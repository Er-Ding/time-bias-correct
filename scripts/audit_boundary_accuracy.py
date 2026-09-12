"""只在评估侧回溯已保存的定位结果；不修改在线定位或实验输入。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.solver import (
    CandidateTrajectory, SolverConfig, _assign_candidates, _objective, _refine_seed,
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def slim(value):
    """Discard repeated cluster payloads after parsing, retaining audit fields."""
    if 'candidate_id' in value and 'members' in value and 'point' in value:
        return {k: v for k, v in value.items() if k != 'members'}
    if 'source_sample_ids' in value and 'representative_sample_id' in value:
        return {k: value[k] for k in (
            'topology_id', 'propagation_interactions', 'representative_sample_id',
            'observed_aoa_global_rad', 'observed_delay_s', 'endpoint_origin_m',
            'point_cluster_id') if k in value}
    return value


def read(path, expected=None):
    raw = Path(path).read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f'文件摘要不匹配：{path}')
    return json.loads(raw, object_hook=slim)


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def point_key(p):
    return p['observation_id'], p['sample_id'], p['topology_id']


def point_at_bias(p, bias_s):
    length = (float(p['observed_delay_s']) - bias_s) * C - p['prefix_length_m']
    if length <= 1e-7 or length >= p['endpoint_free_distance_m'] - 1e-7:
        return None
    return np.asarray(p['endpoint_origin_m']) + length * np.asarray(p['endpoint_direction'])


def closest(points, truth, bias_s):
    distances = defaultdict(list)
    for p in points:
        q = point_at_bias(p, bias_s)
        if q is not None:
            distances[p['observation_id']].append(float(np.linalg.norm(q - truth)))
    return {k: min(v) for k, v in distances.items()}


def support(distances, radius):
    return sum(v <= radius for v in distances.values())


def stage_route(initial, clustered, representatives, radius):
    if support(initial, radius) < 2:
        return 'initial_less_than_two_nearby_observations'
    if support(clustered, radius) < 2:
        return 'dbscan_lost_nearby_support'
    if support(representatives, radius) < 2:
        return 'representatives_lost_nearby_support'
    return 'nearby_representatives_reached_solver'


def candidate(p):
    return CandidateTrajectory(
        observation_id=p['observation_id'], candidate_id=p['candidate_id'],
        anchor_m=p['anchor_m'], direction=p['direction'],
        beta_min_m=p['beta_interval_m'][0], beta_max_m=p['beta_interval_m'][1],
        weight=p['weight'], metadata=p.get('metadata', {}))


def peak_audit(peaks, config, truth_paths, bias_s):
    answer = []
    options = config['music']['spectrum_sampling']
    aw = math.radians(config['music']['angle_step_deg'] * options['aoa_half_width_grid_steps'])
    dw = config['music']['delay_step_s'] * options['delay_half_width_grid_steps']
    maximum = max(p['spectrum_value'] for p in peaks['nominal'])
    for i, peak in enumerate(peaks['nominal']):
        source = peaks['nominal_source_indices'][i]
        coarse = peaks['coarse'][source]
        matches = []
        nearest = None
        for path in truth_paths:
            da = abs(path['aoa_local_rad'] - peak['aoa_rad'])
            dt = abs(path['geometric_delay_s'] + bias_s - peak['delay_s'])
            score = (da / aw) ** 2 + (dt / dw) ** 2
            item = dict(path_index=path['path_index'], angle_error_deg=math.degrees(da),
                        delay_error_ns=dt * 1e9, normalized_distance=score)
            if nearest is None or score < nearest['normalized_distance']:
                nearest = item
            if (abs(path['aoa_local_rad'] - coarse['aoa_rad']) <= aw + 1e-10
                    and abs(path['geometric_delay_s'] + bias_s - coarse['delay_s']) <= dw + 1e-15):
                matches.append(path['path_index'])
        answer.append(dict(observation_id=f'music_path_{source:02d}',
                           aoa_local_deg=math.degrees(peak['aoa_rad']),
                           delay_ns=peak['delay_s'] * 1e9,
                           spectrum_value=peak['spectrum_value'],
                           relative_spectrum=peak['spectrum_value'] / maximum,
                           truth_paths_in_sampling_window=matches, nearest_truth=nearest))
    return answer


def audit_trial(row, point, bias_s, truth_paths, raw_available, radius, do_refine):
    out = {k: row.get(k) for k in ('ue_id', 'repeat_index', 'strategy', 'status',
           'position_error_m', 'clock_bias_error_ns', 'forward_valid', 'identifiable',
           'initial_count', 'cluster_count', 'representative_count', 'error')}
    out.update(retained_path_count=point['retained_path_count'],
               channel_category=point['channel_category'], has_diffraction=point['has_diffraction'],
               result_dir=row['result_dir'], raw_inputs_verified=raw_available)
    if row['status'] != 'success':
        out['route'] = 'no_position_output'
        return out
    root = Path(row['result_dir']) / 'localization'
    manifest = read(root / 'localization_manifest.json')
    if manifest['inputs']['online_measurement']['sha256'] != row['input_sha256']:
        raise ValueError('定位记录与输入绑定不一致')
    def artifact(key):
        entry = manifest['artifacts'][key]
        return read(root / Path(entry['path']).name, entry['sha256'])
    config = read(root / 'localization_config.json', manifest['config_snapshot']['file_sha256'])['resolved_config']
    peaks = artifact('music_peaks')
    out['peaks'] = peak_audit(peaks, config, truth_paths, bias_s)
    result = artifact('result')
    truth = np.asarray(point['position_m'])
    error = float(np.linalg.norm(np.asarray(result['mu_m']) - truth))
    if abs(error - row['position_error_m']) > 1e-8:
        raise ValueError('位置误差重算不一致')
    if abs(abs(result['clock_bias_s'] - bias_s) * 1e9 - row['clock_bias_error_ns']) > 1e-6:
        raise ValueError('时间偏差误差重算不一致')
    initial = artifact('initial_candidates')['points']
    initial_distances = closest(initial, truth, bias_s)
    clusters = artifact('representative_points')
    kept = {point_key(p) for p in clusters['memberships'] if p['role'] != 'noise'}
    cluster_distances = closest((p for p in initial if point_key(p) in kept), truth, bias_s)
    del clusters, initial
    trajectories = [candidate(p) for p in artifact('representative_trajectories')]
    groups = defaultdict(list)
    for p in trajectories:
        groups[p.observation_id].append(p)
    representative_distances = {}
    for name, items in groups.items():
        ds = [float(np.linalg.norm(p.point(bias_s * C) - truth)) for p in items if p.is_valid(bias_s * C, 1e-9)]
        if ds:
            representative_distances[name] = min(ds)
    out.update(initial_min_by_observation_m=initial_distances,
               cluster_min_by_observation_m=cluster_distances,
               representative_min_by_observation_m=representative_distances)
    for label, values in [('initial', initial_distances), ('cluster', cluster_distances), ('representative', representative_distances)]:
        out[label + '_nearest_m'] = min(values.values(), default=None)
        out[label + '_support_count'] = support(values, radius)
    out['route'] = stage_route(initial_distances, cluster_distances, representative_distances, radius)
    out['route_5m'] = stage_route(initial_distances, cluster_distances, representative_distances, 5.0)
    cfg = SolverConfig(huber_delta=float(config['localization']['huber_delta_m']),
                       max_iterations=int(config['localization']['max_iterations']),
                       max_seed_pairs=int(config['localization'].get('max_seed_pairs', 100000)))
    gt_state = np.r_[truth, bias_s * C]
    final_state = np.r_[result['mu_m'], result['distance_bias_m']]
    gt_assignment = _assign_candidates(gt_state, groups, cfg)
    final_assignment = _assign_candidates(final_state, groups, cfg)
    final_cost = _objective(final_state, final_assignment, len(groups), cfg)
    recorded_cost = result['diagnostics']['objective']
    if not np.isclose(final_cost, recorded_cost, atol=1e-7, rtol=1e-8):
        raise ValueError('目标函数重算与保存值不一致')
    gt_cost = _objective(gt_state, gt_assignment, len(groups), cfg)
    out.update(objective_truth=gt_cost, objective_final=final_cost,
               truth_valid_observation_count=len(gt_assignment),
               truth_has_lower_objective=len(gt_assignment) >= 2 and gt_cost + 1e-7 < final_cost,
               final_bias_ns=result['clock_bias_s'] * 1e9,
               final_position_m=result['mu_m'],
               final_residuals_m=result['central_residuals_m'],
               robust_weights=result['diagnostics']['robust_weights'],
               physical_constraints=result['diagnostics']['diffraction_physical_constraints'],
               final_selected={k: {'candidate_id': p.candidate_id,
                                  'interactions': p.metadata.get('propagation_interactions', []),
                                  'origin_m': p.metadata.get('endpoint_origin_m')}
                               for k, p in final_assignment.items()})
    if do_refine and error > 5.0:
        # Diagnostic counterfactual only. Never added to the original trial results.
        refined = _refine_seed(gt_state, groups, cfg)
        out['truth_seed_diagnostic'] = None if refined is None else dict(
            position_error_m=float(np.linalg.norm(refined.state[:2] - truth)),
            clock_bias_ns=float(refined.state[2] / C * 1e9), objective=refined.objective,
            converged=refined.converged,
            candidate_ids={k: v.candidate_id for k, v in refined.assignment.items()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coverage-radius-m', type=float, default=2.0)
    parser.add_argument('--truth-seed-diagnostic', action='store_true')
    parser.add_argument('--ue-ids', default='')
    args = parser.parse_args()
    if args.coverage_radius_m <= 0:
        parser.error('覆盖半径必须为正')
    root = args.experiment.resolve()
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    exp = read(root / 'experiment.json')
    plan = read(root / 'pilot/plan.json')
    rows = [json.loads(line) for line in (root / 'pilot/trials.jsonl').read_text().splitlines()]
    wanted = set(args.ue_ids.split(',')) if args.ue_ids else set()
    known = {p['ue_id'] for p in plan['points']}
    if wanted - known:
        raise ValueError(f'冻结计划中没有这些 UE：{sorted(wanted - known)}')
    row_keys = [(r['ue_id'], r['repeat_index'], r['strategy']) for r in rows]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError('同一次计划请求存在重复记录')
    if wanted:
        rows = [r for r in rows if r['ue_id'] in wanted]
    # The diagnostic replay must use the same objective/trajectory convention as the experiment.
    repo = Path(__file__).resolve().parents[1]
    for name in ['solver.py', 'constants.py']:
        if sha(repo / 'src/time_bias_localization' / name) != exp['source']['files'][name]:
            raise ValueError(f'实验之后 {name} 已变化，不能用当前代码重放旧目标函数')
    provenance = dict(experiment=str(root), plan_sha256=sha(root / 'pilot/plan.json'),
                      trials_sha256=sha(root / 'pilot/trials.jsonl'),
                      audit_script_sha256=sha(__file__), arguments=vars(args).copy(),
                      uses_ground_truth=True, purpose='offline_diagnosis_only',
                      updates_original_results=False, gpu_used=False)
    provenance['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in provenance['arguments'].items()}
    write(dest / 'provenance.json', provenance)
    points = {p['ue_id']: p for p in plan['points']}
    truths = {}
    raw_missing = []
    for point in points.values():
        if wanted and point['ue_id'] not in wanted:
            continue
        gpath = Path(point['probe_root']) / 'geometry_channel.npz'
        paths = []
        if gpath.is_file():
            probe = read(gpath.parent / 'probe.json')
            if sha(gpath) != probe['array_artifact']['sha256']:
                raise ValueError(f'冻结信道摘要错误：{gpath}')
            with np.load(gpath, allow_pickle=False) as d:
                keep = d['retained_mask'].astype(bool)
                if np.count_nonzero(keep) != point['retained_path_count']:
                    raise ValueError('真实路径数与冻结清单不一致')
                powers = np.sum(np.abs(d['path_coefficients']) ** 2, axis=0)
                for j in np.flatnonzero(keep):
                    paths.append(dict(path_index=int(j), aoa_local_rad=float(d['aoa_local_rad'][j]),
                                      aoa_global_rad=float(d['aoa_global_rad'][j]),
                                      geometric_delay_s=float(d['absolute_delays_s'][j]),
                                      relative_power=float(powers[j] / max(powers[keep])),
                                      reflections=int(d['reflection_order'][j]), diffractions=int(d['diffraction_order'][j])))
        else:
            raw_missing.append(str(gpath))
        truths[point['ue_id']] = paths
    write(dest / 'true_paths.json', truths)
    output = []
    started = time.monotonic()
    for i, row in enumerate(rows, 1):
        point = points[row['ue_id']]
        obs = point['observations'][row['repeat_index']]
        if (row['input_sha256'] != obs['input_sha256']
                or row['noise_seed'] != point['noise_seeds'][row['repeat_index']]
                or row['mc_seed'] != point['mc_seeds'][row['repeat_index']]):
            raise ValueError('逐次记录与冻结输入或随机种子不一致')
        available = True
        for key, expected in [('online_npz', 'input_sha256'), ('truth_npz', 'truth_sha256'), ('scene_json', 'scene_sha256'), ('generation_manifest', 'manifest_sha256')]:
            path = Path(obs[key])
            if not path.is_file():
                raw_missing.append(str(path)); available = False
            elif sha(path) != obs[expected]:
                raise ValueError(f'恢复的原始文件与实验不一致：{path}')
        bias_s = float(exp['generation_config']['simulation']['clock_bias_s'])
        if Path(obs['truth_npz']).is_file():
            with np.load(obs['truth_npz'], allow_pickle=False) as d:
                if not np.allclose(d['ue_position_m'], point['position_m'], atol=1e-10, rtol=0):
                    raise ValueError('冻结 UE 与真值坐标不一致')
                bias_s = float(d['clock_bias_s'])
        r = audit_trial(row, point, bias_s, truths[point['ue_id']], available,
                        args.coverage_radius_m, args.truth_seed_diagnostic)
        output.append(r)
        with (dest / 'audit.jsonl').open('a') as stream:
            stream.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
        print(f"[精度回溯] {i}/{len(rows)} {row['ue_id']} repeat={row['repeat_index']} {row['strategy']} 原误差={row['position_error_m']} 路由={r['route']}", flush=True)
    summary = dict(request_count=len(rows), elapsed_seconds=time.monotonic() - started,
                   coverage_radius_m=args.coverage_radius_m,
                   raw_missing_files=sorted(set(raw_missing)), groups=[])
    for strategy in ['single', 'coverage']:
        for label, predicate in [('all', lambda r: True), ('error_over_5m', lambda r: (r['position_error_m'] or 0) > 5),
                                 ('error_over_100m', lambda r: (r['position_error_m'] or 0) > 100),
                                 ('error_within_2m', lambda r: r['position_error_m'] is not None and r['position_error_m'] <= 2)]:
            subset = [r for r in output if r['strategy'] == strategy and predicate(r)]
            summary['groups'].append(dict(strategy=strategy, subset=label, count=len(subset),
                routes=dict(Counter(r['route'] for r in subset)),
                routes_5m=dict(Counter(r.get('route_5m', 'no_position_output') for r in subset)),
                insufficient_true_paths=sum(r['retained_path_count'] < 2 for r in subset),
                truth_has_lower_objective=sum(r.get('truth_has_lower_objective', False) for r in subset),
                truth_seed_within_2m=sum((r.get('truth_seed_diagnostic') or {}).get('position_error_m', math.inf) <= 2 for r in subset)))
    write(dest / 'summary.json', summary)
    fields = ['ue_id', 'repeat_index', 'strategy', 'status', 'position_error_m', 'retained_path_count',
              'channel_category', 'initial_nearest_m', 'cluster_nearest_m', 'representative_nearest_m',
              'initial_support_count', 'cluster_support_count', 'representative_support_count',
              'route', 'route_5m', 'objective_truth', 'objective_final', 'truth_has_lower_objective',
              'final_bias_ns', 'forward_valid', 'identifiable']
    with (dest / 'audit.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(output)
    print(f'回溯完成：{dest}；缺失原始文件 {len(summary["raw_missing_files"])} 个', flush=True)


if __name__ == '__main__':
    main()
