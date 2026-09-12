"""评估侧条件对照：同一保存的观测样本，改变反向追踪参考偏差。"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path

from audit_boundary_accuracy import read, write, sha, closest
from time_bias_localization.initial_candidates import generate_initial_candidate_points
from time_bias_localization.pipeline import PathObservationSample
from time_bias_localization.scene import Scene2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--ue-ids', default='PILOT_0006,PILOT_0011,PILOT_0026,PILOT_0030,PILOT_0008,PILOT_0027')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    exp = read(args.experiment / 'experiment.json')
    points = {p['ue_id']: p for p in read(args.experiment / 'pilot/plan.json')['points']}
    rows = [json.loads(x) for x in (args.experiment / 'pilot/trials.jsonl').read_text().splitlines()]
    source_root = Path(__file__).resolve().parents[1] / 'src/time_bias_localization'
    for name in ['initial_candidates.py', 'diffraction_candidates.py', 'diffraction_prefixes.py', 'diffraction.py']:
        if sha(source_root / name) != exp['source']['files'][name]:
            raise ValueError(f'{name} 已变化，不能用于重放')
    if sha(args.scene) != next(iter(points.values()))['observations'][0]['scene_sha256']:
        raise ValueError('公共地图与原实验摘要不一致')
    scene = Scene2D.from_dict(read(args.scene))
    output = []
    for ue in args.ue_ids.split(','):
        row = next(r for r in rows if r['ue_id'] == ue and r['repeat_index'] == 0 and r['strategy'] == 'coverage')
        root = Path(row['result_dir']) / 'localization'
        manifest = read(root / 'localization_manifest.json')
        peaks = read(root / 'music_peaks.json', manifest['artifacts']['music_peaks']['sha256'])
        original = read(root / 'initial_candidates.json', manifest['artifacts']['initial_candidates']['sha256'])
        config = read(root / 'localization_config.json', manifest['config_snapshot']['file_sha256'])['resolved_config']
        samples = [PathObservationSample(**s) for s in peaks['observation_samples']]
        actual_bias = float(exp['generation_config']['simulation']['clock_bias_s'])
        for reference in [0., actual_bias]:
            print(f'[参考偏差对照] {ue} 参考偏差={reference * 1e9:g} ns', flush=True)
            result = generate_initial_candidate_points(
                scene, config['radio']['bs_position_m'], samples,
                reference_bias_s=reference, max_reflections=config['scene']['max_reflections'],
                max_diffractions=config['scene']['max_diffractions'],
                diffraction_directions_per_sample=config['localization']['diffraction_directions_per_sample'],
                diffraction_angle_tolerance_deg=config['localization']['diffraction_angle_tolerance_deg'])
            candidate_points = [asdict(p) for p in result.points]
            # Check the baseline reproduces positions, order and topology of the saved run.
            baseline_equal = None
            if reference == 0.:
                import numpy as np
                baseline_equal = len(candidate_points) == len(original['points']) and all(
                    p['sample_id'] == q['sample_id'] and p['topology_id'] == q['topology_id']
                    and np.allclose(p['position_m'], q['position_m'], atol=1e-10, rtol=0)
                    for p, q in zip(candidate_points, original['points']))
                if not baseline_equal:
                    raise ValueError('参考偏差为 0 的对照不能重现保存结果')
            distances = closest(candidate_points, points[ue]['position_m'], actual_bias)
            record = dict(ue_id=ue, reference_bias_ns=reference * 1e9, count=len(candidate_points),
                          nearest_by_observation_m=distances, baseline_reproduces_saved=baseline_equal)
            output.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            write(args.output / f'{ue}_reference_{reference * 1e9:g}ns.json', record)
    write(args.output / 'summary.json', dict(
        evaluation_only=True, is_online_fix=False,
        explanation='参考偏差25ns来自冻结生成配置；仅用于隔离参考偏差裁剪影响，不能作为在线已知值。',
        scene=str(args.scene.resolve()), scene_sha256=sha(args.scene), rows=output))


if __name__ == '__main__':
    main()
