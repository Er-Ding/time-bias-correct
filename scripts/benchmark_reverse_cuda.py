#!/usr/bin/env python3
"""冻结 CSI/采样的 CPU-CUDA 反向追踪对照；独立进程，不更改主实验后端。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

import numpy as np

from time_bias_localization import pipeline
from time_bias_localization.config import localization_config_view
from time_bias_localization.diffraction_prefixes import clear_prefix_cache, get_diffraction_prefixes
from time_bias_localization.initial_candidates import generate_initial_candidate_points
from time_bias_localization.reverse_cuda import (
    CudaWallGeometry, clear_cuda_reverse_cache, generate_initial_candidate_points_cuda,
)
from time_bias_localization.scene import Scene2D
from time_bias_localization.timing import collect_timings, register_synchronizer


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(pipeline._jsonable(data), ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialized_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def append(path, value):
    with path.open('a') as output:
        output.write(json.dumps(value, ensure_ascii=False)+'\n')
        output.flush()


def reuse_reverse_measurements(source, cases, repeats, output):
    """报告保存失败时复用完整反向计时；验证算法和输入完全相同，保留来源。"""
    source = source.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('复用来源和新目录必须互相独立')
    protocol = json.loads((source/'protocol.json').read_text())
    if protocol['cases'] != cases or protocol['timing_repeats'] != repeats:
        raise ValueError('复用的观测清单、参数或重复次数不一致')
    project = Path(__file__).resolve().parents[1]
    for name, expected in protocol['source_hashes'].items():
        if name.startswith('src/') and digest(project/name) != expected:
            raise ValueError(f'定位或 CUDA 算法代码改变，不能复用旧计时：{name}')
    rows = [json.loads(line) for line in (source/'timings.jsonl').read_text().splitlines()]
    expected_pairs = {(case['case_id'], repeat) for case in cases for repeat in range(repeats)}
    actual_pairs = {(row['case_id'], row['repeat']) for row in rows}
    if len(rows) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise ValueError('旧反向计时未完整完成，不能作为完成数据复用')
    if any(not (row['comparison']['discrete_fields_equal'] and row['comparison']['numeric_fields_allclose']) for row in rows):
        raise ValueError('旧反向计时存在候选核对失败')
    records = {}
    for path in [source/'timings.jsonl', source/'setup.json', *sorted((source/'cases').rglob('*.json'))]:
        relative = path.relative_to(source)
        target = output/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[str(relative)] = digest(path)
    write(output/'reverse_reuse.json', {'source': str(source), 'copied_sha256': records,
        'source_protocol_sha256': digest(source/'protocol.json'),
        'algorithm_hashes_verified': True, 'reason': 'continue_complete_pipeline_checks_after_report_serialization_fix'})
    prepared = [(case, json.loads((output/'cases'/case['case_id']/'inputs.json').read_text())['samples']) for case in cases]
    print(f'[复用反向计时] {len(rows)} 对；算法和输入已核对，原目录保留：{source}', flush=True)
    return rows, prepared


def snapshot_source(destination):
    root = Path(__file__).resolve().parents[1]
    files = sorted((root/'src/time_bias_localization').glob('*.py'))
    files += [Path(__file__).resolve(), root/'run_reverse_cuda_experiment.sh', root/'tests/test_reverse_cuda.py']
    records = {}
    for source in files:
        relative = source.relative_to(root)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records[str(relative)] = digest(source)
    return records


def extract_samples(config, observation):
    """执行与主流程相同的 MUSIC 和采样函数；不打开真值文件。"""
    manifest, _, bundle_id = pipeline.load_generation_manifest(observation['generation_manifest'])
    stage_name, checked = pipeline.validate_generation_manifest_envelope(manifest)
    if bundle_id != checked:
        raise ValueError('生成清单批次编号不一致')
    scene_capture = pipeline.capture_file(observation['scene_json'])
    online_capture = pipeline.capture_file(observation['online_npz'])
    pipeline.verify_generation_artifact(manifest, 'scene_json', scene_capture)
    pipeline.verify_generation_artifact(manifest, 'online_measurement', online_capture)
    scene = Scene2D.from_dict(json.loads(scene_capture.data))
    measurement = pipeline.load_online_measurement_bytes(online_capture.data, source_path=online_capture.path)
    pipeline.validate_localization_input_contract(manifest, stage_name, config, scene, measurement)
    music = config['music']; compute = config['compute']
    computer = pipeline.get_music_computer(compute['backend'], compute['device_id'],
                                         compute['batch_size'], compute['angle_chunk_size'])
    register_synchronizer(computer.synchronize)
    pipeline._validate_unambiguous_delay_window(music, measurement.subcarrier_frequencies_hz)
    angles, delays = pipeline._make_grids(music)
    prepared = computer.prepare(measurement.csi_observed,
        subcarrier_frequencies_hz=measurement.subcarrier_frequencies_hz,
        carrier_frequency_hz=measurement.carrier_frequency_hz,
        antenna_spacing_m=measurement.antenna_spacing_m,
        num_sources=int(music.get('signal_subspace_rank', music['num_paths'])),
        spatial_subarray_size=int(music['spatial_subarray_size']),
        frequency_subarray_size=int(music['frequency_subarray_size']),
        diagonal_loading=float(music['diagonal_loading']))
    spectrum = prepared.spectrum(aoa_grid_rad=angles, delay_grid_s=delays)
    peaks = pipeline.extract_local_music_peaks(spectrum, aoa_grid_rad=angles, delay_grid_s=delays,
        max_peaks=int(music['num_paths']), minimum_relative_height=0.0,
        minimum_separation_bins=pipeline._separation_bins(music))
    if len(peaks) < 2:
        raise RuntimeError('MUSIC 搜索区域不足两条')
    sampled = pipeline.sample_music_spectrum(prepared, peaks, aoa_grid_rad=angles, delay_grid_s=delays,
        bs_boresight_rad=measurement.bs_boresight_rad, settings=music['spectrum_sampling'],
        seed=int(config['project']['random_seed'])+2,
        minimum_angle_separation_rad=np.deg2rad(float(music['min_angle_separation_deg'])),
        minimum_delay_separation_s=float(music['min_delay_separation_s']))
    if len(sampled.refined_peaks) < 2:
        raise RuntimeError('MUSIC 细谱有效峰不足两条')
    return scene, measurement.bs_position_m, sampled.samples


def compare_points(expected, actual):
    if len(actual.points) != len(expected.points):
        raise AssertionError(f'候选数量不同 CPU={len(expected.points)} CUDA={len(actual.points)}')
    numeric = {'position_m', 'reflection_points_m', 'prefix_length_m', 'endpoint_origin_m',
               'endpoint_direction', 'endpoint_free_distance_m', 'interaction_points_m'}
    maxima = dict.fromkeys(sorted(numeric), 0.)
    for cpu, gpu in zip(expected.points, actual.points, strict=True):
        a, b = asdict(cpu), asdict(gpu)
        for key in a:
            if key not in numeric:
                if a[key] != b[key]:
                    raise AssertionError(f'{cpu.sample_id}: {key} 不一致')
            else:
                x, y = np.asarray(a[key], float), np.asarray(b[key], float)
                if x.shape != y.shape:
                    raise AssertionError(f'{cpu.sample_id}: {key} 形状不一致')
                if x.size:
                    maxima[key] = max(maxima[key], float(np.max(np.abs(x-y))))
                np.testing.assert_allclose(y, x, rtol=1e-12, atol=1e-8,
                                           err_msg=f'{cpu.sample_id}: {key}')
    if expected.rejected_samples != actual.rejected_samples:
        raise AssertionError('纯反射拒绝记录不同')
    for key in ['initial_point_count', 'observation_initial_point_counts']:
        if expected.diagnostics[key] != actual.diagnostics[key]:
            raise AssertionError(f'候选诊断 {key} 不同')
    for key in ['sample_prefix_angle_match_count', 'attempted_direction_count', 'observation_point_counts']:
        if expected.diagnostics['diffraction'][key] != actual.diagnostics['diffraction'][key]:
            raise AssertionError(f'绕射诊断 {key} 不同')
    return {'candidate_count': len(actual.points), 'diffraction_count': sum(p.has_diffraction for p in actual.points),
            'discrete_fields_equal': True, 'numeric_fields_allclose': True, 'maximum_absolute_difference': maxima}


def measured_reverse(function, scene, bs, samples, options, synchronize):
    synchronize()
    started = time.perf_counter()
    with collect_timings(synchronize=synchronize) as recorder:
        result = function(scene, bs, samples, **options)
    synchronize()
    seconds = time.perf_counter()-started
    stages = {row['name']: row['elapsed_s'] for row in recorder.to_dict()['stages']}
    return result, {'reverse_seconds': seconds, 'stage_seconds': stages}


@contextmanager
def candidate_backend(function):
    """仅在这个独立验证进程内切换调用入口，结束后恢复；不写主流程文件。"""
    original = pipeline.generate_initial_candidate_points
    pipeline.generate_initial_candidate_points = function
    try:
        yield
    finally:
        pipeline.generate_initial_candidate_points = original


def full_pipeline_check(case, output, expected_samples):
    """这里只核对最终结果；不把跨流程缓存的计时用作端到端提速结论。"""
    records = []
    for policy in ('single', 'coverage'):
        results = {}
        for backend, function in [('cpu', generate_initial_candidate_points), ('cuda', generate_initial_candidate_points_cuda)]:
            config = deepcopy(case['config'])
            config['localization']['diffraction_representative_policy'] = policy
            destination = output/policy/backend
            seen = []
            def checked(scene, bs, samples, **options):
                samples = list(samples)
                actual_samples = [asdict(sample) for sample in samples]
                if actual_samples != expected_samples:
                    raise AssertionError('完整流程的 MUSIC 采样与冻结反向输入不一致')
                initial = function(scene, bs, samples, **options)
                seen.append(initial)
                return initial
            print(f"[完整流程核对] {case['case_id']} {policy} {backend}", flush=True)
            with candidate_backend(checked), collect_timings() as timer:
                result = pipeline.localize(config, scene_json=case['observation']['scene_json'],
                    online_input=case['observation']['online_npz'],
                    generation_manifest=case['observation']['generation_manifest'], output_root=destination)
            position = result['mu_m']
            results[backend] = {'position_m': position, 'clock_bias_s': result['clock_bias_s'],
                                'initial': seen[0], 'timing': timer.to_dict()}
        comparison = compare_points(results['cpu']['initial'], results['cuda']['initial'])
        np.testing.assert_allclose(results['cuda']['position_m'], results['cpu']['position_m'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(results['cuda']['clock_bias_s'], results['cpu']['clock_bias_s'], atol=1e-14, rtol=0)
        # Evaluation happens only after both online requests have returned.
        observation = case['observation']
        if digest(observation['truth_npz']) != observation['truth_sha256']:
            raise ValueError('独立评估的真值文件摘要不一致')
        with np.load(observation['truth_npz'], allow_pickle=False) as truth:
            truth_position = np.asarray(truth['ue_position_m'], float)
        row = {'case_id': case['case_id'], 'policy': policy, 'candidate_comparison': comparison,
               'position_difference_m': float(np.linalg.norm(np.asarray(results['cuda']['position_m'])-results['cpu']['position_m'])),
               'clock_bias_difference_s': abs(results['cuda']['clock_bias_s']-results['cpu']['clock_bias_s']),
               'timing_use': 'diagnostic_only_not_end_to_end_speedup'}
        for backend in ('cpu', 'cuda'):
            row[backend] = {key: value for key, value in results[backend].items() if key != 'initial'}
            row[backend]['euclidean_error_m'] = float(np.linalg.norm(np.asarray(results[backend]['position_m'])-truth_position))
        records.append(row)
        write(output/policy/'comparison.json', row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ue-ids', required=True)
    parser.add_argument('--noise-repeats', type=int, default=2)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--pipeline-cases', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=2400)
    parser.add_argument('--reuse-reverse-from', type=Path)
    args = parser.parse_args()
    if min(args.noise_repeats, args.timing_repeats, args.timeout_seconds) <= 0 or args.pipeline_cases < 0:
        parser.error('次数和超时必须为正，完整流程个数必须非负')
    def deadline(*_):
        raise TimeoutError(f'独立 CUDA 验证超过 {args.timeout_seconds} 秒')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    root = args.experiment.resolve()
    experiment = json.loads((root/'experiment.json').read_text())
    plan = json.loads((root/'pilot/plan.json').read_text())
    ids = args.ue_ids.split(',')
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('UE 编号不能为空或重复')
    points = {point['ue_id']: point for point in plan['points']}
    cases = []
    for ue_id in ids:
        point = points[ue_id]
        if args.noise_repeats > len(point['observations']):
            raise ValueError(f'{ue_id} 没有足够的冻结噪声观测')
        for repeat in range(args.noise_repeats):
            config = localization_config_view(experiment['generation_config'])
            config['compute']['backend'] = 'cuda'; config['compute']['device_id'] = 0
            config['project']['random_seed'] = point['mc_seeds'][repeat]
            observation = point['observations'][repeat]
            for field, sha in [('online_npz', 'input_sha256'), ('scene_json', 'scene_sha256'),
                               ('generation_manifest', 'manifest_sha256')]:
                if digest(observation[field]) != observation[sha]:
                    raise ValueError(f'{ue_id} 冻结输入 {field} 摘要不一致')
            cases.append({'case_id': f'{ue_id}_noise_{repeat:03d}', 'config': config, 'observation': observation})
    write(output/'protocol.json', {
        'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'cases': cases, 'timing_repeats': args.timing_repeats, 'pipeline_cases': args.pipeline_cases,
        'gpu_mask': os.environ.get('CUDA_VISIBLE_DEVICES'), 'cpu_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
        'source_hashes': snapshot_source(output/'source_snapshot'),
        'reuse_reverse_from': str(args.reuse_reverse_from.resolve()) if args.reuse_reverse_from else None,
        'experiment_sha256': digest(root/'experiment.json'), 'plan_sha256': digest(root/'pilot/plan.json'),
        'scope': 'fixed_case_correctness_and_reverse_latency_not_population_accuracy',
        'timing_definition': 'synchronized wall time, including transfers and candidate objects',
        'steady_state': 'shared immutable CPU prefix prebuilt; both backends exercised before balanced repeated timing',
        'cpu_remaining': ['public_prefix_build', 'pure_specular_branch', 'angle_matching', 'candidate_objects'],
    })
    print(f"[实验计划] {len(cases)} 份冻结 CSI，每份 {args.timing_repeats} 次 CPU/CUDA 对照；输出 {output}", flush=True)
    rows = []; pipeline_rows = []; prepared = []
    if args.reuse_reverse_from:
        rows, prepared = reuse_reverse_measurements(args.reuse_reverse_from, cases, args.timing_repeats, output)
    clear_prefix_cache(); clear_cuda_reverse_cache()
    import cupy as cp
    synchronize = cp.cuda.get_current_stream().synchronize
    for index, case in enumerate([] if args.reuse_reverse_from else cases):
        case_dir = output/'cases'/case['case_id']
        print(f"[观测提取] {index+1}/{len(cases)} {case['case_id']}", flush=True)
        with collect_timings() as capture_timer:
            scene, bs, samples = extract_samples(case['config'], case['observation'])
        sample_rows = [asdict(sample) for sample in samples]
        options = {'reference_bias_s': case['config']['localization']['initial_reference_bias_s'],
                   'max_reflections': case['config']['scene']['max_reflections'], 'max_diffractions': 1,
                   'diffraction_directions_per_sample': case['config']['localization']['diffraction_directions_per_sample'],
                   'diffraction_angle_tolerance_deg': case['config']['localization']['diffraction_angle_tolerance_deg']}
        write(case_dir/'inputs.json', {'samples': sample_rows, 'sample_sha256': serialized_digest(sample_rows),
              'options': options, 'bs_position_m': list(bs), 'source': case['observation'],
              'feature_extraction_timing': capture_timer.to_dict()})
        if index == 0:
            t = time.perf_counter()
            prefixes, _ = get_diffraction_prefixes(scene, bs, options['max_reflections'])
            prefix_seconds = time.perf_counter()-t
            t = time.perf_counter(); geometry = CudaWallGeometry(scene); geometry.synchronize()
            cuda_setup_seconds = time.perf_counter()-t
            setup = {'cpu_public_prefix_seconds': prefix_seconds, 'prefix_count': len(prefixes),
                     'cuda_compile_and_wall_upload_seconds': cuda_setup_seconds,
                     'gpu_name': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                     'cupy_version': cp.__version__, 'kernel_compilation_cache': os.environ.get('CUPY_CACHE_DIR')}
            write(output/'setup.json', setup)
            print(f'[公共路径和 CUDA 初始化] {json.dumps(setup, ensure_ascii=False)}', flush=True)
        # Exercise each backend first. These are retained as initialization checks, not mixed into steady means.
        baseline, cpu_first = measured_reverse(generate_initial_candidate_points, scene, bs, samples, options, synchronize)
        accelerated, gpu_first = measured_reverse(generate_initial_candidate_points_cuda, scene, bs, samples, options, synchronize)
        write(case_dir/'cpu_candidates.json', [asdict(p) for p in baseline.points])
        write(case_dir/'cuda_candidates.json', [asdict(p) for p in accelerated.points])
        comparison = compare_points(baseline, accelerated)
        write(case_dir/'initial_check.json', {'comparison': comparison, 'cpu': cpu_first, 'cuda': gpu_first})
        print(f"[候选核对] {case['case_id']} {comparison['candidate_count']} 个候选一致；"
              f"CPU {cpu_first['reverse_seconds']:.4f}s，CUDA {gpu_first['reverse_seconds']:.4f}s（首次核对）", flush=True)
        for repeat in range(args.timing_repeats):
            order = [('cpu', generate_initial_candidate_points), ('cuda', generate_initial_candidate_points_cuda)]
            if (index+repeat) % 2:
                order.reverse()
            measured = {}
            for backend, function in order:
                result, timing = measured_reverse(function, scene, bs, samples, options, synchronize)
                compare_points(baseline, result)
                measured[backend] = timing
            row = {'case_id': case['case_id'], 'repeat': repeat, 'order': [b for b, _ in order],
                   'comparison': comparison, **measured}
            rows.append(row); append(output/'timings.jsonl', row)
            print(f"[计时] {index+1}/{len(cases)} 第 {repeat+1}/{args.timing_repeats} 次："
                  f"CPU {measured['cpu']['reverse_seconds']:.4f}s，CUDA {measured['cuda']['reverse_seconds']:.4f}s", flush=True)
        prepared.append((case, sample_rows))
    for case, sample_rows in prepared[:args.pipeline_cases]:
        pipeline_rows.extend(full_pipeline_check(case, output/'pipeline'/case['case_id'], sample_rows))
        write(output/'pipeline_comparisons.json', pipeline_rows)
    summary = {'case_count': len(cases), 'timed_pair_count': len(rows), 'all_candidates_equivalent': True,
               'pipeline_policy_pairs': len(pipeline_rows), 'all_pipeline_estimates_equivalent': True if pipeline_rows else None,
               'maximum_candidate_field_differences': {key: max(r['comparison']['maximum_absolute_difference'][key] for r in rows)
                   for key in rows[0]['comparison']['maximum_absolute_difference']},
               'scope': 'reverse_stage_only; not a full_pilot_accuracy_or_end_to_end_speedup_claim'}
    for backend in ('cpu', 'cuda'):
        values = [r[backend]['reverse_seconds'] for r in rows]
        stage_names = sorted(set().union(*(r[backend]['stage_seconds'] for r in rows)))
        summary[backend] = {'reverse_mean_seconds': statistics.mean(values),
            'reverse_median_seconds': statistics.median(values),
            'stage_mean_seconds': {name: statistics.mean(r[backend]['stage_seconds'].get(name, 0.) for r in rows) for name in stage_names}}
    summary['reverse_speedup_ratio_of_means'] = summary['cpu']['reverse_mean_seconds']/summary['cuda']['reverse_mean_seconds']
    summary['setup'] = json.loads((output/'setup.json').read_text())
    write(output/'summary.json', summary)
    print('[实验完成] '+json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # The detached supervisor records the actual exit code; completed JSONL rows remain intact.
        print(f'[实验失败] {type(error).__name__}: {error}', flush=True)
        raise
