#!/usr/bin/env python3
"""只使用冻结实验的公共地图和首份 CSI，检查反向追踪的冷/热耗时。"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

from time_bias_localization import pipeline
from time_bias_localization.config import localization_config_view
from time_bias_localization.timing import collect_timings


class ReverseComplete(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=300)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds 必须为正整数')
    def deadline(_signum, _frame):
        raise TimeoutError(f'CPU 验证超过 {args.timeout_seconds} 秒')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.timeout_seconds)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    base = args.experiment.resolve()
    experiment = json.loads((base / 'experiment.json').read_text())
    config = localization_config_view(experiment['generation_config'])
    config['compute']['backend'] = 'numpy'
    config['localization']['diffraction_representative_policy'] = 'single'
    # Only the seed is used; the UE position stored for evaluation is never passed online.
    plan = json.loads((base / 'pilot' / 'plan.json').read_text())
    point = next(p for p in plan['points'] if p['ue_id'] == 'PILOT_0001')
    config['project']['random_seed'] = point['mc_seeds'][0]
    observation = base / 'pilot/observations/PILOT_0001/repeat_000'
    manifest_path = observation / 'generation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    original = pipeline.generate_initial_candidate_points
    runs = []
    def bounded_generate(*values, **kwargs):
        started = time.perf_counter()
        result = original(*values, **kwargs)
        elapsed = time.perf_counter() - started
        serial = json.dumps([asdict(p) for p in result.points], sort_keys=True, separators=(',', ':'))
        runs.append(dict(reverse_seconds=elapsed, point_count=len(result.points),
                         candidate_sha256=hashlib.sha256(serial.encode()).hexdigest(),
                         diagnostics=result.diagnostics))
        print(f'[反向验证] 完成：{len(result.points)} 个候选，{elapsed:.3f} 秒', flush=True)
        raise ReverseComplete('验证在反向候选完成后停止，不执行聚类/求解')
    pipeline.generate_initial_candidate_points = bounded_generate
    for index in range(2):
        print(f'[反向验证] 第 {index + 1}/2 次，同一冻结 CSI，CPU', flush=True)
        with collect_timings() as recorder:
            try:
                pipeline.localize(config, scene_json=manifest['scene_json'], online_input=manifest['online_input'],
                                  generation_manifest=manifest_path, output_root=output / f'pass_{index}')
            except ReverseComplete:
                pass
        runs[-1]['timing'] = recorder.to_dict()
        (output / 'performance.json').write_text(json.dumps({'runs': runs}, indent=2))
    if runs[0]['candidate_sha256'] != runs[1]['candidate_sha256']:
        raise AssertionError('同一观测的冷/热缓存候选不一致')
    print(json.dumps({'cold_seconds': runs[0]['reverse_seconds'], 'warm_seconds': runs[1]['reverse_seconds'],
                      'candidate_count': runs[0]['point_count'], 'equal_candidates': True}, indent=2), flush=True)


if __name__ == '__main__':
    main()
