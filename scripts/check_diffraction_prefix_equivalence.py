#!/usr/bin/env python3
"""用全部首反射墙建立独立对照表，核对真实地图的保守筛选没有漏项。"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np
from time_bias_localization.diffraction_prefixes import PrefixGeometry, clear_prefix_cache, get_diffraction_prefixes
from time_bias_localization.scene import Scene2D


def serial_prefixes(prefixes):
    return [dict(edge_id=p[0].edge_id, wall_ids=list(p[1]), points_m=[np.asarray(q).tolist() for q in p[2]],
                 length_m=p[3], angle_rad=p[4], toward_bs=p[5].tolist()) for p in prefixes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=int, default=300)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds 必须为正整数')
    def deadline(_signum, _frame):
        raise TimeoutError(f'前缀表核对超过 {args.timeout_seconds} 秒')
    signal.signal(signal.SIGALRM, deadline); signal.alarm(args.timeout_seconds)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.experiment.resolve() / 'pilot/observations/PILOT_0001/repeat_000/generation_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    scene_path = Path(manifest['scene_json'])
    scene = Scene2D.load(scene_path)
    # BS is public online metadata. No UE position, truth path, or noise metadata is loaded.
    with np.load(manifest['online_input'], allow_pickle=False) as measurement:
        source = measurement['bs_position_m'].copy()
    max_reflections = int(manifest['rt_model']['max_reflections'])
    clear_prefix_cache()
    started = time.perf_counter()
    filtered, _ = get_diffraction_prefixes(scene, source, max_reflections)
    filtered_seconds = time.perf_counter() - started
    filtered_serial = serial_prefixes(filtered)
    (output / 'filtered_prefixes.json').write_text(json.dumps(filtered_serial, indent=2))
    clear_prefix_cache()
    print('[前缀完整性核对] 开始全首墙对照；不使用首墙可见性筛选', flush=True)
    started = time.perf_counter()
    with patch.object(PrefixGeometry, 'visible_first_walls',
                      lambda self, _source: np.arange(len(self.walls))):
        complete, _ = get_diffraction_prefixes(scene, source, max_reflections)
    complete_seconds = time.perf_counter() - started
    complete_serial = serial_prefixes(complete)
    (output / 'unpruned_prefixes.json').write_text(json.dumps(complete_serial, indent=2))
    filtered_keys = [(p[0].edge_id, p[1]) for p in filtered]
    complete_keys = [(p[0].edge_id, p[1]) for p in complete]
    assert filtered_keys == complete_keys, '前缀序列及顺序与全部首墙枚举不一致'
    max_difference = 0.
    for left, right in zip(filtered, complete):
        for a, b in ((np.asarray(left[2]).reshape(-1, 2), np.asarray(right[2]).reshape(-1, 2)),
                     (np.asarray(left[3:5]), np.asarray(right[3:5])), (left[5], right[5])):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)
            if np.asarray(a).size:
                max_difference = max(max_difference, float(np.max(np.abs(np.asarray(a) - b))))
    result = dict(scene_json=str(scene_path), scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                  bs_position_m=source.tolist(), wall_count=len(scene.walls), max_reflections=max_reflections,
                  filtered_prefix_count=len(filtered), unpruned_prefix_count=len(complete),
                  same_sequences_and_order=True, exact_serialization_equal=filtered_serial == complete_serial,
                  max_absolute_field_difference=max_difference,
                  filtered_seconds=filtered_seconds, unpruned_seconds=complete_seconds,
                  truth_loaded=False)
    (output / 'equivalence.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
