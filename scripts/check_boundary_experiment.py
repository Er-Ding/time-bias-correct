"""完整回归测试和小规模实验流程检查；不是 30/300 UE 科学实验。"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-u', '-m', 'pytest', '-q'], check=True)
    project = Path(__file__).resolve().parents[1]
    generation = yaml.safe_load((project / 'configs/diffraction_demo.yaml').read_text())
    generation['music']['spectrum_sampling']['samples_per_peak'] = 16
    generation['localization']['candidate_cluster_min_samples'] = 2
    generation['simulation']['clock_bias_s'] = 0.0
    (root / 'fixture_generation.yaml').write_text(yaml.safe_dump(generation))
    from dataclasses import replace
    from time_bias_localization.scene import make_synthetic_room, WallSegment
    scene = make_synthetic_room(bounds_m=generation['scene']['bounds_m'], fixed_height_m=1.5, bev_resolution_m=.05)
    scene = replace(scene, name=generation['scene']['name'], walls=(*scene.walls, WallSegment('screen', (10., 0.), (10., 8.))))
    scene_json = scene.save(root / 'public_scene')['scene_json']
    fixture = dict(public_scene_json=scene_json, generation_config='fixture_generation.yaml', channel_backend='synthetic_fixture',
                   pilot_ue_count=2, formal_ue_count=1, noise_repeats=2, random_seed=20260911,
                   max_proposals=200, trial_timeout_s=120.0, warmup_per_strategy=1, plots=True)
    (root / 'fixture_experiment.yaml').write_text(yaml.safe_dump(fixture))
    for phase in ('pilot', 'formal'):
        subprocess.run([sys.executable, '-u', '-m', 'time_bias_localization.boundary_experiment',
                        '--config', str(root / 'fixture_experiment.yaml'), '--output', str(root / 'experiment'),
                        '--phase', phase], check=True)
        trials = [json.loads(line) for line in (root / 'experiment' / phase / 'trials.jsonl').read_text().splitlines()]
        assert len(trials) == fixture[f'{phase}_ue_count'] * fixture['noise_repeats'] * 2
        assert all(row['status'] != 'pending' for row in trials)
        paired = {}
        for row in trials:
            paired.setdefault((row['ue_id'], row['repeat_index']), []).append(row)
        for pair in paired.values():
            assert len(pair) == 2 and pair[0]['input_sha256'] == pair[1]['input_sha256']
            assert pair[0]['mc_seed'] == pair[1]['mc_seed']
            if all(row['status'] == 'success' for row in pair):
                paths = [Path(row['result_dir']) / 'localization' for row in pair]
                initial = [json.loads((path / 'initial_candidates.json').read_text()) for path in paths]
                assert initial[0]['points'] == initial[1]['points']
                clusters = [json.loads((path / 'representative_points.json').read_text()) for path in paths]
                memberships = [[{k: v for k, v in member.items() if k != 'is_representative'}
                                for member in cluster['memberships']] for cluster in clusters]
                assert memberships[0] == memberships[1]
        assert any(row['position_error_m'] is not None for row in trials), '流程检查需要至少一次数值输出'
    print(f'小规模流程检查完成：{root}；不能据此声明正式实验精度。', flush=True)


if __name__ == '__main__':
    main()
