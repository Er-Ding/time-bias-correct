"""修复实现后，在新实验目录复用已冻结的观测；不复制旧定位结果或改写源目录。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .contracts import validate_generation_manifest_envelope
from .provenance import artifact_record, canonical_json_sha256, file_sha256


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def reuse_prepared_cohort(source_root, destination_root, cohort, settings, config):
    """核对生成参数、完整计划和文件内容后引用同一批观测，保留历史失败现场。"""
    source, destination = Path(source_root).resolve(), Path(destination_root).resolve()
    if source == destination:
        raise ValueError('修复后的实验必须使用新输出目录，不能覆盖源实验')
    source_record_path = source / 'experiment.json'
    source_plan_path = source / cohort / 'plan.json'
    frozen = _read(source_record_path)
    plan = _read(source_plan_path)
    old_settings, old_config = frozen['settings'], frozen['generation_config']
    for field in ('channel_backend', 'random_seed', 'noise_repeats', f'{cohort}_ue_count', 'legal_region'):
        if old_settings.get(field) != settings.get(field):
            raise ValueError(f'复用观测时采样参数不能改变：{field}')
    # MUSIC 峰选择范围也用于生成端的阵列正面角度筛选，必须保持一致。
    for field in ('scene', 'radio', 'simulation', 'project'):
        if old_config[field] != config[field]:
            raise ValueError(f'复用观测时信道生成参数不能改变：{field}')
    for field in ('angle_min_deg', 'angle_max_deg'):
        if old_config['music'][field] != config['music'][field]:
            raise ValueError(f'复用观测时角度接收范围不能改变：{field}')
    source_public = frozen.get('public_scene_sha256')
    current_public = file_sha256(settings['public_scene_json']) if settings.get('public_scene_json') else None
    if source_public != current_public:
        raise ValueError('复用观测的公开场景文件不同')
    n_points, repeats = settings[f'{cohort}_ue_count'], settings['noise_repeats']
    if (plan.get('cohort') != cohort or plan.get('coverage_frozen_before_localization') is not True
            or len(plan.get('points', [])) != n_points or plan.get('expected_requests') != n_points * repeats * 2):
        raise ValueError('源实验没有完整的冻结观测计划，不能以部分样本恢复实验')
    checked = {}
    def verify(path, digest):
        path = str(Path(path).resolve())
        if path not in checked:
            checked[path] = file_sha256(path)
        if checked[path] != digest:
            raise ValueError(f'源观测内容与冻结摘要不一致：{path}')
    identifiers = set()
    for point in plan['points']:
        key = point['ue_id']
        if key in identifiers or point['cohort'] != cohort:
            raise ValueError('源 UE 清单重复或所属阶段不一致')
        identifiers.add(key)
        if any(len(point.get(field, [])) != repeats for field in ('noise_seeds', 'mc_seeds', 'observations')):
            raise ValueError('源 UE 噪声、采样种子或观测数量不完整')
        setup = Path(point['setup_root']) / 'channel_setup.json'
        setup_digest = file_sha256(setup)
        for observation in point['observations']:
            for field, digest in (('online_npz', 'input_sha256'), ('truth_npz', 'truth_sha256'),
                                  ('scene_json', 'scene_sha256'), ('generation_manifest', 'manifest_sha256')):
                verify(observation[field], observation[digest])
            manifest = _read(observation['generation_manifest'])
            validate_generation_manifest_envelope(manifest)
            for artifact, field, digest in (('scene_json', 'scene_json', 'scene_sha256'),
                                           ('online_measurement', 'online_npz', 'input_sha256'),
                                           ('ground_truth', 'truth_npz', 'truth_sha256')):
                record = manifest['artifact_hashes'][artifact]
                if Path(record['path']).resolve() != Path(observation[field]).resolve() or record['sha256'] != observation[digest]:
                    raise ValueError('观测清单与生成清单不属于同一份数据')
            if manifest.get('channel_setup', {}).get('sha256') != setup_digest:
                raise ValueError('观测所属的公开信道设置不一致')
    provenance = {
        'schema_version': 1, 'source_root': str(source), 'source_experiment': artifact_record(source_record_path),
        'source_plan': artifact_record(source_plan_path), 'verified_input_files': checked,
        'observation_count': n_points * repeats,
        'reuse_scope': 'frozen_positions_channels_and_noisy_csi_only_fresh_localization_and_timing',
        'source_compute': old_config.get('compute'), 'current_compute': config.get('compute'),
        'source_code_sha256': frozen.get('source', {}).get('sha256'),
    }
    copied = deepcopy(plan)
    copied['prepared_reuse'] = {'source_root': str(source), 'source_plan_sha256': provenance['source_plan']['sha256']}
    target = destination / cohort
    target_plan = target / 'plan.json'
    if target_plan.exists():
        if canonical_json_sha256(_read(target_plan)) != canonical_json_sha256(copied):
            raise ValueError('新实验目录已绑定另一份观测计划，拒绝覆盖')
    else:
        _write(target / 'prepared_reuse.json', provenance)
        _write(target / 'frozen_points.json', {'points': [{k: v for k, v in point.items() if k != 'observations'}
                                                       for point in copied['points']],
                                             'coverage_frozen_before_localization': True})
        _write(target_plan, copied)
    print(f'[{cohort} 数据复用] 已核对 {n_points} 个 UE、{n_points * repeats} 份 CSI；源目录保留：{source}', flush=True)
    return copied
