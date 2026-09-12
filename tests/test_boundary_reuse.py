from copy import deepcopy
from pathlib import Path

import pytest

from time_bias_localization.boundary_experiment import freeze_experiment, prepare_cohort
from time_bias_localization.boundary_reuse import reuse_prepared_cohort
from time_bias_localization.config import load_config
from time_bias_localization.provenance import file_sha256


@pytest.fixture
def prepared(tmp_path):
    config = load_config(Path(__file__).parents[1] / 'configs/diffraction_demo.yaml')
    settings = dict(channel_backend='synthetic_fixture', random_seed=20260911, pilot_ue_count=1, formal_ue_count=2,
                    noise_repeats=2, legal_region={}, max_proposals=100)
    source = tmp_path / 'old'
    freeze_experiment(source, settings, config)
    plan = prepare_cohort(source, 'pilot', settings, config)
    return source, settings, config, plan


def test_reuse_checks_and_references_same_csi_without_copying_results(prepared, tmp_path):
    source, settings, config, plan = prepared
    before = file_sha256(source / 'experiment.json')
    (source / 'pilot' / 'trials.jsonl').write_text('{"old_failure":"preserved"}\n')
    new_config = deepcopy(config)
    new_config['compute'].update(backend='cuda', device_id=0)
    recovered = reuse_prepared_cohort(source, tmp_path / 'new', 'pilot', settings, new_config)
    assert recovered['points'] == plan['points']
    assert recovered['expected_requests'] == 4
    assert not (tmp_path / 'new' / 'pilot' / 'trials.jsonl').exists()
    assert file_sha256(source / 'experiment.json') == before
    assert (source / 'pilot' / 'trials.jsonl').read_text() == '{"old_failure":"preserved"}\n'
    assert reuse_prepared_cohort(source, tmp_path / 'new', 'pilot', settings, new_config) == recovered


def test_reuse_rejects_modified_csi(prepared, tmp_path):
    source, settings, config, plan = prepared
    path = Path(plan['points'][0]['observations'][0]['online_npz'])
    with path.open('ab') as stream:
        stream.write(b'modified')
    with pytest.raises(ValueError, match='摘要'):
        reuse_prepared_cohort(source, tmp_path / 'new', 'pilot', settings, config)
    assert not (tmp_path / 'new' / 'pilot' / 'plan.json').exists()


def test_reuse_rejects_physical_noise_change_and_incomplete_plan(prepared, tmp_path):
    source, settings, config, plan = prepared
    changed = deepcopy(config)
    changed['radio']['snr_db'] += 1
    with pytest.raises(ValueError, match='radio'):
        reuse_prepared_cohort(source, tmp_path / 'new', 'pilot', settings, changed)
    with pytest.raises(ValueError, match='新输出目录'):
        reuse_prepared_cohort(source, source, 'pilot', settings, config)
