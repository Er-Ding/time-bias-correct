from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from time_bias_localization import boundary_channel as channel
from time_bias_localization.boundary_experiment import (
    derived_seed, load_settings, make_job, prepare_cohort, trial_record, file_sha256,
)
from time_bias_localization.config import load_config
from time_bias_localization.initial_candidates import cluster_initial_candidate_points
from test_diffraction_workflow import arc_point


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / 'configs/diffraction_demo.yaml')


def test_single_policy_changes_only_representative_selection():
    points = [arc_point(i, i * .06) for i in range(20)]
    kwargs = dict(min_samples=2, position_radius_m=1.0, beta_interval_m=(-3., 3.),
                  diffraction_coverage_distance_m=1.0, return_diagnostics=True)
    single = cluster_initial_candidate_points(points, diffraction_representative_policy='single', **kwargs)
    cover = cluster_initial_candidate_points(points, diffraction_representative_policy='coverage', **kwargs)
    assert single.diagnostics['cluster_count'] == cover.diagnostics['cluster_count'] == 1
    assert len(single.representatives) == 1 < len(cover.representatives)
    for left, right in zip(single.memberships, cover.memberships):
        assert {k: v for k, v in left.items() if k != 'is_representative'} == {k: v for k, v in right.items() if k != 'is_representative'}
    for result in (single, cover):
        assert all(rep.point in rep.members and rep.point.weight == 1.0 for rep in result.representatives)
        assert len({rep.metadata['point_cluster_id'] for rep in result.representatives}) == 1
    assert single.representatives[0].metadata['diffraction_representative_policy'] == 'single'


def test_online_job_does_not_receive_truth_or_requested_noise(config, tmp_path):
    observation = dict(scene_json='scene.json', online_npz='observed.npz', truth_npz='secret_truth.npz',
                       generation_manifest='manifest.json', input_sha256='unused')
    job = make_job(config, observation, 'single', 123, tmp_path, True)
    encoded = json.dumps(job)
    for forbidden in ('ue_position_m', 'clock_bias_s', 'snr_db', 'secret_truth.npz', 'simulation'):
        assert forbidden not in encoded
    assert job['config']['project']['random_seed'] == 123
    assert job['config']['radio']['bs_position_m'] == config['simulation']['bs_position_m']


def test_independent_cohort_and_noise_seeds():
    seeds = [derived_seed(123, c, i, purpose) for c in ('pilot', 'formal')
             for i in range(330) for purpose in (0, *range(100, 105), *range(200, 205))]
    assert len(set(seeds)) == len(seeds)
    assert derived_seed(123, 'pilot', 0, 0) == derived_seed(123, 'pilot', 0, 0)


def test_failed_forward_check_still_gets_euclidean_error(tmp_path):
    truth = tmp_path / 'truth.npz'
    np.savez(truth, ue_position_m=[0., 0.], clock_bias_s=1e-9)
    point = dict(cohort='pilot', ue_id='PILOT_0001', noise_seeds=[1], mc_seeds=[2],
                 channel_category='diffraction_only', has_diffraction=True)
    payload = dict(status='success', position_m=[3., 4.], clock_bias_s=3e-9,
                   forward_valid=False, diagnostics={}, processing_seconds=7.,
                   timings=dict(marks={'csi_map_ready': 1., 'position_available': 3., 'checked_complete': 5.}, events=[]))
    row = trial_record(point, 0, 'coverage', payload, dict(truth_npz=str(truth), truth_sha256=file_sha256(truth), input_sha256='hash'), tmp_path)
    assert row['position_error_m'] == 5.0
    assert row['clock_bias_error_ns'] == pytest.approx(2.)
    assert row['forward_valid'] is False
    assert row['localization_seconds'] == 2.0 and row['checked_seconds'] == 4.0


def test_coverage_unknown_retries_same_point_and_never_runs_solver(config, tmp_path, monkeypatch):
    calls = []
    class FakeProvider:
        def __init__(self, root):
            self.setup_root = root
        def close(self):
            pass
        def probe(self, position, seed):
            calls.append((position.copy(), seed))
            if len(calls) == 1:
                return channel.CoverageProbe('unknown', 'rt_error', position, seed, 0.1)
            return channel.CoverageProbe('covered', 'signal', position, seed, 0.1, np.ones((12, 96), complex),
                                         dict(retained_mask=np.array([True]), reflection_order=np.array([0]), diffraction_order=np.array([1])))
    def fake_bundle(probe, cfg, *, setup_root, output_root, noise_seed):
        output_root.mkdir(parents=True)
        online = output_root / 'observed.npz'
        online.write_bytes(str(noise_seed).encode())
        return dict(online_npz=str(online), truth_npz=str(online), scene_json=str(online), generation_manifest=str(online))
    monkeypatch.setattr(channel, 'make_boundary_channel', lambda cfg, root, **kw: FakeProvider(root))
    monkeypatch.setattr(channel, 'write_observation_bundle', fake_bundle)
    settings = dict(pilot_ue_count=2, formal_ue_count=3, noise_repeats=2, random_seed=31,
                    max_proposals=10, channel_backend='synthetic_fixture', legal_region={})
    with pytest.raises(RuntimeError, match='未知'):
        prepare_cohort(tmp_path, 'pilot', settings, config)
    assert not (tmp_path / 'pilot' / 'plan.json').exists()
    plan = prepare_cohort(tmp_path, 'pilot', settings, config)
    assert calls[0][1] == calls[1][1] and np.array_equal(calls[0][0], calls[1][0])
    assert plan['expected_requests'] == 8
    assert len(plan['points']) == 2
    assert len(plan['points'][0]['observations']) == 2
    assert plan['coverage_frozen_before_localization'] is True
    assert len((tmp_path / 'pilot' / 'coverage_proposals.jsonl').read_text().splitlines()) == 3


def test_committed_probe_recovers_original_setup_after_missing_ledger(config, tmp_path, monkeypatch):
    from time_bias_localization import boundary_experiment as runner
    probes = []
    class Provider:
        def __init__(self, root):
            self.setup_root = root
            root.mkdir(parents=True)
            (root / 'channel_setup.json').write_text(json.dumps({'id': str(root)}))
        def close(self):
            pass
        def probe(self, point, seed):
            probes.append(self.setup_root)
            return channel.CoverageProbe('covered', 'signal', point, seed, .1, np.ones((12, 96), complex),
                dict(retained_mask=np.array([True]), reflection_order=np.array([0]), diffraction_order=np.array([1])),
                channel_setup_sha256=file_sha256(self.setup_root / 'channel_setup.json'))
    def bundle(probe, cfg, *, setup_root, output_root, noise_seed):
        assert probe.channel_setup_sha256 == file_sha256(Path(setup_root) / 'channel_setup.json')
        output_root.mkdir(parents=True)
        path = output_root / 'test_file'
        path.write_text('not online input; no localization is invoked by preparation')
        return dict(online_npz=str(path), truth_npz=str(path), scene_json=str(path), generation_manifest=str(path))
    monkeypatch.setattr(channel, 'make_boundary_channel', lambda cfg, root, **kw: Provider(root))
    monkeypatch.setattr(channel, 'write_observation_bundle', bundle)
    original_append = runner.append_json
    def crash(*args):
        raise RuntimeError('simulated stop after probe commit')
    monkeypatch.setattr(runner, 'append_json', crash)
    settings = dict(pilot_ue_count=1, noise_repeats=1, random_seed=9, max_proposals=10,
                    channel_backend='synthetic_fixture', legal_region={})
    with pytest.raises(RuntimeError, match='simulated'):
        prepare_cohort(tmp_path, 'pilot', settings, config)
    monkeypatch.setattr(runner, 'append_json', original_append)
    plan = prepare_cohort(tmp_path, 'pilot', settings, config)
    assert len(probes) == 1  # Committed geometry must not be regenerated or replaced.
    assert plan['points'][0]['setup_root'] == str(probes[0])


def test_truth_tamper_is_rejected_before_accuracy_computation(tmp_path):
    truth = tmp_path / 'truth.npz'
    np.savez(truth, ue_position_m=[0., 0.], clock_bias_s=0.)
    observation = dict(truth_npz=str(truth), truth_sha256=file_sha256(truth))
    np.savez(truth, ue_position_m=[1., 1.], clock_bias_s=0.)
    with pytest.raises(ValueError, match='真值'):
        trial_record({}, 0, 'single', {'status': 'success', 'position_m': [0., 0.]}, observation, tmp_path)
