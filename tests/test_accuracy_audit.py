"""Check diagnostic geometry and preserve source IDs after peak suppression."""
import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[1] / 'scripts/audit_boundary_accuracy.py'
_spec = importlib.util.spec_from_file_location('accuracy_audit_test_module', _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_reference_zero_rejection_does_not_imply_rejection_at_true_bias():
    point = dict(observed_delay_s=20 / audit.C, prefix_length_m=10,
                 endpoint_free_distance_m=5, endpoint_origin_m=[2, 4], endpoint_direction=[1, 0])
    assert audit.point_at_bias(point, 0) is None
    np.testing.assert_allclose(audit.point_at_bias(point, 7 / audit.C), [5, 4])
    assert audit.point_at_bias(point, 10 / audit.C) is None


def test_support_counts_source_observations_and_reports_the_first_loss():
    two = {'a': .1, 'b': .2}
    one = {'a': .1, 'b': 8.}
    assert audit.stage_route(one, one, one, 2) == 'initial_less_than_two_nearby_observations'
    assert audit.stage_route(two, one, one, 2) == 'dbscan_lost_nearby_support'
    assert audit.stage_route(two, two, one, 2) == 'representatives_lost_nearby_support'
    assert audit.stage_route(two, two, two, 2) == 'nearby_representatives_reached_solver'


def test_suppressed_peaks_keep_original_source_id_and_coarse_window():
    coarse = [dict(aoa_rad=0., delay_s=10e-9, spectrum_value=1.) for _ in range(3)]
    coarse[2] = dict(aoa_rad=.3, delay_s=20e-9, spectrum_value=100.)
    peak = dict(aoa_rad=.31, delay_s=20.2e-9, spectrum_value=100.)
    peaks = dict(coarse=coarse, nominal=[peak], nominal_source_indices=[2])
    config = {'music': dict(angle_step_deg=1., delay_step_s=1e-9,
                           spectrum_sampling=dict(aoa_half_width_grid_steps=1.5, delay_half_width_grid_steps=1.5))}
    paths = [dict(path_index=4, aoa_local_rad=.30, geometric_delay_s=15e-9)]
    row = audit.peak_audit(peaks, config, paths, 5e-9)[0]
    assert row['observation_id'] == 'music_path_02'
    assert row['truth_paths_in_sampling_window'] == [4]
