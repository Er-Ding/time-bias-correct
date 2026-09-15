"""连续长度、角度及物理导数必须与独立路径重建一致。"""
from dataclasses import replace
import math

import numpy as np
import pytest

from time_bias_localization.diffraction import diffraction_edges, enumerate_paths, rebuild_path
from time_bias_localization.propagation_hypotheses import build_hypothesis_bank
from time_bias_localization.propagation_model import (
    ContinuousObservation, evaluate_hypothesis, make_hypothesis, wrap_angle_rad,
)
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room


def _screen_scene():
    return Scene2D("screen", (-10., 10., -10., 10.),
                   (WallSegment("screen", (0., -6.), (0., 0.)),
                    WallSegment("right", (8., -9.), (8., 9.))),
                   1.5, .1, "continuous_geometry_test")


def _assert_derivatives(scene, hypothesis, position):
    value = evaluate_hypothesis(scene, hypothesis, position, check_validity=False)
    numerical_length = []
    numerical_angle = []
    for axis in np.eye(2):
        plus = evaluate_hypothesis(scene, hypothesis, np.asarray(position) + 1e-5 * axis,
                                   check_validity=False)
        minus = evaluate_hypothesis(scene, hypothesis, np.asarray(position) - 1e-5 * axis,
                                    check_validity=False)
        numerical_length.append((plus.length_m - minus.length_m) / 2e-5)
        numerical_angle.append(float(wrap_angle_rad(plus.aoa_rad - minus.aoa_rad)) / 2e-5)
    np.testing.assert_allclose(value.length_gradient_xy, numerical_length, atol=2e-8)
    np.testing.assert_allclose(value.aoa_gradient_xy, numerical_angle, atol=2e-8)


def test_direct_and_two_reflection_functions_agree_with_rebuilt_geometry():
    scene, ue, bs = make_synthetic_room(), (7.3, 5.1), (2.1, 3.2)
    paths = enumerate_paths(scene, ue, bs, max_reflections=2)
    assert {path.reflection_order for path in paths} == {0, 1, 2}
    for path in paths:
        interactions = tuple(("reflection", key) for key in path.interaction_wall_ids)
        hypothesis = make_hypothesis(scene, bs, interactions)
        evaluated = evaluate_hypothesis(scene, hypothesis, ue)
        assert evaluated.valid, evaluated.invalid_reason
        assert evaluated.length_m == pytest.approx(path.length_m, abs=1e-10)
        assert float(wrap_angle_rad(evaluated.aoa_rad - math.radians(path.arrival_aoa_deg))) == pytest.approx(0, abs=1e-12)
        _assert_derivatives(scene, hypothesis, ue)


@pytest.mark.parametrize("ue,bs", [((4., -2.), (-4., -2.)), ((-4., -2.), (4., -2.))])
def test_diffraction_and_mixed_paths_have_correct_length_and_zero_angle_derivative(ue, bs):
    scene = _screen_scene()
    paths = [path for path in enumerate_paths(scene, ue, bs, max_reflections=2, max_diffractions=1)
             if path.diffraction_order]
    assert any(path.reflection_order for path in paths)
    for path in paths:
        hypothesis = make_hypothesis(scene, bs, path.propagation_interactions)
        evaluated = evaluate_hypothesis(scene, hypothesis, ue)
        assert evaluated.valid, evaluated.invalid_reason
        assert evaluated.length_m == pytest.approx(path.length_m, abs=1e-10)
        np.testing.assert_array_equal(evaluated.aoa_gradient_xy, [0., 0.])
        _assert_derivatives(scene, hypothesis, ue)


def test_diffraction_is_continuous_on_a_legal_arc_without_any_direction_samples():
    scene, bs = _screen_scene(), (-4., -2.)
    edge = next(edge for edge in diffraction_edges(scene) if edge.position_m == (0., 0.))
    hypothesis = make_hypothesis(scene, bs, (("diffraction", edge.edge_id),))
    radius = math.sqrt(20)
    lengths = []
    for phi in np.deg2rad([-17.3, -31.7, -62.9]):
        ue = radius * np.asarray([math.cos(phi), math.sin(phi)])
        value = evaluate_hypothesis(scene, hypothesis, ue)
        assert value.valid
        lengths.append(value.length_m)
    np.testing.assert_allclose(lengths, 2 * radius, atol=1e-12)


def test_invalid_path_keeps_finite_extension_and_explicit_reason():
    scene, bs = _screen_scene(), (-4., -2.)
    direct = make_hypothesis(scene, bs)
    blocked = evaluate_hypothesis(scene, direct, (4., -2.))
    assert not blocked.valid
    assert blocked.invalid_reason == "blocked_segment"
    assert math.isfinite(blocked.length_m)
    assert np.all(np.isfinite(blocked.length_gradient_xy))
    unchecked = evaluate_hypothesis(scene, direct, (4., -2.), check_validity=False)
    assert unchecked.valid is None
    assert unchecked.length_m == blocked.length_m
    assert evaluate_hypothesis(scene, direct, (20., -2.)).invalid_reason == "ue_outside_map"
    singular = evaluate_hypothesis(scene, direct, bs)
    assert not singular.valid
    assert singular.invalid_reason == "singular_endpoint"


def test_invalid_diffraction_shadow_is_rejected_after_ue_moves():
    scene, bs = _screen_scene(), (-4., -2.)
    edge = next(edge for edge in diffraction_edges(scene) if edge.position_m == (0., 0.))
    hypothesis = make_hypothesis(scene, bs, (("diffraction", edge.edge_id),))
    assert evaluate_hypothesis(scene, hypothesis, (4., -2.)).valid
    invalid = evaluate_hypothesis(scene, hypothesis, (-3., -3.))
    assert not invalid.valid
    assert invalid.invalid_reason == "diffraction_outside_shadow_or_degenerate"


def test_complete_bank_preserves_all_legal_routes_and_records_exact_enumeration_counts():
    scene, bs, ue = _screen_scene(), (-4., -2.), (4., -2.)
    observation = ContinuousObservation("peak", .4, 20., .1, 1.)
    bank = build_hypothesis_bank(scene, bs, [observation], max_hypotheses=10000,
                                 max_enumerated_sequences=10000)
    report = bank.search_report
    n, e = len(scene.walls), len(diffraction_edges(scene))
    expected = 1 + n + n * (n - 1) + e * (1 + 2 * n + 2 * n * (n - 1) + n * n)
    assert report["enumerated_sequences"] == expected
    assert report["total_sequences"] == expected
    assert report["complete_within_configured_orders"]
    assert not report["budget_exhausted"]
    actual = {hypothesis.interactions for hypothesis in bank.hypotheses
              if evaluate_hypothesis(scene, hypothesis, ue).valid}
    expected_paths = enumerate_paths(scene, ue, bs, max_reflections=2, max_diffractions=1)
    expected_sequences = {path.propagation_interactions or tuple(("reflection", key)
                          for key in path.interaction_wall_ids) for path in expected_paths}
    assert expected_sequences <= actual
    # Reference enumeration historically excludes this split; a same wall on the
    # two sides of a corner must still be constructed and checked independently.
    opposite_bank = build_hypothesis_bank(scene, (4., -2.), [observation],
                                          max_hypotheses=10000,
                                          max_enumerated_sequences=10000)
    assert any(h.interactions[0] == h.interactions[-1]
               for h in opposite_bank.hypotheses if len(h.interactions) == 3
               and h.interactions[1][0] == "diffraction")


def test_budget_is_shared_between_families_and_omissions_are_explicit():
    scene = _screen_scene()
    observation = ContinuousObservation("peak", .4, 20., .1, 1.)
    bank = build_hypothesis_bank(scene, (-4., -2.), [observation],
                                 max_enumerated_sequences=9, max_hypotheses=1000)
    report = bank.search_report
    assert report["enumerated_sequences"] == 9
    assert all(family["enumerated_sequences"] == 1 for family in report["families"])
    assert report["unsearched_sequences"] == report["total_sequences"] - 9
    assert report["budget_exhausted"]
    assert report["stop_reason"] == "max_enumerated_sequences"
    assert sum(family["unsearched_sequences"] for family in report["unsearched_families"]) == report["unsearched_sequences"]


def test_observation_bounds_preserve_true_paths_with_bias_and_angle_wrap():
    scene, bs, ue, beta = _screen_scene(), (-4., -2.), (4., -2.), -3.
    paths = enumerate_paths(scene, ue, bs, max_reflections=2, max_diffractions=1)
    observations = [ContinuousObservation(str(i), math.radians(path.arrival_aoa_deg) + 2 * math.pi,
                                          path.length_m + beta, .01, .1)
                    for i, path in enumerate(paths)]
    bank = build_hypothesis_bank(scene, bs, observations, max_hypotheses=10000,
                                 aoa_gate_rad=1e-5, beta_interval_m=(-4., 2.),
                                 length_gate_sigma=0.)
    for path, indices in zip(paths, bank.observation_hypothesis_indices):
        expected = path.propagation_interactions or tuple(("reflection", key) for key in path.interaction_wall_ids)
        assert expected in {bank.hypotheses[index].interactions for index in indices}


def test_empty_map_and_impossible_length_have_explicit_bank_results():
    scene = replace(_screen_scene(), walls=())
    observation = ContinuousObservation("peak", 0., -100., .1, .1)
    bank = build_hypothesis_bank(scene, (0., 0.), [observation], beta_interval_m=(0., 1.))
    assert bank.hypotheses == ()
    assert bank.search_report["total_sequences"] == 1
    assert bank.search_report["observations_without_hypotheses"] == ["peak"]
    assert bank.search_report["complete_within_configured_orders"]


def test_bank_cannot_depend_on_legacy_points_clusters_or_representatives(monkeypatch):
    import builtins
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.rsplit(".", 1)[-1] in {"initial_candidates", "bias_interval_candidates", "ransac_solver",
                                      "representative_cover", "spectrum_sampling"}:
            pytest.fail("连续分支构造不能导入旧候选点、聚类、代表或 RANSAC")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    bank = build_hypothesis_bank(_screen_scene(), (-4., -2.),
                                 [ContinuousObservation("peak", 0., 10., .1, .5)])
    assert bank.hypotheses


@pytest.mark.parametrize("kwargs", [{"max_reflections": True}, {"max_diffractions": 2},
                                   {"max_hypotheses": 0}, {"max_enumerated_sequences": 0},
                                   {"aoa_gate_rad": 0.}, {"beta_interval_m": (2., 1.)}])
def test_invalid_search_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        build_hypothesis_bank(_screen_scene(), (-4., -2.),
                              [ContinuousObservation("peak", 0., 10., .1, .5)], **kwargs)
