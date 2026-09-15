"""连续求解的物理可解性、量化误差隔离和失败状态检查。

这里用受控真值合成观测，真值只存在于测试；生产求解器不读取它。
"""
from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.continuous_solver import (
    ContinuousSolverConfig, evaluate_continuous_state, solve_continuous_position_and_bias,
)
from time_bias_localization.diffraction import diffraction_edges, enumerate_paths
from time_bias_localization.propagation_hypotheses import HypothesisBank, build_hypothesis_bank
from time_bias_localization.propagation_model import (
    ContinuousObservation, evaluate_hypothesis, make_hypothesis,
)
from time_bias_localization.scene import Scene2D, WallSegment


TRUE_XY = np.array([4.37, -1.63])
TRUE_BETA = 2.731
BS = (-4.0, -2.0)


def _screen(*, bottom=False):
    walls = (WallSegment("screen", (0.0, -6.0), (0.0, 0.0)),
             WallSegment("right", (8.0, -9.0), (8.0, 9.0)))
    if bottom:
        walls += (WallSegment("bottom", (-9.0, -8.0), (9.0, -8.0)),)
    return Scene2D("continuous_screen", (-10.0, 10.0, -10.0, 10.0), walls, 1.5, 0.1, "test")


def _bank_from_routes(scene, routes, *, xy=TRUE_XY, beta=TRUE_BETA,
                      all_routes_for_every_observation=False):
    hypotheses = tuple(make_hypothesis(scene, BS, route) for route in routes)
    values = [evaluate_hypothesis(scene, hypothesis, xy) for hypothesis in hypotheses]
    assert all(value.valid for value in values)
    observations = tuple(ContinuousObservation(f"peak_{i}", value.aoa_rad,
                            value.length_m + beta, math.radians(0.2), 0.05)
                         for i, value in enumerate(values))
    indices = tuple(tuple(range(len(hypotheses))) if all_routes_for_every_observation else (i,)
                    for i in range(len(observations)))
    return HypothesisBank(scene, BS, observations, hypotheses, indices,
                           {"complete_within_configured_orders": True,
                            "source": "controlled_known_routes_for_numerical_diagnostic_only"})


def _diffraction_bank(count=4):
    scene = _screen()
    paths = enumerate_paths(scene, TRUE_XY, BS, max_reflections=1, max_diffractions=1)
    routes = [p.propagation_interactions for p in paths if p.diffraction_order][:count]
    assert len(routes) == count
    return _bank_from_routes(scene, routes)


def _config(**kwargs):
    values = dict(xy_bounds_m=((-9.0, 9.0), (-7.5, 8.0)),
                  bias_bounds_m=(-8.0, 8.0), max_starts=16,
                  max_iterations=60, max_association_iterations=5,
                  max_seed_combinations=2000, ambiguity_cost_tolerance=1e-6)
    values.update(kwargs)
    return ContinuousSolverConfig(**values)


def test_noiseless_off_grid_diffraction_recovers_continuous_position_and_bias():
    bank = _diffraction_bank()
    result = solve_continuous_position_and_bias(bank, _config())
    assert result.status == "success", result.to_dict()
    np.testing.assert_allclose(result.position_m, TRUE_XY, atol=1e-6)
    assert result.beta_m == pytest.approx(TRUE_BETA, abs=1e-6)
    assert result.clock_bias_s == pytest.approx(TRUE_BETA / 299792458.0, abs=1e-14)
    assert result.best_candidate["physical_rank"] == 3
    assert result.best_candidate["objective"] < 1e-12
    # The true direction deliberately falls between a coarse direction grid.
    edge = np.asarray(bank.hypotheses[0].anchor_m)
    radius = np.linalg.norm(TRUE_XY - edge)
    true_angle = math.atan2(*(TRUE_XY - edge)[::-1])
    coarse_angle = round(true_angle / math.radians(15)) * math.radians(15)
    old_point = edge + radius * np.array([math.cos(coarse_angle), math.sin(coarse_angle)])
    assert np.linalg.norm(TRUE_XY - old_point) > 0.2
    assert not result.diagnostics["seed_search"]["legacy_candidates_used"]
    assert not result.best_candidate["covariance"]["statistically_calibrated"]
    assert result.best_candidate["full_path_set_validation"] == "not_performed"


def test_two_pure_diffraction_ranges_cannot_determine_three_unknowns():
    bank = _diffraction_bank(2)
    result = solve_continuous_position_and_bias(
        bank, _config(max_starts=1), initial_states=[[*TRUE_XY, TRUE_BETA]])
    assert result.status == "insufficient_constraints"
    assert result.position_m is None
    assert result.best_candidate["physical_rank"] == 2
    assert result.best_candidate["covariance"] is None


def test_different_prefixes_with_same_equivalent_corner_add_no_range_rank():
    scene = _screen(bottom=True)
    paths = enumerate_paths(scene, TRUE_XY, BS, max_reflections=1, max_diffractions=1)
    routes = [p.propagation_interactions for p in paths
              if p.diffraction_order and p.propagation_interactions[0][0] == "diffraction"]
    bank = _bank_from_routes(scene, routes)
    first = bank.hypotheses[0].equivalent_anchor_m
    keep = [i for i, h in enumerate(bank.hypotheses)
            if np.allclose(h.equivalent_anchor_m, first)]
    assert len(keep) >= 2
    selected = tuple(bank.hypotheses[i].interactions for i in keep)
    bank = _bank_from_routes(scene, selected)
    validation = evaluate_continuous_state(bank, _config(), [*TRUE_XY, TRUE_BETA])
    assert validation["physical_rank"] == 1
    assert not validation["acceptable"]


def test_duplicate_peaks_cannot_reuse_one_physical_path():
    bank = _diffraction_bank(1)
    first = bank.observations[0]
    observations = (first, replace(first, observation_id="duplicate_peak"))
    bank = replace(bank, observations=observations, observation_hypothesis_indices=((0,), (0,)))
    validation = evaluate_continuous_state(bank, _config(), [*TRUE_XY, TRUE_BETA])
    assert validation["matched_observation_count"] == 1
    assert validation["objective"] == pytest.approx(_config().unmatched_cost)
    assert validation["physical_rank"] == 1
    assert not validation["acceptable"]


def test_one_peak_with_multiple_explanations_still_counts_once():
    bank = _diffraction_bank(2)
    bank = replace(bank, observations=bank.observations[:1],
                   observation_hypothesis_indices=((0, 1),))
    value = evaluate_continuous_state(bank, _config(), [*TRUE_XY, TRUE_BETA])
    assert value["matched_observation_count"] == 1
    assert len(value["selected_paths"]) == 1


def test_all_geometrically_invalid_branches_never_succeed():
    scene = _screen()
    hypothesis = make_hypothesis(scene, BS, ())
    observations = (ContinuousObservation("blocked", 0.0, 9.0, 0.1, 0.1),)
    bank = HypothesisBank(scene, BS, observations, (hypothesis,), ((0,),), {})
    config = _config(xy_bounds_m=((3.0, 5.0), (-3.0, -1.0)), max_starts=2,
                     min_observations=1)
    result = solve_continuous_position_and_bias(bank, config)
    assert result.status == "search_exhausted"
    assert result.position_m is None
    assert result.best_candidate["matched_observation_count"] == 0
    assert result.best_candidate["objective"] == config.unmatched_cost
    assert not result.diagnostics["any_legal_path_seen"]


def test_specular_and_diffraction_mixed_functions_recover_joint_state():
    scene = _screen(bottom=True)
    paths = enumerate_paths(scene, TRUE_XY, BS, max_reflections=1, max_diffractions=1)
    specular = next(p for p in paths if not p.diffraction_order)
    diffraction = next(p for p in paths if p.diffraction_order)
    routes = (tuple(("reflection", wall) for wall in specular.interaction_wall_ids),
              diffraction.propagation_interactions)
    bank = _bank_from_routes(scene, routes)
    result = solve_continuous_position_and_bias(bank, _config(max_starts=8))
    assert result.status == "success", result.to_dict()
    np.testing.assert_allclose(result.position_m, TRUE_XY, atol=1e-5)
    assert result.beta_m == pytest.approx(TRUE_BETA, abs=1e-5)


def test_direct_and_reflection_use_original_angle_residual_and_wrap_pi():
    scene = Scene2D("reflector", (-12., 12., -12., 12.),
                    (WallSegment("bottom", (-11., -8.), (11., -8.)),), 1.5, .1, "test")
    xy = np.array([-7.13, -2.01])
    bank = _bank_from_routes(scene, ((), (("reflection", "bottom"),)), xy=xy)
    bank = replace(bank, observations=(replace(bank.observations[0],
                    aoa_rad=bank.observations[0].aoa_rad + 2 * np.pi), bank.observations[1]))
    result = solve_continuous_position_and_bias(bank, _config(max_starts=8))
    assert result.status == "success", result.to_dict()
    np.testing.assert_allclose(result.position_m, xy, atol=1e-5)
    assert result.best_candidate["physical_rank"] == 3


def test_public_bank_runs_independently_of_old_point_and_cluster_modules(monkeypatch):
    scene = _screen(bottom=True)
    known = _diffraction_bank()
    # Direct map discovery receives original observations, not known routes.
    bank = build_hypothesis_bank(scene, BS, known.observations,
                                max_reflections=1, max_diffractions=1,
                                aoa_gate_rad=math.radians(1), beta_interval_m=(-8, 8))
    import time_bias_localization.initial_candidates as old
    monkeypatch.setattr(old, "generate_initial_candidate_points",
                        lambda *_a, **_k: pytest.fail("new solver called legacy point generation"))
    monkeypatch.setattr(old, "cluster_initial_candidate_points",
                        lambda *_a, **_k: pytest.fail("new solver called clustering"))
    result = solve_continuous_position_and_bias(bank, _config(max_starts=12))
    assert result.status == "success", result.to_dict()
    np.testing.assert_allclose(result.position_m, TRUE_XY, atol=1e-5)


def test_full_public_bias_interval_has_no_reference_ray_restriction():
    bank = _diffraction_bank()
    beta = -123.4
    bank = replace(bank, observations=tuple(replace(o, observed_length_m=o.observed_length_m
                          + beta - TRUE_BETA) for o in bank.observations))
    config = _config(bias_bounds_m=(-150, -100), max_starts=8)
    result = solve_continuous_position_and_bias(bank, config)
    assert result.status == "success", result.to_dict()
    np.testing.assert_allclose(result.position_m, TRUE_XY, atol=1e-5)
    assert result.beta_m == pytest.approx(beta, abs=1e-5)


def test_duplicate_observation_ids_and_physical_route_aliases_are_rejected():
    bank = _diffraction_bank(2)
    with pytest.raises(ValueError, match="观测编号"):
        solve_continuous_position_and_bias(replace(bank,
            observations=(bank.observations[0], bank.observations[0])), _config())
    with pytest.raises(ValueError, match="重复物理路线"):
        solve_continuous_position_and_bias(replace(bank,
            hypotheses=(bank.hypotheses[0], replace(bank.hypotheses[0], hypothesis_id="alias"))), _config())


def test_group_huber_and_unmatched_fixed_cost_are_reported():
    bank = _diffraction_bank()
    outlier = ContinuousObservation("outlier", 1.345, 1000., .01, .05)
    bank = replace(bank, observations=(*bank.observations, outlier),
                   observation_hypothesis_indices=(*bank.observation_hypothesis_indices, (0, 1, 2, 3)))
    result = solve_continuous_position_and_bias(bank, _config(max_starts=8))
    assert result.status == "success", result.to_dict()
    assert result.best_candidate["objective"] == pytest.approx(_config().unmatched_cost, abs=1e-8)
    assert result.best_candidate["unmatched_observation_ids"] == ["outlier"]


def test_two_distinct_legal_full_rank_solutions_are_ambiguous():
    scene = Scene2D("ambiguous_corners", (-10., 15., -10., 15.), (
        WallSegment("a", (0., 0.), (.2, 0.)),
        WallSegment("b", (3., 0.), (2.85, .15)),
        WallSegment("c", (0., 3.), (.2, 3.))), 1.5, .1, "test")
    bs, true_xy, true_beta = (5., 5.), np.array([.5, -1.]), 3.
    anchors = {(0., 0.), (3., 0.), (0., 3.)}
    edges = [edge for edge in diffraction_edges(scene) if edge.position_m in anchors]
    hypotheses = tuple(make_hypothesis(scene, bs, (("diffraction", edge.edge_id),)) for edge in edges)
    evaluations = [evaluate_hypothesis(scene, h, true_xy) for h in hypotheses]
    assert len(evaluations) == 3 and all(value.valid for value in evaluations)
    observations = tuple(ContinuousObservation(str(i), value.aoa_rad,
                        value.length_m + true_beta, .001, .01)
                         for i, value in enumerate(evaluations))
    bank = HypothesisBank(scene, bs, observations, hypotheses, ((0,), (1,), (2,)), {})
    config = _config(xy_bounds_m=((-9., 14.), (-9., 14.)), bias_bounds_m=(-2., 5.), max_starts=8)
    result = solve_continuous_position_and_bias(bank, config)
    assert result.status == "ambiguous", result.to_dict()
    assert result.position_m is None and result.beta_m is None
    solutions = [result.best_candidate, *result.alternatives]
    assert all(candidate["physical_rank"] == 3 for candidate in solutions)
    assert min(np.linalg.norm(np.asarray(item["position_m"]) - true_xy) for item in solutions) < 1e-5
    other_xy = np.array([-1.46811095, -4.64123065])
    assert min(np.linalg.norm(np.asarray(item["position_m"]) - other_xy) for item in solutions) < 1e-5
    assert result.diagnostics["competitive_alternative_count"] >= 1


def test_empty_incomplete_bank_is_budget_failure_not_absence_of_physical_routes():
    original = _diffraction_bank(2)
    bank = replace(original, hypotheses=(), observation_hypothesis_indices=((), ()),
                   search_report={"budget_exhausted": True, "unsearched_sequences": 123})
    result = solve_continuous_position_and_bias(bank, _config(max_starts=1))
    assert result.status == "search_exhausted"
    assert result.diagnostics["hypothesis_search_incomplete"]


@pytest.mark.parametrize("boundary", ["bias", "position"])
def test_noisy_optimum_on_public_bound_matches_independent_bounded_fit(boundary):
    from scipy.optimize import least_squares

    scene = Scene2D("bounded_reflector", (-12., 12., -12., 12.),
                    (WallSegment("bottom", (-11., -8.), (11., -8.)),), 1.5, .1, "test")
    # A small common delay perturbation puts the unconstrained optimum beyond
    # the allowed beta maximum. The other case restricts x just below the
    # observed position. Both require an actual constrained boundary optimum.
    beta = 8.05 if boundary == "bias" else TRUE_BETA
    bank = _bank_from_routes(scene, ((), (("reflection", "bottom"),)), beta=beta)
    config = _config(xy_bounds_m=((-9., 4.35 if boundary == "position" else 9.), (-7.5, 8.)),
                     bias_bounds_m=(-8., 8.), max_starts=4)
    low = np.array([config.xy_bounds_m[0][0], config.xy_bounds_m[1][0], -8.])
    high = np.array([config.xy_bounds_m[0][1], config.xy_bounds_m[1][1], 8.])

    def residual(state):
        values = []
        for observation, hypothesis in zip(bank.observations, bank.hypotheses):
            value = evaluate_hypothesis(scene, hypothesis, state[:2], check_validity=False)
            angle = (value.aoa_rad - observation.aoa_rad + np.pi) % (2 * np.pi) - np.pi
            values.extend([angle / observation.angle_scale_rad,
                           (value.length_m + state[2] - observation.observed_length_m)
                           / observation.length_scale_m])
        return np.asarray(values)

    initial = np.array([4.2, -1.5, 7.8 if boundary == "bias" else TRUE_BETA])
    reference = least_squares(residual, initial, bounds=(low, high),
                              xtol=1e-13, ftol=1e-13, gtol=1e-13)
    boundary_index = 2 if boundary == "bias" else 0
    assert reference.x[boundary_index] == pytest.approx(high[boundary_index], abs=1e-7)
    assert np.max(np.linalg.norm(residual(reference.x).reshape(-1, 2), axis=1)) < config.huber_delta
    result = solve_continuous_position_and_bias(bank, config, initial_states=[initial])
    assert result.status == "success", result.to_dict()
    np.testing.assert_allclose([*result.position_m, result.beta_m], reference.x, atol=1e-5)
    assert result.best_candidate["all_selected_paths_valid"]


def _schedule_fixture(*, branches_per_type=40, shared=False, budget=60):
    """Only tests finite-search scheduling; no synthetic map is presented as valid."""
    hypotheses, fixed_angles, rows = [], [], []
    for observation in range(1 if shared else 3):
        indices = []
        for diffracted in (False, True):
            for index in range(branches_per_type):
                branch = len(hypotheses)
                pattern = (("diffraction",) if diffracted else ("reflection",))
                if index % 2:
                    pattern = ("reflection", *pattern)
                hypotheses.append(SimpleNamespace(interactions=tuple(
                    (kind, f"object_{branch}_{part}") for part, kind in enumerate(pattern))))
                fixed_angles.append(0.0 if diffracted else np.nan)
                indices.append(branch)
        rows.append(indices)
    if shared:
        rows *= 3
    allowed = np.zeros((3, len(hypotheses)), bool)
    for i, indices in enumerate(rows):
        allowed[i, indices] = True
    return SimpleNamespace(n=3, hypotheses=tuple(hypotheses),
                           fixed_angles=np.asarray(fixed_angles), allowed=allowed,
                           config=SimpleNamespace(max_seed_combinations=budget))


def test_seed_budget_reaches_all_observation_pairs_and_equation_types():
    from time_bias_localization.continuous_solver import _seed_combination_schedule

    problem = _schedule_fixture()
    schedule, report = _seed_combination_schedule(problem, np.random.default_rng(42))
    attempts = list(schedule)
    assert len(attempts) == 60
    for kind in ("specular_specular", "specular_diffraction", "diffraction_triple"):
        assert report["types"][kind]["attempted_combinations"] > 0
    ss_pairs = {tuple(i for i, _ in entries) for kind, entries in attempts
                if kind == "specular_specular"}
    assert ss_pairs == {(0, 1), (0, 2), (1, 2)}
    # A large first family cannot consume the whole quota while a later
    # observation pair or reflection order receives none.
    ss_families = [row for row in report["families"] if row["type"] == "specular_specular"]
    family_counts = [row["attempted_combinations"] for row in ss_families]
    assert min(family_counts) > 0 and max(family_counts) - min(family_counts) <= 1
    assert report["diffraction_pair_attempts"] == 0
    assert sum(row["attempted_combinations"] for row in report["families"]) == len(attempts)


def test_seed_schedule_counts_exactly_and_never_reuses_a_physical_branch():
    from time_bias_localization.continuous_solver import _seed_combination_schedule

    problem = _schedule_fixture(branches_per_type=4, shared=True, budget=1000)
    schedule, report = _seed_combination_schedule(problem, np.random.default_rng(7))
    attempts = list(schedule)
    # Three observation pairs * 4*3 distinct SS choices, six ordered SD
    # pairs * 4*4 choices, and one triple * 4*3*2 distinct D choices.
    expected = {"specular_specular": 36, "specular_diffraction": 96, "diffraction_triple": 24}
    assert len(attempts) == sum(expected.values())
    assert len({(kind, entries) for kind, entries in attempts}) == len(attempts)
    for kind, entries in attempts:
        assert len({j for _, j in entries}) == len(entries)
        assert len({i for i, _ in entries}) == len(entries)
        if len(entries) == 2:
            assert any(np.isnan(problem.fixed_angles[j]) for _, j in entries)
    for kind, count in expected.items():
        assert report["types"][kind] == {"possible_combinations": count, "attempted_combinations": count}


def test_bounded_cartesian_search_explores_both_axes_before_first_row_exhaustion():
    from itertools import islice
    from time_bias_localization.continuous_solver import _permuted_combinations

    pairs = list(islice(_permuted_combinations((range(100), range(100, 200)),
                                              np.random.default_rng(3)), 20))
    assert len({pair[0] for pair in pairs}) >= 15
    assert len({pair[1] for pair in pairs}) >= 15
