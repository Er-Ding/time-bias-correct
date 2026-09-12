"""绕射几何、完整传播顺序、连续偏差覆盖与候选互斥的回归检查。"""
from dataclasses import replace
import math

import numpy as np
import pytest

from time_bias_localization.candidates import PathObservationSample
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.diffraction import diffraction_edges, enumerate_paths, rebuild_path
from time_bias_localization.diffraction_diagnostics import physical_constraint_rank
from time_bias_localization.forward_check import forward_check_solution
from time_bias_localization.initial_candidates import (
    InitialCandidatePoint, generate_initial_candidate_points, cluster_initial_candidate_points,
    build_representative_trajectories,
)
from time_bias_localization.scene import Scene2D, WallSegment
from time_bias_localization.solver import CandidateTrajectory, solve_position_and_bias


def screen_scene():
    return Scene2D("screen", (-10., 10., -10., 10.),
                   (WallSegment("screen", (0., -6.), (0., 0.)),
                    WallSegment("right", (8., -9.), (8., 9.))), 1.5, .1, "test")


def arc_point(index, phi, *, edge="corner", observation="obs", radius=10., free=40.):
    direction = (math.cos(phi), math.sin(phi))
    return InitialCandidatePoint(
        observation, f"s{index:03d}", edge, 0., tuple(radius * np.asarray(direction)), (), (),
        0., (radius + 5.) / C, 5., (0., 0.), direction, free,
        propagation_interactions=(("diffraction", edge),), interaction_points_m=((0., 0.),),
        parent_sample_id=f"parent{index}",
    )


def test_public_edges_exclude_collinear_and_t_junctions():
    scene = replace(screen_scene(), walls=(WallSegment("a", (0., 0.), (1., 0.)),
        WallSegment("b", (1., 0.), (2., 0.)), WallSegment("c", (.5, 0.), (.5, 2.))))
    positions = {edge.position_m for edge in diffraction_edges(scene)}
    assert (1., 0.) not in positions
    assert (.5, 0.) not in positions
    assert (0., 0.) in positions
    assert diffraction_edges(replace(scene, walls=tuple(reversed(scene.walls)))) == diffraction_edges(scene)


def test_single_diffraction_has_shadow_visibility_and_mixed_reflection_order():
    scene, bs, ue = screen_scene(), (-4., -2.), (4., -2.)
    paths = enumerate_paths(scene, ue, bs, max_reflections=1, max_diffractions=1)
    diffracted = [path for path in paths if path.diffraction_order]
    assert diffracted
    assert any(path.reflection_order for path in diffracted)
    shortest = min(diffracted, key=lambda path: path.length_m)
    assert shortest.length_m == pytest.approx(2 * math.sqrt(20))
    assert rebuild_path(scene, (-3., -3.), bs, shortest.propagation_interactions) is None
    for path in diffracted:
        rebuilt = rebuild_path(scene, ue, bs, path.propagation_interactions)
        np.testing.assert_allclose(rebuilt.nodes, path.nodes)


def test_reverse_uses_observed_samples_and_labels_hypotheses_without_trajectory_objects(monkeypatch):
    import time_bias_localization.solver as solver
    scene, bs = screen_scene(), (-4., -2.)
    target = min((p for p in enumerate_paths(scene, (4., -2.), bs, max_reflections=1, max_diffractions=1)
                  if p.diffraction_order), key=lambda p: p.length_m)
    samples = [PathObservationSample("peak", f"s{i:03d}", math.radians(target.arrival_aoa_deg), target.delay_s)
               for i in range(64)]
    monkeypatch.setattr(solver, "CandidateTrajectory", lambda *_a, **_k: pytest.fail("聚类前不能建立轨迹"))
    generated = generate_initial_candidate_points(scene, bs, samples, max_reflections=1, max_diffractions=1,
                                                   diffraction_directions_per_sample=8)
    points = [p for p in generated.points if p.has_diffraction]
    assert points and all(p.observation_id == "peak" and p.parent_sample_id for p in points)
    assert len({p.sample_id for p in generated.points}) == len(generated.points)
    assert min(np.linalg.norm(np.asarray(p.position_m) - (4., -2.)) for p in points) < .1
    for p in points[::max(1, len(points) // 12)]:
        rebuilt = rebuild_path(scene, p.position_m, bs, tuple(reversed(p.interactions)))
        assert rebuilt is not None
        assert rebuilt.length_m == pytest.approx(target.length_m, abs=1e-7)
    clustered = cluster_initial_candidate_points(points, min_samples=1, beta_interval_m=(-2., 2.), return_diagnostics=True)
    assert clustered.diagnostics["builds_bias_trajectories"] is False


def test_dbscan_keeps_edge_identity_and_full_order_and_multireps_are_one_cluster():
    points = [arc_point(i, i * .1) for i in range(12)]
    points += [arc_point(i + 20, i * .1, edge="other_corner") for i in range(12)]
    result = cluster_initial_candidate_points(points, position_radius_m=1.5, min_samples=2,
        diffraction_coverage_distance_m=1., beta_interval_m=(-2., 2.), return_diagnostics=True)
    assert result.diagnostics["cluster_count"] == 2
    assert len(result.representatives) > 2
    assert len(result.memberships) == len(points)
    assert len({p["candidate_id"] for p in result.memberships}) == 2
    assert len({r.candidate_id for r in result.representatives}) == len(result.representatives)
    for rep in result.representatives:
        assert rep.point in rep.members
        assert all(member.interactions == rep.point.interactions for member in rep.members)
    # 同样的墙编号和边缘编号，顺序不同也不能合簇。
    left = replace(points[0], reflection_wall_ids=("wall",), reflection_points_m=((1., 1.),),
                   propagation_interactions=(("reflection", "wall"), ("diffraction", "corner")),
                   interaction_points_m=((1., 1.), (0., 0.)))
    right = replace(left, sample_id="reverse_order", propagation_interactions=tuple(reversed(left.interactions)),
                    interaction_points_m=tuple(reversed(left.interaction_points_m)))
    assert len(cluster_initial_candidate_points([left, right], min_samples=1, beta_interval_m=(-2., 2.))) == 2


def test_representative_cover_checks_continuous_bias_and_preserves_each_valid_interval():
    points = [arc_point(0, 0.), arc_point(1, .1), arc_point(2, .2, free=15.)]
    result = cluster_initial_candidate_points(points, min_samples=1, position_radius_m=3.,
        diffraction_coverage_distance_m=1.1, beta_interval_m=(-20., 4.), return_diagnostics=True)
    assert result.diagnostics["cluster_count"] == 1
    assert len(result.representatives) > result.representatives[0].metadata["reference_coverage_representative_count"]
    trajectories = build_representative_trajectories(result.representatives, (-20., 4.))
    assert len({t.metadata["point_cluster_id"] for t in trajectories}) == 1
    for p in points:
        for beta in np.linspace(max(-20., 10. - p.endpoint_free_distance_m + 1e-7), 4., 157):
            actual = np.asarray(p.position_m) - beta * np.asarray(p.endpoint_direction)
            valid = [t for t in trajectories if t.is_valid(beta)]
            assert valid
            assert min(np.linalg.norm(t.point(beta) - actual) for t in valid) <= 1.1 + 1e-8
    assert all(t.weight == 1. for t in trajectories)


def test_multiple_representatives_remain_mutually_exclusive_and_two_diffractions_are_rank_two():
    p, beta = np.asarray([2., 3.]), 1.
    candidates = []
    for observation, q in [("a", np.array([0., 0.])), ("b", np.array([5., 0.]))]:
        direction = (p - q) / np.linalg.norm(p - q)
        candidates.append(CandidateTrajectory(observation, observation, p + beta * direction, direction, -2., 2.,
            metadata={"propagation_interactions": [["diffraction", observation]], "interaction_points_m": [q.tolist()]}))
    candidates.append(replace(candidates[0], candidate_id="a_extra", anchor_m=candidates[0].anchor_m + .2))
    result = solve_position_and_bias(candidates)
    assert len(result.selected_candidates) == 2
    diagnostics = physical_constraint_rank(screen_scene(), result.selected_candidates, result.mu)
    assert diagnostics["physical_constraint_count"] == 2
    assert diagnostics["local_constraint_rank"] == 2
    assert diagnostics["locally_identifiable"] is False


def test_forward_rebuilds_mixed_sequence_from_estimated_position_and_map():
    scene, bs = screen_scene(), (-4., -2.)
    target = next(p for p in enumerate_paths(scene, (4., -2.), bs, max_reflections=1, max_diffractions=1)
                  if p.diffraction_order and p.reflection_order)
    samples = [PathObservationSample("peak", f"s{i:03d}", math.radians(target.arrival_aoa_deg), target.delay_s)
               for i in range(32)]
    generated = generate_initial_candidate_points(scene, bs, samples, max_reflections=1, max_diffractions=1,
                                                   diffraction_directions_per_sample=8)
    matching = [p for p in generated.points if p.interactions == tuple(reversed(target.propagation_interactions))]
    assert matching
    reps = cluster_initial_candidate_points(matching[:1], min_samples=1, beta_interval_m=(-2., 2.))
    trajectory = build_representative_trajectories(reps, (-2., 2.))[0]
    check = forward_check_solution(scene, bs, [trajectory], trajectory.point(0.), 0., max_reflections=1, max_diffractions=1)
    assert check["all_selected_paths_valid"]
    assert check["paths"][0]["sample_residuals"]["delay_error_ns"] == pytest.approx(0., abs=1e-5)
    assert not forward_check_solution(scene, bs, [trajectory], trajectory.point(0.), 0., max_reflections=1)["all_selected_paths_valid"]


@pytest.mark.parametrize("value", [True, 2, -1, 0.5])
def test_reject_unsupported_diffraction_count(value):
    with pytest.raises(ValueError):
        generate_initial_candidate_points(screen_scene(), (-4., -2.), [], max_diffractions=value)
