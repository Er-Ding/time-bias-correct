"""固定参考点 -> 纯位置聚类 -> 代表轨迹的物理与数据边界。"""

from dataclasses import asdict, fields, replace
import json

import numpy as np
import pytest

from time_bias_localization import candidates, solver
from time_bias_localization.candidates import PathObservationSample
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.forward_check import forward_check_solution
from time_bias_localization.initial_candidates import (
    InitialCandidateClusteringResult,
    InitialCandidatePoint,
    RepresentativeCandidatePoint,
    build_representative_trajectories,
    cluster_initial_candidate_points,
    generate_initial_candidate_points,
)
from time_bias_localization.raytrace2d import enumerate_specular_paths
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room


def _empty_scene():
    return Scene2D("test", (-10.0, 10.0, -10.0, 10.0), (), 1.5, 0.1, "test")


def _sample(sample_id, distance_m, *, angle=0.0, observation_id="obs"):
    return PathObservationSample(observation_id, sample_id, angle, distance_m / C)


def test_generation_and_clustering_never_build_any_trajectory(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("点生成和点聚类不能先构建轨迹")

    monkeypatch.setattr(candidates, "generate_reverse_candidates", forbidden)
    monkeypatch.setattr(candidates, "reverse_trace_sample", forbidden)
    monkeypatch.setattr(candidates, "RawReverseTrajectory", forbidden)
    monkeypatch.setattr(candidates, "CandidateTrajectory", forbidden)
    monkeypatch.setattr(solver, "CandidateTrajectory", forbidden)
    generated = generate_initial_candidate_points(
        _empty_scene(), (0.0, 0.0), [_sample("a", 1.0), _sample("b", 1.1)]
    )
    representatives = cluster_initial_candidate_points(generated.points, min_samples=1)
    assert len(representatives) == 1
    assert generated.diagnostics["builds_bias_trajectories"] is False
    forbidden_fields = {"anchor_m", "direction", "beta_min_m", "beta_max_m", "beta_interval_m"}
    assert not {field.name for field in fields(InitialCandidatePoint)} & forbidden_fields
    assert not set(asdict(representatives[0].point)) & forbidden_fields


@pytest.mark.parametrize("target, expected_position, expected_walls", [
    (2.0, (2.0, 0.0), ()),
    (8.0, (4.0, 0.0), ("right",)),
    (18.0, (2.0, 0.0), ("right", "left")),
])
def test_reference_endpoint_follows_zero_one_and_two_actual_reflections(target, expected_position, expected_walls):
    scene = replace(_empty_scene(), walls=(
        WallSegment("right", (6.0, -8.0), (6.0, 8.0)),
        WallSegment("left", (-2.0, -8.0), (-2.0, 8.0)),
    ))
    generated = generate_initial_candidate_points(scene, (0.0, 0.0), [_sample("a", target)])
    assert len(generated.points) == 1
    point = generated.points[0]
    np.testing.assert_allclose(point.position_m, expected_position, atol=1e-12)
    assert point.reflection_wall_ids == expected_walls
    assert len(point.reflection_points_m) == len(expected_walls)
    assert not generated.rejected_samples
    json.dumps(asdict(generated), allow_nan=False)


def test_nonzero_reference_bias_changes_actual_endpoint_before_any_clustering():
    reference_bias = 2.0 / C
    generated = generate_initial_candidate_points(
        _empty_scene(), (0.0, 0.0), [_sample("a", 5.0)], reference_bias_s=reference_bias
    )
    point = generated.points[0]
    np.testing.assert_allclose(point.position_m, (3.0, 0.0), atol=1e-12)
    trajectory = build_representative_trajectories(cluster_initial_candidate_points(generated.points, min_samples=1), (-5, 5))[0]
    np.testing.assert_allclose(trajectory.point(2), point.position_m, atol=1e-12)
    np.testing.assert_allclose(trajectory.point(3), (2.0, 0.0), atol=1e-12)
    assert trajectory.beta_interval == pytest.approx((-5.0, 5.0))


@pytest.mark.parametrize("target, max_reflections, reason", [
    (0.0, 2, "nonpositive_reference_length"),
    (6.0, 2, "endpoint_on_wall"),
    (6.0 + 0.5e-7, 2, "endpoint_on_wall"),
    (7.0, 0, "exceeds_max_reflections"),
    (12.0, 1, "endpoint_at_bs"),
    (14.0, 2, "endpoint_on_wall"),
    (15.0, 1, "exceeds_max_reflections"),
    (23.0, 2, "exceeds_max_reflections"),
])
def test_rejected_samples_keep_source_and_explicit_reason(target, max_reflections, reason):
    scene = replace(_empty_scene(), walls=(
        WallSegment("right", (6.0, -8.0), (6.0, 8.0)),
        WallSegment("left", (-2.0, -8.0), (-2.0, 8.0)),
    ))
    result = generate_initial_candidate_points(scene, (0, 0), [_sample("a", target)],
                                               max_reflections=max_reflections)
    assert result.points == []
    assert result.rejected_samples[0]["sample_id"] == "a"
    assert result.rejected_samples[0]["reason"] == reason
    assert result.diagnostics["input_sample_count"] == 1
    assert result.diagnostics["rejection_reason_counts"] == {reason: 1}


@pytest.mark.parametrize("origin, target, reason", [
    ((0.0, 0.0), 10.0, "endpoint_on_scene_boundary"),
    ((0.0, 0.0), 11.0, "leaves_scene_before_endpoint"),
    ((10.0, 0.0), 1.0, "leaves_scene_before_endpoint"),
])
def test_scene_boundary_blocks_wall_outside_scene(origin, target, reason):
    scene = replace(_empty_scene(), walls=(WallSegment("outside", (11.0, -2.0), (11.0, 2.0)),))
    result = generate_initial_candidate_points(scene, origin, [_sample("a", target)])
    assert not result.points
    assert result.rejected_samples[0]["reason"] == reason


def test_fixed_reference_can_explicitly_reject_sample_that_other_bias_would_make_valid():
    scene = replace(_empty_scene(), walls=(WallSegment("right", (6.0, -8.0), (6.0, 8.0)),))
    sample = _sample("a", 7.0)
    at_zero = generate_initial_candidate_points(scene, (0, 0), [sample], max_reflections=0)
    at_two_meters = generate_initial_candidate_points(scene, (0, 0), [sample], max_reflections=0,
                                                       reference_bias_s=2.0 / C)
    assert at_zero.rejected_samples[0]["reason"] == "exceeds_max_reflections"
    np.testing.assert_allclose(at_two_meters.points[0].position_m, (5.0, 0.0))


def test_clustering_uses_xy_even_when_directions_are_opposite():
    points = generate_initial_candidate_points(
        _empty_scene(), (0, 0), [_sample("a", 0.5, angle=np.pi / 2),
                                _sample("b", 0.5, angle=-np.pi / 2)]
    ).points
    assert np.dot(points[0].endpoint_direction, points[1].endpoint_direction) == pytest.approx(-1)
    representatives = cluster_initial_candidate_points(points, position_radius_m=1.1, min_samples=2)
    assert len(representatives) == 1
    assert representatives[0].point in points
    assert representatives[0].metadata["uses_direction_threshold"] is False
    assert len(build_representative_trajectories(representatives, (-5, 5))) == 1


def _points_at(positions, *, prefix="sample", observation_id="obs"):
    base = generate_initial_candidate_points(_empty_scene(), (0, 0), [_sample("base", 1.0)]).points[0]
    return [replace(base, sample_id=f"{prefix}-{index:03d}", observation_id=observation_id,
                    position_m=(float(position[0]), float(position[1])))
            for index, position in enumerate(positions)]


def test_dbscan_keeps_a_long_dense_continuous_cloud_in_one_cluster():
    points = _points_at([(x, 0) for x in np.arange(1.0, 8.01, 0.25)])
    result = cluster_initial_candidate_points(points, position_radius_m=0.6,
                                              return_diagnostics=True)
    assert isinstance(result, InitialCandidateClusteringResult)
    assert len(result.representatives) == 1
    representative = result.representatives[0]
    assert len(representative.members) == len(points)
    assert representative.metadata["maximum_member_point_distance_m"] == pytest.approx(7.0)
    assert representative.metadata["min_samples"] == 5
    assert result.diagnostics["core_point_count"] == len(points) - 4
    assert result.diagnostics["border_point_count"] == 4
    assert result.noise_points == []


def test_dbscan_keeps_disconnected_clouds_and_reports_noise_without_representatives():
    points = _points_at([(0.1 * x, y) for y in (0, 5) for x in range(5)] + [(9, 9)])
    result = cluster_initial_candidate_points(points, return_diagnostics=True)
    assert [len(item.members) for item in result.representatives] == [5, 5]
    assert result.noise_points == [points[-1]]
    assert result.diagnostics["noise_point_count"] == 1
    assert result.diagnostics["clustered_point_count"] == 10
    assert result.memberships[-1]["role"] == "noise"
    assert result.memberships[-1]["neighbor_count"] == 1
    assert result.memberships[-1]["candidate_id"] is None
    assert result.memberships[-1]["is_representative"] is False
    assert all(points[-1] not in item.members for item in result.representatives)
    assert sum(row["is_representative"] for row in result.memberships) == 2
    json.dumps(asdict(result), allow_nan=False)


def test_border_points_cannot_bridge_core_clusters_and_assignment_is_order_independent():
    # 中点同时邻近左右两个核心点，但只有三个邻点（含自身），不能连接两簇。
    left = _points_at([(x, 0) for x in (-1.5, -1.4, -1.3, -1.2, -0.9)], prefix="a")
    right = _points_at([(x, 0) for x in (0.9, 1.2, 1.3, 1.4, 1.5)], prefix="b")
    middle = _points_at([(0, 0)], prefix="z")
    points = left + right + middle
    result = cluster_initial_candidate_points(points, position_radius_m=1.0,
                                              return_diagnostics=True)
    assert [len(item.members) for item in result.representatives] == [6, 5]
    assert middle[0] in result.representatives[0].members
    assert result.memberships[-1]["role"] == "border"
    assert result.memberships[-1]["neighbor_count"] == 3
    assert result.diagnostics["core_point_count"] == 10
    rng = np.random.default_rng(4)
    for _ in range(8):
        reordered = [points[index] for index in rng.permutation(len(points))]
        other = cluster_initial_candidate_points(reordered, position_radius_m=1.0,
                                                  return_diagnostics=True)
        assert asdict(other) == asdict(result)
    # 更接近右团时应归右团，不因为左团排序靠前而归左团。
    moved = left + right + [replace(middle[0], position_m=(0.05, 0))]
    other = cluster_initial_candidate_points(moved, position_radius_m=1.0,
                                              return_diagnostics=True)
    assert [len(item.members) for item in other.representatives] == [5, 6]


def test_sparse_branches_are_reported_separately_and_density_ignores_spectral_weights():
    dense = _points_at([(0.01 * x, 0) for x in range(5)], prefix="dense")
    sparse = [replace(point, weight=1e200) for point in
              _points_at([(0, 0), (0.01, 0)], prefix="sparse", observation_id="weak-branch")]
    result = cluster_initial_candidate_points(dense + sparse, return_diagnostics=True)
    assert len(result.representatives) == 1
    assert result.noise_points == sparse
    assert result.diagnostics["group_summaries"][1]["noise_point_count"] == 2
    assert result.diagnostics["group_summaries"][1]["cluster_count"] == 0
    assert [row["neighbor_count"] for row in result.memberships[-2:]] == [2, 2]


def test_default_min_samples_includes_self_and_all_noise_is_explicit():
    points = _points_at([(0.01 * x, 0) for x in range(5)])
    assert len(cluster_initial_candidate_points(points)) == 1
    result = cluster_initial_candidate_points(points[:4], return_diagnostics=True)
    assert result.representatives == []
    assert result.noise_points == points[:4]
    assert all(row["neighbor_count"] == 4 and row["role"] == "noise"
               for row in result.memberships)
    assert result.diagnostics["cluster_count"] == 0
    assert result.diagnostics["core_point_count"] == 0
    assert result.diagnostics["border_point_count"] == 0
    assert result.diagnostics["noise_point_count"] == 4


def test_weighted_medoid_remains_a_real_member_without_weighting_density():
    points = _points_at([(0.1 * x, 0) for x in range(5)])
    points[-1] = replace(points[-1], weight=100.0)
    result = cluster_initial_candidate_points(points, return_diagnostics=True)
    assert result.representatives[0].point is points[-1]
    assert all(row["neighbor_count"] == 5 and row["role"] == "core"
               for row in result.memberships)


@pytest.mark.parametrize("kwargs", [
    {"position_radius_m": value} for value in (True, np.bool_(False), 0, -1, np.inf, np.nan, "1.5", None)
] + [
    {"min_samples": value} for value in (True, np.bool_(True), 0, -1, 2.0, np.nan, np.inf, "5", None)
])
def test_invalid_dbscan_parameters_fail_even_for_empty_input(kwargs):
    with pytest.raises(ValueError):
        cluster_initial_candidate_points([], **kwargs)


def test_representative_is_actual_medoid_and_sampling_multiplicity_does_not_inflate_solver_weight():
    points = generate_initial_candidate_points(
        _empty_scene(), (0, 0), [_sample("a", 1.0), _sample("b", 1.5), _sample("c", 2.0)]
    ).points
    representative = cluster_initial_candidate_points(points, position_radius_m=1.1, min_samples=2)[0]
    assert representative.point is points[1]
    trajectory = build_representative_trajectories([representative], (-5, 5))[0]
    assert trajectory.weight == 1.0
    assert trajectory.metadata["source_sample_ids"] == ["a", "b", "c"]
    assert trajectory.metadata["initial_point"] == asdict(points[1])
    assert trajectory.metadata["members"] == [asdict(point) for point in points]


def test_point_representative_rejects_synthetic_centroid_and_cross_source_members():
    points = generate_initial_candidate_points(
        _empty_scene(), (0, 0), [_sample("a", 1.0), _sample("b", 2.0)]
    ).points
    synthetic_centroid = replace(points[0], position_m=(1.5, 0.0))
    with pytest.raises(ValueError, match="真实成员"):
        RepresentativeCandidatePoint("fake", synthetic_centroid, tuple(points), {})
    with pytest.raises(ValueError, match="同一观测"):
        RepresentativeCandidatePoint("mixed", points[0],
                                     (points[0], replace(points[1], observation_id="other")), {})


def test_clustering_partitions_source_peak_and_full_wall_sequence_not_ambiguous_joined_name():
    point = generate_initial_candidate_points(_empty_scene(), (0, 0), [_sample("a", 1.0)]).points[0]
    points = [
        point,
        replace(point, observation_id="another"),
        replace(point, sample_id="b", topology_id="a-b-c", reflection_wall_ids=("a-b", "c"),
                reflection_points_m=((1, 1), (2, 2))),
        replace(point, sample_id="c", topology_id="a-b-c", reflection_wall_ids=("a", "b-c"),
                reflection_points_m=((1, 1), (2, 2))),
    ]
    representatives = cluster_initial_candidate_points(points, position_radius_m=100, min_samples=1)
    assert len(representatives) == 4
    assert len({item.candidate_id for item in representatives}) == 4


@pytest.mark.parametrize("backend, chunk_size", [("reference", 1), ("numpy", 1), ("numpy", 8192)])
def test_point_to_cluster_to_trajectory_recovers_known_geometry_and_shared_bias(backend, chunk_size):
    scene = make_synthetic_room()
    bs, ue = np.asarray((2.0, 7.0)), np.asarray((14.0, 4.0))
    beta, reference_bias = 2.0, 0.7e-9
    paths = enumerate_specular_paths(scene, ue, bs, max_reflections=2)
    samples = [
        PathObservationSample(f"obs-{i}", f"sample-{j}", np.deg2rad(path.arrival_aoa_deg),
                              path.delay_s + (beta + delta) / C)
        for i, path in enumerate(paths) for j, delta in enumerate((-0.01, 0.0, 0.01))
    ]
    generated = generate_initial_candidate_points(scene, bs, samples, reference_bias_s=reference_bias,
                                                   backend=backend, wall_chunk_size=chunk_size)
    assert not generated.rejected_samples
    assert len(generated.points) == len(samples)
    assert all(np.linalg.norm(np.asarray(point.position_m) - ue) > 1.0 for point in generated.points)
    representatives = cluster_initial_candidate_points(generated.points, position_radius_m=0.1, min_samples=3)
    assert len(representatives) == len(paths)
    assert all(item.point.sample_id == "sample-1" for item in representatives)
    trajectories = build_representative_trajectories(representatives, (-5, 5))
    for point, trajectory in zip(representatives, trajectories, strict=True):
        np.testing.assert_allclose(trajectory.point(reference_bias * C), point.point.position_m, atol=1e-10)
        np.testing.assert_allclose(trajectory.point(beta), ue, atol=1e-10)
    solution = solver.solve_position_and_bias(trajectories)
    np.testing.assert_allclose(solution.mu, ue, atol=1e-8)
    assert solution.beta == pytest.approx(beta, abs=1e-8)
    checked = forward_check_solution(scene, bs, solution.selected_candidates, solution.mu, solution.beta)
    assert checked["all_selected_paths_valid"]
    assert {len(row["reflection_wall_ids"]) for row in checked["paths"]} == {0, 1, 2}
    assert all(abs(row["sample_residuals"]["delay_error_ns"]) < 1e-6 for row in checked["paths"])


def test_vectorized_and_reference_endpoints_and_rejections_match_exactly():
    rng = np.random.default_rng(7)
    scene = make_synthetic_room()
    samples = [_sample(str(i), float(distance), angle=float(angle), observation_id=f"obs-{i % 3}")
               for i, (angle, distance) in enumerate(zip(rng.uniform(-np.pi, np.pi, 90),
                                                          rng.uniform(0, 100, 90), strict=True))]
    reference = generate_initial_candidate_points(scene, (2, 7), samples, backend="reference")
    vectorized = generate_initial_candidate_points(scene, (2, 7), samples, wall_chunk_size=1)
    assert vectorized.points == reference.points
    assert vectorized.rejected_samples == reference.rejected_samples
    assert vectorized.diagnostics == {**reference.diagnostics, "backend": "numpy"}


def test_empty_inputs_and_empty_clusters():
    generated = generate_initial_candidate_points(_empty_scene(), (0, 0), [])
    assert generated.points == [] and generated.rejected_samples == []
    assert cluster_initial_candidate_points([]) == []
    result = cluster_initial_candidate_points([], return_diagnostics=True)
    assert result.representatives == result.noise_points == result.memberships == []
    assert result.diagnostics["group_summaries"] == []
    assert result.diagnostics["reference_bias_s"] is None
    json.dumps(asdict(result), allow_nan=False)
    assert build_representative_trajectories([], (-1, 1)) == []


def test_duplicate_sources_mixed_references_and_out_of_range_reference_are_rejected():
    sample = _sample("a", 1.0)
    with pytest.raises(ValueError, match="重复"):
        generate_initial_candidate_points(_empty_scene(), (0, 0), [sample, sample])
    point = generate_initial_candidate_points(_empty_scene(), (0, 0), [sample]).points[0]
    with pytest.raises(ValueError, match="重复"):
        cluster_initial_candidate_points([point, point])
    with pytest.raises(ValueError, match="reference_bias_s"):
        cluster_initial_candidate_points([point, replace(point, sample_id="b", reference_bias_s=1e-9)])
    representatives = cluster_initial_candidate_points([point], min_samples=1)
    with pytest.raises(ValueError, match="参考 bias"):
        build_representative_trajectories(representatives, (1, 2))
    with pytest.raises(ValueError, match="candidate_id"):
        build_representative_trajectories(representatives * 2, (-1, 1))


@pytest.mark.parametrize("kwargs", [{"reference_bias_s": np.nan}, {"max_reflections": True},
                                    {"max_reflections": 3}, {"max_reflections": 2.0},
                                    {"backend": "invalid"}])
def test_invalid_generator_settings_fail_before_sampling(kwargs):
    with pytest.raises(ValueError):
        generate_initial_candidate_points(_empty_scene(), (0, 0), [], **kwargs)
