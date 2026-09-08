from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np

from time_bias_localization.candidates import (
    PathObservationSample,
    RawReverseTrajectory,
    cluster_reverse_candidates,
    generate_reverse_candidates,
)
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S
from time_bias_localization.raytrace2d import enumerate_specular_paths
from time_bias_localization.scene import (
    Scene2D,
    make_synthetic_room,
    preprocess_deepmimo_scene,
    preprocess_sionna_exported_scene,
    ray_segment_intersection,
)
from time_bias_localization.solver import solve_position_and_bias


def _raw_candidate(
    *,
    observation_id: str = "obs",
    sample_id: str,
    topology_id: str = "wall-a",
    anchor_m: tuple[float, float] = (0.0, 0.0),
    direction: tuple[float, float] = (1.0, 0.0),
    beta_interval_m: tuple[float, float] = (0.0, 10.0),
) -> RawReverseTrajectory:
    return RawReverseTrajectory(
        observation_id=observation_id,
        sample_id=sample_id,
        topology_id=topology_id,
        anchor_m=anchor_m,
        direction=direction,
        beta_min_m=beta_interval_m[0],
        beta_max_m=beta_interval_m[1],
        prefix_length_m=0.0,
        endpoint_origin_m=(0.0, 0.0),
        reflection_wall_ids=() if topology_id == "los" else (topology_id,),
        reflection_points_m=(),
        observed_aoa_global_rad=0.0,
        observed_delay_s=1e-7,
        weight=1.0,
    )


def test_scene_image_and_coordinate_roundtrip(tmp_path) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.1)
    artifacts = scene.save(tmp_path)
    loaded = Scene2D.load(artifacts["scene_json"])
    point = (4.25, 8.75)
    assert np.allclose(loaded.pixel_to_metric(loaded.metric_to_pixel(point)), point)
    occupancy = np.load(artifacts["occupancy_npy"])
    assert occupancy.shape == (loaded.height_px, loaded.width_px)
    assert np.min(occupancy) == 0


def test_deepmimo_vertical_faces_become_walls() -> None:
    vertical = SimpleNamespace(
        vertices=np.asarray([[0, 0, 0], [5, 0, 0], [5, 0, 3], [0, 0, 3]]),
        normal=np.asarray([0, 1, 0]),
    )
    roof = SimpleNamespace(
        vertices=np.asarray([[0, 0, 3], [5, 0, 3], [5, 5, 3], [0, 5, 3]]),
        normal=np.asarray([0, 0, 1]),
    )
    scene_3d = SimpleNamespace(
        objects=[SimpleNamespace(name="building", faces=[vertical, roof])]
    )
    scene_2d = preprocess_deepmimo_scene(
        scene_3d,
        name="fake",
        fixed_height_m=1.5,
        bev_resolution_m=0.1,
        bounds_m=(-1, 6, -1, 2),
    )
    assert len(scene_2d.walls) == 1
    assert np.isclose(scene_2d.walls[0].length_m, 5.0)


def test_sionna_triangle_slice_does_not_connect_disjoint_components(tmp_path) -> None:
    # 两段墙故意放在同一个导出对象里。凸包会虚构 x=1 到 x=3 的连接墙，
    # 对原始三角形逐面切片则只会保留两段真实墙。
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 3.0],
            [0.0, 0.0, 3.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 3.0],
            [3.0, 0.0, 3.0],
        ]
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int64
    )
    for filename, value in (
        ("sionna_vertices.pkl", vertices),
        ("sionna_faces.pkl", faces),
        ("sionna_objects.pkl", {"two_disjoint_walls": (0, 8)}),
    ):
        with (tmp_path / filename).open("wb") as handle:
            pickle.dump(value, handle)

    scene = preprocess_sionna_exported_scene(
        tmp_path,
        name="raw_mesh",
        fixed_height_m=1.5,
        bev_resolution_m=0.1,
        bounds_m=(-1.0, 5.0, -1.0, 1.0),
    )

    assert scene.source == "sionna_exported_triangle_mesh"
    assert {wall.source_object for wall in scene.walls} == {"two_disjoint_walls"}
    assert [wall.wall_id for wall in scene.walls] == [
        "sionna_0000_000000",
        "sionna_0000_000001",
        "sionna_0000_000002",
        "sionna_0000_000003",
    ]
    assert max(wall.length_m for wall in scene.walls) <= 0.5 + 1e-12
    assert all(
        ray_segment_intersection((2.0, -1.0), (0.0, 1.0), wall) is None
        for wall in scene.walls
    )


def test_true_path_trajectories_recover_position_and_bias() -> None:
    scene = make_synthetic_room()
    bs = np.asarray([2.0, 7.0])
    ue = np.asarray([14.0, 4.0])
    clock_bias_s = 25e-9
    paths = [
        path
        for path in enumerate_specular_paths(scene, ue, bs, max_reflections=2)
        if -89.0 <= path.arrival_aoa_deg <= 89.0
    ][:3]
    samples = [
        PathObservationSample(
            observation_id=f"obs_{index}",
            sample_id=f"sample_{index}",
            aoa_global_rad=np.deg2rad(path.arrival_aoa_deg),
            delay_s=path.delay_s + clock_bias_s,
        )
        for index, path in enumerate(paths)
    ]
    raw = generate_reverse_candidates(
        scene,
        bs,
        samples,
        max_reflections=2,
        beta_interval_m=(-25.0, 25.0),
    )
    result = solve_position_and_bias(cluster_reverse_candidates(raw))
    assert np.allclose(result.mu, ue, atol=1e-8)
    assert np.isclose(
        result.beta, clock_bias_s * SPEED_OF_LIGHT_M_S, atol=1e-8
    )
    assert len(result.selected_candidates) == 3


def test_clustering_never_mixes_observations_or_topologies() -> None:
    scene = make_synthetic_room()
    bs = np.asarray([2.0, 7.0])
    samples = [
        PathObservationSample("obs_a", "a0", 0.1, 70e-9),
        PathObservationSample("obs_b", "b0", 0.1, 70e-9),
    ]
    raw = generate_reverse_candidates(
        scene,
        bs,
        samples,
        max_reflections=1,
        beta_interval_m=(-10.0, 10.0),
    )
    clustered = cluster_reverse_candidates(raw, position_radius_m=100.0)
    keys = {
        (candidate.observation_id, candidate.metadata["topology_id"])
        for candidate in clustered
    }
    assert len(keys) == len(clustered)
    assert {candidate.observation_id for candidate in clustered} == {"obs_a", "obs_b"}


def test_clustering_does_not_merge_trajectories_that_only_cross_at_one_beta() -> None:
    beta_m = 20.0
    first_direction = np.asarray((1.0, 0.0))
    angle_rad = np.deg2rad(5.0)
    second_direction = np.asarray((np.cos(angle_rad), np.sin(angle_rad)))
    true_position = np.asarray((3.0, 4.0))
    raw = [
        _raw_candidate(
            sample_id="sample-0",
            anchor_m=tuple(true_position + beta_m * first_direction),
            direction=tuple(first_direction),
            beta_interval_m=(15.0, 25.0),
        ),
        _raw_candidate(
            sample_id="sample-1",
            anchor_m=tuple(true_position + beta_m * second_direction),
            direction=tuple(second_direction),
            beta_interval_m=(15.0, 25.0),
        ),
    ]

    clustered = cluster_reverse_candidates(
        raw, position_radius_m=0.05, direction_radius_deg=5.1
    )

    # 两个 anchor 相距约 1.74 m，但两条轨迹在 beta=20 m 处相交。
    assert np.linalg.norm(np.asarray(raw[0].anchor_m) - np.asarray(raw[1].anchor_m)) > 1.0
    assert len(clustered) == 2
    assert clustered[0].metadata["cluster_distance_rule"] == (
        "maximum_over_shared_beta_interval"
    )


def test_cluster_preserves_actual_representative_interval_and_rejects_interval_chains() -> None:
    raw = [
        _raw_candidate(sample_id="sample-0", beta_interval_m=(0.0, 4.0)),
        _raw_candidate(sample_id="sample-1", beta_interval_m=(3.0, 7.0)),
        _raw_candidate(sample_id="sample-2", beta_interval_m=(6.0, 10.0)),
    ]

    clustered = cluster_reverse_candidates(raw)

    # 0 和 2 没有共同有效区间，不能通过 1 的链式连接而被合并。
    assert len(clustered) == 2
    assert clustered[0].beta_min_m == 0.0
    assert clustered[0].beta_max_m == 4.0
    assert clustered[0].metadata["representative_rule"] == (
        "weighted_medoid_actual_member"
    )
    assert clustered[0].metadata["member_beta_min_range_m"] == [0.0, 3.0]
    assert clustered[0].metadata["member_beta_max_range_m"] == [4.0, 7.0]
    assert clustered[0].metadata["shared_beta_interval_m"] == [3.0, 4.0]


def test_cluster_diameter_cannot_grow_through_neighbor_chains() -> None:
    raw = [
        _raw_candidate(sample_id=str(index), anchor_m=(0.9 * index, 0.0))
        for index in range(3)
    ]
    clusters = cluster_reverse_candidates(raw, position_radius_m=1.0)
    assert len(clusters) == 2
    assert [item.metadata["raw_count"] for item in clusters] == [2, 1]
    assert all(item.metadata["maximum_member_trajectory_distance_m"] <= 1.0 for item in clusters)


def test_representative_is_medoid_actual_member_with_its_original_geometry() -> None:
    raw = [
        _raw_candidate(sample_id=str(index), anchor_m=(offset, 0.0),
                       beta_interval_m=(index, 10.0 + index))
        for index, offset in enumerate((0.0, 0.2, 0.7))
    ]
    representative, = cluster_reverse_candidates(raw)
    assert representative.metadata["representative_sample_id"] == "1"
    np.testing.assert_array_equal(representative.anchor_m, raw[1].anchor_m)
    np.testing.assert_array_equal(representative.direction, raw[1].direction)
    assert representative.beta_interval == (1.0, 11.0)
    assert representative.metadata["source_sample_ids"] == ["0", "1", "2"]
    assert [item["sample_id"] for item in representative.metadata["members"]] == ["0", "1", "2"]
    reordered, = cluster_reverse_candidates(list(reversed(raw)))
    assert reordered.metadata == representative.metadata


def test_cluster_frequency_is_metadata_not_solver_weight() -> None:
    raw = [
        _raw_candidate(sample_id=f"main-{index}", topology_id="wall-a")
        for index in range(3)
    ]
    raw.append(_raw_candidate(sample_id="rare-0", topology_id="wall-b"))

    clustered = cluster_reverse_candidates(raw)
    by_topology = {candidate.metadata["topology_id"]: candidate for candidate in clustered}

    assert by_topology["wall-a"].weight == 1.0
    assert by_topology["wall-b"].weight == 1.0
    assert by_topology["wall-a"].metadata["empirical_frequency"] == 0.75
    assert by_topology["wall-b"].metadata["empirical_frequency"] == 0.25
