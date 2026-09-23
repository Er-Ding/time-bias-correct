"""重叠三角立面必须恢复为真实墙面，且不制造门洞、拐角或重复路径。"""
from dataclasses import replace
import math

import numpy as np
import pytest

from time_bias_localization.diffraction import diffraction_edges, enumerate_paths
from time_bias_localization.propagation_hypotheses import build_hypothesis_bank
from time_bias_localization.propagation_model import ContinuousObservation, evaluate_hypothesis
from time_bias_localization.raytrace2d import _segment_visible, enumerate_specular_paths
from time_bias_localization.scene import (
    Scene2D, WallSegment, merge_collinear_wall_segments, preprocess_sionna_triangle_mesh,
)


def _scene(walls):
    return Scene2D("walls", (-20., 20., -20., 20.), tuple(walls), 1.5, .1, "test")


def _panel_mesh(panels):
    vertices, faces = [], []
    for start, end, height in panels:
        offset = len(vertices)
        vertices.extend([(*start, 0.), (*end, 0.), (*end, height), (*start, height)])
        faces.extend([(offset, offset+1, offset+2), (offset, offset+2, offset+3)])
    return np.asarray(vertices), np.asarray(faces)


def _preprocess(panels, **kwargs):
    vertices, faces = _panel_mesh(panels)
    return preprocess_sionna_triangle_mesh(vertices, faces, name="mesh", fixed_height_m=1.5,
        bev_resolution_m=.1, bounds_m=(-20., 20., -20., 20.),
        object_vertex_ranges={"building": (0, len(vertices))}, **kwargs)


def _path_key(path):
    return tuple(np.round(path.nodes.ravel(), 7))


def test_opposite_facades_with_different_heights_become_one_wall():
    # 真实故障的结构：同一面墙的两组立面法向相反、高度不同，切分点不同。
    scene = _preprocess([((0., 0.), (10., 0.), 19.1251888275),
                         ((10., 0.), (0., 0.), 18.9558696747)])
    assert len(scene.walls) == 1
    np.testing.assert_allclose([scene.walls[0].start_m, scene.walls[0].end_m], [(0., 0.), (10., 0.)])
    assert {edge.position_m for edge in diffraction_edges(scene)} == {(0., 0.), (10., 0.)}


def test_short_triangle_slices_are_merged_before_minimum_wall_length_filter():
    scene = _preprocess([((0., 0.), (.3, 0.), 3.), ((1., 0.), (1.1, 0.), 3.)])
    assert len(scene.walls) == 1
    assert scene.walls[0].length_m == pytest.approx(.3)
    assert not _segment_visible(scene, np.array([.15, -1.]), np.array([.15, 1.]))


def test_union_does_not_pull_unselected_outside_triangle_slices_into_the_map():
    vertices, faces = _panel_mesh([((-10., -10.), (2., 2.), 3.)])
    scene = preprocess_sionna_triangle_mesh(vertices, faces, name="crop", fixed_height_m=1.5,
        bev_resolution_m=.1, bounds_m=(-1., 3., -1., 3.))
    # 原来的区域筛选仅保留与范围相交的三角切片，即 (-4,-4) 到 (2,2)。
    # 不能因为先合并整个立面，又把完全在区域外的 (-10,-10) 到 (-4,-4) 带回来。
    assert len(scene.walls) == 1
    assert {scene.walls[0].start_m, scene.walls[0].end_m} == {(-4., -4.), (2., 2.)}


@pytest.mark.parametrize("gap", [1., 1e-4, 1e-6])
def test_collinear_union_preserves_openings_smaller_than_old_endpoint_quantization(gap):
    scene = _preprocess([((0., 0.), (2., 0.), 3.), ((2.+gap, 0.), (4., 0.), 3.)])
    assert len(scene.walls) == 2
    assert scene.walls[0].length_m + scene.walls[1].length_m == pytest.approx(4.-gap, abs=1e-12)
    midpoint = 2. + gap/2
    assert _segment_visible(scene, np.array([midpoint, -1.]), np.array([midpoint, 1.]))


def test_parallel_walls_and_corners_are_not_merged():
    scene = _preprocess([((0., 0.), (4., 0.), 3.), ((0., .0001), (4., .0001), 3.),
                         ((4., 0.), (4., -3.), 3.)])
    assert len(scene.walls) == 3
    assert (4., 0.) in {edge.position_m for edge in diffraction_edges(scene)}


def test_different_objects_keep_their_boundary_and_source():
    walls = (WallSegment("a", (0., 0.), (2., 0.), "first_material"),
             WallSegment("b", (2., 0.), (4., 0.), "second_material"))
    assert merge_collinear_wall_segments(walls) == walls


def test_merging_does_not_turn_a_t_junction_into_a_diffraction_edge():
    scene = _preprocess([((0., 0.), (2., 0.), 3.), ((2., 0.), (4., 0.), 3.),
                         ((2., 0.), (2., 3.), 3.)])
    assert len(scene.walls) == 2
    assert (2., 0.) not in {edge.position_m for edge in diffraction_edges(scene)}


def test_reflection_at_a_triangle_seam_is_recovered_once():
    scene = _preprocess([((0., 0.), (10., 0.), 3.), ((10., 0.), (0., 0.), 4.)])
    paths = enumerate_specular_paths(scene, (2., 2.), (8., 2.), max_reflections=2)
    reflected = [p for p in paths if p.reflection_order]
    assert len(reflected) == 1
    np.testing.assert_allclose(reflected[0].interaction_points_m, [(5., 0.)])
    assert reflected[0].length_m == pytest.approx(2 * math.sqrt(13))


@pytest.mark.parametrize("with_diffraction", [False, True])
def test_preprocessed_mesh_has_same_unique_paths_as_independent_whole_wall_scene(with_diffraction):
    if with_diffraction:
        walls = (WallSegment("screen", (0., -6.), (0., 0.)),
                 WallSegment("right", (8., -9.), (8., 9.)))
        positions, bs = ((4., -2.), (5., -3.)), (-4., -2.)
    else:
        walls = (WallSegment("south", (0., 0.), (10., 0.)),
                 WallSegment("east", (10., 0.), (10., 10.)),
                 WallSegment("north", (10., 10.), (0., 10.)),
                 WallSegment("west", (0., 10.), (0., 0.)))
        positions, bs = ((7.3, 5.1), (3.4, 8.2)), (2.1, 3.2)
    expected_scene = _scene(walls)
    panels = [(w.start_m, w.end_m, 3.) for w in walls]
    panels += [(w.end_m, w.start_m, 4.7) for w in walls]
    actual_scene = _preprocess(panels)
    assert len(actual_scene.walls) == len(walls)
    assert {edge.position_m for edge in diffraction_edges(actual_scene)} == {
        edge.position_m for edge in diffraction_edges(expected_scene)}
    for ue in positions:
        expected = enumerate_paths(expected_scene, ue, bs, max_reflections=2,
                                   max_diffractions=int(with_diffraction))
        actual = enumerate_paths(actual_scene, ue, bs, max_reflections=2,
                                 max_diffractions=int(with_diffraction))
        assert expected
        assert len(actual) == len({_path_key(path) for path in actual})
        assert {_path_key(path) for path in actual} == {_path_key(path) for path in expected}
        observations = [ContinuousObservation(str(i), math.radians(p.arrival_aoa_deg),
            p.length_m + 2., .01, .1) for i, p in enumerate(expected)]
        bank = build_hypothesis_bank(actual_scene, bs, observations, max_reflections=2,
            max_diffractions=int(with_diffraction), max_hypotheses=10000,
            max_enumerated_sequences=50000, aoa_gate_rad=.01, beta_interval_m=(-3., 4.))
        assert bank.search_report["complete_within_configured_orders"]
        valid = [evaluation.path for h in bank.hypotheses
                 if (evaluation := evaluate_hypothesis(actual_scene, h, ue)).valid]
        assert len(valid) == len({_path_key(path) for path in valid})
        assert {_path_key(path) for path in valid} == {_path_key(path) for path in expected}


def test_union_is_idempotent_deterministic_and_old_scene_loading_does_not_change_ids():
    walls = (WallSegment("a", (0., 0.), (3., 0.)), WallSegment("b", (4., 0.), (1., 0.)),
             WallSegment("c", (8., 0.), (7., 0.)))
    merged = merge_collinear_wall_segments(walls)
    assert merge_collinear_wall_segments(tuple(reversed(walls))) == merged
    assert merge_collinear_wall_segments(merged) == merged
    scene = _scene(walls)
    assert Scene2D.from_dict(scene.to_dict()).walls == walls
    assert len(replace(scene, walls=merged).walls) == 2


@pytest.mark.parametrize("tolerance", [0., -1., float("nan"), float("inf")])
def test_invalid_collinearity_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="容差"):
        merge_collinear_wall_segments((), tolerance_m=tolerance)
