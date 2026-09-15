"""全部墙同时计算时必须保持原标量遮挡判据，不能因提速改变物理接受。"""
from dataclasses import replace

import numpy as np
import pytest

from time_bias_localization.raytrace2d import (
    _EPS, _segment_intersection_parameters, _segment_visible,
    _visibility_wall_arrays, _VISIBILITY_WALL_CACHE, _VISIBILITY_WALL_CACHE_SIZE,
)
from time_bias_localization.scene import Scene2D, WallSegment


def _scene(walls=()):
    return Scene2D("visibility", (-20., 20., -20., 20.), tuple(walls),
                   1.5, .1, "scalar_vector_equivalence")


def _legacy_visible(scene, start, end, *, allowed_endpoint_walls=()):
    """原 _segment_visible 的逐墙循环；交点函数保留原实现。"""
    allowed = set(allowed_endpoint_walls)
    for wall in scene.walls:
        hit = _segment_intersection_parameters(start, end, wall)
        if hit is None:
            continue
        fraction, _, _ = hit
        if wall.wall_id in allowed and (fraction <= _EPS or fraction >= 1.0 - _EPS):
            continue
        if _EPS < fraction < 1.0 - _EPS:
            return False
        if wall.wall_id not in allowed:
            return False
    return True


def _compare(scene, start, end, allowed=()):
    start, end = np.asarray(start), np.asarray(end)
    assert _segment_visible(scene, start, end, allowed_endpoint_walls=iter(allowed)) == _legacy_visible(
        scene, start, end, allowed_endpoint_walls=allowed)


@pytest.mark.parametrize("seed", range(8))
def test_vector_visibility_equals_scalar_on_random_walls_rays_and_permissions(seed):
    rng = np.random.default_rng(seed)
    scene = _scene(WallSegment(f"wall_{index}", tuple(rng.uniform(-9, 9, 2)),
                               tuple(rng.uniform(-9, 9, 2))) for index in range(37))
    for index in range(160):
        start, end = rng.uniform(-12, 12, (2, 2))
        if index % 4 == 0:
            # Include exact wall endpoints, not just generic crossings.
            start = scene.walls[index % len(scene.walls)].start
        if index % 7 == 0:
            end = scene.walls[(index + 11) % len(scene.walls)].end
        allowed = tuple(wall.wall_id for i, wall in enumerate(scene.walls) if (i + index) % 3 == 0)
        _compare(scene, start, end, allowed)
        _compare(scene, end, start, allowed)


_FRACTIONS = [
    -2 * _EPS, np.nextafter(-_EPS, -np.inf), -_EPS,
    np.nextafter(-_EPS, np.inf), 0., np.nextafter(_EPS, -np.inf),
    _EPS, np.nextafter(_EPS, np.inf), .5,
    np.nextafter(1. - _EPS, -np.inf), 1. - _EPS,
    np.nextafter(1. - _EPS, np.inf), 1.,
    np.nextafter(1. + _EPS, -np.inf), 1. + _EPS,
    np.nextafter(1. + _EPS, np.inf), 1. + 2 * _EPS,
]


@pytest.mark.parametrize("fraction", _FRACTIONS)
@pytest.mark.parametrize("allowed", [(), ("wall",), ("unknown",)])
def test_segment_endpoint_thresholds_match_without_changing_tolerance(fraction, allowed):
    scene = _scene([WallSegment("wall", (fraction, -1.), (fraction, 1.))])
    _compare(scene, (0., 0.), (1., 0.), allowed)


@pytest.mark.parametrize("fraction", _FRACTIONS)
def test_wall_endpoint_thresholds_match_scalar(fraction):
    scene = _scene([WallSegment("wall", (.5, -fraction), (.5, 1. - fraction))])
    _compare(scene, (0., 0.), (1., 0.))
    _compare(scene, (0., 0.), (1., 0.), ("wall",))


@pytest.mark.parametrize("denominator", [
    0., 1e-14, 5e-13, np.nextafter(1e-12, 0.), 1e-12,
    np.nextafter(1e-12, np.inf), 2e-12, 1e-9,
    -1e-14, -1e-12, np.nextafter(-1e-12, -np.inf), -2e-12,
])
def test_parallel_threshold_and_almost_parallel_crossings_match(denominator):
    scene = _scene([WallSegment("wall", (-.5, -denominator / 2),
                                (1.5, denominator / 2))])
    _compare(scene, (0., 0.), (1., 0.))
    _compare(scene, (0., 0.), (1., 0.), ("wall",))


def test_allowed_wall_id_applies_to_every_same_named_segment_and_only_endpoints():
    scene = _scene([WallSegment("shared", (0., -1.), (0., 1.)),
                    WallSegment("shared", (1., -1.), (1., 1.))])
    assert _segment_visible(scene, np.array([0., 0.]), np.array([1., 0.]),
                            allowed_endpoint_walls=("shared",))
    _compare(scene, (0., 0.), (1., 0.), ("shared", "shared", "unknown"))
    middle = replace(scene, walls=(*scene.walls, WallSegment("shared", (.5, -1.), (.5, 1.))))
    assert not _segment_visible(middle, np.array([0., 0.]), np.array([1., 0.]),
                                allowed_endpoint_walls=("shared",))
    _compare(middle, (0., 0.), (1., 0.), ("shared",))


@pytest.mark.parametrize("point", [(0., 0.), (0., 1.), (1., 1.)])
def test_empty_map_and_zero_length_segments_keep_scalar_behavior(point):
    for scene in (_scene(), _scene([WallSegment("wall", (0., -1.), (0., 1.))])):
        for allowed in ((), ("wall",)):
            _compare(scene, point, point, allowed)
    assert _segment_visible(_scene(), np.array([0., 0.]), np.array([1., 1.]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_comparisons_preserve_old_scalar_result(bad):
    scene = _scene([WallSegment("a", (0., -1.), (0., 1.)),
                    WallSegment("b", (-1., 0.), (1., 0.))])
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        for start, end in [((bad, 0.), (1., 1.)), ((0., 0.), (bad, 1.)),
                           ((bad, bad), (bad, bad))]:
            for allowed in ((), ("a",), ("a", "b")):
                _compare(scene, start, end, allowed)


def test_float32_endpoints_keep_subtraction_order_of_scalar_formula():
    rng = np.random.default_rng(825)
    scene = _scene([WallSegment("vertical", (1., -5.), (1., 5.)),
                    WallSegment("slanted", (-2., -.7), (2., 3.))])
    for _ in range(150):
        start, end = rng.uniform(-7, 7, (2, 2)).astype(np.float32)
        _compare(scene, start, end, ("vertical",))


def test_visibility_geometry_cache_is_read_only_bounded_and_uses_scene_identity(monkeypatch):
    scene = _scene([WallSegment("wall", (0., -1.), (0., 1.))])

    def no_scene_hash(_self):
        pytest.fail("热路径不应逐墙计算场景哈希")

    monkeypatch.setattr(Scene2D, "__hash__", no_scene_hash)
    first = _visibility_wall_arrays(scene)
    assert _visibility_wall_arrays(scene) is first
    with pytest.raises(ValueError):
        first.starts[0, 0] = 12.
    with pytest.raises(ValueError):
        first.vectors[0, 0] = 12.
    with pytest.raises(ValueError):
        first.indices_by_id["wall"][0] = 12
    with pytest.raises(TypeError):
        first.indices_by_id["other"] = np.array([0])
    changed = replace(scene, walls=(WallSegment("wall", (2., -1.), (2., 1.)),))
    assert _visibility_wall_arrays(changed) is not first
    np.testing.assert_array_equal(_visibility_wall_arrays(changed).starts, [[2., -1.]])
    retained_scenes = [replace(scene, name=f"scene_{i}") for i in range(_VISIBILITY_WALL_CACHE_SIZE + 3)]
    for current in retained_scenes:
        _visibility_wall_arrays(current)
    assert len(_VISIBILITY_WALL_CACHE) <= _VISIBILITY_WALL_CACHE_SIZE
    _compare(scene, (-1., 0.), (1., 0.))
