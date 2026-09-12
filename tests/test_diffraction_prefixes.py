"""批量绕射前缀必须保持标量参考的全部几何序列和实际候选。"""
from dataclasses import replace
import math
import numpy as np
import pytest

from time_bias_localization.diffraction import diffraction_edges, reflection_leg, reflection_sequences
from time_bias_localization.diffraction_prefixes import PrefixGeometry, get_diffraction_prefixes, clear_prefix_cache
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room
from time_bias_localization.raytrace2d import _segment_visible
from time_bias_localization.candidates import PathObservationSample


def reference_prefixes(scene, source, max_reflections):
    prefixes = []
    for edge in diffraction_edges(scene):
        q = np.asarray(edge.position_m)
        for sequence in reflection_sequences(scene, max_reflections):
            points = reflection_leg(scene, source, q, sequence, receiver_walls=edge.incident_wall_ids)
            if points is None:
                continue
            vectors = np.diff(np.asarray([source, *points, q]), axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            prefixes.append((edge, sequence, points, float(lengths.sum()),
                             math.atan2(vectors[0, 1], vectors[0, 0]), -vectors[-1] / lengths[-1]))
    return tuple(prefixes)


@pytest.mark.parametrize('seed', list(range(12)))
def test_random_crossing_wall_prefixes_match_complete_scalar_enumeration(seed):
    rng = np.random.default_rng(seed)
    walls = tuple(WallSegment(f'w{i}', tuple(rng.uniform(-9, 9, 2)), tuple(rng.uniform(-9, 9, 2)))
                  for i in range(7))
    scene = Scene2D('crossing', (-10., 10., -10., 10.), walls, 1.5, .1, 'test')
    source = rng.uniform(-7, 7, 2)
    expected = reference_prefixes(scene, source, 2)
    actual, _ = get_diffraction_prefixes(scene, source, 2)
    assert [(p[0], p[1]) for p in actual] == [(p[0], p[1]) for p in expected]
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a[2], b[2], atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(a[3:5], b[3:5], atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(a[5], b[5], atol=1e-10, rtol=1e-12)


def test_batched_visibility_endpoint_rules_match_scalar():
    scene = replace(make_synthetic_room(), walls=(
        WallSegment('vertical', (0., -2.), (0., 2.)),
        WallSegment('crossing', (-2., 0.), (2., 0.)),
        WallSegment('parallel', (-2., 1.), (2., 1.))))
    geometry = PrefixGeometry(scene)
    starts = []; ends = []; allowed = []
    rng = np.random.default_rng(113)
    for i in range(200):
        starts.append(rng.uniform(-3, 3, 2)); ends.append(rng.uniform(-3, 3, 2))
        allowed.append(tuple(j for j in range(3) if (i >> j) & 1))
    for delta in [-2e-7, -1e-7, 0., 1e-7, 2e-7]:
        starts.append(np.asarray((-1., .5))); ends.append(np.asarray((delta, .5))); allowed.append((0,))
        starts.append(np.asarray((-1., .5))); ends.append(np.asarray((delta, .5))); allowed.append(())
    expected = [_segment_visible(scene, a, b, allowed_endpoint_walls=[scene.walls[j].wall_id for j in ids])
                for a, b, ids in zip(starts, ends, allowed)]
    np.testing.assert_array_equal(geometry.visible(starts, ends, allowed), expected)


@pytest.mark.parametrize('order', [0, 1, 2])
def test_actual_diffraction_candidates_keep_ids_positions_and_interactions(monkeypatch, order):
    import time_bias_localization.diffraction_candidates as module
    scene = replace(make_synthetic_room(), walls=make_synthetic_room().walls + (
        WallSegment('screen', (10., 0.), (10., 8.)),))
    source = np.asarray((2., 7.))
    reference = reference_prefixes(scene, source, order)
    samples = [PathObservationSample(f'obs{i % 3}', f's{i}', p[4], (p[3] + 4.) / 299792458.)
               for i, p in enumerate(reference)]
    kwargs = dict(reference_bias_s=0., max_reflections=order, directions_per_sample=8, angle_tolerance_deg=3.)
    expected_cache = lambda *args: (reference, False)
    with monkeypatch.context() as m:
        m.setattr(module, 'get_diffraction_prefixes', expected_cache)
        expected, expected_diagnostics = module.generate_diffraction_points(scene, source, samples, **kwargs)
    actual, actual_diagnostics = module.generate_diffraction_points(scene, source, samples, **kwargs)
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert a.sample_id == b.sample_id
        assert a.interactions == b.interactions
        assert a.parent_sample_id == b.parent_sample_id
        assert a.observation_id == b.observation_id
        assert a.weight == b.weight
        np.testing.assert_allclose(a.position_m, b.position_m, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(a.endpoint_direction, b.endpoint_direction, atol=1e-10, rtol=1e-12)
    for key in ('initial_point_count', 'visible_bs_edge_prefix_count', 'attempted_direction_count'):
        assert actual_diagnostics[key] == expected_diagnostics[key]


def test_cache_includes_public_map_bs_and_reflection_budget():
    clear_prefix_cache()
    scene = make_synthetic_room()
    first, hit = get_diffraction_prefixes(scene, (2., 7.), 1)
    assert not hit
    cached, hit = get_diffraction_prefixes(scene, (2., 7.), 1)
    assert hit and first is cached
    assert not get_diffraction_prefixes(scene, (3., 7.), 1)[1]
    assert not get_diffraction_prefixes(scene, (2., 7.), 2)[1]
    changed = replace(scene, walls=scene.walls + (WallSegment('extra', (10., 0.), (10., 8.)),))
    assert not get_diffraction_prefixes(changed, (2., 7.), 1)[1]


def test_unresolved_tiny_visibility_gap_retains_distant_wall():
    scene = Scene2D('tiny_gap', (-1., 25000., -10., 10.), (
        WallSegment('lower', (10000., -.001), (10000., -5e-7)),
        WallSegment('upper', (10000., 5e-7), (10000., .001)),
        WallSegment('distant', (20000., -1.), (20000., 1.)),
        WallSegment('edge', (14000., 2e-7), (14001., 2e-7))), 1.5, .1, 'test')
    source = np.asarray((0., 0.))
    geometry = PrefixGeometry(scene)
    assert 2 in geometry.visible_first_walls(source)
    actual, _ = get_diffraction_prefixes(scene, source, 2)
    expected = reference_prefixes(scene, source, 2)
    assert [(p[0], p[1]) for p in actual] == [(p[0], p[1]) for p in expected]


def test_diffraction_then_reflection_fan_points_rebuild_the_complete_path():
    from time_bias_localization.diffraction import rebuild_path
    from time_bias_localization.diffraction_candidates import generate_diffraction_points
    scene = replace(make_synthetic_room(), walls=make_synthetic_room().walls + (
        WallSegment('screen', (10., 0.), (10., 8.)),))
    source = np.asarray((2., 7.))
    prefixes = reference_prefixes(scene, source, 2)
    samples = [PathObservationSample(f'obs{i}', f's{i}_{length}', p[4], (p[3] + length) / 299792458.)
               for i, p in enumerate(prefixes) for length in (4., 20., 40.)]
    points, _ = generate_diffraction_points(scene, source, samples, reference_bias_s=0., max_reflections=2,
                                            directions_per_sample=8, angle_tolerance_deg=3.)
    mixed = [p for p in points if any(a[0] == 'diffraction' and b[0] == 'reflection'
                                    for a, b in zip(p.interactions, p.interactions[1:]))]
    assert mixed
    for point in mixed:
        rebuilt = rebuild_path(scene, source, point.position_m, point.interactions)
        assert rebuilt is not None
        assert rebuilt.length_m == pytest.approx(point.observed_delay_s * 299792458., abs=1e-7)
        np.testing.assert_allclose(rebuilt.interaction_points_m, point.interaction_points_m, atol=1e-7)


def test_tiny_crossing_walls_do_not_lose_a_valid_first_reflection():
    # Wall-wall determinant is < 1e-12, but the BS-to-reflection ray is valid.
    # Ignoring the two crossings would miss the narrow central visibility interval.
    x, y0, offset = .001, 1.5e-7, 3e-8
    a = lambda y: (x + offset + y - y0, y)
    b = lambda y: (x + offset - y + y0, y)
    scene = Scene2D('short_crossings', (-.01, .01, -.01, .01), (
        WallSegment('central', (x, -3e-7), (x, 3e-7)),
        WallSegment('a', a(-3.5e-7), a(3.2e-7)),
        WallSegment('b', b(-3.2e-7), b(3.5e-7)),
        WallSegment('receiver', (0., 3e-7), (0., 1e-5))), 1.5, .1, 'test')
    source = np.asarray((0., 0.))
    assert reflection_leg(scene, source, np.asarray((0., 3e-7)), ('central',),
                          receiver_walls=('receiver',)) is not None
    assert 0 in PrefixGeometry(scene).visible_first_walls(source)
    actual, _ = get_diffraction_prefixes(scene, source, 2)
    expected = reference_prefixes(scene, source, 2)
    assert [(p[0], p[1]) for p in actual] == [(p[0], p[1]) for p in expected]
