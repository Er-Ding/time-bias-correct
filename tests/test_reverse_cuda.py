"""CUDA 几何必须保留候选来源、传播顺序和边界判定；显式开启才使用 GPU。"""
from dataclasses import asdict, replace
import os

import numpy as np
import pytest

from time_bias_localization.candidates import PathObservationSample
from time_bias_localization.diffraction_candidates import generate_diffraction_points
from time_bias_localization.diffraction_prefixes import get_diffraction_prefixes
from time_bias_localization.initial_candidates import generate_initial_candidate_points
from time_bias_localization.reverse_compute import VectorizedWallIntersector
from time_bias_localization.reverse_cuda import (
    CudaWallGeometry, clear_cuda_reverse_cache, generate_diffraction_points_cuda,
    generate_initial_candidate_points_cuda,
)
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room


cuda = pytest.mark.skipif(os.environ.get("TBC_RUN_CUDA_REVERSE_TESTS") != "1",
                          reason="CUDA 回归必须经独立脚本明确指定 GPU")


def assert_same_points(expected, actual):
    assert len(actual) == len(expected)
    numeric = {"position_m", "reflection_points_m", "prefix_length_m", "endpoint_origin_m",
               "endpoint_direction", "endpoint_free_distance_m", "interaction_points_m"}
    for a, b in zip(expected, actual, strict=True):
        da, db = asdict(a), asdict(b)
        for name in da:
            if name in numeric:
                np.testing.assert_allclose(db[name], da[name], atol=1e-9, rtol=1e-12, err_msg=name)
            else:
                assert db[name] == da[name], name


@pytest.mark.parametrize("keyword,value", [
    ("directions_per_sample", 0), ("directions_per_sample", True),
    ("job_chunk_size", -1), ("job_chunk_size", 1.5),
    ("max_reflections", 3), ("max_reflections", True),
    ("angle_tolerance_deg", 0), ("angle_tolerance_deg", 90),
    ("reference_bias_s", float("nan")),
    ("device_id", True), ("device_id", -1),
])
def test_invalid_parameters_fail_before_gpu_import(keyword, value):
    kwargs = dict(reference_bias_s=0., max_reflections=2, directions_per_sample=4, angle_tolerance_deg=3.)
    kwargs[keyword] = value
    with pytest.raises(ValueError):
        generate_diffraction_points_cuda(make_synthetic_room(), (2., 7.), [], **kwargs)


def test_missing_gpu_selection_fails_without_cuda_initialization(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        CudaWallGeometry(make_synthetic_room())


@cuda
@pytest.mark.parametrize("seed", range(4))
def test_random_batched_intersections_and_stable_ties(seed):
    rng = np.random.default_rng(seed)
    walls = tuple(WallSegment(f'w{i}', tuple(rng.uniform(-9, 9, 2)), tuple(rng.uniform(-9, 9, 2)))
                  for i in range(35))
    scene = Scene2D('crossing', (-10., 10., -10., 10.), walls, 1.5, .1, 'test')
    origins = rng.uniform(-10, 10, (500, 2)); directions = rng.normal(size=(500, 2))
    gpu = CudaWallGeometry(scene)
    indices, values = gpu.nearest_batch(origins, directions)
    cpu = VectorizedWallIntersector(scene)
    for k, (origin, direction) in enumerate(zip(origins, directions, strict=True)):
        hit = cpu.nearest(origin, direction)
        if hit is None:
            assert indices[k] == -1
        else:
            assert gpu.walls[indices[k]].wall_id == hit[1].wall_id
            np.testing.assert_allclose(values[k], [hit[0], *hit[2]], atol=1e-10, rtol=1e-12)


@cuda
def test_corner_ties_wall_endpoints_parallel_and_start_cutoff():
    scene = replace(make_synthetic_room(), walls=(
        WallSegment('z-right', (4., 0.), (4., 4.)),
        WallSegment('a-top', (4., 4.), (0., 4.))))
    rows = [((0., 0.), (1., 0.)), ((0., 0.), (0., 1.)), ((1., 1.), (1., 1.)),
            ((4., 2.), (-1., 0.)), ((4., 2.), (1., 0.)), ((2., 4.), (1., 1e-10)),
            ((2., 4.), (1., 1e-8)), ((0., 0.), (1., 1.+5e-10)), ((0., 0.), (1., 1.+5e-8))]
    gpu = CudaWallGeometry(scene)
    ids, values = gpu.nearest_batch(*zip(*rows))
    for i, (origin, direction) in enumerate(rows):
        expected = VectorizedWallIntersector(scene).nearest(origin, direction)
        if expected is None:
            assert ids[i] == -1
        else:
            assert gpu.walls[ids[i]].wall_id == expected[1].wall_id
            np.testing.assert_allclose(values[i], [expected[0], *expected[2]], atol=1e-10, rtol=1e-12)
    assert gpu.walls[ids[2]].wall_id == 'a-top'
    for d in [.5e-6, 1e-6, np.nextafter(1e-6, np.inf), 2e-6]:
        short = replace(scene, walls=(WallSegment('near', (d, -1.), (d, 1.)),
                                     WallSegment('far', (2., -1.), (2., 1.))))
        geometry = CudaWallGeometry(short)
        ids, _ = geometry.nearest_batch([[0., 0.]], [[1., 0.]])
        assert geometry.walls[ids[0]].wall_id == ('near' if d > 1e-6 else 'far')


@cuda
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("bias", [-30e-9, 0., 30e-9])
def test_complete_fans_and_bias_changes_preserve_all_candidate_fields(order, bias):
    scene = replace(make_synthetic_room(), walls=make_synthetic_room().walls + (
        WallSegment('screen', (10., 0.), (10., 8.)),))
    source = np.asarray((2., 7.))
    prefixes, _ = get_diffraction_prefixes(scene, source, order)
    samples = [PathObservationSample(f'obs{i % 3}', f's{i}_{length}', p[4],
                                    (p[3]+length)/299792458., .6)
               for i, p in enumerate(prefixes) for length in (.00001, 4., 20., 40.)]
    kwargs = dict(reference_bias_s=bias, max_reflections=order, directions_per_sample=8, angle_tolerance_deg=3.)
    expected, expected_info = generate_diffraction_points(scene, source, samples, **kwargs)
    actual, info = generate_diffraction_points_cuda(scene, source, iter(samples), job_chunk_size=7, **kwargs)
    assert_same_points(expected, actual)
    for name in ('initial_point_count', 'sample_prefix_angle_match_count', 'attempted_direction_count',
                 'observation_point_counts'):
        assert info[name] == expected_info[name]
    again, again_info = generate_diffraction_points_cuda(scene, source, samples, job_chunk_size=1024, **kwargs)
    assert again_info['cuda_geometry_cache_hit']
    assert_same_points(actual, again)


@cuda
@pytest.mark.parametrize("seed", range(4))
def test_random_crossing_scenes_and_full_entrypoint(seed):
    rng = np.random.default_rng(92+seed)
    scene = Scene2D('random', (-10., 10., -10., 10.), tuple(
        WallSegment(f'w{i}', tuple(rng.uniform(-9, 9, 2)), tuple(rng.uniform(-9, 9, 2)))
        for i in range(7)), 1.5, .1, 'test')
    samples = [PathObservationSample(f'obs{i%3}', f's{i}', float(rng.uniform(-np.pi, np.pi)),
                                    float(rng.uniform(0., 300e-9))) for i in range(90)]
    kwargs = dict(max_reflections=2, max_diffractions=1, diffraction_directions_per_sample=8)
    expected = generate_initial_candidate_points(scene, (0., 0.), samples, **kwargs)
    actual = generate_initial_candidate_points_cuda(scene, (0., 0.), iter(samples), **kwargs)
    assert_same_points(expected.points, actual.points)
    assert actual.rejected_samples == expected.rejected_samples


@cuda
def test_empty_map_empty_samples_and_cache_map_identity():
    clear_cuda_reverse_cache()
    room = make_synthetic_room()
    empty = replace(room, walls=())
    sample = PathObservationSample('o', 's', 0., 40e-9)
    kwargs = dict(reference_bias_s=0., max_reflections=2, directions_per_sample=4, angle_tolerance_deg=3.)
    for scene, samples in [(empty, [sample]), (room, []), (room, [sample])]:
        expected, _ = generate_diffraction_points(scene, (2., 7.), samples, **kwargs)
        actual, info = generate_diffraction_points_cuda(scene, (2., 7.), samples, **kwargs)
        assert_same_points(expected, actual)
    moved = replace(room, walls=room.walls + (WallSegment('screen', (10., 0.), (10., 8.)),))
    actual, info = generate_diffraction_points_cuda(moved, (2., 7.), [sample], **kwargs)
    assert not info['cuda_geometry_cache_hit']


@cuda
def test_candidate_endpoints_at_wall_and_map_boundary_thresholds():
    scene = replace(make_synthetic_room(), walls=make_synthetic_room().walls + (
        WallSegment('screen', (10., 0.), (10., 8.)),))
    source = np.asarray((2., 7.))
    prefixes, _ = get_diffraction_prefixes(scene, source, 2)
    kwargs = dict(reference_bias_s=0., max_reflections=2, directions_per_sample=8, angle_tolerance_deg=3.)
    base = [PathObservationSample(f'o{i}', f's{i}', p[4], (p[3]+4.)/299792458.)
            for i, p in enumerate(prefixes)]
    found, _ = generate_diffraction_points(scene, source, base, **kwargs)
    assert found
    samples = []
    for i, point in enumerate(found):
        for j, delta in enumerate([-2e-7, -1e-7, 0., 1e-7, 2e-7]):
            length = point.prefix_length_m+point.endpoint_free_distance_m+delta
            # Each observation starts at ordinal zero, as in the initial sample.
            samples.append(PathObservationSample(f'e{i}_{j}', f'end{i}_{j}',
                point.observed_aoa_global_rad, length/299792458.))
    expected, _ = generate_diffraction_points(scene, source, samples, **kwargs)
    actual, _ = generate_diffraction_points_cuda(scene, source, samples, **kwargs)
    assert_same_points(expected, actual)


def test_launcher_requires_explicit_gpu_before_creating_run_directory(tmp_path):
    import subprocess
    from pathlib import Path
    script = Path(__file__).resolve().parents[1]/'run_reverse_cuda_experiment.sh'
    environment = dict(os.environ, RT_CUDA_GPU_ID='', RT_CUDA_MODE='tests',
                       RT_CUDA_OUTPUT_ROOT=str(tmp_path/'no_run'))
    result = subprocess.run(['bash', str(script)], env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'RT_CUDA_GPU_ID' in result.stderr
    assert not (tmp_path/'no_run').exists()
