"""向量化 RT 必须逐字段保持参考实现，包括角点并列与阈值边界。"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from time_bias_localization.candidates import (
    PathObservationSample, _nearest_wall_hit, generate_reverse_candidates,
)
from time_bias_localization.reverse_compute import VectorizedWallIntersector
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room


def _assert_same_hit(scene, origin, direction, chunk_size):
    expected = _nearest_wall_hit(scene, np.asarray(origin), np.asarray(direction))
    actual = VectorizedWallIntersector(scene, wall_chunk_size=chunk_size).nearest(
        np.asarray(origin), np.asarray(direction)
    )
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        np.testing.assert_array_equal(actual[2], expected[2])
    return actual


@pytest.mark.parametrize("chunk_size", [1, 2, 8192])
@pytest.mark.parametrize("origin,direction", [
    ((0.0, 0.0), (1.0, 0.0)),
    ((0.0, 0.0), (0.0, 1.0)),
    ((1.0, 1.0), (1.0, 1.0)),
    ((4.0, 2.0), (-1.0, 0.0)),
    ((4.0, 2.0), (1.0, 0.0)),
    ((2.0, 4.0), (1.0, 1e-10)),
    ((2.0, 4.0), (1.0, 1e-8)),
    ((0.0, 0.0), (1.0, 1.0 + 5e-10)),
    ((0.0, 0.0), (1.0, 1.0 + 5e-8)),
])
def test_nearest_wall_corner_parallel_and_endpoint_tolerances(origin, direction, chunk_size):
    # 顺序刻意与编号排序相反，角点命中并列时仍必须选 a-top。
    scene = replace(make_synthetic_room(), walls=(
        WallSegment("z-right", (4.0, 0.0), (4.0, 4.0)),
        WallSegment("a-top", (4.0, 4.0), (0.0, 4.0)),
    ))
    hit = _assert_same_hit(scene, origin, direction, chunk_size)
    if origin == (1.0, 1.0) and direction == (1.0, 1.0):
        assert hit[1].wall_id == "a-top"


@pytest.mark.parametrize("distance", [0.5e-6, 1e-6, np.nextafter(1e-6, np.inf), 2e-6])
def test_ray_start_distance_cutoff_is_unchanged(distance):
    scene = replace(make_synthetic_room(), walls=(
        WallSegment("near", (distance, -1.0), (distance, 1.0)),
        WallSegment("far", (2.0, -1.0), (2.0, 1.0)),
    ))
    hit = _assert_same_hit(scene, (0.0, 0.0), (1.0, 0.0), 1)
    assert hit[1].wall_id == ("near" if distance > 1e-6 else "far")


@pytest.mark.parametrize("max_reflections", [0, 1, 2])
@pytest.mark.parametrize("chunk_size", [1, 3, 8192])
def test_full_candidates_preserve_order_topology_points_and_bias_intervals(max_reflections, chunk_size):
    rng = np.random.default_rng(61)
    scene = make_synthetic_room()
    bs = (2.0, 7.0)
    angles = np.r_[rng.uniform(-np.pi, np.pi, 40), 0.0, np.pi / 2, -np.pi / 2,
                   np.arctan2(7.0, 18.0), np.arctan2(-7.0, 18.0)]
    samples = [PathObservationSample(f"obs-{i % 3}", f"sample-{i}", float(angle),
                                    float(rng.uniform(0.0, 500e-9)))
               for i, angle in enumerate(angles)]
    kwargs = dict(max_reflections=max_reflections, beta_interval_m=(-24.0, 24.0))
    expected = generate_reverse_candidates(scene, bs, samples, backend="reference", **kwargs)
    actual = generate_reverse_candidates(scene, bs, iter(samples), backend="numpy",
                                         wall_chunk_size=chunk_size, **kwargs)
    assert actual == expected


def test_empty_wall_scene_and_empty_samples_keep_reference_behavior():
    scene = replace(make_synthetic_room(), walls=())
    samples = [PathObservationSample("obs", "sample", 0.0, 60e-9)]
    kwargs = dict(max_reflections=2, beta_interval_m=(-24.0, 24.0))
    assert generate_reverse_candidates(scene, (2.0, 7.0), samples, **kwargs) == (
        generate_reverse_candidates(scene, (2.0, 7.0), samples, backend="reference", **kwargs)
    )
    assert generate_reverse_candidates(scene, (2.0, 7.0), [], **kwargs) == []


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_intersection_chunk_size_must_be_positive_integer(chunk_size):
    with pytest.raises(ValueError, match="wall_chunk_size"):
        VectorizedWallIntersector(make_synthetic_room(), wall_chunk_size=chunk_size)


def test_saved_munich_monte_carlo_samples_match_reference_if_available():
    root = Path(__file__).resolve().parents[1]
    localization = root / "outputs/spectrum_check_cuda_20260908_v1/UE001/repeat_000/localization"
    if not (localization / "spectrum_samples.json").is_file():
        pytest.skip("本机没有固定 Munich 谱面采样产物")
    manifest = json.loads((localization / "localization_manifest.json").read_text())
    scene = Scene2D.load(manifest["scene_input"])
    config = json.loads((localization / "localization_config.json").read_text())["resolved_config"]
    records = json.loads((localization / "spectrum_samples.json").read_text())["samples"]
    # 跨全部来源峰选固定样本；另以本轮387样本全量回放验证并记录耗时。
    chosen = records[::32]
    samples = [PathObservationSample(r["observation_id"], r["sample_id"],
                                    r["aoa_global_rad"], r["delay_s"], r["weight"])
               for r in chosen]
    from time_bias_localization.constants import SPEED_OF_LIGHT_M_S
    kwargs = dict(max_reflections=config["scene"]["max_reflections"],
                  beta_interval_m=(config["localization"]["bias_min_s"] * SPEED_OF_LIGHT_M_S,
                                   config["localization"]["bias_max_s"] * SPEED_OF_LIGHT_M_S))
    expected = generate_reverse_candidates(scene, config["radio"]["bs_position_m"], samples,
                                           backend="reference", **kwargs)
    actual = generate_reverse_candidates(scene, config["radio"]["bs_position_m"], samples,
                                         wall_chunk_size=127, **kwargs)
    assert actual == expected
