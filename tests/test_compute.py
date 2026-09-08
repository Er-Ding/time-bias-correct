"""批量计算与原 MUSIC 的物理/数值一致性，以及找峰规则回归。"""

from __future__ import annotations

import importlib
import json

import numpy as np
import numpy.testing as npt
import pytest

from time_bias_localization.compute import ComputeSettings, MusicComputer
from time_bias_localization.signal import (
    extract_local_music_peaks,
    music_2d_spectrum,
    synthesize_ula_csi,
)


def _parameters() -> dict:
    return {
        "subcarrier_frequencies_hz": (np.arange(12) - 5.5) * 2e6,
        "carrier_frequency_hz": 3.5e9,
        "antenna_spacing_m": None,
        "aoa_grid_rad": np.deg2rad(np.arange(-60, 61, 3)),
        "delay_grid_s": np.arange(0, 161, 4) * 1e-9,
        "num_sources": 2,
        "spatial_subarray_size": 3,
        "frequency_subarray_size": 6,
        "diagonal_loading": 0.01,
    }


def _observations() -> np.ndarray:
    parameters = _parameters()
    rng = np.random.default_rng(923)
    samples = []
    for aoa in [-24, 3, 18, 36, 48]:
        csi = synthesize_ula_csi(
            path_aoa_rad=np.deg2rad([aoa, -45]),
            path_delay_s=[64e-9, 112e-9],
            path_coefficients=[1 + 0.4j, 0.8 - 0.7j],
            subcarrier_frequencies_hz=parameters["subcarrier_frequencies_hz"],
            num_bs_antennas=5,
            carrier_frequency_hz=parameters["carrier_frequency_hz"],
        )
        noise = 0.03 * (rng.standard_normal((3, 5, 12))
                        + 1j * rng.standard_normal((3, 5, 12)))
        samples.append(csi[None] + noise)
    return np.stack(samples)


@pytest.mark.parametrize("batch_size,angle_chunk", [(1, 1), (2, 7), (4, 100)])
def test_batched_numpy_matches_reference_and_preserves_observation_order(
    batch_size: int, angle_chunk: int
) -> None:
    observations = _observations()
    parameters = _parameters()
    expected = np.stack([
        music_2d_spectrum(observation, **parameters) for observation in observations
    ])
    computer = MusicComputer(ComputeSettings(
        batch_size=batch_size, angle_chunk_size=angle_chunk
    ))
    actual = computer.spectra(observations, **parameters)
    npt.assert_allclose(actual, expected, rtol=2e-9, atol=1e-9)
    assert actual.dtype == np.float64
    assert computer.metadata()["completed_csi"] == observations.shape[0]
    assert computer.metadata()["completed_batches"] == int(np.ceil(5 / batch_size))
    assert computer.metadata()["largest_batch"] == min(batch_size, 5)
    for reference, batched in zip(expected, actual):
        peak_args = {key: parameters[key] for key in ("aoa_grid_rad", "delay_grid_s")}
        ref_peaks = extract_local_music_peaks(reference, max_peaks=2, **peak_args)
        batch_peaks = extract_local_music_peaks(batched, max_peaks=2, **peak_args)
        assert [(p.aoa_index, p.delay_index) for p in ref_peaks] == [
            (p.aoa_index, p.delay_index) for p in batch_peaks
        ]


def test_single_csi_shapes_and_mutable_grid_cache() -> None:
    parameters = _parameters()
    observation = _observations()[0, 0]
    computer = MusicComputer()
    first = computer.spectrum(observation, **parameters)
    second = computer.spectrum(observation[None], **parameters)
    npt.assert_array_equal(first, second)
    assert computer.metadata()["steering_cache_hits"] == 1
    parameters["delay_grid_s"][10] += 0.3e-9
    changed = computer.spectrum(observation, **parameters)
    expected = music_2d_spectrum(observation, **parameters)
    npt.assert_allclose(changed, expected, rtol=2e-9)
    assert computer.metadata()["steering_cache_entries"] == 2
    json.dumps(computer.metadata(), allow_nan=False)


def test_steering_cache_has_entry_and_memory_bounds() -> None:
    computer = MusicComputer()
    parameters = _parameters()
    observation = _observations()[0, 0]
    for index in range(6):
        parameters["delay_grid_s"] = _parameters()["delay_grid_s"] + index * 1e-10
        computer.spectrum(observation, **parameters)
    assert computer.metadata()["steering_cache_entries"] == 4
    assert computer.metadata()["steering_cache_misses"] == 6
    computer._CACHE_MAX_BYTES = 1
    parameters["delay_grid_s"] += 0.2e-10
    computer.spectrum(observation, **parameters)
    # 超过内存上限的新网格不放入缓存。
    assert computer.metadata()["steering_cache_entries"] == 4


@pytest.mark.parametrize("settings", [
    {"backend": "auto"}, {"device_id": -1}, {"device_id": True},
    {"batch_size": 0}, {"batch_size": 1.5}, {"angle_chunk_size": False},
])
def test_settings_fail_early(settings: dict) -> None:
    with pytest.raises(ValueError):
        ComputeSettings(**settings)


@pytest.mark.parametrize("parameter,value", [
    ("num_sources", 0), ("num_sources", True),
    ("frequency_subarray_size", 13),
    ("diagonal_loading", -0.1), ("carrier_frequency_hz", 0),
    ("antenna_spacing_m", -1), ("delay_grid_s", [-1e-9, 0]),
    ("delay_grid_s", [0, 0]), ("aoa_grid_rad", [0.1, 0]),
    ("aoa_grid_rad", [-2.0, 0]),
])
def test_validation_agrees_with_reference(parameter: str, value: object) -> None:
    parameters = _parameters()
    parameters[parameter] = value
    observation = _observations()[0]
    with pytest.raises(ValueError):
        music_2d_spectrum(observation, **parameters)
    with pytest.raises(ValueError):
        MusicComputer().spectrum(observation, **parameters)


def test_empty_invalid_and_nonfinite_batches_are_rejected() -> None:
    computer = MusicComputer()
    for shape in [(0, 1, 5, 12), (1, 0, 5, 12), (1, 5, 12), (1, 1, 5, 11)]:
        with pytest.raises(ValueError):
            computer.spectra(np.zeros(shape), **_parameters())
    invalid = _observations()
    invalid[4, 2, 1, 4] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        computer.spectra(invalid, **_parameters())


def test_requested_cuda_never_silently_falls_back(monkeypatch) -> None:
    original = importlib.import_module

    def without_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("test: CuPy missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", without_cupy)
    with pytest.raises(RuntimeError, match="不会自动改用 CPU"):
        MusicComputer(ComputeSettings(backend="cuda"))


def _original_peak_indices(values, max_peaks, relative_height, separation):
    """优化前逐格找峰的完整规则，作为独立回归依据。"""
    threshold = float(np.max(values)) * relative_height
    candidates = []
    for angle in range(values.shape[0]):
        for delay in range(values.shape[1]):
            value = float(values[angle, delay])
            if value < threshold:
                continue
            neighborhood = values[
                max(0, angle - 1):min(values.shape[0], angle + 2),
                max(0, delay - 1):min(values.shape[1], delay + 2),
            ]
            if value >= float(np.max(neighborhood)):
                candidates.append((-value, angle, delay))
    candidates.sort()
    result = []
    for negative_value, angle, delay in candidates:
        if any(abs(angle - a) <= separation[0] and abs(delay - d) <= separation[1]
               for a, d, _ in result):
            continue
        result.append((angle, delay, -negative_value))
        if max_peaks is not None and len(result) >= max_peaks:
            break
    return result


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (15, 1), (23, 27)])
@pytest.mark.parametrize("separation", [(0, 0), (1, 1), (2, 4)])
@pytest.mark.parametrize("relative_height,limit", [(0.0, None), (0.2, 7), (1.0, 3)])
def test_vectorized_peaks_preserve_plateaus_edges_ties_and_suppression(
    shape: tuple[int, int], separation: tuple[int, int],
    relative_height: float, limit: int | None,
) -> None:
    rng = np.random.default_rng(614)
    # 整数谱刻意包含大量同分；全零和全常数谱检查平台行为。
    for values in (rng.integers(0, 5, shape).astype(float),
                   np.zeros(shape), np.full(shape, 3.0)):
        expected = _original_peak_indices(values, limit, relative_height, separation)
        peaks = extract_local_music_peaks(
            values, aoa_grid_rad=np.linspace(-1, 1, shape[0]),
            delay_grid_s=np.arange(shape[1]) * 1e-9, max_peaks=limit,
            minimum_relative_height=relative_height,
            minimum_separation_bins=separation,
        )
        assert [(p.aoa_index, p.delay_index, p.spectrum_value) for p in peaks] == expected


def test_cuda_matches_reference_for_multiple_independent_observations() -> None:
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("没有可用 CUDA GPU")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA 驱动无法访问")
    computer = MusicComputer(ComputeSettings(
        backend="cuda", batch_size=3, angle_chunk_size=7
    ))
    observations = _observations()
    parameters = _parameters()
    actual = computer.spectra(observations, **parameters)
    expected = np.stack([
        music_2d_spectrum(observation, **parameters) for observation in observations
    ])
    npt.assert_allclose(actual, expected, rtol=2e-8, atol=1e-8)
    assert computer.metadata()["backend"] == "cuda"
    assert computer.metadata()["largest_batch"] == 3
    assert computer.metadata()["completed_batches"] == 2
    assert isinstance(actual, np.ndarray)
    for reference, accelerated in zip(expected, actual):
        for limit in (1, 2, 3):
            arguments = dict(aoa_grid_rad=parameters["aoa_grid_rad"],
                             delay_grid_s=parameters["delay_grid_s"], max_peaks=limit)
            reference_peaks = extract_local_music_peaks(reference, **arguments)
            accelerated_peaks = extract_local_music_peaks(accelerated, **arguments)
            assert [(p.aoa_index, p.delay_index) for p in reference_peaks] == [
                (p.aoa_index, p.delay_index) for p in accelerated_peaks
            ]
