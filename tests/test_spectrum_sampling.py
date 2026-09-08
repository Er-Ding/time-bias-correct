"""单份带噪 CSI 的子空间复用与连续谱面采样，独立于定位真值。"""

import json

import numpy as np
import pytest

from time_bias_localization.compute import ComputeSettings, MusicComputer
from time_bias_localization.signal import (
    MusicPeak2D, extract_local_music_peaks, music_2d_spectrum, synthesize_ula_csi,
)
from time_bias_localization.spectrum_sampling import sample_music_spectrum


def _input():
    frequencies = (np.arange(12) - 5.5) * 2e6
    kwargs = dict(
        subcarrier_frequencies_hz=frequencies, carrier_frequency_hz=3.5e9,
        antenna_spacing_m=None, num_sources=2, spatial_subarray_size=3,
        frequency_subarray_size=6, diagonal_loading=0.01,
    )
    grids = dict(aoa_grid_rad=np.deg2rad(np.arange(-60, 61, 3)),
                 delay_grid_s=np.arange(0, 161, 4) * 1e-9)
    csi = synthesize_ula_csi(
        path_aoa_rad=np.deg2rad([12.4, -33.7]), path_delay_s=[63.7e-9, 116.3e-9],
        path_coefficients=[1 + 0.4j, 0.8 - 0.7j], num_bs_antennas=5,
        subcarrier_frequencies_hz=frequencies, carrier_frequency_hz=3.5e9,
    )
    rng = np.random.default_rng(17)
    csi += 0.03 * (rng.normal(size=csi.shape) + 1j * rng.normal(size=csi.shape))
    return csi, kwargs, grids


def test_prepared_grid_and_paired_values_match_reference_with_only_one_eigh(monkeypatch):
    csi, kwargs, grids = _input()
    original = csi.copy()
    expected = music_2d_spectrum(csi, **kwargs, **grids)
    calls = []
    eigh = np.linalg.eigh

    def counted(value):
        calls.append(value.shape)
        return eigh(value)

    monkeypatch.setattr(np.linalg, "eigh", counted)
    computer = MusicComputer(ComputeSettings(angle_chunk_size=3))
    prepared = computer.prepare(csi, **kwargs)
    actual = prepared.spectrum(**grids)
    angles, delays = np.meshgrid(grids["aoa_grid_rad"], grids["delay_grid_s"], indexing="ij")
    paired = prepared.values(aoa_rad=angles.ravel(), delay_s=delays.ravel()).reshape(actual.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-9)
    np.testing.assert_allclose(paired, expected, rtol=2e-9)
    np.testing.assert_array_equal(csi, original)
    assert len(calls) == 1
    assert prepared.metadata()["eigendecomposition_count"] == 1
    assert computer.metadata()["completed_csi"] == 1
    assert computer.metadata()["eigendecomposition_count"] == 1


def test_fixed_csi_produces_reproducible_subgrid_samples_with_valid_provenance():
    csi, kwargs, grids = _input()
    computer = MusicComputer()
    prepared = computer.prepare(csi, **kwargs)
    peaks = extract_local_music_peaks(prepared.spectrum(**grids), max_peaks=2, **grids)
    args = dict(**grids, bs_boresight_rad=2.8, settings={}, seed=612)
    result = sample_music_spectrum(prepared, peaks, **args)
    repeated = sample_music_spectrum(prepared, peaks, **args)
    assert result.samples == repeated.samples
    assert result.records == repeated.records
    assert len(result.samples) == 2 * 129
    assert len({sample.sample_id for sample in result.samples}) == len(result.samples)
    for peak_index, peak in enumerate(peaks):
        records = [r for r in result.records if r["peak_index"] == peak_index]
        assert records[0]["sampling_kind"] == "nominal"
        assert records[0]["aoa_local_rad"] == peak.aoa_rad
        assert records[0]["delay_s"] == peak.delay_s
        mc = records[1:]
        assert len({(r["aoa_local_rad"], r["delay_s"]) for r in mc}) == 128
        assert all(not np.any(np.isclose(grids["aoa_grid_rad"], r["aoa_local_rad"], atol=1e-14, rtol=0)) for r in mc)
        assert all(not np.any(np.isclose(grids["delay_grid_s"], r["delay_s"], atol=1e-20, rtol=0)) for r in mc)
    assert all(s.weight == 1 for s in result.samples)
    for record in result.records:
        expected_angle = (record["aoa_local_rad"] + 2.8 + np.pi) % (2 * np.pi) - np.pi
        assert record["aoa_global_rad"] == expected_angle
    assert computer.metadata()["eigendecomposition_count"] == 1
    assert result.diagnostics["added_csi_noise"] is False
    assert result.diagnostics["proposal_is_calibrated_probability"] is False
    json.dumps(dict(records=result.records, regions=result.regions, diagnostics=result.diagnostics), allow_nan=False)
    changed = sample_music_spectrum(prepared, peaks, **{**args, "seed": 613})
    assert changed.samples != result.samples


@pytest.mark.parametrize("angle_end,delay_end", [(0, 0), (-1, -1), (0, -1), (-1, 0)])
def test_search_boundary_clipping_and_uniform_cell_mass(angle_end, delay_end):
    csi, kwargs, grids = _input()
    prepared = MusicComputer().prepare(csi, **kwargs)
    peak = MusicPeak2D(float(grids["aoa_grid_rad"][angle_end]),
                      float(grids["delay_grid_s"][delay_end]), 1.0, angle_end, delay_end)
    result = sample_music_spectrum(prepared, [peak], **grids, bs_boresight_rad=0,
                                   settings={"uniform_mixture": 1.0, "include_nominal": False}, seed=3)
    assert len(result.samples) == 128
    region = result.regions[0]
    assert region["bounds_clipped"]
    probabilities = np.asarray(region["cell_probabilities"])
    np.testing.assert_allclose(probabilities, 1 / probabilities.size, rtol=1e-12)
    for sample in result.samples:
        assert grids["aoa_grid_rad"][0] <= sample.aoa_global_rad <= grids["aoa_grid_rad"][-1]
        assert grids["delay_grid_s"][0] <= sample.delay_s <= grids["delay_grid_s"][-1]


@pytest.mark.parametrize("options", [
    {"samples_per_peak": 0}, {"samples_per_peak": True},
    {"local_grid_points_per_axis": 1}, {"spectrum_power": 0},
    {"uniform_mixture": -0.1}, {"uniform_mixture": float("nan")},
    {"aoa_half_width_grid_steps": -1}, {"include_nominal": 1}, {"truth": 4},
])
def test_invalid_sampling_settings_rejected(options):
    csi, kwargs, grids = _input()
    prepared = MusicComputer().prepare(csi, **kwargs)
    with pytest.raises(ValueError):
        sample_music_spectrum(prepared, [], **grids, bs_boresight_rad=0, settings=options, seed=0)


def test_cpu_cuda_preparation_and_continuous_values_agree():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("没有可用 CUDA GPU")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA 驱动无法访问")
    csi, kwargs, grids = _input()
    cpu = MusicComputer().prepare(csi, **kwargs)
    gpu = MusicComputer(ComputeSettings(backend="cuda")).prepare(csi, **kwargs)
    peaks = extract_local_music_peaks(cpu.spectrum(**grids), max_peaks=2, **grids)
    args = dict(**grids, bs_boresight_rad=0, settings={}, seed=11)
    cpu_result = sample_music_spectrum(cpu, peaks, **args)
    gpu_result = sample_music_spectrum(gpu, peaks, **args)
    assert cpu_result.samples == gpu_result.samples
    np.testing.assert_allclose([r["spectrum_value"] for r in cpu_result.records],
                               [r["spectrum_value"] for r in gpu_result.records], rtol=2e-8)
    assert gpu.metadata()["eigendecomposition_count"] == 1
