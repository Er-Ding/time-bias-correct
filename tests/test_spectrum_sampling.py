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
    assert result.refined_peak_source_indices == [0, 1]
    assert result.refined_peaks == repeated.refined_peaks
    for peak_index, peak in zip(result.refined_peak_source_indices, result.refined_peaks, strict=True):
        records = [r for r in result.records if r["peak_index"] == peak_index]
        assert records[0]["sampling_kind"] == "nominal"
        assert records[0]["aoa_local_rad"] == peak.aoa_rad
        assert records[0]["delay_s"] == peak.delay_s
        region = result.regions[peak_index]
        fine_spectrum = np.asarray(region["spectrum"])
        assert peak.spectrum_value == fine_spectrum.max()
        assert region["coarse_aoa_local_rad"] == peaks[peak_index].aoa_rad
        assert region["coarse_delay_s"] == peaks[peak_index].delay_s
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
    assert result.diagnostics["peak_and_proposal_share_fine_spectrum"] is True
    json.dumps(dict(records=result.records, regions=result.regions, diagnostics=result.diagnostics), allow_nan=False)
    changed = sample_music_spectrum(prepared, peaks, **{**args, "seed": 613})
    assert changed.samples != result.samples


class _AnalyticPrepared:
    """独立可控谱面：检查找峰和采样之间的数据约束。"""

    def __init__(self, function, diagnostic_value=None):
        self.function = function
        self.diagnostic_value = diagnostic_value
        self.grid_calls = 0

    def spectrum(self, *, aoa_grid_rad, delay_grid_s):
        self.grid_calls += 1
        angles, delays = np.meshgrid(aoa_grid_rad, delay_grid_s, indexing="ij")
        return self.function(angles, delays)

    def values(self, *, aoa_rad, delay_s):
        if self.diagnostic_value is not None:
            return np.full_like(aoa_rad, self.diagnostic_value)
        return self.function(aoa_rad, delay_s)

    def metadata(self):
        return {"eigendecomposition_count": 1}


def test_refined_peak_and_cell_mass_share_one_fine_spectrum():
    def surface(angle, delay):
        return 1 + 10 * np.exp(-((angle - 0.025) / 0.04) ** 2 - ((delay - 12.5e-9) / 4e-9) ** 2)

    prepared = _AnalyticPrepared(surface)
    coarse = MusicPeak2D(0.0, 10e-9, float(surface(0.0, 10e-9)), 2, 1)
    grids = dict(aoa_grid_rad=np.linspace(-0.2, 0.2, 5), delay_grid_s=np.linspace(0, 40e-9, 5))
    options = dict(spectrum_power=2.0, uniform_mixture=0.2)
    args = dict(**grids, bs_boresight_rad=0.0, settings=options, seed=4)
    result = sample_music_spectrum(prepared, [coarse], **args)
    peak = result.refined_peaks[0]
    region = result.regions[0]
    assert prepared.grid_calls == 1  # 不再另外计算单元中心谱。
    assert peak.aoa_rad != coarse.aoa_rad
    assert peak.delay_s != coarse.delay_s
    spectrum = np.asarray(region["spectrum"])
    fine_angles, fine_delays = np.asarray(region["aoa_grid_rad"]), np.asarray(region["delay_grid_s"])
    maximum_index = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    assert (peak.aoa_index, peak.delay_index) == maximum_index
    assert peak.aoa_rad == fine_angles[maximum_index[0]]
    assert peak.delay_s == fine_delays[maximum_index[1]]
    assert peak.spectrum_value == spectrum.max()
    corner_mean = (spectrum[:-1, :-1] + spectrum[1:, :-1] + spectrum[:-1, 1:] + spectrum[1:, 1:]) / 4
    np.testing.assert_allclose(region["cell_spectrum"], corner_mean, rtol=1e-14)
    areas = np.diff(fine_angles)[:, None] * np.diff(fine_delays)[None, :]
    mass = corner_mean ** options["spectrum_power"] * areas
    expected = 0.8 * mass / mass.sum() + 0.2 * areas / areas.sum()
    np.testing.assert_allclose(region["cell_probabilities"], expected, rtol=1e-14)
    assert result.records[0]["aoa_local_rad"] == peak.aoa_rad
    assert result.records[0]["delay_s"] == peak.delay_s
    # 精确谱值仅输出诊断；不能改变抽样概率、抽样坐标或正式峰。
    diagnostics_only = sample_music_spectrum(_AnalyticPrepared(surface, diagnostic_value=1e20), [coarse], **args)
    assert diagnostics_only.samples == result.samples
    assert diagnostics_only.refined_peaks == result.refined_peaks
    assert diagnostics_only.regions == result.regions
    assert diagnostics_only.records[1]["spectrum_value"] != result.records[1]["spectrum_value"]


@pytest.mark.parametrize("second_seed,center,kept,removed", [
    (0.05, 0.025, 0, 1),  # 同强度按原来源编号打破平局。
    (0.04, 0.0275, 1, 0),  # 保留更强细峰，即使它的粗峰排在后面。
])
def test_overlapping_refined_peaks_deduplicate_without_renumbering_sources(second_seed, center, kept, removed):
    def surface(angle, delay):
        return (1 + 10 * np.exp(-((angle - center) / 0.008) ** 2 - ((delay - 12.5e-9) / 1e-9) ** 2)
                + 4 * np.exp(-((angle + 0.175) / 0.008) ** 2 - ((delay - 36.25e-9) / 1e-9) ** 2))

    peaks = [MusicPeak2D(angle, delay, float(surface(angle, delay)), index, index)
             for index, (angle, delay) in enumerate([(0.0, 10e-9), (second_seed, 10e-9), (-0.2, 40e-9)])]
    args = dict(aoa_grid_rad=np.linspace(-0.2, 0.2, 5), delay_grid_s=np.linspace(0, 40e-9, 5),
                bs_boresight_rad=0, settings={}, seed=9,
                minimum_angle_separation_rad=0.02, minimum_delay_separation_s=1e-9)
    result = sample_music_spectrum(_AnalyticPrepared(surface), peaks, **args)
    assert result.refined_peak_source_indices == [kept, 2]
    assert {sample.observation_id for sample in result.samples} == {f"music_path_{kept:02d}", "music_path_02"}
    assert {record["peak_index"] for record in result.records} == {kept, 2}
    assert [region["source_peak_index"] for region in result.regions] == [kept, 2]
    assert len(result.samples) == 2 * 129
    suppressed = result.diagnostics["suppressed_refined_peaks"]
    assert len(suppressed) == 1
    assert suppressed[0]["source_peak_index"] == removed
    assert suppressed[0]["kept_source_peak_index"] == kept
    # 两个物理轴必须同时接近才抑制，不能仅凭同角度删除另一时延的峰。
    distinct = sample_music_spectrum(_AnalyticPrepared(surface), peaks, **{**args,
        "minimum_angle_separation_rad": 1.0, "minimum_delay_separation_s": 1e-9})
    assert len(distinct.refined_peaks) == 2


@pytest.mark.parametrize("name", ["minimum_angle_separation_rad", "minimum_delay_separation_s"])
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_physical_peak_separation_rejected(name, value):
    prepared = _AnalyticPrepared(lambda angle, delay: np.ones_like(angle))
    with pytest.raises(ValueError, match=name):
        sample_music_spectrum(prepared, [], aoa_grid_rad=[-0.1, 0.1], delay_grid_s=[0, 1e-9],
                              bs_boresight_rad=0, settings={}, seed=1, **{name: value})


@pytest.mark.parametrize("global_edge", [False, True])
def test_boundary_maximum_is_flagged_and_distinguishes_search_limits(global_edge):
    prepared = _AnalyticPrepared(lambda angle, delay: 2 + angle + delay / 1e-7)
    grids = dict(aoa_grid_rad=np.linspace(-0.2, 0.2, 5), delay_grid_s=np.linspace(0, 40e-9, 5))
    peak = MusicPeak2D(0.2 if global_edge else 0.0, 40e-9 if global_edge else 10e-9, 1.0, 0, 0)
    result = sample_music_spectrum(prepared, [peak], **grids, bs_boresight_rad=0, settings={}, seed=1)
    region = result.regions[0]
    assert region["peak_on_window_edge"] is True
    assert region["refined_peak_boundary_axes"] == ["aoa", "delay"]
    assert region["refined_peak_search_boundary_axes"] == (["aoa", "delay"] if global_edge else [])
    assert region["unresolved_window_peak"] is (not global_edge)
    assert result.diagnostics["unresolved_window_peak_source_indices"] == ([] if global_edge else [0])


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
