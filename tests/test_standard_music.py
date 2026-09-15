"""标准 MUSIC 直接读峰：真实 CSI、阶段边界和无坐标结果回归。"""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from time_bias_localization.compute import MusicComputer
from time_bias_localization.config import (
    DEFAULT_CONFIG, localization_config_view, validate_config,
)
from time_bias_localization.pipeline import generate_data, localize, prepare_scene
from time_bias_localization.signal import synthesize_ula_csi


AUTO = dict(mode="eigenvalue_threshold", noise_reference="median", threshold_ratio=6.0)


def configuration(root):
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(root)
    config["music"].update(subspace_selection=dict(AUTO),
        path_detection=dict(enabled=False, max_paths=6))
    config["music"]["spectrum_sampling"].update(
        samples_per_peak=16, local_grid_points_per_axis=9)
    config["localization"].update(candidate_bias_mode="full_interval",
                                  require_identifiable_solution=True)
    return config


def generated_input(config, root, monkeypatch, *, paths=None, noise_only=False):
    """在保存之前构造受控观测，生成清单中的文件摘要仍然有效。"""
    import time_bias_localization.pipeline as pipeline

    original_generate = pipeline.generate_synthetic_measurement
    if paths is not None or noise_only:
        def generate(scene, settings):
            online, truth = original_generate(scene, settings)
            rng = np.random.default_rng(2)
            noise = .003 * (rng.normal(size=online.csi_observed.shape)
                           + 1j * rng.normal(size=online.csi_observed.shape)) / np.sqrt(2)
            if noise_only:
                observed = noise
            else:
                angles, delays, coefficients = paths
                clean = synthesize_ula_csi(
                    path_aoa_rad=angles, path_delay_s=delays,
                    path_coefficients=coefficients,
                    num_bs_antennas=online.csi_observed.shape[-2],
                    subcarrier_frequencies_hz=online.subcarrier_frequencies_hz,
                    carrier_frequency_hz=online.carrier_frequency_hz,
                    antenna_spacing_m=online.antenna_spacing_m)
                observed = clean[np.newaxis] + noise
            return replace(online, csi_observed=observed), truth
        monkeypatch.setattr(pipeline, "generate_synthetic_measurement", generate)
    scene = prepare_scene(config, root)
    bundle = generate_data(config, scene_json=scene["scene_json"], output_root=root)
    # 删除离线真值后，在线定位仍须能运行；测试不靠清单标志推断隔离有效。
    Path(bundle["truth_npz"]).unlink()
    Path(bundle["truth_json"]).unlink()
    return dict(scene_json=scene["scene_json"], online_input=bundle["online_npz"],
                output_root=root)


def prohibit_residual_detection(monkeypatch):
    import time_bias_localization.path_detection as detection

    def forbidden(*args, **kwargs):
        pytest.fail("标准 MUSIC 不能调用逐次 CSI 残差拟合或验收")
    monkeypatch.setattr(detection, "detect_csi_paths", forbidden)


def assert_standard_diagnostics(result, *, rank, nominal):
    diagnostic = result["diagnostics"]
    assert diagnostic["music_subspace_selection"]["signal_rank"] == rank
    assert diagnostic["requested_music_peak_count"] == rank
    assert diagnostic["nominal_music_peak_count"] == nominal
    feature = diagnostic["music_feature_extraction"]
    assert feature["method"] == "standard_music"
    assert feature["peak_count_source"] == "observed_signal_subspace_rank"
    assert feature["peak_limit"] == rank
    assert feature["iterative_residual_detection"] is False


@pytest.mark.parametrize("noise_only,rank,reason", [
    (False, 1, "insufficient_music_peaks"),
    (True, 0, "no_signal_subspace_above_threshold"),
])
def test_single_path_and_noise_stop_without_inventing_observations(
        tmp_path, monkeypatch, noise_only, rank, reason):
    import time_bias_localization.pipeline as pipeline

    config = configuration(tmp_path)  # 旧 num_paths=3、检测上限6均不能制造更多观测。
    inputs = generated_input(config, tmp_path, monkeypatch,
        paths=([.173], [73.456e-9], [1.]), noise_only=noise_only)
    prohibit_residual_detection(monkeypatch)
    monkeypatch.setattr(pipeline, "sample_music_spectrum",
        lambda *args, **kwargs: pytest.fail("不足两条路径不能进入 MC"))
    result = localize(localization_config_view(config), **inputs)
    assert result["status"] == "unlocalizable"
    assert result["reason"] == reason
    assert result["mu_m"] is None
    assert_standard_diagnostics(result, rank=rank, nominal=rank)
    progress = json.loads(Path(result["progress_path"]).read_text())
    assert progress["truth_was_loaded"] is False
    assert "04_initial_candidates" not in progress["completed_steps"]


class ReachedSampling(RuntimeError):
    """此测试只核查读峰与细化，不把任意合成角度误当作房间内的真实路径。"""


@pytest.mark.parametrize("paths,expected_rank", [
    (([.173, -.612], [73.456e-9, 127.321e-9], [1., .035]), 2),
    ((np.deg2rad([-65, -43, -21, 1, 23, 45, 67]),
      np.array([32, 48, 64, 80, 96, 112, 128]) * 1e-9, np.ones(7)), 7),
])
def test_observed_rank_controls_peak_count_and_weak_path_survives(
        tmp_path, monkeypatch, paths, expected_rank):
    import time_bias_localization.pipeline as pipeline

    config = configuration(tmp_path)
    inputs = generated_input(config, tmp_path, monkeypatch, paths=paths)
    prohibit_residual_detection(monkeypatch)
    original_sample = pipeline.sample_music_spectrum
    seen = {}

    def capture(prepared, coarse_peaks, **kwargs):
        seen["rank"] = prepared.subspace_diagnostics["signal_rank"]
        seen["coarse"] = list(coarse_peaks)
        assert kwargs["accepted_peak_centers"] is False
        seen["sampled"] = original_sample(prepared, coarse_peaks, **kwargs)
        raise ReachedSampling()

    monkeypatch.setattr(pipeline, "sample_music_spectrum", capture)
    with pytest.raises(ReachedSampling):
        localize(localization_config_view(config), **inputs)
    assert seen["rank"] == expected_rank
    assert len(seen["coarse"]) == expected_rank
    refined = seen["sampled"].refined_peaks
    assert len(refined) == expected_rank
    for angle, delay in zip(paths[0], paths[1], strict=True):
        assert any(abs(peak.aoa_rad - angle) < np.deg2rad(.5)
                   and abs(peak.delay_s - delay) < .4e-9 for peak in refined)
    assert len({sample.observation_id for sample in seen["sampled"].samples}) == expected_rank


@pytest.mark.parametrize("duplicate_after_refinement", [False, True])
def test_too_few_peaks_are_a_normal_no_coordinate_result(
        tmp_path, monkeypatch, duplicate_after_refinement):
    import time_bias_localization.pipeline as pipeline

    config = configuration(tmp_path)
    inputs = generated_input(config, tmp_path, monkeypatch)
    prohibit_residual_detection(monkeypatch)
    original_extract = pipeline.extract_local_music_peaks

    def limited_proposals(*args, **kwargs):
        peaks = original_extract(*args, **kwargs)
        assert len(peaks) == 3
        return [peaks[0], peaks[0]] if duplicate_after_refinement else peaks[:1]

    monkeypatch.setattr(pipeline, "extract_local_music_peaks", limited_proposals)
    import time_bias_localization.bias_interval_candidates as reverse
    monkeypatch.setattr(reverse, "generate_bias_interval_points",
        lambda *args, **kwargs: pytest.fail("不足两条正式峰不能进入反向 RT"))
    result = localize(localization_config_view(config), **inputs)
    assert result["status"] == "unlocalizable"
    assert result["mu_m"] is None
    assert result["reason"] == ("insufficient_music_peaks_after_refinement"
                                if duplicate_after_refinement else "insufficient_music_peaks")
    assert_standard_diagnostics(result, rank=3, nominal=1)


def test_standard_music_runs_the_three_path_chain_without_mutating_csi(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline

    config = configuration(tmp_path)
    inputs = generated_input(config, tmp_path, monkeypatch)
    prohibit_residual_detection(monkeypatch)
    online_path = Path(inputs["online_input"])
    original_bytes = online_path.read_bytes()
    original_prepare = MusicComputer.prepare
    original_sample = pipeline.sample_music_spectrum
    preparations = []
    sampling_calls = []

    def prepare(self, csi, **kwargs):
        before = csi.copy()
        prepared = original_prepare(self, csi, **kwargs)
        preparations.append((csi, before, prepared))
        return prepared

    def sample(prepared, coarse_peaks, **kwargs):
        assert prepared is preparations[0][2]
        assert kwargs["accepted_peak_centers"] is False
        sampling_calls.append(len(coarse_peaks))
        return original_sample(prepared, coarse_peaks, **kwargs)

    monkeypatch.setattr(MusicComputer, "prepare", prepare)
    monkeypatch.setattr(pipeline, "sample_music_spectrum", sample)
    result = localize(localization_config_view(config), **inputs)
    assert result["status"] == "success", result.get("reason")
    assert_standard_diagnostics(result, rank=3, nominal=3)
    assert len(preparations) == 1
    assert sampling_calls == [3]
    np.testing.assert_array_equal(preparations[0][0], preparations[0][1])
    assert online_path.read_bytes() == original_bytes
    assert result["diagnostics"]["initial_candidate_count"] > 0
    assert result["diagnostics"]["representative_trajectory_count"] > 0
    manifest = json.loads((tmp_path / "localization/localization_manifest.json").read_text())
    assert manifest["truth_was_loaded"] is False
    peaks = json.loads((tmp_path / "localization/music_peaks.json").read_text())
    assert "path_detection" not in peaks
    assert len(peaks["nominal"]) == 3
    assert peaks["nominal_resolution"] == "local_fine_spectrum"


def test_automatic_mode_allows_one_configured_path_but_fixed_mode_does_not(tmp_path):
    config = configuration(tmp_path)
    config["music"]["num_paths"] = 1
    validate_config(config)
    config["music"]["subspace_selection"] = dict(mode="fixed")
    with pytest.raises(ValueError, match="至少需要两条"):
        validate_config(config)


def test_fixed_mode_still_uses_configured_number_of_music_peaks(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline

    config = configuration(tmp_path)
    config["music"].update(num_paths=2, signal_subspace_rank=3,
                           subspace_selection=dict(mode="fixed"))
    inputs = generated_input(config, tmp_path, monkeypatch)
    prohibit_residual_detection(monkeypatch)

    def capture(prepared, coarse_peaks, **kwargs):
        assert prepared.subspace_diagnostics["signal_rank"] == 3
        assert len(coarse_peaks) == 2
        raise ReachedSampling()

    monkeypatch.setattr(pipeline, "sample_music_spectrum", capture)
    with pytest.raises(ReachedSampling):
        localize(localization_config_view(config), **inputs)


def test_unresolved_noise_reference_preserves_standard_music_diagnostics(tmp_path, monkeypatch):
    import time_bias_localization.pipeline as pipeline
    from time_bias_localization.music_subspace import select_subspace_rank

    config = configuration(tmp_path)
    inputs = generated_input(config, tmp_path, monkeypatch)
    prohibit_residual_detection(monkeypatch)

    def unresolved(self, csi, **kwargs):
        # 此退化协方差无法提供正的噪声参考，使用真实分界器产生错误及诊断。
        select_subspace_rank([0., 0., 0., 1.], fixed_rank=3, settings=AUTO)
        pytest.fail("退化协方差应抛出 SubspaceSelectionError")

    monkeypatch.setattr(MusicComputer, "prepare", unresolved)
    monkeypatch.setattr(pipeline, "sample_music_spectrum",
        lambda *args, **kwargs: pytest.fail("噪声参考未解析时不能进入 MC"))
    result = localize(localization_config_view(config), **inputs)
    assert result["status"] == "detection_incomplete"
    assert result["reason"] == "music_noise_reference_unresolved"
    assert result["mu_m"] is None
    diagnostic = result["diagnostics"]
    assert diagnostic["music_subspace_selection"]["signal_rank"] is None
    assert diagnostic["requested_music_peak_count"] is None
    assert diagnostic["nominal_music_peak_count"] == 0
    feature = diagnostic["music_feature_extraction"]
    assert feature["method"] == "standard_music"
    assert feature["iterative_residual_detection"] is False
    assert feature["peak_limit"] is None
    progress = json.loads(Path(result["progress_path"]).read_text())
    peaks = json.loads((Path(result["progress_path"]).parent / "music_peaks.json").read_text())
    assert progress["truth_was_loaded"] is False
    assert peaks["feature_extraction"] == feature
