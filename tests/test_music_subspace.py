"""子空间分界、幅度不变性以及 MUSIC -> CSI 验收的真实观测回归。"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from time_bias_localization.compute import MusicComputer, ComputeSettings
from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view, validate_config
from time_bias_localization.music_subspace import (
    select_subspace_rank, subspace_settings, SubspaceSelectionError,
)
from time_bias_localization.path_detection import detect_csi_paths
from time_bias_localization.pipeline import prepare_scene, generate_data, localize
from time_bias_localization.signal import synthesize_ula_csi, extract_local_music_peaks


AUTO = dict(mode="eigenvalue_threshold", noise_reference="median", threshold_ratio=6.0)


def receiver():
    return dict(subcarrier_frequencies_hz=np.arange(64)*3.125e6,
                carrier_frequency_hz=3.5e9, antenna_spacing_m=None,
                spatial_subarray_size=4, frequency_subarray_size=16)


def observation(weak=False, noise_only=False):
    r = receiver()
    clean = synthesize_ula_csi(path_aoa_rad=[.173, -.612] if weak else [.173],
        path_delay_s=[123.456e-9, 217.321e-9] if weak else [123.456e-9],
        path_coefficients=[1., .035] if weak else [1.], num_bs_antennas=8,
        **{key: r[key] for key in ("subcarrier_frequencies_hz", "carrier_frequency_hz", "antenna_spacing_m")})
    rng = np.random.default_rng(2)
    noise = .01*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))/np.sqrt(2)
    return noise if noise_only else clean+noise


def accepted(data, *, max_paths=3, proposals=None):
    args = receiver()
    prepared = MusicComputer().prepare(data, **args, subspace_selection=AUTO)
    angles, delays = np.linspace(-1.2, 1.2, 61), np.linspace(0, 280e-9, 141)
    if proposals is None:
        proposals = extract_local_music_peaks(prepared.spectrum(aoa_grid_rad=angles, delay_grid_s=delays),
            aoa_grid_rad=angles, delay_grid_s=delays, max_peaks=None,
            minimum_relative_height=0., minimum_separation_bins=(1,1))
    detection = detect_csi_paths(data, **{key: args[key] for key in (
        "subcarrier_frequencies_hz", "carrier_frequency_hz", "antenna_spacing_m")},
        aoa_grid_rad=angles, delay_grid_s=delays, proposal_peaks=proposals,
        settings=dict(enabled=True, max_paths=max_paths, false_alarm_probability=.05,
                      calibration_trials=127, max_refine_evaluations=45))
    return prepared, detection


@pytest.mark.parametrize("scale", [1e-16, 1., 1e6])
def test_relative_cutoff_and_equality_are_scale_invariant(scale):
    values = np.array([1.,1.,1.,2.,3.,9.])*scale
    rank, diagnostic = select_subspace_rank(values, fixed_rank=999,
        settings=dict(mode="eigenvalue_threshold", noise_reference="minimum", threshold_ratio=2.))
    assert rank == 2
    assert diagnostic["noise_rank"] == 4  # 等于分界的特征值归入噪声。
    assert diagnostic["threshold"] == pytest.approx(2*scale, rel=1e-12, abs=0)


def test_rank_above_six_and_noise_only_are_not_clamped():
    rank, _ = select_subspace_rank([1.]*10+[10.]*7, fixed_rank=6, settings=AUTO)
    assert rank == 7
    rank, _ = select_subspace_rank([.8,.9,1.,1.1], fixed_rank=6, settings=AUTO)
    assert rank == 0
    with pytest.raises(SubspaceSelectionError):
        select_subspace_rank([0.,0.,0.,1.], fixed_rank=1, settings=AUTO)


@pytest.mark.parametrize("invalid", [{"threshold_ratio":1}, {"threshold_ratio":True},
    {"noise_reference":"true_noise"}, {"mode":"unknown"}, {"snr_db":35}])
def test_selection_rejects_invalid_and_oracle_settings(invalid):
    with pytest.raises(ValueError):
        subspace_settings(invalid)


def test_noise_only_csi_has_zero_signal_rank_and_flat_spectrum():
    prepared = MusicComputer().prepare(observation(noise_only=True), **receiver(), subspace_selection=AUTO)
    assert prepared.subspace_diagnostics["signal_rank"] == 0
    assert np.all(prepared.spectrum(aoa_grid_rad=np.array([-.5,.5]), delay_grid_s=np.array([0.,1e-7])) == 1.)


def test_observed_csi_amplitude_and_loading_do_not_set_the_rank():
    data = observation(weak=True)
    computer = MusicComputer()
    baseline = computer.prepare(data, **receiver(), num_sources=99, subspace_selection=AUTO)
    assert baseline.subspace_diagnostics["signal_rank"] == 2
    for amplitude, loading in [(1e-8,0.), (1e3,0.), (1.,.01)]:
        actual = computer.prepare(amplitude*data, **receiver(), num_sources=1,
                                  diagonal_loading=loading, subspace_selection=AUTO)
        assert actual.subspace_diagnostics["signal_rank"] == 2
        assert actual.subspace_diagnostics["threshold"] / amplitude**2 == pytest.approx(
            baseline.subspace_diagnostics["threshold"], rel=1e-7)


def test_mixed_batch_uses_each_observation_rank_and_matches_noise_projection():
    data = [observation(), observation(weak=True), observation(noise_only=True)]
    r = receiver()
    grid = dict(aoa_grid_rad=np.linspace(-1,1,11), delay_grid_s=np.linspace(0,280e-9,21))
    computer = MusicComputer(ComputeSettings(batch_size=3))
    batched = computer.spectra(np.array(data)[:,None], **r, **grid, subspace_selection=AUTO)
    for i, item in enumerate(data):
        prepared = computer.prepare(item, **r, subspace_selection=AUTO)
        np.testing.assert_allclose(batched[i], prepared.spectrum(**grid), rtol=1e-9)
        # 独立构造噪声投影，避免仅用同一条计算路径相互比较。
        windows = np.lib.stride_tricks.sliding_window_view(item,(4,16)).reshape(-1,64)
        values, vectors = np.linalg.eigh(windows.T @ windows.conj()/len(windows))
        noise_vectors = vectors[:, values <= prepared.subspace_diagnostics["threshold"]]
        spatial = np.exp(1j*np.pi*np.arange(4)[:,None]*np.sin(grid["aoa_grid_rad"]))
        frequency = np.exp(-2j*np.pi*r["subcarrier_frequencies_hz"][:16,None]*grid["delay_grid_s"])
        atoms = (spatial[:,None,:,None]*frequency[None,:,None,:]).reshape(64,-1)/8
        expected = (1/np.sum(abs(noise_vectors.conj().T @ atoms)**2,axis=0)).reshape(11,21)
        np.testing.assert_allclose(batched[i],expected,rtol=1e-8)


@pytest.mark.parametrize("weak,count", [(False,1),(True,2)])
def test_music_proposals_accept_single_and_weak_additional_path(weak,count):
    prepared, result = accepted(observation(weak=weak))
    assert len(result.peaks) == count
    assert result.diagnostics["proposal_source"] == "observed_csi_music_local_maxima"
    assert result.diagnostics["stop_reason"] == "residual_has_no_significant_additional_component"
    if weak:
        assert min(abs(p.aoa_rad+.612) for p in result.peaks) < .01
        assert min(abs(p.delay_s-217.321e-9) for p in result.peaks) < .5e-9


def test_missing_music_peak_is_incomplete_instead_of_falsely_noise_only():
    _, result = accepted(observation(weak=True), proposals=[])
    assert result.diagnostics["stop_reason"] == "music_proposals_exhausted_with_significant_residual"
    assert not result.peaks


@pytest.mark.parametrize("budget", [1,3])
def test_pipeline_music_precedes_acceptance_and_rank_is_independent_of_budget(tmp_path,budget,monkeypatch):
    import time_bias_localization.path_detection as detection_module
    import time_bias_localization.pipeline as pipeline
    import time_bias_localization.compute as compute_module
    config = deepcopy(DEFAULT_CONFIG)
    config["output"]["root"] = str(tmp_path)
    config["music"].update(signal_subspace_rank=999, subspace_selection=AUTO,
        path_detection=dict(enabled=True,max_paths=budget,false_alarm_probability=.05,
                            calibration_trials=127,max_refine_evaluations=45))
    config["music"]["spectrum_sampling"].update(samples_per_peak=16,local_grid_points_per_axis=9)
    config["localization"].update(candidate_bias_mode="full_interval",require_identifiable_solution=True)
    validate_config(config)
    scene = prepare_scene(config,tmp_path)
    bundle = generate_data(config,scene_json=scene["scene_json"],output_root=tmp_path)
    order = []
    original_spectrum, original_detector = compute_module.PreparedMusic.spectrum, detection_module.detect_csi_paths
    def spectrum(self,**kwargs):
        order.append("music")
        return original_spectrum(self,**kwargs)
    def detect(*args,**kwargs):
        assert "music" in order
        assert kwargs["proposal_peaks"]
        order.append("acceptance")
        return original_detector(*args,**kwargs)
    monkeypatch.setattr(compute_module.PreparedMusic,"spectrum",spectrum)
    monkeypatch.setattr(detection_module,"detect_csi_paths",detect)
    if budget == 1:
        monkeypatch.setattr(pipeline,"sample_music_spectrum",lambda *a,**k: pytest.fail("未验收完不能进入 MC"))
    result = localize(localization_config_view(config),scene_json=scene["scene_json"],
                      online_input=bundle["online_npz"],output_root=tmp_path)
    assert order.index("music") < order.index("acceptance")
    assert result["diagnostics"]["music_subspace_selection"]["signal_rank"] == 3
    if budget == 1:
        assert result["status"] == "detection_incomplete"
        assert result["reason"] == "path_detection_budget_exhausted"
        assert result["mu_m"] is None
    else:
        assert result["status"] == "success", result.get("reason")
        peaks=json.loads((tmp_path/"localization/music_peaks.json").read_text())
        assert len(peaks["nominal"]) == 3
        for nominal, accepted_peak in zip(peaks["nominal"],peaks["path_detection"]["accepted_peaks"]):
            assert nominal["aoa_rad"] == accepted_peak["aoa_rad"]
            assert nominal["delay_s"] == accepted_peak["delay_s"]
    metadata_path = (Path(result["progress_path"]) if budget == 1
                     else tmp_path/"localization/localization_manifest.json")
    progress=json.loads(metadata_path.read_text())
    assert progress["truth_was_loaded"] is False
