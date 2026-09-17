"""检查绕射方向、合成 CSI 前的过滤，以及带负偏置的真实观测相位。"""
from copy import deepcopy
import math

import numpy as np
import pytest

from time_bias_localization.path_policy import path_type_counts, uplink_path_mask
from time_bias_localization.sionna_generation import extract_planar_uplink_csi
from time_bias_localization.propagation_hypotheses import build_hypothesis_bank
from time_bias_localization.propagation_model import ContinuousObservation
from time_bias_localization.scene import Scene2D, WallSegment


class OrderedPaths:
    def __init__(self):
        # 上行顺序。前六条应进入 CSI；后四条是不允许的位置/次数。
        sequences = [(), (1,), (1, 1), (8,), (8, 1), (8, 1, 1), (1, 8), (1, 1, 8), (1, 8, 1), (8, 8)]
        n = len(sequences)
        self.interactions = np.zeros((3, 1, 1, n), dtype=int)
        for i, seq in enumerate(sequences):
            self.interactions[:len(seq), 0, 0, i] = seq
        self.vertices = np.zeros((3, 1, 1, n, 3))
        self.vertices[..., 2] = 1.5
        self.phi_r = np.zeros((1, 1, n))
        self.tau = (100 + np.arange(n) * 10).reshape(1, 1, n) * 1e-9

    def cir(self, *, normalize_delays=False, **kwargs):
        assert normalize_delays is False
        return np.ones((1, 2, 1, 1, self.tau.size, 1), complex), self.tau


def test_six_families_are_filtered_before_csi_synthesis():
    paths = OrderedPaths()
    frequencies = np.arange(8) * 1e6
    csi, meta = extract_planar_uplink_csi(paths, frequencies, fixed_height_m=1.5, vertical_tolerance_m=.1,
        max_reflections=2, max_diffractions=1, diffraction_position="last_from_bs", minimum_path_count=0,
        coefficient_zero_threshold=0.0)
    np.testing.assert_array_equal(meta["retained_mask"], [True]*6 + [False]*4)
    assert path_type_counts(meta["interactions"], meta["retained_mask"]) == [1]*6
    expected = np.exp(-2j * np.pi * paths.tau[0, 0, :6, None] * frequencies).sum(axis=0)
    np.testing.assert_allclose(csi, np.tile(expected, (2, 1)))
    assert uplink_path_mask(np.zeros((0, 0), int), "last_from_bs").shape == (0,)
    with pytest.raises(ValueError, match="六类之外"):
        path_type_counts(meta["interactions"], np.ones(paths.tau.size, dtype=bool))


def test_inverse_bank_has_only_same_six_families():
    scene = Scene2D("screen", (-10., 10., -10., 10.),
        (WallSegment("screen", (0., -6.), (0., 0.)), WallSegment("right", (8., -9.), (8., 9.))), 1.5, .1, "test")
    obs = [ContinuousObservation("a", 0., 20., .02, .5)]
    bank = build_hypothesis_bank(scene, (4., 4.), obs, max_reflections=2, max_diffractions=1,
                                diffraction_position="last_from_bs", max_hypotheses=10000)
    assert len(bank.search_report["families"]) == 6
    assert any(h.diffraction_order for h in bank.hypotheses)
    for h in bank.hypotheses:
        if h.diffraction_order:
            assert h.interactions[0][0] == "diffraction"
            assert all(kind == "reflection" for kind, _ in h.interactions[1:])


def test_generation_and_localization_must_agree_on_diffraction_position(tmp_path):
    from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view
    from time_bias_localization.boundary_channel import make_boundary_channel
    from time_bias_localization.contracts import validate_generation_manifest_envelope, validate_localization_input_contract
    from time_bias_localization.data import load_online_measurement
    import json
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["scene"].update(max_reflections=2, max_diffractions=1, diffraction_position="last_from_bs")
    provider = make_boundary_channel(cfg, tmp_path / "setup", backend="synthetic_fixture")
    probe = provider.probe([10., 7.], seed=1)
    bundle = provider.write_observation_bundle(probe, output_root=tmp_path / "observation", noise_seed=9)
    manifest = json.loads(open(bundle["generation_manifest"]).read())
    stage, _ = validate_generation_manifest_envelope(manifest)
    public = localization_config_view(cfg)
    measurement = load_online_measurement(bundle["online_npz"])
    validate_localization_input_contract(manifest, stage, public, provider.scene_2d, measurement)
    public["scene"]["diffraction_position"] = "any"
    with pytest.raises(ValueError, match="绕射位置"):
        validate_localization_input_contract(manifest, stage, public, provider.scene_2d, measurement)
    provider.close()


def test_negative_observed_delay_is_detected_from_biased_csi():
    from time_bias_localization.signal import apply_common_delay_bias, synthesize_ula_csi, music_2d_spectrum, extract_local_music_peaks
    from time_bias_localization.compute import MusicComputer
    from time_bias_localization.spectrum_sampling import refine_music_peaks
    frequencies = np.arange(64) * 2e6
    clean = synthesize_ula_csi(path_aoa_rad=[.2], path_delay_s=[10e-9], path_coefficients=[1.],
        subcarrier_frequencies_hz=frequencies, num_bs_antennas=6, carrier_frequency_hz=3.5e9, antenna_spacing_m=None)
    noisy = apply_common_delay_bias(clean, frequencies, -25e-9, noise_std=.001, seed=19)
    args = dict(subcarrier_frequencies_hz=frequencies, carrier_frequency_hz=3.5e9, antenna_spacing_m=None,
                num_sources=1, spatial_subarray_size=4, frequency_subarray_size=16, diagonal_loading=1e-8)
    grids = dict(aoa_grid_rad=np.linspace(-.4, .4, 41), delay_grid_s=np.arange(-30., 31.) * 1e-9)
    reference = music_2d_spectrum(noisy, **args, **grids)
    computer = MusicComputer()
    spectrum = computer.spectrum(noisy, **args, **grids)
    np.testing.assert_allclose(spectrum, reference, rtol=1e-5, atol=1e-3)
    peaks = extract_local_music_peaks(spectrum, **grids, max_peaks=1)
    assert abs(peaks[0].delay_s - (-15e-9)) < 1e-15
    prepared = computer.prepare(noisy, **args)
    refined = refine_music_peaks(prepared, peaks, **grids, bs_boresight_rad=0., settings={})
    assert abs(refined.refined_peaks[0].delay_s - (-15e-9)) < .2e-9
    assert not refined.samples
    assert np.isfinite(prepared.values(aoa_rad=[.2], delay_s=[-15e-9])).all()
