"""信号模块的核心物理约定与可复现性测试。"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from time_bias_localization.signal import (
    SPEED_OF_LIGHT_M_S,
    apply_common_delay_bias,
    estimate_music_peak_samples,
    extract_local_music_peaks,
    music_2d_spectrum,
    synthesize_ula_csi,
)


CARRIER_FREQUENCY_HZ = 3.5e9
ANTENNA_SPACING_M = SPEED_OF_LIGHT_M_S / CARRIER_FREQUENCY_HZ / 2.0
SUBCARRIER_FREQUENCIES_HZ = (
    np.arange(48, dtype=np.float64) - 23.5
) * 2.0e6
AOA_GRID_RAD = np.deg2rad(np.arange(-60.0, 61.0, 1.0))
DELAY_GRID_S = np.arange(0.0, 202.0, 2.0) * 1e-9


def _single_path_csi(aoa_deg: float, delay_ns: float) -> np.ndarray:
    return synthesize_ula_csi(
        path_aoa_rad=np.deg2rad([aoa_deg]),
        path_delay_s=np.array([delay_ns * 1e-9]),
        path_coefficients=np.array([0.8 - 0.3j]),
        subcarrier_frequencies_hz=SUBCARRIER_FREQUENCIES_HZ,
        num_bs_antennas=8,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
        antenna_spacing_m=ANTENNA_SPACING_M,
    )


def _strongest_music_peak(csi: np.ndarray):
    spectrum = music_2d_spectrum(
        csi,
        subcarrier_frequencies_hz=SUBCARRIER_FREQUENCIES_HZ,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
        antenna_spacing_m=ANTENNA_SPACING_M,
        aoa_grid_rad=AOA_GRID_RAD,
        delay_grid_s=DELAY_GRID_S,
        num_sources=1,
        spatial_subarray_size=5,
        frequency_subarray_size=20,
    )
    return extract_local_music_peaks(
        spectrum,
        aoa_grid_rad=AOA_GRID_RAD,
        delay_grid_s=DELAY_GRID_S,
        max_peaks=1,
    )[0]


def test_single_path_music_recovers_aoa_and_absolute_delay() -> None:
    csi = _single_path_csi(aoa_deg=23.0, delay_ns=80.0)

    peak = _strongest_music_peak(csi)

    assert np.rad2deg(peak.aoa_rad) == 23.0
    assert peak.delay_s == 80.0e-9
    assert peak.spectrum_value > 1.0


def test_positive_common_bias_increases_observed_delay() -> None:
    geometric_delay_ns = 50.0
    positive_bias_ns = 20.0
    geometric = _single_path_csi(aoa_deg=-17.0, delay_ns=geometric_delay_ns)

    observed = apply_common_delay_bias(
        geometric,
        SUBCARRIER_FREQUENCIES_HZ,
        positive_bias_ns * 1e-9,
    )

    expected_multiplier = np.exp(
        -2.0j
        * np.pi
        * SUBCARRIER_FREQUENCIES_HZ
        * positive_bias_ns
        * 1e-9
    )
    npt.assert_allclose(observed, geometric * expected_multiplier, atol=1e-13)
    peak = _strongest_music_peak(observed)
    assert np.rad2deg(peak.aoa_rad) == -17.0
    assert peak.delay_s == (geometric_delay_ns + positive_bias_ns) * 1e-9


def test_sionna_receive_array_phase_recovers_same_aoa_not_mirror() -> None:
    aoa_deg = 23.0
    delay_s = 80.0e-9
    centered_positions = (
        np.arange(8, dtype=float) - 3.5
    ) * ANTENNA_SPACING_M
    spatial = np.exp(
        2.0j
        * np.pi
        * centered_positions
        * np.sin(np.deg2rad(aoa_deg))
        / (SPEED_OF_LIGHT_M_S / CARRIER_FREQUENCY_HZ)
    )
    frequency = np.exp(
        -2.0j * np.pi * SUBCARRIER_FREQUENCIES_HZ * delay_s
    )
    sionna_style_csi = spatial[:, np.newaxis] * frequency[np.newaxis, :]

    peak = _strongest_music_peak(sionna_style_csi)

    assert np.rad2deg(peak.aoa_rad) == aoa_deg
    assert peak.delay_s == delay_s


def test_peak_perturbation_samples_are_reproducible() -> None:
    observed = _single_path_csi(aoa_deg=11.0, delay_ns=96.0)
    keyword_arguments = dict(
        subcarrier_frequencies_hz=SUBCARRIER_FREQUENCIES_HZ,
        carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
        antenna_spacing_m=ANTENNA_SPACING_M,
        aoa_grid_rad=AOA_GRID_RAD,
        delay_grid_s=DELAY_GRID_S,
        noise_std=0.03,
        num_repetitions=4,
        seed=20260904,
        num_sources=1,
        peaks_per_repetition=2,
        spatial_subarray_size=4,
        frequency_subarray_size=16,
    )

    first = estimate_music_peak_samples(observed, **keyword_arguments)
    second = estimate_music_peak_samples(observed, **keyword_arguments)

    npt.assert_array_equal(first.aoa_rad, second.aoa_rad)
    npt.assert_array_equal(first.delay_s, second.delay_s)
    npt.assert_array_equal(first.spectrum_value, second.spectrum_value)
    assert first.aoa_rad.shape == (4, 2)
    assert first.delay_s.shape == (4, 2)


def test_csi_shape_and_input_validation() -> None:
    csi = _single_path_csi(aoa_deg=0.0, delay_ns=40.0)
    assert csi.shape == (8, SUBCARRIER_FREQUENCIES_HZ.size)

    with np.testing.assert_raises_regex(ValueError, "长度必须相同"):
        synthesize_ula_csi(
            path_aoa_rad=[0.0, 0.1],
            path_delay_s=[40e-9],
            path_coefficients=[1.0 + 0.0j],
            subcarrier_frequencies_hz=SUBCARRIER_FREQUENCIES_HZ,
            num_bs_antennas=8,
            carrier_frequency_hz=CARRIER_FREQUENCY_HZ,
            antenna_spacing_m=ANTENNA_SPACING_M,
        )
