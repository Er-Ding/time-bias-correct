from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.compute import MusicComputer, ComputeSettings
from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view, validate_localization_config
from time_bias_localization.music_stage import estimate_batched_peak_samples
from time_bias_localization.signal import estimate_music_peak_samples
from time_bias_localization.sionna_generation import _sionna_runtime_info


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("snapshots", [False, True])
def test_perturbation_batching_preserves_reference_noise_and_peaks(batch_size, snapshots):
    rng = np.random.default_rng(12)
    shape = (2, 4, 10) if snapshots else (4, 10)
    observation = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    parameters = dict(
        subcarrier_frequencies_hz=np.arange(10) * 2e6,
        carrier_frequency_hz=3.5e9, antenna_spacing_m=None,
        aoa_grid_rad=np.linspace(-1, 1, 13), delay_grid_s=np.linspace(0, 200e-9, 21),
        noise_std=0.2, num_repetitions=5, seed=321, num_sources=2,
        peaks_per_repetition=3, spatial_subarray_size=3, frequency_subarray_size=5,
        diagonal_loading=0.01, minimum_relative_height=0.1, minimum_separation_bins=(2, 2),
    )
    expected = estimate_music_peak_samples(observation, **parameters)
    computer = MusicComputer(ComputeSettings(batch_size=batch_size, angle_chunk_size=5))
    actual = estimate_batched_peak_samples(computer, observation, batch_size=batch_size, **parameters)
    np.testing.assert_array_equal(actual.aoa_rad, expected.aoa_rad)
    np.testing.assert_array_equal(actual.delay_s, expected.delay_s)
    np.testing.assert_allclose(actual.spectrum_value, expected.spectrum_value, rtol=1e-9)


@pytest.mark.parametrize("field,value", [
    ("backend", "auto"), ("device_id", -1), ("batch_size", 0),
    ("angle_chunk_size", True), ("ue_truth", [1, 2]),
])
def test_compute_config_rejects_invalid_settings_and_truth(field, value):
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config["compute"][field] = value
    with pytest.raises(ValueError):
        validate_localization_config(config)


def test_legacy_localization_config_without_compute_is_still_valid():
    config = localization_config_view(deepcopy(DEFAULT_CONFIG))
    config.pop("compute")
    validate_localization_config(config)


@pytest.mark.parametrize("variant", [None, "llvm_ad_mono_polarized", "cuda_ad_mono_polarized"])
def test_sionna_actual_device_check(monkeypatch, variant):
    monkeypatch.setattr("time_bias_localization.sionna_generation.importlib.import_module",
                        lambda name: SimpleNamespace(variant=lambda: variant))
    report = _sionna_runtime_info(require_cuda=False)
    assert report["mitsuba_variant"] == variant
    if variant and variant.startswith("cuda_"):
        assert _sionna_runtime_info(require_cuda=True)["uses_cuda"]
    else:
        with pytest.raises(RuntimeError, match="不自动改用 CPU"):
            _sionna_runtime_info(require_cuda=True)
