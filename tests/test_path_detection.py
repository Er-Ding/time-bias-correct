"""从真实观测行为验证停止、弱成分保留和二维整体门槛。"""
import numpy as np
import pytest
from time_bias_localization.path_detection import (
    _Search, detect_csi_paths, estimate_noise_from_csi, detection_settings,
)
from time_bias_localization.signal import synthesize_ula_csi


def setup():
    frequencies = np.arange(64) * 3.125e6
    return dict(subcarrier_frequencies_hz=frequencies, carrier_frequency_hz=3.5e9,
                antenna_spacing_m=None, aoa_grid_rad=np.linspace(-1.2, 1.2, 61),
                delay_grid_s=np.linspace(0, 280e-9, 141),
                settings=dict(enabled=True, max_paths=3, false_alarm_probability=.05,
                              calibration_trials=127, max_refine_evaluations=45))


def observation(angles, delays, powers, seed=2):
    args = setup()
    clean = synthesize_ula_csi(path_aoa_rad=angles, path_delay_s=delays,
                              path_coefficients=powers, num_bs_antennas=8,
                              subcarrier_frequencies_hz=args["subcarrier_frequencies_hz"],
                              carrier_frequency_hz=args["carrier_frequency_hz"])
    rng = np.random.default_rng(seed)
    return clean + .01 * (rng.normal(size=clean.shape) + 1j*rng.normal(size=clean.shape))/np.sqrt(2)


def test_czt_two_dimensional_search_equals_explicit_csi_dictionary():
    args = setup()
    search = _Search(8, args["subcarrier_frequencies_hz"], 3.5e9, 299792458/3.5e9/2,
                     args["aoa_grid_rad"], args["delay_grid_s"], "numpy")
    data = observation([.173], [123.456e-9], [1.])
    scores = search.correlations(data)
    for i,j in [(0,0), (60,140), (31,52), (41,123)]:
        atom = search.dictionary([[np.sin(search.angles[i]), search.delays[j]*search.band]])[:,0]
        assert scores[i,j] == pytest.approx(np.vdot(atom, data.ravel()), rel=1e-10, abs=1e-9)


def test_calibration_accounts_for_every_location_in_nested_2d_searches():
    args = setup()
    options = detection_settings(args["settings"])
    def search(angles, delays):
        return _Search(8,args["subcarrier_frequencies_hz"],3.5e9,299792458/3.5e9/2,angles,delays,"numpy")
    large = search(args["aoa_grid_rad"],args["delay_grid_s"])
    small = search(args["aoa_grid_rad"][20:23],args["delay_grid_s"][60:64])
    _, big_maxima, big_threshold, _ = large.calibrate(np.empty((0,2)),1,options,0)
    _, small_maxima, small_threshold, _ = small.calibrate(np.empty((0,2)),1,options,0)
    assert np.all(big_maxima >= small_maxima-1e-9)
    assert big_threshold > small_threshold


def test_single_off_grid_path_stops_after_one_component():
    result = detect_csi_paths(observation([.173], [123.456e-9], [1.]), **setup())
    assert len(result.peaks) == 1
    assert result.peaks[0].aoa_rad == pytest.approx(.173, abs=.002)
    assert result.peaks[0].delay_s == pytest.approx(123.456e-9, abs=.1e-9)
    assert result.diagnostics["steps"][-1]["decision"] == "stop_no_additional_component"


@pytest.mark.parametrize("factor", [1e-8,1e-5,1e3])
def test_rt_amplitude_scale_does_not_change_refinement_or_path_count(factor):
    data = observation([.173], [123.456e-9], [1.])
    expected = detect_csi_paths(data, **setup())
    scaled = detect_csi_paths(data*factor, **setup())
    assert len(scaled.peaks) == len(expected.peaks) == 1
    assert scaled.peaks[0].aoa_rad == pytest.approx(expected.peaks[0].aoa_rad,abs=1e-8)
    assert scaled.peaks[0].delay_s == pytest.approx(expected.peaks[0].delay_s,abs=1e-15)


def test_weak_component_is_retained_despite_large_power_difference():
    result = detect_csi_paths(observation([.173,-.612], [123.456e-9,217.321e-9], [1.,.035]), **setup())
    assert len(result.peaks) == 2
    assert min(abs(p.delay_s-217.321e-9) for p in result.peaks) < .5e-9
    assert min(abs(p.aoa_rad+.612) for p in result.peaks) < .01


def test_identical_csi_components_count_as_one_path():
    data = observation([.173,.173], [123.456e-9,123.456e-9], [1.,.2j])
    result = detect_csi_paths(data, **setup())
    assert len(result.peaks) == 1


def test_noise_only_and_noise_scale_invariance():
    rng = np.random.default_rng(47)
    data = rng.normal(size=(8,64)) + 1j*rng.normal(size=(8,64))
    assert .7*2 < estimate_noise_from_csi(data) < 1.3*2
    small = detect_csi_paths(data, **setup())
    large = detect_csi_paths(data*1000, **setup())
    assert not small.peaks and not large.peaks
    assert small.diagnostics["steps"][0]["whole_search_p_value"] == large.diagnostics["steps"][0]["whole_search_p_value"]


def test_noise_and_false_alarm_settings_cannot_include_oracle_inputs():
    with pytest.raises(ValueError, match="未定义"):
        detection_settings({"snr_db": 35})
    with pytest.raises(ValueError, match="太少"):
        detection_settings({"calibration_trials": 10})


def test_zero_csi_is_reported_as_unresolved_noise():
    result = detect_csi_paths(np.zeros((8,64), complex), **setup())
    assert not result.peaks
    assert result.diagnostics["stop_reason"] == "noise_level_unresolved"
