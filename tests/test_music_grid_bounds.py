"""MUSIC 粗网格不得越过公开的角度、时延搜索窗口。"""

import numpy as np

from time_bias_localization.pipeline import _make_grids


def _config(**changes):
    return {
        "angle_min_deg": -89.0,
        "angle_max_deg": 90.0,
        "angle_step_deg": 4.0,
        "delay_min_s": 0.0,
        "delay_max_s": 159e-9,
        "delay_step_s": 4e-9,
        **changes,
    }


def test_nondivisible_steps_append_exact_upper_bounds_without_overshoot():
    config = _config()
    angles, delays = _make_grids(config)
    expected_degrees = np.r_[np.arange(-89.0, 90.0, 4.0), 90.0]
    expected_delays = np.r_[np.arange(0.0, 159e-9, 4e-9), 159e-9]
    np.testing.assert_array_equal(angles, np.deg2rad(expected_degrees))
    np.testing.assert_array_equal(delays, expected_delays)
    assert angles[0] == np.deg2rad(config["angle_min_deg"])
    assert angles[-1] == np.pi / 2
    assert delays[0] == config["delay_min_s"]
    assert delays[-1] == config["delay_max_s"]
    assert np.all(np.diff(angles) > 0)
    assert np.all(np.diff(delays) > 0)
    assert np.max(angles) <= np.pi / 2
    assert np.max(delays) <= config["delay_max_s"]


def test_step_larger_than_window_still_produces_both_exact_endpoints():
    config = _config(
        angle_min_deg=1.0, angle_max_deg=2.0, angle_step_deg=5.0,
        delay_min_s=10e-9, delay_max_s=11e-9, delay_step_s=50e-9,
    )
    angles, delays = _make_grids(config)
    np.testing.assert_array_equal(angles, np.deg2rad([1.0, 2.0]))
    np.testing.assert_array_equal(delays, [10e-9, 11e-9])


def test_divisible_step_does_not_add_near_duplicate_endpoint():
    config = _config(
        angle_min_deg=-89.0, angle_max_deg=89.0, angle_step_deg=1.0,
        delay_max_s=160e-9, delay_step_s=1e-9,
    )
    angles, delays = _make_grids(config)
    assert angles.size == 179
    assert delays.size == 161
    assert angles[-1] == np.deg2rad(89.0)
    assert delays[-1] == 160e-9
    np.testing.assert_allclose(np.diff(angles), np.deg2rad(1.0), rtol=1e-12)
    np.testing.assert_allclose(np.diff(delays), 1e-9, rtol=1e-12)


def test_nonzero_delay_origin_keeps_last_short_cell_inside_window():
    config = _config(delay_min_s=12e-9, delay_max_s=25e-9, delay_step_s=4e-9)
    _, delays = _make_grids(config)
    np.testing.assert_allclose(delays, np.array([12, 16, 20, 24, 25]) * 1e-9,
                               rtol=0, atol=1e-22)
    assert delays[0] == config["delay_min_s"]
    assert delays[-1] == config["delay_max_s"]
    assert np.all(delays <= config["delay_max_s"])
