"""按 MUSIC 峰的完整天线×子载波响应筛选实验输入，不读取真值或定位误差。"""
from itertools import combinations
import numpy as np
from .constants import SPEED_OF_LIGHT_M_S


def screen_music_observation(peaks, *, num_antennas, antenna_spacing_m,
                             carrier_frequency_hz, frequencies_hz, settings):
    enabled = settings.get("enabled", False)
    threshold = float(settings.get("max_response_correlation", 0.995))
    if not isinstance(enabled, bool) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("观测筛选需要布尔 enabled 和 (0, 1] 内的相似度门槛")
    report = {"enabled": enabled, "excluded": False, "threshold": threshold,
              "peak_count": len(peaks), "excluded_pairs": [],
              "uses_truth": False, "uses_localization_error": False,
              "scope": "observed_music_response_separability_not_true_path_count",
              "metric": "absolute_normalized_inner_product_full_antenna_subcarrier_response"}
    if not enabled:
        return report
    frequencies = np.asarray(frequencies_hz, dtype=float)
    if frequencies.ndim != 1 or not frequencies.size or not np.all(np.isfinite(frequencies)):
        raise ValueError("子载波频率必须为非空有限一维数组")
    n = np.arange(num_antennas)
    wavelength = SPEED_OF_LIGHT_M_S / carrier_frequency_hz
    maximum = 0.0
    for i, j in combinations(range(len(peaks)), 2):
        first, second = peaks[i], peaks[j]
        angle = abs(np.mean(np.exp(2j * np.pi * n * antenna_spacing_m / wavelength
                                  * (np.sin(first.aoa_rad) - np.sin(second.aoa_rad)))))
        delay = abs(np.mean(np.exp(-2j * np.pi * (frequencies - frequencies[0])
                                  * (first.delay_s - second.delay_s))))
        correlation = float(np.clip(angle * delay, 0, 1))
        maximum = max(maximum, correlation)
        if correlation >= threshold:
            report["excluded_pairs"].append({"first_peak_index": i, "second_peak_index": j,
                "response_correlation": correlation, "angle_response_correlation": float(angle),
                "delay_response_correlation": float(delay),
                "angles_deg": np.rad2deg([first.aoa_rad, second.aoa_rad]).tolist(),
                "delays_ns": [first.delay_s * 1e9, second.delay_s * 1e9]})
    report.update(excluded=bool(report["excluded_pairs"]), maximum_response_correlation=maximum)
    return report
