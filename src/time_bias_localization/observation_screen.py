"""按 MUSIC 峰的完整天线×子载波响应筛选实验输入，不读取真值或定位误差。"""
from itertools import combinations
import math
import numpy as np
from .constants import SPEED_OF_LIGHT_M_S


AMBIGUITY_POLICIES = ("exclude_sample", "enumerate_branches")


def _wrapped_spread_deg(values_deg):
    best = 0.0
    for left in range(len(values_deg)):
        for right in range(left + 1, len(values_deg)):
            best = max(best, abs((values_deg[left] - values_deg[right] + 180.0) % 360.0 - 180.0))
    return best


def ambiguity_groups(excluded_pairs, peak_count):
    """把触发对按共享峰合并成歧义组；每组是一份角度不可分辨的测量。

    同一组内的峰互为角度候选，任何时刻只应有一个参与求解；组与组之间独立，
    因此可用组的笛卡尔积枚举角度分支。不读取真值，也不改变触发判据本身。
    """
    if isinstance(peak_count, bool) or not isinstance(peak_count, int) or peak_count < 0:
        raise ValueError("峰数量必须是非负整数")
    parent = list(range(peak_count))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    involved = set()
    for pair in excluded_pairs:
        members = (int(pair["first_peak_index"]), int(pair["second_peak_index"]))
        if any(not 0 <= index < peak_count for index in members):
            raise ValueError("歧义对引用了不存在的峰编号")
        involved.update(members)
        left, right = root(members[0]), root(members[1])
        if left != right:
            parent[left] = right
    grouped = {}
    for index in sorted(involved):
        grouped.setdefault(root(index), []).append(index)
    return sorted(grouped.values(), key=lambda members: members[0])


def _group_spreads(groups, peaks):
    return [{"peak_indices": list(members),
             "angle_spread_deg": _wrapped_spread_deg(
                 [math.degrees(peaks[index].aoa_rad) for index in members]),
             "delay_spread_ns": float(np.ptp([peaks[index].delay_s for index in members]) * 1e9)}
            for members in groups]


def screen_music_observation(peaks, *, num_antennas, antenna_spacing_m,
                             carrier_frequency_hz, frequencies_hz, settings):
    enabled = settings.get("enabled", False)
    threshold = float(settings.get("max_response_correlation", 0.995))
    policy = settings.get("ambiguity_policy", "exclude_sample")
    if not isinstance(enabled, bool) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("观测筛选需要布尔 enabled 和 (0, 1] 内的相似度门槛")
    if policy not in AMBIGUITY_POLICIES:
        raise ValueError("observation_screen.ambiguity_policy 只能为 "
                         + "、".join(AMBIGUITY_POLICIES))
    report = {"enabled": enabled, "excluded": False, "threshold": threshold,
              "peak_count": len(peaks), "excluded_pairs": [],
              "ambiguity_policy": policy, "ambiguity_groups": [], "branch_count": 1,
              "ambiguity_group_spreads": [],
              "excluded_meaning": "at_least_one_pair_triggered_not_sample_verdict",
              "scope": "observed_music_response_separability_not_true_path_count",
              "metric": "absolute_normalized_inner_product_full_antenna_subcarrier_response",
              "uses_truth": False, "uses_localization_error": False}
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
    groups = ambiguity_groups(report["excluded_pairs"], len(peaks))
    report.update(ambiguity_groups=groups,
                  branch_count=math.prod(len(members) for members in groups) if groups else 1,
                  ambiguity_group_spreads=_group_spreads(groups, peaks))
    return report
