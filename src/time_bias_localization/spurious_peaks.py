"""剔除 MUSIC 谱在角度搜索边界产生的伪峰。

半波长 ULA 在端射附近阵列流形退化：大量不同的 (角度, 时延) 组合给出几乎
共线的导向矢量，MUSIC 噪声子空间投影接近零，伪谱沿角度边界形成一条脊。
沿这条脊，时延维仍反映真实路径，因此伪峰与某条真实路径同延、角度却钉在
网格边界，且谱值明显更低。

合成单路径实验（噪声为信号的 1%，信号秩为 1）复现了这一现象：真值 −86.6°
与 −89° 各自在 +89.000° 处产生伪峰，而 −40° 不产生。伪峰只依赖导向矢量
与 MUSIC 公式，不依赖信道模型，因此真实阵列同样会出现。

判据保守：三条同时成立才剔除，只删更弱的那个，不删边界上确实存在的强峰。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np

from .signal import MusicPeak2D


DEFAULT_SPURIOUS_PEAK_FILTER: dict = {
    "enabled": False,
    "edge_tolerance_deg": 1.0,
    "delay_tolerance_ns": 5.0,
    "maximum_spectrum_ratio": 0.5,
    "peak_margin": 2,
}


@dataclass(frozen=True)
class SpuriousPeakFilterResult:
    peaks: tuple[MusicPeak2D, ...]
    source_indices: tuple[int, ...]
    report: dict


def spurious_peak_filter_settings(settings: Mapping | None) -> dict:
    """校验并补全伪峰剔除参数；未给定时返回关闭状态。"""
    if settings is None:
        return dict(DEFAULT_SPURIOUS_PEAK_FILTER)
    if not isinstance(settings, Mapping):
        raise ValueError("spurious_peak_filter 必须是参数字典")
    unknown = set(settings) - set(DEFAULT_SPURIOUS_PEAK_FILTER)
    if unknown:
        raise ValueError(f"spurious_peak_filter 含未知参数：{sorted(unknown)}")
    values = {**DEFAULT_SPURIOUS_PEAK_FILTER, **settings}
    if not isinstance(values["enabled"], bool):
        raise ValueError("spurious_peak_filter.enabled 必须为布尔值")
    for name in ("edge_tolerance_deg", "delay_tolerance_ns"):
        value = values[name]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"spurious_peak_filter.{name} 必须是非负有限数")
        values[name] = float(value)
    margin = values["peak_margin"]
    if isinstance(margin, bool) or not isinstance(margin, int) or margin < 0:
        raise ValueError("spurious_peak_filter.peak_margin 必须是非负整数")
    values["peak_margin"] = int(margin)
    ratio = values["maximum_spectrum_ratio"]
    if isinstance(ratio, bool) or not math.isfinite(float(ratio)) or not 0 < float(ratio) <= 1:
        raise ValueError("spurious_peak_filter.maximum_spectrum_ratio 必须位于 (0, 1]")
    values["maximum_spectrum_ratio"] = float(ratio)
    return values


def filter_boundary_mirror_peaks(peaks: Sequence[MusicPeak2D],
                                 source_indices: Sequence[int],
                                 *, aoa_grid_rad, settings,
                                 keep_count: int | None = None) -> SpuriousPeakFilterResult:
    """删除角度边界上由更强同延峰产生的伪峰。

    三条判据同时成立才剔除：
    1. 角度与网格最大 |角度| 之差不超过 edge_tolerance_deg；
    2. 存在另一个峰，时延差不超过 delay_tolerance_ns；
    3. 该峰的谱值不超过对方谱值的 maximum_spectrum_ratio 倍。
    更强的那一方本身不因位于边界而被删；判定按谱值降序进行，避免互删。

    伪峰在峰提取阶段就已占用名额，删掉它不会自动找回被挤掉的真实路径。
    因此调用方应先按 ``peak_margin`` 留出余量再提取，过滤后由 ``keep_count``
    截断回原有名额；被挤掉但仍在余量内的真实路径借此补回。
    """
    peaks = tuple(peaks)
    source_indices = tuple(source_indices)
    if len(peaks) != len(source_indices):
        raise ValueError("峰与来源编号必须一一对应")
    values = spurious_peak_filter_settings(settings) if isinstance(settings, Mapping) else settings
    grid = np.asarray(aoa_grid_rad, float)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("角度网格必须是非空一维数组")
    if keep_count is not None and (isinstance(keep_count, bool)
                                   or not isinstance(keep_count, int) or keep_count < 1):
        raise ValueError("keep_count 必须是正整数或空")
    report = {"enabled": bool(values["enabled"]), "removed_count": 0, "removed": [],
              "boundary_angle_deg": float(np.rad2deg(np.max(np.abs(grid)))),
              "rule": "boundary_angle_and_coincident_delay_and_weaker_spectrum",
              "peak_margin": int(values["peak_margin"]),
              "input_count": len(peaks), "keep_count": keep_count,
              "truncated_count": 0, "truncated": [], "uses_truth": False}
    if not values["enabled"] or len(peaks) < 2:
        return SpuriousPeakFilterResult(peaks, source_indices, report)
    boundary = float(np.max(np.abs(grid)))
    edge_tolerance = math.radians(values["edge_tolerance_deg"])
    delay_tolerance = values["delay_tolerance_ns"] * 1e-9
    ratio = values["maximum_spectrum_ratio"]
    order = sorted(range(len(peaks)), key=lambda index: -peaks[index].spectrum_value)
    removed: set[int] = set()
    for index in order:
        peak = peaks[index]
        if abs(abs(peak.aoa_rad) - boundary) > edge_tolerance:
            continue
        for other in order:
            if other == index or other in removed:
                continue
            candidate = peaks[other]
            if abs(abs(candidate.aoa_rad) - boundary) <= edge_tolerance:
                continue
            if abs(candidate.delay_s - peak.delay_s) > delay_tolerance:
                continue
            if peak.spectrum_value > ratio * candidate.spectrum_value:
                continue
            removed.add(index)
            report["removed"].append({
                "removed_aoa_deg": float(np.rad2deg(peak.aoa_rad)),
                "removed_delay_ns": float(peak.delay_s * 1e9),
                "removed_spectrum_value": float(peak.spectrum_value),
                "explaining_aoa_deg": float(np.rad2deg(candidate.aoa_rad)),
                "explaining_delay_ns": float(candidate.delay_s * 1e9),
                "explaining_spectrum_value": float(candidate.spectrum_value),
                "spectrum_ratio": float(peak.spectrum_value / candidate.spectrum_value),
                "delay_difference_ns": float((peak.delay_s - candidate.delay_s) * 1e9),
            })
            break
    kept = [index for index in range(len(peaks)) if index not in removed]
    report["removed_count"] = len(removed)
    if keep_count is not None and len(kept) > keep_count:
        # 余量内保留下来的峰按谱值取前 keep_count 个，对齐来源编号。
        by_value = sorted(kept, key=lambda index: -peaks[index].spectrum_value)
        dropped = by_value[keep_count:]
        report["truncated"] = [{"aoa_deg": float(np.rad2deg(peaks[index].aoa_rad)),
                               "delay_ns": float(peaks[index].delay_s * 1e9),
                               "spectrum_value": float(peaks[index].spectrum_value)}
                              for index in dropped]
        report["truncated_count"] = len(dropped)
        kept = sorted(by_value[:keep_count])
    report["kept_count"] = len(kept)
    return SpuriousPeakFilterResult(tuple(peaks[index] for index in kept),
                                   tuple(source_indices[index] for index in kept), report)
