"""在同一份观测 CSI 的 MUSIC 峰附近采样，构造有来源记录的候选观测。

谱值只定义候选搜索的提议分布，不是路径的后验概率或置信区间。
每个峰独立分配样本；采样权重不再乘谱值，也不制造独立测量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .candidates import PathObservationSample, local_to_global_aoa
from .compute import PreparedMusic
from .signal import MusicPeak2D, _as_real_vector


@dataclass(frozen=True)
class SpectrumSamplingResult:
    samples: list[PathObservationSample]
    records: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    diagnostics: dict[str, Any]


DEFAULT_SPECTRUM_SAMPLING: dict[str, Any] = {
    "samples_per_peak": 128,
    "aoa_half_width_grid_steps": 1.5,
    "delay_half_width_grid_steps": 1.5,
    "local_grid_points_per_axis": 25,
    "spectrum_power": 1.0,
    "uniform_mixture": 0.1,
    "include_nominal": True,
}


def _sampling_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - set(DEFAULT_SPECTRUM_SAMPLING)
    if unknown:
        raise ValueError(f"未知的谱面采样参数: {sorted(unknown)}")
    values = {**DEFAULT_SPECTRUM_SAMPLING, **settings}
    for key, minimum in (("samples_per_peak", 1), ("local_grid_points_per_axis", 2)):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key} 必须为不小于 {minimum} 的整数")
        values[key] = int(value)
    for key in ("aoa_half_width_grid_steps", "delay_half_width_grid_steps", "spectrum_power"):
        value = values[key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须为有限正数")
        values[key] = float(value)
    mixture = values["uniform_mixture"]
    if isinstance(mixture, bool) or not np.isfinite(mixture) or not 0 <= mixture <= 1:
        raise ValueError("uniform_mixture 必须位于 [0, 1]")
    values["uniform_mixture"] = float(mixture)
    if not isinstance(values["include_nominal"], bool):
        raise ValueError("include_nominal 必须为布尔值")
    return values


def sample_music_spectrum(
    prepared: PreparedMusic,
    nominal_peaks: Sequence[MusicPeak2D],
    *,
    aoa_grid_rad: Sequence[float] | np.ndarray,
    delay_grid_s: Sequence[float] | np.ndarray,
    bs_boresight_rad: float,
    settings: Mapping[str, Any],
    seed: int,
) -> SpectrumSamplingResult:
    """精确计算局部谱，按单元概率抽样，再在单元内生成连续坐标。

    单元质量与 ``中心谱值 ** spectrum_power * 单元面积`` 成正比，并与
    覆盖整个局部区域的均匀分布混合。该分布仅用于候选搜索；原始峰作为
    单独参考样本保留。角度是阵列局部角，输出 RT 输入时才转换为全局角。
    """
    options = _sampling_settings(settings)
    angles = _as_real_vector("aoa_grid_rad", aoa_grid_rad)
    delays = _as_real_vector("delay_grid_s", delay_grid_s)
    for name, grid in (("aoa_grid_rad", angles), ("delay_grid_s", delays)):
        if grid.size < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError(f"{name} 必须严格递增且至少包含两个坐标")
    if angles[0] < -np.pi / 2 or angles[-1] > np.pi / 2:
        raise ValueError("局部 AOA 搜索范围必须位于 [-pi/2, pi/2]")
    if delays[0] < 0:
        raise ValueError("观测时延搜索范围必须非负")
    if not np.isfinite(bs_boresight_rad):
        raise ValueError("bs_boresight_rad 必须为有限数")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    rng = np.random.default_rng(seed)
    angle_width = float(np.median(np.diff(angles))) * options["aoa_half_width_grid_steps"]
    delay_width = float(np.median(np.diff(delays))) * options["delay_half_width_grid_steps"]
    samples: list[PathObservationSample] = []
    records: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    count = options["samples_per_peak"]
    grid_size = options["local_grid_points_per_axis"]
    description = (
        "Per-peak proposal only, not a posterior or confidence distribution: "
        "cell mass = (1-uniform_mixture) * normalize(center_spectrum^spectrum_power "
        "* cell_area) + uniform_mixture * normalize(cell_area); "
        "uniform continuous coordinates within selected cell. "
        "Every returned sample has weight 1; samples are alternatives of one observation."
    )
    for peak_index, peak in enumerate(nominal_peaks):
        if not (np.isfinite(peak.aoa_rad) and np.isfinite(peak.delay_s)):
            raise ValueError("原始谱峰角度与时延必须为有限数")
        if not (angles[0] <= peak.aoa_rad <= angles[-1]
                and delays[0] <= peak.delay_s <= delays[-1]):
            raise ValueError("原始谱峰必须位于原始 MUSIC 搜索范围内")
        observation_id = f"music_path_{peak_index:02d}"
        angle_bounds = (max(float(angles[0]), peak.aoa_rad - angle_width),
                        min(float(angles[-1]), peak.aoa_rad + angle_width))
        delay_bounds = (max(float(delays[0]), peak.delay_s - delay_width),
                        min(float(delays[-1]), peak.delay_s + delay_width))
        angle_edges = np.linspace(*angle_bounds, grid_size)
        delay_edges = np.linspace(*delay_bounds, grid_size)
        # 极小区域被浮点精度压缩时明确失败，不能悄悄退化成重复网格点。
        if np.any(np.diff(angle_edges) <= 0) or np.any(np.diff(delay_edges) <= 0):
            raise ValueError("局部采样区域过小，无法在浮点精度内形成连续网格")
        angle_centers = (angle_edges[:-1] + angle_edges[1:]) / 2
        delay_centers = (delay_edges[:-1] + delay_edges[1:]) / 2
        local_spectrum = prepared.spectrum(aoa_grid_rad=angle_edges, delay_grid_s=delay_edges)
        center_spectrum = prepared.spectrum(
            aoa_grid_rad=angle_centers, delay_grid_s=delay_centers,
        )
        if (not np.all(np.isfinite(center_spectrum)) or np.any(center_spectrum <= 0)):
            raise ValueError("局部 MUSIC 谱必须为有限正数")
        area = np.diff(angle_edges)[:, None] * np.diff(delay_edges)[None, :]
        # 先除以最大值，避免大谱值在取幂时溢出；仅改变公共归一化常数。
        spectral_mass = (center_spectrum / np.max(center_spectrum)) ** options["spectrum_power"] * area
        mixture = options["uniform_mixture"]
        probabilities = ((1 - mixture) * spectral_mass / spectral_mass.sum()
                         + mixture * area / area.sum())
        probabilities /= probabilities.sum()
        selected = rng.choice(probabilities.size, size=count, p=probabilities.ravel())
        angle_cells, delay_cells = np.unravel_index(selected, probabilities.shape)
        sampled_angles = angle_edges[angle_cells] + rng.random(count) * np.diff(angle_edges)[angle_cells]
        sampled_delays = delay_edges[delay_cells] + rng.random(count) * np.diff(delay_edges)[delay_cells]
        sample_kinds = ["local_spectrum_mc"] * count
        sample_ids = [f"{observation_id}:mc_{index:05d}" for index in range(count)]
        cells: list[tuple[int | None, int | None]] = list(zip(
            angle_cells.tolist(), delay_cells.tolist(), strict=True,
        ))
        if options["include_nominal"]:
            sampled_angles = np.r_[peak.aoa_rad, sampled_angles]
            sampled_delays = np.r_[peak.delay_s, sampled_delays]
            sample_kinds.insert(0, "nominal")
            sample_ids.insert(0, f"{observation_id}:nominal")
            cells.insert(0, (None, None))
        exact_values = prepared.values(aoa_rad=sampled_angles, delay_s=sampled_delays)
        for index, (angle, delay, value, kind, sample_id, cell) in enumerate(zip(
            sampled_angles, sampled_delays, exact_values, sample_kinds, sample_ids, cells,
            strict=True,
        )):
            global_angle = local_to_global_aoa(float(angle), bs_boresight_rad)
            sample = PathObservationSample(
                observation_id=observation_id, sample_id=sample_id,
                aoa_global_rad=global_angle, delay_s=float(delay), weight=1.0,
            )
            samples.append(sample)
            records.append({
                "observation_id": observation_id, "sample_id": sample_id,
                "peak_index": peak_index, "sample_index": index,
                "sampling_kind": kind, "aoa_local_rad": float(angle),
                "aoa_global_rad": global_angle, "delay_s": float(delay),
                "spectrum_value": float(value), "weight": 1.0,
                "cell_aoa_index": cell[0], "cell_delay_index": cell[1],
            })
        regions.append({
            "observation_id": observation_id, "peak_index": peak_index,
            "nominal_aoa_local_rad": float(peak.aoa_rad),
            "nominal_delay_s": float(peak.delay_s),
            "aoa_grid_rad": angle_edges.tolist(), "delay_grid_s": delay_edges.tolist(),
            "spectrum": local_spectrum.tolist(), "cell_spectrum": center_spectrum.tolist(),
            "cell_probabilities": probabilities.tolist(),
            "sampling_distribution": description,
            "spectrum_power": options["spectrum_power"], "uniform_mixture": mixture,
            "bounds_clipped": bool(
                angle_bounds[0] > peak.aoa_rad - angle_width
                or angle_bounds[1] < peak.aoa_rad + angle_width
                or delay_bounds[0] > peak.delay_s - delay_width
                or delay_bounds[1] < peak.delay_s + delay_width
            ),
        })
    return SpectrumSamplingResult(
        samples=samples, records=records, regions=regions,
        diagnostics={
            "method": "local_music_spectrum_monte_carlo",
            "seed": int(seed), "settings": options,
            "samples_per_peak": count, "num_peaks": len(nominal_peaks),
            "total_samples": len(samples), "monte_carlo_samples": len(nominal_peaks) * count,
            "nominal_samples": len(nominal_peaks) if options["include_nominal"] else 0,
            "added_csi_noise": False, "proposal_is_calibrated_probability": False,
            "sample_weight_rule": "one; never multiply spectrum values again",
            "prepared_music": prepared.metadata(),
        },
    )
