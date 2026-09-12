"""在同一份观测 CSI 的 MUSIC 峰附近采样，构造有来源记录的候选观测。

谱值只定义候选搜索的提议分布，不是路径的后验概率或置信区间。
每个峰独立分配样本；采样权重不再乘谱值，也不制造独立测量。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .candidates import PathObservationSample, local_to_global_aoa
from .compute import PreparedMusic
from .signal import MusicPeak2D, _as_real_vector
from .timing import stage


@dataclass(frozen=True)
class SpectrumSamplingResult:
    samples: list[PathObservationSample]
    records: list[dict[str, Any]]
    regions: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    refined_peaks: list[MusicPeak2D] = field(default_factory=list)
    refined_peak_source_indices: list[int] = field(default_factory=list)


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
    minimum_angle_separation_rad: float = 0.0,
    minimum_delay_separation_s: float = 0.0,
) -> SpectrumSamplingResult:
    """在一份局部细谱上确定正式峰、构造单元概率并生成连续样本。

    ``nominal_peaks`` 是仅用于确定搜索区域的粗峰。返回的正式峰位于各区域
    细谱的最大节点；MC 单元质量使用同一节点谱的四角均值，不另算中心谱。
    单元内采用均匀连续坐标，精确连续谱值仅作为诊断。细化后邻近重复峰按
    物理角度和时延间隔去重，保留较强峰及其原粗峰编号，不重排来源编号。
    角度是阵列局部角，输出 RT 输入时才转换为全局角。
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
    for name, value in (("minimum_angle_separation_rad", minimum_angle_separation_rad),
                        ("minimum_delay_separation_s", minimum_delay_separation_s)):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} 必须为有限非负数")
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
        "cell_spectrum = mean of four corners of the same fine-grid spectrum; "
        "cell mass = (1-uniform_mixture) * normalize(cell_spectrum^spectrum_power "
        "* cell_area) + uniform_mixture * normalize(cell_area); "
        "uniform continuous coordinates within selected cell. "
        "Exact spectrum values at samples are diagnostics only. "
        "Every returned sample has weight 1; samples are alternatives of one observation."
    )
    # 先完成细谱找峰和去重，之后才为保留的观测分配 MC 样本。
    local_regions: list[dict[str, Any]] = []
    for peak_index, coarse_peak in enumerate(nominal_peaks):
        if not (np.isfinite(coarse_peak.aoa_rad) and np.isfinite(coarse_peak.delay_s)):
            raise ValueError("原始谱峰角度与时延必须为有限数")
        if not (angles[0] <= coarse_peak.aoa_rad <= angles[-1]
                and delays[0] <= coarse_peak.delay_s <= delays[-1]):
            raise ValueError("原始谱峰必须位于原始 MUSIC 搜索范围内")
        angle_bounds = (max(float(angles[0]), coarse_peak.aoa_rad - angle_width),
                        min(float(angles[-1]), coarse_peak.aoa_rad + angle_width))
        delay_bounds = (max(float(delays[0]), coarse_peak.delay_s - delay_width),
                        min(float(delays[-1]), coarse_peak.delay_s + delay_width))
        angle_edges = np.linspace(*angle_bounds, grid_size)
        delay_edges = np.linspace(*delay_bounds, grid_size)
        # 极小区域被浮点精度压缩时明确失败，不能悄悄退化成重复网格点。
        if np.any(np.diff(angle_edges) <= 0) or np.any(np.diff(delay_edges) <= 0):
            raise ValueError("局部采样区域过小，无法在浮点精度内形成连续网格")
        with stage('T05_fine_spectrum'):
            local_spectrum = np.asarray(prepared.spectrum(
                aoa_grid_rad=angle_edges, delay_grid_s=delay_edges,
            ), dtype=np.float64)
        with stage('T06_fine_peaks'):
            if (local_spectrum.shape != (grid_size, grid_size)
                    or not np.all(np.isfinite(local_spectrum)) or np.any(local_spectrum <= 0)):
                raise ValueError("局部 MUSIC 谱必须为有限正数")
            angle_index, delay_index = np.unravel_index(np.argmax(local_spectrum), local_spectrum.shape)
            peak = MusicPeak2D(
                aoa_rad=float(angle_edges[angle_index]), delay_s=float(delay_edges[delay_index]),
                spectrum_value=float(local_spectrum[angle_index, delay_index]),
                aoa_index=int(angle_index), delay_index=int(delay_index),
            )
            boundary_axes = []
            if angle_index in (0, grid_size - 1):
                boundary_axes.append("aoa")
            if delay_index in (0, grid_size - 1):
                boundary_axes.append("delay")
            search_boundary_axes = []
            if "aoa" in boundary_axes and peak.aoa_rad in (angles[0], angles[-1]):
                search_boundary_axes.append("aoa")
            if "delay" in boundary_axes and peak.delay_s in (delays[0], delays[-1]):
                search_boundary_axes.append("delay")
            local_regions.append({
                "source_index": peak_index, "peak": peak, "coarse_peak": coarse_peak,
                "angles": angle_edges, "delays": delay_edges, "spectrum": local_spectrum,
                "boundary_axes": boundary_axes, "search_boundary_axes": search_boundary_axes,
                "bounds_clipped": bool(
                    angle_bounds[0] > coarse_peak.aoa_rad - angle_width
                    or angle_bounds[1] < coarse_peak.aoa_rad + angle_width
                    or delay_bounds[0] > coarse_peak.delay_s - delay_width
                    or delay_bounds[1] < coarse_peak.delay_s + delay_width
                ),
            })
    with stage('T06_fine_peaks'):
        selected_regions: list[dict[str, Any]] = []
        suppressed_peaks: list[dict[str, Any]] = []
        for region in sorted(local_regions, key=lambda item: (-item["peak"].spectrum_value, item["source_index"])):
            peak = region["peak"]
            duplicate = next((other for other in selected_regions
                              if abs(peak.aoa_rad - other["peak"].aoa_rad) <= minimum_angle_separation_rad
                              and abs(peak.delay_s - other["peak"].delay_s) <= minimum_delay_separation_s), None)
            if duplicate is None:
                selected_regions.append(region)
            else:
                coarse_peak = region["coarse_peak"]
                suppressed_peaks.append({
                    "source_peak_index": region["source_index"],
                    "observation_id": f"music_path_{region['source_index']:02d}",
                    "kept_source_peak_index": duplicate["source_index"],
                    "kept_observation_id": f"music_path_{duplicate['source_index']:02d}",
                    "coarse_aoa_local_rad": float(coarse_peak.aoa_rad),
                    "coarse_delay_s": float(coarse_peak.delay_s),
                    "refined_aoa_local_rad": peak.aoa_rad, "refined_delay_s": peak.delay_s,
                    "refined_spectrum_value": peak.spectrum_value,
                    "reason": "refined_peak_within_minimum_physical_separation",
                })
        selected_regions.sort(key=lambda item: item["source_index"])
    with stage('T07_feature_sampling'):
        for region in selected_regions:
            peak_index = region["source_index"]
            peak, coarse_peak = region["peak"], region["coarse_peak"]
            angle_edges, delay_edges = region["angles"], region["delays"]
            local_spectrum = region["spectrum"]
            observation_id = f"music_path_{peak_index:02d}"
            # 除以 4 后相加，避免有限但很大的节点谱值在求和时溢出。
            corner_spectrum = (local_spectrum[:-1, :-1] / 4 + local_spectrum[1:, :-1] / 4
                               + local_spectrum[:-1, 1:] / 4 + local_spectrum[1:, 1:] / 4)
            area = np.diff(angle_edges)[:, None] * np.diff(delay_edges)[None, :]
            # 先除以最大值，避免大谱值在取幂时溢出；仅改变公共归一化常数。
            spectral_mass = (corner_spectrum / np.max(corner_spectrum)) ** options["spectrum_power"] * area
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
                    "source_peak_index": peak_index, "peak_grid": "local_fine_spectrum",
                    "sampling_kind": kind, "aoa_local_rad": float(angle),
                    "aoa_global_rad": global_angle, "delay_s": float(delay),
                    "spectrum_value": float(value), "weight": 1.0,
                    "cell_aoa_index": cell[0], "cell_delay_index": cell[1],
                })
            regions.append({
                "observation_id": observation_id, "peak_index": peak_index,
                "source_peak_index": peak_index,
                "coarse_aoa_local_rad": float(coarse_peak.aoa_rad),
                "coarse_delay_s": float(coarse_peak.delay_s),
                "coarse_aoa_grid_index": int(coarse_peak.aoa_index),
                "coarse_delay_grid_index": int(coarse_peak.delay_index),
                "coarse_spectrum_value": float(coarse_peak.spectrum_value),
                "nominal_aoa_local_rad": float(peak.aoa_rad),
                "nominal_delay_s": float(peak.delay_s),
                "nominal_spectrum_value": peak.spectrum_value,
                "refined_aoa_grid_index": peak.aoa_index,
                "refined_delay_grid_index": peak.delay_index,
                "peak_grid": "local_fine_spectrum",
                "peak_on_window_edge": bool(region["boundary_axes"]),
                "refined_peak_at_local_boundary": bool(region["boundary_axes"]),
                "refined_peak_boundary_axes": region["boundary_axes"],
                "refined_peak_search_boundary_axes": region["search_boundary_axes"],
                "unresolved_window_peak": bool(set(region["boundary_axes"]) - set(region["search_boundary_axes"])),
                "boundary_policy": "retain_and_flag_window_maximum; no_unreported_extrapolation",
                "aoa_grid_rad": angle_edges.tolist(), "delay_grid_s": delay_edges.tolist(),
                "spectrum": local_spectrum.tolist(), "cell_spectrum": corner_spectrum.tolist(),
                "cell_spectrum_rule": "arithmetic_mean_of_four_fine_grid_corners",
                "cell_probabilities": probabilities.tolist(),
                "sampling_distribution": description,
                "spectrum_power": options["spectrum_power"], "uniform_mixture": mixture,
                "bounds_clipped": region["bounds_clipped"],
            })
    return SpectrumSamplingResult(
        samples=samples, records=records, regions=regions,
        refined_peaks=[region["peak"] for region in selected_regions],
        refined_peak_source_indices=[region["source_index"] for region in selected_regions],
        diagnostics={
            "method": "local_fine_music_peak_and_monte_carlo",
            "seed": int(seed), "settings": options,
            "samples_per_peak": count, "num_peaks": len(selected_regions),
            "coarse_peak_count": len(nominal_peaks),
            "refined_peak_source_indices": [region["source_index"] for region in selected_regions],
            "suppressed_refined_peaks": suppressed_peaks,
            "minimum_angle_separation_rad": float(minimum_angle_separation_rad),
            "minimum_delay_separation_s": float(minimum_delay_separation_s),
            "total_samples": len(samples), "monte_carlo_samples": len(selected_regions) * count,
            "nominal_samples": len(selected_regions) if options["include_nominal"] else 0,
            "peak_and_proposal_share_fine_spectrum": True,
            "exact_sample_spectrum_used_for_proposal": False,
            "unresolved_window_peak_source_indices": [region["peak_index"] for region in regions
                                                       if region["unresolved_window_peak"]],
            "added_csi_noise": False, "proposal_is_calibrated_probability": False,
            "sample_weight_rule": "one; never multiply spectrum values again",
            "prepared_music": prepared.metadata(),
        },
    )
