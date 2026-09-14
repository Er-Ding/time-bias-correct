"""从观测 CSI 逐条检验路径；二维全搜索的噪声最大值用于决定何时停止。

采用 NOMP 的“检出、连续细化、重新拟合、检查残差”思路。这里的门槛
由本接收阵列和二维搜索网格的模拟噪声最大值求出，不使用一维独立频点
公式。模拟只用于检验门槛，不向观测 CSI 加噪。白噪声假设、估计噪声
和已拟合路径条件下的校准边界均写入诊断，不能解释为真实路径概率。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import least_squares

from .constants import SPEED_OF_LIGHT_M_S
from .signal import MusicPeak2D
from .timing import stage


DEFAULT_PATH_DETECTION = {
    "enabled": False,
    "max_paths": 6,
    "false_alarm_probability": 0.01,
    "calibration_trials": 1023,
    "calibration_seed": 20260912,
    "calibration_batch_size": 8,
    "duplicate_correlation": 0.98,
    "max_refine_evaluations": 60,
}


def detection_settings(settings: Mapping[str, Any]) -> dict:
    if not isinstance(settings, Mapping) or set(settings) - set(DEFAULT_PATH_DETECTION):
        raise ValueError("music.path_detection 包含未定义字段或不是键值映射")
    result = {**DEFAULT_PATH_DETECTION, **settings}
    if not isinstance(result["enabled"], bool):
        raise ValueError("path_detection.enabled 必须为布尔值")
    for key in ("max_paths", "calibration_trials", "calibration_batch_size", "max_refine_evaluations"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"path_detection.{key} 必须为正整数")
    seed = result["calibration_seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("calibration_seed 必须为非负整数")
    for key in ("false_alarm_probability", "duplicate_correlation"):
        if isinstance(result[key], bool) or not np.isfinite(result[key]) or not 0 < result[key] < 1:
            raise ValueError(f"path_detection.{key} 必须位于 (0,1)")
    if 1 / (result["calibration_trials"] + 1) > result["false_alarm_probability"] / (result["max_paths"] + 1):
        raise ValueError("calibration_trials 太少，无法分辨分配给一次完整二维搜索的误检概率")
    return result


def estimate_noise_from_csi(csi: np.ndarray) -> float:
    """周期 Hann 窗 CIR 的稳健中位数噪声底，折算回每个复 CSI 元素的功率。

仅使用观测和窗口能量；稀疏路径占用较少时延单元。密集信道、硬件有色
噪声可能破坏这一假设，不能把估计值当作已知真实噪声功率。
"""
    data = np.asarray(csi, complex)
    if data.ndim not in (2, 3, 4) or data.shape[-1] < 8 or not np.all(np.isfinite(data)):
        raise ValueError("噪声估计需要有限复数 CSI 和至少 8 个子载波")
    n = data.shape[-1]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    power = np.abs(np.fft.ifft(data * window, axis=-1)) ** 2
    return float(np.median(power) * n * n / (np.log(2) * np.sum(window ** 2)))


@dataclass
class PathDetectionResult:
    peaks: list[MusicPeak2D]
    coefficients: np.ndarray
    diagnostics: dict[str, Any]


class _Search:
    """可分离的二维匹配搜索；Chirp-Z 支持任意均匀时延网格。"""
    def __init__(self, antennas, frequencies, carrier, spacing, angles, delays, backend):
        self.m, self.f = antennas, np.asarray(frequencies, float)
        self.angles, self.delays = np.asarray(angles, float), np.asarray(delays, float)
        self.n, self.size = self.f.size, antennas * self.f.size
        if self.n < 8 or antennas < 2 or self.f.ndim != 1 or not np.all(np.isfinite(self.f)):
            raise ValueError("路径检验需要至少 2 根天线和 8 个等间距子载波")
        df = np.diff(self.f)
        if np.any(df <= 0) or not np.allclose(df, df[0], rtol=1e-8, atol=1e-6):
            raise ValueError("路径检验的子载波频率必须等间距递增")
        for grid in (self.angles, self.delays):
            if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
                raise ValueError("二维搜索轴必须有限且严格递增")
        if not np.allclose(np.diff(self.delays), np.diff(self.delays)[0], rtol=1e-8, atol=1e-18):
            raise ValueError("时延搜索轴必须等间距")
        if self.angles[0] < -np.pi/2 or self.angles[-1] > np.pi/2 or self.delays[0] < 0:
            raise ValueError("角度或时延搜索范围不合法")
        if not np.isfinite(carrier) or carrier <= 0 or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("载频和天线间距必须为有限正数")
        if backend not in ("numpy", "cuda"):
            raise ValueError("路径检验后端必须为 numpy 或 cuda")
        self.xp = np
        if backend == "cuda":
            import cupy
            self.xp = cupy
        xp = self.xp
        self.spatial = 2 * np.pi * spacing * carrier / SPEED_OF_LIGHT_M_S * np.arange(antennas)
        self.band = float(df[0] * self.n)
        self.relative_f = (self.f - self.f[0]) / self.band
        self.angle_matrix = xp.asarray(np.exp(1j * self.spatial[:, None] * np.sin(self.angles)))
        length = self.delays.size
        self.nfft = next_fast_len(self.n + length - 1)
        k = np.arange(max(self.n, length), dtype=float)
        chirp = np.exp(1j * np.pi * df[0] * np.diff(self.delays)[0] * k ** 2)
        self.start = xp.asarray(np.exp(2j * np.pi * df[0] * self.delays[0] * k[:self.n]) * chirp[:self.n])
        self.kernel = xp.asarray(np.fft.fft(1 / np.r_[chirp[self.n-1:0:-1], chirp[:length]], self.nfft))
        self.end = xp.asarray(chirp[:length])

    def cpu(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def correlations(self, data):
        xp = self.xp
        array = xp.asarray(data)
        transformed = xp.fft.ifft(xp.fft.fft(array * self.start, self.nfft, axis=-1) * self.kernel, axis=-1)
        transformed = transformed[..., self.n-1:self.n-1+self.delays.size] * self.end
        return xp.einsum("ma,...md->...ad", self.angle_matrix.conj(), transformed, optimize=True)

    def dictionary(self, parameters):
        parameters = np.asarray(parameters).reshape(-1, 2)
        return np.stack([np.exp(1j * self.spatial[:, None] * u - 2j * np.pi * self.relative_f[None, :] * t).ravel()
                         for u, t in parameters], axis=1) if len(parameters) else np.empty((self.size, 0), complex)

    def fit(self, data, parameters, evaluations):
        parameters = np.asarray(parameters, float).reshape(-1, 2)
        y = data.reshape(data.shape[0], -1).T
        if not len(parameters):
            return parameters, np.empty((0, data.shape[0]), complex), data.copy(), 0
        lower = np.tile([np.sin(self.angles[0]), self.delays[0] * self.band], len(parameters))
        upper = np.tile([np.sin(self.angles[-1]), self.delays[-1] * self.band], len(parameters))
        scale = max(float(np.linalg.norm(y)), np.finfo(float).tiny)
        def residual(x):
            dictionary = self.dictionary(x)
            coefficients = np.linalg.lstsq(dictionary, y, rcond=1e-10)[0]
            difference = y - dictionary @ coefficients
            # RT 接收系数可以很小。归一化仅用于优化收敛判断，避免相同
            # 信噪比的数据因绝对幅度小而在粗网格上提前停止；输出保留原单位。
            return np.r_[difference.real.ravel(), difference.imag.ravel()] / scale
        fitted = least_squares(residual, np.clip(parameters.ravel(), lower, upper),
                               bounds=(lower, upper), max_nfev=evaluations,
                               ftol=1e-10, xtol=1e-11, gtol=1e-9, x_scale="jac")
        parameters = fitted.x.reshape(-1, 2)
        dictionary = self.dictionary(parameters)
        coefficients = np.linalg.lstsq(dictionary, y, rcond=1e-10)[0]
        residual_data = (y - dictionary @ coefficients).T.reshape(data.shape)
        return parameters, coefficients, residual_data, int(fitted.nfev)

    def projection(self, parameters):
        dictionary = self.dictionary(parameters)
        q = np.linalg.qr(dictionary, mode="reduced")[0]
        if dictionary.shape[1] and np.linalg.matrix_rank(dictionary) != dictionary.shape[1]:
            raise ValueError("已拟合 CSI 成分线性相关，无法继续进行残差检验")
        denominator = self.xp.full((self.angles.size, self.delays.size), float(self.size))
        if q.shape[1]:
            correlations = self.correlations(q.T.reshape(-1, self.m, self.n))
            denominator -= self.xp.sum(self.xp.abs(correlations) ** 2, axis=0)
        # 完全重复的 CSI 不能成为新观测；近似重复仍须比较删减后的重新拟合。
        denominator = self.xp.where(denominator > self.size * 1e-8, denominator, self.xp.inf)
        return q, denominator

    def statistic(self, residual, denominator):
        return self.xp.sum(self.xp.abs(self.correlations(residual)) ** 2, axis=-3) / denominator

    def calibrate(self, parameters, snapshots, options, step):
        q, denominator = self.projection(parameters)
        rng = np.random.default_rng(np.random.SeedSequence([options["calibration_seed"], step]))
        maxima = []
        xp = self.xp
        q_device = xp.asarray(q)
        n = self.n
        window = xp.asarray(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))
        for start in range(0, options["calibration_trials"], options["calibration_batch_size"]):
            batch = min(options["calibration_batch_size"], options["calibration_trials"] - start)
            # 每个重复使用独立的纯噪声；与观测 CSI 分开存储和处理。
            noise = xp.asarray(rng.normal(size=(batch, snapshots, self.m, n, 2)))
            noise = (noise[..., 0] + 1j * noise[..., 1]) / np.sqrt(2)
            cir_power = xp.abs(xp.fft.ifft(noise * window, axis=-1)) ** 2
            estimated = xp.median(cir_power.reshape(batch, -1), axis=1) * n*n / (np.log(2) * xp.sum(window ** 2))
            if q.shape[1]:
                flat = noise.reshape(batch, snapshots, -1)
                noise = (flat - (flat @ q_device.conj()) @ q_device.T).reshape(noise.shape)
            scores = self.statistic(noise, denominator)
            maxima.extend(self.cpu(xp.max(scores, axis=(-2, -1)) / estimated).tolist())
        maxima = np.asarray(maxima)
        alpha = options["false_alarm_probability"] / (options["max_paths"] + 1)
        order = int(np.ceil((len(maxima) + 1) * (1-alpha))) - 1
        threshold = float(np.sort(maxima)[order])
        return denominator, maxima, threshold, alpha


def detect_csi_paths(csi, *, backend="numpy", device_id=0, **kwargs) -> PathDetectionResult:
    if backend == "cuda":
        import cupy
        with cupy.cuda.Device(device_id):
            return _detect_csi_paths(csi, backend=backend, **kwargs)
    return _detect_csi_paths(csi, backend=backend, **kwargs)


def _detect_csi_paths(csi, *, subcarrier_frequencies_hz, carrier_frequency_hz,
                     antenna_spacing_m, aoa_grid_rad, delay_grid_s, settings,
                     backend="numpy", proposal_peaks=None) -> PathDetectionResult:
    options = detection_settings(settings)
    data = np.asarray(csi, complex)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3 or not data.size or not np.all(np.isfinite(data)):
        raise ValueError("路径检验 CSI 必须为有限 (天线,子载波) 或 (快照,天线,子载波) 数组")
    if data.shape[-1] != len(subcarrier_frequencies_hz):
        raise ValueError("CSI 子载波数量与频率轴不一致")
    spacing = SPEED_OF_LIGHT_M_S / carrier_frequency_hz / 2 if antenna_spacing_m is None else antenna_spacing_m
    search = _Search(data.shape[-2], subcarrier_frequencies_hz, carrier_frequency_hz,
                     spacing, aoa_grid_rad, delay_grid_s, backend)
    remaining_proposals = None
    if proposal_peaks is not None:
        remaining_proposals = []
        for source_index, peak in enumerate(proposal_peaks):
            if (not np.isfinite(peak.aoa_rad) or not np.isfinite(peak.delay_s)
                    or not search.angles[0] <= peak.aoa_rad <= search.angles[-1]
                    or not search.delays[0] <= peak.delay_s <= search.delays[-1]):
                raise ValueError("MUSIC 候选峰必须位于检验网格范围内")
            index = (int(np.argmin(abs(search.angles-peak.aoa_rad))),
                     int(np.argmin(abs(search.delays-peak.delay_s))))
            remaining_proposals.append((source_index, peak, index))
    with stage("T02a_observed_noise"):
        noise = estimate_noise_from_csi(data)
    diagnostics = {
        "method": "sequential_2d_residual_search_joint_refit_and_duplicate_reduction",
        "settings": options, "backend": backend, "noise_power_estimate": noise,
        "noise_source": "observed_csi_periodic_hann_cir_median",
        "noise_model": "circular_complex_white_gaussian",
        "noise_estimator_assumption": "paths_occupy_a_minority_of_windowed_delay_cells",
        "calibration_scope": "maximum_over_full_2d_grid_with_fitted_linear_projection_and_same_noise_estimator",
        "calibration_limit": "conditional_parametric_noise_calibration; fitted_geometry_and_noise_model_are_not_known_truth",
        "search_grid_shape": [len(aoa_grid_rad), len(delay_grid_s)],
        "search_test_budget": options["max_paths"] + 1,
        "continuous_refinement_policy": "only_after_a_full_grid_test_passes; cannot_bypass_detection_gate",
        "steps": [], "duplicate_checks": [], "input_csi_noise_added": False,
    }
    if proposal_peaks is not None:
        diagnostics.update(method="music_peak_proposals_residual_acceptance_joint_refit",
                           proposal_source="observed_csi_music_local_maxima",
                           music_proposal_count=len(remaining_proposals))
    if not np.isfinite(noise) or noise <= 0:
        diagnostics.update(stop_reason="noise_level_unresolved", accepted_path_count=0)
        return PathDetectionResult([], np.empty((0, data.shape[0]), complex), diagnostics)
    parameters = np.empty((0, 2))
    coefficients = np.empty((0, data.shape[0]), complex)
    residual = data.copy()
    stop = "path_limit_reached"
    for step in range(options["max_paths"] + 1):
        with stage("T02b_search_calibration", search_index=step):
            denominator, maxima, threshold, alpha = search.calibrate(parameters, data.shape[0], options, step)
        with stage("T02c_residual_search", search_index=step):
            scores = search.cpu(search.statistic(residual, denominator)) / noise
            index = np.unravel_index(np.argmax(scores), scores.shape)
            statistic = float(scores[index])
            p_value = float((1 + np.count_nonzero(maxima >= statistic)) / (len(maxima) + 1))
        record = {"search_index": step, "existing_path_count": len(parameters),
                  "grid_maximum_statistic": statistic, "threshold": threshold,
                  "whole_search_p_value": p_value, "allocated_false_alarm_probability": alpha,
                  "calibration_p_value_resolution": 1 / (len(maxima)+1),
                  "proposal_aoa_rad": float(search.angles[index[0]]),
                  "proposal_delay_s": float(search.delays[index[1]])}
        diagnostics["steps"].append(record)
        if p_value > alpha:
            record["decision"] = "stop_no_additional_component"
            stop = "residual_has_no_significant_additional_component"
            break
        if step == options["max_paths"]:
            record["decision"] = "unresolved_path_budget_exhausted"
            break
        proposal = [np.sin(search.angles[index[0]]), search.delays[index[1]] * search.band]
        if remaining_proposals is not None:
            # 保留全二维网格的停止检验：MUSIC 漏峰不能被解释成残差里只有噪声。
            record.update(global_grid_maximum_statistic=statistic, global_whole_search_p_value=p_value)
            if not remaining_proposals:
                record["decision"] = "unresolved_music_proposals_exhausted"
                stop = "music_proposals_exhausted_with_significant_residual"
                break
            best = max(range(len(remaining_proposals)), key=lambda i: scores[remaining_proposals[i][2]])
            source_index, peak, index = remaining_proposals.pop(best)
            statistic = float(scores[index])
            p_value = float((1 + np.count_nonzero(maxima >= statistic)) / (len(maxima) + 1))
            record.update(music_proposal_index=source_index, proposal_aoa_rad=float(peak.aoa_rad),
                          proposal_delay_s=float(peak.delay_s), proposal_statistic=statistic,
                          proposal_whole_search_p_value=p_value)
            if p_value > alpha:
                record["decision"] = "unresolved_no_significant_music_proposal"
                stop = "music_proposals_exhausted_with_significant_residual"
                break
            proposal = [np.sin(peak.aoa_rad), peak.delay_s * search.band]
        previous_energy = float(np.sum(np.abs(residual) ** 2))
        with stage("T02d_joint_csi_refit", search_index=step):
            candidate, gains, candidate_residual, evaluations = search.fit(
                data, np.vstack([parameters, proposal]), options["max_refine_evaluations"])
        improvement = (previous_energy - float(np.sum(np.abs(candidate_residual) ** 2))) / noise
        record.update(refit_improvement_statistic=improvement, refit_evaluations=evaluations)
        if improvement <= threshold:
            record["decision"] = "stop_refit_has_insufficient_improvement"
            stop = ("joint_refit_does_not_support_additional_component" if remaining_proposals is None
                    else "music_proposal_refit_incomplete")
            break
        with stage("T02e_duplicate_refit", search_index=step):
            while len(candidate) > 1:
                dictionary = search.dictionary(candidate)
                similarity = np.abs(dictionary.conj().T @ dictionary) / search.size
                pairs = np.argwhere(np.triu(similarity >= options["duplicate_correlation"], 1))
                reduced = False
                for i, j in pairs:
                    alternatives = []
                    for removed in (i, j):
                        init = np.delete(candidate, removed, axis=0)
                        alternatives.append((f"delete_{removed}", search.fit(data, init, options["max_refine_evaluations"])))
                    # 合并的角度/时延以已拟合成分功率加权；所有复系数重新估计。
                    weights = np.sum(np.abs(gains[[i, j]]) ** 2, axis=1)
                    merged = np.average(candidate[[i, j]], axis=0, weights=weights) if weights.sum() else candidate[i]
                    init = np.delete(candidate, j, axis=0)
                    init[i] = merged
                    alternatives.append(("merge", search.fit(data, init, options["max_refine_evaluations"])))
                    chosen, fitted = min(alternatives, key=lambda item: float(np.sum(np.abs(item[1][2])**2)))
                    loss = (float(np.sum(np.abs(fitted[2])**2)) - float(np.sum(np.abs(candidate_residual)**2))) / noise
                    diagnostics["duplicate_checks"].append({"search_index": step, "pair": [int(i), int(j)],
                        "csi_correlation": float(similarity[i,j]), "best_reduced_model": chosen,
                        "reduction_cost_statistic": loss, "threshold": threshold,
                        "alternatives": [{"action": name, "residual_energy": float(np.sum(np.abs(fit[2])**2))}
                                         for name, fit in alternatives], "reduced": loss <= threshold})
                    if loss <= threshold:
                        candidate, gains, candidate_residual, _ = fitted
                        reduced = True
                        break
                if not reduced:
                    break
        if len(candidate) <= len(parameters):
            # 检验预算不重置；不能不停换提案，直到噪声碰巧通过。
            parameters, coefficients, residual = candidate, gains, candidate_residual
            record["decision"] = "stop_duplicate_explanation"
            stop = ("additional_candidate_duplicates_existing_component" if remaining_proposals is None
                    else "music_duplicate_unresolved")
            break
        parameters, coefficients, residual = candidate, gains, candidate_residual
        record["decision"] = "accept_additional_component"
    peaks = [MusicPeak2D(float(np.arcsin(u)), float(t/search.band),
                        float(np.sum(np.abs(coefficients[i])**2)),
                        int(np.argmin(abs(search.angles-np.arcsin(u)))),
                        int(np.argmin(abs(search.delays-t/search.band)))) for i, (u,t) in enumerate(parameters)]
    diagnostics.update(stop_reason=stop, accepted_path_count=len(peaks),
                       residual_energy=float(np.sum(np.abs(residual)**2)),
                       accepted_peaks=[{"aoa_rad": p.aoa_rad, "delay_s": p.delay_s,
                                        "fitted_power": p.spectrum_value} for p in peaks],
                       coefficients_real=coefficients.real.tolist(), coefficients_imag=coefficients.imag.tolist())
    return PathDetectionResult(peaks, coefficients, diagnostics)
