"""由观测协方差的特征值划分 MUSIC 子空间，不使用路径真值或注入噪声。"""
from collections.abc import Mapping

import numpy as np


DEFAULT_SUBSPACE_SELECTION = {
    "mode": "fixed",  # 旧配置的复现入口；新实验显式启用 eigenvalue_threshold。
    "noise_reference": "median",
    "threshold_ratio": 6.0,
}


def subspace_settings(settings=None):
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping) or set(settings) - set(DEFAULT_SUBSPACE_SELECTION):
        raise ValueError("music.subspace_selection 包含未定义字段或不是键值映射")
    result = {**DEFAULT_SUBSPACE_SELECTION, **settings}
    if result["mode"] not in {"fixed", "eigenvalue_threshold"}:
        raise ValueError("subspace_selection.mode 必须为 fixed 或 eigenvalue_threshold")
    if result["noise_reference"] not in {"minimum", "median", "lower_half_median"}:
        raise ValueError("noise_reference 必须为 minimum、median 或 lower_half_median")
    ratio = result["threshold_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or ratio <= 1:
        raise ValueError("threshold_ratio 必须为大于 1 的有限数")
    return result


class SubspaceSelectionError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__("观测协方差无法提供可靠的相对噪声基准")


def select_subspace_rank(eigenvalues, *, fixed_rank, settings=None):
    """输入升序、去掉数值对角加载后的特征值；返回维数及可核查的分界。

    median 假设大部分特征方向由噪声主导；lower_half_median 只取较小一半，
    minimum 对应最小特征值倍数规则。均为可配置估计规则，不宣称已校准误检概率。
    """
    options = subspace_settings(settings)
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values) < 0):
        raise ValueError("特征值必须为至少两个有限数，按升序排列")
    diagnostics = {**options, "covariance_dimension": int(values.size),
                   "eigenvalues_ascending": values.tolist(),
                   "source": "observed_csi_covariance_before_diagonal_loading"}
    if options["mode"] == "fixed":
        rank = int(fixed_rank)
        if isinstance(fixed_rank, bool) or rank != fixed_rank or not 1 <= rank < values.size:
            raise ValueError("固定信号维数必须位于 [1, 协方差维数-1]")
        diagnostics.update(signal_rank=rank, noise_rank=int(values.size-rank))
        return rank, diagnostics
    reference = float(values[0] if options["noise_reference"] == "minimum"
                      else np.median(values if options["noise_reference"] == "median"
                                     else values[:values.size // 2]))
    tolerance = float(np.finfo(float).eps * values.size * np.max(np.abs(values)))
    diagnostics.update(noise_reference_value=reference, numerical_tolerance=tolerance,
                       threshold=float(options["threshold_ratio"] * reference))
    if reference <= tolerance:
        diagnostics.update(status="noise_reference_unresolved", signal_rank=None, noise_rank=None)
        raise SubspaceSelectionError(diagnostics)
    rank = int(np.count_nonzero(values > diagnostics["threshold"]))
    diagnostics.update(status="resolved", signal_rank=rank, noise_rank=int(values.size-rank),
                       comparison="eigenvalue > threshold", rank_was_clamped=False)
    return rank, diagnostics
