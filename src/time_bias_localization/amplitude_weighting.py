"""按 MUSIC 谱幅值为各观测分配不同的噪声尺度。

动机：MUSIC 的角度与时延估计方差随该路径的信噪比上升而下降。谱幅值是
观测端可得的路由强度代理量——不需要材质参数，也不需要真值。因此衰减的
观测应获得更大的噪声尺度，从而在联合求解中获得更小的权重。

尺度仍由公开超参数 ``angle_scale_deg`` 与 ``length_scale_m`` 给定，本模块
只按幅值比值缩放它们，不标定绝对噪声。两条用途因此分离：
相对权重由幅值决定，绝对尺度仍是经验设置（并同时决定 5σ 接受门限）。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np


DEFAULT_AMPLITUDE_WEIGHTING: dict = {
    "enabled": False,
    "exponent": 1.0,
    "reference": "maximum",
    "maximum_factor": 10.0,
}

REFERENCE_MODES = ("maximum", "median", "minimum")


def amplitude_weighting_settings(settings: Mapping | None) -> dict:
    """校验并补全幅值加权参数；未给定时返回关闭状态。"""
    if settings is None:
        return dict(DEFAULT_AMPLITUDE_WEIGHTING)
    if not isinstance(settings, Mapping):
        raise ValueError("amplitude_weighting 必须是参数字典")
    unknown = set(settings) - set(DEFAULT_AMPLITUDE_WEIGHTING)
    if unknown:
        raise ValueError(f"amplitude_weighting 含未知参数：{sorted(unknown)}")
    values = {**DEFAULT_AMPLITUDE_WEIGHTING, **settings}
    if not isinstance(values["enabled"], bool):
        raise ValueError("amplitude_weighting.enabled 必须为布尔值")
    if values["reference"] not in REFERENCE_MODES:
        raise ValueError("amplitude_weighting.reference 只能为 "
                         + "、".join(REFERENCE_MODES))
    for name in ("exponent", "maximum_factor"):
        value = values[name]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"amplitude_weighting.{name} 必须是非负有限数")
        values[name] = float(value)
    if values["enabled"] and values["maximum_factor"] < 1:
        raise ValueError("amplitude_weighting.maximum_factor 启用时必须不小于 1")
    return values


def amplitude_scale_factors(amplitudes: Sequence[float], settings: Mapping | None) -> np.ndarray:
    """返回每个观测的噪声尺度倍数；衰减的观测取大于 1 的倍数。

    倍数为 ``clip((参考幅值 / 该幅值) ** exponent, 1, maximum_factor)``。
    参考幅值由 ``reference`` 决定。幅值必须为正且有限；关闭时全部返回 1。
    """
    values = amplitude_weighting_settings(settings)
    amplitudes = np.asarray(amplitudes, float)
    if amplitudes.ndim != 1 or amplitudes.size == 0:
        raise ValueError("幅值必须是非空一维序列")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes <= 0):
        raise ValueError("幅值必须是有限正数")
    if not values["enabled"]:
        return np.ones(amplitudes.size)
    reference = {"maximum": amplitudes.max(), "median": float(np.median(amplitudes)),
                 "minimum": amplitudes.min()}[values["reference"]]
    factors = np.power(reference / amplitudes, values["exponent"])
    return np.clip(factors, 1.0, values["maximum_factor"])
