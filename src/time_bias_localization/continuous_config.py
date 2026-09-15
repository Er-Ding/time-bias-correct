"""连续定位的公开参数；尺度为工程设置，不使用注入噪声或真值。"""
from collections.abc import Mapping
import math


CONTINUOUS_DEFAULTS = {
    "angle_scale_deg": 1.0, "length_scale_m": 0.75,
    "max_hypotheses": 4096, "max_enumerated_sequences": 50000,
    "aoa_gate_deg": 5.0, "length_gate_sigma": 5.0,
    "max_starts": 96, "max_iterations": 80, "max_association_iterations": 8,
    "huber_delta": 2.0, "unmatched_cost": 16.0, "max_residual_norm": 5.0,
    "min_observations": 2, "max_condition_number": 1e8,
    "rank_relative_tolerance": 1e-8, "convergence_step_m": 1e-7,
    "convergence_cost": 1e-10, "ambiguity_cost_tolerance": 0.5,
    "distinct_position_m": 0.25, "distinct_bias_m": 0.25,
    "max_alternatives": 8, "max_seed_combinations": 20000,
}
HYPOTHESIS_SETTING_KEYS = frozenset({
    "max_hypotheses", "max_enumerated_sequences", "aoa_gate_deg", "length_gate_sigma",
})
OBSERVATION_SETTING_KEYS = frozenset({"angle_scale_deg", "length_scale_m"})


def continuous_settings(settings=None):
    settings = {} if settings is None else settings
    if not isinstance(settings, Mapping):
        raise ValueError("localization.continuous 必须是参数字典")
    unknown = set(settings) - set(CONTINUOUS_DEFAULTS)
    if unknown:
        raise ValueError(f"连续定位含未知参数：{sorted(unknown)}")
    values = {**CONTINUOUS_DEFAULTS, **settings}
    for name, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f"continuous.{name} 不能是布尔值")
        if isinstance(CONTINUOUS_DEFAULTS[name], int):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"continuous.{name} 必须为正整数")
        else:
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"continuous.{name} 必须为有限正数")
            values[name] = float(value)
    if not values["angle_scale_deg"] < 90 or not values["aoa_gate_deg"] < 180:
        raise ValueError("连续定位角度尺度必须小于 90 度，分支角门槛必须小于 180 度")
    if values["min_observations"] < 2:
        raise ValueError("连续定位至少需要两条观测，再按物理约束检查是否足够")
    if values["rank_relative_tolerance"] >= 1:
        raise ValueError("连续定位的相对秩容差必须小于 1")
    return values
