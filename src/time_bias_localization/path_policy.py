"""统一路径方向：数据和函数内部为 UE→BS，六类统计对外按 BS→UE。"""
from __future__ import annotations

import numpy as np

PATH_TYPE_NAMES = ("直达", "一次反射", "两次反射", "单次绕射", "一次反射后绕射", "两次反射后绕射")
UPLINK_SEQUENCES = ((), (1,), (1, 1), (8,), (8, 1), (8, 1, 1))


def validate_diffraction_position(value: str) -> str:
    if value not in ("any", "last_from_bs"):
        raise ValueError("diffraction_position 只能为 any 或 last_from_bs")
    return value


def uplink_path_mask(interactions: np.ndarray, policy: str) -> np.ndarray:
    """last_from_bs 表示绕射后直接到 UE；上行数组中绕射必须在最前面。"""
    validate_diffraction_position(policy)
    kinds = np.asarray(interactions)
    if kinds.ndim != 2:
        raise ValueError("交互数组必须为 [交互深度, 路径数]")
    if policy == "any":
        return np.ones(kinds.shape[1], dtype=bool)
    return np.asarray([tuple(column[column != 0]) in UPLINK_SEQUENCES for column in kinds.T], dtype=bool)


def path_type_counts(interactions: np.ndarray, retained: np.ndarray) -> list[int]:
    """只统计实际合成 CSI 的路径；遇到六类以外的保留路径必须报错。"""
    kinds, kept = np.asarray(interactions), np.asarray(retained, dtype=bool)
    if kinds.ndim != 2 or kept.shape != (kinds.shape[1],):
        raise ValueError("路径交互与保留掩码形状不一致")
    counts = [0] * 6
    for column in kinds[:, kept].T:
        sequence = tuple(column[column != 0])
        if sequence not in UPLINK_SEQUENCES:
            raise ValueError(f"发现第一阶段六类之外的上行路径：{sequence}")
        counts[UPLINK_SEQUENCES.index(sequence)] += 1
    return counts


def selected_path_type_counts(paths: list[dict]) -> list[int]:
    counts = [0] * 6
    for path in paths:
        sequence = tuple({"reflection": 1, "diffraction": 8}[kind]
                         for kind, _ in path["propagation_interactions"])
        if sequence not in UPLINK_SEQUENCES:
            raise ValueError(f"定位结果包含第一阶段六类之外的路径：{sequence}")
        counts[UPLINK_SEQUENCES.index(sequence)] += 1
    return counts
