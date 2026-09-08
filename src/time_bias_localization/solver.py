"""二维 UE 位置与公共距离偏差的联合求解器。

每个候选解释提供一条随公共距离偏差变化的轨迹：

    p(beta) = anchor - beta * direction

其中 ``beta = c * b``，单位为米。求解器要求所有入选观测在同一个
``beta`` 下尽量汇聚到同一个二维位置。来自同一 ``observation_id`` 的候选
彼此互斥，任一轮计算最多选择其中一个。

本模块只依赖 NumPy，且不输出 accept/reject 一类科学判定。数值上不可辨识的
输入会抛出 :class:`SolverError`，正常结果则始终包含位置均值和完整协方差矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np


Array = np.ndarray


class SolverError(ValueError):
    """输入无法形成可辨识的位置与公共偏差解。"""


def _as_readonly_vector2(value: Sequence[float], name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} 必须是长度为 2 的向量，实际形状为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须全部为有限数值")
    array = array.copy()
    array.setflags(write=False)
    return array


def _as_readonly_array(value: Array, shape: tuple[int, ...], name: str) -> Array:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} 的形状必须为 {shape}，实际为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 必须全部为有限数值")
    array = array.copy()
    array.setflags(write=False)
    return array


def _require_hashable(value: Hashable, name: str) -> None:
    try:
        hash(value)
    except TypeError as exc:
        raise ValueError(f"{name} 必须是可哈希的标识") from exc


def _stable_key(value: Hashable) -> tuple[str, str, str]:
    """为不同类型的标识生成可重复的排序键。"""

    value_type = type(value)
    return (value_type.__module__, value_type.__qualname__, repr(value))


@dataclass(frozen=True)
class CandidateTrajectory:
    """一条观测路径的一种几何解释。

    ``direction`` 必须是单位向量，以保证 ``beta`` 的单位仍为米。
    ``beta_min_m`` 和 ``beta_max_m`` 两端均包含在有效区间内，可使用正负
    无穷表示无界。
    ``weight`` 是相对可信度；数值越大，该候选在拟合中的作用越强。
    """

    observation_id: Hashable
    candidate_id: Hashable
    anchor_m: Sequence[float] = field(repr=False)
    direction: Sequence[float] = field(repr=False)
    beta_min_m: float = -np.inf
    beta_max_m: float = np.inf
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        _require_hashable(self.observation_id, "observation_id")
        _require_hashable(self.candidate_id, "candidate_id")

        anchor = _as_readonly_vector2(self.anchor_m, "anchor_m")
        direction = _as_readonly_vector2(self.direction, "direction")
        direction_norm = float(np.linalg.norm(direction))
        if not np.isclose(direction_norm, 1.0, rtol=1e-5, atol=1e-8):
            raise ValueError(
                "direction 必须是单位向量；否则 beta 将不再表示以米为单位的距离偏差"
            )
        # 消除浮点计算造成的极小单位长度误差。
        direction = direction.copy() / direction_norm
        direction.setflags(write=False)

        beta_min = float(self.beta_min_m)
        beta_max = float(self.beta_max_m)
        if np.isnan(beta_min) or np.isnan(beta_max) or beta_min > beta_max:
            raise ValueError("必须满足 beta_min_m <= beta_max_m，且不能含 NaN")

        weight = float(self.weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight 必须是有限正数")

        object.__setattr__(self, "anchor_m", anchor)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "beta_min_m", beta_min)
        object.__setattr__(self, "beta_max_m", beta_max)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def point(self, beta: float) -> Array:
        """返回给定公共距离偏差下的候选 UE 位置。"""

        return self.anchor_m - float(beta) * self.direction

    def is_valid(self, beta: float, tolerance: float = 0.0) -> bool:
        """判断 ``beta`` 是否落在该候选的有效区间内。"""

        return self.beta_min_m - tolerance <= beta <= self.beta_max_m + tolerance

    @property
    def beta_interval(self) -> tuple[float, float]:
        """以二元组形式返回有效区间，供求解器内部使用。"""

        return (self.beta_min_m, self.beta_max_m)


@dataclass(frozen=True)
class SolverConfig:
    """求解器的数值参数。距离相关参数的单位均为米。"""

    huber_delta: float = 1.0
    max_iterations: int = 40
    max_irls_iterations: int = 30
    tolerance: float = 1e-9
    validity_tolerance: float = 1e-9
    rank_tolerance: float = 1e-10
    max_condition_number: float = 1e12
    max_seeds: int = 512
    max_seed_pairs: int = 100_000
    missing_observation_penalty: float | None = None
    covariance_floor: float = 1e-10
    downweight_threshold: float = 0.999

    def __post_init__(self) -> None:
        positive_values = {
            "huber_delta": self.huber_delta,
            "tolerance": self.tolerance,
            "rank_tolerance": self.rank_tolerance,
            "max_condition_number": self.max_condition_number,
        }
        for name, value in positive_values.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        if not np.isfinite(self.validity_tolerance) or self.validity_tolerance < 0.0:
            raise ValueError("validity_tolerance 必须是有限非负数")
        if not np.isfinite(self.covariance_floor) or self.covariance_floor < 0.0:
            raise ValueError("covariance_floor 必须是有限非负数")
        if not 0.0 < self.downweight_threshold <= 1.0:
            raise ValueError("downweight_threshold 必须位于 (0, 1] 内")
        integer_values = {
            "max_iterations": self.max_iterations,
            "max_irls_iterations": self.max_irls_iterations,
            "max_seeds": self.max_seeds,
            "max_seed_pairs": self.max_seed_pairs,
        }
        for name, value in integer_values.items():
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} 必须是正整数")
            if value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.missing_observation_penalty is not None:
            penalty = self.missing_observation_penalty
            if not np.isfinite(penalty) or penalty < 0.0:
                raise ValueError("missing_observation_penalty 必须是有限非负数")


@dataclass(frozen=True)
class SolverDiagnostics:
    """便于实验分析的数值诊断，不代表 accept/reject 判定。"""

    converged: bool
    iterations: int
    seed_count: int
    evaluated_solution_count: int
    objective: float
    design_rank: int
    condition_number: float
    selected_observation_count: int
    total_observation_count: int
    downweighted_observations: tuple[Hashable, ...]
    unused_observations: tuple[Hashable, ...]
    robust_weights: Mapping[Hashable, float]
    joint_covariance: Array = field(repr=False)

    def __post_init__(self) -> None:
        covariance = _as_readonly_array(
            self.joint_covariance, (3, 3), "joint_covariance"
        )
        object.__setattr__(self, "joint_covariance", covariance)
        object.__setattr__(self, "robust_weights", dict(self.robust_weights))


@dataclass(frozen=True)
class SolverResult:
    """联合估计结果。

    ``mu`` 和 ``sigma`` 分别是 UE 二维高斯近似的均值和完整 ``2 x 2``
    协方差矩阵。``beta`` 是以米为单位的公共距离偏差。
    """

    mu: Array = field(repr=False)
    sigma: Array = field(repr=False)
    beta: float
    selected_candidates: Mapping[Hashable, CandidateTrajectory]
    residuals: Mapping[Hashable, float]
    diagnostics: SolverDiagnostics

    def __post_init__(self) -> None:
        mu = _as_readonly_array(self.mu, (2,), "mu")
        sigma = _as_readonly_array(self.sigma, (2, 2), "sigma")
        if not np.isfinite(self.beta):
            raise ValueError("beta 必须是有限数值")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "selected_candidates", dict(self.selected_candidates))
        object.__setattr__(self, "residuals", dict(self.residuals))

    @property
    def selected_candidate_ids(self) -> dict[Hashable, Hashable]:
        """返回 ``observation_id -> candidate_id`` 的简化映射。"""

        return {
            observation_id: candidate.candidate_id
            for observation_id, candidate in self.selected_candidates.items()
        }


@dataclass
class _RefinedSolution:
    state: Array
    assignment: dict[Hashable, CandidateTrajectory]
    residuals: dict[Hashable, float]
    robust_weights: dict[Hashable, float]
    objective: float
    iterations: int
    converged: bool
    rank: int
    condition_number: float


def _group_candidates(
    candidates: Iterable[CandidateTrajectory],
) -> dict[Hashable, tuple[CandidateTrajectory, ...]]:
    grouped: dict[Hashable, list[CandidateTrajectory]] = {}
    count = 0
    for candidate in candidates:
        if not isinstance(candidate, CandidateTrajectory):
            raise TypeError("candidates 中的每一项都必须是 CandidateTrajectory")
        grouped.setdefault(candidate.observation_id, []).append(candidate)
        count += 1
    if count == 0:
        raise SolverError("没有候选轨迹")
    if len(grouped) < 2:
        raise SolverError("至少需要两条不同 observation_id 的观测")

    ordered: dict[Hashable, tuple[CandidateTrajectory, ...]] = {}
    for observation_id in sorted(grouped, key=_stable_key):
        observation_candidates = sorted(
            grouped[observation_id], key=lambda item: _stable_key(item.candidate_id)
        )
        candidate_ids = [item.candidate_id for item in observation_candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"观测 {observation_id!r} 内的 candidate_id 必须唯一")
        ordered[observation_id] = tuple(observation_candidates)
    return ordered


def _design_matrix(candidates: Sequence[CandidateTrajectory]) -> Array:
    matrix = np.empty((2 * len(candidates), 3), dtype=float)
    for index, candidate in enumerate(candidates):
        matrix[2 * index] = (1.0, 0.0, candidate.direction[0])
        matrix[2 * index + 1] = (0.0, 1.0, candidate.direction[1])
    return matrix


def _weighted_fit(
    candidates: Sequence[CandidateTrajectory],
    weights: Array,
    config: SolverConfig,
) -> tuple[Array, int, float]:
    """在候选固定时求带 beta 区间约束的加权最小二乘解。"""

    if len(candidates) < 2:
        raise SolverError("固定候选后不足两条独立观测")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(candidates),) or np.any(weights <= 0.0):
        raise ValueError("内部权重必须与候选数一致且全部为正")

    beta_min = max(candidate.beta_interval[0] for candidate in candidates)
    beta_max = min(candidate.beta_interval[1] for candidate in candidates)
    if beta_min > beta_max + config.validity_tolerance:
        raise SolverError("所选候选不存在共同有效的 beta 区间")

    anchors = np.stack([candidate.anchor_m for candidate in candidates])
    directions = np.stack([candidate.direction for candidate in candidates])
    weight_sum = float(np.sum(weights))
    anchor_mean = np.sum(weights[:, None] * anchors, axis=0) / weight_sum
    direction_mean = np.sum(weights[:, None] * directions, axis=0) / weight_sum
    centered_anchors = anchors - anchor_mean
    centered_directions = directions - direction_mean
    denominator = float(
        np.sum(weights * np.sum(centered_directions * centered_directions, axis=1))
    )
    if denominator <= config.rank_tolerance:
        raise SolverError("入选轨迹方向几乎相同，公共偏差不可辨识")
    numerator = float(
        np.sum(weights * np.sum(centered_directions * centered_anchors, axis=1))
    )
    beta = numerator / denominator
    beta = float(np.clip(beta, beta_min, beta_max))
    mu = anchor_mean - beta * direction_mean
    state = np.array((mu[0], mu[1], beta), dtype=float)

    matrix = _design_matrix(candidates)
    row_weights = np.repeat(np.sqrt(weights), 2)
    weighted_matrix = matrix * row_weights[:, None]
    rank = int(np.linalg.matrix_rank(weighted_matrix, tol=config.rank_tolerance))
    condition_number = float(np.linalg.cond(weighted_matrix))
    if rank < 3:
        raise SolverError("入选轨迹无法同时辨识二维位置和公共偏差")
    if not np.isfinite(condition_number) or condition_number > config.max_condition_number:
        raise SolverError("轨迹几何条件过差，位置与公共偏差解不稳定")
    return state, rank, condition_number


def _residual(candidate: CandidateTrajectory, state: Array) -> float:
    return float(np.linalg.norm(state[:2] - candidate.point(state[2])))


def _assign_candidates(
    state: Array,
    groups: Mapping[Hashable, tuple[CandidateTrajectory, ...]],
    config: SolverConfig,
) -> dict[Hashable, CandidateTrajectory]:
    """每个观测只选择一个当前残差最小的有效候选。"""

    assignment: dict[Hashable, CandidateTrajectory] = {}
    beta = float(state[2])
    for observation_id, candidates in groups.items():
        choices: list[tuple[float, tuple[str, str, str], CandidateTrajectory]] = []
        for candidate in candidates:
            if not candidate.is_valid(beta, config.validity_tolerance):
                continue
            residual = _residual(candidate, state)
            # weight 视作逆方差，因此在候选之间比较标准化残差。
            cost = candidate.weight * residual * residual
            choices.append((cost, _stable_key(candidate.candidate_id), candidate))
        if choices:
            choices.sort(key=lambda item: (item[0], item[1]))
            assignment[observation_id] = choices[0][2]
    return assignment


def _huber_multiplier(standardized_residuals: Array, delta: float) -> Array:
    multipliers = np.ones_like(standardized_residuals)
    large = standardized_residuals > delta
    multipliers[large] = delta / standardized_residuals[large]
    return multipliers


def _huber_loss(standardized_residual: float, delta: float) -> float:
    if standardized_residual <= delta:
        return 0.5 * standardized_residual * standardized_residual
    return delta * (standardized_residual - 0.5 * delta)


def _fit_assignment(
    assignment: Mapping[Hashable, CandidateTrajectory],
    initial_state: Array,
    config: SolverConfig,
) -> tuple[Array, dict[Hashable, float], int, int, float, bool]:
    observation_ids = list(assignment)
    candidates = [assignment[observation_id] for observation_id in observation_ids]
    base_weights = np.array([candidate.weight for candidate in candidates], dtype=float)
    state = np.asarray(initial_state, dtype=float).copy()
    converged = False
    rank = 0
    condition_number = np.inf

    for iteration in range(1, config.max_irls_iterations + 1):
        residuals = np.array([_residual(candidate, state) for candidate in candidates])
        standardized = np.sqrt(base_weights) * residuals
        multipliers = _huber_multiplier(standardized, config.huber_delta)
        effective_weights = base_weights * multipliers
        new_state, rank, condition_number = _weighted_fit(
            candidates, effective_weights, config
        )
        if np.linalg.norm(new_state - state) <= config.tolerance * (
            1.0 + np.linalg.norm(state)
        ):
            state = new_state
            converged = True
            break
        state = new_state

    final_residuals = np.array([_residual(candidate, state) for candidate in candidates])
    final_standardized = np.sqrt(base_weights) * final_residuals
    final_multipliers = _huber_multiplier(final_standardized, config.huber_delta)
    robust_weights = {
        observation_id: float(multiplier)
        for observation_id, multiplier in zip(observation_ids, final_multipliers)
    }
    return state, robust_weights, iteration, rank, condition_number, converged


def _objective(
    state: Array,
    assignment: Mapping[Hashable, CandidateTrajectory],
    total_observation_count: int,
    config: SolverConfig,
) -> float:
    loss = 0.0
    for candidate in assignment.values():
        standardized = np.sqrt(candidate.weight) * _residual(candidate, state)
        loss += _huber_loss(standardized, config.huber_delta)
    missing_count = total_observation_count - len(assignment)
    missing_penalty = config.missing_observation_penalty
    if missing_penalty is None:
        missing_penalty = 4.0 * config.huber_delta * config.huber_delta
    return float(loss + missing_count * missing_penalty)


def _assignment_signature(
    assignment: Mapping[Hashable, CandidateTrajectory],
) -> tuple[tuple[tuple[str, str, str], tuple[str, str, str]], ...]:
    return tuple(
        (_stable_key(observation_id), _stable_key(candidate.candidate_id))
        for observation_id, candidate in assignment.items()
    )


def _refine_seed(
    seed: Array,
    groups: Mapping[Hashable, tuple[CandidateTrajectory, ...]],
    config: SolverConfig,
) -> _RefinedSolution | None:
    state = seed.copy()
    total_iterations = 0
    converged = False
    rank = 0
    condition_number = np.inf
    robust_weights: dict[Hashable, float] = {}

    for _ in range(config.max_iterations):
        assignment = _assign_candidates(state, groups, config)
        if len(assignment) < 2:
            return None
        signature_before = _assignment_signature(assignment)
        try:
            (
                new_state,
                robust_weights,
                irls_iterations,
                rank,
                condition_number,
                irls_converged,
            ) = _fit_assignment(assignment, state, config)
        except SolverError:
            return None
        total_iterations += irls_iterations
        assignment_after = _assign_candidates(new_state, groups, config)
        signature_after = _assignment_signature(assignment_after)
        state_change = np.linalg.norm(new_state - state)
        state = new_state
        if (
            signature_after == signature_before
            and state_change <= config.tolerance * (1.0 + np.linalg.norm(state))
        ):
            converged = irls_converged
            break

    # 收尾时必须让“候选分配”和“用这些候选拟合出的状态”同时稳定。旧实现
    # 在最后拟合后只重新分配、不再重拟合，边界附近会返回属于两套候选的混合
    # 结果。若有限轮内仍来回切换，宁可丢弃该初值，也不返回口径不一致的解。
    final_assignment = _assign_candidates(state, groups, config)
    assignment_stable = False
    final_converged = False
    for _ in range(config.max_iterations):
        if len(final_assignment) < 2:
            return None
        signature_before = _assignment_signature(final_assignment)
        try:
            (
                state,
                robust_weights,
                final_iterations,
                rank,
                condition_number,
                final_converged,
            ) = _fit_assignment(final_assignment, state, config)
        except SolverError:
            return None
        total_iterations += final_iterations
        reassigned = _assign_candidates(state, groups, config)
        if (
            _assignment_signature(reassigned) == signature_before
            and final_converged
        ):
            final_assignment = reassigned
            assignment_stable = True
            break
        final_assignment = reassigned
    if not assignment_stable:
        return None
    converged = converged and final_converged
    residuals = {
        observation_id: _residual(candidate, state)
        for observation_id, candidate in final_assignment.items()
    }
    objective = _objective(state, final_assignment, len(groups), config)
    return _RefinedSolution(
        state=state,
        assignment=final_assignment,
        residuals=residuals,
        robust_weights=robust_weights,
        objective=objective,
        iterations=total_iterations,
        converged=converged,
        rank=rank,
        condition_number=condition_number,
    )


def _make_seeds(
    groups: Mapping[Hashable, tuple[CandidateTrajectory, ...]],
    config: SolverConfig,
) -> list[Array]:
    # max_seeds 只限制后续精修，无法约束此前的候选对生成和评分。
    # 先显式检查预算，绝不按输入顺序静默丢弃部分观测的候选组合。
    pair_count = sum(
        len(groups[first_id]) * len(groups[second_id])
        for first_id, second_id in combinations(groups, 2)
    )
    if pair_count > config.max_seed_pairs:
        raise SolverError(
            f"跨观测候选对数量 {pair_count} 超过 max_seed_pairs={config.max_seed_pairs}；"
            "请检查第一次聚类半径与采样范围，或明确提高候选对预算"
        )
    seeds: list[Array] = []
    seen: set[tuple[float, float, float]] = set()
    for first_id, second_id in combinations(groups, 2):
        for first in groups[first_id]:
            for second in groups[second_id]:
                beta_min = max(first.beta_interval[0], second.beta_interval[0])
                beta_max = min(first.beta_interval[1], second.beta_interval[1])
                if beta_min > beta_max + config.validity_tolerance:
                    continue
                try:
                    seed, _, _ = _weighted_fit(
                        (first, second),
                        np.array((first.weight, second.weight), dtype=float),
                        config,
                    )
                except SolverError:
                    continue
                key = tuple(float(value) for value in np.round(seed, decimals=12))
                if key not in seen:
                    seen.add(key)
                    seeds.append(seed)

    # 先保留对全部观测解释得更好的初值，避免候选数很大时组合爆炸。
    seeds.sort(
        key=lambda state: (
            _objective(
                state,
                _assign_candidates(state, groups, config),
                len(groups),
                config,
            ),
            float(state[2]),
            float(state[0]),
            float(state[1]),
        )
    )
    return seeds[: config.max_seeds]


def _positive_semidefinite(matrix: Array) -> Array:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _estimate_covariance(
    solution: _RefinedSolution,
    config: SolverConfig,
) -> tuple[Array, Array]:
    candidates = list(solution.assignment.values())
    multipliers = np.array(
        [
            solution.robust_weights.get(candidate.observation_id, 1.0)
            for candidate in candidates
        ],
        dtype=float,
    )
    base_weights = np.array([candidate.weight for candidate in candidates], dtype=float)
    effective_weights = base_weights * multipliers
    matrix = _design_matrix(candidates)
    observations = np.concatenate([candidate.anchor_m for candidate in candidates])
    row_weights = np.repeat(effective_weights, 2)
    normal_matrix = matrix.T @ (row_weights[:, None] * matrix)
    residual_vector = observations - matrix @ solution.state
    weighted_squared_error = float(np.sum(row_weights * residual_vector**2))
    rank = int(np.linalg.matrix_rank(matrix, tol=config.rank_tolerance))
    degrees_of_freedom = max(matrix.shape[0] - rank, 1)
    residual_variance = max(
        weighted_squared_error / degrees_of_freedom, config.covariance_floor
    )
    joint_covariance = np.linalg.pinv(
        normal_matrix, rcond=config.rank_tolerance
    ) * residual_variance
    joint_covariance = _positive_semidefinite(joint_covariance)
    position_covariance = _positive_semidefinite(joint_covariance[:2, :2])
    return position_covariance, joint_covariance


def solve_position_and_bias(
    candidates: Iterable[CandidateTrajectory],
    config: SolverConfig | None = None,
) -> SolverResult:
    """联合估计二维 UE 位置和公共距离偏差。

    求解步骤为：从不同观测的两条候选轨迹产生初值；对每个初值交替执行
    互斥候选选择和稳健加权拟合；最后选择整体稳健代价最小的解。

    Args:
        candidates: 所有观测的候选轨迹，可按任意顺序给出。
        config: 可选数值配置。

    Returns:
        包含 ``mu``、完整 ``sigma``、``beta``、候选选择与诊断的结果。

    Raises:
        SolverError: 少于两条不同观测，或轨迹几何无法辨识公共偏差。
    """

    config = config or SolverConfig()
    groups = _group_candidates(candidates)
    seeds = _make_seeds(groups, config)
    if not seeds:
        raise SolverError("没有两条候选轨迹能够产生可辨识且区间合法的初始解")

    solutions: list[_RefinedSolution] = []
    for seed in seeds:
        solution = _refine_seed(seed, groups, config)
        if solution is not None:
            solutions.append(solution)
    if not solutions:
        raise SolverError("所有初始解均在稳健拟合过程中失效")

    solutions.sort(
        key=lambda item: (
            item.objective,
            -len(item.assignment),
            float(item.state[2]),
            float(item.state[0]),
            float(item.state[1]),
            _assignment_signature(item.assignment),
        )
    )
    best = solutions[0]
    sigma, joint_covariance = _estimate_covariance(best, config)
    unused_observations = tuple(
        observation_id
        for observation_id in groups
        if observation_id not in best.assignment
    )
    downweighted_observations = tuple(
        observation_id
        for observation_id in best.assignment
        if best.robust_weights.get(observation_id, 1.0) < config.downweight_threshold
    )
    diagnostics = SolverDiagnostics(
        converged=best.converged,
        iterations=best.iterations,
        seed_count=len(seeds),
        evaluated_solution_count=len(solutions),
        objective=best.objective,
        design_rank=best.rank,
        condition_number=best.condition_number,
        selected_observation_count=len(best.assignment),
        total_observation_count=len(groups),
        downweighted_observations=downweighted_observations,
        unused_observations=unused_observations,
        robust_weights=best.robust_weights,
        joint_covariance=joint_covariance,
    )
    return SolverResult(
        mu=best.state[:2],
        sigma=sigma,
        beta=float(best.state[2]),
        selected_candidates=best.assignment,
        residuals=best.residuals,
        diagnostics=diagnostics,
    )


# 简短别名，便于流水线模块调用。
solve_trajectories = solve_position_and_bias


__all__ = [
    "CandidateTrajectory",
    "SolverConfig",
    "SolverDiagnostics",
    "SolverError",
    "SolverResult",
    "solve_position_and_bias",
    "solve_trajectories",
]
