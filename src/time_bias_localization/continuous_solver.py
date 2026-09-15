"""直接拟合地图传播函数与原始角度、时延观测。

只优化 ``(x, y, beta=c*b)``。路线选择用一对一匹配；同一个峰或物理
路线不能重复计票。多初值来自允许区域及连续距离/角度方程，不读取旧
候选点、点簇、代表方向或 RANSAC 结果。局部更新始终重新验证物理路径。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
from itertools import combinations, product
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, lsq_linear

from .propagation_model import ContinuousObservation, evaluate_hypothesis


@dataclass(frozen=True)
class ContinuousSolverConfig:
    xy_bounds_m: tuple[tuple[float, float], tuple[float, float]]
    bias_bounds_m: tuple[float, float]
    speed_of_light_mps: float = 299792458.0
    max_starts: int = 96
    max_iterations: int = 80
    max_association_iterations: int = 8
    seed: int = 0
    huber_delta: float = 2.0
    unmatched_cost: float = 16.0
    max_residual_norm: float = 5.0
    min_observations: int = 2
    max_condition_number: float = 1e8
    rank_relative_tolerance: float = 1e-8
    convergence_step_m: float = 1e-7
    convergence_cost: float = 1e-10
    ambiguity_cost_tolerance: float = 0.5
    distinct_position_m: float = 0.25
    distinct_bias_m: float = 0.25
    max_alternatives: int = 8
    max_seed_combinations: int = 20000

    def __post_init__(self) -> None:
        bounds = np.asarray((*self.xy_bounds_m, self.bias_bounds_m), float)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError("位置和偏差范围必须为三个有限上下界")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("位置和偏差的上界必须大于下界")
        for name in ("max_starts", "max_iterations", "max_association_iterations",
                     "min_observations", "max_alternatives", "max_seed_combinations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed 必须是非负整数")
        for name in ("speed_of_light_mps", "huber_delta", "unmatched_cost",
                     "max_residual_norm", "max_condition_number",
                     "rank_relative_tolerance", "convergence_step_m", "convergence_cost",
                     "distinct_position_m", "distinct_bias_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        if (not math.isfinite(self.ambiguity_cost_tolerance)
                or self.ambiguity_cost_tolerance < 0):
            raise ValueError("ambiguity_cost_tolerance 必须是有限非负数")


@dataclass(frozen=True)
class ContinuousSolverResult:
    status: str
    position_m: tuple[float, float] | None
    beta_m: float | None
    clock_bias_s: float | None
    selected_paths: tuple[dict[str, Any], ...]
    best_candidate: dict[str, Any] | None
    alternatives: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    scientific_validation_status: str = "not_validated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _huber(squared_norm, delta):
    value = np.asarray(squared_norm)
    return np.where(value <= delta * delta, value,
                    2 * delta * np.sqrt(np.maximum(value, 0)) - delta * delta)


class _Problem:
    def __init__(self, bank, config):
        self.bank, self.config = bank, config
        self.observations = tuple(bank.observations)
        self.hypotheses = tuple(bank.hypotheses)
        self.n = len(self.observations)
        self.h = len(self.hypotheses)
        ids = [o.observation_id for o in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError("同一个观测编号不能重复输入")
        # A bank should already canonicalize IDs. Reject aliases rather than let
        # an accidental duplicate column create a second vote for one path.
        route_ids = [(tuple(h.interactions), tuple(h.receiver_m)) for h in self.hypotheses]
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("传播函数库包含重复物理路线")
        self.bounds = np.asarray((*config.xy_bounds_m, config.bias_bounds_m), float)
        self.a = np.asarray([h.affine_image_matrix for h in self.hypotheses], float).reshape(-1, 2, 2)
        self.t = np.asarray([h.affine_image_offset for h in self.hypotheses], float).reshape(-1, 2)
        self.anchors = np.asarray([h.anchor_m for h in self.hypotheses], float).reshape(-1, 2)
        self.fixed = np.asarray([h.fixed_length_m for h in self.hypotheses], float)
        self.fixed_angles = np.asarray([np.nan if h.fixed_aoa_rad is None else h.fixed_aoa_rad
                                        for h in self.hypotheses], float)
        self.z = np.asarray([o.observed_length_m for o in self.observations])
        self.angles = np.asarray([o.aoa_rad for o in self.observations])
        self.sa = np.asarray([o.angle_scale_rad for o in self.observations])
        self.sl = np.asarray([o.length_scale_m for o in self.observations])
        self.allowed = np.zeros((self.n, self.h), bool)
        if len(bank.observation_hypothesis_indices) != self.n:
            raise ValueError("观测与传播函数索引的数量不一致")
        for i, indices in enumerate(bank.observation_hypothesis_indices):
            for j in indices:
                if j < 0 or j >= self.h:
                    raise ValueError("传播函数索引越界")
                self.allowed[i, j] = True
        self.path_checks = 0
        self.invalid_checks = 0
        self.invalid_reason_counts: dict[str, int] = {}
        self.backtracked_steps = 0

    def in_bounds(self, state):
        return (np.asarray(state).shape == (3,) and np.all(np.isfinite(state))
                and np.all(state >= self.bounds[:, 0]) and np.all(state <= self.bounds[:, 1]))

    def smooth_residuals(self, state):
        vectors = np.einsum("hij,j->hi", self.a, state[:2]) + self.t - self.anchors
        lengths = np.linalg.norm(vectors, axis=1) + self.fixed
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angles = np.where(np.isnan(self.fixed_angles), angles, self.fixed_angles)
        angle_r = _wrap(angles[None, :] - self.angles[:, None]) / self.sa[:, None]
        length_r = (lengths[None, :] + state[2] - self.z[:, None]) / self.sl[:, None]
        residuals = np.stack((angle_r, length_r), axis=-1)
        singular = np.linalg.norm(vectors, axis=1) <= 1e-9
        return residuals, singular

    def evaluation(self, j, state):
        evaluation = evaluate_hypothesis(self.bank.scene, self.hypotheses[j], state[:2])
        self.path_checks += 1
        if not evaluation.valid:
            self.invalid_checks += 1
            reason = evaluation.invalid_reason or "unspecified_geometry_failure"
            self.invalid_reason_counts[reason] = self.invalid_reason_counts.get(reason, 0) + 1
        return evaluation

    def assignment(self, state, *, exploratory=False):
        """Match one peak to at most one distinct legal path, with dummy outliers.

        Analytic costs are cheap. Geometry is evaluated only for columns selected
        by the assignment; a rejected column is removed and matching repeated.
        This is equivalent to eagerly checking every column at this same state.
        """
        if not self.in_bounds(state) or not self.n or not self.h:
            return (), self.n * self.config.unmatched_cost, {}
        residuals, singular = self.smooth_residuals(state)
        norms2 = np.sum(residuals * residuals, axis=2)
        costs = _huber(norms2, self.config.huber_delta)
        allowed = self.allowed & ~singular[None, :] & np.isfinite(costs)
        if not exploratory:
            allowed &= norms2 <= self.config.max_residual_norm ** 2
        dummy = (max(float(np.max(costs[allowed], initial=0)) + 1,
                     self.config.unmatched_cost) if exploratory else self.config.unmatched_cost)
        forbidden = max(dummy * (self.n + 2) + 1, 1e12)
        real_costs = np.where(allowed & (costs < dummy), costs, forbidden)
        table = np.concatenate((real_costs, np.full((self.n, self.n), dummy)), axis=1)
        evaluations = {}
        while True:
            rows, cols = linear_sum_assignment(table)
            selected = [(int(i), int(j)) for i, j in zip(rows, cols) if j < self.h]
            rejected = []
            for _, j in selected:
                if j not in evaluations:
                    evaluations[j] = self.evaluation(j, state)
                if not evaluations[j].valid:
                    rejected.append(j)
            if not rejected:
                break
            table[:, rejected] = forbidden
        score = sum(float(costs[i, j]) for i, j in selected)
        score += (self.n - len(selected)) * self.config.unmatched_cost
        return tuple(selected), float(score), evaluations

    def residual_jacobian(self, state, assignment):
        residuals, jacobians = [], []
        for i, j in assignment:
            value = evaluate_hypothesis(self.bank.scene, self.hypotheses[j], state[:2],
                                        check_validity=False)
            residuals.append([float(_wrap(value.aoa_rad - self.angles[i]) / self.sa[i]),
                              (value.length_m + state[2] - self.z[i]) / self.sl[i]])
            jacobians.append([
                [*np.asarray(value.aoa_gradient_xy) / self.sa[i], 0.0],
                [*np.asarray(value.length_gradient_xy) / self.sl[i], 1.0 / self.sl[i]],
            ])
        return np.asarray(residuals).reshape(-1, 2), np.asarray(jacobians).reshape(-1, 2, 3)

    def local_fit(self, state, assignment):
        """Group-Huber Gauss-Newton with legal-step backtracking.

        Fixed assignment means fixed residual dimension. No fabricated residual
        is substituted for an invalid path. An invalid proposal is shortened.
        """
        state = np.asarray(state, float).copy()
        if not assignment:
            return state, False, "no_assignment", 0
        damping = 1e-6
        for iteration in range(self.config.max_iterations):
            residuals, jac = self.residual_jacobian(state, assignment)
            norms = np.linalg.norm(residuals, axis=1)
            weights = np.minimum(1.0, self.config.huber_delta / np.maximum(norms, 1e-15))
            j = (jac * np.sqrt(weights)[:, None, None]).reshape(-1, 3)
            r = (residuals * np.sqrt(weights)[:, None]).reshape(-1)
            gradient = j.T @ r
            normal = j.T @ j
            diagonal = np.maximum(np.diag(normal), 1e-12)
            # At a box boundary, an outward gradient is compatible with a
            # constrained minimum. Test the feasible gradient, not the raw one.
            projected_gradient = gradient.copy()
            at_lower = state <= self.bounds[:, 0] + self.config.convergence_step_m
            at_upper = state >= self.bounds[:, 1] - self.config.convergence_step_m
            projected_gradient[at_lower & (gradient > 0)] = 0.0
            projected_gradient[at_upper & (gradient < 0)] = 0.0
            gradient_norm = float(np.linalg.norm(projected_gradient / np.sqrt(diagonal)))
            if gradient_norm <= 1e-9:
                return state, True, "projected_gradient_stationary", iteration + 1
            # Only three unknowns: solve the bounded linearized problem directly
            # so a clock-boundary optimum is reachable. Geometry still needs the
            # independent physical backtracking checks below.
            augmented_j = np.vstack((j, np.diag(np.sqrt(damping * diagonal))))
            augmented_r = np.r_[-r, np.zeros(3)]
            subproblem = lsq_linear(
                augmented_j, augmented_r,
                bounds=(self.bounds[:, 0] - state, self.bounds[:, 1] - state),
                method="bvls", tol=1e-12, max_iter=30,
            )
            step = subproblem.x
            if np.linalg.norm(step) <= self.config.convergence_step_m:
                if damping <= 1e-3 and subproblem.optimality <= 1e-7:
                    return state, True, "stationary", iteration + 1
                return state, False, "damping_stalled", iteration + 1
            base = float(np.sum(_huber(norms * norms, self.config.huber_delta)))
            accepted = False
            alpha = 1.0
            for _ in range(30):
                proposal = np.clip(state + alpha * step, self.bounds[:, 0], self.bounds[:, 1])
                if self.in_bounds(proposal):
                    proposed_r, _ = self.residual_jacobian(proposal, assignment)
                    score = float(np.sum(_huber(np.sum(proposed_r * proposed_r, axis=1),
                                                self.config.huber_delta)))
                    if np.isfinite(score) and score < base:
                        legal = all(self.evaluation(k, proposal).valid for _, k in assignment)
                        if legal:
                            accepted = True
                            break
                alpha *= 0.5
                self.backtracked_steps += 1
            if accepted:
                improvement = base - score
                state = proposal
                damping = max(damping / 3, 1e-12)
                # Do not call a tiny boundary-limited step convergence. A nearby
                # legal boundary can prevent descent without locating a minimum.
                if (improvement <= self.config.convergence_cost
                        and alpha >= 0.25 and np.linalg.norm(alpha * step) < 1e-4):
                    return state, True, "cost_stable", iteration + 1
            else:
                damping *= 10
                if damping > 1e10:
                    return state, False, "blocked_or_no_descent", iteration + 1
        return state, False, "iteration_budget", self.config.max_iterations


def _halton(index, base):
    value, factor = 0.0, 1.0
    while index:
        factor /= base
        value += factor * (index % base)
        index //= base
    return value


def _specular_ray(problem, i, j):
    direction = np.array([math.cos(problem.angles[i]), math.sin(problem.angles[i])])
    a = problem.a[j].T @ (problem.anchors[j] - problem.t[j]
                            + (problem.z[i] - problem.fixed[j]) * direction)
    u = problem.a[j].T @ direction
    return a, u


def _pair_roots(problem, first, second):
    (i, j), (k, l) = first, second
    spec_first, spec_second = np.isnan(problem.fixed_angles[[j, l]])
    if not spec_first and not spec_second:
        return []
    if not spec_first:
        return _pair_roots(problem, second, first)
    a, u = _specular_ray(problem, i, j)
    if spec_second:
        other_a, other_u = _specular_ray(problem, k, l)
        du = u - other_u
        denominator = float(du @ du)
        if denominator < 1e-12:
            return []
        beta = float(du @ (a - other_a) / denominator)
        if min(problem.z[i] - problem.fixed[j] - beta,
               problem.z[k] - problem.fixed[l] - beta) <= 0:
            return []
        xy = 0.5 * (a - beta * u + other_a - beta * other_u)
    else:
        q = problem.a[l] @ a + problem.t[l] - problem.anchors[l]
        v = problem.a[l] @ u
        s = problem.z[k] - problem.fixed[l]
        denominator = 2 * (s - float(q @ v))
        if abs(denominator) < 1e-10:
            return []
        beta = (s * s - float(q @ q)) / denominator
        if min(s - beta, problem.z[i] - problem.fixed[j] - beta) <= 0:
            return []
        xy = a - beta * u
    return [np.r_[xy, beta]]


def _diffraction_roots(problem, entries):
    centers = np.asarray([problem.a[j].T @ (problem.anchors[j] - problem.t[j])
                          for _, j in entries])
    s = np.asarray([problem.z[i] - problem.fixed[j] for i, j in entries])
    matrix = 2 * (centers[1:] - centers[0])
    if np.linalg.matrix_rank(matrix, tol=1e-10) < 2:
        return []
    p = np.linalg.solve(matrix, np.sum(centers[1:] ** 2, axis=1)
                         - centers[0] @ centers[0] - s[1:] ** 2 + s[0] ** 2)
    q = np.linalg.solve(matrix, 2 * (s[1:] - s[0]))
    d = p - centers[0]
    coefficients = [float(q @ q) - 1, 2 * (float(q @ d) + s[0]), float(d @ d) - s[0] ** 2]
    roots = np.roots(np.trim_zeros(coefficients, trim="f")) if any(coefficients) else []
    result = []
    for root in roots:
        if abs(np.imag(root)) <= 1e-7:
            beta = float(np.real(root))
            if np.all(s - beta > 0):
                result.append(np.r_[p + beta * q, beta])
    return result


def _distinct_combination_count(rows):
    """Number of tuples with distinct physical branch IDs (two or three rows)."""
    sets = [set(row) for row in rows]
    count = math.prod(len(row) for row in rows)
    if len(rows) == 2:
        return count - len(sets[0] & sets[1])
    return (count - len(sets[0] & sets[1]) * len(rows[2])
            - len(sets[0] & sets[2]) * len(rows[1])
            - len(sets[1] & sets[2]) * len(rows[0])
            + 2 * len(sets[0] & sets[1] & sets[2]))


def _permuted_combinations(rows, rng):
    """Visit a Cartesian product without exhausting the first row first.

    A coprime stride gives a bijection of the complete finite product. Its
    initial value changes all tuple coordinates together; row shuffles remove
    dependence on map object IDs. This affects initial-search coverage only.
    """
    rows = [list(row) for row in rows]
    for row in rows:
        rng.shuffle(row)
    sizes = [len(row) for row in rows]
    count = math.prod(sizes)
    stride = 1 + sum(math.prod(sizes[k:]) for k in range(1, len(sizes)))
    while math.gcd(stride, count) != 1:
        stride += 1
    offset = int(rng.integers(count)) if count else 0
    for index in range(count):
        flat = (offset + index * stride) % count
        digits = []
        for size in reversed(sizes):
            digits.append(flat % size)
            flat //= size
        values = tuple(row[digit] for row, digit in zip(rows, reversed(digits)))
        if len(set(values)) == len(values):
            yield values


def _seed_combination_schedule(problem, rng):
    """Fair finite coverage of SS, SD and DDD continuous equation systems.

    Two diffraction ranges cannot seed all three unknowns, so DD pairs never
    consume this budget. Within a type, observation combinations and propagation
    orders take turns. SS gets half the weighted turns; SD and DDD each get a
    quarter while available. Unused capacity automatically goes to other types.
    """
    specular, diffracted = [], []
    for i in range(problem.n):
        spec_groups, diff_groups = {}, {}
        for j in np.flatnonzero(problem.allowed[i]):
            pattern = tuple(kind for kind, _ in problem.hypotheses[j].interactions)
            groups = spec_groups if np.isnan(problem.fixed_angles[j]) else diff_groups
            groups.setdefault(pattern, []).append(int(j))
        specular.append(spec_groups)
        diffracted.append(diff_groups)
    types = ("specular_specular", "specular_diffraction", "diffraction_triple")
    groups_by_type = {kind: [] for kind in types}
    family_reports = []

    def register(kind, observations, patterns, rows):
        possible = _distinct_combination_count(rows)
        if possible <= 0:
            return
        record = {"type": kind, "observation_indices": list(observations),
                  "propagation_patterns": [list(pattern) for pattern in patterns],
                  "possible_combinations": possible, "attempted_combinations": 0}
        family_reports.append(record)
        iterator = _permuted_combinations(rows, rng)
        groups_by_type[kind].append((tuple(observations), tuple(patterns), iterator, record))

    for i, k in combinations(range(problem.n), 2):
        for (p, first), (q, second) in product(specular[i].items(), specular[k].items()):
            register(types[0], (i, k), (p, q), (first, second))
    for i in range(problem.n):
        for k in range(problem.n):
            if i == k:
                continue
            for (p, first), (q, second) in product(specular[i].items(), diffracted[k].items()):
                register(types[1], (i, k), (p, q), (first, second))
    for i, k, m in combinations(range(problem.n), 3):
        for (p, first), (q, second), (r, third) in product(
                diffracted[i].items(), diffracted[k].items(), diffracted[m].items()):
            register(types[2], (i, k, m), (p, q, r), (first, second, third))
    queues = {}
    for kind, groups in groups_by_type.items():
        # Lower orders receive the first turn. Every active family then receives
        # one attempt per round, so a large first observation pair cannot starve
        # another pair or interaction pattern.
        groups.sort(key=lambda group: (sum(map(len, group[1])), group[0], group[1]))
        queues[kind] = deque(groups)
    type_reports = {
        kind: {"possible_combinations": sum(row["possible_combinations"]
                                             for row in family_reports if row["type"] == kind),
               "attempted_combinations": 0}
        for kind in types
    }
    report = {
        "policy": "weighted_types_round_robin_observations_and_orders_permuted_product",
        "type_turn_weights": {types[0]: 2, types[1]: 1, types[2]: 1},
        "types": type_reports, "families": family_reports,
        "diffraction_pair_attempts": 0,
        "diffraction_pairs_cannot_initialize_three_unknowns": True,
    }

    def generate():
        checked = 0
        cycle = (types[0], types[1], types[2], types[0])
        while any(queues.values()) and checked < problem.config.max_seed_combinations:
            for kind in cycle:
                queue = queues[kind]
                while queue:
                    observations, patterns, iterator, record = queue.popleft()
                    try:
                        branches = next(iterator)
                    except StopIteration:
                        continue
                    queue.append((observations, patterns, iterator, record))
                    record["attempted_combinations"] += 1
                    type_reports[kind]["attempted_combinations"] += 1
                    checked += 1
                    yield kind, tuple(zip(observations, branches))
                    break
                if checked >= problem.config.max_seed_combinations:
                    break
    return generate(), report


def _make_starts(problem, initial_states):
    config = problem.config
    rng = np.random.default_rng(config.seed)
    candidates, seen = [], set()
    proposals_seen = 0

    def add(state, origin):
        nonlocal proposals_seen
        state = np.asarray(state, float)
        if not problem.in_bounds(state):
            return
        key = tuple(np.round(state, 7))
        if key in seen:
            return
        seen.add(key)
        proposals_seen += 1
        # Bounded unbiased reservoir; numerical duplicate removal is only for
        # initial states, never a clustering or representative constraint.
        capacity = max(config.max_starts * 4, 32)
        if len(candidates) < capacity:
            candidates.append((state, origin))
        else:
            slot = int(rng.integers(proposals_seen))
            if slot < capacity:
                candidates[slot] = (state, origin)

    explicit = []
    if initial_states is not None:
        for state in initial_states:
            state = np.asarray(state, float)
            if not problem.in_bounds(state):
                raise ValueError("显式初值必须是允许范围内的有限 (x,y,beta)")
            explicit.append((state, "explicit_continuous_state"))
    schedule, coverage = _seed_combination_schedule(problem, rng)
    for kind, entries in schedule:
        roots = (_diffraction_roots(problem, entries) if kind == "diffraction_triple"
                 else _pair_roots(problem, *entries))
        for state in roots:
            add(state, f"continuous_{kind}_equations")
    counts = coverage["types"]
    checked = sum(item["attempted_combinations"] for item in counts.values())
    pair_checked = sum(counts[key]["attempted_combinations"]
                       for key in ("specular_specular", "specular_diffraction"))
    pair_total = sum(counts[key]["possible_combinations"]
                     for key in ("specular_specular", "specular_diffraction"))
    triple_total = counts["diffraction_triple"]["possible_combinations"]
    ranked = []
    print(f"[continuous_solver] 检查初值路径合法性：0/{len(candidates)}", flush=True)
    for candidate_index, (state, origin) in enumerate(candidates, 1):
        assignment, score, _ = problem.assignment(state)
        ranked.append((-len(assignment), score, tuple(state), origin))
        if candidate_index % 64 == 0 or candidate_index == len(candidates):
            print(f"[continuous_solver] 检查初值路径合法性：{candidate_index}/{len(candidates)}", flush=True)
    ranked.sort()
    domain_count = min(16, max(1, config.max_starts // 4))
    analytic_limit = max(0, config.max_starts - len(explicit) - domain_count)
    starts = explicit[:config.max_starts]
    starts.extend((np.asarray(state), origin) for _, _, state, origin in ranked[:analytic_limit])
    low, span = problem.bounds[:, 0], np.diff(problem.bounds, axis=1)[:, 0]
    domain = [(np.full(3, 0.5), "domain_center"),
              (np.array([0.5, 0.5, 0.0]), "domain_beta_lower_endpoint"),
              (np.array([0.5, 0.5, 1.0]), "domain_beta_upper_endpoint")]
    for index in range(1, config.max_starts + 1):
        domain.append((np.array([_halton(index, p) for p in (2, 3, 5)]),
                       "permitted_domain_halton"))
    for unit, origin in domain:
        if len(starts) >= config.max_starts:
            break
        state = low + unit * span
        if not any(np.linalg.norm(state - old) < 1e-7 for old, _ in starts):
            starts.append((state, origin))
    return starts, {
        "continuous_seed_combinations_checked": checked,
        "pair_combinations_checked": pair_checked,
        "pair_combinations_possible": pair_total,
        "diffraction_triple_combinations_possible": triple_total,
        "seed_combination_budget_exhausted": checked < pair_total + triple_total,
        "combination_coverage": coverage,
        "distinct_in_bounds_analytic_states_seen": proposals_seen,
        "analytic_states_retained_for_ranking": len(candidates),
        "initial_state_source": "continuous_constraints_and_public_domain",
        "legacy_candidates_used": False,
    }


def _candidate(problem, state, *, converged=True, termination="state_validation"):
    assignment, score, evaluations = problem.assignment(state)
    residuals, jac = problem.residual_jacobian(state, assignment)
    matrix = jac.reshape(-1, 3)
    singular = np.linalg.svd(matrix, compute_uv=False) if len(matrix) else np.array([])
    threshold = max(float(singular[0]) * problem.config.rank_relative_tolerance, 1e-12) if singular.size else 1e-12
    rank = int(np.sum(singular > threshold))
    condition = float(singular[0] / singular[2]) if rank == 3 else None
    reasons = []
    if not problem.in_bounds(state):
        reasons.append("outside_permitted_domain")
    if len(assignment) < problem.config.min_observations:
        reasons.append("insufficient_observations")
    if rank < 3:
        reasons.append("insufficient_independent_physical_constraints")
    if condition is not None and condition > problem.config.max_condition_number:
        reasons.append("ill_conditioned_physical_constraints")
    if not converged:
        reasons.append("local_search_not_converged")
    covariance = None
    if rank == 3:
        conditional = np.linalg.pinv(matrix.T @ matrix, rcond=problem.config.rank_relative_tolerance ** 2)
        covariance = {
            "matrix_xy_beta_m2": conditional.tolist(),
            "variables": ["x_m", "y_m", "beta_m"],
            "conditional_on_selected_branches": True,
            "statistically_calibrated": False,
            "assumed_scales": "engineering_scales_not_measured_noise_covariance",
            "includes_branch_ambiguity": False,
        }
    selected = []
    for (i, j), residual in zip(assignment, residuals):
        value = evaluations[j]
        h = problem.hypotheses[j]
        selected.append({
            "observation_id": problem.observations[i].observation_id,
            "observation_index": i, "hypothesis_id": h.hypothesis_id,
            "hypothesis_index": j, "propagation_interactions": [list(item) for item in h.interactions],
            "predicted_length_m": float(value.length_m), "predicted_aoa_rad": float(value.aoa_rad),
            "angle_residual_rad": float(residual[0] * problem.sa[i]),
            "length_residual_m": float(residual[1] * problem.sl[i]),
            "normalized_residual": residual.tolist(),
            "normalized_residual_norm": float(np.linalg.norm(residual)),
            "path_valid": bool(value.valid), "invalid_reason": value.invalid_reason,
            "interaction_points_m": ([list(point) for point in value.path.interaction_points_m]
                                      if value.path is not None else []),
        })
    assigned = {i for i, _ in assignment}
    return {
        "position_m": np.asarray(state[:2]).tolist(), "beta_m": float(state[2]),
        "clock_bias_s": float(state[2] / problem.config.speed_of_light_mps),
        "objective": float(score), "selected_paths": selected,
        "unmatched_observation_ids": [o.observation_id for i, o in enumerate(problem.observations) if i not in assigned],
        "matched_observation_count": len(assignment), "physical_rank": rank,
        "physical_jacobian_xy_beta": matrix.tolist(), "jacobian_singular_values": singular.tolist(),
        "condition_number": condition, "covariance": covariance,
        "all_selected_paths_valid": bool(assignment) and all(row["path_valid"] for row in selected),
        "acceptable": not reasons, "rejection_reasons": reasons,
        "converged": converged, "termination": termination,
        "validation_scope": "selected_paths_and_raw_peaks_only",
        "full_path_set_validation": "not_performed",
    }


def evaluate_continuous_state(bank, config: ContinuousSolverConfig,
                              state: Sequence[float]) -> dict[str, Any]:
    """Apply the same physical residual/rank checks to any numeric state.

    Re-matches against this bank; it does not validate a historical solver's
    selected representatives. Useful for fair comparison of frozen estimates.
    """
    problem = _Problem(bank, config)
    value = np.asarray(state, float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("待验证状态必须是有限 (x,y,beta)")
    return _candidate(problem, value)


def _structural_rank_upper_bound(problem):
    """Conservative algebraic bound, independent of the finite starts tried."""
    active = np.flatnonzero(np.any(problem.allowed, axis=0))
    active_observations = np.flatnonzero(np.any(problem.allowed, axis=1))
    if not len(active):
        return 0
    row_count = 0
    for i in active_observations:
        allowed = np.flatnonzero(problem.allowed[i])
        row_count += 2 if np.any(np.isnan(problem.fixed_angles[allowed])) else 1
    route_row_count = sum(2 if np.isnan(problem.fixed_angles[j]) else 1 for j in active)
    upper = min(3, row_count, route_row_count)
    if np.all(~np.isnan(problem.fixed_angles[active])):
        centers = [problem.a[j].T @ (problem.anchors[j] - problem.t[j]) for j in active]
        unique_centers = {tuple(center) for center in centers}
        upper = min(upper, len(unique_centers))
    return upper


def solve_continuous_position_and_bias(bank, config: ContinuousSolverConfig,
                                       *, initial_states=None) -> ContinuousSolverResult:
    """Jointly estimate location and the one shared clock bias from raw peaks."""
    problem = _Problem(bank, config)
    print(f"[continuous_solver] 构造连续初值：观测 {problem.n} 条，传播函数 {problem.h} 个", flush=True)
    starts, seed_report = _make_starts(problem, initial_states)
    print(f"[continuous_solver] 开始联合优化：初值 0/{len(starts)}", flush=True)
    candidates, records = [], []
    any_legal_path = False
    for start_index, (initial, origin) in enumerate(starts):
        state = initial.copy()
        converged = False
        termination = "association_budget"
        total_iterations = 0
        old_signature = None
        for association_iteration in range(config.max_association_iterations):
            assignment, _, _ = problem.assignment(state)
            if len(assignment) < config.min_observations:
                # A capped outlier objective is flat far from every fit. Explore
                # a fixed legal explanation first, then return to capped matching.
                assignment, _, _ = problem.assignment(state, exploratory=True)
            any_legal_path |= bool(assignment)
            if not assignment:
                termination = "no_legal_assignment_at_start"
                break
            new_state, fit_converged, fit_reason, iterations = problem.local_fit(state, assignment)
            total_iterations += iterations
            updated, _, _ = problem.assignment(new_state)
            movement = float(np.linalg.norm(new_state - state))
            state = new_state
            signature = tuple(updated)
            if signature == assignment and fit_converged:
                converged, termination = True, fit_reason
                break
            if (signature == old_signature and movement <= config.convergence_step_m):
                converged, termination = fit_converged, fit_reason
                break
            old_signature = signature
            termination = fit_reason
        candidate = _candidate(problem, state, converged=converged, termination=termination)
        candidate["initial_state_index"] = start_index
        records.append({"initial_state_index": start_index, "origin": origin,
                        "initial_state_xy_beta_m": initial.tolist(),
                        "final_state_xy_beta_m": state.tolist(),
                        "objective": candidate["objective"], "iterations": total_iterations,
                        "termination": termination, "converged": converged,
                        "physical_rank": candidate["physical_rank"],
                        "matched_observation_count": candidate["matched_observation_count"]})
        candidates.append(candidate)
        if (start_index + 1) % 16 == 0 or start_index + 1 == len(starts):
            accepted_count = sum(item["acceptable"] for item in candidates)
            print(f"[continuous_solver] 已完成初值 {start_index + 1}/{len(starts)}，"
                  f"当前残差代价 {candidate['objective']:.6g}，"
                  f"独立约束秩 {candidate['physical_rank']}/3，"
                  f"已找到满足接受条件的初值结果 {accepted_count} 个", flush=True)
    # Prefer an acceptable solution; failing candidates are still preserved in
    # the best-candidate record if no acceptable solution exists.
    candidates.sort(key=lambda item: (not item["acceptable"], item["objective"],
                                      -item["physical_rank"], -item["matched_observation_count"]))
    best = candidates[0] if candidates else None
    distinct = []
    for candidate in candidates:
        if not candidate["acceptable"]:
            continue
        if all(np.linalg.norm(np.subtract(candidate["position_m"], other["position_m"]))
               >= config.distinct_position_m
               or abs(candidate["beta_m"] - other["beta_m"]) >= config.distinct_bias_m
               for other in distinct):
            distinct.append(candidate)
    competitive = ([candidate for candidate in distinct[1:]
                    if candidate["objective"] <= distinct[0]["objective"] + config.ambiguity_cost_tolerance]
                   if distinct else [])
    structural_rank_bound = _structural_rank_upper_bound(problem)
    bank_search_incomplete = bool(bank.search_report.get("budget_exhausted", False))
    if best is not None and best["acceptable"]:
        status = "ambiguous" if competitive else "success"
    elif not problem.h or not np.any(problem.allowed):
        status = "search_exhausted" if bank_search_incomplete else "no_valid_branches"
    elif problem.n < config.min_observations:
        status = "insufficient_constraints"
    elif not any_legal_path:
        status = "search_exhausted"
    elif structural_rank_bound < 3 and not bank_search_incomplete:
        status = "insufficient_constraints"
    else:
        status = "search_exhausted"
    # Finite starts never certify absence of another solution, even when every
    # enumerated branch has been constructed. Keep this evidence boundary explicit.
    diagnostics = {
        "config": asdict(config), "observation_count": problem.n,
        "hypothesis_count": problem.h, "hypothesis_search": dict(bank.search_report),
        "seed_search": seed_report, "starts_attempted": len(starts), "starts": records,
        "path_checks": problem.path_checks, "invalid_path_checks": problem.invalid_checks,
        "invalid_path_reason_counts": problem.invalid_reason_counts,
        "backtracked_steps": problem.backtracked_steps, "any_legal_path_seen": any_legal_path,
        "acceptable_distinct_solutions": len(distinct), "competitive_alternative_count": len(competitive),
        "global_optimality_certified": False, "full_path_set_validation": "not_performed",
        "success_meaning": "converged_selected_path_fit_with_full_local_physical_rank",
        "finite_start_budget": True,
        "physical_rank_upper_bound_for_built_bank": structural_rank_bound,
        "structural_insufficiency_certified_for_built_bank": structural_rank_bound < 3,
        "hypothesis_search_incomplete": bank_search_incomplete,
    }
    usable = best if status == "success" else None
    alternatives = tuple((distinct[1:] if distinct else [])[:config.max_alternatives])
    print(f"[continuous_solver] 完成：状态 {status}，初值 {len(starts)}/{len(starts)}，"
          f"不同的可接受解 {len(distinct)} 个", flush=True)
    return ContinuousSolverResult(
        status=status, position_m=tuple(usable["position_m"]) if usable else None,
        beta_m=usable["beta_m"] if usable else None,
        clock_bias_s=usable["clock_bias_s"] if usable else None,
        selected_paths=tuple(usable["selected_paths"]) if usable else (),
        best_candidate=best, alternatives=alternatives, diagnostics=diagnostics,
    )
