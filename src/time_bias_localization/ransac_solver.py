"""有固定抽样预算的分组 RANSAC，联合估计 (x, y, beta=c*b)。

每条观测最多一票；缺失候选与超过距离门槛的候选具有相同封顶代价。
优先比较支持观测数，再比较封顶平方残差。抽样方向只是离散候选，
不会被误当作绕射提供的独立角度约束。固定次数不宣称全局搜索置信度。
"""
from collections import defaultdict
import numpy as np
from .scene import reflect_point
from .timing import stage
from .solver import (SolverError, RansacSearchError, SolverDiagnostics, SolverResult,
    _group_candidates, _weighted_fit, _fit_assignment, _assignment_signature,
    _RefinedSolution, _estimate_covariance)


def _diffraction(candidate):
    return any(kind == "diffraction" for kind, _ in candidate.metadata.get("propagation_interactions", []))


class _CandidateIndex:
    def __init__(self, groups, scene):
        self.ids = list(groups)
        self.groups = groups
        self.flat = [candidate for group in groups.values() for candidate in group]
        lengths = [len(group) for group in groups.values()]
        self.offsets = np.cumsum([0, *lengths])
        self.anchors = np.array([c.anchor_m for c in self.flat])
        self.directions = np.array([c.direction for c in self.flat])
        self.lower = np.array([c.beta_min_m for c in self.flat])
        self.upper = np.array([c.beta_max_m for c in self.flat])
        self.centers = {}
        walls = {wall.wall_id: wall for wall in scene.walls} if scene is not None else {}
        self.clusters = []
        for group in groups.values():
            clusters = defaultdict(list)
            for c in group:
                clusters[c.metadata.get("point_cluster_id", c.candidate_id)].append(c)
                if not _diffraction(c):
                    continue
                sequence = c.metadata["propagation_interactions"]
                index = next(i for i, (kind, _) in enumerate(sequence) if kind == "diffraction")
                center = np.array(c.metadata["interaction_points_m"][index], dtype=float)
                for kind, wall_id in sequence[index + 1:]:
                    if kind == "reflection":
                        if wall_id not in walls:
                            raise ValueError("含绕射后反射的 RANSAC 输入需要提供 scene")
                        center = reflect_point(center, walls[wall_id])
                self.centers[id(c)] = center
            # 同簇先采中心的概率较高；输入顺序不影响复现。
            self.clusters.append(list(clusters.values()))

    def sample(self, index, rng):
        clusters = self.clusters[index]
        cluster = clusters[int(rng.integers(len(clusters)))]
        return cluster[0] if rng.random() < 0.5 else cluster[int(rng.integers(len(cluster)))]

    def assignment(self, state, config):
        residual2 = np.sum((self.anchors - state[2] * self.directions - state[:2])**2, axis=1)
        valid = ((self.lower <= state[2] + config.validity_tolerance)
                 & (self.upper >= state[2] - config.validity_tolerance))
        residual2[~valid] = np.inf
        assignment, residuals = {}, {}
        threshold2 = config.ransac_inlier_distance_m**2
        cost = 0.0
        for observation, lo, hi in zip(self.ids, self.offsets[:-1], self.offsets[1:]):
            best = int(lo + np.argmin(residual2[lo:hi]))
            value = float(residual2[best])
            cost += min(value, threshold2)
            if value <= threshold2:
                assignment[observation] = self.flat[best]
                residuals[observation] = float(np.sqrt(value))
        return assignment, residuals, cost

    def physical_rank(self, assignment, state):
        rows = []
        for c in assignment.values():
            if id(c) not in self.centers:
                rows.extend(((1., 0., c.direction[0]), (0., 1., c.direction[1])))
            else:
                vector = state[:2] - self.centers[id(c)]
                length = np.linalg.norm(vector)
                if length > 1e-9:
                    rows.append((*tuple(vector / length), 1.))
        return int(np.linalg.matrix_rank(np.asarray(rows), tol=1e-8)) if rows else 0


def _score(assignment, cost, state):
    return (-len(assignment), cost, float(state[2]), float(state[0]), float(state[1]))


def _refine(seed, index, config):
    state = seed.copy()
    iterations = 0
    for _ in range(config.max_iterations):
        assignment, _, _ = index.assignment(state, config)
        if len(assignment) < 2 or index.physical_rank(assignment, state) < 3:
            return None
        try:
            fitted, weights, count, rank, condition, converged = _fit_assignment(assignment, state, config)
        except SolverError:
            return None
        iterations += count
        after, residuals, cost = index.assignment(fitted, config)
        stable = _assignment_signature(after) == _assignment_signature(assignment)
        state = fitted
        if stable and converged and index.physical_rank(after, state) == 3:
            return _RefinedSolution(state, after, residuals, weights, cost, iterations, True, rank, condition)
    return None


def solve_ransac(candidates, config, *, scene=None):
    groups = _group_candidates(candidates)
    if len(groups) == 2 and all(_diffraction(c) for group in groups.values() for c in group):
        raise SolverError("两条纯绕射观测至多提供两个独立距离约束，不足以联合确定 x、y、b")
    index = _CandidateIndex(groups, scene)
    rng = np.random.default_rng(config.ransac_seed)
    stats = {"method": "ransac", "random_seed": config.ransac_seed,
             "max_trials": config.ransac_max_trials, "draw_count": 0,
             "valid_hypothesis_count": 0, "sample_size_counts": {"2": 0, "3": 0},
             "inlier_distance_m": config.ransac_inlier_distance_m,
             "max_refinements": config.ransac_max_refinements,
             "sampling": "uniform_distinct_observations_then_uniform_clusters_then_center_or_uniform_member",
             "one_vote_per_observation": True, "exhaustive_pairs_materialized": False,
             "score": "maximum_inlier_observations_then_truncated_squared_distance",
             "sampling_confidence_guaranteed": False, "global_uniqueness_proven": False}
    pool = {}
    with stage("T12_ransac_sampling"):
        for _ in range(config.ransac_max_trials):
            stats["draw_count"] += 1
            observation_indices = rng.choice(len(groups), 2, replace=False).tolist()
            chosen = [index.sample(i, rng) for i in observation_indices]
            if all(_diffraction(c) for c in chosen):
                remaining = [i for i in range(len(groups)) if i not in observation_indices]
                if not remaining:
                    continue
                chosen.append(index.sample(remaining[int(rng.integers(len(remaining)))], rng))
            stats["sample_size_counts"][str(len(chosen))] += 1
            if max(c.beta_min_m for c in chosen) > min(c.beta_max_m for c in chosen):
                continue
            try:
                state, _, _ = _weighted_fit(chosen, np.array([c.weight for c in chosen]), config)
            except SolverError:
                continue
            if any(np.linalg.norm(c.point(state[2]) - state[:2]) > config.ransac_inlier_distance_m for c in chosen):
                continue
            assignment, _, cost = index.assignment(state, config)
            if len(assignment) < 2 or index.physical_rank(assignment, state) < 3:
                continue
            stats["valid_hypothesis_count"] += 1
            key = tuple(np.round(state / 0.25).astype(np.int64))
            score = _score(assignment, cost, state)
            if key not in pool or score < pool[key][0]:
                pool[key] = (score, state)
    with stage("T12_seed_scoring"):
        seeds = sorted(pool.values(), key=lambda item: item[0])[:config.ransac_max_refinements]
    stats["distinct_hypothesis_count"] = len(pool)
    with stage("T12_iterations"):
        solutions = [solution for _, seed in seeds if (solution := _refine(seed, index, config)) is not None]
    stats["refined_solution_count"] = len(solutions)
    if not solutions:
        raise RansacSearchError(stats)
    with stage("T12_result"):
        solutions.sort(key=lambda item: _score(item.assignment, item.objective, item.state))
        best = solutions[0]
        stats["inlier_observation_count"] = len(best.assignment)
        stats["outlier_observation_ids"] = [str(key) for key in groups if key not in best.assignment]
        alternatives = []
        for solution in solutions[1:]:
            if np.linalg.norm(solution.state[:2] - best.state[:2]) > config.ransac_inlier_distance_m:
                alternatives.append({"position_m": solution.state[:2].tolist(), "distance_bias_m": float(solution.state[2]),
                                     "inlier_count": len(solution.assignment), "score": solution.objective})
        stats["alternative_hypotheses"] = alternatives[:5]
        sigma, covariance = _estimate_covariance(best, config)
        diagnostics = SolverDiagnostics(converged=best.converged, iterations=best.iterations,
            seed_count=len(seeds), evaluated_solution_count=len(solutions), objective=best.objective,
            design_rank=best.rank, condition_number=best.condition_number,
            selected_observation_count=len(best.assignment), total_observation_count=len(groups),
            downweighted_observations=tuple(key for key in best.assignment
                if best.robust_weights.get(key, 1.) < config.downweight_threshold),
            unused_observations=tuple(key for key in groups if key not in best.assignment),
            robust_weights=best.robust_weights, joint_covariance=covariance, search=stats)
        return SolverResult(best.state[:2], sigma, float(best.state[2]), best.assignment, best.residuals, diagnostics)
