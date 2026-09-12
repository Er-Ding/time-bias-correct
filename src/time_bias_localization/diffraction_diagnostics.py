"""把绕射出射方向作为未知量检查局部可辨识性，避免采样方向制造假约束。"""
import numpy as np
from .scene import reflect_point


def physical_constraint_rank(scene, selected_candidates, position_m):
    rows, diffraction_count = [], 0
    walls = {wall.wall_id: wall for wall in scene.walls}
    position = np.asarray(position_m)
    for candidate in selected_candidates.values():
        sequence = candidate.metadata.get("propagation_interactions", [])
        d_indices = [i for i, (kind, _) in enumerate(sequence) if kind == "diffraction"]
        if not d_indices:
            rows.extend([(1., 0., candidate.direction[0]), (0., 1., candidate.direction[1])])
            continue
        diffraction_count += 1
        index = d_indices[0]
        # 将绕射点经过其后的反射墙展开；UE 到该虚拟点只有一个距离约束。
        origin = np.asarray(candidate.metadata["interaction_points_m"][index])
        for kind, key in sequence[index + 1:]:
            origin = reflect_point(origin, walls[key])
        vector = position - origin
        length = np.linalg.norm(vector)
        if length > 1e-9:
            rows.append((*tuple(vector / length), 1.))
    rank = int(np.linalg.matrix_rank(np.asarray(rows), tol=1e-8)) if rows else 0
    return {
        "selected_observation_count": len(selected_candidates),
        "selected_diffraction_observation_count": diffraction_count,
        "physical_constraint_count": len(rows), "position_and_bias_dimension": 3,
        "local_constraint_rank": rank, "locally_identifiable": rank == 3,
        "checks_global_uniqueness": False,
        "diffraction_direction_is_independently_observed": False,
        "covariance_interpretation": "conditional_on_discrete_representatives_not_calibrated_uncertainty",
    }
