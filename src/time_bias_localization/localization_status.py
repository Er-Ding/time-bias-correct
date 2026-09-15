"""没有位置输出的正常终态；计算未完成与物理约束不足分别记录。"""

NO_POSITION_STATUSES = frozenset({
    "unlocalizable", "detection_incomplete", "solver_budget_exhausted", "excluded_observation",
    "ambiguous", "geometry_failed",
})
COMPLETED_WORKER_STATUSES = NO_POSITION_STATUSES | {"success"}
