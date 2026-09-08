"""反向射线与墙段的分块 NumPy 求交，保持标量参考实现的几何规则。"""

from __future__ import annotations

import numpy as np

from .scene import Scene2D, WallSegment, _GEOMETRY_EPS


class VectorizedWallIntersector:
    """同一场景复用墙坐标；每次查询只分配至多 wall_chunk_size 行数组。

    按墙编号稳定排序后，最小距离并列时的第一次 argmin，等价于参考实现的
    min(distance, wall_id)。不合并共线墙、不近似交点，也不改变墙端点容差。
    """

    def __init__(self, scene: Scene2D, *, wall_chunk_size: int = 8192) -> None:
        if (
            isinstance(wall_chunk_size, (bool, np.bool_))
            or not isinstance(wall_chunk_size, (int, np.integer))
            or wall_chunk_size < 1
        ):
            raise ValueError("wall_chunk_size 必须为正整数")
        self.wall_chunk_size = int(wall_chunk_size)
        self.walls = tuple(sorted(scene.walls, key=lambda wall: wall.wall_id))
        self.starts = np.asarray([wall.start_m for wall in self.walls], dtype=float).reshape(-1, 2)
        self.vectors = np.asarray([wall.vector for wall in self.walls], dtype=float).reshape(-1, 2)
        self.starts.setflags(write=False)
        self.vectors.setflags(write=False)

    def nearest(
        self, origin: np.ndarray, direction: np.ndarray,
    ) -> tuple[float, WallSegment, np.ndarray] | None:
        if not self.walls:
            return None
        # ray_segment_intersection 在每次墙求交前做同样的单位化，这里只需一次。
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= _GEOMETRY_EPS:
            raise ValueError("射线方向不能为零")
        direction = direction / norm
        best_distance = float("inf")
        best_index: int | None = None
        for start in range(0, len(self.walls), self.wall_chunk_size):
            stop = min(start + self.wall_chunk_size, len(self.walls))
            vectors = self.vectors[start:stop]
            offsets = self.starts[start:stop] - origin
            denominator = direction[0] * vectors[:, 1] - direction[1] * vectors[:, 0]
            nonparallel = np.abs(denominator) > _GEOMETRY_EPS
            distances = np.full(stop - start, np.inf, dtype=float)
            fractions = np.full(stop - start, np.inf, dtype=float)
            np.divide(
                offsets[:, 0] * vectors[:, 1] - offsets[:, 1] * vectors[:, 0],
                denominator, out=distances, where=nonparallel,
            )
            np.divide(
                offsets[:, 0] * direction[1] - offsets[:, 1] * direction[0],
                denominator, out=fractions, where=nonparallel,
            )
            valid = (
                nonparallel
                & (distances > 1e-6)
                & (fractions >= -_GEOMETRY_EPS)
                & (fractions <= 1.0 + _GEOMETRY_EPS)
            )
            distances[~valid] = np.inf
            index = int(np.argmin(distances))
            distance = float(distances[index])
            # 块内和块间均保留按墙编号排序后第一个等距离交点。
            if distance < best_distance:
                best_distance = distance
                best_index = start + index
        if best_index is None:
            return None
        point = origin + best_distance * direction
        return best_distance, self.walls[best_index], point


__all__ = ["VectorizedWallIntersector"]
