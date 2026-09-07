"""用于二维闭环验证的镜面路径枚举。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Sequence

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .scene import Scene2D, WallSegment, cross_2d, reflect_point


_EPS = 1e-7


@dataclass(frozen=True)
class GeometricPath2D:
    """一条从 UE 到 BS 的二维传播路径。"""

    path_id: str
    source_m: tuple[float, float]
    receiver_m: tuple[float, float]
    interaction_wall_ids: tuple[str, ...]
    interaction_points_m: tuple[tuple[float, float], ...]
    length_m: float
    arrival_aoa_deg: float

    @property
    def delay_s(self) -> float:
        return self.length_m / SPEED_OF_LIGHT_M_S

    @property
    def reflection_order(self) -> int:
        return len(self.interaction_wall_ids)

    @property
    def nodes(self) -> np.ndarray:
        return np.asarray(
            [self.source_m, *self.interaction_points_m, self.receiver_m], dtype=float
        )


def _segment_intersection_parameters(
    start: np.ndarray,
    end: np.ndarray,
    wall: WallSegment,
) -> tuple[float, float, np.ndarray] | None:
    direction = end - start
    wall_vec = wall.vector
    denominator = cross_2d(direction, wall_vec)
    if abs(denominator) <= 1e-12:
        return None
    offset = wall.start - start
    segment_fraction = cross_2d(offset, wall_vec) / denominator
    wall_fraction = cross_2d(offset, direction) / denominator
    if (
        segment_fraction < -_EPS
        or segment_fraction > 1.0 + _EPS
        or wall_fraction < -_EPS
        or wall_fraction > 1.0 + _EPS
    ):
        return None
    point = start + segment_fraction * direction
    return float(segment_fraction), float(wall_fraction), point


def _segment_visible(
    scene: Scene2D,
    start: np.ndarray,
    end: np.ndarray,
    *,
    allowed_endpoint_walls: Iterable[str] = (),
) -> bool:
    allowed = set(allowed_endpoint_walls)
    for wall in scene.walls:
        hit = _segment_intersection_parameters(start, end, wall)
        if hit is None:
            continue
        fraction, _, _ = hit
        if wall.wall_id in allowed and (fraction <= _EPS or fraction >= 1.0 - _EPS):
            continue
        if _EPS < fraction < 1.0 - _EPS:
            return False
        if wall.wall_id not in allowed:
            return False
    return True


def _backtrack_reflection_points(
    source: np.ndarray,
    receiver: np.ndarray,
    walls: Sequence[WallSegment],
) -> list[np.ndarray] | None:
    images = [source]
    for wall in walls:
        images.append(reflect_point(images[-1], wall))

    current = receiver
    points_reversed: list[np.ndarray] = []
    for wall_index in range(len(walls) - 1, -1, -1):
        image = images[wall_index + 1]
        hit = _segment_intersection_parameters(current, image, walls[wall_index])
        if hit is None:
            return None
        fraction, wall_fraction, point = hit
        if not (_EPS < fraction < 1.0 - _EPS):
            return None
        if not (_EPS < wall_fraction < 1.0 - _EPS):
            return None
        points_reversed.append(point)
        current = point
    return list(reversed(points_reversed))


def _is_path_visible(
    scene: Scene2D,
    source: np.ndarray,
    receiver: np.ndarray,
    walls: Sequence[WallSegment],
    points: Sequence[np.ndarray],
) -> bool:
    nodes = [source, *points, receiver]
    for index, (start, end) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        allowed: list[str] = []
        if index > 0:
            allowed.append(walls[index - 1].wall_id)
        if index < len(walls):
            allowed.append(walls[index].wall_id)
        if not _segment_visible(scene, start, end, allowed_endpoint_walls=allowed):
            return False
    return True


def _build_path(
    source: np.ndarray,
    receiver: np.ndarray,
    walls: Sequence[WallSegment],
    points: Sequence[np.ndarray],
) -> GeometricPath2D:
    nodes = np.asarray([source, *points, receiver], dtype=float)
    segment_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    previous = nodes[-2]
    arrival_direction = previous - receiver
    arrival_direction /= np.linalg.norm(arrival_direction)
    aoa_deg = math.degrees(math.atan2(arrival_direction[1], arrival_direction[0]))
    topology = "los" if not walls else "-".join(wall.wall_id for wall in walls)
    return GeometricPath2D(
        path_id=topology,
        source_m=(float(source[0]), float(source[1])),
        receiver_m=(float(receiver[0]), float(receiver[1])),
        interaction_wall_ids=tuple(wall.wall_id for wall in walls),
        interaction_points_m=tuple((float(p[0]), float(p[1])) for p in points),
        length_m=float(np.sum(segment_lengths)),
        arrival_aoa_deg=float(aoa_deg),
    )


def enumerate_specular_paths(
    scene: Scene2D,
    source_m: Sequence[float],
    receiver_m: Sequence[float],
    *,
    max_reflections: int = 2,
) -> list[GeometricPath2D]:
    """枚举直射及最多二次镜面反射路径。

    该实现用于离线闭环和单元测试。Sionna RT 数据进入系统后，正向路径由
    Sionna 计算，定位端仍只使用同一份二维墙线。
    """

    if max_reflections not in (0, 1, 2):
        raise ValueError("第一版只支持最多二次反射")
    source = np.asarray(source_m, dtype=float)
    receiver = np.asarray(receiver_m, dtype=float)
    if source.shape != (2,) or receiver.shape != (2,):
        raise ValueError("source_m 和 receiver_m 必须是二维坐标")
    if not scene.contains(source) or not scene.contains(receiver):
        raise ValueError("UE 和 BS 必须位于场景边界内")

    paths: list[GeometricPath2D] = []
    if _segment_visible(scene, source, receiver):
        paths.append(_build_path(source, receiver, (), ()))

    for order in range(1, max_reflections + 1):
        for wall_sequence in product(scene.walls, repeat=order):
            if any(
                wall_sequence[idx].wall_id == wall_sequence[idx + 1].wall_id
                for idx in range(order - 1)
            ):
                continue
            points = _backtrack_reflection_points(source, receiver, wall_sequence)
            if points is None:
                continue
            if not _is_path_visible(scene, source, receiver, wall_sequence, points):
                continue
            paths.append(_build_path(source, receiver, wall_sequence, points))

    paths.sort(key=lambda path: (path.reflection_order, path.length_m, path.path_id))
    return paths

