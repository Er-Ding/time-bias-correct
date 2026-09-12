"""固定高度、竖直墙边缘的一次绕射几何；不计算 UTD 复振幅。

传播顺序由 (类型, 地图对象编号) 明确表示。角度扇面只在墙角的阴影侧
采样；每条路径的各段仍须通过地图遮挡检查。在线不读取生成路径或 UE。
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from functools import lru_cache
import json
import math
from typing import Sequence

import numpy as np

from .raytrace2d import (
    GeometricPath2D, _backtrack_reflection_points, _segment_visible,
    enumerate_specular_paths,
)
from .scene import Scene2D, cross_2d

InteractionSequence = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class DiffractionEdge2D:
    edge_id: str
    position_m: tuple[float, float]
    incident_wall_ids: tuple[str, ...]


@lru_cache(maxsize=8)
def diffraction_edges(scene: Scene2D) -> tuple[DiffractionEdge2D, ...]:
    """提取墙线端点；排除共线接缝、多墙交汇和墙段内部的交汇点。"""
    groups: dict[tuple[float, float], list] = {}
    for wall in scene.walls:
        for point in (wall.start, wall.end):
            key = tuple(float(x) for x in np.round(point, 7))
            groups.setdefault(key, []).append(wall)
    edges = []
    starts = np.asarray([wall.start_m for wall in scene.walls], float).reshape(-1, 2)
    vectors = np.asarray([wall.vector for wall in scene.walls], float).reshape(-1, 2)
    vector_lengths_squared = np.sum(vectors * vectors, axis=1)
    wall_indices = {wall.wall_id: index for index, wall in enumerate(scene.walls)}
    for point, walls in sorted(groups.items()):
        if len(walls) > 2:
            continue
        q = np.asarray(point)
        directions = []
        for wall in walls:
            other = wall.end if np.linalg.norm(q - wall.start) < 1e-6 else wall.start
            directions.append((other - q) / np.linalg.norm(other - q))
        if len(walls) == 2 and abs(float(np.dot(*directions))) > 1 - 1e-8:
            continue
        # T 形接点不能作为独立的自由边缘。
        fractions = np.sum((q - starts) * vectors, axis=1) / vector_lengths_squared
        interior = ((fractions > 1e-7) & (fractions < 1 - 1e-7)
                    & (np.linalg.norm(q - starts - fractions[:, None] * vectors, axis=1) < 1e-6))
        interior[[wall_indices[wall.wall_id] for wall in walls]] = False
        interior_join = bool(np.any(interior))
        if interior_join:
            continue
        ids = tuple(sorted(wall.wall_id for wall in walls))
        digest = sha256(json.dumps([point, ids], separators=(",", ":")).encode()).hexdigest()[:16]
        edges.append(DiffractionEdge2D(f"edge_{digest}", point, ids))
    return tuple(edges)


@lru_cache(maxsize=8)
def _wall_lookup(scene: Scene2D):
    return {wall.wall_id: wall for wall in scene.walls}


def shadow_directions(scene: Scene2D, edge: DiffractionEdge2D,
                      toward_previous: np.ndarray, toward_next: np.ndarray) -> bool:
    """局部绕角阴影检查，与末段长度无关，因而随时间偏差变化仍成立。"""
    q = np.asarray(edge.position_m)
    lookup = _wall_lookup(scene)
    local = tuple(lookup[key] for key in edge.incident_wall_ids)
    # 不把沿墙面或穿过绕射点的退化直线算作绕射。
    for direction in (toward_previous, toward_next):
        for wall in local:
            if abs(cross_2d(wall.vector, direction)) < 1e-8 * wall.length_m:
                return False
    if np.linalg.norm(toward_previous + toward_next) < 1e-7:
        return False
    scale = min(wall.length_m for wall in local) * 1e-3
    from .raytrace2d import _segment_intersection_parameters
    for wall in local:
        hit = _segment_intersection_parameters(q + scale * toward_previous,
                                               q + scale * toward_next, wall)
        if hit is not None and 1e-7 < hit[0] < 1 - 1e-7:
            return True
    return False


def reflection_leg(scene: Scene2D, source: np.ndarray, receiver: np.ndarray,
                   wall_ids: Sequence[str], *, source_walls=(), receiver_walls=()):
    lookup = _wall_lookup(scene)
    if any(key not in lookup for key in wall_ids):
        return None
    walls = [lookup[key] for key in wall_ids]
    if any(a == b for a, b in zip(wall_ids, wall_ids[1:])):
        return None
    points = _backtrack_reflection_points(source, receiver, walls)
    if points is None:
        return None
    nodes = [source, *points, receiver]
    for i, (start, end) in enumerate(zip(nodes[:-1], nodes[1:])):
        if np.linalg.norm(end - start) <= 1e-7:
            return None
        allowed = list(source_walls if i == 0 else (wall_ids[i - 1],))
        allowed.extend(receiver_walls if i == len(walls) else (wall_ids[i],))
        if not _segment_visible(scene, start, end, allowed_endpoint_walls=allowed):
            return None
    return points


def rebuild_path(scene: Scene2D, source_m: Sequence[float], receiver_m: Sequence[float],
                 interactions: InteractionSequence) -> GeometricPath2D | None:
    """按明确的 UE→BS 顺序重新计算反射点和绕射点，验证各段可见。"""
    source, receiver = np.asarray(source_m, float), np.asarray(receiver_m, float)
    if not scene.contains(source) or not scene.contains(receiver):
        return None
    if any(kind not in {"reflection", "diffraction"} for kind, _ in interactions):
        return None
    d_indices = [i for i, (kind, _) in enumerate(interactions) if kind == "diffraction"]
    if len(d_indices) > 1:
        return None
    if not d_indices:
        points = reflection_leg(scene, source, receiver, [key for _, key in interactions])
        if points is None:
            return None
    else:
        index = d_indices[0]
        edge = {e.edge_id: e for e in diffraction_edges(scene)}.get(interactions[index][1])
        if edge is None:
            return None
        q = np.asarray(edge.position_m)
        before = reflection_leg(scene, source, q, [key for _, key in interactions[:index]],
                                receiver_walls=edge.incident_wall_ids)
        after = reflection_leg(scene, q, receiver, [key for _, key in interactions[index + 1:]],
                               source_walls=edge.incident_wall_ids)
        if before is None or after is None:
            return None
        previous = before[-1] if before else source
        following = after[0] if after else receiver
        if not shadow_directions(scene, edge, (previous - q) / np.linalg.norm(previous - q),
                                  (following - q) / np.linalg.norm(following - q)):
            return None
        points = [*before, q, *after]
    nodes = np.asarray([source, *points, receiver])
    arrival = nodes[-2] - receiver
    return GeometricPath2D(
        path_id=json.dumps(interactions, separators=(",", ":")),
        source_m=tuple(source), receiver_m=tuple(receiver),
        interaction_wall_ids=tuple(key for kind, key in interactions if kind == "reflection"),
        interaction_points_m=tuple(tuple(point) for point in points),
        length_m=float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()),
        arrival_aoa_deg=math.degrees(math.atan2(arrival[1], arrival[0])),
        propagation_interactions=interactions,
    )


def reflection_sequences(scene: Scene2D, max_reflections: int):
    for order in range(max_reflections + 1):
        for sequence in product((wall.wall_id for wall in scene.walls), repeat=order):
            if not any(a == b for a, b in zip(sequence, sequence[1:])):
                yield sequence


def enumerate_paths(scene: Scene2D, source_m, receiver_m, *, max_reflections=2,
                    max_diffractions=0) -> list[GeometricPath2D]:
    if isinstance(max_diffractions, bool) or max_diffractions not in (0, 1):
        raise ValueError("当前只支持最多一次绕射")
    paths = enumerate_specular_paths(scene, source_m, receiver_m, max_reflections=max_reflections)
    if max_diffractions:
        for edge in diffraction_edges(scene):
            for sequence in reflection_sequences(scene, max_reflections):
                for index in range(len(sequence) + 1):
                    interactions = (tuple(("reflection", key) for key in sequence[:index])
                                    + (("diffraction", edge.edge_id),)
                                    + tuple(("reflection", key) for key in sequence[index:]))
                    path = rebuild_path(scene, source_m, receiver_m, interactions)
                    if path is not None:
                        paths.append(path)
    return sorted(paths, key=lambda p: (p.reflection_order + p.diffraction_order, p.length_m, p.path_id))
