"""公共地图和 BS 的绕射前缀表；批量实现与标量镜像法使用相同几何判据。"""
from __future__ import annotations

from functools import lru_cache
import math
import time
import numpy as np

from .diffraction import diffraction_edges

_EPS = 1e-7


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _intersection(start, end, wall_start, wall_vector):
    direction = end - start
    offset = wall_start - start
    denominator = _cross(direction, wall_vector)
    valid = np.abs(denominator) > 1e-12
    fraction = np.zeros(np.broadcast_shapes(direction.shape[:-1], wall_vector.shape[:-1]))
    wall_fraction = np.zeros_like(fraction)
    np.divide(_cross(offset, wall_vector), denominator, out=fraction, where=valid)
    np.divide(_cross(offset, direction), denominator, out=wall_fraction, where=valid)
    valid &= ((fraction > _EPS) & (fraction < 1 - _EPS)
              & (wall_fraction > _EPS) & (wall_fraction < 1 - _EPS))
    return valid, start + fraction[..., None] * direction


class PrefixGeometry:
    def __init__(self, scene):
        self.scene = scene
        self.walls = scene.walls
        self.starts = np.asarray([w.start_m for w in self.walls], float).reshape(-1, 2)
        self.vectors = np.asarray([w.vector for w in self.walls], float).reshape(-1, 2)
        self.normals = np.asarray([w.normal for w in self.walls], float).reshape(-1, 2)
        self.wall_index = {w.wall_id: i for i, w in enumerate(self.walls)}
        self._image_cache = None

    def image_sources(self, source, first_indices, max_reflections):
        key = (tuple(source), tuple(first_indices), max_reflections)
        if self._image_cache is None or self._image_cache[0] != key:
            first = self.reflect(source, first_indices)
            # Keep this public geometry cache bounded even for much larger maps.
            second = (self.reflect(first[:, None, :], np.arange(len(self.walls)))
                      if max_reflections == 2 and len(first_indices) * len(self.walls) <= 2_000_000
                      else None)
            self._image_cache = (key, first, second)
        return self._image_cache[1:]

    def reflect(self, points, wall_indices):
        normals = self.normals[wall_indices]
        return points - 2 * np.sum((points - self.starts[wall_indices]) * normals, axis=-1)[..., None] * normals

    def visible(self, starts, ends, allowed):
        """逐条线段的完整遮挡检查，与 raytrace2d._segment_visible 等价。"""
        starts, ends = np.asarray(starts), np.asarray(ends)
        answer = np.ones(len(starts), bool)
        for begin in range(0, len(starts), 256):
            end = min(begin + 256, len(starts))
            direction = (ends[begin:end] - starts[begin:end])[:, None, :]
            offset = self.starts[None, :, :] - starts[begin:end, None, :]
            denominator = _cross(direction, self.vectors)
            nonparallel = np.abs(denominator) > 1e-12
            fractions = np.zeros_like(denominator)
            wall_fractions = np.zeros_like(denominator)
            np.divide(_cross(offset, self.vectors), denominator, out=fractions, where=nonparallel)
            np.divide(_cross(offset, direction), denominator, out=wall_fractions, where=nonparallel)
            hits = (nonparallel & (fractions >= -_EPS) & (fractions <= 1 + _EPS)
                    & (wall_fractions >= -_EPS) & (wall_fractions <= 1 + _EPS))
            endpoint = (fractions <= _EPS) | (fractions >= 1 - _EPS)
            for row, indices in enumerate(allowed[begin:end]):
                for index in indices:
                    hits[row, index] &= not endpoint[row, index]
            answer[begin:end] = ~np.any(hits, axis=1)
        return answer

    def visible_first_walls(self, source):
        """所有可能成为第一处镜面反射的墙，不使用观测或 UE。

        两条墙的前后遮挡次序仅在墙端点方向或墙间交点方向改变。
        将这些方向分段，并查询每段中点的最近墙，可获得必要候选集合。
        极窄角区间与贴近 BS 的退化墙直接保留，避免浮点角度分辨率删点。
        此处不判断镜面反射是否成立，后续仍做完整回溯和遮挡检查。
        """
        n = len(self.walls)
        if not n:
            return np.empty(0, int)
        endpoints = np.concatenate([self.starts, self.starts + self.vectors])
        endpoint_offsets = endpoints - source
        endpoint_angles = np.arctan2(endpoint_offsets[:, 1], endpoint_offsets[:, 0])
        critical = [endpoint_angles]
        uncertain_crossings = np.zeros(n, bool)
        lower = np.minimum(self.starts, self.starts + self.vectors)
        upper = np.maximum(self.starts, self.starts + self.vectors)
        for first in range(n):
            offsets = self.starts[first + 1:] - self.starts[first]
            vectors = self.vectors[first + 1:]
            denominator = _cross(self.vectors[first], vectors)
            # These are wall-wall intersections, not a propagation ray query.
            # A fixed ray tolerance would discard real crossings of short walls.
            valid = denominator != 0
            uncertain = (valid & (np.abs(denominator) <= 1e-12)
                         & np.all(upper[first] + _EPS >= lower[first + 1:], axis=1)
                         & np.all(upper[first + 1:] + _EPS >= lower[first], axis=1))
            if np.any(uncertain):
                uncertain_crossings[first] = True
                uncertain_crossings[first + 1:] |= uncertain
            t = np.zeros(len(offsets)); u = np.zeros(len(offsets))
            np.divide(_cross(offsets, vectors), denominator, out=t, where=valid)
            np.divide(_cross(offsets, self.vectors[first]), denominator, out=u, where=valid)
            valid &= (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            if np.any(valid):
                crossings = self.starts[first] + t[valid, None] * self.vectors[first] - source
                critical.append(np.arctan2(crossings[:, 1], crossings[:, 0]))
        angles = np.unique(np.concatenate(critical))
        extended = np.r_[angles, angles[0] + 2 * np.pi]
        midpoints = (extended[:-1] + extended[1:]) / 2
        selected = uncertain_crossings.copy()
        # Query the boundary rays too: this conservatively retains ties.
        queries = np.r_[angles, midpoints]
        offsets = self.starts - source
        numerator = _cross(offsets, self.vectors)
        for begin in range(0, len(queries), 256):
            a = queries[begin:begin + 256]
            directions = np.stack([np.cos(a), np.sin(a)], axis=-1)
            denominator = _cross(directions[:, None, :], self.vectors)
            valid = np.abs(denominator) > 1e-12
            distances = np.full_like(denominator, np.inf)
            fractions = np.full_like(denominator, np.inf)
            np.divide(numerator, denominator, out=distances, where=valid)
            np.divide(_cross(offsets, directions[:, None, :]), denominator, out=fractions, where=valid)
            valid &= (distances > 0) & (fractions >= -1e-12) & (fractions <= 1 + 1e-12)
            distances[~valid] = np.inf
            nearest = np.min(distances, axis=1, keepdims=True)
            selected |= np.any(np.isfinite(distances) & (distances <= nearest + 1e-7), axis=0)
        angular_width = np.abs((endpoint_angles[:n] - endpoint_angles[n:] + np.pi) % (2 * np.pi) - np.pi)
        selected |= angular_width < 1e-10
        selected |= np.minimum(np.linalg.norm(endpoint_offsets[:n], axis=1),
                               np.linalg.norm(endpoint_offsets[n:], axis=1)) < 1e-6
        # Almost coincident angular breakpoints are below the robust sweep resolution.
        narrow = np.flatnonzero(np.diff(extended) < 1e-10)
        signed_width = ((endpoint_angles[n:] - endpoint_angles[:n] + np.pi)
                        % (2 * np.pi) - np.pi)
        wall_middle = endpoint_angles[:n] + signed_width / 2
        for index in narrow:
            a = midpoints[index]
            distance = np.abs((wall_middle - a + np.pi) % (2 * np.pi) - np.pi)
            # Keep every wall intersecting an unresolved angular sliver, including
            # a distant wall visible only through an extremely narrow foreground gap.
            selected |= distance <= np.abs(signed_width) / 2 + 1e-10
        return np.flatnonzero(selected)

    def edge_prefixes(self, source, edge, max_reflections, first_indices):
        q = np.asarray(edge.position_m)
        incident = tuple(self.wall_index[key] for key in edge.incident_wall_ids)
        result = []
        if np.linalg.norm(q - source) > _EPS and self.visible([source], [q], [incident])[0]:
            result.append(((), ()))
        if not max_reflections or not len(first_indices):
            return result
        first_images, second_images = self.image_sources(source, first_indices, max_reflections)
        valid, p1 = _intersection(q, first_images, self.starts[first_indices], self.vectors[first_indices])
        indices = np.flatnonzero(valid)
        if len(indices):
            w = first_indices[indices]; p = p1[indices]
            good = (np.linalg.norm(p - source, axis=1) > _EPS) & (np.linalg.norm(p - q, axis=1) > _EPS)
            good &= self.visible(np.broadcast_to(source, p.shape), p, [(int(i),) for i in w])
            good &= self.visible(p, np.broadcast_to(q, p.shape), [(*incident, int(i)) for i in w])
            result.extend(((int(i),), (point,)) for i, point in zip(w[good], p[good]))
        if max_reflections < 2:
            return result
        # At most a few visible first walls are active in an urban map; chunk by wall
        # to bound temporary memory independently of the number of edges.
        n = len(self.walls)
        for begin in range(0, len(first_indices), 32):
            indices = first_indices[begin:begin + 32]
            image1 = first_images[begin:begin + 32]
            images2 = (second_images[begin:begin + 32] if second_images is not None
                       else self.reflect(image1[:, None, :], np.arange(n)))
            valid2, p2 = _intersection(q, images2, self.starts, self.vectors)
            valid2[np.arange(len(indices)), indices] = False
            first_rows, selected = np.nonzero(valid2)
            if not len(selected):
                continue
            i = indices[first_rows]
            second_points = p2[first_rows, selected]
            valid1, first_points = _intersection(second_points, image1[first_rows], self.starts[i], self.vectors[i])
            i = i[valid1]; selected = selected[valid1]
            second_points = second_points[valid1]; first_points = first_points[valid1]
            if not len(selected):
                continue
            good = ((np.linalg.norm(first_points - source, axis=1) > _EPS)
                    & (np.linalg.norm(second_points - first_points, axis=1) > _EPS)
                    & (np.linalg.norm(second_points - q, axis=1) > _EPS))
            # Check cheap first-segment occlusion before allocating the remaining checks.
            good &= self.visible(np.broadcast_to(source, first_points.shape), first_points,
                                 [(int(k),) for k in i])
            kept = np.flatnonzero(good)
            if not len(kept):
                continue
            i = i[kept]; j = selected[kept]; a = first_points[kept]; b = second_points[kept]
            good = self.visible(a, b, [(int(k), int(l)) for k, l in zip(i, j)])
            good &= self.visible(b, np.broadcast_to(q, b.shape), [(*incident, int(k)) for k in j])
            result.extend(((int(k), int(l)), (x, y))
                          for k, l, x, y in zip(i[good], j[good], a[good], b[good]))
        return result


@lru_cache(maxsize=4)
def _cached_prefixes(scene, source_tuple, max_reflections):
    started = last_log = time.monotonic()
    source = np.asarray(source_tuple, float)
    geometry = PrefixGeometry(scene)
    edges = diffraction_edges(scene)
    first_indices = geometry.visible_first_walls(source) if max_reflections else np.empty(0, int)
    print(f"[绕射前缀] 公共地图 {len(scene.walls)} 面墙，{len(edges)} 个边缘，"
          f"首反射候选墙 {len(first_indices)}；开始构建可复用前缀表", flush=True)
    prefixes = []
    for edge_index, edge in enumerate(edges):
        for indices, points in geometry.edge_prefixes(source, edge, max_reflections, first_indices):
            wall_ids = tuple(scene.walls[i].wall_id for i in indices)
            nodes = np.asarray([source, *points, edge.position_m])
            vectors = np.diff(nodes, axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            prefixes.append((edge, wall_ids, points, float(lengths.sum()),
                             math.atan2(vectors[0, 1], vectors[0, 0]), -vectors[-1] / lengths[-1]))
        now = time.monotonic()
        if now - last_log >= 5:
            print(f"[绕射前缀] 已处理 {edge_index + 1}/{len(edges)} 个边缘，"
                  f"保留 {len(prefixes)} 条前缀，用时 {now - started:.1f} 秒", flush=True)
            last_log = now
    print(f"[绕射前缀] 完成：{len(prefixes)} 条，用时 {time.monotonic() - started:.2f} 秒", flush=True)
    return tuple(prefixes)


def get_diffraction_prefixes(scene, source, max_reflections):
    if max_reflections not in (0, 1, 2):
        raise ValueError("当前绕射前缀只支持最多二次反射")
    before = _cached_prefixes.cache_info()
    prefixes = _cached_prefixes(scene, tuple(float(x) for x in source), int(max_reflections))
    after = _cached_prefixes.cache_info()
    return prefixes, after.hits > before.hits


def clear_prefix_cache():
    _cached_prefixes.cache_clear()
