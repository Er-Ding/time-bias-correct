"""独立的 CUDA 绕射候选后端；公共路径表仍复用 CPU 实现。

不修改主流程默认后端。双精度批量推进完整分支，保留 CPU 候选顺序、
来源、权重、传播交互及偏差相关字段；GPU 初始化失败直接报错。
"""
from __future__ import annotations

from functools import lru_cache
import importlib
import json
import math
import os

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .diffraction import diffraction_edges
from .diffraction_prefixes import get_diffraction_prefixes
from .initial_candidates import InitialCandidateGenerationResult, InitialCandidatePoint
from .initial_candidates import generate_initial_candidate_points as generate_initial_cpu
from .timing import stage


_CUDA_SOURCE = r'''
// NVRTC provides device math intrinsics without host C library headers.
#define INFINITY (__longlong_as_double(0x7ff0000000000000LL))

__device__ double cross2(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

// Walls are sorted by wall_id. Strict '<' preserves the first equal hit.
__device__ int nearest_wall(const double* walls, int nw,
    double ox, double oy, double dx, double dy, double* distance, double* px, double* py) {
    double norm = sqrt(dx * dx + dy * dy);
    dx /= norm; dy /= norm;
    int best = -1;
    double best_t = INFINITY;
    for (int w = 0; w < nw; ++w) {
        const double* a = walls + w * 9;
        double det = cross2(dx, dy, a[2], a[3]);
        if (fabs(det) <= 1e-9) continue;
        double ax = a[0] - ox, ay = a[1] - oy;
        double t = cross2(ax, ay, a[2], a[3]) / det;
        double u = cross2(ax, ay, dx, dy) / det;
        if (t > 1e-6 && u >= -1e-9 && u <= 1.0 + 1e-9 && t < best_t) {
            best = w; best_t = t;
        }
    }
    *distance = best_t;
    *px = ox + best_t * dx; *py = oy + best_t * dy;
    return best;
}

extern "C" __global__ void nearest_many(const double* walls, int nw,
    const double* origins, const double* directions, int n, int* ids, double* result) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    ids[k] = nearest_wall(walls, nw, origins[2*k], origins[2*k+1],
        directions[2*k], directions[2*k+1], result+3*k, result+3*k+1, result+3*k+2);
}

__device__ bool in_shadow(const double* walls, const int* incident,
    const double* prefix, double dx, double dy) {
    double ux = prefix[3], uy = prefix[4];
    for (int j = 0; j < 2; ++j) {
        int wi = incident[j];
        if (wi < 0) continue;
        const double* w = walls + wi * 9;
        if (fabs(cross2(w[2], w[3], ux, uy)) < 1e-8 * w[6]
            || fabs(cross2(w[2], w[3], dx, dy)) < 1e-8 * w[6]) return false;
    }
    if (sqrt((ux+dx)*(ux+dx) + (uy+dy)*(uy+dy)) < 1e-7) return false;
    double scale = prefix[7];
    double ax = prefix[0] + scale * ux, ay = prefix[1] + scale * uy;
    double bx = prefix[0] + scale * dx, by = prefix[1] + scale * dy;
    for (int j = 0; j < 2; ++j) {
        int wi = incident[j];
        if (wi < 0) continue;
        const double* w = walls + wi * 9;
        double vx = bx-ax, vy = by-ay;
        double det = cross2(vx, vy, w[2], w[3]);
        if (fabs(det) <= 1e-12) continue;
        double rx = w[0]-ax, ry = w[1]-ay;
        double t = cross2(rx, ry, w[2], w[3]) / det;
        double u = cross2(rx, ry, vx, vy) / det;
        if (t >= -1e-7 && t <= 1.0+1e-7 && u >= -1e-7 && u <= 1.0+1e-7
            && t > 1e-7 && t < 1.0-1e-7) return true;
    }
    return false;
}

extern "C" __global__ void trace_fans(const double* walls, int nw,
    const double* prefixes, const int* incidents, const int* job_prefix,
    const double* directions, const double* targets, int n, int max_reflections,
    const double* bounds, double bsx, double bsy,
    int* accepted, double* result, int* reflection_count, int* reflection_walls,
    double* reflection_points) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= n) return;
    accepted[k] = 0;
    int pi = job_prefix[k];
    const double* p = prefixes + pi*8;
    double dx = directions[2*k], dy = directions[2*k+1];
    if (!in_shadow(walls, incidents + pi*2, p, dx, dy)) return;
    double ox = p[0], oy = p[1], length = p[2], target = targets[k];
    int added = 0, capacity = max_reflections > 0 ? max_reflections : 1;
    for (int order = (int)p[5]; order <= max_reflections; ++order) {
        double distance, hx, hy;
        int hit = nearest_wall(walls, nw, ox, oy, dx, dy, &distance, &hx, &hy);
        double bx = INFINITY, by = INFINITY;
        if (dx > 1e-12) bx = (bounds[1]-ox)/dx;
        else if (dx < -1e-12) bx = (bounds[0]-ox)/dx;
        if (dy > 1e-12) by = (bounds[3]-oy)/dy;
        else if (dy < -1e-12) by = (bounds[2]-oy)/dy;
        double boundary = fmax(0.0, fmin(bx, by));
        bool inside = hit >= 0 && distance <= boundary + 1e-7;
        double free_distance = inside ? fmin(distance, boundary) : boundary;
        double remaining = target - length;
        if (remaining > 1e-7 && remaining < free_distance - 1e-7) {
            double ex = ox + remaining*dx, ey = oy + remaining*dy;
            if (sqrt((ex-bsx)*(ex-bsx)+(ey-bsy)*(ey-bsy)) <= 1e-7) return;
            double* out = result + k*8;
            out[0]=ex; out[1]=ey; out[2]=ox; out[3]=oy;
            out[4]=dx; out[5]=dy; out[6]=length; out[7]=free_distance;
            reflection_count[k] = added;
            accepted[k] = 1;
            return;
        }
        if (remaining <= free_distance+1e-7 || !inside || order == max_reflections) return;
        const double* w = walls + hit*9;
        double d1 = sqrt((hx-w[0])*(hx-w[0])+(hy-w[1])*(hy-w[1]));
        double d2 = sqrt((hx-w[7])*(hx-w[7])+(hy-w[8])*(hy-w[8]));
        if (fmin(d1,d2) <= 1e-7) return;
        reflection_walls[k*capacity+added] = hit;
        reflection_points[(k*capacity+added)*2] = hx;
        reflection_points[(k*capacity+added)*2+1] = hy;
        ++added;
        length += distance; ox=hx; oy=hy;
        double norm=sqrt(dx*dx+dy*dy); dx/=norm; dy/=norm;
        double dot=dx*w[4]+dy*w[5];
        dx-=2.0*dot*w[4]; dy-=2.0*dot*w[5];
        norm=sqrt(dx*dx+dy*dy); dx/=norm; dy/=norm;
    }
}
'''


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} 必须为正整数")
    return int(value)


class CudaWallGeometry:
    """一张地图的双精度墙数组常驻用户指定 GPU，支持批量最近交点查询。"""

    def __init__(self, scene, *, device_id=0):
        if (isinstance(device_id, (bool, np.bool_))
                or not isinstance(device_id, (int, np.integer)) or device_id < 0):
            raise ValueError("device_id 必须为非负整数")
        if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
            raise RuntimeError("必须先明确指定 CUDA_VISIBLE_DEVICES；不默认开放全部 GPU")
        self.cp = importlib.import_module("cupy")
        if device_id >= self.cp.cuda.runtime.getDeviceCount():
            raise ValueError("device_id 超出本进程可用的 GPU 范围")
        self.device_id = int(device_id)
        self.device = self.cp.cuda.Device(self.device_id)
        self.scene = scene
        self.walls = tuple(sorted(scene.walls, key=lambda w: w.wall_id))
        self.wall_indices = {w.wall_id: i for i, w in enumerate(self.walls)}
        wall_rows = [[*w.start_m, *w.vector, *w.normal, w.length_m, *w.end_m] for w in self.walls]
        with self.device:
            self.device_walls = self.cp.asarray(np.asarray(wall_rows, float).reshape(-1, 9))
            self.device_bounds = self.cp.asarray(scene.bounds_m, dtype=self.cp.float64)
            self.module = self.cp.RawModule(code=_CUDA_SOURCE,
                options=("--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"))
            self.nearest_kernel = self.module.get_function("nearest_many")
            self.fan_kernel = self.module.get_function("trace_fans")
            self.synchronize()

    def synchronize(self):
        with self.device:
            self.cp.cuda.get_current_stream().synchronize()

    def nearest_batch(self, origins, directions):
        """返回墙编号、距离和交点；无交点时墙编号为 -1，其他字段不可用。"""
        origins = np.asarray(origins, float)
        directions = np.asarray(directions, float)
        if (origins.ndim != 2 or origins.shape[1:] != (2,) or directions.shape != origins.shape
                or not np.all(np.isfinite(origins)) or not np.all(np.isfinite(directions))
                or np.any(np.linalg.norm(directions, axis=1) <= 1e-9)):
            raise ValueError("起点和非零方向必须为有限的 N×2 数组")
        n = len(origins)
        if not n:
            return np.empty(0, np.int32), np.empty((0, 3), float)
        with self.device:
            cp = self.cp
            ids = cp.empty(n, cp.int32); result = cp.empty((n, 3), cp.float64)
            self.nearest_kernel(((n+127)//128,), (128,), (
                self.device_walls, np.int32(len(self.walls)), cp.asarray(origins),
                cp.asarray(directions), np.int32(n), ids, result))
            return cp.asnumpy(ids), cp.asnumpy(result)


class CudaDiffractionTracer(CudaWallGeometry):
    """公共前缀由 CPU 构建一次；每次观测的绕射扇面在 CUDA 中批量追踪。"""

    def __init__(self, scene, bs, max_reflections, *, device_id=0):
        super().__init__(scene, device_id=device_id)
        self.bs = np.asarray(bs, float)
        self.max_reflections = int(max_reflections)
        self.prefixes, self.prefix_cache_hit = get_diffraction_prefixes(scene, self.bs, max_reflections)
        prefix_rows = []; incidents = []
        for edge, wall_ids, _, length, angle, previous in self.prefixes:
            ids = [self.wall_indices[key] for key in edge.incident_wall_ids]
            if not 1 <= len(ids) <= 2:
                raise ValueError("CUDA 第一版只支持与一面或两面墙关联的绕射边缘")
            scale = min(self.walls[i].length_m for i in ids) * 1e-3
            prefix_rows.append([*edge.position_m, length, *previous, len(wall_ids), angle, scale])
            incidents.append(ids + [-1] * (2-len(ids)))
        self.prefix_host = np.asarray(prefix_rows, float).reshape(-1, 8)
        with self.device:
            self.device_prefixes = self.cp.asarray(self.prefix_host)
            self.device_incidents = self.cp.asarray(np.asarray(incidents, np.int32).reshape(-1, 2))
            self.synchronize()

    def generate(self, samples, *, reference_bias_s, directions_per_sample, angle_tolerance_deg,
                 job_chunk_size=8192):
        cp = self.cp
        samples = list(samples)
        counts = {}; points = []; ordinals = {}; angle_matches = 0; attempted = 0
        directions = []
        for sample in samples:
            ordinal = ordinals.get(sample.observation_id, 0)
            ordinals[sample.observation_id] = ordinal + 1
            rows = []
            for branch in range(directions_per_sample):
                fraction = ((ordinal + 0.5) * 0.6180339887498949 + branch / directions_per_sample) % 1.0
                phi = 2 * np.pi * fraction
                rows.append((math.cos(phi), math.sin(phi)))
            directions.append(rows)
        directions = np.asarray(directions, float).reshape(-1, directions_per_sample, 2)
        capacity = max(1, self.max_reflections)
        # Bound the CPU angle-matching matrix and all device scratch arrays.
        with self.device, stage("T08_fan_trace"):
            for begin in range(0, len(samples), 256):
                batch = samples[begin:begin+256]
                angles = np.asarray([s.aoa_global_rad for s in batch])
                targets = np.asarray([(s.delay_s-reference_bias_s)*SPEED_OF_LIGHT_M_S for s in batch])
                error = (self.prefix_host[None, :, 6] - angles[:, None] + np.pi) % (2*np.pi) - np.pi
                match = ((np.abs(error) <= math.radians(angle_tolerance_deg))
                         & (targets[:, None] > self.prefix_host[None, :, 2]+1e-7))
                sample_indices, prefix_indices = np.nonzero(match)
                angle_matches += len(sample_indices)
                sj = np.repeat(sample_indices, directions_per_sample)
                pj = np.repeat(prefix_indices, directions_per_sample).astype(np.int32)
                bj = np.tile(np.arange(directions_per_sample), len(sample_indices))
                attempted += len(sj)
                for offset in range(0, len(sj), job_chunk_size):
                    ss = sj[offset:offset+job_chunk_size]
                    pp = pj[offset:offset+job_chunk_size]
                    bb = bj[offset:offset+job_chunk_size]
                    n = len(ss)
                    accepted = cp.empty(n, cp.int32)
                    result = cp.empty((n, 8), cp.float64)
                    nref = cp.empty(n, cp.int32)
                    wref = cp.empty((n, capacity), cp.int32)
                    pref = cp.empty((n, capacity, 2), cp.float64)
                    self.fan_kernel(((n+127)//128,), (128,), (
                        self.device_walls, np.int32(len(self.walls)), self.device_prefixes,
                        self.device_incidents, cp.asarray(pp), cp.asarray(directions[ss+begin, bb]),
                        cp.asarray(targets[ss]), np.int32(n), np.int32(self.max_reflections),
                        self.device_bounds, np.float64(self.bs[0]), np.float64(self.bs[1]),
                        accepted, result, nref, wref, pref))
                    chosen = cp.flatnonzero(accepted)
                    selected = cp.asnumpy(chosen)
                    values = cp.asnumpy(result[chosen]); nr = cp.asnumpy(nref[chosen])
                    wr = cp.asnumpy(wref[chosen]); pr = cp.asnumpy(pref[chosen])
                    for row, k in enumerate(selected):
                        sample = batch[ss[k]]
                        prefix_index = int(pp[k]); branch = int(bb[k])
                        edge, wall_ids, hit_points, _, _, _ = self.prefixes[prefix_index]
                        added_ids = tuple(self.walls[i].wall_id for i in wr[row, :nr[row]])
                        added_points = tuple(tuple(v) for v in pr[row, :nr[row]])
                        interactions = (*(("reflection", key) for key in wall_ids),
                                        ("diffraction", edge.edge_id),
                                        *(("reflection", key) for key in added_ids))
                        out = values[row]
                        points.append(InitialCandidatePoint(
                            observation_id=sample.observation_id,
                            sample_id=f"{sample.sample_id}:d{prefix_index:05d}:{branch:04d}",
                            parent_sample_id=sample.sample_id,
                            topology_id=json.dumps(interactions, separators=(",", ":")),
                            reference_bias_s=float(reference_bias_s), position_m=tuple(out[:2]),
                            reflection_wall_ids=(*wall_ids, *added_ids),
                            reflection_points_m=(*map(tuple, hit_points), *added_points),
                            observed_aoa_global_rad=float(sample.aoa_global_rad),
                            observed_delay_s=float(sample.delay_s), prefix_length_m=float(out[6]),
                            endpoint_origin_m=tuple(out[2:4]), endpoint_direction=tuple(out[4:6]),
                            endpoint_free_distance_m=float(out[7]), weight=float(sample.weight),
                            propagation_interactions=interactions,
                            interaction_points_m=(*map(tuple, hit_points), edge.position_m, *added_points)))
                        counts[sample.observation_id] = counts.get(sample.observation_id, 0)+1
            self.synchronize()
        return points, {
            "model": "2d_vertical_edges_single_shadow_diffraction_with_specular_reflections",
            "amplitude_model": "not_used_in_reverse_candidate_generation",
            "edge_count": len(diffraction_edges(self.scene)),
            "visible_bs_edge_prefix_count": len(self.prefixes),
            "prefix_geometry": "batched_exact_image_method_with_public_first_wall_visibility",
            "sample_prefix_angle_match_count": angle_matches, "attempted_direction_count": attempted,
            "directions_per_sample": directions_per_sample, "angle_tolerance_deg": float(angle_tolerance_deg),
            "initial_point_count": len(points), "observation_point_counts": counts,
            "mechanism_label_source": "map_hypothesis_not_observed_path_ground_truth",
            "angular_sampling": "deterministic_stratified_fan_across_observation_samples",
            "backend": "cuda", "device_id": self.device_id, "dtype": "float64",
            "public_prefix_backend": "numpy", "job_chunk_size": job_chunk_size,
            "cpu_stages": ["public_prefixes", "angle_matching", "candidate_objects"],
        }


@lru_cache(maxsize=2)
def _cached_tracer(scene, bs_tuple, max_reflections, device_id):
    return CudaDiffractionTracer(scene, bs_tuple, max_reflections, device_id=device_id)


def clear_cuda_reverse_cache():
    """仅释放本模块持有的引用，不清理其他模块的 CUDA 内存或 CPU 路径缓存。"""
    _cached_tracer.cache_clear()


def generate_diffraction_points_cuda(scene, bs, samples, *, reference_bias_s,
        max_reflections, directions_per_sample, angle_tolerance_deg, device_id=0, job_chunk_size=8192):
    directions_per_sample = _positive_integer("directions_per_sample", directions_per_sample)
    job_chunk_size = _positive_integer("job_chunk_size", job_chunk_size)
    if (isinstance(device_id, (bool, np.bool_))
            or not isinstance(device_id, (int, np.integer)) or device_id < 0):
        raise ValueError("device_id 必须为非负整数")
    if (isinstance(max_reflections, (bool, np.bool_))
            or not isinstance(max_reflections, (int, np.integer)) or max_reflections not in (0, 1, 2)):
        raise ValueError("max_reflections 必须为 0、1、2")
    if not np.isfinite(angle_tolerance_deg) or not 0 < angle_tolerance_deg < 90:
        raise ValueError("绕射角度匹配容差必须介于 0 与 90 度")
    bs = np.asarray(bs, float)
    if bs.shape != (2,) or not np.all(np.isfinite(bs)) or not scene.contains(bs):
        raise ValueError("BS 必须是场景内的有限二维坐标")
    if not np.isfinite(reference_bias_s):
        raise ValueError("reference_bias_s 必须为有限数")
    before = _cached_tracer.cache_info()
    with stage("T08_cuda_setup"):
        tracer = _cached_tracer(scene, tuple(bs), max_reflections, device_id)
    reused = _cached_tracer.cache_info().hits > before.hits
    points, diagnostics = tracer.generate(samples, reference_bias_s=reference_bias_s,
        directions_per_sample=directions_per_sample, angle_tolerance_deg=angle_tolerance_deg,
        job_chunk_size=job_chunk_size)
    diagnostics.update(cuda_geometry_cache_hit=reused,
                       public_prefix_cache_hit=reused or tracer.prefix_cache_hit)
    return points, diagnostics


def generate_initial_candidate_points_cuda(scene, bs_position_m, samples, *,
        reference_bias_s=0.0, max_reflections=2, backend="numpy", wall_chunk_size=8192,
        max_diffractions=0, diffraction_directions_per_sample=4, diffraction_angle_tolerance_deg=3.0,
        device_id=0, job_chunk_size=8192):
    """与现有候选入口兼容；直达/纯反射走 CPU，绕射分支走 CUDA。"""
    if isinstance(max_diffractions, (bool, np.bool_)) or max_diffractions not in (0, 1):
        raise ValueError("当前只支持 0 或 1 次绕射")
    samples = list(samples)
    initial = generate_initial_cpu(scene, bs_position_m, samples,
        reference_bias_s=reference_bias_s, max_reflections=max_reflections,
        backend=backend, wall_chunk_size=wall_chunk_size, max_diffractions=0)
    if not max_diffractions:
        return initial
    with stage("T08_diffraction"):
        extra, info = generate_diffraction_points_cuda(scene, bs_position_m, samples,
            reference_bias_s=reference_bias_s, max_reflections=max_reflections,
            directions_per_sample=diffraction_directions_per_sample,
            angle_tolerance_deg=diffraction_angle_tolerance_deg,
            device_id=device_id, job_chunk_size=job_chunk_size)
    points = initial.points + extra
    counts = dict.fromkeys(initial.diagnostics["observation_input_sample_counts"], 0)
    for point in points:
        counts[point.observation_id] += 1
    diagnostics = {**initial.diagnostics, "one_point_per_sample": False,
        "specular_branch_rejected_count": len(initial.rejected_samples),
        "rejected_sample_semantics": "rejected_specular_branch_not_all_hypotheses",
        "initial_point_count": len(points), "diffraction": info, "max_diffractions": 1,
        "observation_initial_point_counts": counts, "backend": "numpy_specular_cuda_diffraction"}
    return InitialCandidateGenerationResult(points, initial.rejected_samples, diagnostics)
