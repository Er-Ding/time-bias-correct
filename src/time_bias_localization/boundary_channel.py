"""边界实验的离线覆盖判定与固定几何信道；本模块不得由在线定位器导入。

覆盖资格只使用公开放置区域与 RT 有效路径。二维反向几何的一致性检查在
资格确定之后单独记录，不能导致换点。真实 Sionna 系数不做按路径强度选点。
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import importlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters import load_sionna_rt_module
from .candidates import global_to_local_aoa
from .constants import SPEED_OF_LIGHT_M_S
from .data import OnlineMeasurement, build_subcarrier_frequencies
from .diffraction import enumerate_paths
from .provenance import artifact_record, canonical_json_sha256, generation_bundle_id
from .scene import Scene2D, make_synthetic_room, preprocess_sionna_triangle_mesh
from .signal import apply_common_delay_bias, synthesize_ula_csi
from .sionna_generation import (
    _path_selection_summary, _save_real_bundle, _sionna_runtime_info,
    _validate_reverse_scene_consistency, _write_json_atomic, extract_planar_uplink_csi,
)


def _xy(value: Sequence[float]) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError("位置必须是两个有限数")
    return point


def _point_in_polygon(point: np.ndarray, vertices: Sequence[Sequence[float]]) -> bool:
    polygon = np.asarray(vertices, dtype=float)
    x, y = point
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crossing = (a[:, 1] > y) != (b[:, 1] > y)
    edge = np.flatnonzero(crossing)
    return bool(np.count_nonzero(x < a[edge, 0] + (y - a[edge, 1]) *
                                (b[edge, 0] - a[edge, 0]) /
                                (b[edge, 1] - a[edge, 1])) % 2)


@dataclass
class LegalRegion:
    """公共地图上的可放置区域。自动模式明确限制为屋顶/顶棚之外的露天区域。"""

    scene: Scene2D
    bs_position_m: Sequence[float]
    wall_clearance_m: float = 0.5
    bs_min_distance_m: float = 2.0
    include_polygons: Sequence[Sequence[Sequence[float]]] = ()
    exclude_polygons: Sequence[Sequence[Sequence[float]]] = ()
    overhead_triangles_m: np.ndarray | None = None
    source: str = "declared_public_polygons"

    def __post_init__(self) -> None:
        self.bs_position_m = _xy(self.bs_position_m)
        for key in ("wall_clearance_m", "bs_min_distance_m"):
            value = float(getattr(self, key))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} 必须是非负有限数")
        for polygon in (*self.include_polygons, *self.exclude_polygons):
            p = np.asarray(polygon, dtype=float)
            if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.all(np.isfinite(p)):
                raise ValueError("公开区域多边形必须包含至少三个有限二维顶点")
            area = abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))))
            if area <= 1e-10:
                raise ValueError("公开区域多边形面积必须大于零")
        self._wall_start = np.asarray([w.start_m for w in self.scene.walls], dtype=float).reshape(-1, 2)
        self._wall_vector = np.asarray([w.end - w.start for w in self.scene.walls], dtype=float).reshape(-1, 2)
        self._wall_length_squared = np.sum(self._wall_vector ** 2, axis=1)
        if self.overhead_triangles_m is None:
            self.overhead_triangles_m = np.empty((0, 3, 3), dtype=float)
        triangles = np.asarray(self.overhead_triangles_m, dtype=float)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.all(np.isfinite(triangles)):
            raise ValueError("公开顶面三角形必须为有限 [N,3,3] 数组")
        # 保留可能位于 UE 上方、且投影具有面积的面；垂直墙由墙边距处理。
        a = triangles[:, 1, :2] - triangles[:, 0, :2]
        b = triangles[:, 2, :2] - triangles[:, 0, :2]
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        keep = (np.abs(det) > 1e-12) & (np.max(triangles[:, :, 2], axis=1) >= self.scene.fixed_height_m)
        self.overhead_triangles_m = triangles[keep]
        self._overhead_min = np.min(self.overhead_triangles_m[:, :, :2], axis=1)
        self._overhead_max = np.max(self.overhead_triangles_m[:, :, :2], axis=1)

    def classify(self, position_m: Sequence[float]) -> tuple[bool, str]:
        point = _xy(position_m)
        if not self.scene.contains(point):
            return False, "outside_public_bounds"
        if self.include_polygons and not any(_point_in_polygon(point, p) for p in self.include_polygons):
            return False, "outside_declared_polygons"
        if any(_point_in_polygon(point, p) for p in self.exclude_polygons):
            return False, "inside_excluded_polygon"
        if float(np.linalg.norm(point - self.bs_position_m)) < self.bs_min_distance_m:
            return False, "too_close_to_bs"
        if len(self._wall_start):
            t = np.clip(np.sum((point - self._wall_start) * self._wall_vector, axis=1) /
                        self._wall_length_squared, 0.0, 1.0)
            distance = np.linalg.norm(point - self._wall_start - t[:, None] * self._wall_vector, axis=1)
            if np.any(distance <= max(self.wall_clearance_m, 1e-8)):
                return False, "wall_clearance"
        mask = np.all(point >= self._overhead_min - 1e-9, axis=1) & np.all(point <= self._overhead_max + 1e-9, axis=1)
        triangles = self.overhead_triangles_m[mask]
        if len(triangles):
            a, b = triangles[:, 1, :2] - triangles[:, 0, :2], triangles[:, 2, :2] - triangles[:, 0, :2]
            q = point - triangles[:, 0, :2]
            det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            u = (q[:, 0] * b[:, 1] - q[:, 1] * b[:, 0]) / det
            v = (a[:, 0] * q[:, 1] - a[:, 1] * q[:, 0]) / det
            inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
            height = triangles[:, 0, 2] + u * (triangles[:, 1, 2] - triangles[:, 0, 2]) + v * (triangles[:, 2, 2] - triangles[:, 0, 2])
            if np.any(inside & (height >= self.scene.fixed_height_m - 1e-8)):
                return False, "inside_public_overhead_footprint"
        return True, "legal"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "bounds_m": list(self.scene.bounds_m),
                "wall_clearance_m": self.wall_clearance_m, "bs_min_distance_m": self.bs_min_distance_m,
                "include_polygons": self.include_polygons, "exclude_polygons": self.exclude_polygons,
                "overhead_triangle_count": len(self.overhead_triangles_m),
                "meaning": "declared outdoor open-sky area; roofs, overhead structures, walls and BS margin excluded"}


@dataclass
class CoverageProbe:
    status: str
    reason: str
    position_m: np.ndarray
    seed: int
    rt_seconds: float
    csi_geometric: np.ndarray | None = None
    path_metadata: dict[str, np.ndarray] = field(default_factory=dict)
    geometry_diagnostic: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    channel_setup_sha256: str | None = None

    def summary(self) -> dict[str, Any]:
        kept = np.asarray(self.path_metadata.get("retained_mask", []), dtype=bool)
        reflection = np.asarray(self.path_metadata.get("reflection_order", np.zeros(len(kept))), dtype=int)
        diffraction = np.asarray(self.path_metadata.get("diffraction_order", np.zeros(len(kept))), dtype=int)
        has_los = bool(np.any(kept & (reflection == 0) & (diffraction == 0)))
        has_reflection = bool(np.any(kept & (reflection > 0) & (diffraction == 0)))
        has_diffraction = bool(np.any(kept & (diffraction > 0)))
        category = "los" if has_los else "reflection_without_los" if has_reflection else "diffraction_only" if has_diffraction else "none"
        return {"status": self.status, "reason": self.reason, "position_m": self.position_m.tolist(),
                "seed": self.seed, "rt_seconds": self.rt_seconds, "retained_path_count": int(np.sum(kept)),
                "has_los": has_los, "has_reflection": has_reflection, "has_diffraction": has_diffraction,
                "channel_category": category, "signal_mean_power": (float(np.mean(np.abs(self.csi_geometric) ** 2))
                    if self.csi_geometric is not None else None),
                "geometry_diagnostic": self.geometry_diagnostic, "error": self.error,
                "channel_setup_sha256": self.channel_setup_sha256}


def save_probe(probe: CoverageProbe, root: str | Path) -> dict[str, str]:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    arrays = {"position_m": probe.position_m, **probe.path_metadata}
    if probe.csi_geometric is not None:
        arrays["csi_geometric"] = probe.csi_geometric
    np.savez_compressed(root / "geometry_channel.npz", **arrays)
    record = {**probe.summary(), "schema_version": 1,
              "array_artifact": artifact_record(root / "geometry_channel.npz")}
    _write_json_atomic(root / "probe.json", record)
    return {"probe_json": str(root / "probe.json"), "geometry_npz": str(root / "geometry_channel.npz")}


def load_probe(root: str | Path) -> CoverageProbe:
    root = Path(root).resolve()
    record = json.loads((root / "probe.json").read_text())
    path = root / "geometry_channel.npz"
    if artifact_record(path)["sha256"] != record["array_artifact"]["sha256"]:
        raise ValueError("冻结几何信道的内容指纹不一致")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return CoverageProbe(record["status"], record["reason"], arrays.pop("position_m"), int(record["seed"]),
                         float(record["rt_seconds"]), arrays.pop("csi_geometric", None), arrays,
                         record.get("geometry_diagnostic", {}), record.get("error"), record.get("channel_setup_sha256"))


class BoundaryChannel:
    """公开配置与地图只加载一次；所有 UE 使用完全相同的传播和筛选规则。"""

    backend = "abstract"

    def __init__(self, config: Mapping[str, Any], setup_root: str | Path):
        self.config = deepcopy(dict(config))
        self.setup_root = Path(setup_root).resolve()
        self.setup_root.mkdir(parents=True, exist_ok=False)
        self.frequencies_hz = build_subcarrier_frequencies(bandwidth_hz=float(config["radio"]["bandwidth_hz"]),
                                                         num_subcarriers=int(config["radio"]["num_subcarriers"]))
        self.provenance: dict[str, Any] = {}
        self.rt_params: dict[str, Any] = {}
        self.closed = False

    def _finish_setup(self, legal_options: Mapping[str, Any] | None, triangles: np.ndarray | None = None) -> None:
        options = dict(legal_options or {})
        allowed = {"wall_clearance_m", "bs_min_distance_m", "include_polygons", "exclude_polygons"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"未知公开区域参数：{sorted(unknown)}")
        self.legal_region = LegalRegion(self.scene_2d, self.config["simulation"]["bs_position_m"],
                                        overhead_triangles_m=triangles,
                                        source="original_mesh_open_sky" if triangles is not None else "declared_fixture_region",
                                        **options)
        self.scene_artifacts = self.scene_2d.save(self.setup_root / "scene")
        self.scene_json = self.scene_artifacts["scene_json"]
        self.provenance.update(schema_version=1, backend=self.backend, scene_artifacts=self.scene_artifacts,
                               scene_fingerprint=artifact_record(self.scene_json), legal_region=self.legal_region.to_dict(),
                               coefficient_zero_threshold=0.0, coverage_rule="at_least_one_finite_nonzero_supported_path",
                               radio={key: value for key, value in self.config["radio"].items() if key != "snr_db"},
                               bs_position_m=list(self.config["simulation"]["bs_position_m"]),
                               propagation={"max_reflections": self.config["scene"]["max_reflections"],
                                            "max_diffractions": self.config["scene"].get("max_diffractions", 0)},
                               rt_parameters=self.rt_params)
        self.setup_json = _write_json_atomic(self.setup_root / "channel_setup.json", self.provenance)

    def _compute(self, position_m: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raise NotImplementedError

    def probe(self, position_m: Sequence[float], seed: int) -> CoverageProbe:
        if self.closed:
            raise RuntimeError("信道提供器已经关闭")
        point = _xy(position_m)
        valid, reason = self.legal_region.classify(point)
        if not valid:
            return CoverageProbe("illegal", reason, point, int(seed), 0.0)
        started = perf_counter()
        try:
            csi, metadata = self._compute(point, int(seed))
        except Exception as error:
            return CoverageProbe("unknown", "rt_or_export_error", point, int(seed), perf_counter() - started,
                                 error=f"{type(error).__name__}: {error}")
        elapsed = perf_counter() - started
        retained = int(np.sum(metadata["retained_mask"]))
        if retained == 0:
            coefficients = np.asarray(metadata.get("path_coefficients", []))
            if coefficients.size and not np.all(np.isfinite(coefficients)):
                return CoverageProbe("unknown", "nonfinite_rt_coefficients", point, int(seed), elapsed,
                                     None, metadata)
            return CoverageProbe("no_signal", "no_supported_finite_nonzero_path_found", point, int(seed), elapsed,
                                 csi, metadata)
        if not np.all(np.isfinite(csi)) or float(np.mean(np.abs(csi) ** 2)) <= 0.0:
            return CoverageProbe("unknown", "nonfinite_or_cancelled_csi_after_valid_paths", point, int(seed), elapsed,
                                 None, metadata)
        # 信号资格已确定。反向地图解释失败只留下诊断，仍保留该 UE。
        try:
            diagnostic = _validate_reverse_scene_consistency(
                self.scene_2d, bs_xy=np.asarray(self.config["simulation"]["bs_position_m"]), ue_xy=point,
                path_metadata=metadata, tolerance_m=max(1e-3, 0.5 * self.scene_2d.bev_resolution_m))
        except Exception as error:
            diagnostic = {"passed": False, "error": f"{type(error).__name__}: {error}"}
        return CoverageProbe("covered", "finite_nonzero_supported_path_found", point, int(seed), elapsed,
                             csi, metadata, diagnostic, channel_setup_sha256=artifact_record(self.setup_json)["sha256"])

    def write_observation_bundle(self, probe: CoverageProbe, *, output_root: str | Path, noise_seed: int,
                                 config: Mapping[str, Any] | None = None) -> dict[str, str]:
        cfg = self.config if config is None else config
        if probe.status != "covered" or probe.csi_geometric is None:
            raise ValueError("只有已获得信号资格的冻结几何信道可以生成观测")
        if probe.channel_setup_sha256 != artifact_record(self.setup_json)["sha256"]:
            raise ValueError("冻结几何信道必须绑定同一公共信道设置")
        if {key: value for key, value in cfg["radio"].items() if key != "snr_db"} != self.provenance["radio"]:
            raise ValueError("冻结几何信道后不能改变无线配置；仅可改变噪声强度")
        if list(cfg["simulation"]["bs_position_m"]) != self.provenance["bs_position_m"]:
            raise ValueError("冻结几何信道后不能改变 BS 位置")
        if {"max_reflections": cfg["scene"]["max_reflections"], "max_diffractions": cfg["scene"].get("max_diffractions", 0)} != self.provenance["propagation"]:
            raise ValueError("冻结几何信道后不能改变传播模型")
        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        radio, simulation = cfg["radio"], cfg["simulation"]
        geometric = probe.csi_geometric
        if geometric.ndim == 2:
            geometric = np.repeat(geometric[None, ...], int(radio["num_snapshots"]), axis=0)
        noise_std = float(np.sqrt(np.mean(np.abs(geometric) ** 2))) * 10.0 ** (-float(radio["snr_db"]) / 20.0)
        observed = apply_common_delay_bias(geometric, self.frequencies_hz, float(simulation["clock_bias_s"]),
                                          noise_std=noise_std, seed=int(noise_seed))
        online = OnlineMeasurement(observed, self.frequencies_hz, float(radio["carrier_hz"]),
                                   SPEED_OF_LIGHT_M_S / float(radio["carrier_hz"]) * float(radio["antenna_spacing_wavelength"]),
                                   np.asarray(simulation["bs_position_m"], dtype=float), math.radians(float(radio["bs_boresight_deg"])))
        artifacts = _save_real_bundle(root, online, ue_position_m=probe.position_m,
                                     clock_bias_s=float(simulation["clock_bias_s"]), csi_geometric=geometric,
                                     injected_noise_std=noise_std, path_metadata=probe.path_metadata)
        selection = _path_selection_summary(probe.path_metadata)
        selection.pop("total_sionna_path_count")
        # 不指定 MUSIC 模型阶数；这里记录已冻结且全数合成的有效路径数量。
        selection["requested_generated_path_count"] = int(np.sum(probe.path_metadata["retained_mask"]))
        manifest: dict[str, Any] = {
            "schema_version": 2, "stage": "sionna_rt_boundary_v1" if self.backend == "sionna" else "synthetic_rt_boundary_fixture_v1",
            "scene_json": self.scene_json, "online_input": artifacts["online_npz"], "truth_input": artifacts["truth_npz"],
            "separation_rule": "localization 只允许读取 online 目录", "link_direction": "uplink_ue_to_bs",
            "absolute_delay_normalization": False, "delay_convention": "observed_delay=geometric_delay+common_bias+noise",
            "path_selection": selection,
            "rt_model": {"los": True, "specular_reflection": True,
                         "max_reflections": int(cfg["scene"]["max_reflections"]),
                         "diffraction": bool(cfg["scene"].get("max_diffractions", 0)),
                         "diffuse_reflection": False, "transmission": False, "front_facing_only": True},
            "artifact_hashes": {"scene_json": artifact_record(self.scene_json),
                                "online_measurement": artifact_record(artifacts["online_npz"]),
                                "ground_truth": artifact_record(artifacts["truth_npz"])},
        }
        manifest["channel_setup"] = artifact_record(self.setup_json)
        if self.backend == "sionna":
            manifest["rt_params"] = {**self.rt_params, "seed": int(probe.seed)}
        manifest["bundle_id"] = generation_bundle_id(manifest)
        manifest_path = _write_json_atomic(root / "generation_manifest.json", manifest)
        _write_json_atomic(root / "data" / "truth" / "boundary_generation.json",
                           {"allowed_for_localization": False, "backend": self.backend, "noise_seed": int(noise_seed),
                            "noise_mode": "per_ue_fixed_target_snr", "snr_db": float(radio["snr_db"]),
                            "geometry_seed": probe.seed, "geometry_diagnostic": probe.geometry_diagnostic,
                            "channel_summary": probe.summary(), "setup": artifact_record(self.setup_json),
                            "fixture_scope": self.provenance.get("validation_scope")})
        return {**artifacts, "scene_json": self.scene_json, "generation_manifest": manifest_path}

    def close(self) -> None:
        self.closed = True


class SyntheticFixtureChannel(BoundaryChannel):
    """几何及流程检查专用，使用预先声明的幅度规则；不属于真实 RT 精度实验。"""
    backend = "synthetic_fixture"

    def __init__(self, config: Mapping[str, Any], setup_root: str | Path, *, public_scene_json: str | Path | None = None,
                 legal_region: Mapping[str, Any] | None = None):
        super().__init__(config, setup_root)
        if public_scene_json is None:
            self.scene_2d = replace(make_synthetic_room(bounds_m=config["scene"]["bounds_m"],
                               fixed_height_m=float(config["scene"]["fixed_height_m"]),
                               bev_resolution_m=float(config["scene"]["bev_resolution_m"])),
                               name=str(config["scene"]["name"]), source=str(config["scene"]["source"]))
        else:
            self.scene_2d = Scene2D.from_dict(json.loads(Path(public_scene_json).read_text()))
        self.provenance["validation_scope"] = "configured_amplitude_synthetic_fixture_only_not_calibrated_utd"
        self._finish_setup(legal_region)

    def _compute(self, position_m: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg, radio = self.config, self.config["radio"]
        all_paths = enumerate_paths(self.scene_2d, source_m=position_m, receiver_m=cfg["simulation"]["bs_position_m"],
                                    max_reflections=int(cfg["scene"]["max_reflections"]),
                                    max_diffractions=int(cfg["scene"].get("max_diffractions", 0)))
        n, depth = len(all_paths), int(cfg["scene"]["max_reflections"]) + int(cfg["scene"].get("max_diffractions", 0))
        vertices = np.zeros((depth, n, 3))
        interactions = np.zeros((depth, n), dtype=int)
        for j, path in enumerate(all_paths):
            sequence = path.propagation_interactions or tuple(("reflection", wall) for wall in path.interaction_wall_ids)
            for k, (kind, _) in enumerate(sequence):
                interactions[k, j] = 8 if kind == "diffraction" else 1
                vertices[k, j] = [*path.interaction_points_m[k], self.scene_2d.fixed_height_m]
        boresight = math.radians(float(radio["bs_boresight_deg"]))
        global_angles = np.radians([path.arrival_aoa_deg for path in all_paths])
        local_angles = np.asarray([global_to_local_aoa(angle, boresight) for angle in global_angles])
        low, high = math.radians(float(cfg["music"]["angle_min_deg"])), math.radians(float(cfg["music"]["angle_max_deg"]))
        retained = (local_angles >= low) & (local_angles <= high)
        delays = np.asarray([path.delay_s for path in all_paths])
        amplitudes = np.asarray(cfg["simulation"].get("path_amplitudes", [1.0]), dtype=float)
        if amplitudes.size == 0 or np.any(~np.isfinite(amplitudes)) or np.any(amplitudes < 0):
            raise ValueError("验证幅度必须为非负有限数")
        # 此固定规则与 MUSIC.num_paths 无关，不能用生成真值指定在线模型阶数。
        coefficients = np.resize(amplitudes, n) * np.exp(1j * np.random.default_rng(seed).uniform(-np.pi, np.pi, n))
        retained &= np.abs(coefficients) > 0.0
        if np.any(retained):
            csi = synthesize_ula_csi(path_aoa_rad=local_angles[retained], path_delay_s=delays[retained],
                                    path_coefficients=coefficients[retained], subcarrier_frequencies_hz=self.frequencies_hz,
                                    num_bs_antennas=int(radio["num_bs_antennas"]), carrier_frequency_hz=float(radio["carrier_hz"]),
                                    antenna_spacing_m=SPEED_OF_LIGHT_M_S / float(radio["carrier_hz"]) * float(radio["antenna_spacing_wavelength"]))
        else:
            csi = np.zeros((int(radio["num_bs_antennas"]), len(self.frequencies_hz)), dtype=complex)
        metadata = {"retained_mask": retained, "planar_retained_mask_before_front_filter": np.ones(n, dtype=bool),
                    "front_facing_angle_mask": (local_angles >= low) & (local_angles <= high),
                    "absolute_delays_s": delays, "aoa_global_rad": global_angles, "aoa_local_rad": local_angles,
                    "interaction_order": np.sum(interactions != 0, axis=0), "reflection_order": np.sum(interactions == 1, axis=0),
                    "diffraction_order": np.sum(interactions == 8, axis=0), "vertices_m": vertices, "interactions": interactions,
                    "path_coefficients": coefficients[None, :], "front_facing_only": np.asarray(True),
                    "bs_boresight_rad": np.asarray(boresight), "local_angle_min_rad": np.asarray(low), "local_angle_max_rad": np.asarray(high)}
        return csi, metadata


class SionnaBoundaryChannel(BoundaryChannel):
    backend = "sionna"

    def __init__(self, config: Mapping[str, Any], setup_root: str | Path, *, legal_region: Mapping[str, Any] | None = None):
        super().__init__(config, setup_root)
        started = perf_counter()
        rt = load_sionna_rt_module()
        mi = importlib.import_module("mitsuba")
        self.provenance["runtime"] = _sionna_runtime_info(require_cuda=os.environ.get("TBC_REQUIRE_CUDA") == "1")
        scene_asset = getattr(getattr(rt, "scene"), str(config["scene"]["name"]), None)
        if scene_asset is None or not Path(scene_asset).is_file():
            raise FileNotFoundError(f"Sionna 内置场景不存在：{config['scene']['name']}")
        self.rt_scene = rt.load_scene(scene_asset)
        radio, sc = config["radio"], config["scene"]
        self.rt_scene.frequency, self.rt_scene.bandwidth = float(radio["carrier_hz"]), float(radio["bandwidth_hz"])
        self.rt_scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                               horizontal_spacing=0.5, pattern="iso", polarization="V")
        self.rt_scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=int(radio["num_bs_antennas"]), vertical_spacing=0.5,
                                               horizontal_spacing=float(radio["antenna_spacing_wavelength"]), pattern="iso", polarization="V")
        bs, z = _xy(config["simulation"]["bs_position_m"]), float(sc["fixed_height_m"])
        self.transmitter = rt.Transmitter("boundary_ue", position=[float(bs[0]), float(bs[1]), z])
        receiver = rt.Receiver("boundary_bs", position=[float(bs[0]), float(bs[1]), z])
        receiver.orientation = [math.radians(float(radio["bs_boresight_deg"])), 0.0, 0.0]
        self.rt_scene.add(self.transmitter)
        self.rt_scene.add(receiver)
        self.path_solver = rt.PathSolver()
        self.rt_params = {"max_depth": int(sc["max_reflections"]) + int(sc.get("max_diffractions", 0)), "los": True,
                          "specular_reflection": True, "diffuse_reflection": False, "diffraction": bool(sc.get("max_diffractions", 0)),
                          "refraction": False, "samples_per_src": int(config["simulation"]["samples_per_source"]),
                          "max_num_paths_per_src": int(config["simulation"].get("max_num_paths_per_source", 100000)),
                          "synthetic_array": True, "seed": int(config["project"]["random_seed"])}
        if sc.get("max_diffractions", 0):
            self.rt_params.update(edge_diffraction=True, diffraction_lit_region=False)
        # 从已载入的原始 Mitsuba 网格读取世界坐标，不经 DeepMIMO 凸包简化。
        vertices, faces, owners, materials = [], [], {}, {}
        offset = 0
        for name, obj in sorted(self.rt_scene.objects.items()):
            parameters = mi.traverse(obj.mi_mesh)
            v = np.asarray(parameters["vertex_positions"]).reshape(-1, 3).astype(float)
            f = np.asarray(parameters["faces"]).reshape(-1, 3).astype(np.int64)
            vertices.append(v)
            faces.append(f + offset)
            owners[name] = [offset, offset + len(v)]
            material = obj.radio_material
            materials[name] = {"name": str(material.name), "relative_permittivity": np.asarray(material.relative_permittivity).tolist(),
                               "conductivity": np.asarray(material.conductivity).tolist()}
            offset += len(v)
        all_vertices, all_faces = np.vstack(vertices), np.vstack(faces)
        mesh_path = self.setup_root / "public_original_mesh.npz"
        np.savez_compressed(mesh_path, vertices_m=all_vertices, faces=all_faces)
        self.scene_2d = preprocess_sionna_triangle_mesh(
            all_vertices, all_faces, name=f"{sc['name']}_bev", fixed_height_m=z,
            bev_resolution_m=float(sc["bev_resolution_m"]), object_vertex_ranges=owners,
            bounds_m=sc.get("localization_bounds_m", sc["bounds_m"]))
        self.provenance.update(source_scene=artifact_record(scene_asset), public_mesh=artifact_record(mesh_path),
                               object_vertex_ranges=owners, material_parameters=materials,
                               material_sha256=canonical_json_sha256(materials), setup_seconds=perf_counter() - started)
        self._finish_setup(legal_region, all_vertices[all_faces])

    def _compute(self, position_m: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg, radio = self.config, self.config["radio"]
        self.transmitter.position = [float(position_m[0]), float(position_m[1]), float(cfg["scene"]["fixed_height_m"])]
        paths = self.path_solver(scene=self.rt_scene, **{**self.rt_params, "seed": int(seed)})
        return extract_planar_uplink_csi(
            paths, self.frequencies_hz, fixed_height_m=float(cfg["scene"]["fixed_height_m"]),
            vertical_tolerance_m=float(cfg["scene"]["vertical_path_tolerance_m"]),
            max_reflections=int(cfg["scene"]["max_reflections"]), max_diffractions=int(cfg["scene"].get("max_diffractions", 0)),
            bs_boresight_rad=math.radians(float(radio["bs_boresight_deg"])),
            local_angle_min_rad=math.radians(float(cfg["music"]["angle_min_deg"])),
            local_angle_max_rad=math.radians(float(cfg["music"]["angle_max_deg"])),
            front_facing_only=True, minimum_path_count=0, coefficient_zero_threshold=0.0)

    def close(self) -> None:
        self.path_solver = self.rt_scene = self.transmitter = None
        super().close()
        import gc
        gc.collect()
        try:
            importlib.import_module("drjit").flush_malloc_cache()
        except (ImportError, AttributeError):
            pass


def make_boundary_channel(config: Mapping[str, Any], setup_root: str | Path, *, backend: str = "sionna",
                          public_scene_json: str | Path | None = None,
                          legal_region: Mapping[str, Any] | None = None) -> BoundaryChannel:
    if not bool(config["radio"].get("front_facing_only", True)):
        raise ValueError("第一版边界实验要求阵列正面角度限制")
    if backend == "sionna":
        if public_scene_json is not None:
            raise ValueError("Sionna 实验的公共地图必须由同一原始网格生成")
        return SionnaBoundaryChannel(config, setup_root, legal_region=legal_region)
    if backend == "synthetic_fixture":
        return SyntheticFixtureChannel(config, setup_root, public_scene_json=public_scene_json, legal_region=legal_region)
    raise ValueError(f"未知信道后端：{backend}")


def write_observation_bundle(probe: CoverageProbe, config: Mapping[str, Any], *, setup_root: str | Path,
                             output_root: str | Path, noise_seed: int) -> dict[str, str]:
    """从已保存的几何信道生成另一份独立噪声 CSI，不载入 RT/GPU。"""
    provider = BoundaryChannel.__new__(BoundaryChannel)
    provider.config = deepcopy(dict(config))
    provider.setup_root = Path(setup_root).resolve()
    provider.setup_json = str(provider.setup_root / "channel_setup.json")
    provider.provenance = json.loads(Path(provider.setup_json).read_text())
    provider.backend = provider.provenance["backend"]
    provider.scene_artifacts = provider.provenance["scene_artifacts"]
    provider.scene_json = provider.scene_artifacts["scene_json"]
    if artifact_record(provider.scene_json)["sha256"] != provider.provenance["scene_fingerprint"]["sha256"]:
        raise ValueError("公共场景与冻结信道的指纹不一致")
    provider.rt_params = provider.provenance["rt_parameters"]
    provider.frequencies_hz = build_subcarrier_frequencies(bandwidth_hz=float(config["radio"]["bandwidth_hz"]),
                                                         num_subcarriers=int(config["radio"]["num_subcarriers"]))
    return provider.write_observation_bundle(probe, output_root=output_root, noise_seed=noise_seed)


__all__ = ["CoverageProbe", "LegalRegion", "BoundaryChannel", "make_boundary_channel", "save_probe", "load_probe", "write_observation_bundle"]
