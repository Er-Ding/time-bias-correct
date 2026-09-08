"""Sionna RT 内置场景到 DeepMIMO V4 与在线 CSI 的生成入口。"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import numpy as np

from .adapters import (
    AdapterContractError,
    load_deepmimo_module,
    load_sionna_rt_module,
    sionna_paths_cir,
)
from .candidates import global_to_local_aoa
from .constants import SPEED_OF_LIGHT_M_S
from .data import OnlineMeasurement, build_subcarrier_frequencies
from .provenance import (
    artifact_record,
    exclusive_output_root_lock,
    file_sha256,
    generation_bundle_id,
    localization_config_snapshot,
)
from .scene import Scene2D, preprocess_sionna_exported_scene, ray_segment_intersection
from .signal import apply_common_delay_bias


@contextmanager
def _working_directory(path: Path):
    """临时切换工作目录，并保证异常时也能恢复。"""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_json_atomic(path: Path, data: Any) -> str:
    """在目标目录写完并同步临时文件后，原子发布 JSON。"""

    encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return str(path)


def _sionna_runtime_info(*, require_cuda: bool) -> dict[str, Any]:
    """记录射线计算实际使用的设备；GPU 实验不允许静默退回 CPU。"""
    try:
        mitsuba = importlib.import_module("mitsuba")
        variant = getattr(mitsuba, "variant", lambda: None)()
    except ImportError:
        variant = None
    using_cuda = isinstance(variant, str) and variant.startswith("cuda_")
    if require_cuda and not using_cuda:
        raise RuntimeError(
            f"GPU 实验要求 Sionna 使用 CUDA，但当前 Mitsuba variant={variant!r}。"
            "请检查驱动及 Sionna 环境；本次不自动改用 CPU。"
        )
    return {
        "mitsuba_variant": variant,
        "uses_cuda": using_cuda,
        "cuda_required": require_cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_id": os.getpid(),
    }


def _ensure_fresh_generation_targets(root: Path) -> None:
    """一次检查 Sionna 生成拥有的所有固定目标，拒绝覆盖任何旧内容。"""

    targets = (
        root / "generation_manifest.json",
        root / "provenance" / "generation_config.json",
        root / "provenance" / "generation_runtime.json",
        root / "deepmimo_source",
        root / "deepmimo_scenarios",
        root / "scene",
        root / "data",
    )
    existing = [path for path in targets if path.exists() or path.is_symlink()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Sionna 生成目标已存在；allow_overwrite=false，拒绝覆盖旧批次：\n"
            f"{formatted}\n请换用新的 output.root"
        )


def _normalize_deepmimo_export_scalars(export_dir: Path) -> tuple[str, ...]:
    """修正 DeepMIMO 4.0.5 无法读取 Sionna 单元素标量数组的问题。"""

    parameter_path = export_dir / "sionna_rt_params.pkl"
    with parameter_path.open("rb") as handle:
        parameters = pickle.load(handle)
    if not isinstance(parameters, dict):
        raise AdapterContractError("Sionna 导出的射线参数不是键值映射")

    normalized: list[str] = []
    for key in ("frequency", "bandwidth"):
        value = parameters.get(key)
        array = np.asarray(value)
        if array.size != 1:
            raise AdapterContractError(
                f"Sionna 导出参数 {key} 应只有一个值，实际形状为 {array.shape}"
            )
        if array.ndim > 0:
            parameters[key] = float(array.reshape(-1)[0])
            normalized.append(key)
    if normalized:
        with parameter_path.open("wb") as handle:
            pickle.dump(parameters, handle)

    material_path = export_dir / "sionna_materials.pkl"
    with material_path.open("rb") as handle:
        materials = pickle.load(handle)
    if not isinstance(materials, list):
        raise AdapterContractError("Sionna 导出的材料参数不是列表")
    material_scalar_keys = (
        "conductivity",
        "relative_permittivity",
        "scattering_coefficient",
        "xpd_coefficient",
        "alpha_r",
        "alpha_i",
        "lambda_",
    )
    materials_changed = False
    for material_index, material in enumerate(materials):
        if not isinstance(material, dict):
            raise AdapterContractError("Sionna 导出的单项材料参数不是键值映射")
        for key in material_scalar_keys:
            value = material.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim == 0:
                continue
            if array.size != 1:
                raise AdapterContractError(
                    f"Sionna 材料 {material_index} 的 {key} 应只有一个值，"
                    f"实际形状为 {array.shape}"
                )
            material[key] = array.reshape(-1)[0].item()
            normalized.append(f"materials[{material_index}].{key}")
            materials_changed = True
    if materials_changed:
        with material_path.open("wb") as handle:
            pickle.dump(materials, handle)
    return tuple(normalized)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value)


def _single_link_tau(tau: np.ndarray, num_paths: int) -> np.ndarray:
    if tau.ndim == 3:  # synthetic array: [rx, tx, path]
        selected = tau[0, 0]
    elif tau.ndim == 5:  # explicit arrays: [rx, rx_ant, tx, tx_ant, path]
        selected = tau[0, 0, 0, 0]
    else:
        raise AdapterContractError(f"不支持的 Sionna tau 形状：{tau.shape}")
    if selected.shape != (num_paths,):
        raise AdapterContractError("Sionna tau 路径轴与 CIR 不一致")
    return np.asarray(selected, dtype=float)


def _single_link_vertices(paths: Any, num_paths: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = _to_numpy(paths.vertices)
    interactions = _to_numpy(paths.interactions)
    if vertices.ndim == 5:  # [depth, rx, tx, path, xyz]
        vertices = vertices[:, 0, 0]
        interactions = interactions[:, 0, 0]
    elif vertices.ndim == 7:  # [depth, rx, rx_ant, tx, tx_ant, path, xyz]
        vertices = vertices[:, 0, 0, 0, 0]
        interactions = interactions[:, 0, 0, 0, 0]
    else:
        raise AdapterContractError(f"不支持的 Sionna vertices 形状：{vertices.shape}")
    if vertices.shape[1:] != (num_paths, 3) or interactions.shape[1] != num_paths:
        raise AdapterContractError("Sionna 交互点或交互类型的路径轴不一致")
    return np.asarray(vertices, dtype=float), np.asarray(interactions)


def _single_link_path_angles(paths: Any, num_paths: int) -> np.ndarray:
    phi_r = _to_numpy(paths.phi_r)
    if phi_r.ndim == 3:
        angles = phi_r[0, 0]
    elif phi_r.ndim == 5:
        angles = phi_r[0, 0, 0, 0]
    else:
        raise AdapterContractError(f"不支持的 Sionna phi_r 形状：{phi_r.shape}")
    if angles.shape != (num_paths,):
        raise AdapterContractError("Sionna phi_r 路径轴与 CIR 不一致")
    return np.asarray(angles, dtype=float)


def _localization_scene_bounds(
    scene_config: dict[str, Any],
) -> tuple[float, float, float, float]:
    """读取固定定位范围；该函数不接收 UE 或真实路径信息。"""

    if "localization_bounds_m" not in scene_config:
        raise ValueError(
            "sionna_builtin 场景必须显式提供 "
            "scene.localization_bounds_m=[x_min, x_max, y_min, y_max]"
        )
    try:
        bounds = np.asarray(scene_config["localization_bounds_m"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("scene.localization_bounds_m 必须是四个有限数") from error
    if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
        raise ValueError("scene.localization_bounds_m 必须是四个有限数")
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    if not (x_min < x_max and y_min < y_max):
        raise ValueError(
            "scene.localization_bounds_m 必须满足 x_min < x_max 且 y_min < y_max"
        )
    return x_min, x_max, y_min, y_max


def _nearest_scene_wall_hit(
    scene: Scene2D,
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
) -> tuple[float, str, np.ndarray] | None:
    hits: list[tuple[float, str, np.ndarray]] = []
    for wall in scene.walls:
        hit = ray_segment_intersection(
            origin_xy, direction_xy, wall, min_distance_m=1e-6
        )
        if hit is not None:
            hits.append((float(hit[0]), wall.wall_id, np.asarray(hit[2], dtype=float)))
    return min(hits, key=lambda item: (item[0], item[1])) if hits else None


def _validate_reverse_scene_consistency(
    scene: Scene2D,
    *,
    bs_xy: np.ndarray,
    ue_xy: np.ndarray,
    path_metadata: dict[str, np.ndarray],
    tolerance_m: float,
) -> dict[str, Any]:
    """确认定位器从 BS 反向追踪时首先命中 Sionna 的真实表面。"""

    retained = np.asarray(path_metadata["retained_mask"], dtype=bool)
    angles = np.asarray(path_metadata["aoa_global_rad"], dtype=float)
    interactions = np.asarray(path_metadata["interactions"])
    vertices = np.asarray(path_metadata["vertices_m"], dtype=float)
    failures: list[str] = []
    los_count = 0
    reflected_count = 0
    x_min, x_max, y_min, y_max = scene.bounds_m

    def inside_fixed_bounds(point_xy: np.ndarray) -> bool:
        point = np.asarray(point_xy, dtype=float)
        return bool(
            point.shape == (2,)
            and np.all(np.isfinite(point))
            and x_min <= point[0] <= x_max
            and y_min <= point[1] <= y_max
        )

    if not inside_fixed_bounds(bs_xy):
        failures.append(f"BS 坐标 {np.asarray(bs_xy).tolist()} 超出固定定位区域")
    if not inside_fixed_bounds(ue_xy):
        failures.append(f"UE 坐标 {np.asarray(ue_xy).tolist()} 超出固定定位区域")

    for path_index in np.flatnonzero(retained):
        direction = np.asarray(
            [math.cos(angles[path_index]), math.sin(angles[path_index])], dtype=float
        )
        nearest = _nearest_scene_wall_hit(scene, bs_xy, direction)
        active_depths = np.flatnonzero(interactions[:, path_index] != 0)
        if active_depths.size == 0:
            los_count += 1
            ue_distance = float(np.linalg.norm(ue_xy - bs_xy))
            if nearest is not None and nearest[0] < ue_distance - tolerance_m:
                failures.append(
                    f"路径 {path_index} 是直射，但墙 {nearest[1]} 在 "
                    f"{nearest[0]:.3f} m 处提前挡住 UE（UE 距离 {ue_distance:.3f} m）"
                )
            continue

        reflected_count += 1
        outside_depths = [
            int(depth)
            for depth in active_depths
            if not inside_fixed_bounds(vertices[int(depth), path_index, :2])
        ]
        if outside_depths:
            failures.append(
                f"路径 {path_index} 的交互点层 {outside_depths} 超出固定定位区域"
            )
            continue
        expected = vertices[int(active_depths[-1]), path_index, :2]
        if nearest is None:
            failures.append(f"路径 {path_index} 的首个反向交互点没有对应二维墙")
            continue
        hit_error = float(np.linalg.norm(nearest[2] - expected))
        if hit_error > tolerance_m:
            failures.append(
                f"路径 {path_index} 应先命中 {expected.tolist()}，实际先命中墙 "
                f"{nearest[1]} 的 {nearest[2].tolist()}，相差 {hit_error:.3f} m"
            )

    if failures:
        details = "；".join(failures[:5])
        suffix = "" if len(failures) <= 5 else f"；另有 {len(failures) - 5} 条失败"
        raise RuntimeError(f"Sionna 原始网格与二维反向射线不一致：{details}{suffix}")
    return {
        "checked_path_count": int(np.sum(retained)),
        "los_path_count": los_count,
        "reflected_path_count": reflected_count,
        "position_tolerance_m": float(tolerance_m),
        "passed": True,
    }


def extract_planar_uplink_csi(
    paths: Any,
    frequencies_hz: np.ndarray,
    *,
    fixed_height_m: float,
    vertical_tolerance_m: float,
    max_reflections: int,
    bs_boresight_rad: float = 0.0,
    local_angle_min_rad: float = -math.pi / 2.0,
    local_angle_max_rad: float = math.pi / 2.0,
    front_facing_only: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """从 Sionna 路径保留二维、阵列正面可解释路径并合成绝对时延 CSI。

    ``phi_r`` 是地图全局到达角。先减去 ``bs_boresight_rad`` 得到阵列
    局部角，再按 MUSIC 的局部角窗口筛选。第一版用
    ``front_facing_only=True`` 显式排除 ULA 无法唯一解释的背面路径。
    """

    boresight = float(bs_boresight_rad)
    angle_min = float(local_angle_min_rad)
    angle_max = float(local_angle_max_rad)
    if not np.isfinite(boresight):
        raise ValueError("bs_boresight_rad 必须是有限数")
    if not (
        np.isfinite(angle_min)
        and np.isfinite(angle_max)
        and -math.pi / 2.0 <= angle_min < angle_max <= math.pi / 2.0
    ):
        raise ValueError("局部角范围必须位于 [-pi/2, pi/2] 且下界小于上界")
    if not isinstance(front_facing_only, (bool, np.bool_)):
        raise ValueError("front_facing_only 必须是布尔值")

    cir_coefficients, tau = sionna_paths_cir(
        paths,
        sampling_frequency=1.0,
        num_time_steps=1,
        out_type="numpy",
    )
    coefficients = _to_numpy(cir_coefficients)
    if coefficients.ndim != 6:
        raise AdapterContractError(
            "Sionna CIR 应为 [rx,rx_ant,tx,tx_ant,path,time]，"
            f"实际是 {coefficients.shape}"
        )
    if coefficients.shape[0] != 1 or coefficients.shape[2] != 1 or coefficients.shape[3] != 1:
        raise AdapterContractError("第一版 Sionna 入口只支持一个 UE 和一个 BS")
    path_coefficients = coefficients[0, :, 0, 0, :, 0]
    num_paths = path_coefficients.shape[1]
    absolute_delays = _single_link_tau(_to_numpy(tau), num_paths)
    vertices, interactions = _single_link_vertices(paths, num_paths)
    aoa_global_rad = _single_link_path_angles(paths, num_paths)

    active_interactions = interactions != 0
    interaction_order = np.sum(active_interactions, axis=0)
    planar_valid = np.isfinite(absolute_delays) & (absolute_delays >= 0.0)
    planar_valid &= interaction_order <= int(max_reflections)
    for path_index in range(num_paths):
        active = active_interactions[:, path_index]
        if not np.any(active):
            continue
        z_values = vertices[active, path_index, 2]
        if np.any(~np.isfinite(z_values)) or np.any(
            np.abs(z_values - fixed_height_m) > vertical_tolerance_m
        ):
            planar_valid[path_index] = False

    aoa_local_rad = np.asarray(
        [global_to_local_aoa(angle, boresight) for angle in aoa_global_rad],
        dtype=float,
    )
    angle_tolerance = 32.0 * np.finfo(float).eps
    front_facing_angle_mask = (
        np.isfinite(aoa_local_rad)
        & (aoa_local_rad >= angle_min - angle_tolerance)
        & (aoa_local_rad <= angle_max + angle_tolerance)
    )
    valid = (
        planar_valid & front_facing_angle_mask
        if bool(front_facing_only)
        else planar_valid.copy()
    )
    if int(np.sum(valid)) < 2:
        raise RuntimeError(
            "二维高度、反射次数和阵列正面角度筛选后少于两条 Sionna 路径；"
            "请调整 UE/BS 位置、阵列朝向、角度范围或射线采样数"
        )

    phase = np.exp(
        -2.0j
        * np.pi
        * absolute_delays[valid, np.newaxis]
        * np.asarray(frequencies_hz)[np.newaxis, :]
    )
    csi = path_coefficients[:, valid] @ phase
    metadata = {
        "retained_mask": valid,
        "planar_retained_mask_before_front_filter": planar_valid,
        "front_facing_angle_mask": front_facing_angle_mask,
        "absolute_delays_s": absolute_delays,
        "aoa_global_rad": aoa_global_rad,
        "aoa_local_rad": aoa_local_rad,
        "interaction_order": interaction_order,
        "vertices_m": vertices,
        "interactions": interactions,
        "front_facing_only": np.asarray(bool(front_facing_only)),
        "bs_boresight_rad": np.asarray(boresight),
        "local_angle_min_rad": np.asarray(angle_min),
        "local_angle_max_rad": np.asarray(angle_max),
    }
    return np.asarray(csi, dtype=np.complex128), metadata


def _path_selection_summary(path_metadata: dict[str, np.ndarray]) -> dict[str, Any]:
    """把真值掩码转成可写入生成清单的明确计数。"""

    retained = np.asarray(path_metadata["retained_mask"], dtype=bool)
    planar = np.asarray(
        path_metadata.get("planar_retained_mask_before_front_filter", retained),
        dtype=bool,
    )
    front = np.asarray(
        path_metadata.get("front_facing_angle_mask", np.ones_like(retained)),
        dtype=bool,
    )
    if retained.shape != planar.shape or retained.shape != front.shape:
        raise ValueError("路径筛选掩码形状必须一致")
    return {
        "rule": "planar_height_and_order_then_front_facing_local_angle_window",
        "front_facing_only": bool(
            np.asarray(path_metadata.get("front_facing_only", True)).item()
        ),
        "bs_boresight_rad": float(
            np.asarray(path_metadata.get("bs_boresight_rad", 0.0)).item()
        ),
        "local_angle_min_rad": float(
            np.asarray(path_metadata.get("local_angle_min_rad", -math.pi / 2.0)).item()
        ),
        "local_angle_max_rad": float(
            np.asarray(path_metadata.get("local_angle_max_rad", math.pi / 2.0)).item()
        ),
        "total_sionna_path_count": int(retained.size),
        "planar_path_count_before_front_filter": int(np.sum(planar)),
        "front_facing_angle_path_count": int(np.sum(front)),
        "retained_path_count_after_front_filter": int(np.sum(retained)),
    }


def _dataset_with_scene(dataset: Any) -> Any:
    if getattr(dataset, "scene", None) is not None:
        return dataset
    try:
        first = dataset[0]
    except (TypeError, IndexError, KeyError) as error:
        raise AdapterContractError("DeepMIMO 转换结果中没有可用的 scene") from error
    if getattr(first, "scene", None) is None:
        raise AdapterContractError("DeepMIMO 转换结果中没有可用的 scene")
    return first


def _local_scenario_name(name: str) -> str:
    """DeepMIMO 加载器会转小写，转换和保存阶段必须使用同一个名字。"""
    normalized = str(name).strip().lower()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ValueError("DeepMIMO 场景名必须是非空的单个目录名")
    return normalized


def _load_local_deepmimo_scene(deepmimo: Any, converted_name: str, scenario_store: Path) -> Any:
    """只加载已转换的本地场景；缺失时不调用会询问下载的加载器。"""
    name = _local_scenario_name(converted_name)
    local_directory = scenario_store / name
    if not local_directory.is_dir() or not (local_directory / "params.json").is_file():
        raise FileNotFoundError(
            f"DeepMIMO 本地转换产物不完整：{local_directory}；"
            "需要该目录及 params.json。已停止，不尝试在线下载。"
        )
    return _dataset_with_scene(deepmimo.load(name))


def _finalize_generation_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """为新生成清单补齐版本和由三个核心产物决定的批次号。"""

    finalized = {**manifest, "schema_version": 2}
    finalized["bundle_id"] = generation_bundle_id(finalized)
    return finalized


def _save_real_bundle(
    root: Path,
    online: OnlineMeasurement,
    *,
    ue_position_m: np.ndarray,
    clock_bias_s: float,
    csi_geometric: np.ndarray,
    injected_noise_std: float,
    path_metadata: dict[str, np.ndarray],
) -> dict[str, str]:
    online_dir = root / "data" / "online"
    truth_dir = root / "data" / "truth"
    online_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    online_path = online_dir / "measurement.npz"
    truth_path = truth_dir / "ground_truth.npz"
    path_selection = _path_selection_summary(path_metadata)
    np.savez_compressed(
        online_path,
        csi_observed=online.csi_observed,
        subcarrier_frequencies_hz=online.subcarrier_frequencies_hz,
        carrier_frequency_hz=np.asarray(online.carrier_frequency_hz),
        antenna_spacing_m=np.asarray(online.antenna_spacing_m),
        bs_position_m=online.bs_position_m,
        bs_boresight_rad=np.asarray(online.bs_boresight_rad),
    )
    (online_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "link_direction": "uplink_ue_to_bs",
                "contains_ground_truth": False,
                "allowed_for_localization": True,
                "absolute_delay_was_preserved": True,
                "csi_axes": ["snapshot", "bs_antenna", "subcarrier"],
                "array_path_prior": {
                    key: path_selection[key]
                    for key in (
                        "rule",
                        "front_facing_only",
                        "bs_boresight_rad",
                        "local_angle_min_rad",
                        "local_angle_max_rad",
                    )
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        truth_path,
        ue_position_m=ue_position_m,
        clock_bias_s=np.asarray(clock_bias_s),
        distance_bias_m=np.asarray(clock_bias_s * SPEED_OF_LIGHT_M_S),
        csi_geometric=csi_geometric,
        injected_noise_std=np.asarray(injected_noise_std),
        **path_metadata,
    )
    (truth_dir / "ground_truth.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "allowed_for_localization": False,
                "evaluation_only": True,
                "note": "包含 UE、注入偏差和 Sionna 路径真值",
                "path_selection": path_selection,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"online_npz": str(online_path), "truth_npz": str(truth_path)}


def generate_sionna_deepmimo_bundle(
    config: dict[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """运行 Sionna RT，转换成 DeepMIMO V4，并生成定位在线输入。"""

    simulation = config["simulation"]
    if bool(simulation.get("allow_overwrite", False)):
        raise ValueError(
            "simulation.allow_overwrite 必须为 false；生成阶段不允许覆盖旧批次"
        )
    root = Path(output_root).expanduser().resolve()
    with exclusive_output_root_lock(root):
        _ensure_fresh_generation_targets(root)
        return _generate_sionna_deepmimo_bundle_locked(config, root=root)


def _generate_sionna_deepmimo_bundle_locked(
    config: dict[str, Any], *, root: Path
) -> dict[str, Any]:
    """在调用方持有输出根锁且完成无覆盖检查后执行实际生成。"""

    scene_config = config["scene"]
    radio = config["radio"]
    simulation = config["simulation"]
    if scene_config["source"] != "sionna_builtin":
        raise ValueError("该入口要求 scene.source=sionna_builtin")
    localization_scene_bounds = _localization_scene_bounds(scene_config)
    config_snapshot = localization_config_snapshot(config)
    config_snapshot_path = root / "provenance" / "generation_config.json"
    _write_json_atomic(config_snapshot_path, config_snapshot)
    config_snapshot_record = {
        "path": str(config_snapshot_path),
        "source_path": config_snapshot["source_config_path"],
        "canonical_sha256": config_snapshot["canonical_sha256"],
        "file_sha256": file_sha256(config_snapshot_path),
    }
    sionna_rt = load_sionna_rt_module()
    runtime_info = _sionna_runtime_info(require_cuda=os.environ.get("TBC_REQUIRE_CUDA") == "1")
    _write_json_atomic(root / "provenance" / "generation_runtime.json", runtime_info)
    deepmimo = load_deepmimo_module()
    exporter_module = importlib.import_module("deepmimo.exporters.sionna_exporter")
    sionna_exporter = getattr(exporter_module, "sionna_exporter")

    scene_catalog = getattr(sionna_rt, "scene", None)
    scene_asset = getattr(scene_catalog, str(scene_config["name"]), None)
    if scene_asset is None:
        raise ValueError(f"Sionna RT 没有内置场景：{scene_config['name']}")
    scene = sionna_rt.load_scene(scene_asset)
    scene.frequency = float(radio["carrier_hz"])
    scene.bandwidth = float(radio["bandwidth_hz"])
    if hasattr(scene, "synthetic_array"):
        scene.synthetic_array = True

    PlanarArray = getattr(sionna_rt, "PlanarArray")
    Transmitter = getattr(sionna_rt, "Transmitter")
    Receiver = getattr(sionna_rt, "Receiver")
    PathSolver = getattr(sionna_rt, "PathSolver")
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=int(radio["num_bs_antennas"]),
        vertical_spacing=0.5,
        horizontal_spacing=float(radio["antenna_spacing_wavelength"]),
        pattern="iso",
        polarization="V",
    )
    fixed_height = float(scene_config["fixed_height_m"])
    ue_xy = np.asarray(simulation["ue_position_m"], dtype=float)
    bs_xy = np.asarray(simulation["bs_position_m"], dtype=float)
    transmitter = Transmitter(
        "ue_tx_0", position=[float(ue_xy[0]), float(ue_xy[1]), fixed_height]
    )
    receiver = Receiver(
        "bs_rx_0", position=[float(bs_xy[0]), float(bs_xy[1]), fixed_height]
    )
    if hasattr(receiver, "orientation"):
        receiver.orientation = [math.radians(float(radio["bs_boresight_deg"])), 0.0, 0.0]
    scene.add(transmitter)
    scene.add(receiver)

    rt_params = {
        "max_depth": int(scene_config["max_reflections"]),
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": False,
        "diffraction": False,
        "refraction": False,
        "samples_per_src": int(simulation["samples_per_source"]),
        "max_num_paths_per_src": int(
            simulation.get("max_num_paths_per_source", 100000)
        ),
        "synthetic_array": True,
        "seed": int(config["project"]["random_seed"]),
    }
    paths = PathSolver()(scene=scene, **rt_params)
    if int(paths.tau.shape[-1]) == 0:
        raise RuntimeError("Sionna RT 没有找到传播路径")

    scenario_name = _local_scenario_name(simulation["deepmimo_scenario_name"])
    export_dir = root / "deepmimo_source" / scenario_name
    if export_dir.exists() and any(export_dir.iterdir()):
        raise FileExistsError(
            f"射线源目录已存在且非空：{export_dir}；请换输出目录，避免覆盖旧数据"
        )
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    sionna_exporter(scene, paths, rt_params, str(export_dir))
    normalized_export_scalars = _normalize_deepmimo_export_scalars(export_dir)

    scenario_store = root / "deepmimo_scenarios"
    scenario_store.mkdir(parents=True, exist_ok=True)
    if hasattr(deepmimo, "config"):
        # DeepMIMO 4.0.5 的加载器读取 config，而转换器仍使用相对目录常量。
        # 在输出根目录内完成两步，确保它们指向同一个、可追溯的场景目录。
        deepmimo.config.set("scenarios_folder", "deepmimo_scenarios")
    with _working_directory(root):
        converted_name = deepmimo.convert(
            str(export_dir),
            scenario_name=scenario_name,
            overwrite=False,
            vis_scene=False,
            lossless=bool(simulation.get("deepmimo_lossless_scene", True)),
        )
        if not converted_name:
            raise RuntimeError("DeepMIMO 转换没有返回场景名")
        # 转换结果仍作为 DeepMIMO V4 数据产物保留并做一次加载校验；定位地图不能
        # 使用 dataset.scene，因为该转换器会先把连通组件简化为二维凸包。
        _load_local_deepmimo_scene(deepmimo, str(converted_name), scenario_store)

    frequencies = build_subcarrier_frequencies(
        bandwidth_hz=float(radio["bandwidth_hz"]),
        num_subcarriers=int(radio["num_subcarriers"]),
    )
    csi_geometric, path_metadata = extract_planar_uplink_csi(
        paths,
        frequencies,
        fixed_height_m=fixed_height,
        vertical_tolerance_m=float(scene_config["vertical_path_tolerance_m"]),
        max_reflections=int(scene_config["max_reflections"]),
        bs_boresight_rad=math.radians(float(radio["bs_boresight_deg"])),
        local_angle_min_rad=math.radians(float(config["music"]["angle_min_deg"])),
        local_angle_max_rad=math.radians(float(config["music"]["angle_max_deg"])),
        front_facing_only=bool(radio.get("front_facing_only", True)),
    )
    path_selection = _path_selection_summary(path_metadata)

    bev_resolution = float(scene_config["bev_resolution_m"])
    scene_2d = preprocess_sionna_exported_scene(
        export_dir,
        name=f"{scene_config['name']}_bev",
        fixed_height_m=fixed_height,
        bev_resolution_m=bev_resolution,
        bounds_m=localization_scene_bounds,
    )
    scene_consistency = _validate_reverse_scene_consistency(
        scene_2d,
        bs_xy=bs_xy,
        ue_xy=ue_xy,
        path_metadata=path_metadata,
        tolerance_m=max(1e-3, 0.5 * bev_resolution),
    )
    scene_artifacts = scene_2d.save(root / "scene")
    signal_rms = float(np.sqrt(np.mean(np.abs(csi_geometric) ** 2)))
    noise_std = signal_rms * 10.0 ** (-float(radio["snr_db"]) / 20.0)
    observed = apply_common_delay_bias(
        csi_geometric[np.newaxis, ...],
        frequencies,
        float(simulation["clock_bias_s"]),
        noise_std=noise_std,
        seed=int(config["project"]["random_seed"]),
    )
    online = OnlineMeasurement(
        csi_observed=observed,
        subcarrier_frequencies_hz=frequencies,
        carrier_frequency_hz=float(radio["carrier_hz"]),
        antenna_spacing_m=(
            SPEED_OF_LIGHT_M_S
            / float(radio["carrier_hz"])
            * float(radio["antenna_spacing_wavelength"])
        ),
        bs_position_m=bs_xy,
        bs_boresight_rad=math.radians(float(radio["bs_boresight_deg"])),
    )
    data_artifacts = _save_real_bundle(
        root,
        online,
        ue_position_m=ue_xy,
        clock_bias_s=float(simulation["clock_bias_s"]),
        csi_geometric=csi_geometric[np.newaxis, ...],
        injected_noise_std=noise_std,
        path_metadata=path_metadata,
    )
    manifest = _finalize_generation_manifest(
        {
            "stage": "sionna_rt_to_deepmimo_v4",
            "sionna_scene": str(scene_config["name"]),
            "deepmimo_scenario": str(converted_name),
            "deepmimo_scenario_store": str(scenario_store),
            "source_export_dir": str(export_dir),
            "absolute_delay_normalization": False,
            "link_direction": "uplink_ue_to_bs",
            "localization_scene_geometry_source": "sionna_exported_triangle_mesh",
            "localization_scene_bounds_m": list(localization_scene_bounds),
            "bounds_source": "fixed_config_not_ue_or_truth_paths",
            "scene_consistency": scene_consistency,
            "rt_params": rt_params,
            "deepmimo_export_scalars_normalized": normalized_export_scalars,
            "scene_artifacts": scene_artifacts,
            "data_artifacts": data_artifacts,
            "config_snapshot": config_snapshot_record,
            "artifact_hashes": {
                "scene_json": artifact_record(scene_artifacts["scene_json"]),
                "online_measurement": artifact_record(data_artifacts["online_npz"]),
                "ground_truth": artifact_record(data_artifacts["truth_npz"]),
            },
            "path_selection": path_selection,
            "planar_path_count_before_front_filter": path_selection[
                "planar_path_count_before_front_filter"
            ],
            "retained_path_count": path_selection[
                "retained_path_count_after_front_filter"
            ],
        }
    )
    manifest_path = root / "generation_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return {**manifest, "manifest": str(manifest_path)}


__all__ = ["extract_planar_uplink_csi", "generate_sionna_deepmimo_bundle"]
