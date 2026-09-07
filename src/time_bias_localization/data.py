"""离线仿真数据生成与在线输入/真值隔离。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from .candidates import global_to_local_aoa
from .constants import SPEED_OF_LIGHT_M_S
from .raytrace2d import GeometricPath2D, enumerate_specular_paths
from .scene import Scene2D
from .signal import apply_common_delay_bias, synthesize_ula_csi


@dataclass(frozen=True)
class OnlineMeasurement:
    """定位程序被允许读取的全部测量信息。"""

    csi_observed: np.ndarray
    subcarrier_frequencies_hz: np.ndarray
    carrier_frequency_hz: float
    antenna_spacing_m: float
    bs_position_m: np.ndarray
    bs_boresight_rad: float


def build_subcarrier_frequencies(*, bandwidth_hz: float, num_subcarriers: int) -> np.ndarray:
    """生成从零开始、严格递增的基带子载波频率。"""

    if bandwidth_hz <= 0.0 or num_subcarriers < 2:
        raise ValueError("带宽必须为正，子载波数至少为 2")
    spacing_hz = float(bandwidth_hz) / int(num_subcarriers)
    return np.arange(int(num_subcarriers), dtype=float) * spacing_hz


def _select_paths(
    paths: Sequence[GeometricPath2D],
    *,
    count: int,
    bs_boresight_rad: float,
    local_angle_min_rad: float,
    local_angle_max_rad: float,
    front_facing_only: bool = True,
) -> list[GeometricPath2D]:
    eligible: list[GeometricPath2D] = []
    for path in paths:
        local_angle = global_to_local_aoa(
            math.radians(path.arrival_aoa_deg), bs_boresight_rad
        )
        in_local_window = local_angle_min_rad <= local_angle <= local_angle_max_rad
        if not front_facing_only or in_local_window:
            eligible.append(path)
    if len(eligible) < count:
        raise RuntimeError(
            f"阵列正面角度范围内只有 {len(eligible)} 条路径，"
            f"少于配置要求的 {count} 条"
        )

    # 先覆盖不同反射次数，再按路径长度补齐，避免离线例子只选到一类几何。
    selected: list[GeometricPath2D] = []
    for order in (0, 1, 2):
        order_paths = [path for path in eligible if path.reflection_order == order]
        if order_paths and len(selected) < count:
            selected.append(min(order_paths, key=lambda item: (item.length_m, item.path_id)))
    for path in eligible:
        if path not in selected and len(selected) < count:
            selected.append(path)
    return selected[:count]


def generate_synthetic_measurement(
    scene: Scene2D,
    config: dict[str, Any],
) -> tuple[OnlineMeasurement, dict[str, Any]]:
    """由二维镜面路径生成带公共时延偏差的上行 CSI。"""

    radio = config["radio"]
    simulation = config["simulation"]
    music = config["music"]
    project = config["project"]
    bs_position = np.asarray(simulation["bs_position_m"], dtype=float)
    ue_position = np.asarray(simulation["ue_position_m"], dtype=float)
    bs_boresight_rad = math.radians(float(radio.get("bs_boresight_deg", 0.0)))
    front_facing_only = bool(radio.get("front_facing_only", True))
    local_angle_min_rad = math.radians(float(music["angle_min_deg"]))
    local_angle_max_rad = math.radians(float(music["angle_max_deg"]))
    all_paths = enumerate_specular_paths(
        scene,
        source_m=ue_position,
        receiver_m=bs_position,
        max_reflections=int(scene_max_reflections(config)),
    )
    selected = _select_paths(
        all_paths,
        count=int(music["num_paths"]),
        bs_boresight_rad=bs_boresight_rad,
        local_angle_min_rad=local_angle_min_rad,
        local_angle_max_rad=local_angle_max_rad,
        front_facing_only=front_facing_only,
    )
    front_facing_path_count = sum(
        local_angle_min_rad
        <= global_to_local_aoa(
            math.radians(path.arrival_aoa_deg), bs_boresight_rad
        )
        <= local_angle_max_rad
        for path in all_paths
    )

    carrier_hz = float(radio["carrier_hz"])
    frequencies_hz = build_subcarrier_frequencies(
        bandwidth_hz=float(radio["bandwidth_hz"]),
        num_subcarriers=int(radio["num_subcarriers"]),
    )
    wavelength_m = SPEED_OF_LIGHT_M_S / carrier_hz
    antenna_spacing_m = wavelength_m * float(radio["antenna_spacing_wavelength"])
    amplitudes = np.asarray(simulation["path_amplitudes"], dtype=float)
    if amplitudes.size < len(selected):
        raise ValueError("path_amplitudes 的数量少于要生成的路径数")
    amplitudes = amplitudes[: len(selected)]
    rng = np.random.default_rng(int(project["random_seed"]))
    num_snapshots = int(radio.get("num_snapshots", 1))
    snapshot_csi: list[np.ndarray] = []
    coefficient_history: list[np.ndarray] = []
    local_angles = np.asarray(
        [
            global_to_local_aoa(math.radians(path.arrival_aoa_deg), bs_boresight_rad)
            for path in selected
        ]
    )
    path_delays = np.asarray([path.delay_s for path in selected])
    base_phases = rng.uniform(-np.pi, np.pi, size=len(selected))
    for snapshot_index in range(num_snapshots):
        if snapshot_index == 0:
            phases = base_phases
        else:
            # 可选多快照模式；默认静态场景只有一个快照。
            phases = base_phases + rng.normal(0.0, 0.05, size=len(selected))
        coefficients = amplitudes * np.exp(1j * phases)
        coefficient_history.append(coefficients)
        snapshot_csi.append(
            synthesize_ula_csi(
                path_aoa_rad=local_angles,
                path_delay_s=path_delays,
                path_coefficients=coefficients,
                subcarrier_frequencies_hz=frequencies_hz,
                num_bs_antennas=int(radio["num_bs_antennas"]),
                carrier_frequency_hz=carrier_hz,
                antenna_spacing_m=antenna_spacing_m,
            )
        )
    csi_geometric = np.stack(snapshot_csi, axis=0)
    signal_rms = float(np.sqrt(np.mean(np.abs(csi_geometric) ** 2)))
    noise_std = signal_rms * 10.0 ** (-float(radio["snr_db"]) / 20.0)
    clock_bias_s = float(simulation["clock_bias_s"])
    csi_observed = apply_common_delay_bias(
        csi_geometric,
        frequencies_hz,
        clock_bias_s,
        noise_std=noise_std,
        seed=int(project["random_seed"]) + 1,
    )
    online = OnlineMeasurement(
        csi_observed=csi_observed,
        subcarrier_frequencies_hz=frequencies_hz,
        carrier_frequency_hz=carrier_hz,
        antenna_spacing_m=antenna_spacing_m,
        bs_position_m=bs_position,
        bs_boresight_rad=bs_boresight_rad,
    )
    truth = {
        "ue_position_m": ue_position,
        "clock_bias_s": clock_bias_s,
        "distance_bias_m": clock_bias_s * SPEED_OF_LIGHT_M_S,
        "csi_geometric": csi_geometric,
        "injected_noise_std": noise_std,
        "path_coefficients": np.stack(coefficient_history, axis=0),
        "paths": selected,
        "path_selection": {
            "rule": "planar_height_and_order_then_front_facing_local_angle_window",
            "front_facing_only": front_facing_only,
            "bs_boresight_rad": bs_boresight_rad,
            "local_angle_min_rad": local_angle_min_rad,
            "local_angle_max_rad": local_angle_max_rad,
            "planar_path_count_before_front_filter": len(all_paths),
            "front_facing_angle_path_count": front_facing_path_count,
            "retained_path_count_after_front_filter": len(selected),
            "requested_generated_path_count": int(music["num_paths"]),
        },
    }
    return online, truth


def scene_max_reflections(config: dict[str, Any]) -> int:
    return int(config["scene"]["max_reflections"])


def _stage_binary_file(path: Path, writer: Any) -> Path:
    """在目标同目录写好并同步一个临时文件；失败文件原样保留。"""

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f".tmp{path.suffix}",
    )
    temporary_path = Path(temporary_name)
    with os.fdopen(file_descriptor, "w+b") as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary_path


def _backup_path(path: Path) -> Path:
    """返回同目录、尚不存在的唯一备份名，但不提前创建文件。"""

    while True:
        candidate = path.with_name(
            f".{path.name}.{uuid4().hex}.backup{path.suffix}"
        )
        if not (candidate.exists() or candidate.is_symlink()):
            return candidate


def _sync_directories(paths: Sequence[Path]) -> None:
    for directory in {path.parent for path in paths}:
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _publish_staged_files(
    staged_files: Sequence[tuple[Path, Path]], *, allow_overwrite: bool
) -> None:
    """发布一组已落盘文件；失败时恢复发布前的整组正式目标。"""

    conflicts = [
        target
        for target, _ in staged_files
        if target.exists() or target.is_symlink()
    ]
    if conflicts and not allow_overwrite:
        joined = "、".join(str(path) for path in conflicts)
        raise FileExistsError(f"保存目标在准备期间被创建，拒绝覆盖：{joined}")

    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        if allow_overwrite:
            for target, _ in staged_files:
                if not (target.exists() or target.is_symlink()):
                    continue
                backup = _backup_path(target)
                os.replace(target, backup)
                backups.append((target, backup))
        for target, temporary in staged_files:
            os.replace(temporary, target)
            published.append((target, temporary))
        _sync_directories([target for target, _ in staged_files])
    except BaseException:
        # 先把本轮已经发布的新文件移回临时名，再恢复全部旧文件。这样
        # 既不暴露新旧混合数据，也不删除失败证据。
        for target, temporary in reversed(published):
            if (target.exists() or target.is_symlink()) and not (
                temporary.exists() or temporary.is_symlink()
            ):
                try:
                    os.replace(target, temporary)
                except OSError:
                    pass
        for target, backup in reversed(backups):
            if (backup.exists() or backup.is_symlink()) and not (
                target.exists() or target.is_symlink()
            ):
                try:
                    os.replace(backup, target)
                except OSError:
                    pass
        raise
    else:
        # 调用方明确允许覆盖且整组新文件已经同步落盘后，旧文件备份才可清理。
        for _, backup in backups:
            try:
                backup.unlink()
            except OSError:
                pass


def save_measurement_bundle(
    output_root: str | Path,
    online: OnlineMeasurement,
    truth: dict[str, Any],
    *,
    allow_overwrite: bool = False,
) -> dict[str, str]:
    """原子保存在线输入和评估真值；已有任一目标时整体拒绝。"""

    root = Path(output_root).expanduser().resolve()
    online_dir = root / "online"
    truth_dir = root / "truth"
    online_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    online_npz = online_dir / "measurement.npz"
    online_manifest = online_dir / "manifest.json"
    truth_npz = truth_dir / "ground_truth.npz"
    truth_json = truth_dir / "ground_truth.json"
    if not isinstance(allow_overwrite, bool):
        raise ValueError("allow_overwrite 必须是布尔值")
    targets = (online_npz, online_manifest, truth_npz, truth_json)
    conflicts = [path for path in targets if path.exists() or path.is_symlink()]
    if conflicts and not allow_overwrite:
        joined = "、".join(str(path) for path in conflicts)
        raise FileExistsError(f"测量数据目标已存在，拒绝覆盖整组文件：{joined}")

    manifest = {
        "schema_version": 1,
        "link_direction": "uplink_ue_to_bs",
        "allowed_for_localization": True,
        "contains_ground_truth": False,
        "csi_shape": list(online.csi_observed.shape),
        "csi_axes": ["snapshot", "bs_antenna", "subcarrier"],
        "frequency_unit": "Hz",
        "position_unit": "m",
        "delay_convention": "observed_delay=geometric_delay+common_bias+noise",
        "array_path_prior": {
            key: truth["path_selection"][key]
            for key in (
                "rule",
                "front_facing_only",
                "bs_boresight_rad",
                "local_angle_min_rad",
                "local_angle_max_rad",
            )
        },
    }
    truth_metadata = {
        "schema_version": 1,
        "allowed_for_localization": False,
        "evaluation_only": True,
        "path_selection": truth["path_selection"],
        "paths": [
            {
                "path_id": path.path_id,
                "reflection_order": path.reflection_order,
                "interaction_wall_ids": list(path.interaction_wall_ids),
                "interaction_points_m": [list(point) for point in path.interaction_points_m],
                "length_m": path.length_m,
                "delay_s": path.delay_s,
                "arrival_aoa_global_deg": path.arrival_aoa_deg,
            }
            for path in truth["paths"]
        ],
    }
    online_manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2
    ).encode("utf-8")
    truth_metadata_bytes = json.dumps(
        truth_metadata, ensure_ascii=False, indent=2
    ).encode("utf-8")

    staged_files: list[tuple[Path, Path]] = []
    staged_files.append(
        (
            online_npz,
            _stage_binary_file(
                online_npz,
                lambda handle: np.savez_compressed(
                    handle,
                    csi_observed=online.csi_observed,
                    subcarrier_frequencies_hz=online.subcarrier_frequencies_hz,
                    carrier_frequency_hz=np.asarray(online.carrier_frequency_hz),
                    antenna_spacing_m=np.asarray(online.antenna_spacing_m),
                    bs_position_m=online.bs_position_m,
                    bs_boresight_rad=np.asarray(online.bs_boresight_rad),
                ),
            ),
        )
    )
    staged_files.append(
        (
            online_manifest,
            _stage_binary_file(
                online_manifest, lambda handle: handle.write(online_manifest_bytes)
            ),
        )
    )
    staged_files.append(
        (
            truth_npz,
            _stage_binary_file(
                truth_npz,
                lambda handle: np.savez_compressed(
                    handle,
                    ue_position_m=truth["ue_position_m"],
                    clock_bias_s=np.asarray(truth["clock_bias_s"]),
                    distance_bias_m=np.asarray(truth["distance_bias_m"]),
                    csi_geometric=truth["csi_geometric"],
                    injected_noise_std=np.asarray(truth["injected_noise_std"]),
                    path_coefficients=truth["path_coefficients"],
                    path_delays_s=np.asarray(
                        [path.delay_s for path in truth["paths"]]
                    ),
                    path_aoa_global_deg=np.asarray(
                        [path.arrival_aoa_deg for path in truth["paths"]]
                    ),
                ),
            ),
        )
    )
    staged_files.append(
        (
            truth_json,
            _stage_binary_file(
                truth_json, lambda handle: handle.write(truth_metadata_bytes)
            ),
        )
    )
    _publish_staged_files(staged_files, allow_overwrite=allow_overwrite)
    return {
        "online_npz": str(online_npz),
        "online_manifest": str(online_manifest),
        "truth_npz": str(truth_npz),
        "truth_json": str(truth_json),
    }


def load_online_measurement_bytes(
    data_bytes: bytes, *, source_path: str | Path
) -> OnlineMeasurement:
    """从已捕获字节加载在线 NPZ，同时严格执行六字段白名单。"""

    resolved = Path(source_path).expanduser().resolve()
    if "ground_truth" in resolved.name or resolved.parent.name == "truth":
        raise ValueError("定位模块拒绝读取真值目录")
    with np.load(BytesIO(data_bytes), allow_pickle=False) as data:
        expected_fields = {
            "csi_observed",
            "subcarrier_frequencies_hz",
            "carrier_frequency_hz",
            "antenna_spacing_m",
            "bs_position_m",
            "bs_boresight_rad",
        }
        actual_fields = set(data.files)
        missing = sorted(expected_fields.difference(actual_fields))
        unexpected = sorted(actual_fields.difference(expected_fields))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"缺少字段：{missing}")
            if unexpected:
                details.append(f"不允许的额外字段：{unexpected}")
            raise ValueError("在线输入字段集合不符合契约；" + "；".join(details))

        raw_csi = np.asarray(data["csi_observed"])
        raw_frequencies = np.asarray(data["subcarrier_frequencies_hz"])
        raw_carrier = np.asarray(data["carrier_frequency_hz"])
        raw_spacing = np.asarray(data["antenna_spacing_m"])
        raw_bs_position = np.asarray(data["bs_position_m"])
        raw_boresight = np.asarray(data["bs_boresight_rad"])

        if raw_csi.ndim != 3:
            raise ValueError(
                "csi_observed 原始形状必须是三维 "
                "(snapshot, bs_antenna, subcarrier)"
            )
        if raw_frequencies.ndim != 1:
            raise ValueError("subcarrier_frequencies_hz 原始形状必须是一维")
        for field_name, value in (
            ("carrier_frequency_hz", raw_carrier),
            ("antenna_spacing_m", raw_spacing),
            ("bs_boresight_rad", raw_boresight),
        ):
            if value.shape != ():
                raise ValueError(f"{field_name} 原始形状必须是标量 shape=()")
        if raw_bs_position.shape != (2,):
            raise ValueError("bs_position_m 原始形状必须是 (2,)")

        try:
            csi_real_finite = bool(np.all(np.isfinite(raw_csi.real)))
            csi_imag_finite = bool(np.all(np.isfinite(raw_csi.imag)))
        except TypeError as error:
            raise ValueError("csi_observed 必须是数值数组") from error
        if not csi_real_finite or not csi_imag_finite:
            raise ValueError("csi_observed 的实部和虚部必须全部为有限值")

        real_fields = (
            ("subcarrier_frequencies_hz", raw_frequencies),
            ("carrier_frequency_hz", raw_carrier),
            ("antenna_spacing_m", raw_spacing),
            ("bs_position_m", raw_bs_position),
            ("bs_boresight_rad", raw_boresight),
        )
        for field_name, value in real_fields:
            if np.iscomplexobj(value):
                raise ValueError(f"{field_name} 必须是实数")
            try:
                all_finite = bool(np.all(np.isfinite(value)))
            except TypeError as error:
                raise ValueError(f"{field_name} 必须是数值") from error
            if not all_finite:
                raise ValueError(f"{field_name} 必须全部为有限值")

        try:
            frequencies = np.asarray(raw_frequencies, dtype=float)
            carrier_frequency = float(raw_carrier.item())
            antenna_spacing = float(raw_spacing.item())
            bs_position = np.asarray(raw_bs_position, dtype=float)
            bs_boresight = float(raw_boresight.item())
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("在线输入数值无法转换为浮点数") from error
        return OnlineMeasurement(
            csi_observed=raw_csi,
            subcarrier_frequencies_hz=frequencies,
            carrier_frequency_hz=carrier_frequency,
            antenna_spacing_m=antenna_spacing,
            bs_position_m=bs_position,
            bs_boresight_rad=bs_boresight,
        )


def load_online_measurement(path: str | Path) -> OnlineMeasurement:
    """只加载字段集合严格符合在线契约的 NPZ。"""

    resolved = Path(path).expanduser().resolve()
    return load_online_measurement_bytes(resolved.read_bytes(), source_path=resolved)


__all__ = [
    "OnlineMeasurement",
    "build_subcarrier_frequencies",
    "generate_synthetic_measurement",
    "load_online_measurement",
    "load_online_measurement_bytes",
    "save_measurement_bundle",
]
