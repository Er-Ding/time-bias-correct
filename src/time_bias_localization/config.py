"""配置读取、默认值和基础校验。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"random_seed": 20260904},
    "compute": {"backend": "numpy", "device_id": 0, "batch_size": 4, "angle_chunk_size": 32},
    "scene": {
        "source": "synthetic_room",
        "name": "offline_room",
        "bounds_m": [0.0, 20.0, 0.0, 14.0],
        "fixed_height_m": 1.5,
        "bev_resolution_m": 0.05,
        "max_reflections": 2,
        "max_diffractions": 0,
    },
    "radio": {
        "carrier_hz": 3.5e9,
        "bandwidth_hz": 400e6,
        "num_subcarriers": 96,
        "num_bs_antennas": 12,
        "antenna_spacing_wavelength": 0.5,
        "bs_boresight_deg": 0.0,
        "front_facing_only": True,
        "num_snapshots": 1,
        "snr_db": 35.0,
    },
    "simulation": {
        "bs_position_m": [2.0, 7.0],
        "ue_position_m": [14.0, 4.0],
        "clock_bias_s": 25e-9,
        "path_amplitudes": [1.0, 0.75, 0.55],
    },
    "music": {
        "num_paths": 3,
        "signal_subspace_rank": 3,
        "angle_min_deg": -89.0,
        "angle_max_deg": 89.0,
        "angle_step_deg": 1.0,
        "delay_min_s": 0.0,
        "delay_max_s": 160e-9,
        "delay_step_s": 1e-9,
        "min_angle_separation_deg": 5.0,
        "min_delay_separation_s": 3e-9,
        "spectrum_sampling": {
            "samples_per_peak": 128,
            "aoa_half_width_grid_steps": 1.5,
            "delay_half_width_grid_steps": 1.5,
            "local_grid_points_per_axis": 25,
            "spectrum_power": 1.0,
            "uniform_mixture": 0.1,
            "include_nominal": True,
        },
        "spatial_subarray_size": 8,
        "frequency_subarray_size": 32,
        "diagonal_loading": 1e-8,
    },
    "localization": {
        "bias_min_s": -80e-9,
        "bias_max_s": 80e-9,
        # 初始候选点共用的公开参考偏差，不是已知真值或最终估计值。
        "initial_reference_bias_s": 0.0,
        # 仅为旧生成配置快照保持稳定；定位白名单不接收这两个未使用字段。
        "candidate_angle_samples": 5,
        "candidate_delay_samples": 5,
        "huber_delta_m": 0.75,
        "max_iterations": 20,
        "max_seed_pairs": 100000,
        # DBSCAN 的邻域距离 eps，允许一个连续簇的总跨度超过此值。
        "candidate_cluster_radius_m": 1.5,
        "candidate_cluster_min_samples": 5,
        "diffraction_directions_per_sample": 4,
        "diffraction_angle_tolerance_deg": 3.0,
        "diffraction_coverage_distance_m": 1.0,
        "diffraction_representative_policy": "coverage",
        # 兼容旧配置快照；点聚类主流程不再使用方向阈值。
        "candidate_direction_radius_deg": 5.0,
    },
    "output": {"root": "outputs/offline_demo"},
}

_LOCALIZATION_SECTION_FIELDS: dict[str, frozenset[str]] = {
    "project": frozenset({"random_seed"}),
    "compute": frozenset({"backend", "device_id", "batch_size", "angle_chunk_size"}),
    "scene": frozenset(
        {
            "source",
            "name",
            "bounds_m",
            "localization_bounds_m",
            "fixed_height_m",
            "bev_resolution_m",
            "max_reflections",
            "max_diffractions",
        }
    ),
    "radio": frozenset(
        {
            "carrier_hz",
            "bandwidth_hz",
            "num_subcarriers",
            "num_bs_antennas",
            "antenna_spacing_wavelength",
            "bs_position_m",
            "bs_boresight_deg",
            "front_facing_only",
            "num_snapshots",
        }
    ),
    "music": frozenset(
        {
            "num_paths",
            "signal_subspace_rank",
            "angle_min_deg",
            "angle_max_deg",
            "angle_step_deg",
            "delay_min_s",
            "delay_max_s",
            "delay_step_s",
            "min_angle_separation_deg",
            "min_delay_separation_s",
            "spectrum_sampling",
            "spatial_subarray_size",
            "frequency_subarray_size",
            "diagonal_loading",
        }
    ),
    "localization": frozenset(
        {
            "bias_min_s",
            "bias_max_s",
            "initial_reference_bias_s",
            "candidate_cluster_radius_m",
            "candidate_cluster_min_samples",
            "diffraction_directions_per_sample",
            "diffraction_angle_tolerance_deg",
            "diffraction_coverage_distance_m",
            "diffraction_representative_policy",
            "candidate_direction_radius_deg",
            "huber_delta_m",
            "max_iterations",
            "max_seed_pairs",
        }
    ),
    "output": frozenset({"root"}),
}
_LOCALIZATION_SECTION_NAMES = tuple(_LOCALIZATION_SECTION_FIELDS)

_REMOVED_MUSIC_FIELDS = frozenset({
    "association_max_normalized_distance", "false_peak_penalty", "missed_peak_penalty",
    "uncertainty_repeats", "uncertainty_extra_peaks",
    "uncertainty_min_relative_height", "uncertainty_noise_scale",
})


def _reject_removed_music_fields(config: Mapping[str, Any]) -> None:
    music = config.get("music", {})
    if isinstance(music, Mapping):
        removed = sorted(_REMOVED_MUSIC_FIELDS.intersection(music))
        if removed:
            raise ValueError(
                f"CSI 重复加噪流程已移除，旧 music 参数不能继续使用：{removed}；"
                "请改用 music.spectrum_sampling 配置谱面采样"
            )


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError("配置文件顶层必须是键值映射")
    return config_path, loaded


def _localization_defaults() -> dict[str, Any]:
    # 默认配置同时服务于数据生成，其中含有 snr_db 等仿真注入参数。
    # 定位默认值必须逐字段过滤，不能把生成专用参数悄悄带进定位进程。
    return {
        section_name: {
            field_name: deepcopy(field_value)
            for field_name, field_value in DEFAULT_CONFIG[section_name].items()
            if field_name in allowed_fields
        }
        for section_name, allowed_fields in _LOCALIZATION_SECTION_FIELDS.items()
    }


def _apply_legacy_signal_rank(config: dict[str, Any], loaded: dict[str, Any]) -> None:
    # 旧配置只有 num_paths 时，保持原先“路径数同时作为子空间阶数”的行为。
    # 新配置显式填写 signal_subspace_rank 后，两者才独立变化。
    loaded_music = loaded.get("music")
    if isinstance(loaded_music, dict) and "signal_subspace_rank" not in loaded_music:
        config["music"]["signal_subspace_rank"] = config["music"]["num_paths"]


def _validate_localization_input_fields(loaded: Mapping[str, Any]) -> None:
    """拒绝定位配置各段中未定义的字段，防止把真值藏在合法段名下。"""

    _reject_removed_music_fields(loaded)
    for section_name, allowed_fields in _LOCALIZATION_SECTION_FIELDS.items():
        if section_name not in loaded:
            continue
        section = loaded[section_name]
        if not isinstance(section, Mapping):
            raise ValueError(f"定位专用配置的 {section_name} 必须是键值映射")
        unsupported = sorted(
            (str(name) for name in section if name not in allowed_fields)
        )
        if unsupported:
            raise ValueError(
                f"定位专用配置的 {section_name} 包含未定义字段：{unsupported}"
            )


def _filter_localization_sections(source: Mapping[str, Any]) -> dict[str, Any]:
    """只复制定位白名单字段；完整生成配置中的其他字段不会进入定位器。"""

    selected: dict[str, Any] = {}
    for section_name, allowed_fields in _LOCALIZATION_SECTION_FIELDS.items():
        if section_name not in source:
            continue
        section = source[section_name]
        if not isinstance(section, Mapping):
            raise ValueError(f"配置的 {section_name} 必须是键值映射")
        selected[section_name] = {
            name: deepcopy(value)
            for name, value in section.items()
            if name in allowed_fields
        }
    return selected


def load_config(path: str | Path) -> dict[str, Any]:
    """读取包含仿真参数的完整配置。"""

    config_path, loaded = _load_yaml_mapping(path)
    config = _merge_dict(DEFAULT_CONFIG, loaded)
    _apply_legacy_signal_rank(config, loaded)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def load_localization_config(path: str | Path) -> dict[str, Any]:
    """读取不含仿真真值的定位专用配置。

    定位入口只接受算法和公开先验参数。若文件出现 ``simulation``，即使其中
    字段看似没有被求解器使用，也直接拒绝，避免定位进程接触 UE 位置或注入
    时钟偏差。
    """

    config_path, loaded = _load_yaml_mapping(path)
    if "simulation" in loaded:
        raise ValueError("定位专用配置禁止包含 simulation 仿真真值")
    unsupported = sorted(set(loaded).difference(_LOCALIZATION_SECTION_NAMES))
    if unsupported:
        raise ValueError(f"定位专用配置包含不支持的顶层字段：{unsupported}")
    _validate_localization_input_fields(loaded)
    isolated_input = deepcopy(loaded)
    isolated_input["_config_path"] = str(config_path)
    config = localization_config_view(isolated_input)
    validate_localization_config(config)
    return config


def validate_localization_config(config: Mapping[str, Any]) -> None:
    """严格校验可直接交给定位器的公开配置，不允许夹带真值或未知字段。"""

    if not isinstance(config, Mapping):
        raise TypeError("定位专用配置必须是键值映射")
    if "simulation" in config:
        raise ValueError(
            "定位专用配置禁止包含 simulation 仿真真值；"
            "必须先生成定位专用配置再调用 localize"
        )
    allowed_top_level = set(_LOCALIZATION_SECTION_NAMES) | {"_config_path"}
    unsupported = sorted(str(name) for name in config if name not in allowed_top_level)
    if unsupported:
        raise ValueError(f"定位专用配置包含不支持的顶层字段：{unsupported}")
    # 计算设备属于可选的执行设置；旧的公开定位配置继续使用 NumPy。
    missing = sorted((set(_LOCALIZATION_SECTION_NAMES) - {"compute"}).difference(config))
    if missing:
        raise ValueError(f"定位专用配置缺少必需段：{missing}")
    _validate_localization_input_fields(config)
    _validate_localization_sections(dict(config))
    if "bs_position_m" not in config["radio"]:
        raise ValueError("定位专用配置必须提供已知的 BS 二维位置 radio.bs_position_m")


def localization_config_view(full_config: dict[str, Any]) -> dict[str, Any]:
    """从完整配置生成不含 ``simulation`` 的定位配置深拷贝。"""

    if not isinstance(full_config, dict):
        raise TypeError("full_config 必须是键值映射")
    _reject_removed_music_fields(full_config)
    selected = _filter_localization_sections(full_config)
    # 旧生成配置把 BS 与 UE 一起放在 simulation。BS 是定位时已知的接收端
    # 先验，因此只提升 BS 位置；UE 位置、时钟偏差和信噪比均不会进入定位视图。
    selected.setdefault("radio", {})
    if "bs_position_m" not in selected["radio"]:
        simulation = full_config.get("simulation")
        if isinstance(simulation, Mapping) and "bs_position_m" in simulation:
            selected["radio"]["bs_position_m"] = deepcopy(
                simulation["bs_position_m"]
            )
    config = _merge_dict(_localization_defaults(), selected)
    _apply_legacy_signal_rank(config, selected)
    _validate_localization_sections(config)
    if "_config_path" in full_config:
        config["_config_path"] = deepcopy(full_config["_config_path"])
    return config


def _finite_float(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数")
    return number


def _positive_float(name: str, value: Any) -> float:
    number = _finite_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} 必须为正数")
    return number


def _nonnegative_integer(name: str, value: Any) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} 必须是非负整数") from error
    if isinstance(value, bool) or integer != value or integer < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return integer


def _positive_integer(name: str, value: Any) -> int:
    integer = _nonnegative_integer(name, value)
    if integer < 1:
        raise ValueError(f"{name} 必须是正整数")
    return integer


def _validated_bounds(name: str, value: Any) -> list[float]:
    try:
        if len(value) != 4:
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 必须为 [x_min, x_max, y_min, y_max]") from error
    bounds = [_finite_float(name, item) for item in value]
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError(f"{name} 必须为 [x_min, x_max, y_min, y_max]")
    return bounds


def validate_config(config: dict[str, Any]) -> None:
    """只校验会造成物理含义错误或数组越界的配置。"""

    _validate_localization_sections(config)
    _validate_simulation_section(config)


def _validate_localization_sections(config: dict[str, Any]) -> None:
    """校验生成和定位共同使用、且不含仿真真值的配置段。"""

    _reject_removed_music_fields(config)
    compute = config.get("compute", DEFAULT_CONFIG["compute"])
    if not isinstance(compute, Mapping):
        raise ValueError("compute 必须是键值映射")
    if compute.get("backend", "numpy") not in ("numpy", "cuda"):
        raise ValueError("compute.backend 只能是 numpy 或 cuda")
    _nonnegative_integer("compute.device_id", compute.get("device_id", 0))
    _positive_integer("compute.batch_size", compute.get("batch_size", 4))
    _positive_integer("compute.angle_chunk_size", compute.get("angle_chunk_size", 32))

    radio = config["radio"]
    scene = config["scene"]
    localization = config["localization"]
    music = config["music"]

    for field_name in ("source", "name"):
        field_value = scene.get(field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"scene.{field_name} 必须是非空字符串")
    output_root = config["output"].get("root")
    if not isinstance(output_root, str) or not output_root.strip():
        raise ValueError("output.root 必须是非空字符串")

    num_subcarriers = _positive_integer("num_subcarriers", radio["num_subcarriers"])
    num_antennas = _positive_integer("num_bs_antennas", radio["num_bs_antennas"])
    if num_subcarriers < 8:
        raise ValueError("num_subcarriers 至少为 8")
    if num_antennas < 2:
        raise ValueError("num_bs_antennas 至少为 2")
    _positive_float("bandwidth_hz", radio["bandwidth_hz"])
    _positive_float("carrier_hz", radio["carrier_hz"])
    _positive_float("antenna_spacing_wavelength", radio["antenna_spacing_wavelength"])
    _finite_float("bs_boresight_deg", radio["bs_boresight_deg"])
    front_facing_only = radio.get("front_facing_only", True)
    if not isinstance(front_facing_only, bool):
        raise ValueError("front_facing_only 必须是布尔值")
    if not front_facing_only:
        raise ValueError("第一版均匀线阵要求 front_facing_only=true")
    _positive_integer("num_snapshots", radio["num_snapshots"])
    if "snr_db" in radio:
        _finite_float("snr_db", radio["snr_db"])
    if "bs_position_m" in radio:
        position = radio["bs_position_m"]
        try:
            if len(position) != 2:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError("radio.bs_position_m 必须是二维坐标") from error
        for value in position:
            _finite_float("radio.bs_position_m", value)
    max_diffractions = _nonnegative_integer("max_diffractions", scene.get("max_diffractions", 0))
    if max_diffractions not in (0, 1):
        raise ValueError("当前只支持最多一次绕射")
    _positive_integer("diffraction_directions_per_sample", localization.get("diffraction_directions_per_sample", 4))
    if localization.get("diffraction_representative_policy", "coverage") not in {"single", "coverage"}:
        raise ValueError("diffraction_representative_policy 只能为 single 或 coverage")
    _positive_float("diffraction_coverage_distance_m", localization.get("diffraction_coverage_distance_m", 1.0))
    angle_tolerance = _positive_float("diffraction_angle_tolerance_deg", localization.get("diffraction_angle_tolerance_deg", 3.0))
    if angle_tolerance >= 90:
        raise ValueError("绕射角度容差必须小于 90 度")
    max_reflections = _nonnegative_integer("max_reflections", scene["max_reflections"])
    if max_reflections not in (0, 1, 2):
        raise ValueError("第一版只支持 0、1 或 2 次镜面反射")
    _finite_float("fixed_height_m", scene["fixed_height_m"])
    _positive_float("bev_resolution_m", scene["bev_resolution_m"])
    _validated_bounds("bounds_m", scene["bounds_m"])
    if "localization_bounds_m" in scene:
        _validated_bounds("localization_bounds_m", scene["localization_bounds_m"])
    bias_min = _finite_float("bias_min_s", localization["bias_min_s"])
    bias_max = _finite_float("bias_max_s", localization["bias_max_s"])
    if bias_min >= bias_max:
        raise ValueError("bias_min_s 必须小于 bias_max_s")
    reference_value = localization.get("initial_reference_bias_s", 0.0)
    if isinstance(reference_value, bool):
        raise ValueError("initial_reference_bias_s 必须是有限数，不能是布尔值")
    try:
        reference_bias = _finite_float("initial_reference_bias_s", reference_value)
    except (TypeError, ValueError) as error:
        raise ValueError("initial_reference_bias_s 必须是有限数") from error
    if not bias_min <= reference_bias <= bias_max:
        raise ValueError("initial_reference_bias_s 必须位于 [bias_min_s, bias_max_s] 范围内")
    if isinstance(localization["candidate_cluster_radius_m"], bool):
        raise ValueError("candidate_cluster_radius_m 必须为有限正数，不能是布尔值")
    if not isinstance(localization["candidate_cluster_min_samples"], int):
        raise ValueError("candidate_cluster_min_samples 必须为正整数")
    for name in (
        "candidate_cluster_radius_m",
        "candidate_direction_radius_deg",
        "huber_delta_m",
    ):
        _positive_float(name, localization[name])
    _positive_integer("max_iterations", localization["max_iterations"])
    _positive_integer("candidate_cluster_min_samples", localization["candidate_cluster_min_samples"])
    _positive_integer("max_seed_pairs", localization.get("max_seed_pairs", 100000))

    angle_min = _finite_float("angle_min_deg", music["angle_min_deg"])
    angle_max = _finite_float("angle_max_deg", music["angle_max_deg"])
    if not (-90.0 <= angle_min < angle_max <= 90.0):
        raise ValueError("MUSIC 角度范围必须位于 [-90, 90] 且下界小于上界")
    delay_min = _finite_float("delay_min_s", music["delay_min_s"])
    delay_max = _finite_float("delay_max_s", music["delay_max_s"])
    if delay_min < 0.0 or delay_min >= delay_max:
        raise ValueError("MUSIC 时延范围必须非负且下界小于上界")
    num_paths = _positive_integer("num_paths", music["num_paths"])
    if num_paths < 2:
        raise ValueError("联合位置与偏差估计至少需要两条 MUSIC 路径")
    spatial_size = _positive_integer(
        "spatial_subarray_size", music["spatial_subarray_size"]
    )
    frequency_size = _positive_integer(
        "frequency_subarray_size", music["frequency_subarray_size"]
    )
    if not 2 <= spatial_size <= num_antennas:
        raise ValueError("spatial_subarray_size 必须位于 [2, num_bs_antennas]")
    if not 2 <= frequency_size <= num_subcarriers:
        raise ValueError("frequency_subarray_size 必须位于 [2, num_subcarriers]")
    signal_subspace_rank = _positive_integer(
        "signal_subspace_rank", music.get("signal_subspace_rank", num_paths)
    )
    subspace_dimension = spatial_size * frequency_size
    if signal_subspace_rank >= subspace_dimension:
        raise ValueError(
            "signal_subspace_rank 必须小于 spatial_subarray_size * "
            "frequency_subarray_size"
        )
    diagonal_loading = _finite_float("diagonal_loading", music["diagonal_loading"])
    if diagonal_loading < 0.0:
        raise ValueError("diagonal_loading 不能为负数")
    for name in (
        "angle_step_deg",
        "delay_step_s",
        "min_angle_separation_deg",
        "min_delay_separation_s",
    ):
        _positive_float(name, music[name])
    sampling = music.get("spectrum_sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("music.spectrum_sampling 必须是键值映射")
    expected = set(DEFAULT_CONFIG["music"]["spectrum_sampling"])
    if set(sampling) != expected:
        raise ValueError(
            "music.spectrum_sampling 字段不完整或包含未定义字段："
            f"缺少={sorted(expected.difference(sampling))}；"
            f"未定义={sorted(str(key) for key in set(sampling).difference(expected))}"
        )
    _positive_integer("samples_per_peak", sampling["samples_per_peak"])
    if _positive_integer("local_grid_points_per_axis", sampling["local_grid_points_per_axis"]) < 3:
        raise ValueError("local_grid_points_per_axis 至少为 3")
    for name in ("aoa_half_width_grid_steps", "delay_half_width_grid_steps", "spectrum_power"):
        _positive_float(name, sampling[name])
    mixture = _finite_float("uniform_mixture", sampling["uniform_mixture"])
    if not 0.0 <= mixture <= 1.0:
        raise ValueError("uniform_mixture 必须位于 [0, 1]")
    if not isinstance(sampling["include_nominal"], bool):
        raise ValueError("include_nominal 必须为布尔值")

    _nonnegative_integer("random_seed", config["project"]["random_seed"])


def _validate_simulation_section(config: dict[str, Any]) -> None:
    """只校验生成阶段使用的仿真真值段。"""

    simulation = config["simulation"]
    for name in ("bs_position_m", "ue_position_m"):
        position = simulation[name]
        if len(position) != 2:
            raise ValueError(f"{name} 必须是二维坐标")
        for value in position:
            _finite_float(name, value)
    _finite_float("clock_bias_s", simulation["clock_bias_s"])
    if "samples_per_source" in simulation:
        _positive_integer("samples_per_source", simulation["samples_per_source"])
    if "max_num_paths_per_source" in simulation:
        _positive_integer(
            "max_num_paths_per_source", simulation["max_num_paths_per_source"]
        )
