"""生成清单与定位公开输入之间的纯数据契约校验。"""

from __future__ import annotations

from collections.abc import Mapping
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S
from .provenance import generation_artifact_record, generation_bundle_id


_SUPPORTED_GENERATION_STAGES = frozenset(
    {"synthetic_csi_generation", "sionna_rt_to_deepmimo_v4"}
)
_CORE_ARTIFACT_NAMES = (
    "scene_json",
    "online_measurement",
    "ground_truth",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_PATH_SELECTION_RULE = (
    "planar_height_and_order_then_front_facing_local_angle_window"
)
_SYNTHETIC_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "scene_json",
        "online_input",
        "truth_input",
        "separation_rule",
        "link_direction",
        "absolute_delay_normalization",
        "delay_convention",
        "path_selection",
        "rt_model",
        "artifact_hashes",
        "bundle_id",
    }
)
_SIONNA_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "stage",
        "sionna_scene",
        "deepmimo_scenario",
        "deepmimo_scenario_store",
        "source_export_dir",
        "absolute_delay_normalization",
        "link_direction",
        "localization_scene_geometry_source",
        "localization_scene_bounds_m",
        "bounds_source",
        "scene_consistency",
        "rt_params",
        "deepmimo_export_scalars_normalized",
        "scene_artifacts",
        "data_artifacts",
        "config_snapshot",
        "artifact_hashes",
        "path_selection",
        "planar_path_count_before_front_filter",
        "retained_path_count",
        "bundle_id",
    }
)
_PROHIBITED_MANIFEST_KEYS = frozenset(
    {
        "ue_position_m",
        "clock_bias_s",
        "distance_bias_m",
        "snr_db",
        "csi_geometric",
        "clean_csi",
        "injected_noise_std",
        "path_coefficients",
        "path_delays_s",
        "path_aoa_global_deg",
    }
)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}必须是键值映射")
    return value


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    actual = set(mapping)
    if actual != set(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(f"{label}字段集合不符合契约；缺少={missing}；额外={extra}")


def _reject_prohibited_keys(value: Any, *, location: str = "生成清单") -> None:
    """递归拒绝把定位真值或注噪参数藏进允许的元数据结构。"""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in _PROHIBITED_MANIFEST_KEYS:
                raise ValueError(f"{location}禁止包含字段 {key_text}")
            _reject_prohibited_keys(nested, location=f"{location}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_prohibited_keys(nested, location=f"{location}[{index}]")


def _require_bool(
    mapping: Mapping[str, Any], key: str, expected: bool, label: str
) -> None:
    value = mapping.get(key)
    if value is not expected:
        expected_text = "true" if expected else "false"
        raise ValueError(f"{label}.{key} 必须为 {expected_text}")


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}必须是非空字符串")
    return value


def _assert_same_path(left: Any, right: Any, label: str) -> None:
    left_text = _require_nonempty_string(left, f"{label}左侧路径 ")
    right_text = _require_nonempty_string(right, f"{label}右侧路径 ")
    if Path(left_text).expanduser().resolve() != Path(right_text).expanduser().resolve():
        raise ValueError(f"{label}记录了两个不同路径")


def _require_reflection_depth(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2:
        raise ValueError(f"{label}必须是 0、1 或 2")
    return value


def _finite_scalar(value: Any, label: str) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label}必须是有限标量") from error
    if array.ndim != 0 or not np.isfinite(array.item()):
        raise ValueError(f"{label}必须是有限标量")
    return float(array.item())


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label}必须是长度为 {length} 的有限数组") from error
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label}必须是长度为 {length} 的有限数组")
    return array


def _finite_bounds(value: Any, label: str) -> np.ndarray:
    bounds = _finite_vector(value, 4, label)
    if not (bounds[0] < bounds[1] and bounds[2] < bounds[3]):
        raise ValueError(f"{label}必须满足 x_min<x_max 且 y_min<y_max")
    return bounds


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label}必须是正整数")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{label}必须是正整数")
    return integer


def _assert_close(actual: float, expected: float, label: str) -> None:
    absolute_tolerance = max(1e-12, abs(expected) * 1e-9)
    if not math.isclose(
        actual, expected, rel_tol=1e-9, abs_tol=absolute_tolerance
    ):
        raise ValueError(f"{label}与公开配置不一致：输入={actual}，配置={expected}")


def _assert_vector_close(
    actual: np.ndarray, expected: np.ndarray, label: str
) -> None:
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    if not np.allclose(actual, expected, rtol=1e-9, atol=scale * 1e-9):
        raise ValueError(f"{label}与公开配置不一致")


def _validate_stage_model(manifest: Mapping[str, Any], stage: str) -> None:
    if stage == "sionna_rt_to_deepmimo_v4":
        if manifest.get("link_direction") != "uplink_ue_to_bs":
            raise ValueError("Sionna 生成清单的 link_direction 必须为 uplink_ue_to_bs")
        if manifest.get("absolute_delay_normalization") is not False:
            raise ValueError("Sionna 生成清单必须声明 absolute_delay_normalization=false")
        if (
            manifest.get("localization_scene_geometry_source")
            != "sionna_exported_triangle_mesh"
        ):
            raise ValueError("Sionna 定位场景必须来自原始导出三角网格")
        if manifest.get("bounds_source") != "fixed_config_not_ue_or_truth_paths":
            raise ValueError("Sionna 定位范围必须来自固定公开配置，不能来自 UE 或路径真值")
        model = _require_mapping(manifest.get("rt_params"), "Sionna rt_params")
        _require_exact_keys(
            model,
            {
                "max_depth",
                "los",
                "specular_reflection",
                "diffuse_reflection",
                "diffraction",
                "refraction",
                "samples_per_src",
                "max_num_paths_per_src",
                "synthetic_array",
                "seed",
            },
            "Sionna rt_params ",
        )
        _require_bool(model, "los", True, "Sionna rt_params")
        _require_bool(model, "specular_reflection", True, "Sionna rt_params")
        _require_reflection_depth(model.get("max_depth"), "Sionna rt_params.max_depth")
        for key in ("diffuse_reflection", "diffraction", "refraction"):
            _require_bool(model, key, False, "Sionna rt_params")
        _require_bool(model, "synthetic_array", True, "Sionna rt_params")
        return

    model = _require_mapping(manifest.get("rt_model"), "离线 rt_model")
    _require_exact_keys(
        model,
        {
            "los",
            "specular_reflection",
            "max_reflections",
            "diffraction",
            "diffuse_reflection",
            "transmission",
            "front_facing_only",
        },
        "离线 rt_model ",
    )
    _require_bool(model, "los", True, "离线 rt_model")
    _require_bool(model, "specular_reflection", True, "离线 rt_model")
    _require_reflection_depth(
        model.get("max_reflections"), "离线 rt_model.max_reflections"
    )
    for key in ("diffraction", "diffuse_reflection", "transmission"):
        _require_bool(model, key, False, "离线 rt_model")
    _require_bool(model, "front_facing_only", True, "离线 rt_model")


def _validate_manifest_metadata_shape(
    manifest: Mapping[str, Any], stage: str
) -> None:
    """限制清单的嵌套结构；只核类型，不用路径数量决定是否定位。"""

    if stage == "synthetic_csi_generation":
        for key in ("scene_json", "online_input", "truth_input", "separation_rule"):
            _require_nonempty_string(manifest.get(key), f"离线生成清单 {key} ")
        if manifest.get("separation_rule") != "localization 只允许读取 online 目录":
            raise ValueError("离线生成清单的在线输入与真值隔离约定不一致")
        artifact_hashes = _require_mapping(
            manifest.get("artifact_hashes"), "离线 artifact_hashes"
        )
        _assert_same_path(
            manifest.get("scene_json"),
            _require_mapping(artifact_hashes.get("scene_json"), "场景记录").get(
                "path"
            ),
            "离线场景",
        )
        _assert_same_path(
            manifest.get("online_input"),
            _require_mapping(
                artifact_hashes.get("online_measurement"), "在线 CSI 记录"
            ).get("path"),
            "离线在线 CSI",
        )
        _assert_same_path(
            manifest.get("truth_input"),
            _require_mapping(artifact_hashes.get("ground_truth"), "真值记录").get(
                "path"
            ),
            "离线评估真值",
        )
        return

    for key in (
        "sionna_scene",
        "deepmimo_scenario",
        "deepmimo_scenario_store",
        "source_export_dir",
    ):
        _require_nonempty_string(manifest.get(key), f"Sionna 生成清单 {key} ")
    _finite_bounds(
        manifest.get("localization_scene_bounds_m"),
        "Sionna 生成清单 localization_scene_bounds_m",
    )
    for key in ("planar_path_count_before_front_filter", "retained_path_count"):
        value = manifest.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Sionna 生成清单 {key} 必须是整数")

    scene_consistency = _require_mapping(
        manifest.get("scene_consistency"), "Sionna scene_consistency"
    )
    _require_exact_keys(
        scene_consistency,
        {
            "checked_path_count",
            "los_path_count",
            "reflected_path_count",
            "position_tolerance_m",
            "passed",
        },
        "Sionna scene_consistency ",
    )
    for key in ("checked_path_count", "los_path_count", "reflected_path_count"):
        value = scene_consistency[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Sionna scene_consistency.{key} 必须是整数")
    _finite_scalar(
        scene_consistency["position_tolerance_m"],
        "Sionna scene_consistency.position_tolerance_m",
    )
    if not isinstance(scene_consistency["passed"], bool):
        raise ValueError("Sionna scene_consistency.passed 必须是布尔值")

    scene_artifacts = _require_mapping(
        manifest.get("scene_artifacts"), "Sionna scene_artifacts"
    )
    _require_exact_keys(
        scene_artifacts,
        {"scene_json", "bev_png", "occupancy_npy"},
        "Sionna scene_artifacts ",
    )
    data_artifacts = _require_mapping(
        manifest.get("data_artifacts"), "Sionna data_artifacts"
    )
    _require_exact_keys(
        data_artifacts,
        {"online_npz", "truth_npz"},
        "Sionna data_artifacts ",
    )
    for section_label, section in (
        ("scene_artifacts", scene_artifacts),
        ("data_artifacts", data_artifacts),
    ):
        for key, value in section.items():
            _require_nonempty_string(value, f"Sionna {section_label}.{key} ")
    artifact_hashes = _require_mapping(
        manifest.get("artifact_hashes"), "Sionna artifact_hashes"
    )
    _assert_same_path(
        scene_artifacts["scene_json"],
        _require_mapping(artifact_hashes.get("scene_json"), "场景记录").get(
            "path"
        ),
        "Sionna 场景",
    )
    _assert_same_path(
        data_artifacts["online_npz"],
        _require_mapping(
            artifact_hashes.get("online_measurement"), "在线 CSI 记录"
        ).get("path"),
        "Sionna 在线 CSI",
    )
    _assert_same_path(
        data_artifacts["truth_npz"],
        _require_mapping(artifact_hashes.get("ground_truth"), "真值记录").get(
            "path"
        ),
        "Sionna 评估真值",
    )

    config_snapshot = _require_mapping(
        manifest.get("config_snapshot"), "Sionna config_snapshot"
    )
    _require_exact_keys(
        config_snapshot,
        {"path", "source_path", "canonical_sha256", "file_sha256"},
        "Sionna config_snapshot ",
    )
    for key in ("path", "source_path"):
        _require_nonempty_string(
            config_snapshot.get(key), f"Sionna config_snapshot.{key} "
        )
    for key in ("canonical_sha256", "file_sha256"):
        digest = config_snapshot.get(key)
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"Sionna config_snapshot.{key} 必须是小写 SHA-256")

    normalized = manifest.get("deepmimo_export_scalars_normalized")
    if not isinstance(normalized, (list, tuple)) or not all(
        isinstance(item, str) and item for item in normalized
    ):
        raise ValueError("deepmimo_export_scalars_normalized 必须是字符串列表")


def validate_generation_manifest_envelope(
    manifest: Mapping[str, Any],
) -> tuple[str, str]:
    """校验第 2 版生成清单的身份、核心产物绑定和公开物理模型。"""

    manifest = _require_mapping(manifest, "生成清单")
    _reject_prohibited_keys(manifest)
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("生成清单 schema_version 必须严格等于 2")

    stage = manifest.get("stage")
    if stage not in _SUPPORTED_GENERATION_STAGES:
        supported = "、".join(sorted(_SUPPORTED_GENERATION_STAGES))
        raise ValueError(f"生成清单 stage 不受支持；只允许：{supported}")
    assert isinstance(stage, str)
    _require_exact_keys(
        manifest,
        (
            _SIONNA_TOP_LEVEL_FIELDS
            if stage == "sionna_rt_to_deepmimo_v4"
            else _SYNTHETIC_TOP_LEVEL_FIELDS
        ),
        "生成清单顶层 ",
    )

    artifact_hashes = _require_mapping(
        manifest.get("artifact_hashes"), "生成清单 artifact_hashes"
    )
    _require_exact_keys(
        artifact_hashes, set(_CORE_ARTIFACT_NAMES), "生成清单 artifact_hashes "
    )
    records: list[dict[str, str]] = []
    for artifact_name in _CORE_ARTIFACT_NAMES:
        raw_record = _require_mapping(
            artifact_hashes.get(artifact_name),
            f"生成清单核心产物 {artifact_name}",
        )
        _require_exact_keys(
            raw_record, {"path", "sha256"}, f"生成清单核心产物 {artifact_name} "
        )
        raw_path = raw_record.get("path")
        raw_digest = raw_record.get("sha256")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"生成清单核心产物 {artifact_name}.path 必须是非空字符串")
        if not isinstance(raw_digest, str) or not _SHA256_PATTERN.fullmatch(
            raw_digest
        ):
            raise ValueError(
                f"生成清单核心产物 {artifact_name}.sha256 必须是 64 位小写十六进制"
            )
        try:
            records.append(generation_artifact_record(manifest, artifact_name))
        except (TypeError, ValueError) as error:
            raise ValueError(f"生成清单核心产物 {artifact_name} 记录无效：{error}") from error
    resolved_paths = [str(Path(record["path"]).resolve()) for record in records]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("生成清单的场景、在线 CSI 和真值必须使用三个不同路径")
    digests = [record["sha256"] for record in records]
    if len(set(digests)) != len(digests):
        raise ValueError("生成清单的场景、在线 CSI 和真值不能声明相同内容摘要")

    declared_bundle_id = manifest.get("bundle_id")
    if not isinstance(declared_bundle_id, str) or not _SHA256_PATTERN.fullmatch(
        declared_bundle_id
    ):
        raise ValueError("生成清单必须声明 64 位小写十六进制 bundle_id")
    computed_bundle_id = generation_bundle_id(manifest)
    if declared_bundle_id != computed_bundle_id:
        raise ValueError("生成清单 bundle_id 与三个核心产物摘要的重算结果不一致")

    _validate_stage_model(manifest, stage)
    _validate_manifest_metadata_shape(manifest, stage)
    path_selection = _require_mapping(
        manifest.get("path_selection"), "生成清单 path_selection"
    )
    required_selection_fields = {
        "rule",
        "front_facing_only",
        "bs_boresight_rad",
        "local_angle_min_rad",
        "local_angle_max_rad",
    }
    stage_count_fields = (
        {
            "total_sionna_path_count",
            "planar_path_count_before_front_filter",
            "front_facing_angle_path_count",
            "retained_path_count_after_front_filter",
        }
        if stage == "sionna_rt_to_deepmimo_v4"
        else {
            "planar_path_count_before_front_filter",
            "front_facing_angle_path_count",
            "retained_path_count_after_front_filter",
            "requested_generated_path_count",
        }
    )
    _require_exact_keys(
        path_selection,
        required_selection_fields | stage_count_fields,
        "生成清单 path_selection ",
    )
    for count_field in stage_count_fields:
        count_value = path_selection[count_field]
        if isinstance(count_value, bool) or not isinstance(count_value, int):
            raise ValueError(f"生成清单 path_selection.{count_field} 必须是整数")
    if path_selection.get("rule") != _PATH_SELECTION_RULE:
        raise ValueError("生成清单 path_selection.rule 与固定二维筛选规则不一致")

    if stage == "synthetic_csi_generation":
        if manifest.get("link_direction") != "uplink_ue_to_bs":
            raise ValueError("离线生成清单的 link_direction 必须为 uplink_ue_to_bs")
        if manifest.get("absolute_delay_normalization") is not False:
            raise ValueError("离线生成清单必须声明 absolute_delay_normalization=false")
        if (
            manifest.get("delay_convention")
            != "observed_delay=geometric_delay+common_bias+noise"
        ):
            raise ValueError("离线生成清单的公共时延偏差符号约定不一致")
    return stage, declared_bundle_id


def validate_localization_input_contract(
    manifest: Mapping[str, Any],
    stage: str,
    config: Mapping[str, Any],
    scene: Any,
    measurement: Any,
) -> None:
    """仅用公开先验检查场景、在线 CSI 与生成清单是否物理一致。"""

    manifest = _require_mapping(manifest, "生成清单")
    config = _require_mapping(config, "定位配置")
    if stage not in _SUPPORTED_GENERATION_STAGES or manifest.get("stage") != stage:
        raise ValueError("输入契约的 stage 与生成清单不一致或不受支持")
    scene_config = _require_mapping(config.get("scene"), "定位配置 scene")
    radio_config = _require_mapping(config.get("radio"), "定位配置 radio")
    music_config = _require_mapping(config.get("music"), "定位配置 music")

    configured_scene_name = scene_config.get("name")
    if not isinstance(configured_scene_name, str) or not configured_scene_name:
        raise ValueError("定位配置 scene.name 必须是非空字符串")
    if stage == "sionna_rt_to_deepmimo_v4":
        expected_scene_name = f"{configured_scene_name}_bev"
        expected_config_source = "sionna_builtin"
        expected_scene_source = "sionna_exported_triangle_mesh"
        if manifest.get("sionna_scene") != configured_scene_name:
            raise ValueError("Sionna 生成清单的场景名称与公开配置不一致")
    else:
        expected_scene_name = configured_scene_name
        expected_config_source = scene_config.get("source")
        if not isinstance(expected_config_source, str) or not expected_config_source:
            raise ValueError("定位配置 scene.source 必须是非空字符串")
        expected_scene_source = expected_config_source
    if scene_config.get("source") != expected_config_source:
        raise ValueError(f"当前生成阶段要求 config.scene.source={expected_config_source}")
    if getattr(scene, "name", None) != expected_scene_name:
        raise ValueError(
            f"二维场景名称不一致：输入={getattr(scene, 'name', None)}，"
            f"应为={expected_scene_name}"
        )
    if getattr(scene, "source", None) != expected_scene_source:
        raise ValueError(f"二维场景来源必须为 {expected_scene_source}")

    scene_bounds = _finite_bounds(getattr(scene, "bounds_m", None), "二维场景 bounds_m")
    configured_bounds = _finite_bounds(
        scene_config.get("bounds_m"), "定位配置 scene.bounds_m"
    )
    _assert_vector_close(scene_bounds, configured_bounds, "二维场景范围")
    if stage == "sionna_rt_to_deepmimo_v4":
        localization_bounds = _finite_bounds(
            scene_config.get("localization_bounds_m"),
            "定位配置 scene.localization_bounds_m",
        )
        manifest_bounds = _finite_bounds(
            manifest.get("localization_scene_bounds_m"),
            "Sionna 生成清单 localization_scene_bounds_m",
        )
        _assert_vector_close(scene_bounds, localization_bounds, "二维场景固定定位范围")
        _assert_vector_close(scene_bounds, manifest_bounds, "生成清单固定定位范围")

    scene_height = _finite_scalar(
        getattr(scene, "fixed_height_m", None), "二维场景 fixed_height_m"
    )
    configured_height = _finite_scalar(
        scene_config.get("fixed_height_m"), "定位配置 scene.fixed_height_m"
    )
    _assert_close(scene_height, configured_height, "二维场景固定高度")
    scene_resolution = _finite_scalar(
        getattr(scene, "bev_resolution_m", None), "二维场景 bev_resolution_m"
    )
    configured_resolution = _finite_scalar(
        scene_config.get("bev_resolution_m"), "定位配置 scene.bev_resolution_m"
    )
    if scene_resolution <= 0.0 or configured_resolution <= 0.0:
        raise ValueError("二维场景俯视图分辨率必须为正数")
    _assert_close(scene_resolution, configured_resolution, "二维场景俯视图分辨率")

    configured_reflections = _require_reflection_depth(
        scene_config.get("max_reflections"), "定位配置 scene.max_reflections"
    )
    model_key = "rt_params" if stage == "sionna_rt_to_deepmimo_v4" else "rt_model"
    model = _require_mapping(manifest.get(model_key), f"生成清单 {model_key}")
    reflection_key = (
        "max_depth" if stage == "sionna_rt_to_deepmimo_v4" else "max_reflections"
    )
    manifest_reflections = _require_reflection_depth(
        model.get(reflection_key), f"生成清单 {model_key}.{reflection_key}"
    )
    if manifest_reflections != configured_reflections:
        raise ValueError("生成清单的最大反射次数与公开定位配置不一致")

    snapshot_count = _positive_integer(
        radio_config.get("num_snapshots"), "定位配置 radio.num_snapshots"
    )
    antenna_count = _positive_integer(
        radio_config.get("num_bs_antennas"), "定位配置 radio.num_bs_antennas"
    )
    subcarrier_count = _positive_integer(
        radio_config.get("num_subcarriers"), "定位配置 radio.num_subcarriers"
    )
    if subcarrier_count < 2:
        raise ValueError("定位配置 radio.num_subcarriers 至少为 2")
    try:
        csi = np.asarray(measurement.csi_observed)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("在线 CSI 无法转换为数组") from error
    expected_shape = (snapshot_count, antenna_count, subcarrier_count)
    if csi.ndim != 3 or csi.shape != expected_shape:
        raise ValueError(
            f"在线 CSI 必须严格为 (S,M,K)={expected_shape}，当前为 {csi.shape}"
        )
    try:
        csi_is_finite = bool(np.all(np.isfinite(csi)))
    except TypeError as error:
        raise ValueError("在线 CSI 必须全部为有限数") from error
    if not csi_is_finite:
        raise ValueError("在线 CSI 必须全部为有限数")

    try:
        frequencies = np.asarray(
            measurement.subcarrier_frequencies_hz, dtype=float
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("子载波频率必须是一维有限数组") from error
    if frequencies.shape != (subcarrier_count,) or not np.all(np.isfinite(frequencies)):
        raise ValueError(
            f"子载波频率必须是一维长度 {subcarrier_count} 的有限数组"
        )
    differences = np.diff(frequencies)
    if np.any(differences <= 0.0):
        raise ValueError("子载波频率必须严格递增")
    bandwidth_hz = _finite_scalar(
        radio_config.get("bandwidth_hz"), "定位配置 radio.bandwidth_hz"
    )
    if bandwidth_hz <= 0.0:
        raise ValueError("定位配置 radio.bandwidth_hz 必须为正数")
    expected_spacing_hz = bandwidth_hz / subcarrier_count
    spacing_tolerance_hz = max(1e-9, expected_spacing_hz * 1e-9)
    if not np.allclose(
        differences,
        expected_spacing_hz,
        rtol=1e-9,
        atol=spacing_tolerance_hz,
    ):
        raise ValueError("子载波频率必须等间隔，且间隔等于 bandwidth/num_subcarriers")
    if not math.isclose(
        float(frequencies[0]), 0.0, rel_tol=0.0, abs_tol=spacing_tolerance_hz
    ):
        raise ValueError("第一个子载波频率必须为 0 Hz")

    carrier_hz = _finite_scalar(
        getattr(measurement, "carrier_frequency_hz", None),
        "在线测量 carrier_frequency_hz",
    )
    configured_carrier_hz = _finite_scalar(
        radio_config.get("carrier_hz"), "定位配置 radio.carrier_hz"
    )
    if carrier_hz <= 0.0 or configured_carrier_hz <= 0.0:
        raise ValueError("载频必须为正数")
    _assert_close(carrier_hz, configured_carrier_hz, "在线测量载频")

    antenna_spacing_m = _finite_scalar(
        getattr(measurement, "antenna_spacing_m", None),
        "在线测量 antenna_spacing_m",
    )
    spacing_wavelength = _finite_scalar(
        radio_config.get("antenna_spacing_wavelength"),
        "定位配置 radio.antenna_spacing_wavelength",
    )
    if antenna_spacing_m <= 0.0 or spacing_wavelength <= 0.0:
        raise ValueError("阵元间距必须为正数")
    expected_antenna_spacing_m = (
        SPEED_OF_LIGHT_M_S / configured_carrier_hz * spacing_wavelength
    )
    _assert_close(
        antenna_spacing_m, expected_antenna_spacing_m, "在线测量阵元间距"
    )

    bs_position = _finite_vector(
        getattr(measurement, "bs_position_m", None), 2, "在线测量 bs_position_m"
    )
    configured_bs_position = _finite_vector(
        radio_config.get("bs_position_m"), 2, "定位配置 radio.bs_position_m"
    )
    _assert_vector_close(bs_position, configured_bs_position, "在线测量 BS 位置")
    boresight_rad = _finite_scalar(
        getattr(measurement, "bs_boresight_rad", None),
        "在线测量 bs_boresight_rad",
    )
    configured_boresight_rad = math.radians(
        _finite_scalar(
            radio_config.get("bs_boresight_deg"),
            "定位配置 radio.bs_boresight_deg",
        )
    )
    _assert_close(boresight_rad, configured_boresight_rad, "在线测量 BS 朝向")

    if radio_config.get("front_facing_only") is not True:
        raise ValueError("定位公开配置必须声明 radio.front_facing_only=true")
    path_selection = _require_mapping(
        manifest.get("path_selection"), "生成清单 path_selection"
    )
    if path_selection.get("front_facing_only") is not True:
        raise ValueError("生成清单 path_selection.front_facing_only 必须为 true")
    selection_boresight = _finite_scalar(
        path_selection.get("bs_boresight_rad"),
        "生成清单 path_selection.bs_boresight_rad",
    )
    _assert_close(selection_boresight, configured_boresight_rad, "路径筛选 BS 朝向")
    expected_angle_min = math.radians(
        _finite_scalar(
            music_config.get("angle_min_deg"), "定位配置 music.angle_min_deg"
        )
    )
    expected_angle_max = math.radians(
        _finite_scalar(
            music_config.get("angle_max_deg"), "定位配置 music.angle_max_deg"
        )
    )
    selection_angle_min = _finite_scalar(
        path_selection.get("local_angle_min_rad"),
        "生成清单 path_selection.local_angle_min_rad",
    )
    selection_angle_max = _finite_scalar(
        path_selection.get("local_angle_max_rad"),
        "生成清单 path_selection.local_angle_max_rad",
    )
    if not selection_angle_min < selection_angle_max:
        raise ValueError("生成清单的路径筛选局部角窗必须满足最小角小于最大角")
    _assert_close(selection_angle_min, expected_angle_min, "路径筛选最小局部角")
    _assert_close(selection_angle_max, expected_angle_max, "路径筛选最大局部角")


__all__ = [
    "validate_generation_manifest_envelope",
    "validate_localization_input_contract",
]
