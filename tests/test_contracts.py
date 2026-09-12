from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from time_bias_localization.config import load_localization_config
from time_bias_localization.constants import SPEED_OF_LIGHT_M_S
from time_bias_localization.contracts import (
    validate_generation_manifest_envelope,
    validate_localization_input_contract,
)
from time_bias_localization.data import load_online_measurement
from time_bias_localization.provenance import generation_bundle_id
from time_bias_localization.scene import Scene2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN11_ROOT = PROJECT_ROOT / "outputs" / "deepmimo_sionna_smoke_run11"


def _artifact_hashes(stage: str) -> dict[str, dict[str, str]]:
    names = ("scene_json", "online_measurement", "ground_truth")
    return {
        name: {
            "path": f"/contract-test/{stage}/{name}",
            "sha256": sha256(f"{stage}:{name}".encode()).hexdigest(),
        }
        for name in names
    }


def _declare_bundle(manifest: dict[str, object]) -> dict[str, object]:
    manifest["bundle_id"] = generation_bundle_id(manifest)
    return manifest


def _synthetic_manifest() -> dict[str, object]:
    return _declare_bundle(
        {
            "schema_version": 2,
            "stage": "synthetic_csi_generation",
            "scene_json": "/contract-test/synthetic/scene_json",
            "online_input": "/contract-test/synthetic/online_measurement",
            "truth_input": "/contract-test/synthetic/ground_truth",
            "separation_rule": "localization 只允许读取 online 目录",
            "link_direction": "uplink_ue_to_bs",
            "absolute_delay_normalization": False,
            "delay_convention": (
                "observed_delay=geometric_delay+common_bias+noise"
            ),
            "artifact_hashes": _artifact_hashes("synthetic"),
            "rt_model": {
                "los": True,
                "specular_reflection": True,
                "max_reflections": 2,
                "diffraction": False,
                "diffuse_reflection": False,
                "transmission": False,
                "front_facing_only": True,
            },
            "path_selection": {
                "rule": (
                    "planar_height_and_order_then_front_facing_local_angle_window"
                ),
                "front_facing_only": True,
                "bs_boresight_rad": 0.0,
                "local_angle_min_rad": math.radians(-89.0),
                "local_angle_max_rad": math.radians(89.0),
                "planar_path_count_before_front_filter": 3,
                "front_facing_angle_path_count": 3,
                "retained_path_count_after_front_filter": 3,
                "requested_generated_path_count": 3,
            },
        }
    )


def _sionna_manifest() -> dict[str, object]:
    return _declare_bundle(
        {
            "schema_version": 2,
            "stage": "sionna_rt_to_deepmimo_v4",
            "sionna_scene": "munich",
            "deepmimo_scenario": "contract_munich",
            "deepmimo_scenario_store": "/contract-test/deepmimo",
            "source_export_dir": "/contract-test/source",
            "artifact_hashes": _artifact_hashes("sionna"),
            "link_direction": "uplink_ue_to_bs",
            "absolute_delay_normalization": False,
            "localization_scene_geometry_source": (
                "sionna_exported_triangle_mesh"
            ),
            "localization_scene_bounds_m": [-2.0, 2.0, -3.0, 3.0],
            "bounds_source": "fixed_config_not_ue_or_truth_paths",
            "scene_consistency": {
                "checked_path_count": 2,
                "los_path_count": 1,
                "reflected_path_count": 1,
                "position_tolerance_m": 0.1,
                "passed": True,
            },
            "rt_params": {
                "los": True,
                "specular_reflection": True,
                "max_depth": 2,
                "diffuse_reflection": False,
                "diffraction": False,
                "refraction": False,
                "samples_per_src": 100,
                "max_num_paths_per_src": 100,
                "synthetic_array": True,
                "seed": 1,
            },
            "deepmimo_export_scalars_normalized": [],
            "scene_artifacts": {
                "scene_json": "/contract-test/sionna/scene_json",
                "bev_png": "/contract-test/sionna/scene.png",
                "occupancy_npy": "/contract-test/sionna/scene.npy",
            },
            "data_artifacts": {
                "online_npz": "/contract-test/sionna/online_measurement",
                "truth_npz": "/contract-test/sionna/ground_truth",
            },
            "config_snapshot": {
                "path": "/contract-test/sionna/config.json",
                "source_path": "/contract-test/sionna/config.yaml",
                "canonical_sha256": "a" * 64,
                "file_sha256": "b" * 64,
            },
            "path_selection": {
                "rule": (
                    "planar_height_and_order_then_front_facing_local_angle_window"
                ),
                "front_facing_only": True,
                "bs_boresight_rad": 0.0,
                "local_angle_min_rad": math.radians(-89.0),
                "local_angle_max_rad": math.radians(89.0),
                "total_sionna_path_count": 3,
                "planar_path_count_before_front_filter": 3,
                "front_facing_angle_path_count": 2,
                "retained_path_count_after_front_filter": 2,
            },
            "planar_path_count_before_front_filter": 3,
            "retained_path_count": 2,
        }
    )


def _contract_case(
    stage: str = "synthetic_csi_generation",
) -> tuple[dict[str, object], dict[str, object], SimpleNamespace, SimpleNamespace]:
    is_sionna = stage == "sionna_rt_to_deepmimo_v4"
    bounds = [-2.0, 2.0, -3.0, 3.0] if is_sionna else [0.0, 8.0, 0.0, 6.0]
    config: dict[str, object] = {
        "scene": {
            "source": "sionna_builtin" if is_sionna else "synthetic_room",
            "name": "munich" if is_sionna else "offline_room",
            "bounds_m": bounds,
            "fixed_height_m": 1.5,
            "bev_resolution_m": 0.2,
            "max_reflections": 2,
        },
        "radio": {
            "carrier_hz": 3.5e9,
            "bandwidth_hz": 400e6,
            "num_subcarriers": 8,
            "num_bs_antennas": 2,
            "antenna_spacing_wavelength": 0.5,
            "bs_position_m": [1.0, 2.0],
            "bs_boresight_deg": 0.0,
            "front_facing_only": True,
            "num_snapshots": 1,
        },
        "music": {
            "angle_min_deg": -89.0,
            "angle_max_deg": 89.0,
        },
    }
    if is_sionna:
        config["scene"]["localization_bounds_m"] = bounds
    scene = SimpleNamespace(
        name="munich_bev" if is_sionna else "offline_room",
        source=(
            "sionna_exported_triangle_mesh" if is_sionna else "synthetic_room"
        ),
        bounds_m=tuple(bounds),
        fixed_height_m=1.5,
        bev_resolution_m=0.2,
    )
    carrier_hz = float(config["radio"]["carrier_hz"])
    measurement = SimpleNamespace(
        csi_observed=np.zeros((1, 2, 8), dtype=np.complex128),
        subcarrier_frequencies_hz=np.arange(8, dtype=float) * (400e6 / 8),
        carrier_frequency_hz=carrier_hz,
        antenna_spacing_m=SPEED_OF_LIGHT_M_S / carrier_hz * 0.5,
        bs_position_m=np.asarray([1.0, 2.0]),
        bs_boresight_rad=0.0,
    )
    manifest = _sionna_manifest() if is_sionna else _synthetic_manifest()
    return manifest, config, scene, measurement


@pytest.mark.parametrize(
    "manifest_factory,expected_stage",
    [
        (_synthetic_manifest, "synthetic_csi_generation"),
        (_sionna_manifest, "sionna_rt_to_deepmimo_v4"),
    ],
)
def test_manifest_envelope_accepts_supported_schema2_batches(
    manifest_factory, expected_stage: str
) -> None:
    manifest = manifest_factory()

    stage, bundle_id = validate_generation_manifest_envelope(manifest)

    assert stage == expected_stage
    assert bundle_id == generation_bundle_id(manifest)


@pytest.mark.parametrize("schema_version", [None, 1, 2.0, True, "2"])
def test_manifest_envelope_requires_integer_schema2(schema_version) -> None:
    manifest = _synthetic_manifest()
    manifest["schema_version"] = schema_version

    with pytest.raises(ValueError, match="schema_version.*2"):
        validate_generation_manifest_envelope(manifest)


def test_manifest_envelope_rejects_unknown_stage() -> None:
    manifest = _synthetic_manifest()
    manifest["stage"] = "unknown_generator"

    with pytest.raises(ValueError, match="stage.*不受支持"):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize("bundle_id", [None, "ABC", "F" * 64, "0" * 63])
def test_manifest_envelope_requires_legal_declared_bundle_id(bundle_id) -> None:
    manifest = _synthetic_manifest()
    manifest["bundle_id"] = bundle_id

    with pytest.raises(ValueError, match="bundle_id"):
        validate_generation_manifest_envelope(manifest)


def test_manifest_envelope_rejects_bundle_id_not_bound_to_core_hashes() -> None:
    manifest = _synthetic_manifest()
    manifest["artifact_hashes"]["online_measurement"]["sha256"] = "a" * 64

    with pytest.raises(ValueError, match="重算结果不一致"):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize("duplicate_field", ["path", "sha256"])
def test_manifest_envelope_requires_three_unique_core_artifacts(
    duplicate_field: str,
) -> None:
    manifest = _synthetic_manifest()
    artifacts = manifest["artifact_hashes"]
    artifacts["ground_truth"][duplicate_field] = artifacts["scene_json"][
        duplicate_field
    ]
    manifest["bundle_id"] = generation_bundle_id(manifest)

    with pytest.raises(ValueError, match="三个不同路径|相同内容摘要"):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize(
    "field,value,expected_message",
    [
        ("path", "", "path.*非空字符串"),
        ("sha256", "not-a-digest", "sha256.*64 位"),
        ("sha256", "A" * 64, "sha256.*小写"),
    ],
)
def test_manifest_envelope_rejects_invalid_core_records(
    field: str, value: object, expected_message: str
) -> None:
    manifest = _synthetic_manifest()
    manifest["artifact_hashes"]["scene_json"][field] = value

    with pytest.raises(ValueError, match=expected_message):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize(
    "stage,field_path,bad_value,expected_message",
    [
        ("sionna", ("link_direction",), "downlink_bs_to_ue", "uplink"),
        ("sionna", ("absolute_delay_normalization",), True, "normalization"),
        (
            "sionna",
            ("localization_scene_geometry_source",),
            "derived_wall_hypotheses",
            "原始导出三角网格",
        ),
        ("sionna", ("bounds_source",), "truth_paths", "固定公开配置"),
        ("sionna", ("rt_params", "los"), False, "los"),
        ("sionna", ("rt_params", "specular_reflection"), False, "specular"),
        ("sionna", ("rt_params", "max_depth"), 3, "max_depth"),
        ("sionna", ("rt_params", "diffuse_reflection"), True, "diffuse"),
        ("sionna", ("rt_params", "diffraction"), True, "diffraction"),
        ("sionna", ("rt_params", "refraction"), True, "refraction"),
        ("synthetic", ("rt_model", "los"), False, "los"),
        ("synthetic", ("rt_model", "specular_reflection"), False, "specular"),
        ("synthetic", ("rt_model", "max_reflections"), 3, "max_reflections"),
        ("synthetic", ("rt_model", "diffraction"), "true", "diffraction"),
        ("synthetic", ("rt_model", "diffuse_reflection"), True, "diffuse"),
        ("synthetic", ("rt_model", "transmission"), True, "transmission"),
    ],
)
def test_manifest_envelope_rejects_unsupported_physics(
    stage: str,
    field_path: tuple[str, ...],
    bad_value: object,
    expected_message: str,
) -> None:
    manifest = _sionna_manifest() if stage == "sionna" else _synthetic_manifest()
    target = manifest
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = bad_value

    with pytest.raises(ValueError, match=expected_message):
        validate_generation_manifest_envelope(manifest)


@pytest.mark.parametrize(
    "stage", ["synthetic_csi_generation", "sionna_rt_to_deepmimo_v4"]
)
def test_localization_input_contract_accepts_publicly_consistent_inputs(
    stage: str,
) -> None:
    manifest, config, scene, measurement = _contract_case(stage)

    validate_localization_input_contract(
        manifest, stage, config, scene, measurement
    )


@pytest.mark.parametrize("stage", ["synthetic_csi_generation", "sionna_rt_to_deepmimo_v4"])
def test_diffraction_contract_requires_matching_public_model(stage):
    manifest, config, scene, measurement = _contract_case(stage)
    key = "rt_params" if stage == "sionna_rt_to_deepmimo_v4" else "rt_model"
    manifest[key]["diffraction"] = True
    if key == "rt_params":
        manifest[key].update(max_depth=3, edge_diffraction=True, diffraction_lit_region=False)
    manifest["bundle_id"] = generation_bundle_id(manifest)
    validate_generation_manifest_envelope(manifest)
    with pytest.raises(ValueError, match="绕射"):
        validate_localization_input_contract(manifest, stage, config, scene, measurement)
    config["scene"]["max_diffractions"] = 1
    validate_localization_input_contract(manifest, stage, config, scene, measurement)


def test_synthetic_scene_source_is_compared_directly_to_public_config() -> None:
    manifest, config, scene, measurement = _contract_case()
    config["scene"]["source"] = "public_synthetic_geometry"
    scene.source = "public_synthetic_geometry"

    validate_localization_input_contract(
        manifest, "synthetic_csi_generation", config, scene, measurement
    )


@pytest.mark.parametrize(
    "field,bad_value,expected_message",
    [
        ("name", "munich", "名称不一致"),
        ("source", "sionna_builtin", "sionna_exported_triangle_mesh"),
        ("bounds_m", (-2.0, 2.1, -3.0, 3.0), "场景范围"),
        ("bounds_m", (2.0, -2.0, -3.0, 3.0), "x_min<x_max"),
        ("fixed_height_m", 1.6, "固定高度"),
        ("bev_resolution_m", 0.3, "分辨率"),
    ],
)
def test_sionna_scene_contract_rejects_scene_mismatch(
    field: str, bad_value: object, expected_message: str
) -> None:
    manifest, config, scene, measurement = _contract_case(
        "sionna_rt_to_deepmimo_v4"
    )
    setattr(scene, field, bad_value)

    with pytest.raises(ValueError, match=expected_message):
        validate_localization_input_contract(
            manifest, "sionna_rt_to_deepmimo_v4", config, scene, measurement
        )


def test_sionna_scene_contract_requires_manifest_and_config_name_mapping() -> None:
    manifest, config, scene, measurement = _contract_case(
        "sionna_rt_to_deepmimo_v4"
    )
    manifest["sionna_scene"] = "etoile"

    with pytest.raises(ValueError, match="场景名称.*公开配置"):
        validate_localization_input_contract(
            manifest, "sionna_rt_to_deepmimo_v4", config, scene, measurement
        )


def test_localization_input_contract_rejects_reflection_count_mismatch() -> None:
    manifest, config, scene, measurement = _contract_case()
    config["scene"]["max_reflections"] = 1

    with pytest.raises(ValueError, match="最大反射次数.*不一致"):
        validate_localization_input_contract(
            manifest, "synthetic_csi_generation", config, scene, measurement
        )


@pytest.mark.parametrize(
    "bad_csi,expected_message",
    [
        (np.zeros((2, 8), dtype=np.complex128), "严格为.*S,M,K"),
        (np.zeros((1, 3, 8), dtype=np.complex128), "严格为.*S,M,K"),
        (
            np.full((1, 2, 8), np.nan + 0.0j, dtype=np.complex128),
            "全部为有限数",
        ),
    ],
)
def test_localization_input_contract_rejects_bad_csi(
    bad_csi: np.ndarray, expected_message: str
) -> None:
    manifest, config, scene, measurement = _contract_case()
    measurement.csi_observed = bad_csi

    with pytest.raises(ValueError, match=expected_message):
        validate_localization_input_contract(
            manifest, "synthetic_csi_generation", config, scene, measurement
        )


@pytest.mark.parametrize(
    "frequencies,expected_message",
    [
        (np.arange(8, dtype=float)[None, :], "一维长度"),
        (np.asarray([0, 50, 100, 100, 200, 250, 300, 350]) * 1e6, "严格递增"),
        (np.asarray([0, 50, 100, 151, 200, 250, 300, 350]) * 1e6, "等间隔"),
        (np.arange(8, dtype=float) * 40e6, "bandwidth/num_subcarriers"),
        (1.0 + np.arange(8, dtype=float) * 50e6, "第一个子载波频率"),
        (
            np.asarray([0, 50, 100, np.nan, 200, 250, 300, 350]) * 1e6,
            "有限数组",
        ),
    ],
)
def test_localization_input_contract_rejects_bad_frequency_axis(
    frequencies: np.ndarray, expected_message: str
) -> None:
    manifest, config, scene, measurement = _contract_case()
    measurement.subcarrier_frequencies_hz = frequencies

    with pytest.raises(ValueError, match=expected_message):
        validate_localization_input_contract(
            manifest, "synthetic_csi_generation", config, scene, measurement
        )


@pytest.mark.parametrize(
    "field,bad_value,expected_message",
    [
        ("carrier_frequency_hz", np.inf, "载频.*有限标量|carrier_frequency"),
        ("carrier_frequency_hz", 3.6e9, "载频.*不一致"),
        ("antenna_spacing_m", np.nan, "antenna_spacing_m.*有限标量"),
        ("antenna_spacing_m", 0.2, "阵元间距.*不一致"),
        ("bs_position_m", [np.nan, 2.0], "BS 位置.*有限数组|bs_position"),
        ("bs_position_m", [1.1, 2.0], "BS 位置.*不一致"),
        ("bs_boresight_rad", np.nan, "bs_boresight_rad.*有限标量"),
        ("bs_boresight_rad", 0.1, "BS 朝向.*不一致"),
    ],
)
def test_localization_input_contract_rejects_bad_radio_metadata(
    field: str, bad_value: object, expected_message: str
) -> None:
    manifest, config, scene, measurement = _contract_case()
    setattr(measurement, field, bad_value)

    with pytest.raises(ValueError, match=expected_message):
        validate_localization_input_contract(
            manifest, "synthetic_csi_generation", config, scene, measurement
        )


@pytest.mark.parametrize(
    "target,field,bad_value,expected_message",
    [
        ("radio", "front_facing_only", False, "front_facing_only=true"),
        ("selection", "front_facing_only", False, "front_facing_only.*true"),
        ("selection", "bs_boresight_rad", 0.1, "路径筛选 BS 朝向"),
        ("selection", "local_angle_min_rad", -1.0, "最小局部角"),
        ("selection", "local_angle_max_rad", 1.0, "最大局部角"),
    ],
)
def test_localization_input_contract_rejects_path_window_mismatch(
    target: str, field: str, bad_value: object, expected_message: str
) -> None:
    manifest, config, scene, measurement = _contract_case()
    mapping = config["radio"] if target == "radio" else manifest["path_selection"]
    mapping[field] = bad_value

    with pytest.raises(ValueError, match=expected_message):
        validate_localization_input_contract(
            manifest, "synthetic_csi_generation", config, scene, measurement
        )


def test_localization_contract_does_not_use_truth_or_path_count_diagnostics() -> None:
    manifest, config, scene, measurement = _contract_case()
    manifest["path_selection"].update(
        {
            "planar_path_count_before_front_filter": -1,
            "front_facing_angle_path_count": -1,
            "retained_path_count_after_front_filter": -1,
            "requested_generated_path_count": -1,
        }
    )
    measurement.ue_position_m = [np.nan, np.nan]
    measurement.clock_bias_s = np.nan

    validate_localization_input_contract(
        manifest, "synthetic_csi_generation", config, scene, measurement
    )


@pytest.mark.parametrize(
    "forbidden_key",
    ["ue_position_m", "clock_bias_s", "snr_db", "csi_geometric"],
)
def test_manifest_envelope_rejects_truth_or_generation_only_payloads(
    forbidden_key: str,
) -> None:
    manifest = _synthetic_manifest()
    manifest["path_selection"][forbidden_key] = 1

    with pytest.raises(ValueError, match=rf"禁止包含字段 {forbidden_key}"):
        validate_generation_manifest_envelope(manifest)


def test_manifest_envelope_rejects_changed_path_rule_and_array_mode() -> None:
    manifest = _sionna_manifest()
    manifest["path_selection"]["rule"] = "truth_selected_paths"
    with pytest.raises(ValueError, match="path_selection.rule"):
        validate_generation_manifest_envelope(manifest)

    manifest = _sionna_manifest()
    manifest["rt_params"]["synthetic_array"] = False
    with pytest.raises(ValueError, match="synthetic_array.*true"):
        validate_generation_manifest_envelope(manifest)


def test_existing_run11_manifest_scene_and_online_file_satisfy_contract() -> None:
    manifest_path = RUN11_ROOT / "generation_manifest.json"
    localization_config_path = (
        PROJECT_ROOT / "configs" / "deepmimo_sionna_smoke_localization.yaml"
    )
    manifest_bytes_before = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes_before.decode("utf-8"))
    scene_path = Path(manifest["artifact_hashes"]["scene_json"]["path"])
    online_path = Path(manifest["artifact_hashes"]["online_measurement"]["path"])
    protected_bytes_before = {
        scene_path: scene_path.read_bytes(),
        online_path: online_path.read_bytes(),
    }

    stage, bundle_id = validate_generation_manifest_envelope(manifest)
    config = load_localization_config(localization_config_path)
    scene = Scene2D.load(scene_path)
    measurement = load_online_measurement(online_path)
    validate_localization_input_contract(manifest, stage, config, scene, measurement)

    assert stage == "sionna_rt_to_deepmimo_v4"
    # 外部 smoke 产物可由用户重新生成；校验当前清单和内容绑定，
    # 不把某次历史运行的摘要写死成场景语义要求。
    assert bundle_id == manifest["bundle_id"] == generation_bundle_id(manifest)
    assert scene.name == "munich_bev"
    assert scene.source == "sionna_exported_triangle_mesh"
    assert manifest_path.read_bytes() == manifest_bytes_before
    assert {path: path.read_bytes() for path in protected_bytes_before} == (
        protected_bytes_before
    )
