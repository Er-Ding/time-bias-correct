from __future__ import annotations

from copy import deepcopy
import inspect
import json
import multiprocessing
from pathlib import Path
import pickle
from queue import Empty
from types import SimpleNamespace

import numpy as np
import pytest

import time_bias_localization.sionna_generation as sionna_generation_module
from time_bias_localization.config import DEFAULT_CONFIG
from time_bias_localization.data import OnlineMeasurement
from time_bias_localization.provenance import (
    artifact_record,
    exclusive_output_root_lock,
    generation_bundle_id,
    load_generation_manifest,
)
from time_bias_localization.scene import Scene2D, WallSegment
from time_bias_localization.sionna_generation import (
    _finalize_generation_manifest,
    _localization_scene_bounds,
    _path_selection_summary,
    _normalize_deepmimo_export_scalars,
    _save_real_bundle,
    _validate_reverse_scene_consistency,
    extract_planar_uplink_csi,
    generate_sionna_deepmimo_bundle,
)


def _sionna_test_config() -> dict[str, object]:
    config = deepcopy(DEFAULT_CONFIG)
    config["scene"].update(
        {
            "source": "sionna_builtin",
            "name": "munich",
            "localization_bounds_m": [-2.0, 2.0, -2.0, 2.0],
            "vertical_path_tolerance_m": 0.1,
        }
    )
    config["radio"].update({"num_bs_antennas": 2, "num_subcarriers": 8})
    config["simulation"].update(
        {
            "allow_overwrite": False,
            "samples_per_source": 8,
            "max_num_paths_per_source": 16,
            "deepmimo_scenario_name": "fake_scenario",
            "deepmimo_lossless_scene": True,
        }
    )
    return config


def test_local_deepmimo_missing_output_never_calls_download_loader(tmp_path):
    def forbidden_load(name):
        raise AssertionError("本地文件缺失时不允许调用 DeepMIMO load")
    deepmimo = SimpleNamespace(load=forbidden_load)
    with pytest.raises(FileNotFoundError, match="不尝试在线下载"):
        sionna_generation_module._load_local_deepmimo_scene(deepmimo, "Mixed_UE001", tmp_path)
    (tmp_path / "mixed_ue001").mkdir()
    with pytest.raises(FileNotFoundError, match="params.json"):
        sionna_generation_module._load_local_deepmimo_scene(deepmimo, "Mixed_UE001", tmp_path)


def test_local_deepmimo_loader_uses_canonical_name(tmp_path):
    directory = tmp_path / "mixed_ue001"
    directory.mkdir()
    (directory / "params.json").write_text("{}")
    received = []
    def load(name):
        received.append(name)
        return SimpleNamespace(scene=object())
    sionna_generation_module._load_local_deepmimo_scene(SimpleNamespace(load=load), "Mixed_UE001", tmp_path)
    assert received == ["mixed_ue001"]


def _generate_in_child(config, output_root: str, messages) -> None:
    messages.put("started")
    try:
        result = generate_sionna_deepmimo_bundle(config, output_root=output_root)
    except BaseException as error:
        messages.put(("error", type(error).__name__, str(error)))
    else:
        messages.put(("finished", result))


def _minimal_generation_manifest(tmp_path) -> dict[str, object]:
    artifact_paths = {
        "scene_json": tmp_path / "scene_2d.json",
        "online_measurement": tmp_path / "measurement.npz",
        "ground_truth": tmp_path / "ground_truth.npz",
    }
    for artifact_name, artifact_path in artifact_paths.items():
        artifact_path.write_bytes(f"fixed-{artifact_name}".encode("utf-8"))
    return {
        "stage": "sionna_rt_to_deepmimo_v4",
        "artifact_hashes": {
            artifact_name: artifact_record(artifact_path)
            for artifact_name, artifact_path in artifact_paths.items()
        },
    }


def test_legacy_sionna_manifest_without_declared_bundle_id_remains_loadable(
    tmp_path,
) -> None:
    legacy_manifest = _minimal_generation_manifest(tmp_path)
    manifest_path = tmp_path / "legacy_generation_manifest.json"
    manifest_path.write_text(
        json.dumps(legacy_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    loaded, _, computed_bundle_id = load_generation_manifest(manifest_path)

    assert "schema_version" not in loaded
    assert "bundle_id" not in loaded
    assert computed_bundle_id == generation_bundle_id(legacy_manifest)


def test_new_sionna_manifest_declares_matching_bundle_id(tmp_path) -> None:
    manifest = _finalize_generation_manifest(_minimal_generation_manifest(tmp_path))
    manifest_path = tmp_path / "generation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    loaded, _, computed_bundle_id = load_generation_manifest(manifest_path)

    assert loaded["schema_version"] == 2
    assert loaded["bundle_id"] == computed_bundle_id
    assert computed_bundle_id == generation_bundle_id(loaded)


def test_sionna_generation_rejects_allow_overwrite_before_creating_root(
    tmp_path, monkeypatch
) -> None:
    config = _sionna_test_config()
    config["simulation"]["allow_overwrite"] = True
    output_root = tmp_path / "must_not_be_created"

    def must_not_run(*args, **kwargs):
        raise AssertionError("早拒绝后不应进入 Sionna 生成")

    monkeypatch.setattr(
        sionna_generation_module,
        "_generate_sionna_deepmimo_bundle_locked",
        must_not_run,
    )

    with pytest.raises(ValueError, match="allow_overwrite 必须为 false"):
        generate_sionna_deepmimo_bundle(config, output_root=output_root)

    assert not output_root.exists()


def test_sionna_generation_rejects_all_existing_targets_without_changing_them(
    tmp_path, monkeypatch
) -> None:
    output_root = tmp_path / "old_batch"
    existing_files = {
        output_root / "generation_manifest.json": b"old-generation-manifest",
        output_root / "provenance" / "generation_config.json": b"old-config",
        output_root / "deepmimo_source" / "old.bin": b"old-source",
        output_root / "deepmimo_scenarios" / "old.bin": b"old-scenario",
        output_root / "scene" / "old.bin": b"old-scene",
        output_root / "data" / "old.bin": b"old-data",
    }
    for path, content in existing_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def must_not_run(*args, **kwargs):
        raise AssertionError("发现旧目标后不应导入或求解 Sionna")

    monkeypatch.setattr(
        sionna_generation_module,
        "_generate_sionna_deepmimo_bundle_locked",
        must_not_run,
    )

    with pytest.raises(FileExistsError) as error_info:
        generate_sionna_deepmimo_bundle(
            _sionna_test_config(), output_root=output_root
        )

    message = str(error_info.value)
    for target_name in (
        "generation_manifest.json",
        "generation_config.json",
        "deepmimo_source",
        "deepmimo_scenarios",
        "scene",
        "data",
    ):
        assert target_name in message
    assert {path: path.read_bytes() for path in existing_files} == existing_files


def test_sionna_generation_waits_for_shared_output_root_lock(
    tmp_path, monkeypatch
) -> None:
    output_root = tmp_path / "locked_generation"
    context = multiprocessing.get_context("fork")
    messages = context.Queue()

    def fake_locked_generation(config, *, root):
        messages.put("entered-generation")
        return {"manifest": str(root / "generation_manifest.json")}

    monkeypatch.setattr(
        sionna_generation_module,
        "_generate_sionna_deepmimo_bundle_locked",
        fake_locked_generation,
    )
    process = context.Process(
        target=_generate_in_child,
        args=(_sionna_test_config(), str(output_root), messages),
    )

    with exclusive_output_root_lock(output_root):
        process.start()
        assert messages.get(timeout=2.0) == "started"
        with pytest.raises(Empty):
            messages.get(timeout=0.2)

    assert messages.get(timeout=2.0) == "entered-generation"
    status, result = messages.get(timeout=2.0)
    assert status == "finished"
    assert result["manifest"] == str(output_root / "generation_manifest.json")
    process.join(timeout=2.0)
    assert process.exitcode == 0


def test_sionna_failure_keeps_residue_but_does_not_publish_manifest(
    tmp_path, monkeypatch
) -> None:
    output_root = tmp_path / "failed_generation"

    def fail_before_sionna_import():
        raise RuntimeError("fake-sionna-import-failure")

    monkeypatch.setattr(
        sionna_generation_module,
        "load_sionna_rt_module",
        fail_before_sionna_import,
    )

    with pytest.raises(RuntimeError, match="fake-sionna-import-failure"):
        generate_sionna_deepmimo_bundle(
            _sionna_test_config(), output_root=output_root
        )

    config_snapshot_path = output_root / "provenance" / "generation_config.json"
    snapshot_before_retry = config_snapshot_path.read_bytes()
    assert not (output_root / "generation_manifest.json").exists()

    with pytest.raises(FileExistsError, match="generation_config.json"):
        generate_sionna_deepmimo_bundle(
            _sionna_test_config(), output_root=output_root
        )
    assert config_snapshot_path.read_bytes() == snapshot_before_retry
    assert not (output_root / "generation_manifest.json").exists()


@pytest.mark.parametrize("scenario_name", ["fake_scenario", "Experiment_20260907T092242_UE001"])
def test_sionna_generation_publishes_manifest_after_all_core_artifacts(
    tmp_path, monkeypatch, scenario_name
) -> None:
    config = _sionna_test_config()
    config["simulation"]["deepmimo_scenario_name"] = scenario_name
    output_root = tmp_path / "complete_generation"
    events: list[str] = []

    class FakeScene:
        synthetic_array = False

        def add(self, item) -> None:
            return None

    fake_paths = SimpleNamespace(tau=np.ones((1,), dtype=float))
    fake_sionna_rt = SimpleNamespace(
        scene=SimpleNamespace(munich="fake-scene-asset"),
        load_scene=lambda asset: FakeScene(),
        PlanarArray=lambda **kwargs: SimpleNamespace(),
        Transmitter=lambda name, position: SimpleNamespace(),
        Receiver=lambda name, position: SimpleNamespace(),
        PathSolver=lambda: (lambda **kwargs: fake_paths),
    )

    class FakeDeepMIMOConfig:
        def set(self, name, value) -> None:
            return None

    def fake_convert(source_dir, *, scenario_name, overwrite, **kwargs):
        assert overwrite is False
        assert scenario_name == scenario_name.lower()
        directory = Path.cwd() / "deepmimo_scenarios" / scenario_name
        directory.mkdir(parents=True)
        (directory / "params.json").write_text("{}")
        events.append("deepmimo")
        return scenario_name

    fake_deepmimo = SimpleNamespace(
        config=FakeDeepMIMOConfig(),
        convert=fake_convert,
        load=lambda scenario_name: SimpleNamespace(scene=object()),
    )

    def fake_exporter(scene, paths, rt_params, export_dir) -> None:
        directory = Path(export_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "sionna_paths.pkl").write_bytes(b"fake-paths")
        events.append("sionna-export")

    class FakeLocalizationScene:
        def save(self, output_dir) -> dict[str, str]:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            paths = {
                "scene_json": directory / "scene_2d.json",
                "bev_png": directory / "scene_bev.png",
                "occupancy_npy": directory / "scene_occupancy.npy",
            }
            paths["scene_json"].write_text("{}", encoding="utf-8")
            paths["bev_png"].write_bytes(b"fake-bev")
            paths["occupancy_npy"].write_bytes(b"fake-occupancy")
            events.append("scene")
            return {name: str(path) for name, path in paths.items()}

    def fake_save_bundle(root, online, **kwargs) -> dict[str, str]:
        online_path = Path(root) / "data" / "online" / "measurement.npz"
        truth_path = Path(root) / "data" / "truth" / "ground_truth.npz"
        online_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        online_path.write_bytes(b"fake-online")
        truth_path.write_bytes(b"fake-truth")
        events.append("data")
        return {"online_npz": str(online_path), "truth_npz": str(truth_path)}

    path_selection = {
        "rule": "fake-selection",
        "front_facing_only": True,
        "bs_boresight_rad": 0.0,
        "local_angle_min_rad": -1.0,
        "local_angle_max_rad": 1.0,
        "total_sionna_path_count": 1,
        "planar_path_count_before_front_filter": 1,
        "front_facing_angle_path_count": 1,
        "retained_path_count_after_front_filter": 1,
    }
    real_atomic_write = sionna_generation_module._write_json_atomic

    def record_atomic_write(path, data):
        events.append(Path(path).name)
        return real_atomic_write(path, data)

    monkeypatch.setattr(
        sionna_generation_module, "load_sionna_rt_module", lambda: fake_sionna_rt
    )
    monkeypatch.setattr(
        sionna_generation_module, "load_deepmimo_module", lambda: fake_deepmimo
    )
    monkeypatch.setattr(
        sionna_generation_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(sionna_exporter=fake_exporter),
    )
    monkeypatch.setattr(
        sionna_generation_module,
        "_normalize_deepmimo_export_scalars",
        lambda export_dir: (),
    )
    monkeypatch.setattr(
        sionna_generation_module,
        "extract_planar_uplink_csi",
        lambda *args, **kwargs: (np.ones((2, 8), dtype=np.complex128), {}),
    )
    monkeypatch.setattr(
        sionna_generation_module,
        "_path_selection_summary",
        lambda metadata: path_selection,
    )
    monkeypatch.setattr(
        sionna_generation_module,
        "preprocess_sionna_exported_scene",
        lambda *args, **kwargs: FakeLocalizationScene(),
    )
    monkeypatch.setattr(
        sionna_generation_module,
        "_validate_reverse_scene_consistency",
        lambda *args, **kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        sionna_generation_module, "_save_real_bundle", fake_save_bundle
    )
    monkeypatch.setattr(
        sionna_generation_module, "_write_json_atomic", record_atomic_write
    )

    result = generate_sionna_deepmimo_bundle(config, output_root=output_root)

    manifest_path = output_root / "generation_manifest.json"
    loaded, _, computed_bundle_id = load_generation_manifest(manifest_path)
    assert Path(result["manifest"]) == manifest_path
    assert loaded["bundle_id"] == computed_bundle_id
    assert events[-1] == "generation_manifest.json"
    assert events.index("generation_config.json") < events.index("sionna-export")
    assert events.index("sionna-export") < events.index("scene")
    assert events.index("scene") < events.index("data")
    assert events.index("data") < events.index("generation_manifest.json")


class FakePaths:
    def __init__(self) -> None:
        # [rx, rx_ant, tx, tx_ant, path, time]
        self._a = np.ones((1, 2, 1, 1, 3, 1), dtype=np.complex128)
        self._tau = np.asarray([[[10e-9, 20e-9, 30e-9]]])
        # 第三条路径落在地面 z=0，应该被二维筛选丢弃。
        self.vertices = np.zeros((1, 1, 1, 3, 3), dtype=float)
        self.vertices[0, 0, 0, 1, 2] = 1.5
        self.vertices[0, 0, 0, 2, 2] = 0.0
        self.interactions = np.asarray([[[[0, 1, 1]]]])
        self.phi_r = np.asarray([[[0.1, 0.2, 0.3]]])

    def cir(self, **kwargs):
        assert kwargs["normalize_delays"] is False
        return self._a, self._tau


def test_sionna_planar_filter_preserves_absolute_delay() -> None:
    frequencies = np.asarray([0.0, 1e6, 2e6])
    csi, metadata = extract_planar_uplink_csi(
        FakePaths(),
        frequencies,
        fixed_height_m=1.5,
        vertical_tolerance_m=0.1,
        max_reflections=2,
    )
    assert csi.shape == (2, 3)
    assert metadata["retained_mask"].tolist() == [True, True, False]
    assert np.allclose(metadata["absolute_delays_s"], [10e-9, 20e-9, 30e-9])


@pytest.mark.parametrize("max_diffractions,expected", [
    (0, [True, True, False, False, False, False]),
    (1, [True, True, True, True, False, False]),
])
def test_sionna_filters_by_interaction_type_and_excludes_multiple_diffractions(max_diffractions, expected):
    paths = FakePaths()
    paths._a = np.ones((1, 2, 1, 1, 6, 1), dtype=complex)
    paths._tau = np.arange(1, 7).reshape(1, 1, 6) * 10e-9
    paths.vertices = np.zeros((2, 1, 1, 6, 3))
    paths.vertices[..., 2] = 1.5
    paths.interactions = np.asarray([[0, 1, 8, 1, 8, 2], [0, 0, 0, 8, 8, 0]]).reshape(2, 1, 1, 6)
    paths.phi_r = np.zeros((1, 1, 6))
    _, metadata = extract_planar_uplink_csi(paths, np.array([0., 1e6]), fixed_height_m=1.5,
        vertical_tolerance_m=.1, max_reflections=1, max_diffractions=max_diffractions)
    assert metadata["retained_mask"].tolist() == expected
    assert metadata["reflection_order"].tolist() == [0, 1, 0, 1, 0, 0]
    assert metadata["diffraction_order"].tolist() == [0, 0, 1, 1, 2, 0]


def test_sionna_front_filter_uses_boresight_and_records_both_counts(tmp_path) -> None:
    paths = FakePaths()
    # 让三条路径都先通过二维高度筛选；相对 30° 阵列朝向，它们的局部角是
    # 0°、40°、-110°，因此最后一条必须在 CSI 合成前被剔除。
    paths.vertices[0, 0, 0, 2, 2] = 1.5
    paths.phi_r = np.deg2rad(np.asarray([[[30.0, 70.0, -80.0]]]))
    frequencies = np.asarray([0.0, 1e6, 2e6])

    csi, metadata = extract_planar_uplink_csi(
        paths,
        frequencies,
        fixed_height_m=1.5,
        vertical_tolerance_m=0.1,
        max_reflections=2,
        bs_boresight_rad=np.deg2rad(30.0),
        local_angle_min_rad=np.deg2rad(-60.0),
        local_angle_max_rad=np.deg2rad(60.0),
        front_facing_only=True,
    )

    assert metadata["planar_retained_mask_before_front_filter"].tolist() == [
        True,
        True,
        True,
    ]
    assert metadata["front_facing_angle_mask"].tolist() == [True, True, False]
    assert metadata["retained_mask"].tolist() == [True, True, False]
    assert np.allclose(np.rad2deg(metadata["aoa_local_rad"]), [0.0, 40.0, -110.0])
    assert csi.shape == (2, 3)

    summary = _path_selection_summary(metadata)
    assert summary["total_sionna_path_count"] == 3
    assert summary["planar_path_count_before_front_filter"] == 3
    assert summary["retained_path_count_after_front_filter"] == 2

    online = OnlineMeasurement(
        csi_observed=csi[np.newaxis, ...],
        subcarrier_frequencies_hz=frequencies,
        carrier_frequency_hz=3.5e9,
        antenna_spacing_m=0.04,
        bs_position_m=np.asarray([0.0, 0.0]),
        bs_boresight_rad=np.deg2rad(30.0),
    )
    _save_real_bundle(
        tmp_path,
        online,
        ue_position_m=np.asarray([1.0, 1.0]),
        clock_bias_s=0.0,
        csi_geometric=csi[np.newaxis, ...],
        injected_noise_std=0.0,
        path_metadata=metadata,
    )
    import json

    truth_metadata = json.loads(
        (tmp_path / "data" / "truth" / "ground_truth.json").read_text(
            encoding="utf-8"
        )
    )
    assert truth_metadata["path_selection"] == summary


def test_deepmimo_export_scalar_arrays_are_normalized(tmp_path) -> None:
    parameter_path = tmp_path / "sionna_rt_params.pkl"
    with parameter_path.open("wb") as handle:
        pickle.dump(
            {
                "frequency": np.asarray([3.5e9], dtype=np.float32),
                "bandwidth": np.asarray([400e6], dtype=np.float32),
                "los": True,
            },
            handle,
        )
    with (tmp_path / "sionna_materials.pkl").open("wb") as handle:
        pickle.dump(
            [
                {
                    "conductivity": np.asarray([0.1]),
                    "relative_permittivity": np.asarray([4.0]),
                    "scattering_coefficient": np.asarray([0.0]),
                    "xpd_coefficient": np.asarray([0.0]),
                    "alpha_r": None,
                    "alpha_i": None,
                    "lambda_": None,
                }
            ],
            handle,
        )

    changed = _normalize_deepmimo_export_scalars(tmp_path)

    with parameter_path.open("rb") as handle:
        normalized = pickle.load(handle)
    with (tmp_path / "sionna_materials.pkl").open("rb") as handle:
        materials = pickle.load(handle)
    assert changed[:2] == ("frequency", "bandwidth")
    assert isinstance(normalized["frequency"], float)
    assert isinstance(normalized["bandwidth"], float)
    assert isinstance(materials[0]["conductivity"], float)
    assert isinstance(materials[0]["relative_permittivity"], float)


def test_localization_scene_bounds_do_not_depend_on_ue_or_path_truth() -> None:
    scene_config = {
        "source": "sionna_builtin",
        "localization_bounds_m": [-100.0, 160.0, -80.0, 180.0],
    }
    ue_position = np.asarray([50.0, 50.0])
    path_metadata = {
        "vertices_m": np.asarray([[[150.0, -40.0, 1.5]]]),
    }

    # 固定范围助手在接口层就不能接收 UE 或路径真值；以后若重新引入这些参数，
    # 该检查会直接失败。
    assert tuple(inspect.signature(_localization_scene_bounds).parameters) == (
        "scene_config",
    )
    first = _localization_scene_bounds(scene_config)
    ue_position[:] = [900.0, -900.0]
    path_metadata["vertices_m"][:] = 1.0e6
    second = _localization_scene_bounds(scene_config)

    assert first == (-100.0, 160.0, -80.0, 180.0)
    assert second == first


@pytest.mark.parametrize(
    "bounds",
    (
        [-100.0, 160.0, -80.0],
        [-100.0, np.inf, -80.0, 180.0],
        [160.0, -100.0, -80.0, 180.0],
        [-100.0, 160.0, 180.0, -80.0],
    ),
)
def test_localization_scene_bounds_reject_invalid_values(bounds: list[float]) -> None:
    with pytest.raises(ValueError, match="localization_bounds_m"):
        _localization_scene_bounds(
            {"source": "sionna_builtin", "localization_bounds_m": bounds}
        )


def test_localization_scene_bounds_are_required() -> None:
    with pytest.raises(ValueError, match="必须显式提供"):
        _localization_scene_bounds({"source": "sionna_builtin"})


def test_reverse_scene_self_check_rejects_wall_in_front_of_los() -> None:
    scene = Scene2D(
        name="bad_slice",
        bounds_m=(-1.0, 2.0, -1.0, 1.0),
        walls=(WallSegment("fabricated", (0.5, -1.0), (0.5, 1.0)),),
        fixed_height_m=1.5,
        bev_resolution_m=0.1,
        source="test",
    )
    metadata = {
        "retained_mask": np.asarray([True]),
        "aoa_global_rad": np.asarray([0.0]),
        "interactions": np.zeros((1, 1), dtype=int),
        "vertices_m": np.zeros((1, 1, 3), dtype=float),
    }

    with pytest.raises(RuntimeError, match="直射.*提前挡住"):
        _validate_reverse_scene_consistency(
            scene,
            bs_xy=np.asarray([0.0, 0.0]),
            ue_xy=np.asarray([1.0, 0.0]),
            path_metadata=metadata,
            tolerance_m=0.01,
        )


def test_reverse_scene_self_check_accepts_true_reflection_point() -> None:
    scene = Scene2D(
        name="good_slice",
        bounds_m=(-1.0, 2.0, -1.0, 1.0),
        walls=(WallSegment("true_wall", (1.0, -1.0), (1.0, 1.0)),),
        fixed_height_m=1.5,
        bev_resolution_m=0.1,
        source="test",
    )
    metadata = {
        "retained_mask": np.asarray([True]),
        "aoa_global_rad": np.asarray([0.0]),
        "interactions": np.ones((1, 1), dtype=int),
        "vertices_m": np.asarray([[[1.0, 0.0, 1.5]]]),
    }

    diagnostics = _validate_reverse_scene_consistency(
        scene,
        bs_xy=np.asarray([0.0, 0.0]),
        ue_xy=np.asarray([-0.5, 0.0]),
        path_metadata=metadata,
        tolerance_m=0.01,
    )

    assert diagnostics == {
        "checked_path_count": 1,
        "los_path_count": 0,
        "reflected_path_count": 1,
        "position_tolerance_m": 0.01,
        "passed": True,
    }


def test_reverse_scene_self_check_rejects_ue_outside_fixed_bounds() -> None:
    scene = Scene2D(
        name="fixed_slice",
        bounds_m=(-1.0, 2.0, -1.0, 1.0),
        walls=(),
        fixed_height_m=1.5,
        bev_resolution_m=0.1,
        source="test",
    )
    metadata = {
        "retained_mask": np.asarray([True]),
        "aoa_global_rad": np.asarray([0.0]),
        "interactions": np.zeros((1, 1), dtype=int),
        "vertices_m": np.zeros((1, 1, 3), dtype=float),
    }

    with pytest.raises(RuntimeError, match="UE 坐标.*超出固定定位区域"):
        _validate_reverse_scene_consistency(
            scene,
            bs_xy=np.asarray([0.0, 0.0]),
            ue_xy=np.asarray([3.0, 0.0]),
            path_metadata=metadata,
            tolerance_m=0.01,
        )


def test_reverse_scene_self_check_rejects_interaction_outside_fixed_bounds() -> None:
    scene = Scene2D(
        name="fixed_slice",
        bounds_m=(-1.0, 2.0, -1.0, 1.0),
        walls=(WallSegment("inside_wall", (1.0, -1.0), (1.0, 1.0)),),
        fixed_height_m=1.5,
        bev_resolution_m=0.1,
        source="test",
    )
    metadata = {
        "retained_mask": np.asarray([True]),
        "aoa_global_rad": np.asarray([0.0]),
        "interactions": np.ones((1, 1), dtype=int),
        "vertices_m": np.asarray([[[3.0, 0.0, 1.5]]]),
    }

    with pytest.raises(RuntimeError, match="交互点层.*超出固定定位区域"):
        _validate_reverse_scene_consistency(
            scene,
            bs_xy=np.asarray([0.0, 0.0]),
            ue_xy=np.asarray([0.0, 0.0]),
            path_metadata=metadata,
            tolerance_m=0.01,
        )
