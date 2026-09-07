from __future__ import annotations

import numpy as np
import pytest

import time_bias_localization.scene as scene_module
from time_bias_localization.scene import Scene2D, WallSegment, make_synthetic_room


_TARGETS = ("scene_2d.json", "scene_bev.png", "scene_occupancy.npy")


def _formal_paths(root):
    return tuple(root / name for name in _TARGETS)


def test_scene_save_success_writes_readable_complete_files(tmp_path) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)

    artifacts = scene.save(tmp_path)

    assert {path for path in artifacts.values()} == {
        str(path) for path in _formal_paths(tmp_path)
    }
    loaded = Scene2D.load(tmp_path / "scene_2d.json")
    assert loaded.name == scene.name
    assert loaded.bounds_m == scene.bounds_m
    assert [wall.wall_id for wall in loaded.walls] == [
        wall.wall_id for wall in scene.walls
    ]
    assert all(
        np.allclose(loaded_wall.start, scene_wall.start)
        and np.allclose(loaded_wall.end, scene_wall.end)
        for loaded_wall, scene_wall in zip(loaded.walls, scene.walls, strict=True)
    )
    assert np.load(tmp_path / "scene_occupancy.npy").shape == (
        scene.height_px,
        scene.width_px,
    )
    with scene_module.Image.open(tmp_path / "scene_bev.png") as image:
        assert image.size == (scene.width_px, scene.height_px)
    assert not tuple(tmp_path.glob(".*.tmp.*"))


def test_scene_from_dict_rejects_extra_truth_fields() -> None:
    data = make_synthetic_room(bev_resolution_m=0.2).to_dict()
    data["ue_position_m"] = [14.0, 4.0]

    with pytest.raises(ValueError, match="字段集合不符合契约.*额外"):
        Scene2D.from_dict(data)

    data = make_synthetic_room(bev_resolution_m=0.2).to_dict()
    data["walls"][0]["clock_bias_s"] = 25.0e-9
    with pytest.raises(ValueError, match="墙的字段集合不符合契约"):
        Scene2D.from_dict(data)


@pytest.mark.parametrize(
    "start_m,end_m,expected_message",
    [
        ((0.0,), (1.0, 0.0), "形状 \\(2,\\)"),
        ((0.0, 0.0, 0.0), (1.0, 0.0), "形状 \\(2,\\)"),
        ((0.0, np.nan), (1.0, 0.0), "全部为有限数"),
        ((0.0, 0.0), (np.inf, 0.0), "全部为有限数"),
    ],
)
def test_wall_segment_rejects_non_2d_or_nonfinite_endpoints(
    start_m, end_m, expected_message: str
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        WallSegment("bad_wall", start_m, end_m, "test")


@pytest.mark.parametrize("conflicting_name", _TARGETS)
def test_scene_save_single_conflict_rejects_before_any_write(
    tmp_path, conflicting_name: str
) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)
    conflicting_path = tmp_path / conflicting_name
    old_bytes = b"preserve-existing-scene-file-byte-for-byte"
    conflicting_path.write_bytes(old_bytes)

    with pytest.raises(FileExistsError, match="拒绝覆盖整组文件"):
        scene.save(tmp_path)

    assert conflicting_path.read_bytes() == old_bytes
    assert all(
        path == conflicting_path or not path.exists()
        for path in _formal_paths(tmp_path)
    )


def test_scene_save_rerun_keeps_every_old_file_unchanged(tmp_path) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)
    scene.save(tmp_path)
    before = {path: path.read_bytes() for path in _formal_paths(tmp_path)}

    with pytest.raises(FileExistsError, match="拒绝覆盖整组文件"):
        scene.save(tmp_path)

    assert {path: path.read_bytes() for path in before} == before


def test_scene_save_treats_broken_symlink_as_existing_target(tmp_path) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)
    target = tmp_path / "scene_2d.json"
    target.symlink_to(tmp_path / "missing-scene.json")

    with pytest.raises(FileExistsError, match="拒绝覆盖整组文件"):
        scene.save(tmp_path)

    assert target.is_symlink()
    assert all(path == target or not path.exists() for path in _formal_paths(tmp_path))


def test_scene_save_explicit_overwrite_replaces_complete_group(tmp_path) -> None:
    first_scene = make_synthetic_room(bev_resolution_m=0.2)
    second_scene = make_synthetic_room(
        bounds_m=(0.0, 10.0, 0.0, 6.0),
        fixed_height_m=2.0,
        bev_resolution_m=0.25,
    )
    first_scene.save(tmp_path)
    before = {path: path.read_bytes() for path in _formal_paths(tmp_path)}

    second_scene.save(tmp_path, allow_overwrite=True)

    after = {path: path.read_bytes() for path in _formal_paths(tmp_path)}
    assert all(after[path] != before[path] for path in before)
    loaded = Scene2D.load(tmp_path / "scene_2d.json")
    assert loaded.bounds_m == second_scene.bounds_m
    assert loaded.fixed_height_m == second_scene.fixed_height_m
    assert np.load(tmp_path / "scene_occupancy.npy").shape == (
        second_scene.height_px,
        second_scene.width_px,
    )
    with scene_module.Image.open(tmp_path / "scene_bev.png") as image:
        assert image.size == (second_scene.width_px, second_scene.height_px)
    assert not tuple(tmp_path.glob(".*.tmp.*"))
    assert not tuple(tmp_path.glob(".*.backup.*"))


def test_scene_save_overwrite_publish_failure_restores_old_group(
    tmp_path, monkeypatch
) -> None:
    first_scene = make_synthetic_room(bev_resolution_m=0.2)
    second_scene = make_synthetic_room(
        bounds_m=(0.0, 10.0, 0.0, 6.0),
        fixed_height_m=2.0,
        bev_resolution_m=0.25,
    )
    first_scene.save(tmp_path)
    formal_paths = set(_formal_paths(tmp_path))
    before = {path: path.read_bytes() for path in formal_paths}
    real_replace = scene_module.os.replace
    publish_count = 0

    def fail_second_new_publish(source, destination):
        nonlocal publish_count
        source_path = scene_module.Path(source)
        destination_path = scene_module.Path(destination)
        if destination_path in formal_paths and ".tmp" in source_path.name:
            publish_count += 1
            if publish_count == 2:
                raise OSError("planned-overwrite-publish-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(scene_module.os, "replace", fail_second_new_publish)

    with pytest.raises(OSError, match="planned-overwrite-publish-failure"):
        second_scene.save(tmp_path, allow_overwrite=True)

    assert {path: path.read_bytes() for path in formal_paths} == before
    assert tuple(tmp_path.glob(".*.tmp.*")), "失败临时文件应保留用于排查"
    assert not tuple(tmp_path.glob(".*.backup.*"))


def test_scene_save_stage_failure_leaves_no_formal_target(tmp_path, monkeypatch) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)

    def fail_npy_write(handle, array):
        handle.write(b"partial-occupancy-npy")
        raise OSError("planned-npy-write-failure")

    monkeypatch.setattr(scene_module.np, "save", fail_npy_write)

    with pytest.raises(OSError, match="planned-npy-write-failure"):
        scene.save(tmp_path)

    assert all(not path.exists() for path in _formal_paths(tmp_path))
    assert tuple(tmp_path.glob(".*.tmp.*")), "失败临时文件应保留用于排查"


def test_scene_save_json_stage_failure_leaves_no_formal_target(
    tmp_path, monkeypatch
) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)
    real_stage = scene_module._stage_scene_file

    def fail_scene_json(path, writer):
        if path.name == "scene_2d.json":
            def write_partial_json_then_fail(handle):
                handle.write(b'{"partial":')
                raise OSError("planned-json-write-failure")

            return real_stage(path, write_partial_json_then_fail)
        return real_stage(path, writer)

    monkeypatch.setattr(scene_module, "_stage_scene_file", fail_scene_json)

    with pytest.raises(OSError, match="planned-json-write-failure"):
        scene.save(tmp_path)

    assert all(not path.exists() for path in _formal_paths(tmp_path))
    assert tuple(tmp_path.glob(".*.tmp.*")), "失败临时文件应保留用于排查"


def test_scene_save_publish_failure_rolls_back_formal_targets(
    tmp_path, monkeypatch
) -> None:
    scene = make_synthetic_room(bev_resolution_m=0.2)
    real_replace = scene_module.os.replace
    publish_count = 0
    formal_paths = set(_formal_paths(tmp_path))

    def fail_second_publish(source, destination):
        nonlocal publish_count
        destination_path = scene_module.Path(destination)
        if destination_path in formal_paths:
            publish_count += 1
            if publish_count == 2:
                raise OSError("planned-publish-failure")
        return real_replace(source, destination)

    monkeypatch.setattr(scene_module.os, "replace", fail_second_publish)

    with pytest.raises(OSError, match="planned-publish-failure"):
        scene.save(tmp_path)

    assert all(not path.exists() for path in formal_paths)
    assert tuple(tmp_path.glob(".*.tmp.*")), "失败临时文件应保留用于排查"
