"""GPU 编号只来自用户输入，且经过 UUID 映射；无需实际使用 GPU。"""
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
SPEC = spec_from_file_location("select_boundary_gpus", ROOT / "scripts/select_boundary_gpus.py")
selection = module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


@pytest.mark.parametrize("value", ["", " ", "2～7", "2-7", "-1", "1.0", "1,,2", "1,", "1,1", "1,01", "all"])
def test_no_implicit_or_ambiguous_gpu_selection(value):
    with pytest.raises(ValueError):
        selection.parse_gpu_ids(value)


def test_cpu_requires_explicit_validation_choice(monkeypatch):
    monkeypatch.setattr(selection, "query_inventory", lambda: pytest.fail("CPU 检查不应探测 GPU"))
    with pytest.raises(ValueError, match="仅用于 validate"):
        selection.resolve_allocation("cpu")
    result = selection.resolve_allocation("cpu", allow_cpu=True)
    assert result["cuda_visible_devices"] == ""
    assert result["devices"] == []
    assert result["default_logical_device"] is None


def test_physical_selection_maps_to_ordered_uuid_not_physical_cuda_index(monkeypatch):
    inventory = {
        index: dict(physical_index=index, uuid=f"GPU-a000000{index}", name=f"GPU {index}")
        for index in range(8)
    }
    monkeypatch.setattr(selection, "query_inventory", lambda: inventory)
    result = selection.resolve_allocation("7, 2,5")
    assert result["cuda_visible_devices"] == "GPU-a0000007,GPU-a0000002,GPU-a0000005"
    assert [item["physical_index"] for item in result["devices"]] == [7, 2, 5]
    assert [item["logical_index"] for item in result["devices"]] == [0, 1, 2]
    assert result["cuda_device_order"] == "PCI_BUS_ID"
    assert result["default_logical_device"] == 0
    with pytest.raises(ValueError, match="不存在"):
        selection.resolve_allocation("8")


def test_inventory_uses_nvidia_smi_and_rejects_invalid_uuid(monkeypatch):
    calls = []
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="2, GPU-abcd-1234, Tesla V100S\n7, GPU-dcba-4321, Tesla V100S\n")
    monkeypatch.setattr(selection.subprocess, "run", query)
    inventory = selection.query_inventory()
    assert set(inventory) == {2, 7}
    assert calls[0][0] == ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"]
    assert calls[0][1]["timeout"] == 15
    monkeypatch.setattr(selection.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout="2, invalid;command, name\n"))
    with pytest.raises(ValueError, match="UUID 无效"):
        selection.query_inventory()


def test_cli_cpu_record_is_explicit_and_cannot_overwrite(tmp_path):
    record = tmp_path / "gpu_allocation.json"
    command = [os.sys.executable, str(ROOT / "scripts/select_boundary_gpus.py"),
               "--gpu-ids", "cpu", "--allow-cpu", "--record", str(record)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0 and result.stdout == "\n"
    original = record.read_text()
    assert json.loads(original)["execution_device"] == "cpu"
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert record.read_text() == original


def test_launcher_missing_gpu_selection_fails_before_starting(tmp_path):
    output = tmp_path / "must_not_start"
    env = dict(os.environ, BOUNDARY_OUTPUT_ROOT=str(output), BOUNDARY_PYTHON_BIN=os.sys.executable,
               BOUNDARY_MODE="experiment", BOUNDARY_PHASE="pilot", BOUNDARY_GPU_IDS="",
               BOUNDARY_CONFIG_PATH=str(ROOT / "configs/diffraction_boundary_experiment.yaml"),
               BOUNDARY_SOURCE_ROOT="")
    result = subprocess.run(["bash", str(ROOT / "run_boundary_experiment.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "请先填写 BOUNDARY_GPU_IDS" in result.stderr
    assert not output.exists()


def test_status_does_not_need_gpu_selection(tmp_path):
    record = tmp_path / "run_records" / "finished"
    record.mkdir(parents=True)
    (record / "task.json").write_text("{}")
    (record / "completion.json").write_text(json.dumps({"exit_code": 0, "status": "success"}))
    (tmp_path / "latest_run.txt").write_text(str(record) + "\n")
    env = dict(os.environ, BOUNDARY_PYTHON_BIN=os.sys.executable, BOUNDARY_GPU_IDS="")
    result = subprocess.run(["bash", str(ROOT / "run_boundary_experiment.sh"), "status", str(tmp_path)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert '"status": "success"' in result.stdout
    assert "请先填写" not in result.stderr
