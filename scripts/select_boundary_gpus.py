"""核对用户指定的物理 GPU 编号，并保存 CUDA 进程内编号对应关系。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess


def parse_gpu_ids(value: str, *, allow_cpu: bool = False) -> list[int] | None:
    """None 只表示用户明确选择 CPU；空值绝不表示使用全部 GPU。"""
    value = value.strip()
    if value == "cpu":
        if not allow_cpu:
            raise ValueError("真实场景实验需要明确指定 GPU；cpu 仅用于 validate 流程检查。")
        return None
    if not value:
        raise ValueError("请填写 BOUNDARY_GPU_IDS，例如 2,3,4,5,6,7；不会默认使用全部 GPU。")
    values = value.split(",")
    if any(re.fullmatch(r"[0-9]+", item.strip()) is None for item in values):
        raise ValueError("BOUNDARY_GPU_IDS 必须是逗号分隔的非负整数，例如 2,3；不能填写范围或空项。")
    indices = [int(item.strip()) for item in values]
    if len(set(indices)) != len(indices):
        raise ValueError("BOUNDARY_GPU_IDS 中的 GPU 编号不能重复。")
    return indices


def query_inventory() -> dict[int, dict]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"无法通过 nvidia-smi 核对 GPU 编号：{exc}") from exc
    inventory = {}
    for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
        if not row:
            continue
        if len(row) < 3 or re.fullmatch(r"[0-9]+", row[0].strip()) is None:
            raise ValueError("nvidia-smi 返回了无法识别的 GPU 记录。")
        index = int(row[0].strip())
        uuid = row[1].strip()
        if re.fullmatch(r"GPU-[0-9a-fA-F-]+", uuid) is None:
            raise ValueError(f"GPU {index} 的 UUID 无效，无法可靠限定可用设备。")
        if index in inventory or any(item["uuid"] == uuid for item in inventory.values()):
            raise ValueError("nvidia-smi 返回了重复的 GPU 编号或 UUID。")
        inventory[index] = {"physical_index": index, "uuid": uuid, "name": ", ".join(row[2:]).strip()}
    return inventory


def resolve_allocation(value: str, *, allow_cpu: bool = False) -> dict:
    indices = parse_gpu_ids(value, allow_cpu=allow_cpu)
    selected = []
    if indices is not None:
        inventory = query_inventory()
        missing = [index for index in indices if index not in inventory]
        if missing:
            raise ValueError(f"nvidia-smi 中不存在指定的 GPU 编号：{missing}。")
        selected = [dict(inventory[index], logical_index=local) for local, index in enumerate(indices)]
    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "requested_gpu_ids": value,
        "execution_device": "cpu" if indices is None else "cuda",
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": ",".join(item["uuid"] for item in selected),
        "devices": selected,
        "default_logical_device": None if indices is None else 0,
        "library_search_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "selection_rule": "user_supplied_nvidia_smi_indices_resolved_to_uuids",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    try:
        allocation = resolve_allocation(args.gpu_ids, allow_cpu=args.allow_cpu)
        # 每次启动有独立目录；拒绝覆盖已有选择记录。
        with args.record.open("x", encoding="utf-8") as output:
            json.dump(allocation, output, ensure_ascii=False, indent=2)
            output.write("\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"GPU 选择失败：{exc}\n")
    print(allocation["cuda_visible_devices"])


if __name__ == "__main__":
    main()
