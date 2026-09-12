"""用少量数组计算核验 Sionna/DrJit 与 CuPy 的实际 GPU 使用范围。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys


def apps_for_pid(pid: int) -> list[dict]:
    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    records = []
    for row in csv.reader(io.StringIO(output.stdout), skipinitialspace=True):
        if len(row) >= 3 and row[1].strip().isdigit() and int(row[1]) == pid:
            records.append({"gpu_uuid": row[0].strip(), "pid": pid, "used_memory_mib": row[2].strip()})
    return records


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def probe(backend: str, allocation: dict, output: Path) -> None:
    allowed = {item["uuid"] for item in allocation["devices"]}
    result = {"backend": backend, "pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "snapshots": []}
    def snapshot(stage: str, require_context: bool = False) -> list[dict]:
        own_apps = apps_for_pid(os.getpid())
        result["snapshots"].append({"stage": stage, "apps": own_apps})
        write(output, result)
        unauthorized = {app["gpu_uuid"] for app in own_apps} - allowed
        if unauthorized:
            raise RuntimeError(f"探针触及未授权的 GPU：{sorted(unauthorized)}")
        if require_context and not own_apps:
            raise RuntimeError("计算后未能从 nvidia-smi 查到探针 PID，不能判定实际 GPU 使用范围。")
        return own_apps
    try:
        if result["cuda_visible_devices"] != allocation["cuda_visible_devices"]:
            raise RuntimeError("当前 CUDA_VISIBLE_DEVICES 与启动记录不一致。")
        snapshot("before_import")
        count = 4096
        expected = count * (count - 1) / 2
        if backend == "sionna":
            import sionna.rt as rt
            import drjit as dr
            import mitsuba as mi
            result["versions"] = {"sionna_rt": rt.__version__, "drjit": dr.__version__, "mitsuba": mi.__version__}
            result["variant"] = mi.variant()
            snapshot("after_import")
            if not mi.variant().startswith("cuda_"):
                raise RuntimeError("Sionna 没有选择 CUDA，无法完成 GPU 范围验证。")
            values = dr.arange(mi.Float, count)
            total = dr.sum(values)
            dr.eval(total)
            dr.sync_thread()
            result["sum"] = float(total[0])
            result["dlpack_device"] = list(values.__dlpack_device__())
            result["logical_device"] = result["dlpack_device"][1]
            if result["dlpack_device"] != [2, 0]:
                raise RuntimeError("DrJit 数组没有位于 CUDA 逻辑设备 0。")
        else:
            import cupy as cp
            result["versions"] = {"cupy": cp.__version__}
            result["visible_device_count"] = cp.cuda.runtime.getDeviceCount()
            snapshot("after_import")
            if result["visible_device_count"] != len(allowed):
                raise RuntimeError("CuPy 可见设备数量与 GPU 掩码不一致。")
            with cp.cuda.Device(0):
                values = cp.arange(count, dtype=cp.float32)
                total = values.sum()
                cp.cuda.get_current_stream().synchronize()
                result["sum"] = float(total.get())
                result["logical_device"] = cp.cuda.runtime.getDevice()
        if result["sum"] != expected:
            raise RuntimeError("GPU 数组求和结果不正确。")
        active = snapshot("after_computation", require_context=True)
        first_gpu = allocation["devices"][0]["uuid"]
        if first_gpu not in {app["gpu_uuid"] for app in active}:
            raise RuntimeError("程序内设备 0 没有映射到用户选择的第一张物理卡。")
        result["status"] = "success"
        print(f"{backend}：计算正确，逻辑设备 0 对应物理 GPU {allocation['devices'][0]['physical_index']}；仅观察到授权 GPU。", flush=True)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(output, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=("sionna", "cupy"))
    args = parser.parse_args()
    allocation = json.loads(args.allocation.read_text())
    if args.backend:
        probe(args.backend, allocation, args.output)
        return
    args.output.mkdir(exist_ok=False)
    statuses = []
    for backend in ("sionna", "cupy"):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--allocation", str(args.allocation),
                   "--output", str(args.output / f"{backend}.json"), "--backend", backend]
        print(f"开始 {backend} 小数组计算检查。", flush=True)
        worker = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        write(args.output / f"{backend}_process.json", {"pid": worker.pid, "command": command})
        try:
            code = worker.wait(timeout=120)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
            code = 124
        statuses.append({"backend": backend, "pid": worker.pid, "exit_code": code})
        write(args.output / "summary.json", {"status": "success" if len(statuses) == 2 and all(x['exit_code'] == 0 for x in statuses) else "incomplete", "probes": statuses})
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
