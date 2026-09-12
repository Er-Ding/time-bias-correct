"""后台任务的运行记录、状态查询与整组停止。由 .sh 配置工作目录和环境。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": pid, "start_ticks": stat[19], "pgid": int(stat[2])}
    except (FileNotFoundError, ProcessLookupError):
        return None


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status", "stop"))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    record_path = directory / "task.json"
    if args.action != "run":
        record = json.loads(record_path.read_text())
        finished = directory / "completion.json"
        if finished.is_file():
            print(f"运行记录：{directory}")
            print(finished.read_text())
            return
        task_live = identity(record["task"]["pid"]) == record["task"]
        supervisor_live = identity(record["supervisor"]["pid"]) == record["supervisor"]
        if args.action == "status":
            print(json.dumps({"task_running": task_live, "supervisor_running": supervisor_live,
                              "completion_record_exists": False, "record": record}, ensure_ascii=False, indent=2))
            if not task_live and not supervisor_live:
                print("进程已不在，但没有结束记录，不能判为成功。")
            return
        if not supervisor_live or record["supervisor"]["pgid"] != record["supervisor"]["pid"]:
            raise SystemExit("监督进程身份不匹配；拒绝停止，避免误停其他任务。")
        write(directory / "stop_request.json", {"time": timestamp(), "signal": "SIGTERM"})
        os.killpg(record["supervisor"]["pgid"], signal.SIGTERM)
        print("已核对进程身份，并向整项任务及其子进程发送停止信号。")
        return
    if record_path.exists() or (directory / "completion.json").exists():
        raise SystemExit("运行记录已存在，拒绝覆盖。")
    if os.getpid() != os.getpgrp():
        raise SystemExit("任务必须由 setsid 启动独立进程组。")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("没有任务命令。")
    # SIGTERM 同时发给整个进程组；监督进程等待子任务退出并写好结束记录。
    signal.signal(signal.SIGTERM, lambda *_: None)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    started = timestamp()
    timer = time.monotonic()
    child = None
    exit_code = 1
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        task_identity = identity(child.pid)
        if task_identity is None:
            raise RuntimeError("任务过早退出，无法保存进程身份")
        write(record_path, {"started_at": started, "cwd": str(Path.cwd()), "command": command,
                            "supervisor": identity(os.getpid()), "task": task_identity,
                            "log": str(directory / "task.log")})
        (directory / "task.pid").write_text(f"{child.pid}\n")
        print(f"后台任务 PID={child.pid}，运行记录={directory}", flush=True)
        code = child.wait()
        exit_code = code if code >= 0 else 128 - code
    finally:
        write(directory / "completion.json", {"started_at": started, "finished_at": timestamp(),
             "duration_s": time.monotonic() - timer, "exit_code": exit_code,
             "status": "success" if exit_code == 0 else "failed"})
        print(f"任务结束，退出码={exit_code}", flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
