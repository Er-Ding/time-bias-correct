"""可选的逐阶段墙钟计时；不改变未开启计时的定位路径。

嵌套记录同时保留包含子步骤和扣除子步骤的时间，汇总时不能重复相加。
尚未执行的步骤不会生成记录；异常退出保留失败步骤已消耗的时间。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Iterator


_CURRENT: ContextVar[TimingRecorder | None] = ContextVar("localization_timing", default=None)


@dataclass
class TimingRecorder:
    synchronize: Callable[[], None] | None = None
    clock: Callable[[], float] = time.perf_counter
    snapshot_path: str | Path | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    marks: dict[str, float] = field(default_factory=dict)
    mark_data: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stack: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _started: float = field(init=False)
    _finished: float | None = field(default=None, init=False)
    _status: str = field(default="running", init=False)
    _snapshot_write_s: float = field(default=0.0, init=False)
    _snapshot_write_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._started = self.clock()

    def sync(self) -> None:
        if self.synchronize is not None:
            self.synchronize()

    def _snapshot(self) -> None:
        if self.snapshot_path is None:
            return
        started = time.perf_counter()
        path = Path(self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
            self._snapshot_write_s += time.perf_counter() - started
            self._snapshot_write_count += 1

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[None]:
        if not name:
            raise ValueError("计时步骤名称不能为空")
        # 上一步异步任务不能算进本步骤。结束同步计入本步骤的墙钟时间。
        self.sync()
        started = self.clock()
        event: dict[str, Any] = {
            "event_id": len(self.events), "name": name,
            "parent_event_id": self._stack[-1]["event_id"] if self._stack else None,
            "depth": len(self._stack), "start_s": started - self._started,
            "end_s": None, "elapsed_s": None, "exclusive_s": None,
            "status": "running", "metadata": metadata, "_child_s": 0.0,
        }
        self.events.append(event)
        self._stack.append(event)
        try:
            self._snapshot()
            yield
        except BaseException as error:
            event["status"] = "failed"
            event["error_type"] = type(error).__name__
            raise
        else:
            event["status"] = "complete"
        finally:
            # 同步失败仍保存时间，并避免覆盖原始定位异常。
            try:
                self.sync()
            except BaseException as error:
                already_failed = event["status"] == "failed"
                event["status"] = "failed"
                event["synchronization_error"] = f"{type(error).__name__}: {error}"
                if not already_failed:
                    raise
            finally:
                ended = self.clock()
                elapsed = max(0.0, ended - started)
                event["end_s"] = ended - self._started
                event["elapsed_s"] = elapsed
                event["exclusive_s"] = max(0.0, elapsed - event.pop("_child_s"))
                self._stack.pop()
                if self._stack:
                    self._stack[-1]["_child_s"] += elapsed
                self._snapshot()

    def mark(self, name: str, **data: Any) -> None:
        if name in self.marks:
            raise ValueError(f"计时边界重复: {name}")
        self.sync()
        self.marks[name] = self.clock() - self._started
        if data:
            self.mark_data[name] = data
        self._snapshot()

    def to_dict(self) -> dict[str, Any]:
        ended = self.clock() if self._finished is None else self._finished
        events = [{key: value for key, value in event.items() if not key.startswith("_")}
                  for event in self.events]
        groups: dict[str, dict[str, Any]] = {}
        for event in events:
            row = groups.setdefault(event["name"], {
                "name": event["name"], "count": 0, "complete_count": 0,
                "failed_count": 0, "running_count": 0, "elapsed_s": 0.0,
                "exclusive_s": 0.0, "complete_elapsed_s": 0.0,
                "failed_elapsed_s": 0.0,
            })
            row["count"] += 1
            row[f"{event['status']}_count"] += 1
            if event["elapsed_s"] is not None:
                row["elapsed_s"] += event["elapsed_s"]
                row["exclusive_s"] += event["exclusive_s"]
                row[f"{event['status']}_elapsed_s"] += event["elapsed_s"]
        for row in groups.values():
            row["status"] = ("failed" if row["failed_count"] else
                             "running" if row["running_count"] else "complete")
        elapsed = max(0.0, ended - self._started)
        recorded = sum(event["exclusive_s"] or 0.0 for event in events)
        return {
            "schema_version": 1, "clock": "perf_counter_wall_time",
            "started_perf_counter_s": self._started if self.clock is time.perf_counter else None,
            "status": self._status, "elapsed_s": elapsed,
            "synchronized": self.synchronize is not None,
            "events": events, "stages": list(groups.values()),
            "marks": dict(self.marks), "mark_data": dict(self.mark_data),
            "recorded_exclusive_s": recorded,
            "unattributed_s": max(0.0, elapsed - recorded),
            "snapshot_write_s": self._snapshot_write_s,
            "snapshot_write_count": self._snapshot_write_count,
            "snapshot_policy": "atomic replace; running event has no completed duration",
            "not_executed_policy": "absent; never a successful zero duration",
            "summation_rule": "sum exclusive_s; never add parents and inclusive children",
        }


@contextmanager
def collect_timings(
    *, synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    snapshot_path: str | Path | None = None,
) -> Iterator[TimingRecorder]:
    """例如 ``with collect_timings() as recorder: localize(...)``。"""
    recorder = TimingRecorder(synchronize=synchronize, clock=clock, snapshot_path=snapshot_path)
    token = _CURRENT.set(recorder)
    try:
        yield recorder
    except BaseException:
        recorder._status = "failed"
        raise
    else:
        recorder._status = "complete"
    finally:
        recorder._finished = recorder.clock()
        _CURRENT.reset(token)
        recorder._snapshot()


@contextmanager
def stage(name: str, **metadata: Any) -> Iterator[None]:
    recorder = _CURRENT.get()
    if recorder is None:
        yield
    else:
        with recorder.stage(name, **metadata):
            yield


def mark(name: str, **data: Any) -> None:
    recorder = _CURRENT.get()
    if recorder is not None:
        recorder.mark(name, **data)


def register_synchronizer(synchronize: Callable[[], None]) -> None:
    """只补充尚未指定的同步器，保留实验调用方显式指定的设备同步。"""
    recorder = _CURRENT.get()
    if recorder is not None and recorder.synchronize is None:
        recorder.synchronize = synchronize


def current_timings() -> dict[str, Any] | None:
    recorder = _CURRENT.get()
    return None if recorder is None else recorder.to_dict()
