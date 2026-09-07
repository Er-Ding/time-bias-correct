"""定位产物的稳定序列化与内容哈希工具。"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np


_GENERATION_ARTIFACT_LABELS = {
    "scene_json": "二维场景",
    "online_measurement": "在线 CSI",
    "ground_truth": "评估真值",
}


@dataclass(frozen=True)
class CapturedFile:
    """一次读取后冻结的文件路径、原始字节和同一份字节的摘要。"""

    path: Path
    data: bytes
    sha256: str

    def artifact_record(self) -> dict[str, str]:
        """返回可直接写进来源清单的路径和摘要。"""

        return {"path": str(self.path), "sha256": self.sha256}


def capture_file(path: str | Path) -> CapturedFile:
    """只打开文件一次，并用实际返回给解析器的字节计算 SHA-256。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"需要读取的文件不存在：{resolved}")
    data = resolved.read_bytes()
    return CapturedFile(path=resolved, data=data, sha256=sha256(data).hexdigest())


@contextmanager
def exclusive_output_root_lock(output_root: str | Path) -> Iterator[Path]:
    """用输出根目录内的同一文件锁串行化定位和评估进程。"""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".time_bias_localization.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _json_value(value: Any) -> Any:
    """转成可稳定排序的普通 JSON 值。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_sha256(value: Any) -> str:
    """计算与缩进和键顺序无关的 JSON SHA-256。"""

    encoded = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """流式计算文件 SHA-256，避免一次读入较大的 CSI 文件。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"需要计算哈希的文件不存在：{resolved}")
    digest = sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: str | Path) -> dict[str, str]:
    """记录一个产物的绝对路径和当前内容哈希。"""

    resolved = Path(path).expanduser().resolve()
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def generation_artifact_record(
    manifest: Mapping[str, Any], artifact_name: str
) -> dict[str, str]:
    """读取并规范化生成清单中的一个必需文件记录。"""

    if artifact_name not in _GENERATION_ARTIFACT_LABELS:
        raise ValueError(f"未知生成产物类型：{artifact_name}")
    try:
        record = manifest["artifact_hashes"][artifact_name]
        path_value = record["path"]
        digest = str(record["sha256"])
    except (KeyError, TypeError) as error:
        label = _GENERATION_ARTIFACT_LABELS[artifact_name]
        raise ValueError(f"生成清单缺少{label}的 path+sha256 记录") from error
    try:
        resolved = Path(path_value).expanduser().resolve()
    except (TypeError, ValueError, OSError) as error:
        label = _GENERATION_ARTIFACT_LABELS[artifact_name]
        raise ValueError(f"生成清单中的{label}路径无效") from error
    if len(digest) != 64:
        label = _GENERATION_ARTIFACT_LABELS[artifact_name]
        raise ValueError(f"生成清单中的{label} SHA-256 无效")
    try:
        int(digest, 16)
    except ValueError as error:
        label = _GENERATION_ARTIFACT_LABELS[artifact_name]
        raise ValueError(f"生成清单中的{label} SHA-256 无效") from error
    return {"path": str(resolved), "sha256": digest.lower()}


def generation_bundle_id(manifest: Mapping[str, Any]) -> str:
    """由场景、在线 CSI 和真值的内容摘要生成稳定批次编号。"""

    hashes = {
        artifact_name: generation_artifact_record(manifest, artifact_name)["sha256"]
        for artifact_name in _GENERATION_ARTIFACT_LABELS
    }
    return canonical_json_sha256(
        {"schema_version": 1, "generation_artifact_sha256": hashes}
    )


def load_generation_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, str], str]:
    """一次读取生成清单，并返回内容、文件记录和稳定批次编号。"""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"生成清单不存在：{resolved}")
    captured = capture_file(resolved)
    try:
        manifest = json.loads(captured.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"生成清单不是有效 JSON：{resolved}") from error
    if not isinstance(manifest, dict):
        raise ValueError("生成清单顶层必须是键值映射")
    bundle_id = generation_bundle_id(manifest)
    declared_bundle_id = manifest.get("bundle_id")
    if declared_bundle_id is not None and declared_bundle_id != bundle_id:
        raise ValueError("生成清单的 bundle_id 与其文件摘要记录不一致")
    manifest_record = captured.artifact_record()
    return manifest, manifest_record, bundle_id


def verify_generation_artifact(
    manifest: Mapping[str, Any],
    artifact_name: str,
    actual_path: str | Path | CapturedFile,
) -> dict[str, str]:
    """确认调用方文件的路径和内容均与生成清单属于同一批。"""

    record = generation_artifact_record(manifest, artifact_name)
    captured = (
        actual_path
        if isinstance(actual_path, CapturedFile)
        else capture_file(actual_path)
    )
    resolved = captured.path
    label = _GENERATION_ARTIFACT_LABELS[artifact_name]
    if Path(record["path"]) != resolved:
        raise ValueError(
            f"{label}的路径与生成清单不一致：输入={resolved}，"
            f"清单={record['path']}"
        )
    actual_sha256 = captured.sha256
    if actual_sha256 != record["sha256"]:
        raise ValueError(f"{label} SHA-256 与生成清单不一致；文件可能已被修改")
    return {"path": str(resolved), "sha256": actual_sha256}


def localization_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """生成不含运行时内部字段的完整定位配置快照。"""

    source_path_value = config.get("_config_path")
    source_path = (
        str(Path(str(source_path_value)).expanduser().resolve())
        if source_path_value is not None
        else None
    )
    resolved_config = {
        str(key): _json_value(value)
        for key, value in config.items()
        if key != "_config_path"
    }
    return {
        "schema_version": 1,
        "source_config_path": source_path,
        "canonical_sha256": canonical_json_sha256(resolved_config),
        "resolved_config": resolved_config,
    }


__all__ = [
    "CapturedFile",
    "artifact_record",
    "canonical_json_sha256",
    "capture_file",
    "exclusive_output_root_lock",
    "file_sha256",
    "generation_artifact_record",
    "generation_bundle_id",
    "load_generation_manifest",
    "localization_config_snapshot",
    "verify_generation_artifact",
]
