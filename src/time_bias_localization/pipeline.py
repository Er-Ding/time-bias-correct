"""从场景和 CSI 到位置均值/协方差的分阶段流水线。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Sequence
from uuid import uuid4

import numpy as np

from .candidates import (
    PathObservationSample,
    RawReverseTrajectory,
    cluster_reverse_candidates,
    generate_reverse_candidates,
    local_to_global_aoa,
)
from .config import (
    load_config,
    localization_config_view,
    validate_localization_config,
)
from .constants import SPEED_OF_LIGHT_M_S
from .contracts import (
    validate_generation_manifest_envelope,
    validate_localization_input_contract,
)
from .data import (
    OnlineMeasurement,
    generate_synthetic_measurement,
    load_online_measurement_bytes,
    save_measurement_bundle,
)
from .provenance import (
    artifact_record,
    capture_file,
    exclusive_output_root_lock,
    file_sha256,
    generation_bundle_id,
    load_generation_manifest,
    localization_config_snapshot,
    verify_generation_artifact,
)
from .scene import Scene2D, make_synthetic_room
from .music_stage import get_music_computer
from .spectrum_sampling import sample_music_spectrum
from .forward_check import forward_check_solution
from .signal import (
    MusicPeak2D,
    MusicPeakSamples,
    estimate_music_peak_samples,
    extract_local_music_peaks,
    music_2d_spectrum,
)
from .solver import CandidateTrajectory, SolverConfig, SolverError, solve_position_and_bias


@dataclass(frozen=True)
class PeakSetAssociation:
    """一次扰动峰集合与 nominal 路径集合的整体匹配结果。"""

    matches: dict[int, tuple[float, float]]
    nominal_to_sample_index: dict[int, int]
    total_cost: float
    unmatched_sample_indices: tuple[int, ...]
    missed_nominal_indices: tuple[int, ...]
    valid_sample_indices: tuple[int, ...]

    @property
    def distribution_weight(self) -> float:
        """把匹配质量变成扰动解的相对权重；不解释为物理概率。"""

        return 1.0 / (1.0 + self.total_cost)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _encoded_json(data: Any) -> bytes:
    return json.dumps(_jsonable(data), ensure_ascii=False, indent=2).encode("utf-8")


def _write_json(path: Path, data: Any) -> str:
    """在目标目录完整写好临时文件，再原子替换正式 JSON。"""

    encoded = _encoded_json(data)
    _write_bytes_atomic(path, encoded)
    return str(path)


def _stage_bytes_in_target_directory(path: Path, data: bytes) -> Path:
    """在正式目标同目录写完并同步字节，但暂不发布。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise
    return temporary_path


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """在同目录写完字节并同步落盘后，原子替换目标文件。"""

    temporary_path = _stage_bytes_in_target_directory(path, data)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _sync_parent_directories(paths: Sequence[Path]) -> None:
    """同步一组正式目标所在目录的目录项。"""

    for directory in {path.parent for path in paths}:
        directory_descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _evaluation_backup_path(path: Path, transaction_id: str) -> Path:
    """生成与正式目标同目录且不会覆盖现有文件的备份路径。"""

    candidate = path.with_name(
        f".{path.name}.{transaction_id}.evaluation-backup"
    )
    if candidate.exists():
        raise FileExistsError(f"评估事务备份目标已存在：{candidate}")
    return candidate


def _publish_evaluation_pair(
    *,
    metrics_path: Path,
    metrics_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> None:
    """成组发布评估指标和定位清单，失败时恢复两者原状态。"""

    if metrics_path == manifest_path:
        raise ValueError("评估指标与定位清单不能使用同一路径")
    target_payloads = (
        (metrics_path, metrics_bytes),
        (manifest_path, manifest_bytes),
    )
    for target, _ in target_payloads:
        if target.exists() and not target.is_file():
            raise ValueError(f"评估事务目标不是普通文件：{target}")

    staged_files: list[tuple[Path, Path, bytes]] = []
    try:
        for target, payload in target_payloads:
            staged_files.append(
                (
                    target,
                    _stage_bytes_in_target_directory(target, payload),
                    payload,
                )
            )
    except BaseException:
        for _, temporary, _ in staged_files:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass
        raise

    transaction_id = uuid4().hex
    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        for target, _, _ in staged_files:
            if not target.exists():
                continue
            backup = _evaluation_backup_path(target, transaction_id)
            os.replace(target, backup)
            backups[target] = backup
        _sync_parent_directories([target for target, _, _ in staged_files])

        for target, temporary, _ in staged_files:
            os.replace(temporary, target)
            published.add(target)
        _sync_parent_directories([target for target, _, _ in staged_files])

        for target, _, expected_bytes in staged_files:
            if target.read_bytes() != expected_bytes:
                raise RuntimeError(f"评估事务发布后字节校验失败：{target}")
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        # 先恢复或撤回 metrics，再恢复 manifest，避免回滚途中出现清单
        # 已恢复但仍残留本轮新指标的组合。
        for target, _, _ in staged_files:
            backup = backups.get(target)
            try:
                if backup is not None:
                    if not backup.exists():
                        rollback_errors.append(f"旧文件备份意外消失：{backup}")
                        continue
                    os.replace(backup, target)
                elif target in published and target.exists():
                    target.unlink()
            except OSError as error:
                rollback_errors.append(f"恢复评估事务目标失败 {target}：{error}")
        try:
            _sync_parent_directories([target for target, _, _ in staged_files])
        except OSError as error:
            rollback_errors.append(f"同步评估事务回滚目录失败：{error}")
        if rollback_errors:
            raise RuntimeError(
                "评估文件成组发布失败，且回滚不完整："
                + "；".join(rollback_errors)
            ) from publish_error
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink()
            except OSError:
                pass
        try:
            _sync_parent_directories([target for target, _, _ in staged_files])
        except OSError:
            # 两个正式文件已经同步发布；目录二次同步失败时保留成功状态。
            pass
    finally:
        for _, temporary, _ in staged_files:
            if temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass


def _write_new_json_atomic(path: Path, data: Any) -> None:
    """原子创建新的 JSON；目标已存在时绝不覆盖。"""

    encoded = _encoded_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    published = False
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        published = True
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                if not published:
                    raise
                # 正式回执已经原子出现。此时不能因临时文件清理异常把整轮
                # 发布判为失败，否则会留下“回执存在、定位已回滚”的假状态。
                pass


_LEGACY_LOCALIZATION_ARTIFACT_FILENAMES = {
    "result": "localization_result.json",
    "music_spectrum": "music_spectrum.npz",
    "uncertainty_solutions": "uncertainty_solutions.npz",
    "music_peaks": "music_peaks.json",
    "raw_reverse_candidates": "raw_reverse_candidates.json",
    "clustered_candidates": "clustered_candidates.json",
    "bootstrap_diagnostics": "bootstrap_diagnostics.json",
}

WORKFLOW = "music_spectrum_sampling_v1"
_LOCALIZATION_ARTIFACT_FILENAMES = {
    "result": "localization_result.json",
    "music_spectrum": "music_spectrum.npz",
    "music_peaks": "music_peaks.json",
    "spectrum_samples": "spectrum_samples.json",
    "raw_reverse_candidates": "raw_reverse_candidates.json",
    "clustered_candidates": "clustered_candidates.json",
    "forward_check": "forward_check.json",
}


def _manifest_artifact_filenames(manifest: dict[str, Any]) -> dict[str, str]:
    if manifest.get("schema_version") == 4:
        if manifest.get("workflow") != WORKFLOW:
            raise ValueError("第 4 版定位清单必须明确记录谱面采样流程")
        return _LOCALIZATION_ARTIFACT_FILENAMES
    return _LEGACY_LOCALIZATION_ARTIFACT_FILENAMES


def _archive_previous_evaluation(
    root: Path, output_dir: Path, *, archive_sources: bool = False
) -> bool:
    """验证并复制上一轮完整运行到历史目录；返回是否有固定指标待移除。"""

    manifest_path = output_dir / "localization_manifest.json"
    fixed_metrics_path = root / "evaluation" / "metrics.json"
    if not manifest_path.exists():
        if fixed_metrics_path.exists():
            raise ValueError(
                "固定评估指标存在，但旧定位清单缺失；拒绝覆盖或静默保留该文件"
            )
        return False

    manifest_capture = capture_file(manifest_path)
    try:
        manifest = json.loads(manifest_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("旧定位清单不是有效 JSON，拒绝发布新定位产物") from error
    if not isinstance(manifest, dict):
        raise ValueError("旧定位清单顶层不是键值映射，拒绝发布新定位产物")

    evaluation_pending = manifest.get("evaluation_pending")
    if evaluation_pending is True:
        if fixed_metrics_path.exists():
            raise ValueError(
                "旧定位清单仍标记为待评估，但固定 metrics.json 已存在；"
                "无法确认其来源，拒绝继续"
            )
        return False
    if evaluation_pending is not False:
        raise ValueError("旧定位清单缺少明确的评估状态，拒绝发布新定位产物")

    try:
        manifest_schema_version = int(manifest["schema_version"])
        old_run_id = str(manifest["run_id"])
        evaluation_record = manifest["evaluation"]
        recorded_metrics_path = Path(evaluation_record["path"]).expanduser().resolve()
        recorded_metrics_sha256 = str(evaluation_record["sha256"])
        manifest_artifacts = manifest["artifacts"]
        result_record = manifest_artifacts["result"]
        recorded_result_sha256 = str(result_record["sha256"])
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError("旧定位清单的评估来源记录不完整，拒绝发布新定位产物") from error
    if manifest_schema_version not in (2, 3, 4):
        raise ValueError(
            f"旧定位清单版本 {manifest_schema_version} 不支持无损归档"
        )
    if not old_run_id or Path(old_run_id).name != old_run_id or old_run_id in {".", ".."}:
        raise ValueError("旧定位清单的 run_id 不能安全用作历史目录名")
    if recorded_metrics_path != fixed_metrics_path.resolve():
        raise ValueError(
            "旧定位清单记录的评估路径不是固定 evaluation/metrics.json；"
            "拒绝猜测或移动其他文件"
        )

    metrics_capture = capture_file(fixed_metrics_path)
    if metrics_capture.sha256 != recorded_metrics_sha256:
        raise ValueError("旧固定评估指标的 SHA-256 与定位清单不一致")
    try:
        metrics = json.loads(metrics_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("旧固定评估指标不是有效 JSON，拒绝归档") from error
    if not isinstance(metrics, dict) or metrics.get("localization_run_id") != old_run_id:
        raise ValueError("旧固定评估指标的 run_id 与定位清单不一致")
    if metrics.get("source_result_sha256") != recorded_result_sha256:
        raise ValueError("旧固定评估指标的结果摘要与旧定位清单不一致")
    if manifest_schema_version >= 3:
        try:
            old_generation_bundle = manifest["generation_bundle"]
            old_bundle_id = str(old_generation_bundle["bundle_id"])
            recorded_generation_manifest_sha256 = str(
                old_generation_bundle["manifest"]["sha256"]
            )
        except (KeyError, TypeError) as error:
            raise ValueError(
                "第 3 版旧定位清单缺少生成数据批次来源记录"
            ) from error
        if (
            metrics.get("generation_bundle_id") != old_bundle_id
            or metrics.get("source_generation_manifest_sha256")
            != recorded_generation_manifest_sha256
        ):
            raise ValueError("旧固定评估指标的生成批次记录与旧定位清单不一致")

    history_dir = (root / "evaluation" / "history" / old_run_id).resolve()
    files_to_archive: dict[Path, bytes] = {}
    archived_artifacts: dict[str, dict[str, str]] = {}
    migrated_artifact_names: list[str] = []
    for artifact_name, filename in _manifest_artifact_filenames(manifest).items():
        active_path = (output_dir / filename).resolve()
        captured = capture_file(active_path)
        record = manifest_artifacts.get(artifact_name)
        if isinstance(record, dict):
            try:
                recorded_path = Path(record["path"]).expanduser().resolve()
                recorded_sha256 = str(record["sha256"])
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ValueError(f"旧定位清单中的 {artifact_name} 记录无效") from error
            if recorded_path != active_path or recorded_sha256 != captured.sha256:
                raise ValueError(f"旧定位产物 {artifact_name} 的路径或哈希不一致")
        elif manifest_schema_version == 2 and artifact_name != "result":
            # 第 2 版没有登记这些诊断文件。迁移时保存实际字节和现场摘要，
            # 并在历史清单中明确标出摘要的产生时点，不能伪装成旧记录。
            migrated_artifact_names.append(artifact_name)
        else:
            raise ValueError(f"旧定位清单缺少必需产物记录：{artifact_name}")
        history_path = history_dir / filename
        files_to_archive[history_path] = captured.data
        archived_artifacts[artifact_name] = {
            "path": str(history_path),
            "sha256": captured.sha256,
        }

    try:
        config_snapshot = manifest["config_snapshot"]
        config_snapshot_path = Path(config_snapshot["path"]).expanduser().resolve()
        config_snapshot_sha256 = str(config_snapshot["file_sha256"])
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError("旧定位清单的配置快照记录不完整") from error
    config_snapshot_capture = capture_file(config_snapshot_path)
    if config_snapshot_capture.sha256 != config_snapshot_sha256:
        raise ValueError("旧定位配置快照的 SHA-256 与清单不一致")
    history_config_path = history_dir / "localization_config.json"
    files_to_archive[history_config_path] = config_snapshot_capture.data

    history_metrics_path = history_dir / "metrics.json"
    history_manifest_path = history_dir / "localization_manifest.json"
    files_to_archive[history_metrics_path] = metrics_capture.data
    archived_manifest = dict(manifest)
    archived_manifest["result"] = archived_artifacts["result"]["path"]
    archived_manifest["artifacts"] = archived_artifacts
    archived_config_snapshot = dict(config_snapshot)
    archived_config_snapshot["path"] = str(history_config_path)
    archived_manifest["config_snapshot"] = archived_config_snapshot
    archived_manifest["evaluation"] = {
        "path": str(history_metrics_path),
        "sha256": metrics_capture.sha256,
    }
    if migrated_artifact_names:
        archived_manifest["archive_migration"] = {
            "source_schema_version": 2,
            "artifact_hashes_computed_during_archive": sorted(
                migrated_artifact_names
            ),
        }

    if archive_sources:
        archived_sources: dict[str, dict[str, str]] = {}
        for input_name, history_name in (
            ("scene", "scene_input.json"),
            ("online_measurement", "online_measurement.npz"),
        ):
            try:
                input_record = manifest["inputs"][input_name]
                input_path = Path(input_record["path"]).expanduser().resolve()
                input_sha256 = str(input_record["sha256"])
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ValueError(f"旧定位清单缺少 {input_name} 来源记录") from error
            input_capture = capture_file(input_path)
            if input_capture.sha256 != input_sha256:
                raise ValueError(f"旧定位输入 {input_name} 的 SHA-256 与清单不一致")
            history_input_path = history_dir / history_name
            files_to_archive[history_input_path] = input_capture.data
            archived_sources[input_name] = {
                "path": str(history_input_path),
                "sha256": input_capture.sha256,
            }
            archived_manifest["inputs"] = dict(archived_manifest["inputs"])
            archived_manifest["inputs"][input_name] = archived_sources[input_name]
        archived_manifest["scene_input"] = archived_sources["scene"]["path"]
        archived_manifest["online_measurement_input"] = archived_sources[
            "online_measurement"
        ]["path"]

        truth_path = (root / "data" / "truth" / "ground_truth.npz").resolve()
        truth_capture = capture_file(truth_path)
        if truth_capture.sha256 != metrics.get("source_truth_sha256"):
            raise ValueError("旧真值文件的 SHA-256 与旧评估指标不一致")
        history_truth_path = history_dir / "ground_truth.npz"
        files_to_archive[history_truth_path] = truth_capture.data
        archived_sources["ground_truth"] = {
            "path": str(history_truth_path),
            "sha256": truth_capture.sha256,
        }

        if manifest_schema_version >= 3:
            generation_record = manifest["generation_bundle"]["manifest"]
            generation_path = Path(generation_record["path"]).expanduser().resolve()
            generation_capture = capture_file(generation_path)
            if generation_capture.sha256 != str(generation_record["sha256"]):
                raise ValueError("旧生成清单的 SHA-256 与旧定位清单不一致")
            history_generation_path = history_dir / "generation_manifest.json"
            files_to_archive[history_generation_path] = generation_capture.data
            archived_sources["generation_manifest"] = {
                "path": str(history_generation_path),
                "sha256": generation_capture.sha256,
            }
            archived_generation_bundle = dict(archived_manifest["generation_bundle"])
            archived_generation_bundle["manifest"] = archived_sources[
                "generation_manifest"
            ]
            archived_manifest["generation_bundle"] = archived_generation_bundle
        archived_manifest["archived_sources"] = archived_sources

    archived_manifest_bytes = json.dumps(
        _jsonable(archived_manifest), ensure_ascii=False, indent=2
    ).encode("utf-8")
    files_to_archive[history_manifest_path] = archived_manifest_bytes

    for target_path, expected_bytes in files_to_archive.items():
        if target_path.exists() and target_path.read_bytes() != expected_bytes:
            raise ValueError(f"历史目标已存在且内容冲突，拒绝覆盖：{target_path}")

    for target_path, expected_bytes in files_to_archive.items():
        if not target_path.exists():
            _write_bytes_atomic(target_path, expected_bytes)
    for target_path, expected_bytes in files_to_archive.items():
        if file_sha256(target_path) != sha256(expected_bytes).hexdigest():
            raise RuntimeError(f"历史归档写入后校验失败：{target_path}")
    return True


def _staged_record(staged_path: Path, final_path: Path) -> dict[str, str]:
    """为暂存文件生成指向最终发布位置的来源记录。"""

    return {"path": str(final_path.resolve()), "sha256": file_sha256(staged_path)}


def _verify_staged_localization(
    staging_dir: Path, final_dir: Path, manifest: dict[str, Any]
) -> None:
    """发布前逐项核验暂存目录和最终清单。"""

    manifest_path = staging_dir / "localization_manifest.json"
    manifest_capture = capture_file(manifest_path)
    try:
        parsed_manifest = json.loads(manifest_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("暂存定位清单不是有效 JSON") from error
    if parsed_manifest != _jsonable(manifest):
        raise RuntimeError("暂存定位清单内容与待发布记录不一致")

    filenames = _manifest_artifact_filenames(manifest)
    expected_names = {
        *filenames.values(),
        "localization_config.json",
        "localization_manifest.json",
    }
    actual_names = {path.name for path in staging_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise RuntimeError(
            f"暂存定位文件集合不完整：期望={sorted(expected_names)}，"
            f"实际={sorted(actual_names)}"
        )
    for filename in expected_names:
        if (staging_dir / filename).stat().st_size <= 0:
            raise RuntimeError(f"暂存定位文件为空：{filename}")
    for filename in expected_names:
        if not filename.endswith(".json"):
            continue
        try:
            parsed = json.loads((staging_dir / filename).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"暂存 JSON 无法解析：{filename}") from error
        if filename == "localization_result.json" and parsed.get(
            "localization_run_id"
        ) != manifest["run_id"]:
            raise RuntimeError("暂存定位结果与定位清单的 run_id 不一致")
    for filename in (name for name in expected_names if name.endswith(".npz")):
        try:
            with np.load(staging_dir / filename, allow_pickle=False) as staged_npz:
                if not staged_npz.files:
                    raise RuntimeError(f"暂存 NPZ 没有数组：{filename}")
        except (OSError, ValueError) as error:
            raise RuntimeError(f"暂存 NPZ 无法读取：{filename}") from error
    for artifact_name, filename in filenames.items():
        staged_path = staging_dir / filename
        record = manifest["artifacts"][artifact_name]
        if Path(record["path"]).resolve() != (final_dir / filename).resolve():
            raise RuntimeError(f"暂存清单中的 {artifact_name} 最终路径不正确")
        if file_sha256(staged_path) != record["sha256"]:
            raise RuntimeError(f"暂存定位产物 {artifact_name} 哈希校验失败")
    config_record = manifest["config_snapshot"]
    if Path(config_record["path"]).resolve() != (
        final_dir / "localization_config.json"
    ).resolve():
        raise RuntimeError("暂存清单中的定位配置最终路径不正确")
    if file_sha256(staging_dir / "localization_config.json") != config_record[
        "file_sha256"
    ]:
        raise RuntimeError("暂存定位配置快照哈希校验失败")


def _publish_staged_localization(
    *,
    root: Path,
    staging_dir: Path,
    output_dir: Path,
    remove_fixed_metrics: bool,
    run_receipt_path: Path | None,
    run_receipt: dict[str, Any] | None,
) -> None:
    """目录级切换新定位运行；任一步失败都恢复旧活动目录和指标。"""

    transaction_id = uuid4().hex
    old_directory_backup = root / f".localization-backup-{transaction_id}"
    metrics_path = root / "evaluation" / "metrics.json"
    metrics_backup = root / f".evaluation-metrics-backup-{transaction_id}.json"
    old_directory_moved = False
    metrics_moved = False
    new_directory_published = False
    receipt_write_started = False
    receipt_created = False
    try:
        if run_receipt_path is not None and run_receipt_path.exists():
            raise FileExistsError(f"运行回执已存在，拒绝覆盖：{run_receipt_path}")
        if output_dir.exists():
            os.replace(output_dir, old_directory_backup)
            old_directory_moved = True
        if remove_fixed_metrics:
            if not metrics_path.is_file():
                raise RuntimeError("提交新定位运行前，旧固定评估指标意外消失")
            os.replace(metrics_path, metrics_backup)
            metrics_moved = True
        os.replace(staging_dir, output_dir)
        new_directory_published = True
        if run_receipt_path is not None:
            if run_receipt is None:
                raise RuntimeError("内部错误：缺少待发布的运行回执内容")
            receipt_write_started = True
            _write_new_json_atomic(run_receipt_path, run_receipt)
            receipt_created = True
    except BaseException as publish_error:
        rollback_errors: list[str] = []
        if (
            receipt_write_started
            and run_receipt_path is not None
            and run_receipt is not None
            and run_receipt_path.exists()
        ):
            try:
                if run_receipt_path.read_bytes() == _encoded_json(run_receipt):
                    run_receipt_path.unlink()
                elif receipt_created:
                    rollback_errors.append("本次运行回执发布后内容又发生变化，拒绝删除")
            except OSError as error:
                rollback_errors.append(f"撤回本次运行回执失败：{error}")
        if new_directory_published:
            try:
                os.replace(output_dir, staging_dir)
                new_directory_published = False
            except OSError as error:
                rollback_errors.append(f"撤回新定位目录失败：{error}")
        if metrics_moved:
            try:
                os.replace(metrics_backup, metrics_path)
            except OSError as error:
                rollback_errors.append(f"恢复固定评估指标失败：{error}")
        if old_directory_moved:
            try:
                os.replace(old_directory_backup, output_dir)
            except OSError as error:
                rollback_errors.append(f"恢复旧定位目录失败：{error}")
        if rollback_errors:
            raise RuntimeError(
                "新定位发布失败，且回滚不完整：" + "；".join(rollback_errors)
            ) from publish_error
        raise
    else:
        if old_directory_backup.exists():
            try:
                shutil.rmtree(old_directory_backup)
            except OSError:
                # 新目录已经完整发布；保留唯一命名的备份比误报发布失败更安全。
                pass
        if metrics_backup.exists():
            try:
                metrics_backup.unlink()
            except OSError:
                pass


def resolve_output_root(config: dict[str, Any]) -> Path:
    configured = Path(str(config["output"]["root"])).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    config_path = Path(str(config.get("_config_path", Path.cwd()))).resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    return (project_root / configured).resolve()


def _reject_existing_output_targets(
    targets: Sequence[Path], *, allow_overwrite: bool, label: str
) -> None:
    """一次检查整组正式目标，断链符号链接也视为已占用。"""

    if not isinstance(allow_overwrite, bool):
        raise ValueError("allow_overwrite 必须是布尔值")
    conflicts = [path for path in targets if os.path.lexists(path)]
    if conflicts and not allow_overwrite:
        joined = "、".join(str(path) for path in conflicts)
        raise FileExistsError(f"{label}目标已存在，拒绝覆盖整组文件：{joined}")


def prepare_scene(
    config: dict[str, Any], output_root: str | Path | None = None
) -> dict[str, str]:
    """生成或预处理二维场景。当前离线闭环使用矩形场景。"""

    root = Path(output_root).resolve() if output_root else resolve_output_root(config)
    with exclusive_output_root_lock(root):
        return _prepare_scene_locked(config, root, allow_overwrite=False)


def _prepare_scene_locked(
    config: dict[str, Any], root: Path, *, allow_overwrite: bool
) -> dict[str, str]:
    """在输出根目录排他锁内生成场景。"""

    scene_dir = root / "scene"
    _reject_existing_output_targets(
        (
            scene_dir / "scene_2d.json",
            scene_dir / "scene_bev.png",
            scene_dir / "scene_occupancy.npy",
            scene_dir / "preprocess_manifest.json",
        ),
        allow_overwrite=allow_overwrite,
        label="场景流水线",
    )
    scene_config = config["scene"]
    source = str(scene_config["source"])
    if source != "synthetic_room":
        raise ValueError(
            "此入口只处理 synthetic_room；DeepMIMO/Sionna 场景请使用 prepare-sionna-scene"
        )
    scene = make_synthetic_room(
        scene_config["bounds_m"],
        fixed_height_m=float(scene_config["fixed_height_m"]),
        bev_resolution_m=float(scene_config["bev_resolution_m"]),
    )
    artifacts = scene.save(scene_dir, allow_overwrite=allow_overwrite)
    _write_json(
        scene_dir / "preprocess_manifest.json",
        {
            "stage": "scene_preprocess",
            "source": source,
            "height_policy": "BS 和 UE 使用同一固定高度；二维定位不读取高度",
            "wall_count": len(scene.walls),
            "artifacts": artifacts,
        },
    )
    return artifacts


def generate_data(
    config: dict[str, Any],
    *,
    scene_json: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, str]:
    """生成离线 CSI，并将 online 与 truth 物理分目录保存。"""

    root = Path(output_root).resolve() if output_root else resolve_output_root(config)
    with exclusive_output_root_lock(root):
        return _generate_data_locked(
            config,
            scene_json=scene_json,
            root=root,
            allow_overwrite=False,
        )


def _generate_data_locked(
    config: dict[str, Any],
    *,
    scene_json: str | Path | None,
    root: Path,
    allow_overwrite: bool,
) -> dict[str, str]:
    """在输出根目录排他锁内生成并发布一整组离线数据。"""

    data_root = root / "data"
    generation_manifest_path = data_root / "generation_manifest.json"
    fixed_targets = (
        data_root / "online" / "measurement.npz",
        data_root / "online" / "manifest.json",
        data_root / "truth" / "ground_truth.npz",
        data_root / "truth" / "ground_truth.json",
        generation_manifest_path,
    )
    _reject_existing_output_targets(
        fixed_targets,
        allow_overwrite=allow_overwrite,
        label="生成数据流水线",
    )
    scene_path = (
        Path(scene_json).expanduser().resolve()
        if scene_json is not None
        else (root / "scene" / "scene_2d.json").resolve()
    )
    if scene_path in {path.resolve() for path in fixed_targets}:
        raise ValueError("scene_json 不能指向将由生成数据流水线发布的固定目标")
    scene_capture = capture_file(scene_path)
    try:
        scene_data = json.loads(scene_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"二维场景不是有效 JSON：{scene_path}") from error
    if not isinstance(scene_data, dict):
        raise ValueError("二维场景顶层必须是键值映射")
    scene = Scene2D.from_dict(scene_data)
    online, truth = generate_synthetic_measurement(scene, config)
    artifacts = save_measurement_bundle(
        data_root, online, truth, allow_overwrite=allow_overwrite
    )
    generation_manifest = {
        "schema_version": 2,
        "stage": "synthetic_csi_generation",
        "scene_json": str(scene_path),
        "online_input": artifacts["online_npz"],
        "truth_input": artifacts["truth_npz"],
        "separation_rule": "localization 只允许读取 online 目录",
        "link_direction": "uplink_ue_to_bs",
        "absolute_delay_normalization": False,
        "delay_convention": (
            "observed_delay=geometric_delay+common_bias+noise"
        ),
        "path_selection": truth["path_selection"],
        "rt_model": {
            "los": True,
            "specular_reflection": True,
            "max_reflections": int(config["scene"]["max_reflections"]),
            "diffraction": False,
            "diffuse_reflection": False,
            "transmission": False,
            "front_facing_only": bool(
                config["radio"].get("front_facing_only", True)
            ),
        },
        "artifact_hashes": {
            "scene_json": scene_capture.artifact_record(),
            "online_measurement": artifact_record(artifacts["online_npz"]),
            "ground_truth": artifact_record(artifacts["truth_npz"]),
        },
    }
    generation_manifest["bundle_id"] = generation_bundle_id(generation_manifest)
    _write_json(generation_manifest_path, generation_manifest)
    return {
        **artifacts,
        "generation_manifest": str(generation_manifest_path.resolve()),
        "generation_bundle_id": str(generation_manifest["bundle_id"]),
    }


def estimate_noise_std_from_observed_csi(csi: np.ndarray, num_sources: int) -> float:
    """按 MUSIC 信号子空间阶数做低秩残差，估计复噪声均方根。"""

    array = np.asarray(csi, dtype=np.complex128)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError("CSI 必须为 (M,K) 或 (S,M,K)")
    estimates: list[float] = []
    for snapshot in array:
        singular_values = np.linalg.svd(snapshot, compute_uv=False)
        rank = min(int(num_sources), singular_values.size - 1)
        residual_energy = float(np.sum(singular_values[rank:] ** 2))
        degrees = max(1, (snapshot.shape[0] - rank) * (snapshot.shape[1] - rank))
        estimates.append(math.sqrt(max(0.0, residual_energy / degrees)))
    estimate = float(np.median(estimates))
    if estimate <= 0.0:
        # 只在完全无噪的数值测试中触发，仍由观测 CSI 的尺度决定。
        estimate = float(np.sqrt(np.mean(np.abs(array) ** 2))) * 1e-8
    return estimate


def _make_grids(music_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    def bounded_grid(lower: float, upper: float, step: float) -> np.ndarray:
        """保留主步长，必要时以最后一个短格精确覆盖上界，绝不越界。"""
        if not all(np.isfinite(value) for value in (lower, upper, step)):
            raise ValueError("MUSIC 网格上下界和步长必须为有限数")
        if lower >= upper or step <= 0:
            raise ValueError("MUSIC 网格要求下界小于上界且步长为正数")
        grid = np.arange(lower, upper, step)
        grid = grid[(grid >= lower) & (grid < upper)]
        if grid.size == 0:
            grid = np.asarray([lower])
        # 整除步长时，浮点误差可能留下一个紧挨上界的重复末格。
        tolerance = 8 * np.finfo(float).eps * max(abs(lower), abs(upper), abs(step))
        if grid.size > 1 and upper - grid[-1] <= tolerance:
            grid[-1] = upper
        else:
            grid = np.r_[grid, upper]
        if np.any(np.diff(grid) <= 0):
            raise ValueError("MUSIC 网格步长过小，无法在浮点精度内形成不同坐标")
        return grid

    angle_deg = bounded_grid(
        float(music_config["angle_min_deg"]),
        float(music_config["angle_max_deg"]),
        float(music_config["angle_step_deg"]),
    )
    delays = bounded_grid(
        float(music_config["delay_min_s"]),
        float(music_config["delay_max_s"]),
        float(music_config["delay_step_s"]),
    )
    return np.deg2rad(angle_deg), delays


def _validate_unambiguous_delay_window(
    music_config: dict[str, Any],
    subcarrier_frequencies_hz: np.ndarray,
) -> float:
    """检查时延搜索窗没有跨过等间隔子载波的周期重复范围。

    对间隔为 ``delta_f`` 的频率采样，``tau`` 与
    ``tau + 1/delta_f`` 产生完全相同的频率相位。若搜索窗跨过这个周期，
    MUSIC 会把同一路径的周期副本当成额外路径，后续几何无法补救。
    """

    frequencies = np.asarray(subcarrier_frequencies_hz, dtype=float).reshape(-1)
    if frequencies.size < 2 or np.any(~np.isfinite(frequencies)):
        raise ValueError("至少需要两个有限子载波频率")
    differences = np.diff(frequencies)
    if np.any(differences <= 0.0):
        raise ValueError("子载波频率必须严格递增")
    spacing_hz = float(np.median(differences))
    if not np.allclose(differences, spacing_hz, rtol=1e-9, atol=max(1e-9, spacing_hz * 1e-9)):
        raise ValueError("二维 MUSIC 的子载波必须等间隔")
    unambiguous_period_s = 1.0 / spacing_hz
    search_span_s = float(music_config["delay_max_s"]) - float(
        music_config["delay_min_s"]
    )
    if search_span_s >= unambiguous_period_s * (1.0 - 1e-9):
        raise ValueError(
            "时延搜索范围跨过子载波的无混叠周期："
            f"搜索宽度={search_span_s * 1e9:.3f} ns，"
            f"无混叠周期={unambiguous_period_s * 1e9:.3f} ns；"
            "请增加子载波数、减小带宽或缩小时延搜索范围"
        )
    return unambiguous_period_s


def _separation_bins(music_config: dict[str, Any]) -> tuple[int, int]:
    angle_bins = int(
        math.ceil(
            float(music_config["min_angle_separation_deg"])
            / float(music_config["angle_step_deg"])
        )
    )
    delay_bins = int(
        math.ceil(
            float(music_config["min_delay_separation_s"])
            / float(music_config["delay_step_s"])
        )
    )
    return angle_bins, delay_bins


def _peak_distance(
    nominal: MusicPeak2D,
    aoa_rad: float,
    delay_s: float,
    *,
    angle_scale_rad: float,
    delay_scale_s: float,
) -> float:
    angle_delta = abs(float(aoa_rad) - nominal.aoa_rad)
    delay_delta = abs(float(delay_s) - nominal.delay_s)
    return (angle_delta / angle_scale_rad) ** 2 + (delay_delta / delay_scale_s) ** 2


def associate_perturbed_peaks(
    nominal_peaks: Sequence[MusicPeak2D],
    samples: MusicPeakSamples,
    *,
    angle_scale_rad: float,
    delay_scale_s: float,
    max_normalized_distance: float = 3.0,
    false_peak_penalty: float = 4.0,
    missed_peak_penalty: float = 9.0,
) -> list[PeakSetAssociation]:
    """把每次扰动的无序峰集合整体配到固定观测编号。

    匹配允许某个扰动峰不属于任何 nominal 路径，也允许某条 nominal 路径在
    本次扰动中没有检出。后者的惩罚更大，符合“虚假峰可作为离群点处理，漏检
    会直接丢失信息”的第一版约定。每次扰动使用全局最小代价，而不是按输入
    顺序逐峰贪心匹配。
    """

    if angle_scale_rad <= 0.0 or delay_scale_s <= 0.0:
        raise ValueError("角度和时延匹配尺度必须为正数")
    if max_normalized_distance <= 0.0:
        raise ValueError("max_normalized_distance 必须为正数")
    if false_peak_penalty < 0.0:
        raise ValueError("false_peak_penalty 不能为负数")
    if missed_peak_penalty <= false_peak_penalty:
        raise ValueError("missed_peak_penalty 必须大于 false_peak_penalty")

    associations: list[PeakSetAssociation] = []
    nominal_count = len(nominal_peaks)
    maximum_squared_distance = float(max_normalized_distance) ** 2
    for repetition in range(samples.aoa_rad.shape[0]):
        valid = tuple(
            index
            for index in range(samples.aoa_rad.shape[1])
            if np.isfinite(samples.aoa_rad[repetition, index])
            and np.isfinite(samples.delay_s[repetition, index])
        )

        distances = {
            (sample_index, nominal_index): _peak_distance(
                nominal_peaks[nominal_index],
                samples.aoa_rad[repetition, sample_index],
                samples.delay_s[repetition, sample_index],
                angle_scale_rad=angle_scale_rad,
                delay_scale_s=delay_scale_s,
            )
            for sample_index in valid
            for nominal_index in range(nominal_count)
        }

        @lru_cache(maxsize=None)
        def search(
            valid_position: int, used_nominal_mask: int
        ) -> tuple[float, tuple[tuple[int, int], ...]]:
            if valid_position == len(valid):
                missed_count = nominal_count - used_nominal_mask.bit_count()
                return float(missed_count * missed_peak_penalty), ()

            sample_index = valid[valid_position]
            tail_cost, tail_pairs = search(valid_position + 1, used_nominal_mask)
            options: list[tuple[float, tuple[tuple[int, int], ...]]] = [
                (float(false_peak_penalty + tail_cost), tail_pairs)
            ]
            for nominal_index in range(nominal_count):
                bit = 1 << nominal_index
                if used_nominal_mask & bit:
                    continue
                distance = float(distances[(sample_index, nominal_index)])
                if distance > maximum_squared_distance:
                    continue
                remaining_cost, remaining_pairs = search(
                    valid_position + 1, used_nominal_mask | bit
                )
                options.append(
                    (
                        distance + remaining_cost,
                        ((nominal_index, sample_index),) + remaining_pairs,
                    )
                )
            # 同代价时优先保留更多匹配，再按编号固定结果，保证可复现。
            return min(options, key=lambda item: (item[0], -len(item[1]), item[1]))

        total_cost, matched_pairs = search(0, 0)
        assignment = {
            nominal_index: (
                float(samples.aoa_rad[repetition, sample_index]),
                float(samples.delay_s[repetition, sample_index]),
            )
            for nominal_index, sample_index in matched_pairs
        }
        matched_sample_indices = {sample_index for _, sample_index in matched_pairs}
        matched_nominal_indices = {nominal_index for nominal_index, _ in matched_pairs}
        associations.append(
            PeakSetAssociation(
                matches=assignment,
                nominal_to_sample_index=dict(matched_pairs),
                total_cost=float(total_cost),
                unmatched_sample_indices=tuple(
                    index for index in valid if index not in matched_sample_indices
                ),
                missed_nominal_indices=tuple(
                    index
                    for index in range(nominal_count)
                    if index not in matched_nominal_indices
                ),
                valid_sample_indices=valid,
            )
        )
    return associations


def _build_observation_samples(
    nominal_peaks: Sequence[MusicPeak2D],
    associations: Sequence[PeakSetAssociation],
    *,
    bs_boresight_rad: float,
) -> tuple[list[PathObservationSample], list[list[PathObservationSample]]]:
    nominal_samples: list[PathObservationSample] = []
    per_repetition: list[list[PathObservationSample]] = [[] for _ in associations]
    for observation_index, peak in enumerate(nominal_peaks):
        observation_id = f"music_path_{observation_index:02d}"
        nominal_samples.append(
            PathObservationSample(
                observation_id=observation_id,
                sample_id=f"{observation_id}:nominal",
                aoa_global_rad=local_to_global_aoa(peak.aoa_rad, bs_boresight_rad),
                delay_s=peak.delay_s,
            )
        )
        for repetition_index, association in enumerate(associations):
            if observation_index not in association.matches:
                continue
            local_angle, delay = association.matches[observation_index]
            sample = PathObservationSample(
                observation_id=observation_id,
                sample_id=f"{observation_id}:repeat_{repetition_index:03d}",
                aoa_global_rad=local_to_global_aoa(local_angle, bs_boresight_rad),
                delay_s=delay,
            )
            per_repetition[repetition_index].append(sample)
    return nominal_samples, per_repetition


def _solver_config(localization_config: dict[str, Any]) -> SolverConfig:
    return SolverConfig(
        huber_delta=float(localization_config["huber_delta_m"]),
        max_iterations=int(localization_config["max_iterations"]),
        max_seed_pairs=int(localization_config.get("max_seed_pairs", 100000)),
    )


def _raw_candidate_dict(candidate: RawReverseTrajectory) -> dict[str, Any]:
    return asdict(candidate)


def _clustered_candidate_dict(candidate: CandidateTrajectory) -> dict[str, Any]:
    return {
        "observation_id": candidate.observation_id,
        "candidate_id": candidate.candidate_id,
        "anchor_m": candidate.anchor_m,
        "direction": candidate.direction,
        "beta_interval_m": [candidate.beta_min_m, candidate.beta_max_m],
        "weight": candidate.weight,
        "metadata": candidate.metadata,
    }


def _bootstrap_joint_solutions(
    scene: Scene2D,
    measurement: OnlineMeasurement,
    per_repetition: Sequence[Sequence[PathObservationSample]],
    associations: Sequence[PeakSetAssociation],
    *,
    beta_interval_m: tuple[float, float],
    max_reflections: int,
    solver_config: SolverConfig,
    position_radius_m: float,
    direction_radius_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if len(per_repetition) != len(associations):
        raise ValueError("扰动观测与峰集合匹配结果数量不一致")
    positions: list[np.ndarray] = []
    betas: list[float] = []
    conditional_covariances: list[np.ndarray] = []
    solution_weights: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for repetition_index, (samples, association) in enumerate(
        zip(per_repetition, associations, strict=True)
    ):
        association_diagnostics = {
            "observation_samples": [asdict(sample) for sample in samples],
            "association_cost": association.total_cost,
            "distribution_weight": association.distribution_weight,
            "matched_peak_count": len(association.matches),
            "missed_nominal_indices": association.missed_nominal_indices,
            "unmatched_sample_indices": association.unmatched_sample_indices,
        }
        # 漏掉 nominal 路径意味着这次扰动少了一条原本用于约束位置/偏差的
        # 独立信息。不能再把这个不完整解与完整解一起平均，否则少数错误的
        # 两路径解可能把最终均值拉走。这里把它记为失败；失败比例会在
        # _distribution_statistics 中放大协方差。额外虚假峰仍允许保留为
        # unmatched 诊断，不会触发本规则。
        if association.missed_nominal_indices:
            diagnostics.append(
                {
                    "repetition": repetition_index,
                    "solved": False,
                    "reason": "扰动中漏掉 nominal 路径",
                    **association_diagnostics,
                }
            )
            continue
        if len({sample.observation_id for sample in samples}) < 2:
            diagnostics.append(
                {
                    "repetition": repetition_index,
                    "solved": False,
                    "reason": "少于两条观测",
                    **association_diagnostics,
                }
            )
            continue
        raw = generate_reverse_candidates(
            scene,
            measurement.bs_position_m,
            samples,
            max_reflections=max_reflections,
            beta_interval_m=beta_interval_m,
        )
        clustered = cluster_reverse_candidates(
            raw,
            position_radius_m=position_radius_m,
            direction_radius_deg=direction_radius_deg,
        )
        association_diagnostics.update({
            "raw_reverse_candidates": [_raw_candidate_dict(item) for item in raw],
            "clustered_candidates": [_clustered_candidate_dict(item) for item in clustered],
        })
        try:
            result = solve_position_and_bias(clustered, solver_config)
        except SolverError as error:
            diagnostics.append(
                {
                    "repetition": repetition_index,
                    "solved": False,
                    "reason": str(error),
                    **association_diagnostics,
                }
            )
            continue
        positions.append(np.asarray(result.mu))
        betas.append(float(result.beta))
        conditional_covariances.append(np.asarray(result.sigma))
        solution_weights.append(association.distribution_weight)
        diagnostics.append(
            {
                "repetition": repetition_index,
                "solved": True,
                **association_diagnostics,
                "position_m": result.mu,
                "beta_m": result.beta,
                "sigma_m2": result.sigma,
                "selected_candidates": {
                    str(key): _clustered_candidate_dict(value)
                    for key, value in result.selected_candidates.items()
                },
                "residuals_m": result.residuals,
                "solver_diagnostics": asdict(result.diagnostics),
                "selected_topologies": {
                    str(observation_id): candidate.metadata.get("topology_id")
                    for observation_id, candidate in result.selected_candidates.items()
                },
            }
        )
    position_array = np.asarray(positions, dtype=float).reshape((-1, 2))
    beta_array = np.asarray(betas, dtype=float)
    covariance_array = np.asarray(conditional_covariances, dtype=float).reshape(
        (-1, 2, 2)
    )
    weight_array = np.asarray(solution_weights, dtype=float)
    return position_array, beta_array, covariance_array, weight_array, diagnostics


def _make_positive_semidefinite(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _distribution_statistics(
    central_mu: np.ndarray,
    central_sigma: np.ndarray,
    central_beta_m: float,
    sampled_positions: np.ndarray,
    sampled_betas_m: np.ndarray,
    sampled_sigmas_m2: np.ndarray,
    sample_weights: np.ndarray,
    requested_count: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """从扰动解形成高斯，并对未求解重复实验做保守膨胀。"""

    positions = np.asarray(sampled_positions, dtype=float).reshape((-1, 2))
    betas = np.asarray(sampled_betas_m, dtype=float).reshape((-1,))
    conditional_sigmas = np.asarray(sampled_sigmas_m2, dtype=float).reshape(
        (-1, 2, 2)
    )
    weights = np.asarray(sample_weights, dtype=float).reshape((-1,))
    if not (
        positions.shape[0]
        == betas.shape[0]
        == conditional_sigmas.shape[0]
        == weights.shape[0]
    ):
        raise ValueError("位置、偏差、条件协方差和权重的扰动解数量必须相同")
    if requested_count < positions.shape[0] or requested_count < 1:
        raise ValueError("requested_count 必须为正且不小于成功扰动解数量")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("扰动解权重必须是有限正数")

    success_count = positions.shape[0]
    failure_inflation = float(requested_count / max(1, success_count))
    if success_count < 2:
        source = (
            "nominal_fallback_no_solved_perturbation"
            if success_count == 0
            else "nominal_fallback_single_solved_perturbation"
        )
        return (
            np.asarray(central_mu, dtype=float),
            _make_positive_semidefinite(
                np.asarray(central_sigma, dtype=float) * failure_inflation
            ),
            float(central_beta_m),
            source,
        )

    weight_sum = float(np.sum(weights))
    effective_denominator = weight_sum - float(np.sum(weights * weights)) / weight_sum
    if effective_denominator <= np.finfo(float).eps:
        return (
            np.asarray(central_mu, dtype=float),
            _make_positive_semidefinite(
                np.asarray(central_sigma, dtype=float) * failure_inflation
            ),
            float(central_beta_m),
            "nominal_fallback_degenerate_perturbation_weights",
        )

    normalized_weights = weights / weight_sum
    mu = np.sum(normalized_weights[:, None] * positions, axis=0)
    beta_m = float(np.sum(normalized_weights * betas))
    centered = positions - mu
    between_solution_sigma = np.einsum(
        "n,ni,nj->ij", weights, centered, centered
    ) / effective_denominator
    mean_conditional_sigma = np.einsum(
        "n,nij->ij", normalized_weights, conditional_sigmas
    )
    # 全方差公式：每个扰动解内部的不确定度均值 + 扰动解均值之间的协方差。
    sigma = mean_conditional_sigma + between_solution_sigma
    sigma *= failure_inflation
    return (
        mu,
        _make_positive_semidefinite(sigma),
        beta_m,
        "total_variance_of_quality_weighted_perturbation_solutions",
    )


def _selection_frequencies(
    bootstrap_diagnostics: Sequence[dict[str, Any]],
    *,
    nominal_count: int,
) -> dict[str, Any]:
    solved = [item for item in bootstrap_diagnostics if item.get("solved")]
    counts: dict[str, dict[str, int]] = {}
    for item in solved:
        for observation_id, topology_id in item.get("selected_topologies", {}).items():
            topology_counts = counts.setdefault(str(observation_id), {})
            key = str(topology_id)
            topology_counts[key] = topology_counts.get(key, 0) + 1
    all_count = len(bootstrap_diagnostics)
    solved_count = len(solved)
    over_all = {
        observation_id: {
            topology_id: count / max(1, all_count)
            for topology_id, count in sorted(topology_counts.items())
        }
        for observation_id, topology_counts in sorted(counts.items())
    }
    conditioned = {
        observation_id: {
            topology_id: count / max(1, solved_count)
            for topology_id, count in sorted(topology_counts.items())
        }
        for observation_id, topology_counts in sorted(counts.items())
    }
    miss_counts = {index: 0 for index in range(nominal_count)}
    for item in bootstrap_diagnostics:
        for index in item.get("missed_nominal_indices", ()):
            miss_counts[int(index)] += 1
    miss_rate = {
        f"music_path_{index:02d}": count / max(1, all_count)
        for index, count in miss_counts.items()
    }
    return {
        "frequency_over_all_repetitions": over_all,
        "frequency_conditioned_on_solved": conditioned,
        "observation_miss_rate": miss_rate,
    }


def _resolve_generation_manifest(
    output_root: Path, generation_manifest: str | Path | None
) -> Path:
    """解析显式清单，或在受控的两个标准位置中唯一发现它。"""

    if generation_manifest is not None:
        return Path(generation_manifest).expanduser().resolve()
    candidates = (
        output_root / "generation_manifest.json",
        output_root / "data" / "generation_manifest.json",
    )
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            "找不到生成清单；请通过 generation_manifest 显式传入，"
            "或将其放在输出目录的标准位置"
        )
    if len(existing) > 1:
        raise ValueError(
            "输出目录中发现多份生成清单，无法可靠判断数据批次；"
            "请通过 generation_manifest 显式指定"
        )
    return existing[0]


def localize(
    config: dict[str, Any],
    *,
    scene_json: str | Path,
    online_input: str | Path,
    generation_manifest: str | Path | None = None,
    output_root: str | Path | None = None,
    run_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """只从二维场景和 online CSI 联合估计位置与公共时延偏差。"""

    validate_localization_config(config)
    root = Path(output_root).resolve() if output_root else resolve_output_root(config)
    with exclusive_output_root_lock(root):
        return _localize_locked(
            config,
            scene_json=scene_json,
            online_input=online_input,
            generation_manifest=generation_manifest,
            output_root=root,
            run_receipt=run_receipt,
        )


def _localize_locked(config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """运行单份观测；异常时保存已完成步骤，不覆盖上一轮成功产物。"""
    progress: dict[str, Any] = {
        "workflow": WORKFLOW, "run_id": str(uuid4()),
        "completed_steps": [], "failed_step": "01_csi_input", "payloads": {},
    }
    try:
        return _localize_locked_impl(config, progress=progress, **kwargs)
    except Exception as error:
        if "inputs" in progress:
            try:
                directory = Path(kwargs["output_root"]) / "localization_failures" / progress["run_id"]
                directory.mkdir(parents=True, exist_ok=False)
                records = {}
                for key, payload in progress["payloads"].items():
                    path = directory / _LOCALIZATION_ARTIFACT_FILENAMES[key]
                    if path.suffix == ".npz":
                        np.savez_compressed(path, **payload)
                    else:
                        _write_json(path, payload)
                    records[key] = artifact_record(path)
                snapshot_path = directory / "localization_config.json"
                snapshot = progress["config_snapshot_data"]
                _write_json(snapshot_path, snapshot)
                metadata = {
                    key: value for key, value in progress.items()
                    if key not in {"payloads", "config_snapshot_data"}
                }
                metadata.update(
                    status="failed", error=f"{type(error).__name__}: {error}",
                    artifacts=records,
                    config_snapshot={
                        "path": str(snapshot_path.resolve()),
                        "file_sha256": file_sha256(snapshot_path),
                        "canonical_sha256": snapshot["canonical_sha256"],
                        "source_config_path": snapshot["source_config_path"],
                    },
                )
                progress_path = directory / "progress.json"
                _write_json(progress_path, metadata)
                error.failure_progress = str(progress_path.resolve())
            except Exception as save_error:
                error.add_note(f"保存失败步骤时另遇到错误：{save_error}")
        raise


def _localize_locked_impl(
    config: dict[str, Any],
    *,
    scene_json: str | Path,
    online_input: str | Path,
    generation_manifest: str | Path | None,
    output_root: Path,
    archive_previous: bool = True,
    previous_evaluation_archived: bool = False,
    run_receipt: str | Path | None = None,
    progress: dict[str, Any],
) -> dict[str, Any]:
    """在输出根目录排他锁已持有时执行完整定位。"""

    stage_timings: dict[str, float] = {}
    stage_started = time.perf_counter()

    def mark_stage(name: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        stage_timings[name] = now - stage_started
        stage_started = now

    root = output_root
    scene_path = Path(scene_json).expanduser().resolve()
    online_path = Path(online_input).expanduser().resolve()
    generation_manifest_path = _resolve_generation_manifest(root, generation_manifest)
    (
        generation_manifest_data,
        generation_manifest_record,
        bundle_id,
    ) = load_generation_manifest(generation_manifest_path)
    generation_stage, validated_bundle_id = validate_generation_manifest_envelope(
        generation_manifest_data
    )
    if validated_bundle_id != bundle_id:
        raise ValueError("生成清单批次编号的两次独立校验结果不一致")
    scene_capture = capture_file(scene_path)
    online_capture = capture_file(online_path)
    scene_input_record = verify_generation_artifact(
        generation_manifest_data, "scene_json", scene_capture
    )
    online_input_record = verify_generation_artifact(
        generation_manifest_data, "online_measurement", online_capture
    )
    output_dir = root / "localization"
    run_receipt_path = (
        Path(run_receipt).expanduser().resolve() if run_receipt is not None else None
    )
    if run_receipt_path is not None and (
        run_receipt_path == output_dir or output_dir in run_receipt_path.parents
    ):
        raise ValueError("运行回执不能写入将被整体切换的 localization 目录")
    localization_run_id = progress["run_id"]
    config_snapshot = localization_config_snapshot(config)
    try:
        scene_data = json.loads(scene_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"二维场景不是有效 JSON：{scene_path}") from error
    if not isinstance(scene_data, dict):
        raise ValueError("二维场景顶层必须是键值映射")
    scene = Scene2D.from_dict(scene_data)
    measurement = load_online_measurement_bytes(
        online_capture.data, source_path=online_capture.path
    )
    validate_localization_input_contract(
        generation_manifest_data,
        generation_stage,
        config,
        scene,
        measurement,
    )
    progress.update(
        generation_bundle={"bundle_id": bundle_id, "manifest": generation_manifest_record},
        inputs={"scene": scene_input_record, "online_measurement": online_input_record},
        config_snapshot_data=config_snapshot,
        truth_was_loaded=False,
    )
    progress["completed_steps"].append("01_csi_input")
    music_config = config["music"]
    localization_config = config["localization"]
    compute_config = config.get("compute", {})
    computer = get_music_computer(
        compute_config.get("backend", "numpy"), int(compute_config.get("device_id", 0)),
        int(compute_config.get("batch_size", 4)), int(compute_config.get("angle_chunk_size", 32)),
    )
    compute_before = computer.metadata()
    mark_stage("input_validation_and_device_setup")
    progress["failed_step"] = "02_music"
    unambiguous_delay_period_s = _validate_unambiguous_delay_window(
        music_config, measurement.subcarrier_frequencies_hz
    )
    num_paths = int(music_config["num_paths"])
    signal_subspace_rank = int(music_config.get("signal_subspace_rank", num_paths))
    aoa_grid, delay_grid = _make_grids(music_config)
    prepared = computer.prepare(
        measurement.csi_observed,
        subcarrier_frequencies_hz=measurement.subcarrier_frequencies_hz,
        carrier_frequency_hz=measurement.carrier_frequency_hz,
        antenna_spacing_m=measurement.antenna_spacing_m,
        num_sources=signal_subspace_rank,
        spatial_subarray_size=int(music_config["spatial_subarray_size"]),
        frequency_subarray_size=int(music_config["frequency_subarray_size"]),
        diagonal_loading=float(music_config["diagonal_loading"]),
    )
    spectrum = prepared.spectrum(aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid)
    nominal_peaks = extract_local_music_peaks(
        spectrum, aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid,
        max_peaks=num_paths, minimum_relative_height=0.0,
        minimum_separation_bins=_separation_bins(music_config),
    )
    peak_output = {
        "workflow": WORKFLOW,
        "note": "谱值用于候选搜索，不是经过校准的路径概率；不对观测 CSI 额外加噪",
        "nominal": [asdict(peak) for peak in nominal_peaks],
    }
    progress["payloads"].update(
        music_spectrum=dict(spectrum=spectrum, aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid),
        music_peaks=peak_output,
    )
    progress["completed_steps"].append("02_music")
    mark_stage("music_observed")
    if len(nominal_peaks) < 2:
        raise RuntimeError(f"二维 MUSIC 只找到 {len(nominal_peaks)} 条路径，无法联合求解")

    progress["failed_step"] = "03_spectrum_sampling"
    sampled = sample_music_spectrum(
        prepared, nominal_peaks, aoa_grid_rad=aoa_grid, delay_grid_s=delay_grid,
        bs_boresight_rad=measurement.bs_boresight_rad,
        settings=music_config["spectrum_sampling"],
        seed=int(config["project"]["random_seed"]) + 2,
    )
    peak_output["observation_samples"] = [asdict(sample) for sample in sampled.samples]
    progress["payloads"]["spectrum_samples"] = {
        "workflow": WORKFLOW, "samples": sampled.records,
        "regions": sampled.regions, "diagnostics": sampled.diagnostics,
    }
    progress["completed_steps"].append("03_spectrum_sampling")
    mark_stage("spectrum_sampling")

    progress["failed_step"] = "04_reverse_candidates"
    beta_interval_m = (
        float(localization_config["bias_min_s"]) * SPEED_OF_LIGHT_M_S,
        float(localization_config["bias_max_s"]) * SPEED_OF_LIGHT_M_S,
    )
    raw_candidates = generate_reverse_candidates(
        scene, measurement.bs_position_m, sampled.samples,
        max_reflections=int(config["scene"]["max_reflections"]),
        beta_interval_m=beta_interval_m,
    )
    progress["payloads"]["raw_reverse_candidates"] = [
        _raw_candidate_dict(candidate) for candidate in raw_candidates
    ]
    progress["completed_steps"].append("04_reverse_candidates")
    mark_stage("reverse_candidates")

    progress["failed_step"] = "05_first_clustering"
    clustered_candidates = cluster_reverse_candidates(
        raw_candidates,
        position_radius_m=float(localization_config["candidate_cluster_radius_m"]),
        direction_radius_deg=float(localization_config["candidate_direction_radius_deg"]),
    )
    progress["payloads"]["clustered_candidates"] = [
        _clustered_candidate_dict(candidate) for candidate in clustered_candidates
    ]
    progress["completed_steps"].append("05_first_clustering")
    mark_stage("first_clustering")

    progress["failed_step"] = "06_joint_solution"
    central = solve_position_and_bias(clustered_candidates, _solver_config(localization_config))
    progress["completed_steps"].append("06_joint_solution")
    mark_stage("joint_solution")
    selected = {
        str(observation_id): _clustered_candidate_dict(candidate)
        for observation_id, candidate in central.selected_candidates.items()
    }
    # 此协方差来自最终几何残差近似；未把采样数当成独立观测数，
    # 也未标定谱面采样本身的不确定性。采样会通过代表选择间接影响残差。
    result = {
        "schema_version": 2, "workflow": WORKFLOW,
        "localization_run_id": localization_run_id,
        "output_type": "point_estimate_with_geometric_residual_covariance",
        "mu_m": central.mu, "sigma_m2": central.sigma,
        "distance_bias_m": central.beta,
        "clock_bias_s": central.beta / SPEED_OF_LIGHT_M_S,
        "central_solution": {
            "mu_m": central.mu, "sigma_m2": central.sigma,
            "distance_bias_m": central.beta,
            "clock_bias_s": central.beta / SPEED_OF_LIGHT_M_S,
        },
        "central_selected_candidates": selected,
        "central_residuals_m": central.residuals,
        "diagnostics": {
            **asdict(central.diagnostics),
            "stage_timings_s": stage_timings,
            "stage_timing_scope": "输入验证到正向检查；不含文件发布及独立评估",
            "music_signal_subspace_rank": signal_subspace_rank,
            "requested_music_peak_count": num_paths,
            "unambiguous_delay_period_s": unambiguous_delay_period_s,
            "nominal_music_peak_count": len(nominal_peaks),
            "spectrum_sample_count": len(sampled.samples),
            "sampling": sampled.diagnostics,
            "raw_candidate_count": len(raw_candidates),
            "clustered_candidate_count": len(clustered_candidates),
            "covariance_source": "selected_candidate_geometric_residual_approximation",
            "covariance_calibrated": False,
            "no_accept_reject_output": True,
        },
    }
    progress["payloads"]["result"] = result
    progress["failed_step"] = "07_forward_check"
    forward_check = forward_check_solution(
        scene, measurement.bs_position_m, central.selected_candidates,
        central.mu, central.beta, max_reflections=int(config["scene"]["max_reflections"]),
        observed_peaks={
            f"music_path_{index:02d}": {
                "aoa_global_rad": local_to_global_aoa(peak.aoa_rad, measurement.bs_boresight_rad),
                "delay_s": peak.delay_s,
            }
            for index, peak in enumerate(nominal_peaks)
        },
    )
    progress["payloads"]["forward_check"] = forward_check
    progress["completed_steps"].append("07_forward_check")
    mark_stage("forward_check")
    result["forward_check"] = forward_check
    compute_report = computer.metadata()
    compute_report["counter_scope"] = "worker_lifetime"
    compute_report["this_localization"] = {
        name: compute_report[name] - compute_before.get(name, 0)
        for name in ("completed_batches", "completed_csi", "steering_cache_hits", "steering_cache_misses", "eigendecomposition_count")
        if name in compute_report
    }
    result["diagnostics"]["compute"] = compute_report
    progress["failed_step"] = "artifact_publication"

    remove_fixed_metrics = (
        _archive_previous_evaluation(root, output_dir)
        if archive_previous
        else previous_evaluation_archived
    )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".localization-staging-", dir=root)
    ).resolve()
    try:
        staged_paths = {
            artifact_name: staging_dir / filename
            for artifact_name, filename in _LOCALIZATION_ARTIFACT_FILENAMES.items()
        }
        for artifact_name, payload in progress["payloads"].items():
            staged_path = staged_paths[artifact_name]
            if staged_path.suffix == ".npz":
                np.savez_compressed(staged_path, **payload)
            else:
                _write_json(staged_path, payload)
        staged_config_path = staging_dir / "localization_config.json"
        _write_json(staged_config_path, config_snapshot)
        config_snapshot_record = {
            "path": str((output_dir / "localization_config.json").resolve()),
            "source_config_path": config_snapshot["source_config_path"],
            "canonical_sha256": config_snapshot["canonical_sha256"],
            "file_sha256": file_sha256(staged_config_path),
        }
        _write_json(staged_paths["result"], result)
        artifact_records = {
            artifact_name: _staged_record(
                staged_paths[artifact_name], output_dir / filename
            )
            for artifact_name, filename in _LOCALIZATION_ARTIFACT_FILENAMES.items()
        }
        localization_manifest = {
            "schema_version": 4,
            "workflow": WORKFLOW,
            "stage": "localization",
            "run_id": localization_run_id,
            "generation_bundle": {
                "bundle_id": bundle_id,
                "manifest": generation_manifest_record,
            },
            "config_snapshot": config_snapshot_record,
            "scene_input": str(scene_path),
            "online_measurement_input": str(online_path),
            "truth_was_loaded": False,
            "truth_access": {
                "truth_file_content_loaded": False,
                "generation_manifest_truth_metadata_visible": True,
                "note": "生成清单整体可见，但定位未打开真值文件或使用真值内容",
            },
            "result": artifact_records["result"]["path"],
            "inputs": {
                "scene": scene_input_record,
                "online_measurement": online_input_record,
            },
            "artifacts": artifact_records,
            "evaluation_pending": True,
        }
        _write_json(
            staging_dir / "localization_manifest.json", localization_manifest
        )
        _verify_staged_localization(staging_dir, output_dir, localization_manifest)
        run_receipt_data = (
            {
                "run_id": localization_run_id,
                "result": artifact_records["result"],
                "manifest": {
                    "path": str(
                        (output_dir / "localization_manifest.json").resolve()
                    ),
                    "sha256": file_sha256(
                        staging_dir / "localization_manifest.json"
                    ),
                },
            }
            if run_receipt_path is not None
            else None
        )
        _publish_staged_localization(
            root=root,
            staging_dir=staging_dir,
            output_dir=output_dir,
            remove_fixed_metrics=remove_fixed_metrics,
            run_receipt_path=run_receipt_path,
            run_receipt=run_receipt_data,
        )
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
    return result


def _load_evaluation_truth(data_bytes: bytes) -> tuple[np.ndarray, float]:
    """严格读取评估所需的二维位置和标量时钟偏差。"""

    try:
        with np.load(BytesIO(data_bytes), allow_pickle=False) as truth:
            true_position = np.asarray(truth["ue_position_m"], dtype=float)
            true_bias = np.asarray(truth["clock_bias_s"], dtype=float)
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError("评估真值缺少可读取的 ue_position_m 或 clock_bias_s") from error
    if true_position.shape != (2,):
        raise ValueError(
            f"评估真值 ue_position_m 必须严格为形状 (2,)，实际为 {true_position.shape}"
        )
    if true_bias.shape != ():
        raise ValueError(
            f"评估真值 clock_bias_s 必须严格为标量形状 ()，实际为 {true_bias.shape}"
        )
    if not np.all(np.isfinite(true_position)):
        raise ValueError("评估真值 ue_position_m 必须全部为有限数")
    if not bool(np.isfinite(true_bias)):
        raise ValueError("评估真值 clock_bias_s 必须为有限数")
    return true_position, float(true_bias)


def _require_fixed_evaluation_output_path(
    *, result_path: Path, output_path: Path
) -> None:
    """评估指标只能写到定位输出根目录下的固定位置。"""

    required_output_path = (
        result_path.parent.parent / "evaluation" / "metrics.json"
    ).resolve()
    if output_path.resolve() != required_output_path:
        raise ValueError(
            "评估输出路径必须固定为 "
            f"{required_output_path}；实际收到 {output_path.resolve()}"
        )


def evaluate(
    *,
    result_json: str | Path,
    truth_npz: str | Path,
    output_json: str | Path,
    expected_run_id: str,
) -> dict[str, Any]:
    """单独读取真值进行评估，不回写定位结果。"""

    if not isinstance(expected_run_id, str) or not expected_run_id.strip():
        raise ValueError("expected_run_id 必须是非空字符串")
    result_path = Path(result_json).expanduser().resolve()
    output_path = Path(output_json).expanduser().resolve()
    _require_fixed_evaluation_output_path(
        result_path=result_path,
        output_path=output_path,
    )
    truth_path = Path(truth_npz).expanduser().resolve()
    output_root = result_path.parent.parent
    with exclusive_output_root_lock(output_root):
        return _evaluate_locked(
            result_path=result_path,
            truth_path=truth_path,
            output_path=output_path,
            expected_run_id=expected_run_id,
        )


def _evaluate_locked(
    *,
    result_path: Path,
    truth_path: Path,
    output_path: Path,
    expected_run_id: str,
) -> dict[str, Any]:
    """在与定位相同的输出根目录排他锁内执行完整评估。"""

    _require_fixed_evaluation_output_path(
        result_path=result_path,
        output_path=output_path,
    )
    history_root = (result_path.parent.parent / "evaluation" / "history").resolve()
    if output_path == history_root or history_root in output_path.parents:
        raise ValueError("评估输出路径不能写入 evaluation/history 历史归档目录")

    manifest_path = result_path.parent / "localization_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"定位结果旁缺少定位清单：{manifest_path}")
    manifest_capture = capture_file(manifest_path)
    try:
        manifest = json.loads(manifest_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"定位清单不是有效 JSON：{manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("定位清单顶层必须是键值映射")

    localization_run_id = manifest.get("run_id")
    if not isinstance(localization_run_id, str) or not localization_run_id:
        raise ValueError("定位清单缺少有效的 run_id，无法确认产物来源")
    if localization_run_id != expected_run_id:
        raise ValueError(
            "期望的定位 run_id 与当前定位清单不一致："
            f"期望={expected_run_id}，当前={localization_run_id}"
        )
    try:
        result_record = manifest["artifacts"]["result"]
        recorded_result_path = Path(result_record["path"]).expanduser().resolve()
        recorded_result_sha256 = str(result_record["sha256"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("定位清单缺少结果文件的 path+sha256 来源记录") from error
    if recorded_result_path != result_path:
        raise ValueError(
            "待评估结果路径与定位清单记录不一致："
            f"输入={result_path}，清单={recorded_result_path}"
        )
    result_capture = capture_file(result_path)
    actual_result_sha256 = result_capture.sha256
    if actual_result_sha256 != recorded_result_sha256:
        raise ValueError(
            "定位结果 SHA-256 与定位清单不一致；文件可能在定位完成后被修改"
        )
    try:
        result = json.loads(result_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"定位结果不是有效 JSON：{result_path}") from error
    if not isinstance(result, dict):
        raise ValueError("定位结果顶层必须是键值映射")
    result_run_id = result.get("localization_run_id")
    if result_run_id != expected_run_id:
        raise ValueError(
            "期望的定位 run_id 与定位结果不一致："
            f"期望={expected_run_id}，结果={result_run_id}"
        )

    try:
        generation_bundle = manifest["generation_bundle"]
        recorded_bundle_id = str(generation_bundle["bundle_id"])
        recorded_generation_manifest = generation_bundle["manifest"]
        generation_manifest_path = Path(
            recorded_generation_manifest["path"]
        ).expanduser().resolve()
        recorded_generation_manifest_sha256 = str(
            recorded_generation_manifest["sha256"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("定位清单缺少生成数据批次的 manifest path+sha256 绑定") from error
    (
        generation_manifest_data,
        actual_generation_manifest_record,
        actual_bundle_id,
    ) = load_generation_manifest(generation_manifest_path)
    if (
        actual_generation_manifest_record["sha256"]
        != recorded_generation_manifest_sha256
    ):
        raise ValueError("生成清单 SHA-256 与定位时记录不一致；清单可能已被修改")
    if actual_bundle_id != recorded_bundle_id:
        raise ValueError("生成数据 bundle_id 与定位清单不一致")

    protected_paths: dict[Path, set[str]] = {}

    def protect_path(label: str, value: Any) -> None:
        if value is None:
            return
        try:
            path = Path(value).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            return
        protected_paths.setdefault(path, set()).add(label)

    # 当前 manifest.evaluation 指向的评估文件允许被下一次评估更新；其余
    # 定位来源文件都只能读，绝不能被 output_json 覆盖。
    protect_path("定位结果", result_path)
    protect_path("评估真值", truth_path)
    protect_path("定位清单", manifest_path)
    protect_path("生成清单", generation_manifest_path)
    protect_path("定位结果", manifest.get("result"))
    protect_path("二维场景输入", manifest.get("scene_input"))
    protect_path("在线 CSI 输入", manifest.get("online_measurement_input"))

    config_snapshot = manifest.get("config_snapshot")
    if isinstance(config_snapshot, dict):
        protect_path("定位配置快照", config_snapshot.get("path"))
        protect_path("定位配置源文件", config_snapshot.get("source_config_path"))

    generation_config_snapshot = generation_manifest_data.get("config_snapshot")
    if isinstance(generation_config_snapshot, dict):
        for path_key in ("path", "source_path", "source_config_path"):
            protect_path(
                f"生成配置 {path_key}", generation_config_snapshot.get(path_key)
            )

    for section_name, section_label in (
        ("scene_artifacts", "生成场景产物"),
        ("data_artifacts", "生成数据产物"),
        ("artifact_hashes", "生成哈希产物"),
    ):
        section = generation_manifest_data.get(section_name)
        if not isinstance(section, dict):
            continue
        for record_name, record in section.items():
            if isinstance(record, dict):
                for path_key in ("path", "source_path", "source_config_path"):
                    protect_path(
                        f"{section_label} {record_name}", record.get(path_key)
                    )
            else:
                protect_path(f"{section_label} {record_name}", record)

    for section_name, section_label in (
        ("inputs", "定位输入"),
        ("artifacts", "定位产物"),
    ):
        section = manifest.get(section_name)
        if not isinstance(section, dict):
            continue
        for record_name, record in section.items():
            if isinstance(record, dict):
                protect_path(f"{section_label} {record_name}", record.get("path"))

    if output_path in protected_paths:
        labels = "、".join(sorted(protected_paths[output_path]))
        raise ValueError(
            f"评估输出路径不能覆盖定位来源文件（{labels}）：{output_path}；"
            "请改用单独的评估结果路径"
        )

    def verify_localization_input(
        input_name: str, generation_artifact_name: str, label: str
    ) -> None:
        try:
            record = manifest["inputs"][input_name]
            input_path = Path(record["path"]).expanduser().resolve()
            recorded_sha256 = str(record["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"定位清单缺少{label}的 path+sha256 记录") from error
        verified = verify_generation_artifact(
            generation_manifest_data, generation_artifact_name, input_path
        )
        if verified["sha256"] != recorded_sha256:
            raise ValueError(f"{label} SHA-256 在生成清单与定位清单之间不一致")

    verify_localization_input("scene", "scene_json", "二维场景")
    verify_localization_input(
        "online_measurement", "online_measurement", "在线 CSI"
    )
    truth_capture = capture_file(truth_path)
    verified_truth_record = verify_generation_artifact(
        generation_manifest_data, "ground_truth", truth_capture
    )

    truth_sha256 = verified_truth_record["sha256"]
    true_position, true_bias_s = _load_evaluation_truth(truth_capture.data)
    estimated_position = np.asarray(result["mu_m"], dtype=float)
    estimated_bias_s = float(result["clock_bias_s"])
    metrics = {
        "schema_version": 3,
        "localization_run_id": localization_run_id,
        "generation_bundle_id": actual_bundle_id,
        "source_generation_manifest_sha256": actual_generation_manifest_record[
            "sha256"
        ],
        "source_result_sha256": actual_result_sha256,
        "source_truth_sha256": truth_sha256,
        "localization_error_m": float(np.linalg.norm(estimated_position - true_position)),
        "clock_bias_error_s": estimated_bias_s - true_bias_s,
        "clock_bias_error_ns": (estimated_bias_s - true_bias_s) * 1e9,
        "estimated_position_m": estimated_position,
        "true_position_m": true_position,
        "estimated_clock_bias_s": estimated_bias_s,
        "true_clock_bias_s": true_bias_s,
        "evaluation_only": True,
    }
    latest_manifest_capture = capture_file(manifest_path)
    try:
        latest_manifest = json.loads(latest_manifest_capture.data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("定位清单在评估运行期间变成了无效 JSON，拒绝写回") from error
    if not isinstance(latest_manifest, dict):
        raise ValueError("定位清单在评估运行期间改变了顶层类型，拒绝写回")
    if latest_manifest.get("run_id") != localization_run_id:
        raise ValueError("定位清单 run_id 在评估运行期间发生变化，拒绝写回")
    if latest_manifest_capture.sha256 != manifest_capture.sha256:
        raise ValueError("定位清单内容在评估运行期间发生变化，拒绝写回")

    metrics_bytes = _encoded_json(metrics)
    manifest["evaluation_pending"] = False
    manifest["evaluation"] = {
        "path": str(output_path.resolve()),
        "sha256": sha256(metrics_bytes).hexdigest(),
    }
    _publish_evaluation_pair(
        metrics_path=output_path,
        metrics_bytes=metrics_bytes,
        manifest_path=manifest_path,
        manifest_bytes=_encoded_json(manifest),
    )
    return metrics


def run_offline_demo(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    root = resolve_output_root(config)
    localization_config = localization_config_view(config)
    validate_localization_config(localization_config)
    with exclusive_output_root_lock(root):
        previous_evaluation_archived = _archive_previous_evaluation(
            root, root / "localization", archive_sources=True
        )
        scene_artifacts = _prepare_scene_locked(
            config,
            root,
            allow_overwrite=previous_evaluation_archived,
        )
        data_artifacts = _generate_data_locked(
            config,
            scene_json=scene_artifacts["scene_json"],
            root=root,
            allow_overwrite=previous_evaluation_archived,
        )
        result = _localize_locked(
            localization_config,
            scene_json=scene_artifacts["scene_json"],
            online_input=data_artifacts["online_npz"],
            generation_manifest=data_artifacts["generation_manifest"],
            output_root=root,
            archive_previous=False,
            previous_evaluation_archived=previous_evaluation_archived,
        )
        metrics = _evaluate_locked(
            result_path=(root / "localization" / "localization_result.json").resolve(),
            truth_path=Path(data_artifacts["truth_npz"]).resolve(),
            output_path=(root / "evaluation" / "metrics.json").resolve(),
            expected_run_id=str(result["localization_run_id"]),
        )
        _write_json(
            root / "run_manifest.json",
            {
                "stage": "offline_end_to_end",
                "config": str(Path(config_path).resolve()),
                "scene_artifacts": scene_artifacts,
                "data_artifacts": data_artifacts,
                "localization_result": str(
                    root / "localization" / "localization_result.json"
                ),
                "evaluation_metrics": str(root / "evaluation" / "metrics.json"),
            },
        )
    return {"result": result, "metrics": metrics, "output_root": str(root)}


__all__ = [
    "associate_perturbed_peaks",
    "estimate_noise_std_from_observed_csi",
    "evaluate",
    "generate_data",
    "localize",
    "prepare_scene",
    "run_offline_demo",
]
