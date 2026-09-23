"""无损归档四份旧报告的逐样本 SVG；数值、PNG、PDF 和原始清单保留原位。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import time
import uuid


REPORTS = (
    "outputs/step_report_20260908T021953_1933279",
    "outputs/spectrum_experiment_20260908T091621_2543248/step_report",
    "outputs/point_clustering_experiment_20260909T022303_3991465/step_report",
    "outputs/fine_dbscan_experiment_20260909T041220_4156404/step_report",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def identity(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError(f"不是普通文件：{path}")
    return {key: getattr(value, key) for key in (
        "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink", "st_blocks")}


def source_path(workspace, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".svg":
        raise ValueError(f"不是允许的相对 SVG 路径：{relative}")
    roots = [Path(root) for root in REPORTS if relative.is_relative_to(Path(root) / "samples")]
    if len(roots) != 1:
        raise ValueError(f"SVG 不在四份报告的 samples 目录中：{relative}")
    path = workspace / relative
    if path.resolve() != path or path.is_symlink():
        raise ValueError(f"拒绝符号链接或路径逃逸：{path}")
    return path, workspace / roots[0]


def check_counterparts(path):
    for suffix in (".png", ".pdf"):
        counterpart = path.with_suffix(suffix)
        if counterpart.resolve() != counterpart or identity(counterpart)["st_size"] == 0:
            raise ValueError(f"PNG/PDF 不存在、为空或是链接：{counterpart}")


def progress(stage, count, total):
    if count == total or count % 500 == 0:
        print(f"{stage}：{count}/{total}", flush=True)


def verify_archive(archive, records):
    expected = {row["relative_path"]: row for row in records}
    if len(expected) != len(records):
        raise ValueError("归档清单有重复路径")
    seen = set()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            row = expected.get(member.name)
            if row is None or member.name in seen or not member.isfile() or member.size != row["bytes"]:
                raise ValueError(f"归档含多余、重复或不匹配的文件：{member.name}")
            with stream.extractfile(member) as contents:
                actual = hashlib.file_digest(contents, "sha256").hexdigest()
            if actual != row["sha256"]:
                raise ValueError(f"归档解压后的 SHA256 不符：{member.name}")
            seen.add(member.name)
            progress("流式解压校验", len(seen), len(records))
    if seen != expected.keys():
        raise ValueError("归档缺少清单中的文件")


def archive_reports(workspace, candidates, directory):
    directory.mkdir(parents=True, exist_ok=False)
    requested = json.loads(candidates.read_text())
    if not isinstance(requested, list) or not requested:
        raise ValueError("候选清单必须是非空列表")
    records, manifests = [], {}
    for number, candidate in enumerate(requested, 1):
        path, report = source_path(workspace, Path(candidate["path"]).relative_to(workspace))
        check_counterparts(path)
        if report not in manifests:
            manifest = report / "report_manifest.json"
            manifests[report] = {
                "path": str(manifest), "sha256": digest(manifest),
                "artifacts": {item["path"]: item["sha256"] for item in json.loads(manifest.read_text())["artifacts"]},
            }
        if manifests[report]["artifacts"].get(str(path)) != candidate["sha256"]:
            raise ValueError(f"候选与原始报告清单不一致：{path}")
        before = identity(path)
        if before["st_size"] != candidate["bytes"] or digest(path) != candidate["sha256"] or identity(path) != before:
            raise ValueError(f"源文件内容或身份已变化：{path}")
        records.append({"relative_path": str(path.relative_to(workspace)), "sha256": candidate["sha256"],
                        "bytes": candidate["bytes"], "identity": before})
        progress("源文件 SHA256 校验", number, len(requested))
    if len({row["relative_path"] for row in records}) != len(records):
        raise ValueError("候选清单有重复路径")
    plan = {"schema_version": 1, "workspace": str(workspace), "created_at_epoch_s": time.time(),
            "candidate_manifest": {"path": str(candidates), "sha256": digest(candidates)},
            "report_manifests": [{k: v for k, v in row.items() if k != "artifacts"} for row in manifests.values()],
            "records": records}
    write_json(directory / "archive_plan.json", plan)
    temporary, archive = directory / "report_svgs.tar.gz.part", directory / "report_svgs.tar.gz"
    with tarfile.open(temporary, "w:gz", compresslevel=1, dereference=False) as stream:
        for number, row in enumerate(records, 1):
            path, _ = source_path(workspace, row["relative_path"])
            if identity(path) != row["identity"]:
                raise ValueError(f"压缩前源文件已变化：{path}")
            stream.add(path, arcname=row["relative_path"], recursive=False)
            if identity(path) != row["identity"]:
                raise ValueError(f"压缩期间源文件已变化：{path}")
            progress("无损压缩", number, len(records))
    verify_archive(temporary, records)
    temporary.rename(archive)
    write_json(directory / "archive_verified.json", {"sha256": digest(archive), "bytes": archive.stat().st_size,
                                                     "verified_file_count": len(records)})
    with archive.open("rb") as saved_archive:
        os.fsync(saved_archive.fileno())
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    for manifest in plan["report_manifests"]:
        if digest(manifest["path"]) != manifest["sha256"]:
            raise ValueError(f"原始报告清单已变化：{manifest['path']}")
    # 先整批检查，再逐个检查；中途变化会停止删除，已归档的所有内容仍可恢复。
    for row in records:
        path, _ = source_path(workspace, row["relative_path"])
        check_counterparts(path)
        if identity(path) != row["identity"]:
            raise ValueError(f"删除前源文件已变化：{path}")
    released = 0
    with (directory / "deletions.jsonl").open("x") as log:
        for number, row in enumerate(records, 1):
            path, _ = source_path(workspace, row["relative_path"])
            check_counterparts(path)
            if identity(path) != row["identity"]:
                raise ValueError(f"删除时源文件已变化：{path}")
            path.unlink()
            # 多链接文件删除一个名字不会释放数据块；不要夸大回收量。
            released += row["identity"]["st_blocks"] * 512 if row["identity"]["st_nlink"] == 1 else 0
            log.write(json.dumps({"relative_path": row["relative_path"], "deleted_at_epoch_s": time.time()}) + "\n")
            log.flush()
            progress("删除已校验的原 SVG", number, len(records))
        os.fsync(log.fileno())
    footprint = sum(item.stat().st_blocks * 512 for item in directory.iterdir() if item.is_file())
    result = {"status": "success", "deleted_svg_count": len(records), "original_file_bytes": sum(row["bytes"] for row in records),
              "released_original_allocated_bytes": released, "archive_directory_allocated_bytes_before_result": footprint,
              "net_reclaimed_allocated_bytes_before_result": released - footprint,
              "archive": str(archive), "restore_required_for_original_report_manifest_full_check": True}
    write_json(directory / "archive_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def restore_reports(workspace, directory):
    plan = json.loads((directory / "archive_plan.json").read_text())
    if plan["workspace"] != str(workspace):
        raise ValueError("恢复工作区与归档记录不一致")
    records = plan["records"]
    archive = directory / "report_svgs.tar.gz"
    verified = json.loads((directory / "archive_verified.json").read_text())
    if digest(archive) != verified["sha256"]:
        raise ValueError("归档本身的 SHA256 已变化")
    for row in records:
        path, _ = source_path(workspace, row["relative_path"])
        if path.exists() and (identity(path)["st_size"] != row["bytes"] or digest(path) != row["sha256"]):
            raise FileExistsError(f"恢复拒绝覆盖已有不同内容：{path}")
    verify_archive(archive, records)
    restored = 0
    by_path = {row["relative_path"]: row for row in records}
    with tarfile.open(archive, "r|gz") as stream:
        for number, member in enumerate(stream, 1):
            row = by_path[member.name]
            path, _ = source_path(workspace, member.name)
            if path.exists():
                if digest(path) != row["sha256"]:
                    raise FileExistsError(f"恢复期间原路径出现不同文件：{path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.restore-{uuid.uuid4().hex}.tmp")
            try:
                with stream.extractfile(member) as source, temporary.open("xb") as destination:
                    while block := source.read(1024 * 1024):
                        destination.write(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                if digest(temporary) != row["sha256"]:
                    raise ValueError(f"恢复内容校验失败：{path}")
                os.chmod(temporary, stat.S_IMODE(row["identity"]["st_mode"]))
                os.utime(temporary, ns=(row["identity"]["st_mtime_ns"], row["identity"]["st_mtime_ns"]))
                source_path(workspace, member.name)
                os.link(temporary, path)  # 原子创建，已有目标时失败；随即移除临时名字。
                restored += 1
            finally:
                temporary.unlink(missing_ok=True)
            progress("恢复 SVG", number, len(records))
    result = {"status": "success", "restored_svg_count": restored, "already_present_identical_count": len(records) - restored}
    write_json(directory / f"restore_result_{time.time_ns()}.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "restore"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    workspace, directory = args.workspace.resolve(), args.archive_dir.absolute()
    if directory.resolve() != directory or not directory.is_relative_to(workspace / "outputs/workspace_maintenance_20260921"):
        raise ValueError("归档目录必须在本工作区维护目录内，且不能经过符号链接")
    if args.action == "archive":
        if args.candidates is None:
            parser.error("archive 需要 --candidates")
        archive_reports(workspace, args.candidates.resolve(), directory)
    else:
        restore_reports(workspace, directory)


if __name__ == "__main__":
    main()
