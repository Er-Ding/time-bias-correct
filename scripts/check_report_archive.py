"""小样本验证归档、恢复、内容冲突及校验失败时不删除源文件。"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import archive_report_svgs as archive


def main():
    with tempfile.TemporaryDirectory(prefix="report_archive_check_") as temporary:
        workspace = Path(temporary).resolve()
        report = workspace / archive.REPORTS[0]
        folder = report / "samples/UE001/repeat_000/02_music"
        folder.mkdir(parents=True)
        svg = folder / "music_spectrum.svg"
        original = b"<svg><text>scientific figure</text></svg>\n" * 100
        svg.write_bytes(original)
        for suffix in (".png", ".pdf"):
            svg.with_suffix(suffix).write_bytes(b"keep counterpart")
        manifest = report / "report_manifest.json"
        manifest.write_text(json.dumps({"artifacts": [{"path": str(svg), "sha256": archive.digest(svg)}]}))
        manifest_sha = archive.digest(manifest)
        candidate = workspace / "candidates.json"
        candidate.write_text(json.dumps([{"path": str(svg), "bytes": svg.stat().st_size, "sha256": archive.digest(svg)}]))
        directory = workspace / "outputs/workspace_maintenance_20260921/roundtrip"
        archive.archive_reports(workspace, candidate, directory)
        assert not svg.exists() and archive.digest(manifest) == manifest_sha
        assert all(svg.with_suffix(suffix).read_bytes() == b"keep counterpart" for suffix in (".png", ".pdf"))
        archive.restore_reports(workspace, directory)
        assert svg.read_bytes() == original
        before = archive.identity(svg)
        archive.restore_reports(workspace, directory)
        assert archive.identity(svg) == before

        svg.write_bytes(b"newer user content")
        try:
            archive.restore_reports(workspace, directory)
            raise AssertionError("覆盖已有不同内容时应拒绝恢复")
        except FileExistsError:
            assert svg.read_bytes() == b"newer user content"
        svg.write_bytes(original)
        with patch.object(archive, "verify_archive", side_effect=ValueError("simulated archive corruption")):
            try:
                archive.archive_reports(workspace, candidate, directory.parent / "corrupt")
                raise AssertionError("归档校验失败后仍然删除了源文件")
            except ValueError:
                assert svg.read_bytes() == original

        link = folder / "unsafe.svg"
        link.symlink_to(svg)
        for relative in (link.relative_to(workspace), Path("../escape.svg"), Path("outputs/other/samples/a.svg")):
            try:
                archive.source_path(workspace, relative)
                raise AssertionError(f"应拒绝路径：{relative}")
            except ValueError:
                pass
        print("通过：无损归档与恢复、不覆盖冲突文件、校验失败保留源文件、拒绝越界和符号链接。", flush=True)


if __name__ == "__main__":
    main()
