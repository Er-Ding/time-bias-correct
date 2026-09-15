"""在独立目录运行连续传播模型的 CPU 检查，并保存可核对的测试汇总。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from time_bias_localization.boundary_experiment import source_fingerprint
from time_bias_localization.config import load_localization_config
from time_bias_localization.provenance import file_sha256


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def implementation_fingerprint() -> dict:
    project = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), project / "scripts" / "run_continuous_comparison.py",
             project / "scripts" / "detached_task.py", project / "run_continuous_model_check.sh",
             project / "run_continuous_model_experiment.sh", project / "run_continuous_model_report.sh"]
    files += sorted(project.glob("tests/test_continuous*.py"))
    files += sorted(project.glob("tests/test_propagation*.py"))
    return {**source_fingerprint(), "continuous_runner_and_test_files": {
        str(path.relative_to(project)): file_sha256(path) for path in files}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("focused", "all"), default="focused")
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    config_path = args.config.resolve()
    config = load_localization_config(config_path)
    if config["localization"]["solver_method"] != "continuous":
        parser.error("验收配置必须使用 continuous 求解器")
    output.mkdir(parents=True, exist_ok=False)
    started_at = datetime.now(timezone.utc).isoformat()
    source = implementation_fingerprint()
    config_hash = file_sha256(config_path)
    write_json(output / "source.json", source)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    command = [sys.executable, "-u", "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(output / "pytest.xml")]
    if args.scope == "focused":
        tests = sorted(set(project.glob("tests/test_continuous*.py"))
                       | set(project.glob("tests/test_propagation*.py")))
        if not tests:
            raise RuntimeError("没有找到连续模型测试；不能报告空测试通过")
        tests += [project / "tests/test_config.py"]
        command += [str(path) for path in tests]
    write_json(output / "command.json", {"cwd": str(project), "command": command})
    print(f"[1/2] 连续模型 CPU 检查开始，范围={args.scope}，输出={output}", flush=True)
    completed = subprocess.run(command, cwd=project, stdin=subprocess.DEVNULL, check=False)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    junit = output / "pytest.xml"
    if junit.is_file():
        for suite in ET.parse(junit).getroot().iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.get(key, 0))
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    unchanged = implementation_fingerprint() == source and file_sha256(config_path) == config_hash
    ok = completed.returncode == 0 and counts["passed"] > 0 and unchanged
    write_json(output / "validation_summary.json", {
        "status": "passed" if ok else "failed", "scope": args.scope,
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "pytest_exit_code": completed.returncode, "tests": counts,
        "source_and_config_unchanged": unchanged,
        "config_path": str(config_path), "config_sha256": config_hash,
        "gpu_used": False, "full_frozen_30_by_5_experiment_run": False,
        "historical_results_modified": False,
        "scientific_acceptance": "not_established_by_implementation_tests",
    })
    print(f"[2/2] {'通过' if ok else '失败'}：{counts}；{output / 'validation_summary.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
