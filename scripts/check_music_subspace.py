"""CPU 回归验收；保存源码、配置摘要和测试结果，不重跑历史预跑。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from time_bias_localization.boundary_experiment import source_fingerprint
from time_bias_localization.config import load_config
from time_bias_localization.provenance import file_sha256


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scope", choices=("focused", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    config_hash = file_sha256(args.config)
    config = load_config(args.config)
    assert config["music"]["subspace_selection"]["mode"] == "eigenvalue_threshold"
    write(args.output/"source.json", source)
    (args.output/"config.yaml").write_bytes(args.config.read_bytes())
    command = [sys.executable,"-u","-m","pytest","-q","-p","no:cacheprovider",
               "--junitxml="+str(args.output/"pytest.xml")]
    if args.scope == "focused":
        command += ["tests/test_music_subspace.py","tests/test_compute.py",
                    "tests/test_path_detection.py","tests/test_accuracy_fix_integration.py"]
    print(f"[1/2] CPU 测试开始，范围={args.scope}，记录={args.output}", flush=True)
    completed = subprocess.run(command)
    counts = {"tests":0,"failures":0,"errors":0,"skipped":0}
    if (args.output/"pytest.xml").is_file():
        for suite in ET.parse(args.output/"pytest.xml").getroot().iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.get(key,0))
    counts["passed"] = counts["tests"]-counts["failures"]-counts["errors"]-counts["skipped"]
    unchanged = source_fingerprint() == source and file_sha256(args.config) == config_hash
    ok = completed.returncode == 0 and unchanged
    write(args.output/"validation_summary.json",dict(status="passed" if ok else "failed",
        scope=args.scope, tests=counts, source_and_config_unchanged=unchanged,
        config_path=str(args.config.resolve()), config_sha256=config_hash,
        selection=config["music"]["subspace_selection"],
        gpu_used=False, scientific_pilot_rerun=False, historical_results_modified=False,
        threshold_is_scientifically_calibrated=False))
    print(f"[2/2] {'通过' if ok else '失败'}：{counts}；{args.output}/validation_summary.json",flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
