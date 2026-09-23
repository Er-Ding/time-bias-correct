"""对照组的开关和样本分组必须绑定同一份配置与结果。"""
import importlib.util
import json
from pathlib import Path

import pytest

from time_bias_localization.config import DEFAULT_CONFIG, localization_config_view


def test_arm_controls_and_current_attempt(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_arm_comparison.py"
    spec = importlib.util.spec_from_file_location("arm_comparison", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    base = localization_config_view(DEFAULT_CONFIG)
    base["music"]["spurious_peak_filter"]["enabled"] = True
    base["localization"]["amplitude_weighting"]["enabled"] = True
    configs = {arm: runner.arm_config(base, arm) for arm in runner.ARMS}
    for arm, expected in (("B0", (False, False)), ("M", (True, False)), ("W", (False, True))):
        config = configs[arm]
        assert (config["music"]["spurious_peak_filter"]["enabled"],
                config["localization"]["amplitude_weighting"]["enabled"]) == expected
        config["music"]["spurious_peak_filter"]["enabled"] = True
        config["localization"]["amplitude_weighting"]["enabled"] = True
        assert config == base  # 开关以外没有变量改变。

    sample = tmp_path / "samples/SAMPLE_000001"
    attempt = sample / "localization_attempts/current"
    old = sample / "localization_attempts/stale/localization"
    old.mkdir(parents=True)
    (old / "music_peaks.json").write_text(json.dumps({"nominal": [{"aoa_rad": 1.56}]}))
    (attempt / "localization").mkdir(parents=True)
    peaks = attempt / "localization/music_peaks.json"
    peaks.write_text(json.dumps({"nominal": [{"aoa_rad": 0.1}]}))
    (sample / "result.json").write_text(json.dumps({
        "status": "success", "rt_path_count": 2, "true_position_m": [1, 2],
        "true_clock_bias_ns": 3, "attempt_dir": str(attempt)}))
    groups, _ = runner.classify_samples(tmp_path)
    assert groups["success_without_boundary"] == [sample.name]
    assert groups["success_with_boundary"] == []
    peaks.unlink()
    with pytest.raises(ValueError, match="唯一"):
        runner.classify_samples(tmp_path)
