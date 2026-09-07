"""项目命令行入口。"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from .config import load_config, load_localization_config
from .pipeline import (
    evaluate,
    generate_data,
    localize,
    prepare_scene,
    resolve_output_root,
    run_offline_demo,
)


def _print_summary(data: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"无法转成 JSON：{type(value).__name__}")

    print(json.dumps(data, ensure_ascii=False, indent=2, default=convert))


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML 配置文件")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="time-bias-localize",
        description="二维多径定位与公共到达时间偏差联合估计",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    offline = commands.add_parser("offline-demo", help="运行不依赖外部数据的完整闭环")
    _common_config(offline)

    scene = commands.add_parser("prepare-scene", help="生成离线二维场景与俯视图")
    _common_config(scene)

    data = commands.add_parser("generate-data", help="由离线场景生成带偏差 CSI")
    _common_config(data)
    data.add_argument("--scene-json", help="二维场景 JSON；默认取输出目录中的场景")

    locate = commands.add_parser("localize", help="只读取在线 CSI 和二维场景进行定位")
    _common_config(locate)
    locate.add_argument("--scene-json", required=True, help="二维场景 JSON")
    locate.add_argument("--online-input", required=True, help="online/measurement.npz")
    locate.add_argument(
        "--generation-manifest",
        help="生成清单 JSON；不填时只在输出目录的两个标准位置中查找",
    )
    locate.add_argument(
        "--run-receipt",
        help="成功后原子创建且不覆盖的运行回执 JSON",
    )

    score = commands.add_parser("evaluate", help="定位结束后单独读取真值评估")
    score.add_argument("--result-json", required=True)
    score.add_argument("--truth-npz", required=True)
    score.add_argument("--output-json", required=True)
    score.add_argument(
        "--expected-run-id",
        required=True,
        help="必须与定位回执中的 run_id 完全一致",
    )

    sionna = commands.add_parser(
        "prepare-sionna-scene", help="运行 Sionna RT 并转换为 DeepMIMO V4"
    )
    _common_config(sionna)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "offline-demo":
        summary = run_offline_demo(args.config)
        _print_summary(
            {
                "output_root": summary["output_root"],
                "mu_m": summary["result"]["mu_m"],
                "sigma_m2": summary["result"]["sigma_m2"],
                "clock_bias_s": summary["result"]["clock_bias_s"],
                "metrics": summary["metrics"],
            }
        )
        return 0

    if args.command == "prepare-scene":
        config = load_config(args.config)
        _print_summary(prepare_scene(config))
        return 0

    if args.command == "generate-data":
        config = load_config(args.config)
        _print_summary(generate_data(config, scene_json=args.scene_json))
        return 0

    if args.command == "localize":
        config = load_localization_config(args.config)
        result = localize(
            config,
            scene_json=args.scene_json,
            online_input=args.online_input,
            generation_manifest=args.generation_manifest,
            run_receipt=args.run_receipt,
        )
        _print_summary(
            {
                "mu_m": result["mu_m"],
                "sigma_m2": result["sigma_m2"],
                "clock_bias_s": result["clock_bias_s"],
                "localization_run_id": result["localization_run_id"],
            }
        )
        return 0

    if args.command == "evaluate":
        _print_summary(
            evaluate(
                result_json=args.result_json,
                truth_npz=args.truth_npz,
                output_json=args.output_json,
                expected_run_id=args.expected_run_id,
            )
        )
        return 0

    if args.command == "prepare-sionna-scene":
        from .sionna_generation import generate_sionna_deepmimo_bundle

        config = load_config(args.config)
        _print_summary(
            generate_sionna_deepmimo_bundle(config, output_root=resolve_output_root(config))
        )
        return 0
    raise AssertionError(f"未知命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
