# 工作区空间检查与 SVG 无损归档

检查范围为 `/data/zhujun/differt_projects/time-bias-correct`。对 `/data/zhujun` 仅统计一级目录大小，没有读取其他用户目录，也不清理其他项目。检查日期：2026-09-21 UTC。

## 空间主要花在哪里

开始检查时，`/data` 可用 **260,759,277,568 字节，约 242.85 GiB**，占用率 97%。当前仓库占 **81,257,123,840 字节，约 75.68 GiB**，其中 `outputs` 占 **80,859,566,080 字节，约 75.31 GiB**。以下大小含目录本身，由 `du -x -B1` 取得。

| 当前仓库中的目录 | 检查前字节数 | 处理 |
| --- | ---: | --- |
| `outputs/diffraction_boundary_music_v5_20260914_01` | 16,392,572,928 | 保留实验与失败证据 |
| `outputs/diffraction_boundary_v3` | 10,514,538,496 | 保留实验与失败证据 |
| `outputs/fine_dbscan_experiment_20260909T041220_4156404` | 9,285,251,072 | 仅归档报告内逐样本 SVG |
| `outputs/point_clustering_experiment_20260909T022303_3991465` | 9,009,016,832 | 仅归档报告内逐样本 SVG |
| `outputs/spectrum_experiment_20260908T091621_2543248` | 8,927,121,408 | 仅归档报告内逐样本 SVG |
| `outputs/step_report_20260908T021953_1933279` | 8,628,629,504 | 仅归档逐样本 SVG，报告数据必须保留 |
| `outputs/diffraction_boundary_v2` | 5,293,285,376 | 保留 |
| `outputs/diffraction_boundary_ransac_v6_20260914_01` | 4,181,901,312 | 保留 |
| `outputs/monte_carlo_munich_1000_20260915_01` | 4,061,958,144 | 保留 |
| `outputs/arm_comparison_20260922_03` | 3,841,896,448 | 保留 |

`outputs` 共统计 151,655 个普通文件。按文件内容大小汇总：JSON 41,072,072,737 字节，SVG 26,410,458,659 字节，NPZ 7,087,253,352 字节，PDF 2,559,422,532 字节，PNG 2,102,752,958 字节。目录占用、文件逻辑大小和文件实际分配块数是不同口径，不能混加。

主要原因：

- `music_spectrum.svg` 合计 **23,424,604,441 字节**。旧绘图把大谱面的每个网格保存成矢量对象，单图约 39 MB；同一图同时保留 PNG、PDF 和 SVG。
- `representative_points.json` 与 `representative_trajectories.json` 合计 **30,963,223,160 字节**。最大的单文件超过 300 MB，多处属于无法定位时保存的中间证据。没有因目录名带 `unavailable`、旧日期或失败状态而删除这些文件。
- 本仓库 `.cache/pip` 是可重新下载的 pip 缓存，占 112,881,664 字节；占比较小，本次不动。`.venv`、`.sionna-venv` 是运行环境，不按缓存处理。

## 具体清理边界

只选择以下四份报告 **`samples/` 之下的 `.svg`**，保留 `summary/` 的全部图片、原数值文件、PNG、PDF、README、配置、运行日志、生成清单和失败记录。

| 报告目录 | SVG 数量 | 原 SVG 内容字节数 | 报告所引用输入的现存情况 |
| --- | ---: | ---: | --- |
| `outputs/step_report_20260908T021953_1933279` | 4,785 | 6,740,004,865 | 2,065 条来源路径全部缺失 |
| `outputs/spectrum_experiment_20260908T091621_2543248/step_report` | 2,250 | 6,536,822,711 | 2,131 条来源路径全部存在 |
| `outputs/point_clustering_experiment_20260909T022303_3991465/step_report` | 2,250 | 6,528,561,056 | 2,281 条来源路径全部存在 |
| `outputs/fine_dbscan_experiment_20260909T041220_4156404/step_report` | 2,700 | 6,587,429,963 | 2,281 条来源路径全部存在 |
| **合计** | **11,985** | **26,392,818,595** | 每个候选都有同名 PNG 和 PDF |

特别注意：第一份旧报告的来源都指向已不存在的 `outputs/gpu_munich_20260908T020052_1904235`。其中复制的数值文件可能是唯一留存，不能整目录删除，也不能声称一定能重新计算出原报告。本次保留所有原图的无损压缩副本，需要时可以原字节恢复，避免依赖重跑旧算法。

依据代码：`visualization.create_report()` 读取定位产物后调用 `step_visualization.export_steps()` 导出报告；`visualization._save()` 写三种图片；报告末尾生成 `report_manifest.json`。在线定位不读取这些报告 SVG。代码搜索未发现生产流程消费 `report_manifest.json`；研究背景文档引用的是 `summary/summary.json` 和 `summary/per_attempt.csv`，这两类文件原位保留。

**原 `report_manifest.json` 不修改。它仍记载原 SVG 的路径和 SHA256，因此在恢复 SVG 之前，对报告全部图片做原始完整性校验会发现这些 SVG 不在原路径。** 归档计划另存原始清单指纹、每张 SVG 的原路径、SHA256、大小和文件身份。按原路径恢复后可验证原清单中的 SVG；这不修复第一份报告在清理前就已经丢失的上游输入。

## 可复查记录与执行脚本

维护目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921`。

- `file_inventory.json`：检查前逐文件大小、实际分配块、inode 和链接数。
- `storage_audit_summary.json`：按目录、文件类型和同名文件归总，以及来源路径存在性检查。
- `report_svg_archive_candidates.json`：11,985 张 SVG 的精确候选清单，含原始清单 SHA256。
- `svg_archive_20260921T142951_1139999/archive_plan.json`：执行时全量复核后的归档计划、原始清单指纹与文件身份。
- `svg_archive_20260921T142951_1139999/report_svgs.tar.gz`：保留工作区相对路径的无损归档。
- `svg_archive_20260921T142951_1139999/archive_verified.json`：归档 SHA256 及流式解压逐文件验证结果。
- `svg_archive_20260921T142951_1139999/deletions.jsonl`：实际删除记录；只有全部解压校验成功后才开始删除。
- `svg_archive_20260921T142951_1139999/archive_result.json`：最终实际数量、原文件分配块、归档目录占用和净减少量。

执行入口是 `/data/zhujun/differt_projects/time-bias-correct/run_archive_report_svgs.sh`。它复用 `scripts/detached_task.py`，使用 `nohup + setsid`，保存实际任务 PID、进程身份、参数、时间、退出码和日志。归档程序只用 Python 标准库，拒绝越界路径、符号链接、变化的文件和缺失 PNG/PDF 的候选；恢复不覆盖已有不同内容。

小样本验证已通过，退出码 0，记录在 `runs/check_20260921T142944_1139833`。覆盖无损往返、重复恢复不改写、已有内容冲突、归档验证失败时保留源文件、拒绝符号链接及越界路径。

本次真实运行记录为 `runs/archive_20260921T142951_1139999`。直接复制以下命令即可管理任务，重新连接服务器后仍有效：

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/archive_20260921T142951_1139999/task.log
/home/zhujun/miniconda3/bin/python3 /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py status /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/archive_20260921T142951_1139999
/home/zhujun/miniconda3/bin/python3 /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py stop /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/archive_20260921T142951_1139999
```

查看日志时按 `Ctrl+C` 只退出查看，不停止后台任务。停止命令先核对进程身份，再停止该任务的整个进程组。

需要恢复全部 SVG 时，可使用以下一组命令。恢复也在后台运行；固定的新记录目录已存在时会拒绝覆盖，需要在参数区换一个新目录名。

```bash
ACTION=restore ARCHIVE_DIR=/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/svg_archive_20260921T142951_1139999 RUN_DIR=/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/restore_svgs_manual_01 bash /data/zhujun/differt_projects/time-bias-correct/run_archive_report_svgs.sh
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/restore_svgs_manual_01/task.log
/home/zhujun/miniconda3/bin/python3 /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py status /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/restore_svgs_manual_01
/home/zhujun/miniconda3/bin/python3 /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py stop /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_maintenance_20260921/runs/restore_svgs_manual_01
```

## 当前仓库之外

`/data/zhujun` 合计 1,716,700,798,976 字节。以下仅定位大头，不代表可删除：

| 一级目录 | 字节数 |
| --- | ---: |
| `Vision-try` | 650,422,648,832 |
| `liangle-dataset` | 394,545,213,440 |
| `Multimodal-Wireless` | 334,165,069,824 |
| `conda_envs` | 116,165,103,616 |
| `differt_projects` | 103,777,267,712 |
| `transfer_bundle_20260323` | 57,648,250,880 |
| `zhujun_WirelessGPT_downstreamtask_EnvRecon` | 29,563,367,424 |
| `CARLA_0.9.15` | 20,279,173,120 |

这些目录没有进行文件级来源核对，本次不清理。也没有为追求更大回收量，删除本仓库现有场景、CSI、真值、仿真输入、定位结果、失败证据或安装环境。

## 完成结果

**已完成。** 实际任务于 2026-09-21 14:29:52 UTC 开始，14:35:02 UTC 结束，用时 310.72 秒，退出码 0。11,985 张 SVG 全部按原始 SHA256 校验并无损归档；归档再全量流式解压校验通过后，才删除原文件。随后检查确认 11,985 张原 SVG 已移走，23,970 张同名 PNG/PDF 均仍存在，四份报告根目录均新增 `SVG_ARCHIVE_NOTICE.md` 指向恢复说明。

| 指标 | 实测字节数 |
| --- | ---: |
| 被移走的原 SVG 实际分配块 | 26,417,590,272 |
| `.tar.gz` 内容大小 | 2,895,324,776 |
| 完整归档目录占用，含计划、校验及删除记录 | 2,904,313,856 |
| **归档本身净回收** | **23,513,276,416，约 21.90 GiB** |
| 仓库总占用：检查前 → 完成后 | 81,257,123,840 → 57,821,732,864 |
| **仓库净减少，包含本次清单和日志等新增开销** | **23,435,390,976，约 21.83 GiB** |
| `/data` 可用空间：检查前 → 完成后 | 260,759,277,568 → 284,194,643,968 |

`/data` 可用空间实测增加 23,435,366,400 字节，完成后约 **264.68 GiB**。整个文件系统可能有其他并发写入，因此 `df` 差值与本项目文件块的净回收量不要求完全相等。快照保存于 `storage_after_cleanup.json`。

归档 SHA256：`156082957306f0cce8813624a2ca9f615b39f34a109ef0c4421d4735f906a44c`。本次没有将旧实验状态改成成功，也没有重算或改变任何定位数值。
