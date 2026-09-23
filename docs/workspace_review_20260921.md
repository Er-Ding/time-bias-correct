# 工作区整理与代码审查（2026-09-21）

本次针对 `/data/zhujun/differt_projects/time-bias-correct` 的当前连续定位、最近的场景修复、三组对照实验和历史输出进行了检查与修复。原有未提交改动保留，没有改写历史实验结果，没有重新启动整批 1000 点实验。

## 现在从哪里进入

| 工作 | 入口及依据 |
| --- | --- |
| 当前连续定位实验 | [run_monte_carlo_experiment.sh](../run_monte_carlo_experiment.sh)，[配置](../configs/monte_carlo_continuous_munich.yaml) |
| 冻结已有观测的连续模型检查 | [run_continuous_model_experiment.sh](../run_continuous_model_experiment.sh)，[实施记录](continuous_model_implementation_20260915.md) |
| 最新墙面修复的几何依据 | [wall_geometry_fix_20260918.md](wall_geometry_fix_20260918.md)，[接收端几何修复](rt_receiver_geometry_fix_20260918.md) |
| 剔除伪峰／幅值加权的三组对照 | [run_arm_comparison.sh](../run_arm_comparison.sh)，各组使用同一份冻结公开配置 |
| 全量实现检查 | [run_tests.sh](../run_tests.sh)，默认后台执行 |
| 当前真实观测 CPU/GPU 检查 | [run_workspace_gpu_check.sh](../run_workspace_gpu_check.sh)，[实测报告](workspace_gpu_review_20260921.md) |
| 历史图表归档与恢复 | [run_archive_report_svgs.sh](../run_archive_report_svgs.sh)，[磁盘审计](workspace_storage_audit_20260921.md) |

历史方法和运行目录保持原名。它们的清单、报告和脚本含绝对路径，移动目录会破坏来源引用；本次用这份索引整理入口。

## 已发现并处理的问题

| 问题及影响 | 处理 |
| --- | --- |
| 在线处理选完角度分支后只保存剩下的峰，但筛选编号仍指向选择前的列表；冻结回放可能索引越界或再次删掉真实峰 | 保存选择前的完整峰与编号；回放使用同一份公开接收机参数重新筛选。无法正确恢复的旧产物明确报错 |
| 所有角度分支未求解成功时，一律记成“观测被剔除”，混淆预算耗尽、多解和几何失败 | 保留真实求解状态，不把求解失败计入观测剔除 |
| 不同角度分支都能解释观测，却直接挑一个最低代价位置输出 | 复用已有多解门限检查跨分支结果；幅值加权导致代价不可比时，保守保留多解 |
| 旧候选流程中，被剔除的伪峰仍留在采样记录里进入反向追踪 | 同步过滤采样点、区域、来源编号和统计 |
| 对照脚本的 B0 没有明确关掉两个开关，输入配置可能污染基准；每个子进程还会重新读取可变配置文件 | 明确固定 B0/M/W 的两个开关，复用正式配置加载器；在启动前冻结公开配置，子进程不接触完整生成配置 |
| 对照分组会扫描所有历史尝试，可能使用另一轮的峰；地图也未按样本绑定的来源读取 | 只读取 `result.json` 对应尝试；地图、CSI 和生成清单按样本记录读取并核对指纹 |
| 对照结果每 20 项才保存一次且直接覆盖 JSON，技术异常也可能整体退出 0 | 每完成一项原子保存；技术异常整体退出 2；新增带日志和进程身份核对的后台脚本 |
| 离线报告只按已完成结果计算成功率，漏掉未完成任务；只跑部分组时，空耗时还会导致报告报错 | 使用计划任务数作分母，单列未完成任务，按实际选择的组生成报告；后台入口支持 `ARM_TASK=analyze` |
| 连续几何检查每次失败都重新遍历所有墙，构造同一查询表 | 复用已有只读查询表，不另加缓存系统 |
| 极短墙的两端都落进同一端点容差，绕射端点判断会除零 | 共线判断直接复用墙自身的单位方向；原短墙测试改为遇到无效数或除零直接失败 |
| 旧 GPU 测量辅助函数漏传自动子空间设置，实际比较了另一种配置 | 补传原配置；重新以当前自动选维设置进行 CPU/GPU 对照 |
| 逐格矢量导出的 MUSIC 热图导致 SVG 累积数十 GiB | 六处热图统一按 300 dpi 嵌入 SVG/PDF；坐标、文字和峰标记继续保留矢量形式，原始数值不变 |

详细调用关系和回归证据见 [流程审查](workspace_review_flow_20260921.md)。本次更改会影响失败分类和多解接受结果，旧统计不能直接作为修复后的成功率。

## GPU 能解决多少

已有 CUDA MUSIC 和旧反向追踪实现可以复用，未新增依赖。本次查看的 999 份历史计时记录中，连续优化占约 98.47%，MUSIC 主要矩阵计算阶段占约 0.38%。这些比例对应被检查的那批运行，并非所有配置的固定比例。

当前真实观测的 MUSIC 采用同样的自动选维设置：CPU 中位约 1.670 秒，V100 中位约 0.132 秒，约 12.63 倍；谱的相对差异约 `1.40e-8`。这是单阶段结果，不能写成整个定位流程加速 12 倍。复用墙查询表后，同一个真实观测的小规模连续求解中位耗时由 13.24 秒降为 12.21 秒，减少约 7.8%，完整输出逐项一致。预算与计时范围见 GPU 报告。

`MC_COMPUTE_BACKEND=cuda` 现可直接启用 MUSIC GPU 计算；留空继续沿用配置。它与射线追踪的计算设备设置分开。连续求解仍包含逐条传播路线检查、匹配、分支和小规模有界求解，尚未完成批量 GPU 改写及等价验证，因此本次保留 CPU 实现。

以下命令用于之后的小批量检查，本次没有执行这批新实验。工作目录和解释器已写入脚本，参数集中在顶部；物理 GPU 0 暴露给子进程后，对应 `compute.device_id=0`。

```bash
MC_COMPUTE_BACKEND=cuda MC_GPU_ID=0 MC_WORKERS=2 MC_SAMPLE_COUNT=3 bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh start /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_gpu_manual_01
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_gpu_manual_01/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_gpu_manual_01
bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_gpu_manual_01
```

日志索引为上面的绝对路径，实际日志和运行记录位于该输出目录的 `run_records/时间_编号/`，具体路径由启动器打印并写入 `latest_run.txt`。查看日志时按 `Ctrl+C` 只退出查看。停止命令会先核对进程身份，再停止整项任务。新代码、设备或参数与原冻结记录不一致时，应使用新输出目录。

## 空间整理与保留边界

开始时 `/data` 可用约 242.85 GiB；本仓库约 75.68 GiB，其中输出约 75.31 GiB。较大的另两个类别是旧候选／代表的 JSON，以及谱数组。它们被历史清单和报告引用，本次未按文件名或修改时间删除。

四份历史逐样本报告共有 11,985 张 SVG，原始文件约 24.58 GiB，每张都有原位 PNG 和 PDF。处理顺序是：核对原报告清单和源 SHA-256 → 无损压缩 → 全量解压比对 SHA-256 → 同步写盘 → 再核对源文件身份 → 删除已归档的原 SVG。归档包含恢复原路径所需清单；PNG、PDF、CSI、场景、真值、数值结果、配置和失败记录保留。

其中独立旧报告的 2065 个上游来源在本次清理前已经不存在；报告里复制的数据可能是唯一留存，因此不能整目录删除。归档 SVG 后，原报告的完整图表清单校验需要先恢复 SVG；这不修复原本已经缺失的上游来源。

归档任务已结束，退出码 0；归档本身净释放 **21.90 GiB**，计入本次清单和日志等开销后，仓库净减少约 **21.83 GiB**。`/data` 可用空间约 **264.68 GiB**，本仓库约从 **75.68 GiB 降到 53.85 GiB**。实际字节数、归档路径和恢复命令见 [磁盘审计最终记录](workspace_storage_audit_20260921.md)。四份报告根目录另有 `SVG_ARCHIVE_NOTICE.md` 指明归档位置。

## 检查和复现

- 初始全量检查：1045 通过、23 跳过，退出码 0。日志：[baseline_tests/task.log](../outputs/workspace_maintenance_20260921/baseline_tests/task.log)。
- 最终全量检查：**1059 通过、23 跳过、0 失败，无警告，退出码 0**；耗时 82.48 秒。[final_tests_v2/task.log](../outputs/workspace_maintenance_20260921/final_tests_v2/task.log)，结束状态：[completion.json](../outputs/workspace_maintenance_20260921/final_tests_v2/completion.json)。跳过的检查不计作通过，真实 CUDA 检查单列在 GPU 报告中。
- 极短墙修复后，相关 79 项检查全部通过，无警告；Shell 语法和补丁格式检查通过。
- 后台对照入口：3 个样本 × 3 组，刻意使用 10 秒时限，9 项均正确记录 `timeout`、单线程和退出记录；用于验证启动、保存和限时，不能当作定位精度证据。记录：[arm_runner_smoke](../outputs/workspace_maintenance_20260921/arm_runner_smoke)。
- 同一入口的 `ARM_TASK=analyze` 已完成只读汇总，退出码 0；产物：[analysis.md](../outputs/workspace_maintenance_20260921/arm_analysis_smoke/analysis/analysis.md)。
- 无损归档小样本检查覆盖恢复一致性、拒绝覆盖、源指纹不符不删除及符号链接／路径逃逸拒绝；实际归档还执行全量源与解压内容校验。
- 未跑修复后的整批 1000 点实验，也未重新标定幅值加权的观测误差尺度。实现检查通过不代表新的精度、成功率或置信区间已经获得科学验证。

重跑实现检查时用新的目录；所有命令可在重新登录后直接执行：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_tests.sh start /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_tests_manual_01
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_tests_manual_01/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_tests.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_tests_manual_01
bash /data/zhujun/differt_projects/time-bias-correct/run_tests.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_tests_manual_01
```

工作目录固定为本仓库，解释器为 `/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python`。实际测试进程、命令、开始时间、结束时间和退出码直接保存在上面输出目录。

## 代码风格和后续边界

按 Ponytail 原则复用现有配置加载器、墙体查询缓存、原子 JSON 写入和 `detached_task.py`；删除对照脚本的重复递归合并逻辑与逐结果线性查找，没有添加第三方依赖或新的执行框架。历史分支仍有用途，不因文件长或年代旧就删除；后台管理仍复用一个监督程序，未大规模改写历史启动器。

开始审查前的源代码、脚本、测试、配置和文档快照为 [source_before_review.tar.gz](../outputs/workspace_maintenance_20260921/source_before_review.tar.gz)，当时的 Git 差异与状态也保存在同目录。本次变更与用户原改动可通过 [review_changes.patch](../outputs/workspace_maintenance_20260921/review_changes.patch) 和 [文件清单](../outputs/workspace_maintenance_20260921/review_changed_files.json) 区分。本次没有提交或重置用户已有改动。
