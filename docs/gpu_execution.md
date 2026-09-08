# GPU 批量实验

本次调整把 MUSIC 的矩阵运算和谱搜索交给 GPU，把不同 UE 分给独立进程执行。UE 采样计划、噪声种子、定位步骤、每个 sample 的中间输出，以及独立评估的真值隔离规则继续保留。

## 运行命令

工作目录固定为 `/data/zhujun/differt_projects/time-bias-correct`，默认解释器为该目录下的 `.sionna-venv/bin/python`。进入目录后，单卡运行：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_IDS=0 ./run_gpu_experiment.sh
```

两张卡并行，每张卡一个常驻进程：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_IDS=0,1 ./run_gpu_experiment.sh
```

`GPU_IDS` 使用 `nvidia-smi` 中的设备编号。每个进程只看见分给自己的那张卡，进程内部的 `device_id` 因此固定为 `0`。同一进程内，Sionna 信道生成和 MUSIC 使用同一张卡。所有指定的卡通过 CUDA 预检后，才开始分配 UE；无法使用 GPU 会明确报错。

脚本参数集中在文件顶部，也可通过同名环境变量覆盖：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `EXPERIMENT_CONFIG` | `configs/visualization_experiment_munich.yaml` | UE 数、噪声重复数、采样区域、采样种子；文件内部引用信道生成配置 |
| `GPU_IDS` | `0` | 使用哪些 GPU；GPU 模式的工作进程数由设备数决定 |
| `MUSIC_BATCH_SIZE` | `4` | 一次提交多少份独立 CSI 扰动，增加此值会增加显存需求 |
| `MUSIC_ANGLE_CHUNK_SIZE` | `32` | 每次计算多少个角度网格，控制导向矩阵临时占用 |
| `CPU_THREADS` | `1` | 每个工作进程允许的 CPU 数值计算线程数 |
| `EXPERIMENT_ROOT` | 自动生成的新目录 | 固定计划、UE 数据、定位结果和执行记录 |
| `REPORT_ROOT` | 实验目录下的 `step_report` | 完成计算后生成的分步图和汇总图；必须为新目录 |
| `PLAN_ONLY` | `0` | 设为 `1` 时只固定采样计划并生成已有信息的图 |
| `RESUME` | `0` | 设为 `1` 时使用 `EXPERIMENT_ROOT` 内的原计划 |

例如，修改 `EXPERIMENT_CONFIG` 指向的 YAML 中的 `ue_count` 和 `noise_repeats` 后，用新输出目录启动即可；计算设备设置不决定样本数量。`MUSIC_BATCH_SIZE` 也不是 UE 数或每点独立噪声重复数，它只控制定位内部扰动的分批运算。

原入口仍可使用 CPU：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
COMPUTE_BACKEND=numpy WORKERS=2 CPU_THREADS=1 ./run_visualization_experiment.sh
```

`WORKERS` 只控制 CPU 模式的 UE 并行进程数。CPU 模式的 MUSIC 保留原 NumPy 计算路径；GPU 模式才把内部 CSI 扰动分批提交给 GPU。Sionna 的运行设备独立记录，不能仅根据 MUSIC 为 CPU 就推断信道生成也使用 CPU。

以上 `.sh` 入口会在启动 Python 前设置 CPU 线程数。直接调用 Python 的 `run_experiment(..., workers=1, compute_backend='numpy')` 时，为保持已有调用行为，会沿用当前进程的线程设置；传入 `cpu_threads` 不会重新初始化已经导入的 NumPy。执行记录中的 `thread_limit_method` 会注明这一点。

新环境缺少 CuPy 时，可执行 `./setup_gpu_environment.sh` 安装并检查。安装脚本的解释器、CUDA 路径和 CuPy 版本同样集中在顶部。该脚本会修改指定的 Python 环境，不用于正常每轮实验启动。

## 计算怎样分工

```text
主进程：读取固定采样计划，持有实验目录锁
  ├─ GPU 0 常驻进程：UE001 → 后续空闲时领取的 UE
  │    生成固定信道 → 独立噪声重复 → 定位 → 独立评估
  └─ GPU 1 常驻进程：UE002 → 后续空闲时领取的 UE
       生成固定信道 → 独立噪声重复 → 定位 → 独立评估
全部 UE 任务结束 → 主进程读取结果并绘图
```

- `experiment.py`：一个 UE 是一个任务；同一 UE 的重复按固定种子依次执行，空闲进程继续领取下一个 UE。每个工作进程具有独立工作目录和 DeepMIMO 进程内设置。线程数和可见 GPU 在新解释器启动前写入环境，避免导入 NumPy 后再限制线程已经太晚。
- `compute.py`：可复用的 MUSIC 计算器。CUDA 模式使用 CuPy，以双精度计算协方差、特征分解及 MUSIC 谱；按批大小和角度块大小限制临时数组。相同搜索网格的导向向量会缓存，缓存有数量和内存上限。
- `music_stage.py`：仍按原来的 NumPy 随机序列生成各次 CSI 扰动，然后分批交给计算器。不同输入分别建立协方差，避免把不同 UE 或不同重复混成一次观测。同一进程在多个 UE 之间复用计算器和缓存。
- `pipeline.py`：保持原定位步骤和输出协议，通过 `config['compute']` 选择计算设备，并记录各阶段耗时。

仍在 CPU 上运行的部分包括：二维墙线提取、峰值筛选、候选轨迹生成、第一次聚类、候选组合和位置/偏差求解、文件写入与绘图。GPU 利用率因此会随阶段变化，不能期望从开始到结束始终满载。

## 输出和核对

```text
实验目录/
├── experiment_plan.json             原始采样计划，不被并行调度修改
├── generation_template.yaml
├── UE001/
│   ├── generation.yaml
│   ├── channel/                     此点固定信道
│   └── repeat_000/
│       ├── localization.yaml        本次公开定位参数，包含 compute
│       ├── data/                    输入 CSI 和独立评估真值
│       ├── localization/            定位结果与每步数值
│       ├── evaluation/              独立评估误差
│       └── attempt.json             本次终态、耗时、进程和 GPU 信息
├── execution_runs/本次执行编号/
│   ├── execution_config.json        设备、批大小、线程数、解释器、计划指纹
│   ├── status.json                  本次调度是否完成
│   ├── tasks/UE001.json             每个 UE 的执行状态及成功/失败次数
│   └── workers/worker_000/
│       ├── worker.json              PID、可见 GPU、实际设备、线程环境和工作目录
│       └── exit.json                进程退出码；启动/运行异常另存 failure.json
└── step_report/
    ├── samples/UE001/repeat_000/     00–08 步骤的图和数据
    └── summary/                     全部 sample 的误差表、Med、P90 和 CDF
```

Sionna 生成阶段在对应信道目录的 `provenance/generation_runtime.json` 记录实际 Mitsuba 计算模式与可见 GPU。请求 CUDA 而实际生成模式不是 `cuda_*` 时会报错。

每次定位的 `localization/localization_result.json` 中，`diagnostics.compute` 保存 MUSIC 的实际设备及批处理信息，`diagnostics.stage_timings_s` 保存输入检查、原始 MUSIC、扰动 MUSIC、峰配对、反向候选、第一次聚类、主求解及扰动求解的耗时。`attempt.json` 的 `localization_seconds` 记录整个定位调用的耗时，包括输出写入；它不包含信道生成、独立评估和后续绘图。首次 CUDA 初始化与后续缓存复用的耗时不同，比较性能时应分别记录。

`diagnostics.compute.counter_scope=worker_lifetime` 表示主计数属于整个常驻进程，`this_localization` 单独保存本次定位计算的 CSI 数、批次数和缓存命中次数。`stage_timings_s` 截止分布计算，不含之后的压缩、归档及文件发布；不能用各阶段之和代替完整调用耗时。

计算完成只表示各任务已产生终态。定位失败、评估失败及误差较大的结果都保留，统计表继续使用原先的失败分母规则；成功运行本身不等于定位精度已通过科学验证。

## 恢复和独立绘图

已有计划可以继续执行尚未产生终态的 UE；固定计划不会重新读取修改后的 UE 数、坐标或噪声种子。已有 `attempt.json` 的成功或失败重复都会跳过。发现没有终态却已有中间文件的重复时，会保留现场并标记中断失败，不覆盖重跑。需要重新实验时应新建输出目录。

GPU 不可用导致工作进程启动失败时，UE 任务尚未发布，可以修复环境后继续原计划。工作进程在执行中异常退出时，已经完成的结果保留，未完成项明确记录为 `worker_failed`，不会无限等待缺失的返回消息。

启动失败或用户中断时，本轮 `execution_runs` 中尚未开始的任务记为 `cancelled`，正在处理的记为 `interrupted`；不为未执行的样本生成 `attempt.json`。已写出的中间文件继续保留。

恢复时同时指定一个新的报告目录，避免覆盖已有报告：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
EXPERIMENT_ROOT=/绝对路径/已有实验目录 \
REPORT_ROOT=/绝对路径/新的报告目录 \
GPU_IDS=0,1 RESUME=1 ./run_gpu_experiment.sh
```

命令行模块不传 `--report-output` 时只执行计算。之后可独立绘图：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
EXPERIMENT_ROOT=/绝对路径/已有实验目录 \
REPORT_ROOT=/绝对路径/新的报告目录 \
./run_visualize_results.sh
```

绘图不在 GPU 工作进程内执行，也不会影响采样计划或修改定位结果。

## CPU/GPU 对照与已完成验证

对现有 CSI 运行完整定位对照，不重新生成信道：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_ID=0 EVALUATE=1 ./run_benchmark_gpu.sh
```

`INPUT_ROOT` 默认指向已有 Munich 30 UE 实验的 `UE001/repeat_000`，可在脚本顶部修改。`OUTPUT_ROOT` 默认自动生成新目录。CPU 与 GPU 使用同一输入、随机种子和双精度；先分别做一次单谱预热，再测整个定位调用。只有两次定位完成后，`EVALUATE=1` 才独立读取真实位置和偏差。默认 `EVALUATE=0` 只比较 CPU/GPU 输出。

2026-09-08 在本机 Tesla V100S-PCIE-32GB 上完成一次完整对照，结果存于
`outputs/gpu_rebuild_validation_20260908_full/benchmark.json` 和 `timings.csv`：

| 本次测量 | CPU | GPU |
|---|---:|---:|
| 完整定位，含结果写入 | 15.319 s | 4.011 s |
| 原始 MUSIC | 1.112 s | 0.130 s |
| 12 次扰动 MUSIC | 12.654 s | 2.288 s |
| 扰动路径和位置求解 | 1.272 s | 1.295 s |
| 独立评估的位置误差 | 0.066532 m | 0.066532 m |
| 独立评估的偏差误差 | −0.150389 ns | −0.150389 ns |

本输入的完整定位加速比为 3.82。原始峰索引、扰动峰配对、反向候选、第一次聚类、选中路径及每次扰动求解状态均一致；最终位置和偏差之差均为 0，MUSIC 谱的相对二范数差为约 `3.52e-10`。这不表示谱逐位相同。

CPU 和 GPU 的首次单谱预热分别用了 1.050 s 和 8.443 s，未计入表中的完整定位耗时；首次批量调用仍可能需要编译。这里比较的是重构后的 CPU 与 GPU，一次输入、一次计时，不用历史实验的耗时推算加速比，也不代表全部 sample 的平均表现。表中不含射线生成、独立评估及绘图。

另一份 UE003 输入也完成了逐步骤对照，CPU 为 15.062 s、GPU 为 2.234 s，最终位置与偏差之差同样为 0；记录在 `outputs/gpu_rebuild_validation_20260908_full_ue003/benchmark.json`。该轮发生在首次 GPU 编译缓存建立之后，预热分别为 1.043 s、1.333 s；不能把它与第一轮的差异全归因于 UE 本身。两份输入都通过一致性检查，但不足以估计全部位置上的平均加速比。

小规模多卡验证保留正式配置中的全部物理和定位参数，只减少 UE 数与独立噪声重复数：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
EXPERIMENT_CONFIG="${PWD}/configs/visualization_gpu_validation.yaml" \
GPU_IDS=0,1 ./run_gpu_experiment.sh
```

本次输出 `outputs/gpu_rebuild_validation_20260908_dual` 中，4 UE × 2 次噪声的 8 次定位和评估全部完成。两个 GPU 进程各处理了两个 UE；每个进程的 CSI 累计计数从 13、26 增加至 39、52，验证了跨重复与跨 UE 的复用。四份 `generation_runtime.json` 均记录 `cuda_ad_mono_polarized`。这用于检查并行执行和产物完整性，不作为多场景精度结论。

常驻进程再次导出 Sionna 场景时，物体自动名称可能改变。汇总绘图已改为比较墙编号、端点、顺序、范围与高度，忽略 `source_object` 说明名称；真实几何变化仍会被拒绝。每个样本仍使用自己的原始场景文件及文件指纹，没有改写定位输入。

双卡验证的完整图表位于 `step_report/`：每次定位的 9 个步骤均有数据，每次导出 33 张 PNG 及对应 PDF/SVG；全部 1701 个报告文件已核对指纹。汇总 Med 为 0.1201 m，P90 为 0.4130 m。此次绘图代码已另存于实验目录的 `validation_plotting_source/`；之后界面中的函数说明补充了 CPU/GPU 两个计算入口，保留已生成报告作为当时版本的证据。

最终 CPU 测试为 384 项通过、1 项 GPU 检查因该解释器没有 CuPy 而跳过；GPU 环境中 `test_compute.py` 和 `test_signal.py` 共 65 项通过、无跳过。另有两份真实 CSI 完整对照及上述双卡运行证据。
