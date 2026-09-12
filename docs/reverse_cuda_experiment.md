# CUDA 绕射反向追踪实验

第一版仅增加独立后端和验证脚本，现有预跑的默认执行入口保持原样。公开地图、BS、同一份 CSI 提取的 MUSIC 采样是反向输入；真值只在完整定位结束后用于误差评价。

## 实现范围

- `src/time_bias_localization/reverse_cuda.py` 提供 `generate_initial_candidate_points_cuda`，参数和返回的候选格式与原入口兼容，另有 `device_id` 和 `job_chunk_size` 参数。
- `generate_diffraction_points_cuda` 单独生成绕射候选；`CudaWallGeometry.nearest_batch` 支持批量墙面求交。
- CUDA 负责局部阴影判断、最近墙面求交、地图边界检查、剩余距离定位、绕射后的继续反射。采用双精度，关闭浮点乘加融合，保留原几何容差和等距离时的墙编号选择规则。
- CPU 负责公共 BS→边缘路径表、纯直达/反射分支、样本与公共路径的角度匹配、候选对象组装。公共路径表继续使用现有缓存；不能将首次 CPU 建表时间当作已经由 GPU 消除。
- 地图和公共路径的数值数组常驻指定 GPU；最多缓存两个场景/BS/反射上限/设备组合，临时数组按分支数分块处理。默认每块 8192 条分支。
- 输出保留来源峰、原采样编号、传播顺序、反射/绕射交互点、权重、参考偏差、末段方向及有效长度。聚类和求解器无需改写。

## 执行脚本

绝对路径：`/data/zhujun/differt_projects/time-bias-correct/run_reverse_cuda_experiment.sh`。默认通过 `nohup setsid` 后台执行，标准输入接到 `/dev/null`，每次新建运行目录。

参数集中在脚本顶部，也可用同名环境变量覆盖：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `RT_CUDA_GPU_ID` | 必填，nvidia-smi 物理卡编号；本实验一次使用一张卡 | 无 |
| `RT_CUDA_MODE` | `tests` 几何测试；`benchmark` 冻结观测对照 | `benchmark` |
| `RT_CUDA_INPUT_ROOT` | 原冻结实验根目录 | `outputs/diffraction_boundary_v1` 的绝对路径 |
| `RT_CUDA_REUSE_ROOT` | 可选，已有完整反向计时的 `artifacts` 目录；核对算法、输入和完整性后，继续完整定位验证 | 无 |
| `RT_CUDA_OUTPUT_ROOT` | 新的独立运行目录 | 带 UTC 时间和启动器编号的绝对路径 |
| `RT_CUDA_UE_IDS` | 固定 UE 编号，逗号分隔 | 0001、0002、0006、0017、0022、0030 |
| `RT_CUDA_NOISE_REPEATS` | 每个 UE 使用前几份原噪声 CSI | 2 |
| `RT_CUDA_TIMING_REPEATS` | 每份观测重复 CPU/CUDA 对照次数 | 3 |
| `RT_CUDA_PIPELINE_CASES` | 前几份观测额外跑完整定位，两种代表策略均核对 | 2 |
| `RT_CUDA_CPU_THREADS` | CPU 数值库线程上限 | 1 |
| `RT_CUDA_TIMEOUT_SECONDS` | 整个对照实验超时 | 2400 秒 |

脚本将物理 GPU 编号解析成唯一编号，设置 `CUDA_VISIBLE_DEVICES`，在进程内使用逻辑设备 0。没有显式 GPU 参数会报错，日志、状态和停止命令不需要 GPU 参数。

下面是一组完整的可复制命令。示例目录若已存在，请更换末尾名称；脚本拒绝覆盖旧结果。

```bash
# 启动。物理 GPU 3 是本次用户授权 3～7 中选取的一张。
RT_CUDA_GPU_ID=3 \
RT_CUDA_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_manual_01 \
bash /data/zhujun/differt_projects/time-bias-correct/run_reverse_cuda_experiment.sh

# 实时日志。Ctrl+C 只退出查看，不会停止后台任务。
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_manual_01/task.log

# 查询实际 PID、身份和最终退出码。
bash /data/zhujun/differt_projects/time-bias-correct/run_reverse_cuda_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_manual_01

# 脚本核对进程身份后，停止任务进程组。
bash /data/zhujun/differt_projects/time-bias-correct/run_reverse_cuda_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_manual_01
```

## 实验口径和输出

CPU/CUDA 使用同一组 MUSIC 采样、同一地图和相同传播上限。先分别执行一次并核对，然后交替执行顺序，重复记录缓存复用后的墙钟耗时。每次计时等待 GPU 完成，包含数据传输、候选对象组装和纯反射分支。包含子步骤的时间不能再次相加。

完整定位的额外检查用于确认候选、聚类和最终位置/公共偏差的兼容性。它的计时只作诊断，不用作独立进程公平预热后的端到端提速结论。

运行目录保存：

- `task.log`、`task.json`、`task.pid`、`completion.json`：日志、实际任务身份、起止时间及退出码。
- `gpu_allocation.json`：本次物理卡与进程逻辑编号对应关系。
- `artifacts/protocol.json`、`artifacts/source_snapshot/`：预先固定的观测清单、输入摘要、实际代码快照。
- `artifacts/setup.json`：公共路径首次 CPU 构建、CUDA 编译及墙面上传时间。
- `artifacts/cases/*/inputs.json`：冻结的 MUSIC 采样和在线输入出处。
- `artifacts/cases/*/cpu_candidates.json`、`cuda_candidates.json`：两套后端的完整候选字段。
- `artifacts/timings.jsonl`：逐次计时；失败前已完成记录保留。
- `artifacts/pipeline/`：两种代表策略的 CPU/CUDA 完整定位产物与独立误差比较。
- `artifacts/summary.json`：全部核对通过后的最终汇总。

候选来源、编号和传播顺序要求完全相同；浮点字段核对使用绝对容差 1e-8、相对容差 1e-12，并保存实际最大差值。最终位置容差 1e-6 米、公共偏差容差 1e-14 秒。该固定小批次用于实现一致性和性能验证，不代表全体 UE 的定位精度结论。

## 2026-09-11 实测结果

用户授权物理 GPU 3～7；性能实验使用 GPU 3，额外几何及报告回归使用 GPU 4。两者均通过 GPU 唯一编号限定设备，未启动其他卡上的计算。

6 个 UE × 2 份原噪声 CSI = 12 份输入，每份进行 3 次交替顺序 CPU/CUDA 对照，共 36 组。以下为已复用公共路径缓存时的单次平均墙钟耗时，包含 GPU 数据传输和 CPU 候选对象组装：

| 阶段 | 原 CPU 实现 | CUDA 实验版 |
|---|---:|---:|
| 纯直达/反射分支（两者仍使用 CPU） | 0.106069 秒 | 0.106298 秒 |
| 绕射候选部分（包含其子步骤） | 4.859785 秒 | 0.024235 秒 |
| 完整反向候选阶段 | 4.969489 秒 | 0.134241 秒 |

完整反向阶段的均值比为 **37.019 倍**。这不是整个定位系统的端到端提速倍数。首份观测之前的公共路径 CPU 建表耗时 **56.352 秒**，该成本仍然存在。CUDA 首次编译及墙面上传实测 0.260 秒；另有首次分支调用的辅助算子编译，首调用记录保留在各观测的 `initial_check.json` 中，不混入重复计时均值。

12 份输入的候选数量、来源编号、权重和传播顺序全部一致；候选坐标最大绝对差值为 **4.2633e-14 米**，其他数值字段的最大绝对差值不超过 7.1055e-14。

额外核对了 UE0001 的两份噪声 CSI，每份分别执行单代表/多代表、CPU/CUDA，共 **8 次完整定位**。两种后端的最终位置差和公共偏差差均为 **0**；两份 CSI 的欧式误差分别约 **0.117857 米、0.049684 米**。这些额外请求仅验证后端替换兼容性，不代表总体定位精度。

最终相关测试 **104 项通过、1 项跳过**。跳过的是未找到本机旧固定样本的已有回放测试；新增 CUDA 测试实际执行，包含随机交叉墙、角点并列、近乎平行、端点阈值、偏差变化、不同批大小和绕射后反射。已有极短墙测试产生的 CPU 警告保留在日志中。

最终汇总及完整定位核对运行（已成功退出，退出码 0）：

`/data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095927_513640`

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095927_513640/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_reverse_cuda_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095927_513640
bash /data/zhujun/differt_projects/time-bias-correct/run_reverse_cuda_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095927_513640
```

上述 `stop` 对已结束的任务只展示完成记录。查看日志时 Ctrl+C 只结束查看。

主要证据的绝对路径：

- 最终结果：`/data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095927_513640/artifacts/summary.json`
- 完整定位比较：`/data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095927_513640/artifacts/pipeline_comparisons.json`
- 原始反向计时：`/data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T095202_502667/artifacts/timings.jsonl`
- 最终测试：`/data/zhujun/differt_projects/time-bias-correct/outputs/reverse_cuda_20260911T100027_515138/task.log`

失败现场也保留。`reverse_cuda_20260911T094730_495893` 是最初 CUDA 编译时使用宿主 `math.h` 引起的失败，已改为使用 NVRTC 自带设备数学函数。`reverse_cuda_20260911T095202_502667` 已完成全部 36 组反向计时，但在完整定位比较写 JSON 时遇到 NumPy 数组序列化错误。最终运行修复序列化后，核对相同算法代码和相同输入，复用这些完整计时并重做完整定位核对；`artifacts/reverse_reuse.json` 记录来源和复制文件摘要。未把失败任务标记为成功，也没有覆盖它们的日志。
