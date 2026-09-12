# 绕射边界实验：执行与结果说明

入口为 `/data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh`。

## 运行

所有持续任务默认脱离终端运行，关闭终端或 SSH 断线不会停止任务。每次启动必须填写允许使用的 **物理 GPU 编号**；不填写就拒绝启动。下面的 `2,3,4,5,6,7` 是本次授权示例，之后请按当时的空闲情况填写。脚本通过 `nvidia-smi` 将编号转换为唯一编号，限制全部子进程，并在运行目录保存 `gpu_allocation.json`。首次执行：

```bash
BOUNDARY_GPU_IDS=2,3,4,5,6,7 \
BOUNDARY_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2 \
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh
```

默认先做 **30 UE × 5 份独立噪声 × 2 种策略，共 300 次定位请求**。每次启动都会打印实际 PID、日志绝对路径、运行记录及管理命令。程序按顺序完成覆盖采样、固定信道、CSI 生成、预热、两种策略成对定位和汇总报告。前两个阶段使用的 RT 进程退出后才开始定位计时。

预跑结束、检查报告后，在**同一输出目录**启动正式实验：

```bash
BOUNDARY_PHASE=formal \
BOUNDARY_GPU_IDS=2,3,4,5,6,7 \
BOUNDARY_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2 \
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh
```

正式实验另取 **300 UE × 5 份噪声 × 2 种策略，共 3000 次定位请求**。正式点和预跑点使用不同随机流；正式预热使用预跑输入。代码、公开配置、环境版本和被引用的公开场景文件均有指纹。正式阶段先保存 `formal_parameters_frozen.json`。本版沿用预跑参数；如需根据预跑调参，修改配置后另开输出目录重新预跑，避免旧、新参数混入同一统计。

配置集中在：

- `/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_experiment.yaml`：UE 数量、噪声次数、随机种子、合法放置区域、单次请求预算、预热及计时快照。
- `/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_generation.yaml`：场景、BS、OFDM、MUSIC、反向追踪和聚类求解参数。当前默认 CUDA、单 CPU 线程。`compute.device_id=0` 表示本次选择列表中的第一张卡，例如选择 2～7 时表示物理 GPU 2；不要将物理编号直接填入程序配置。GPU 数量是可用范围，当前定位仍按请求顺序计时，不同时铺满所有卡。绕射几何部分使用 CPU 批量计算，两种策略各有独立进程和缓存。
- 启动脚本顶部参数区：项目目录、解释器、输入配置、输出目录、实验阶段、运行模式及线程数。

## 修复后复用已生成的 30 UE / 150 份 CSI

原失败结果保留在 `outputs/diffraction_boundary_v1`。新目录引用同一组冻结点、几何信道、噪声 CSI 和随机种子；启动前核对原文件摘要和生成参数，不继承旧定位结果。`prepared_reuse.json` 记录来源、已核对文件及旧/新计算配置。源目录在新实验期间必须保留。

```bash
BOUNDARY_GPU_IDS=2,3,4,5,6,7 \
BOUNDARY_SOURCE_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v1 \
BOUNDARY_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2 \
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh
```

## 日志、状态和停止

以下命令在重新连接服务器后仍能直接执行：

```bash
# 实时查看最新一次启动的日志：内部使用 tail -n 100 -F，并打印实际绝对路径。
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh log /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2

# 查看实际任务 PID、是否仍在运行，以及最终退出码。
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2

# 核对进程身份后，向整组任务及其子进程发送停止信号。
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2
```

查看日志时按 **Ctrl+C 只退出查看，不停止后台任务**。`latest_run.txt` 保存最新运行记录目录的绝对路径；每次独立运行位于 `run_records/<时间及进程号>/`，内有 `task.log`、`task.json`、`task.pid`、`completion.json`。成功取决于退出码，不能仅凭进程消失判为成功。同一输出目录不允许并发启动。服务器重启不会自动恢复任务。管理命令不要求再次提供 GPU 编号。

中断后，用原来的阶段、GPU 设置和输出目录重新启动。若更换代码、配置或 GPU，使用新目录并复用原观测，避免将不同运行条件的时间混在一起。已记录的定位请求（包括失败和超时）保留，不重新挑选 UE；尚未记录的请求在新的尝试目录中执行。RT 技术错误导致的“覆盖未知”会重试同一提案。未完成的目录保留。代码或参数已改变则拒绝续跑，要求新目录。

## 采样和公平对照

在固定的 Munich 公开地图范围 `[-100,160] × [-80,180]` 米内，均匀随机提出 XY 位置。高度固定为 1.5 米。默认放置区是原始网格顶面投影以外的露天区域，另留 0.5 米墙边距、2 米 BS 距离；它也会排除桥下、顶棚下等有顶区域。可通过公开多边形改变放置区，但必须在实验前固定。

RT 仅保留当前系统支持的二维、阵列正面、最多两次反射和一次绕射的有限非零路径。至少一条这样的路径即取得信号资格；不会要求两条路径、足够多的 MUSIC 峰或定位成功。所有提案、拒绝原因、信道分类和覆盖计算耗时都保存在 `coverage_proposals.jsonl`。技术错误单列为未知并终止本次采样，不能当作无信号跳过。

先固定全部合格位置和几何信道，再生成每个 UE 的 5 份独立噪声观测。本版是每个 UE 固定目标 **35 dB SNR** 的算法边界实验，**没有接收机灵敏度门限或固定噪声功率覆盖判定**。因此结论应描述为“当前 RT 预算和传播模型找到信号的露天范围内、给定 SNR 下的结果”。RT 未找到路径也不等于证明现实中没有信号。

两组都使用含绕射的同一份 CSI、同一 MC 随机种子、同一聚类结果、同一求解参数。`single` 对每个绕射簇选一个真实成员；`coverage` 按既有覆盖距离选择多个真实成员，并检查整个合法偏差区间。一个簇的多个代表仍互斥，不能增加来源峰的观测权重。两组都完整重做 MUSIC 和后续在线步骤，各用独立的常驻进程，执行顺序交替平衡，不共享对方刚计算的谱或几何缓存。预热失败后保留独立失败记录，重建两组进程并继续真实请求，不再对困难输入无限重复预热。实际请求失败后也会记录并继续下一个配对。

## 用时

主表对应 13 个步骤：协方差、特征分解、粗 MUSIC 谱、粗谱找峰、局部细谱、细谱找峰去重、特征采样、反向追踪、DBSCAN、代表点选择、代表轨迹、联合求解、在线几何检查。反向追踪和求解另有细分子项。

- `localization_seconds`：CSI 和公开地图就绪到位置输出；提前失败则截至失败，超时则截至请求超时边界。
- `checked_seconds`：同一起点到在线几何检查结束；失败请求截至失败边界。
- `processing_seconds`：在线入口到返回，包含输入读取、检查和结果发布；硬超时使用父进程实际等待时长并标明其口径。
- 独立的预热、进程启动、离线 RT、加噪声、真值评估和绘图不算在线定位用时。

请求会标明首次冷启动、已预热或进程内复用；`latency_by_execution_state` 分别统计这些状态。中断恢复时只剩半对且缓存状态不对应的记录不算成对时间差，精度和主时间表仍保留。公共几何缓存首次构建耗时计入实际请求，不当作免费步骤。每个阶段记录包含子步骤的耗时及扣除子步骤后的自身耗时。主图只用 13 个主阶段，不能再次加上其子项。未执行、执行失败、超时中断分别标记，缺失耗时不是零。主表列出统计数量、平均值、分位数，以及所有已测请求和有数值输出请求两种统计范围。

NumPy 用墙钟计时；CuPy 在相关阶段边界等待设备完成。默认将计时现场原子写入文件，以便硬超时后检查最后阶段。该开销列在 `snapshot_write_s`，同时保留在实际总用时内。两组开关一致；多代表增加阶段次数也可能增加这一开销。关闭 `timing_snapshots` 可减少写盘干扰，但硬超时会失去未返回的细分计时。本版输入起点是已有 CSI，尚未测量 OFDM 波形接收、同步、FFT 或信道估计前端。

## 精度和产物

每份 CSI 独立计算 `||估计位置 - 真值位置||₂`，单位米，固定高度下即 XY 欧式距离。先得到每次误差，再汇总 UE 的 5 次重复，不能先平均位置再算误差。所有有限位置输出都保留误差，包括正向检查未通过的结果；没有输出的请求保留原因，以全部计划请求为分母统计输出率和误差不超过 0.5/1/2/5 米的比例。

每个阶段目录 `pilot/`、`formal/` 保存：

| 文件或目录 | 内容 |
| --- | --- |
| `coverage_proposals.jsonl` | 全部位置提案、覆盖结果及原因 |
| `frozen_points.json`、`plan.json` | 固定 UE、信道分类、随机种子、共享观测路径 |
| `channel_setups/`、`frozen_channels/` | 公开网格、材料参数、场景指纹、原始信道 |
| `observations/` | 独立噪声 CSI、生成清单、分开保存的真值 |
| `benchmark_attempts/` | 每次启动、预热、每份 CSI 两种策略的完整定位产物和计时现场 |
| `trials.jsonl` | 每项请求结果，失败和超时均保留 |
| `report/trials.csv`、`stages.csv`、`per_ue.csv` | 单次、分阶段、每 UE 明细 |
| `report/summary.json` | 完整统计及统计范围 |
| `report/summary/precision.csv`、`timing.csv`、`paired.csv` | 误差、用时、成对差值 |
| `report/summary/*_by_channel.csv` | 按是否含绕射、直射/反射可达/仅绕射可达分类，查看绕射部分的变化 |
| `cohort_completion.json` | 全部计划请求是否已经有记录；不等同于每次都定位成功 |

报告还生成误差分布、达到误差门限的比例、各阶段时间、空间误差/用时图，以及代表数量与用时的关系图。分类仅用于解释原均匀样本，不重新筛点或改变总体权重。

## 代码自检

```bash
BOUNDARY_MODE=validate \
BOUNDARY_GPU_IDS=cpu \
BOUNDARY_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_code_validation \
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh
```

此模式先运行完整测试，再执行带屏风的合成场景小流程：预跑 2 UE、正式 1 UE，各 2 份噪声、2 种策略，并检查共享 CSI、共享随机种子和完整请求数量。合成场景使用声明的幅度规则，只验证实现，不能代替真实 Sionna 或正式定位精度实验。

## 本次实现验证记录（2026-09-11）

完整回归为 **676 项通过、1 项跳过**；跳过项依赖未提供的旧 Munich 固定样本。小规模流程的 8 次预跑请求及 4 次正式流程请求均完成，退出码 0，报表和图像生成完整。逐对核对了 CSI 指纹、MC 种子、全部初始点及聚类归属完全相同。第二个预跑 UE 的两份噪声输入中，单代表策略各保留 3 个绕射代表，多代表策略分别保留 21、22 个。

- 完整日志：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_framework_validation_20260911_02/run_records/20260911T074257_301437/task.log`
- 结束记录：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_framework_validation_20260911_02/run_records/20260911T074257_301437/completion.json`
- 验证汇总：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_framework_validation_20260911_02/run_records/20260911T074257_301437/validation_summary.json`
- 小规模报告：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_framework_validation_20260911_02/run_records/20260911T074257_301437/validation/experiment/pilot/report/summary.json`，同级正式流程目录为 `formal/report/summary.json`。
- 真实 Sionna 单点生成及数据契约检查：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_sionna_evidence_20260911T073812/README.md`。
- 较早一次验证被执行期间代码指纹变化拦截，已保留：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_framework_validation_20260911_01`。

以上只说明实现和统计流程已验证，尚未执行 30/300 UE 的真实 Sionna 实验，也不能据此判断多代表在真实场景中的精度收益。
