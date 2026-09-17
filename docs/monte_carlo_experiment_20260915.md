# 单场景 1000 个随机位置的连续定位实验

## 默认实验

- 场景：Sionna Munich；BS 平面坐标 `[20, 50]` m；BS/UE 高度 1.5 m。
- 采样区域：`x ∈ [-100,160]` m、`y ∈ [-80,180]` m。沿用当前露天定位条件，排除原始网格顶面覆盖区、距墙不足 0.5 m 和距 BS 不足 2 m 的位置。
- 最终接受 **1000 个至少包含一条有效路径的位置**。非法位置和零路径位置记录后补采；最多检查 100000 个提案，达不到数量则明确失败。
- 每个位置生成一份 CSI，12 根 BS 天线、512 个子载波、400 MHz 带宽、3.5 GHz 载频。
- 所有进入 CSI 的路径都共用该位置独立抽取的时间偏置，分布为 `Uniform(-50,50)` ns。
- CSI 加入复高斯噪声；每个位置根据自身信号功率设置噪声，目标信噪比 35 dB。
- RT 在 GPU 0 上运行；定位默认使用 4 个 CPU 进程，每进程一个数学库线程。每个位置的定位请求预算为 1800 秒。

本次“蒙特卡罗”指随机位置、随机公共时间偏置和随机 CSI 噪声。定位仍走连续模型，不会重新引入代表点、空间聚类或 RANSAC。

## 六维数组的顺序

所有面向用户的路径类别按 **BS→UE** 排序：

```text
[直达, 一次反射, 两次反射, 单次绕射, 一次反射后绕射, 两次反射后绕射]
```

真实信号和路径数组内部方向为 **UE→BS**，所以对应的交互序列为：

```text
[空序列, R, RR, D, DR, DRR]
```

`scene.diffraction_position: last_from_bs` 同时控制 CSI 合成前的路径筛选和连续传播函数构造。`UE→反射→绕射→BS`、`UE→反射→绕射→反射→BS` 等不在这次模型中的路径会在合成 CSI 前排除。旧配置不写该字段时继续采用原有 `any` 规则。

“有效路径”还要求符合当前二维高度、阵列正面接收角度、反射/绕射次数以及有限非零系数条件。零路径表示在声明的 RT 射线预算和这些条件下没有找到有效路径，不等于证明物理场景中不存在任何传播路径。

## 启动与管理

工作目录：`/data/zhujun/differt_projects/time-bias-correct`。

解释器：`/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python`。

默认启动命令：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh start
```

默认输出目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01`。

实时日志：

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01/task.log
```

查看状态：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01
```

停止整项任务：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01
```

`Ctrl+C` 只退出日志查看，不会停止后台任务。任务由 `nohup` 和 `setsid` 启动，脱离当前终端；这不代表服务器重启后自动恢复。

每次启动的实际日志、进程身份、启动命令、起止时间和退出码保存在输出目录的 `run_records/时间戳_启动进程号/` 中。根目录 `task.log` 是最新实际日志的链接，`latest_run.txt` 保存对应运行记录目录。停止前会核对实际进程身份并向整个任务进程组发送停止信号。

## 修改参数和续跑

启动器参数区可修改 Python、配置文件、输出目录、GPU 编号、CPU 线程数，并可通过 `MC_SAMPLE_COUNT`、`MC_WORKERS` 覆盖配置中的样本数和定位进程数。信道、时间偏置、采样范围和求解预算集中在：

`/data/zhujun/differt_projects/time-bias-correct/configs/monte_carlo_continuous_munich.yaml`

例如另开一个输出目录、使用 GPU 1 和 8 个定位进程：

```bash
MC_GPU_ID=1 MC_WORKERS=8 bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_experiment.sh start /data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_02
```

中断后用**同样的参数和输出目录**再次执行启动命令，即可续跑：

- 已固定的采样点和已完成的 CSI 不重新抽取。
- 已完成的定位结果和失败结果不重新运行。
- 被中断但尚未提交结果的样本保留旧现场，在新的尝试目录继续。
- RT 技术错误保留同一个提案，续跑时重试，不能当作无路径位置换掉。
- 配置、代码或冻结的环境记录变化时拒绝混入原实验，应改用新的输出目录。

`MC_PREPARE_ONLY=1` 只固定采样并生成 CSI。之后去掉该选项、保持其他参数不变，可以继续定位。

## 输出和统计口径

以默认输出目录为根：

| 路径 | 内容 |
|---|---|
| `experiment.json` | 冻结参数、代码指纹和运行环境 |
| `plan.json` | 所有接受样本、位置、随机种子、偏置和六维真实路径数量；仅离线端读取 |
| `sampling/proposals/` | 包括无路径与非法点在内的每个提案记录 |
| `sampling/channels/` | 已保存的 RT 路径、掩码和干净几何信道 |
| `progress.json` | 当前阶段和已完成数量 |
| `samples/SAMPLE_000001/observation.json` | 该样本 CSI、地图、真值和来源清单的文件指纹 |
| `samples/SAMPLE_000001/observation_attempts/` | CSI 生成记录；`data/online/measurement.npz` 为定位输入，`data/truth/` 仅供评估 |
| `samples/SAMPLE_000001/localization_attempts/` | 连续定位产物、逐样本 `online.log`、计时、求解结果和事后评估 |
| `samples/SAMPLE_000001/result.json` | 该样本最终状态、误差、路径数量及指向原始记录的指纹 |
| `report/samples.csv` | 全部接受样本的逐行结果，未完成、失败、超时也保留 |
| `report/summary.json` | 成功率、位置/偏置误差、六维路径数量及单路径/多路径子组 |
| `report/summary.md` | 可直接阅读的实验汇总 |
| `report/error_cdf.png`、`report/error_cdf.pdf` | 成功样本的位置误差和时间偏置误差累计分布 |

路径数量分别统计：

- `rt_path_type_counts`：所有接受样本实际进入 CSI 的六类路径总数。
- `samples_containing_path_type`：至少含有对应路径类别的样本数。
- `selected_path_type_counts_on_success`：成功定位时最终选中的六类路径总数。
- 每个样本的 `path_type_counts` 和 `selected_path_type_counts` 均为固定六维数组。

“两条以上”按 **至少两条** 统计，字段为 `samples_with_at_least_two_paths`；另提供严格超过两条的 `samples_with_more_than_two_paths`。

位置误差使用平面欧氏距离，单位米；时间偏置误差使用绝对误差，单位 ns，CSV 另外保留带符号误差。均输出均值、中位数、均方根误差、P90、P95 和最大值。

误差分布只包含成功输出，成功率分母始终包含该组全部接受样本，不能通过删除失败样本抬高成功率。单路径、不可区分观测、约束不足、歧义和超时不触发补采。`success` 表示满足现有选中路径的接受条件，仍受有限传播函数和初值搜索预算限制。

CSI 中存在负观测时延时，MUSIC 允许在负时延范围搜索；几何传播时延仍保持非负。默认搜索窗 `[-80,1150]` ns 小于当前子载波对应的 1280 ns 周期；超出该窗口的较长路径仍可能发生时延混叠，不利用真值把这些样本剔除。

每个在线子进程只收到公开算法/无线配置、BS 信息、地图、带噪 CSI 及来源校验清单。UE 真值、注入偏置、噪声种子、目标 SNR 和真实路径数量不进入在线求解请求；父进程在求解结果提交后才加载真值计算误差。

完整遍历结束后，正常的定位失败或超时仍属于实验结果。RT、文件、进程或其他技术异常会保留现场；RT 错误停止采样等待重试，逐样本定位技术异常在其余样本完成后使整项任务返回非零退出码。

## 代码检查

以下脚本在后台运行 CPU 测试，包括小型场景的完整 CSI→MUSIC→连续定位→评估→续跑流程：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_monte_carlo_check.sh start
```

启动后输出其独立日志、状态和停止命令。小规模检查不替代正式 1000 点实验。

### 本次验证记录

- 最终 CPU 检查：339 项，337 项通过、2 项跳过、0 项失败。覆盖六类路径过滤、CSI 噪声重建、负观测时延经过完整定位流程、失败分母、RT 错误重试、超时结果保留和中断恢复。
- 测试记录：`/data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_check_20260915_04/run_records/20260915T134909_689793/`，其中 `tests.xml` 为测试明细。
- 真实 Munich 检查：检查 8 个提案，6 个属于不可放置区域，接受 2 个点；两点都完成 96 个初值的连续求解，整项任务退出码为 0。
- 真实检查结果：`/data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_smoke_20260915_01/report/summary.json`。
- 两点位置误差分别为 `0.0097668637 m`、`0.0567373023 m`；时间偏置绝对误差分别为 `0.0158985623 ns`、`0.1301923531 ns`。进入 CSI 的总路径数组为 `[2,5,3,0,0,0]`，所以这两点真实场景检查没有验证实际绕射定位精度；绕射位置规则另由包含绕射的测试检查。
- 真实场景检查启动后，修正了一处已完成评估文件的路径类型转换；最终续跑测试覆盖该修正。真实检查目录保留启动时的代码指纹，不改写原始记录。
- 正式 1000 点任务尚未启动。正式输出目录与上述所有检查目录分开。
