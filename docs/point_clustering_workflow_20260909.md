# 先聚类初始点，再生成代表轨迹

更新日期：2026-09-09。流程标识：`music_point_clustering_v2`。定位清单版本：5。

本次修改落实“先对蒙特卡洛采样反向 RT 得到的位置点聚类，再为代表点生成轨迹”的顺序。
旧版 `music_spectrum_sampling_v1` 先生成全部候选轨迹、再比较轨迹；两者的聚类对象不同。
旧实验目录保持原样，重新绘制时也沿用各自真实执行的步骤，不能通过改图把旧结果变成新结果。

## 当前执行顺序

```text
一份已带噪的接收 CSI
  → 计算 MUSIC 谱，按 num_paths 上限选峰
  → 各峰附近采样连续角度和观测时延
  → 在统一的参考 bias 下反向 RT，得到地图上的初始候选点
  → 同来源峰、同反射墙顺序内，对初始点按位置距离聚类
  → 从每簇选一个真实成员作为代表点
  → 只为代表点建立位置随 bias 变化的轨迹
  → 联合估计一个公共 bias 和 UE 位置
  → 正向路径检查
  → 独立读取真值计算误差
```

`music.num_paths` 继续作为输出峰数上限，本次不更改其默认值 3。
定位内部继续不对 CSI 额外加噪，MUSIC 的全局谱、局部细谱和采样谱值复用一次计算结果。
每个 UE 的 5 次独立噪声仍属于实验中生成的 5 份接收数据，每份数据各执行一次以上流程。

## 初始点如何定义

未知 bias 时，同一个角度和观测时延不能直接确定唯一的 UE 位置。因此先统一使用一个
公开参考值 `localization.initial_reference_bias_s`，默认是 0 秒。每个样本的反向传播长度为：

```text
初始传播长度 = 光速 ×（采样观测时延 − 参考 bias）
```

反向 RT 从 BS 出发，按样本角度行进并做镜面反射；走完这段传播长度的位置就是初始点。
初始点保留来源峰、样本编号、实际反射墙及其顺序、反射位置和末段方向。
这一阶段不为所有样本构建位置随 bias 变化的轨迹。

参考 bias 是形成可比较点集的统一假设，既不读取真实偏差，也不固定最后求解的偏差。
例如参考值为 0、真实偏差为正时，初始点通常不是最终 UE 位置，这并非点生成出错。
参考值必须是有限数，并位于公开的 `[bias_min_s, bias_max_s]` 搜索范围内。
非正传播长度、参考位置无法形成允许反射次数内有效路径等情况会保留排除原因；
不会悄悄使用另外一个 bias 给这些样本补点。

## 聚类对象与代表轨迹

聚类只读取初始点的位置。在相同来源峰、相同反射墙顺序的组内，以 XY 欧式距离判断
哪些点足够接近，默认距离阈值为 1.5 米。每个簇选择实际存在的一个成员作为代表，
不使用可能落到障碍物里或失去路径解释的坐标平均值。

方向变化和完整 bias 区间内的轨迹差异不再作为这一步的聚类条件。
旧字段 `candidate_direction_radius_deg` 仍可读取，默认配置补齐时也保留它，
用于兼容已有配置快照；它不参与当前点聚类，调整它不会改变此步骤。

完成点聚类后，才为代表点构建固定反射结构下的位置关系：

```text
beta = 光速 × 时间 bias
代表位置(beta) = 初始代表点 −（beta − 参考 beta）× 末段单位方向
```

每条代表轨迹保留合法 bias 区间，求解器从不同来源峰的候选中选组合，在同一个 bias
下寻找位置一致的解。同一来源峰的多个采样结果仍是互斥解释，不被算作更多独立观测。

## 脚本与参数

复用已有 CSI 检查三份观测：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_point_clustering_check.sh
```

默认输入是 `outputs/spectrum_experiment_20260908T091621_2543248` 中的
UE001、UE010、UE006，各取 `repeat_000`。它只重放定位，不生成新信道和新噪声。
默认输出是独立的新目录 `outputs/point_clustering_check_时间_进程号`。

脚本顶部集中设置输入 `SOURCE_EXPERIMENT`、定位配置 `CONFIG_PATH`、输出 `OUTPUT_ROOT`，
以及 `UE_IDS`、`NOISE_REPEATS`、`SAMPLES_PER_PEAK`、`REFERENCE_BIAS_S` 和计算设备。
`REFERENCE_BIAS_S` 的单位是秒；检查脚本将它显式写入本次定位配置，默认覆盖为 0。
例如只重放 UE010，用相同参考 bias：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
UE_IDS=UE010 NOISE_REPEATS=1 REFERENCE_BIAS_S=0.0 DEVICE_ID=0 \
./run_point_clustering_check.sh
```

新建 30 UE × 5 次独立噪声实验：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_IDS=0,1,2,3,4,5,6,7 PLAN_ONLY=0 RESUME=0 \
./run_point_clustering_experiment.sh
```

该脚本默认输出 `outputs/point_clustering_experiment_时间_进程号`，默认报告在其 `step_report`。
`EXPERIMENT_CONFIG` 指向的 YAML 设置 UE 数、实验噪声次数和区域；该文件引用的
`generation_config` 的 `localization` 段设置参考 bias 和点聚类半径：

```yaml
localization:
  bias_min_s: -80.0e-9
  bias_max_s: 80.0e-9
  initial_reference_bias_s: 0.0
  candidate_cluster_radius_m: 1.5
```

`PLAN_ONLY=1` 只保存固定计划和已有信息的图。`RESUME=1` 只继续同一流程的原计划，
不会重新读入修改后的配置；修改参数后应使用新的实验目录。
旧 `run_spectrum_sampling_*.sh` 仍调用当前定位代码，脚本名不意味着能够复现旧版算法；
新实验建议使用上面的点聚类入口，以便从目录名区分不同批次。

## 每份观测的步骤图和原始数据

报告结构为 `step_report/samples/UE编号/repeat编号/`：

| 目录 | 实际内容 |
|---|---|
| `00_scene_truth` | 场景、BS、真实 UE 和实际参与 CSI 的真实路径，仅供独立展示 |
| `01_csi_input` | 本次接收 CSI 与公开参数 |
| `02_music` | MUSIC 谱和入选的粗网格峰 |
| `03_spectrum_sampling` | 各峰局部细谱、连续采样点、样本来源 |
| `04_initial_candidates` | 在统一参考 bias 下得到的全部有效初始位置点及排除原因 |
| `05_point_clustering` | 点簇、所有成员、选出的真实代表点及簇大小 |
| `06_representative_trajectories` | 只从代表点生成的合法 bias 轨迹 |
| `07_joint_solution` | 代表轨迹参与的共同 bias 和位置估计 |
| `08_forward_check` | 最终所选反射路径的可见性、几何及角度/时延检查 |
| `09_final_evaluation` | 同一个最终解的真实/估计坐标、位置和 bias 误差 |

04 和 05 展示位置点，06 才展示轨迹。轨迹表示 UE 候选位置随 bias 的变化，不是完整的
BS—反射墙—UE 传播折线；传播折线在真实路径和正向检查步骤展示。
03 图中的原始峰来自粗网格，背景来自细网格，两者最高点不必重合。

定位目录中的主要新文件为：

- `initial_candidates.json`：全部有效初始点、被排除样本及原因、参考值和统计。
- `representative_points.json`：每簇真实代表、成员及聚类说明。
- `representative_trajectories.json`：只为上述代表生成的轨迹及合法 bias 区间。

上述文件与 CSI、配置、结果和评估一起通过清单绑定。中途失败时保留已经完成步骤的
产物，报告明确区分失败和未执行步骤。总体 `summary` 继续保存定位误差 CDF、Med/P90
以及逐观测/逐 UE 的表，失败项不写成零误差，也不从计划总数中消失。

## 验证范围与尚存限制

2026-09-09 在 Tesla V100S 上使用原实验中三份已保存的 `repeat_000` 接收 CSI，
完成当前流程定位及独立评估。参考 bias 为 0 秒，每峰使用 128 次随机采样和一个原始峰，
未重新生成信道或接收噪声。结果记录于
[replay_results.json](../outputs/point_clustering_check_20260909_v1/replay_results.json)，
逐步图和汇总见
[逐步报告](../outputs/point_clustering_check_20260909_v1/step_report_readable/README.md)。

| UE | 有效初始点 | 聚类代表点 | 生成轨迹 | 位置误差（米） | bias 误差（ns） |
|---|---:|---:|---:|---:|---:|
| UE001 | 387 | 16 | 16 | 0.087314 | -0.053010 |
| UE010 | 360 | 20 | 20 | 0.038712 | +0.016037 |
| UE006 | 169 | 9 | 9 | 1.642396 | +2.626924 |

bias 误差为“估计减真实”。每份观测均只计算一次 MUSIC 子空间；表中轨迹数等于代表点数，
不是初始点数。UE006 仍明显弱于另外两例，不应只展示误差较小的样本。

另外对 UE010 的同一份 CSI、配置和随机种子执行完整 CPU/GPU 对照，
[benchmark.json](../outputs/point_clustering_cpu_cuda_20260909_v1/benchmark.json)
记录的 23 项检查全部通过，覆盖谱面样本、被排除样本、初始点、点簇、代表轨迹和最终解。
CPU/GPU 最终位置差为 **0 米**，bias 差为 **0 ns**；谱值有浮点计算差异并在容差内。
该对照只有一个输入，不把其中耗时比推广为全量实验加速结论。

完整回归测试为 **546 passed、2 skipped**（21.73 秒）；两项跳过是受限运行环境内无法访问
CUDA 所致，不是定位失败。随后在真实 GPU 上补跑对应计算与采样测试，结果为
**3 passed、73 deselected**（1.35 秒），覆盖此前跳过的两项。
配置专项的 54 项已包含在完整测试中，不额外相加，覆盖新字段的类型、有限性、范围端点、
默认值与真实 bias 隔离。两份新脚本的语法及参数传递检查通过。默认实验配置也实际执行了
计划创建检查：场景文件有效，计划为新流程的 30 UE × 5 份观测，峰上限为 3、参考 bias 为 0；
该检查只创建计划，没有执行 150 次定位。

参考 bias 会影响初始点落在哪条路径段、哪些样本有效以及哪些点被合并，这是当前方法
的明确取舍。只在参考位置靠近的两点，随着 bias 改变可能逐渐分开；新聚类不会提前
比较它们整条轨迹。后续可以在相同 CSI 上改变公开参考值、采样数、聚类半径做对照。

少量重放只验证代码和输出链路。新版 30 × 5 结果必须来自新版实际运行，不能把先前
`spectrum_experiment_20260908T091621_2543248` 的结果改标成新版统计。
几何检查通过也不等于已经完成真实 CSI 或论文层面的精度验证。
