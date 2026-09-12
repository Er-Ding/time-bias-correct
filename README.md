# 时间偏差校正的二维多径定位

默认定位主流程为 `music_fine_spectrum_dbscan_v3`，定位清单格式为第 6 版。
现已增加可选的一次绕射版本 `music_diffraction_cover_v4`（第 7 版清单）：
保留 DBSCAN，在绕射簇内按覆盖距离选多个实际代表，并检查整个合法时间偏差区间的覆盖。
实现范围、验证结果、参数和后台脚本见 [一次绕射与簇内多代表](docs/diffraction_v1.md)。

绕射超时修复、GPU 限制及原 CSI 恢复说明见 [修复记录](docs/boundary_repair_20260911.md)。每次启动必须明确填写 `BOUNDARY_GPU_IDS`，不会默认开放全部 GPU。

绕射边界实验已提供后台入口 `./run_boundary_experiment.sh`：先做 30 UE × 5 次噪声预跑，再做独立 300 UE × 5 次正式实验，逐份 CSI 对比单代表和多代表。采样规则、13 阶段计时、欧式误差、失败统计及完整启动/查看/停止命令见 [实验执行说明](docs/diffraction_boundary_experiment_usage.md)。

默认关闭绕射时的流程为：

```text
一份接收到的带噪 CSI → 一次协方差与特征分解 → 粗谱确定区域 → 局部细谱正式找峰
  → 在同一份局部细谱上连续采样角度和观测时延 → 参考 bias 下反向 RT 得到初始位置点
  → 同来源峰、同反射墙顺序内做 DBSCAN 点聚类，每簇选一个真实代表，单独记录离群点
  → 只为代表点生成随 bias 变化的轨迹 → 一次位置与公共 bias 联合求解
  → 正向路径检查 → 独立读取真值评估
```

定位内部不再给 CSI 额外加噪，也不再逐次扰动求解后平均。实验中每个 UE 的
5 次独立噪声重复仍保留，用于生成 5 份接收数据；每份接收数据分别执行上述流程。
实现、运行命令与验证结果见 [统一细谱与 DBSCAN 流程说明](docs/fine_spectrum_dbscan_workflow_20260909.md)。
先检查三份已有 CSI：`./run_fine_dbscan_check.sh`；全量新实验：`./run_fine_dbscan_experiment.sh`。
初始参考偏差 `localization.initial_reference_bias_s` 默认为 0 秒，是公开假设，不是实际 bias；
它只用于形成聚类点集，最终 bias 仍由联合求解得到。

项目同时提供 DeepMIMO V4 + Sionna RT 2.0 的可选入口。定位算法本身不依赖
这两个大型仿真包，因此即使尚未安装它们，也能先验证数学和数据链路。

## 已固定的物理约定

- UE 单天线发射，BS 均匀线阵接收。
- BS 的二维位置、阵列朝向和阵元参数是定位时已知的公开先验；UE 位置和注入
  时钟偏差只存在于生成与独立评估侧。
- 第一阶段只估计二维坐标；BS、UE 使用同一固定高度。
- 使用直射和最多二次镜面反射；`scene.max_diffractions=1` 可加入固定高度二维边缘的最多一次阴影侧绕射，允许与反射组合。漫反射和穿透仍未接入定位。
- 观测时延为 `几何时延 + 公共偏差 + 单路径误差`。
- 内部使用距离偏差 `beta = c * b`，候选轨迹为
  `p(beta) = anchor - beta * direction`。
- 同一个 MUSIC 峰产生的不同反射解释互斥，求解时最多选择一个。
- 正常输出只有 `mu`、完整 `sigma`、公共偏差和数值诊断，不输出
  `accept/reject/ambiguous`。
- MUSIC 伪谱值用于找峰及构造局部候选搜索分布，不当作已校准的路径概率。
  同一来源峰的样本是互斥候选解释，不因采样数量增加而获得更多独立观测权重。
- `sigma` 是所选候选的几何残差近似协方差，不是 CSI 扰动解的分布，也未经覆盖率校准。
- `music.signal_subspace_rank` 表示 MUSIC 计算时认为 CSI 中有多少个信号成分；
  `music.num_paths` 表示最后向定位器输出多少个峰。两者用途不同，不再要求相等。

## 立即运行离线闭环

```bash
cd /data/zhujun/differt_projects/time-bias-correct
/data/zhujun/differt_projects/time-bias-correct/run_offline_demo.sh
```

脚本默认使用：

```text
/home/zhujun/miniconda3/bin/python3
```

若要换解释器：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
PYTHON_BIN=/绝对路径/python /data/zhujun/differt_projects/time-bias-correct/run_offline_demo.sh
```

默认产物位于 `outputs/offline_demo/`：

- `scene/scene_bev.png`：定位使用的二维俯视图。
- `scene/scene_2d.json`：与俯视图严格对齐的米制墙线和坐标变换。
- `data/online/measurement.npz`：定位程序唯一允许读取的 CSI 输入，字段必须严格为
  `csi_observed`、`subcarrier_frequencies_hz`、`carrier_frequency_hz`、
  `antenna_spacing_m`、`bs_position_m`、`bs_boresight_rad` 这 6 项。
- `data/generation_manifest.json` 或根目录的 `generation_manifest.json`：把场景、
  在线 CSI 和评估真值绑定为同一个生成批次；定位使用场景、CSI 来源记录和公开的
  物理模型声明做一致性检查，但不会打开其中记录的真值文件。
- `data/truth/ground_truth.npz`：只供独立评估读取的 UE 与偏差真值。
- `localization/music_spectrum.npz`：二维 MUSIC 谱和坐标网格。
- `localization/spectrum_samples.json`：每个峰的局部细谱面、连续采样坐标、来源和采样参数。
- `localization/initial_candidates.json`：参考 bias 下的初始位置点、路径来源和排除原因。
- `localization/representative_points.json`：点簇、成员及所选真实代表点。
- `localization/representative_trajectories.json`：只为代表点生成的轨迹及合法 bias 范围。
- `localization/localization_result.json`：最终 `mu`、`sigma` 和公共偏差。
- `localization/forward_check.json`：最终解对应的路径合法性，以及预测角度、时延与输入观测的差异。
- `localization/localization_config.json`：补齐默认值后的定位专用配置快照及其文件指纹；
  其中含有公开的 `radio.bs_position_m`，但不含 `simulation`、UE 位置、注入偏差
  或生成时使用的 `snr_db`。
- `localization/localization_manifest.json`：本次定位编号和批次、场景、CSI、主结果及
  全部诊断产物的文件指纹。
- `evaluation/metrics.json`：单独读取真值后得到的误差。
- `evaluation/history/<旧运行编号>/`：当前版本在重新定位前保存旧主结果、全部诊断文件、
  定位配置、指标及路径已改写的旧定位清单；固定 `metrics.json` 只代表当前已经
  完成评估的运行。离线闭环还会把该次使用的场景、CSI、真值和生成清单一并保存。
- `receipts/*.json`：分阶段运行的不可覆盖回执，记录本次定位编号以及结果和清单
  指纹；评估必须拿着该编号运行，不能猜“当前最新的一次”。

## 分阶段运行

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_preprocess_scene.sh
./run_generate_simulated_csi.sh
RECEIPT_PATH="${PWD}/outputs/offline_demo/receipts/manual_$(date -u +%Y%m%dT%H%M%S)_$$.json"
RECEIPT_JSON="${RECEIPT_PATH}" ./run_localization.sh
RUN_RECEIPT_JSON="${RECEIPT_PATH}" ./run_evaluate.sh
```

每个脚本顶部都有独立“用户参数区”，路径和解释器均为清楚的绝对路径。
离线分阶段脚本中，场景和数据生成使用 `configs/offline_demo.yaml`，定位使用不含
`simulation` 的 `configs/offline_demo_localization.yaml`；不要把生成配置传给
`run_localization.sh`。场景和数据生成入口默认不覆盖旧批次，上面前两步只适用于
新的 `output.root`；要重做数据，请同时为生成配置和定位配置换一个新目录。
若只想用已有数据重跑定位，从设置 `RECEIPT_PATH` 的一行开始即可。

## DeepMIMO V4 + Sionna RT

小规模工程检查使用两份配置：`configs/deepmimo_sionna_smoke.yaml` 只负责生成，
`configs/deepmimo_sionna_smoke_localization.yaml` 只负责定位。后续更完整的场景实验
使用对应的 `deepmimo_sionna_munich.yaml` 和
`deepmimo_sionna_munich_localization.yaml`。定位配置显式包含已知 BS 位置，但不含
`simulation` 或生成时的 `snr_db`。生成入口会：

1. 加载 Sionna RT 内置 Munich 场景；
2. 将 UE 设为发射端、BS 阵列设为接收端；
3. 只开启直射和最多二次镜面反射；
4. 强制保留绝对时延；
5. 保存 DeepMIMO V4 转换产物，但不使用其中经过凸包简化的场景做定位；
6. 使用预先写在配置里的固定地图范围 `[-100, 160, -80, 180]`，不根据 UE
   或真实路径改变范围；
7. 直接读取 Sionna 导出的原始顶点和三角面，在固定高度切出二维墙线；
8. 仅在生成阶段检查保留路径从 BS 反向出发时，是否先命中 Sionna 给出的真实墙面；
   若 UE、BS 或交互点在固定范围外则停止，不会扩大地图；
9. 删除不能在固定高度二维化的地面、屋顶或非正面路径；
10. 生成带统一公共时延偏差的在线 CSI；定位命令不会收到真实路径或 UE 真值。

之所以绕开 `DeepMIMO Dataset.scene`，是因为当前转换过程会先把连通物体变成
二维凸包。凹形建筑或同一对象内的分离部分可能因此多出一条现实中不存在的长墙。
第一版定位地图直接切原始 Sionna 三角网格，避免这类“假墙”；DeepMIMO 数据仍
完整保留，供数据格式和后续实验使用。

### 一键运行完整工程检查

环境准备完成后，运行：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_smoke_deepmimo_sionna.sh
```

脚本固定执行三个阶段：

```text
Sionna/DeepMIMO 与在线 CSI 准备 → 定位 → 独立真值评估
```

默认 `REUSE_GENERATED=1`。如果输出目录已有生成清单，脚本会先检查它确实来自
原始 Sionna 三角网格、使用固定地图范围、保留了绝对时延、链路方向正确且场景
自检通过，然后复用耗时的射线结果；生成配置、场景、在线 CSI 和真值的文件指纹
也必须一致。运行前还会核对生成配置与定位配置的公开参数和 `output.root` 一致。
定位只读取不含 `simulation` 的定位配置；配置、场景 JSON、生成清单和在线 NPZ
都采用严格字段检查。主入口还会核对上行方向、绝对时延、固定地图来源、反射模型、
CSI 形状与频率间隔，以及 BS 位置和朝向。它只用公开输入完成这些检查，不会打开
清单所记录的真值文件。定位会单独保存配置快照，定位与评估每次都会重新运行。
若要强制重新生成，先在生成和定位配置中同时换到
一个新的 `output.root` 和 DeepMIMO 场景名，再执行：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
REUSE_GENERATED=0 ./run_smoke_deepmimo_sionna.sh
```

脚本不会删除或覆盖已有射线目录。顶部“用户参数区”集中列出了项目目录、解释器、
`GENERATION_CONFIG_PATH`、`LOCALIZATION_CONFIG_PATH`、`GENERATION_MANIFEST`、
输入和输出路径；
`OUTPUT_ROOT` 默认从两份配置共同的 `output.root` 读取。若两份配置互不匹配，或
手工设置的输出目录与配置不一致，脚本会停止，避免把不同实验的文件混在一起。

### 绝对时延不等于无限时延范围

Sionna 的 `cir()`/`cfr()` 默认会把第一条路径移到零时延。本项目统一强制传入
`normalize_delays=False`，任何尝试打开时延归一化的调用都会直接报错。

但等间隔子载波只能区分一个有限时延区间。若子载波间隔为 `Δf`，不重复区间为
`1/Δf`；本项目的频率生成方式下也可写成 `子载波数/带宽`。例如 512 个子载波、
400 MHz 带宽对应约 1.28 微秒。MUSIC 搜索窗必须小于这个范围，否则同一路径会
每隔 1.28 微秒重复出现。代码会在定位前检查这一点；保留绝对时延并不能代替该检查。

### 均匀线阵的正面约束

单条均匀线阵无法仅靠相位区分阵列正面和背面的镜像方向。第一版通过
`radio.front_facing_only: true` 明确采用正面先验：仿真 CSI 只保留相对 BS 朝向
位于 `[-90°, 90°]` 的到达路径，MUSIC 也只在同一范围搜索。这个开关是第一版的
阵列模型边界，不代表已经解决 360° 到达角；若关闭它，应改用能分辨前后的阵列，
或同时保留两个方向解释。

若还没有项目专用环境，先运行：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./setup_sionna_environment.sh
```

### 历史版本的小规模实跑记录（CSI 扰动流程）

本节保留谱面采样改造之前的运行证据；其中的位置误差、协方差和扰动次数
均属于当时版本，不能作为当前 `music_fine_spectrum_dbscan_v3` 的验证结果。

run11 已在 Munich 场景从零完整生成，并再次执行一键脚本验证了安全复用和旧指标
归档。产物保存在 `outputs/deepmimo_sionna_smoke_run11/`。Sionna 共给出 15 条路径，
其中 10 条满足
固定高度二维条件，6 条同时位于阵列正面；6 条路径全部通过生成阶段的二维墙面
反向自检。最终使用 3 个 MUSIC 峰，4/4 次 CSI 扰动均得到完整解：

```text
位置真值       [50.0000, 50.0000] m
位置估计       [49.9952, 49.8396] m
位置误差       0.1604 m
偏差真值       25.0000 ns
偏差估计       25.0898 ns
偏差误差       0.0898 ns
Σ              [[0.00775565, 0.00080473],
                [0.00080473, 0.00851535]] m²
```

本次生成清单明确记录固定地图范围来自配置，且为第 2 版格式；声明的批次号
`342d0e59…1f48` 与场景、在线 CSI、真值三个文件指纹重新计算的结果一致。定位配置
快照不含 `simulation` 和 `snr_db`，并显式保留 BS 已知位置；在线测量文件也严格只有
CSI、频率和 BS 已知量。批次号标识三个核心文件，完整生成清单自身的 SHA-256
另外绑定上行、时延和地图等物理声明。第二次运行
复用了生成数据，先归档第一次指标，再产生新的定位编号并重新评估；定位清单和指标
仍绑定同一批次。这组数字只用于证明当时版本的代码、物理方向、文件边界和一键脚本彼此
一致，不作为多场景精度结论。当时结果见
`outputs/deepmimo_sionna_smoke_run11/evaluation/metrics.json`。

run11 中编号为 `691931db-55cb-4e92-b2ba-bb74c39dfd67` 的历史目录是在完整归档机制
加入前留下的旧证据，仅含清单和指标，不能当作可独立复现的完整历史；项目保留它
用于追溯，不删除也不伪造补齐。此后的 `5fb6b9df-...`、`8120b76a-...` 等归档才按
当前的完整格式保存。

## 测试

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_tests.sh
```

测试覆盖偏差符号、二维 MUSIC 恢复、场景投影、反向轨迹、候选
互斥、多路径稳健求解、协方差半正定、上下行互易、绝对时延保护、固定地图范围、
严格输入字段与几何、产物防覆盖、批次混用拒绝、并发锁、原子写入、完整历史归档、
定位回执以及端到端真值隔离。

## 当前边界

- 新流程的实现与检查记录见 [点聚类流程说明](docs/point_clustering_workflow_20260909.md)；
  历史版本的工程运行记录保留在上节，不与新流程合并统计。
- DeepMIMO/Sionna 的 Munich run11 从零生成、安全复用和历史指标归档都已打通，
  证明了当时 API、数组形状、固定地图范围、双配置隔离、批次绑定和三阶段脚本能够工作。
- `music.signal_subspace_rank` 和 `music.num_paths` 仍由配置给定。前者控制 MUSIC
  如何分开信号与噪声，后者控制送入定位的峰数；路径数自动估计留到后续实验。
- 局部采样复用同一份 CSI 的子空间，不重复分解，不进行扰动峰配对；候选数量
  由 `music.spectrum_sampling.samples_per_peak` 控制。采样不会补回全局找峰已经漏掉的路径。
- 初始点先聚类，随后只为代表点构建轨迹，再执行一次联合求解；没有足够有效路径时记录失败，失败前已完成的
  步骤数据单独保存。未执行的后续步骤不生成估计结果。
- 旧 `music.uncertainty_*`、`association_max_normalized_distance`、
  `false_peak_penalty`、`missed_peak_penalty` 配置会明确报错。旧报告保持只读兼容；
  重放旧 CSI 时应使用当前定位配置和新的输出目录。
- 每次定位都会生成新的运行编号；`run_localization.sh` 和一键 smoke 脚本还会创建
  不可覆盖回执，并绑定配置、场景、在线 CSI 和结果文件指纹；评估
  前会核对生成批次、场景、在线 CSI、真值和结果记录。定位重跑前，旧指标和清单
  会进入历史目录；固定指标路径在新评估成功前保持为空，不能冒充当前结果。
- 同一输出目录的生成、定位和评估由跨进程排他锁串行执行；场景、NPZ 和 JSON
  先写到同目录临时文件，再整组或原子发布；评估指标和更新后的定位清单也会一起
  提交或一起回滚。普通生成入口拒绝覆盖任一旧目标；
  只有离线闭环在完整归档旧运行后，才允许在同一目录发布下一批。
- 当前仍只覆盖固定高度二维定位、均匀线阵正面、直射和最多二次镜面反射。
- 工程 smoke 只证明程序和数据链路跑通，不等于科学验证。正式结论还需要多位置、
  多随机种子、不同信噪比和偏差、路径漏检/虚警扫掠、与基线比较以及真实 CSI 验证。

更细的数据边界和模块关系见 [docs/architecture.md](docs/architecture.md)。

## 结果图表与多 UE 小实验

运行 `./run_visualize_results.sh` 可直接绘制已有 Munich run11 历史结果；新结果通过
脚本顶部的输入目录参数指定。输出按
`samples/UE编号/repeat编号/00–09步骤/` 组织，每步保存图及原始数据，包括 CSI、
MUSIC、局部谱面采样、初始点、点簇及代表、代表轨迹、唯一联合解、正向检查和独立评估。
历史报告使用其原始步骤名，并明确标明旧版本。
`summary/` 单独保存全部 sample 的定位误差 CDF、Med/P90 图和逐次、逐 UE 汇总表。
不包含候选簇之间的联系图；绘图独立读取已有评估产物，不修改定位结果。

`./run_point_clustering_experiment.sh` 提供一个 BS、30 个 UE、每点 5 次独立噪声的
实验入口。设置 `PLAN_ONLY=1` 只生成固定采样计划和示意图；正式执行时每个 UE
复用同一份无噪声信道生成重复，失败项也计入总数。默认采样范围是已有 UE 附近的
20 m × 20 m 局部空旷区域。参数、路径、恢复运行和图表含义见
[结果可视化说明](docs/visualization_experiments.md)。

GPU 全量实验示例：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_IDS=0,1,2,3,4,5,6,7 PLAN_ONLY=0 RESUME=0 ./run_point_clustering_experiment.sh
```

每卡一个常驻进程处理不同 UE；每份接收 CSI 的特征分解只执行一次，全局谱、
局部谱和连续坐标评价复用该结果。默认仍为 30 UE × 5 次独立噪声。
先检查已有观测而不生成信道时，运行：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_point_clustering_check.sh
```

检查脚本默认读取 `spectrum_experiment_20260908T091621_2543248` 中
UE001、UE010、UE006 各一份已保存的 CSI，在新目录重新
定位并输出逐步骤报告和汇总。两份新脚本的输入、输出和参数都集中在顶部。
安装、单卡/多卡命令、历史执行记录及对照验证见
[GPU 运行说明](docs/gpu_execution.md)。
