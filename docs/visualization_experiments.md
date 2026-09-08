# 定位实验与结果绘图

本次实现沿用现有定位算法，新增独立的批量实验和绘图模块。**不绘制 UE 或候选簇之间的联系图。**
真实路径、真实位置和真实偏差只供生成、评估和绘图使用；定位仍经过原有严格输入检查。

## 图表要回答的问题

图表用于检查：同一 BS 下，不同 UE 和独立噪声重复的最终位置与时间偏差误差如何，以及真实传播路径与初次聚类轨迹是什么样子。没有预设精度结论。

按每个 UE sample 单独建文件夹，文件夹内按独立噪声重复和实际执行步骤保存图与数据。总体 CDF、Med、P90 在独立的 summary 文件夹中。
默认输出中文标注的 PNG（300 dpi）、PDF、可编辑文字 SVG，以及对应 CSV。地图横纵轴统一为米，保持等比例。
当前画布适合查看和后续排版；正式投稿时仍需按目标期刊的最终栏宽调整文字大小。

当前主流程为 `music_spectrum_sampling_v1`，定位清单格式为第 4 版。接收到的一份带噪 CSI 只计算一次子空间，随后计算全局 MUSIC 和各峰附近细谱面，连续采样、汇集反向候选、聚类取代表，再联合求解一个位置和公共 bias。定位内部不再额外加噪。旧实验仍按其真实旧步骤只读展示，不会换成新流程的标签。

## 直接画已有结果

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_visualize_results.sh
```

脚本默认解释器为 `/home/zhujun/miniconda3/bin/python3`，默认读取
`outputs/deepmimo_sionna_smoke_run11/`。需要 `matplotlib>=3.8` 和中文字体；当前机器已有文泉驿微米黑。
依赖也声明在 `pyproject.toml` 的 `visualization` 可选组中。

参数集中在脚本顶部，可以通过环境变量指定输入输出：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
RUN_ROOT="${PWD}/outputs/deepmimo_sionna_smoke_run11" \
REPORT_ROOT="${PWD}/outputs/my_result_figures" \
./run_visualize_results.sh
```

绘图目录必须尚不存在。绘图不重新定位、不重新评估，也不修改已有指标；读取前核对生成清单、结果、诊断、评估的文件指纹和运行编号。未完成评估或文件被修改时停止。

## 一个 BS、30 个 UE、每点 5 次独立噪声

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_IDS=0,1,2,3,4,5,6,7 PLAN_ONLY=0 RESUME=0 ./run_spectrum_sampling_experiment.sh
```

默认解释器为项目的 `.sionna-venv/bin/python`，Sionna 环境设置沿用已有脚本。
默认配置为 `configs/visualization_experiment_munich.yaml`：

| 参数 | 默认值及修改位置 |
|---|---|
| UE 数量、重复次数 | 实验配置的 `ue_count: 30`、`noise_repeats: 5` |
| 采样区域 | 实验配置的 `sampling_bounds_m: [40, 60, 40, 60]` |
| 采样种子 | 实验配置的 `random_seed: 20260907` |
| BS 坐标、真实偏差 | 所引用 `deepmimo_sionna_munich.yaml` 的 `simulation` 段 |
| 阵列朝向、信噪比 | 所引用生成配置的 `radio` 段 |
| MUSIC 和定位参数 | 所引用生成配置的 `music`、`localization` 段 |
| 每峰连续采样数 | 所引用生成配置的 `music.spectrum_sampling.samples_per_peak: 128`，另保留 1 个原始峰参考样本 |
| 输出路径、解释器 | `.sh` 脚本顶部用户参数区 |

默认沿用 BS `(20,50) m`、朝向 `−45°`、偏差 `25 ns` 和信噪比 `35 dB`。
采样矩形位于已有 `(50,50)` UE 所在空旷区域。这是 **20 m × 20 m 局部实验**，不代表整座城市的覆盖。
程序保守检查整个矩形及墙边距不与任一墙段包围盒重叠，并限制 UE 与 BS 最小距离。
二维墙线不能自动判断一个封闭区域是房间还是建筑实体，修改矩形时须依据场景选择可通行区域；不声称支持任意地图自动剔除建筑内部。

先只生成采样计划和场景图：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
PLAN_ONLY=1 EXPERIMENT_ROOT="${PWD}/outputs/my_30ue_experiment" \
./run_spectrum_sampling_experiment.sh
```

之后执行该固定计划：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
RESUME=1 EXPERIMENT_ROOT="${PWD}/outputs/my_30ue_experiment" \
./run_spectrum_sampling_experiment.sh
```

每个 UE 只做一次信道生成。每次噪声重复均从该 UE 保存的无噪声 CSI 重新注入噪声，位置、路径系数和真实偏差保持不变。
这些是实验中独立生成的接收数据；每份接收 CSI 内部的候选全部由局部 MUSIC 谱面采样产生，不再进行 CSI 扰动及重复提峰。
生成模板和所有位置、噪声种子在运行前保存为 `experiment_plan.json` 及配置快照。
定位配置通过已有过滤函数生成，再经过定位专用配置加载器读取，不含 UE、真实偏差和注噪强度。
每次重复保留自己的生成清单、回执、定位结果、独立评估和 `attempt.json`。
`noise_generation.json` 记录原始信道清单和噪声种子；不传入定位器。

恢复运行时跳过已写入终态的尝试，不按定位精度筛选或重新采样。中断残留保留，标记 `interrupted_failed`，需要重做时使用新的实验目录。
某个 UE 的信道生成失败，其全部噪声重复仍作为失败项保留；程序继续后续 UE。终态失败也不会自动重试。
旧版本的 `uncertainty_*` 和峰配对配置被明确拒绝，不能直接用旧计划混跑新流程。只读绘制旧结果仍受支持。

## 先重放已有接收数据检查新流程

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_spectrum_sampling_check.sh
```

默认输入是 `outputs/gpu_munich_20260908T020052_1904235` 中 UE001、UE002、UE006 各一份已保存的接收 CSI，定位配置为 `configs/deepmimo_sionna_munich_localization.yaml`。不重新生成信道或注噪，结果、逐步骤图和总体统计写入新的 `outputs/spectrum_check_时间_进程号/`。

脚本顶部可修改 `SOURCE_EXPERIMENT`、`UE_IDS`、`NOISE_REPEATS`、`SAMPLES_PER_PEAK`、`CONFIG_PATH`、计算设备及 `OUTPUT_ROOT`。`NOISE_REPEATS` 表示读取每个 UE 已有的几份独立观测。检查少量 UE 的结果用于核对流程，不能代替 30 UE × 5 次噪声的全量精度结论。

局部采样参数集中在 `music.spectrum_sampling`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `samples_per_peak` | 128 | 每个来源峰的连续随机样本数 |
| `aoa_half_width_grid_steps` | 1.5 | 角度采样区域的半宽，单位为原始角度网格步长 |
| `delay_half_width_grid_steps` | 1.5 | 时延采样区域的半宽，单位为原始时延网格步长 |
| `local_grid_points_per_axis` | 25 | 局部角度、时延两轴各自的细网格边界数 |
| `spectrum_power` | 1.0 | 构造候选搜索分布时的谱值幂次 |
| `uniform_mixture` | 0.1 | 覆盖整个局部区域的均匀采样比例 |
| `include_nominal` | true | 每个来源峰额外保留一个原始峰参考样本 |

局部区域裁剪到原 MUSIC 搜索范围内。先根据单元中心的谱值及单元面积抽取单元，再在单元内生成连续角度和时延；对应谱值由同一子空间精确计算。该分布只用于候选搜索，不是已校准的路径概率。每个样本保留来源峰，权重为 1，不再乘一次谱值。

DeepMIMO 的本地场景名统一使用小写：其加载器会自动将名称转小写，Linux 目录名却区分大小写。
转换后先检查对应本地目录和 `params.json`，确认存在才调用加载器，缺失时直接停止该次生成，不询问在线下载。
若旧批次出现 `Scenario not found`、在线下载 403，保留旧目录，使用修复后的脚本以 `RESUME=0` 新建实验。
批量入口生成定位配置时，Sionna 的 `scene.bounds_m` 同步为公开的 `localization_bounds_m`，避免合并默认配置时误用合成房间范围。

单独重画整个实验，不重复计算：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
EXPERIMENT_ROOT="${PWD}/outputs/my_30ue_experiment" \
REPORT_ROOT="${PWD}/outputs/my_30ue_figures" \
./run_visualize_results.sh
```

## 输出说明

```text
报告目录/
├── samples/
│   ├── UE001/
│   │   ├── sample_summary.json       # 本 UE 的 Med/P90 和次数
│   │   ├── results.csv               # 本 UE 每次噪声重复的结果
│   │   ├── repeat_000/
│   │   │   ├── 00_scene_truth/       # 场景、BS、真值和参与 CSI 的路径
│   │   │   ├── 01_csi_input/         # 在线 CSI 幅度/相位、原始输入和公开配置
│   │   │   ├── 02_music/             # 二维 MUSIC 谱、原始峰位置和数值
│   │   │   ├── 03_spectrum_sampling/# 局部细谱面、连续采样点、来源及采样分布
│   │   │   ├── 04_reverse_candidates/# 全部谱面采样观测的反向候选轨迹
│   │   │   ├── 05_first_clustering/  # 各簇成员、代表候选、聚类前后对照
│   │   │   ├── 06_joint_solution/    # 代表候选的位置/偏差联合解及残差
│   │   │   ├── 07_forward_check/     # 最终解的预测路径与输入观测检查
│   │   │   └── 08_final_evaluation/  # 同一个联合解的位置、偏差及独立评估
│   │   └── repeat_001/...
│   └── UE002/...
└── summary/
    ├── position_error_cdf.png        # 同时导出 PDF/SVG
    ├── position_error_med_p90.png
    ├── per_attempt.csv
    ├── per_sample.csv
    └── summary.json
```

每个步骤都有 README，写明对应的真实函数名；每个重复目录有 step_index.json。
图之外同时保存该步骤的 JSON/NPZ/CSV，能直接检查实际数值。

| 产物 | 内容 |
|---|---|
| `summary/01_scene_samples.*`、`01_scene_samples_zoom.*` | 全局场景和采样区域放大图 |
| `summary/03_localization_map.*` | 所有真实位置、估计位置及误差连线 |
| `summary/position_error_cdf.*` | 成功定位误差的标准经验 CDF，图例标出 Med/P90 |
| `summary/position_error_med_p90.*` | 全部成功尝试的 Med/P90、每个 UE 的 Med/P90 |
| `summary/all_attempt_attainment_and_bias.*` | 全部计划次数中的达标比例、偏差误差分布 |
| `summary/per_attempt.csv` | 每次真实/估计坐标、位置误差、偏差误差、状态及耗时 |
| `summary/per_sample.csv` | 每个 UE 的成功/失败/待运行次数、噪声重复内的 Med/P90 |
| `summary/summary.csv`、`summary.json` | 全部成功尝试的误差中位数、均方根、P90 及整体计数 |
| `report_manifest.json` | 输入文件和图表、表格的指纹，以及绘图代码指纹 |

误差定义：位置为 `||估计坐标 − 真实坐标||₂`（m），时间偏差为 `估计偏差 − 真实偏差`（ns）。
整体 Med/P90 和标准 CDF 使用全部成功尝试的误差，同时报告失败/待运行次数。
每个 UE 的 Med/P90 在该 UE 成功的噪声重复中计算，不先平均估计坐标再算误差。
分位数使用 numpy.quantile 默认线性插值；小样本下标注分位数不一定恰好在经验 CDF 的阶跃处。
另存的 1/2/5 m 达标率和 all_attempt_attainment_and_bias 以全部计划尝试为分母，失败、待运行项不计为达标。
失败项误差留空，不写成零。成功表示程序完成并得到评估，不表示达到论文精度要求。
同一位置的五份噪声不是五个独立空间样本，不据此计算空间泛化的置信区间。

04、05 分别画聚类前、聚类后的完整合法轨迹，没有提前代入最终估计偏差。
颜色表示观测，实/虚/点划线表示 0/1/2 次反射，R/C 编号与 trajectories.csv 对应。
06 展示聚类代表参与的唯一联合解，07 对该解做正向路径检查，08 对同一个解独立评估。没有“原始峰解”和“扰动均值解”两套结果。
03 的连续样本全部传入 04；05 保存簇成员与所选真实成员代表。同一来源峰、同一反射结构内才允许聚类，所有候选保留合法 bias 区间。
椭圆仅展示所选候选几何残差得到的协方差尺度，未经覆盖率校准，不能解释为已经验证的 95% 定位置信区间。

新运行的 `spectrum_samples.json` 保存局部谱面和全部观测样本，`forward_check.json` 保存预测路径与输入角度、时延的对照。定位内部不再生成 `bootstrap_diagnostics.json` 或 `uncertainty_solutions.npz`。
若新运行中途失败，已完成阶段的中间产物保存到 `localization_failures/运行编号/`，报告显示失败发生在哪一步；未执行阶段明确标注未完成。旧失败运行没有可核验中间文件时，各步骤保留“缺失/失败”说明，不编造图。
这里展示的是固定步骤的输入输出，不声称保存每次求解器迭代的历史。

历史扰动版本仍使用 `03_peak_perturbations` 和 `07_perturbation_solutions`：当时 04–06 使用原始峰，07 对扰动观测分别求解，08 汇总。报告会标记历史流程；没有保存的内部候选不能事后补写为执行证据。新旧流程结果不能直接拼成一条同方法 CDF。

Sionna 真实路径只画 `retained_mask` 保留并参与 CSI 合成的路径。
离线路径交互点来自真值 JSON，核对其路径长度与 NPZ 时延后绘制，并单独记录文件指纹；旧生成清单本身没有绑定该 JSON。

## 独立绘图与历史验证记录

若定位已完成而绘图报错，只需用 `run_visualize_results.sh` 读取原实验，避免重新生成信道和定位。脚本顶部的 `EXPERIMENT_ROOT` 指定原实验，`REPORT_ROOT` 指定新的报告目录。
`SUMMARY_ONLY=1` 先生成总体场景图、定位误差 CDF、Med/P90 和全部逐次/逐 UE 表格；默认 `SUMMARY_ONLY=0` 还会导出每个 sample 的全部中间步骤。汇总模式仍校验所有成功结果的输入和产物指纹，并明确标注未导出逐步骤图。

当前谱面采样版本的实现与检查见 [谱面采样实现记录](spectrum_sampling_implementation_20260908.md)。以下数值全部来自切换前的 CSI 扰动版本，保留用于追溯，不作为当前流程的精度结论。

2026-09-08 的 `gpu_munich_20260908T020052_1904235` 完成 150 次尝试，145 次成功，UE006 的 5 次均在联合求解入口因有效观测组不足失败。所有 30 份地图的 1440 个墙段几何精确一致，但部分 Sionna 导出编号不同；已修正全局底图判断，原定位结果保持不变。修复后的汇总报告在该实验的 `summary_report_fixed/`，成功结果的 Med/P90 分别为 0.3071/0.6167 m，失败次数另外报告。

已经使用现有 Munich run11 产物生成图表，并执行合成房间的 2 UE × 2 噪声闭环和图表汇总。
上述初次图表验证时，Munich 30 UE × 5 噪声仅生成了采样计划；该说明不代表后续实验的完成情况，具体状态以各实验目录的 `attempt.json` 和汇总表为准。
新增测试覆盖固定信道/独立噪声、在线字段隔离、防覆盖、采样可复现、失败分母以及诊断文件修改后的拒绝读取。

2026-09-08 的 GPU 架构验证另外完成了 Munich 4 UE × 2 次噪声的双卡运行，8 次定位及独立评估全部成功，结果位于 `outputs/gpu_rebuild_validation_20260908_dual`。
其图表汇总位置误差 Med 为 0.1201 m、P90 为 0.4130 m；这是小规模实现验证，不作为论文的空间泛化结论。GPU 与 CPU 对照的命令和完整计时边界见 [GPU 运行说明](gpu_execution.md)。
