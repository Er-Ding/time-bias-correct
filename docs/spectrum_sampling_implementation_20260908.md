# 谱面采样定位：实现与运行

> 版本说明：本文记录 2026-09-08 的谱面采样版本。2026-09-09 已进一步改为“初始位置点聚类后，再生成代表轨迹”；当前实现、脚本和输出见 [点聚类流程说明](point_clustering_workflow_20260909.md)。下文的 v1 结果和流程保留供追溯。

实现日期：2026-09-08。流程标识：`music_spectrum_sampling_v1`。定位清单版本：4。

## 已经实现的流程

接收到的一份带噪 CSI → MUSIC 谱 → 每峰局部谱与连续采样 → 全部样本反向追踪 → 第一次聚类与代表候选 → 一次位置和共同 bias 联合求解 → 已选路径正向检查 → 独立评估。

定位内部不再调用 CSI 额外加噪、重复提峰、扰动峰配对、扰动求解或结果平均。旧的独立辅助函数仅保留给历史回归，不参与新的结果生成。

每个来源峰的样本共用来源编号，各有独立样本编号。反向候选保留位置随 bias 的变化以及合法 bias 区间；第一次聚类在同源峰、同反射结构内进行。簇内每对成员都需要满足距离和方向阈值，代表选簇内真实成员，保留其完整几何和观测信息。

同一路径的候选互为替代，求解器每个来源峰最多选一个代表。采样数量、簇大小不被当成独立测量次数，也不会直接增加观测权重。最终协方差是所选候选几何残差的近似，不是经过校准的采样置信区间。

## 在现有接收数据上检查

不需要重新生成信道，也不会重新加噪。默认重放原实验 UE001、UE002、UE006 的第 0 次接收数据，结果写入独立新目录：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
./run_spectrum_sampling_check.sh
```

参数集中在 `run_spectrum_sampling_check.sh` 的开头。例如：

```bash
UE_IDS=UE001,UE002,UE003 NOISE_REPEATS=5 SAMPLES_PER_PEAK=128 \
DEVICE_ID=0 ./run_spectrum_sampling_check.sh
```

- `SOURCE_EXPERIMENT`：已经保存接收 CSI 的实验目录。
- `CONFIG_PATH`：新的定位专用配置，默认 `configs/deepmimo_sionna_munich_localization.yaml`。
- `UE_IDS`：原实验中的 UE 编号，逗号分隔。
- `NOISE_REPEATS`：重放每个 UE 已保存的前几份接收数据。
- `SAMPLES_PER_PEAK`：每个谱峰的蒙特卡洛样本数，另保留一个原始峰参考样本。
- `COMPUTE_BACKEND`：`cuda` 或 `numpy`；指定 CUDA 后初始化失败会报错，不会自动改用 CPU。
- `OUTPUT_ROOT`：新的结果目录，已存在时拒绝覆盖。

默认解释器是仓库的 `.sionna-venv/bin/python`。不要直接把其指向的基础环境解释器当作等价入口：虚拟环境中安装的依赖可能不同。

## 新建完整实验

```bash
cd /data/zhujun/differt_projects/time-bias-correct
GPU_IDS=0,1,2,3,4,5,6,7 PLAN_ONLY=0 RESUME=0 \
./run_spectrum_sampling_experiment.sh
```

默认仍为 30 个 UE，每个 UE 5 份独立带噪接收数据。UE 数、噪声重复数和区域在 `configs/visualization_experiment_munich.yaml` 修改；BS、信号与谱面采样设置在它引用的 `configs/deepmimo_sionna_munich.yaml` 修改。

一张 GPU 对应一个常驻进程，不同 UE 分配到不同进程；同一个 UE 的各份接收数据按原定种子依次执行。新脚本默认使用独立的 `outputs/spectrum_experiment_时间_进程号/` 目录。旧实验计划不能直接 RESUME 到新流程。

## 当前采样参数

参数位于 `music.spectrum_sampling`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `samples_per_peak` | 128 | 每峰连续蒙特卡洛样本数 |
| `aoa_half_width_grid_steps` | 1.5 | 角度区域半宽，相对于原始搜索网格步长 |
| `delay_half_width_grid_steps` | 1.5 | 时延区域半宽，相对于原始搜索网格步长 |
| `local_grid_points_per_axis` | 25 | 每个局部谱坐标轴的网格点数 |
| `spectrum_power` | 1.0 | 构造搜索分布时采用的谱值幂次 |
| `uniform_mixture` | 0.1 | 混入的局部区域均匀采样比例 |
| `include_nominal` | true | 额外保留原始峰作为参考候选 |

先根据局部网格单元中心谱值、单元面积和均匀混合项选择单元，再在单元内部连续采样，并重新计算该坐标的精确 MUSIC 谱值。整个过程复用同一份 CSI 的子空间；记录中的 `eigendecomposition_count` 应为 1。

采样区域在实际搜索范围内截断。默认 Munich 粗网格为 1 度和 1 ns，因此采样半宽分别为 1.5 度和 1.5 ns。这些是当前实现参数，不代表已经验证最优，也不是误差置信范围。

旧 `uncertainty_*` 和扰动峰配对配置被明确拒绝。修改采样参数时应新建实验目录，保留原参数与结果用于对照。

## 结果在哪里

`step_report/samples/UE编号/repeat_编号/` 包含：

| 文件夹 | 内容 |
| --- | --- |
| `00_scene_truth` | 场景、真实 UE 和真实路径，只用于评估和示意图 |
| `01_csi_input` | 实际接收到的带噪 CSI 与公开定位参数 |
| `02_music` | 全局 MUSIC 谱与原始峰 |
| `03_spectrum_sampling` | 每峰局部谱、连续采样散点、样本表和采样参数 |
| `04_reverse_candidates` | 所有采样输入生成的候选轨迹及来源 |
| `05_first_clustering` | 簇成员、真实代表、数量统计和叠加图 |
| `06_joint_solution` | 唯一联合解、所选代表、位置、bias 和残差 |
| `07_forward_check` | 重算已选路径，核对原始峰与采样代表的角度和时延残差 |
| `08_final_evaluation` | 估计值、真实值、位置欧式误差和 bias 误差 |

`step_report/summary/` 保存总体误差的中位数、P90、CDF 和失败统计。成功误差统计的分母与全部尝试的分母分别记录。

定位失败时，已完成的步骤写入 `localization_failures/运行编号/`；`attempt.json` 明确绑定 `progress.json`，报告据此展示完成到哪一步。不会重新计算未保存的步骤冒充历史输出。

正向检查只重算选中的二维直射/镜面反射路径，检查反射点、遮挡、共同 bias 和观测残差。它不枚举全场景全部路径，也不重建完整 CSI；`all_selected_paths_valid` 不能解释为最终位置已被证明正确。

## 验证记录

本次使用旧实验中完全相同的三份接收 CSI 做实现检查，原数据和旧结果保留。实际 GPU 上每份观测只分解一次；GPU/CPU 的连续谱值对照测试通过。三份输入均得到定位结果：

| UE / 接收数据 | 谱面样本数 | 反向候选数 | 代表数 | 位置误差 / m | bias 有符号误差 / ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| UE001 / repeat_000 | 387 | 387 | 19 | 0.07234 | -0.07862 |
| UE002 / repeat_000 | 387 | 351 | 20 | 0.07753 | 0.01371 |
| UE006 / repeat_000 | 387 | 169 | 9 | 1.64240 | 2.62692 |

387 个样本来自 3 个峰，每峰 128 个连续样本加 1 个原始峰参考样本。反向候选会依据地图和合法 bias 区间筛选，因此数量可以少于样本数；一个样本也可能产生多个反射阶数候选。

这三份输入用于实现验收，包含此前求解失败的 UE006，不是按实验计划重新抽取的统计样本。不能用它们的汇总误差代替 30 UE × 5 次的全量定位结论。

完整重放结果位于 `outputs/spectrum_check_cuda_20260908_v2/`；每个 sample 的原始定位产物、独立指标和执行状态都在各自子目录。三份结果的位置误差中位数为 0.07753 m，P90 为 1.32942 m，仅用于核对汇总程序。

可直接打开 [最终逐步报告](../outputs/spectrum_check_cuda_20260908_v2/step_report_final/README.md)。报告保留各 UE 的 00–08 步骤及总体汇总；其中 03 的谱面样本和 05 的簇成员、代表图已经人工查看。

同一份 UE001 接收 CSI 的完整 CPU/GPU 对照通过：采样来源、候选、簇和最终选择一致；最终位置差为 0 m，bias 差为 0 ns。记录位于 `outputs/spectrum_cpu_cuda_20260908_v1/benchmark.json`。这次 CPU 定位耗时 2.550 s、GPU 1.084 s，计时排除了设备预热、信道生成、独立评估及绘图，不能视为全量实验平均加速比。

MUSIC 的协方差、分解和谱查询使用所选 CPU/GPU 后端；反向追踪的墙求交采用分块 NumPy 向量化，聚类、求解和绘图仍在 CPU 上。对 UE001、UE002 的已保存样本，新墙求交与原标量算法及旧候选文件的全部字段严格相等；RT 阶段分别从约 15.2 s 降到约 0.114 s。详细输入摘要、对照结果和计时保存在 `outputs/reverse_vectorization_check_20260908_v1.json`。

最终常规测试：490 项通过，2 项因沙箱内不能访问 CUDA 驱动而跳过；随后在实际 GPU 上单独执行 CUDA 相关检查，3 项全部通过，覆盖上述跳过项。脚本语法检查通过；真实数据的完整 CPU/GPU 对照也通过。新全量配置的采样计划已验证为 30 UE × 5 次；本次没有重新执行全部 150 份观测。
