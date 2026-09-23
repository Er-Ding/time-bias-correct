# 工作区计算性能与 GPU 审查

核对日期：2026-09-21。仓库：`/data/zhujun/differt_projects/time-bias-correct`。

结论：现有 MUSIC 已支持 GPU，本次用当前自动维数配置在 V100 实测其主要计算部分快 **12.63 倍**。主要等待时间仍来自 CPU 上逐条重建传播路径。本次修复其中重复构造墙字典的问题，同一真实观测的小规模求解耗时从 **13.24 秒降到 12.21 秒**，完整求解结果逐项一致。没有把整个连续求解器替换为 GPU，也没有改变默认实验的科学参数。

## 1. 证据范围

- 历史来源：[1000 样本实验](/data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01)。共找到 1000 个阶段计时文件，其中 999 个记录完成、1 个停留在 `running`。统计仅使用 999 个完成记录；保留未结束记录。
- 当前对照输入：[SAMPLE_000475 的 B0 定位文件](/data/zhujun/differt_projects/time-bias-correct/outputs/arm_comparison_20260922_03/report/runs/B0/SAMPLE_000475/localization)。输入含 8 条观测、1238 个传播函数；只读在线带噪 CSI、公共地图、配置、保存的观测和函数库，没有读取真值。
- 对照记录：[review.json](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353/validation/review.json)，含输入与代码 SHA-256、每次时间、设备和维数信息；输入文件在运行后重新核验未变化。
- 完成状态：[completion.json](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353/completion.json)，退出码 0。

连续求解对照固定为 8 个初值、每次最多 20 次迭代、2000 个初值组合。它核对计算变化，没有重跑全部实验，不能据此宣称定位精度得到改善。GPU 对照核对协方差、分解和粗谱；连续求解前后对照使用同一份保存的观测及函数库，不能把二者合称为完整 CPU/GPU 定位结果验证。

## 2. 时间花在哪里

历史记录按阶段的独占耗时相加，避免重复累计嵌套步骤。原始文件列表与汇总见 [historical_timings.json](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353/historical_timings.json)。这些是多个样本各自运行时间之和，不是实验开始到结束的日历时间。

| 历史阶段 | 累计时间 | 占样本总时间 |
| --- | ---: | ---: |
| 连续位置/偏差优化 | 435871.44 秒 | 98.47% |
| 构造连续传播函数 | 4672.34 秒 | 1.06% |
| MUSIC 协方差、分解、粗谱 | 1663.80 秒 | 0.38% |
| 全部完成样本 | 442624.43 秒 | 100% |

因此，即使仅把上述 MUSIC 阶段耗时压到零，历史总耗时的改善也不到 0.4%。历史运行的代码、函数库和预算与本次不同，不用历史总时间计算本次修改的加速比。

当前真实输入的首次函数耗时检查：连续求解共 18.79 秒，其中路径几何检查 15.62 秒；可见性检查调用 66846 次，累计 5.66 秒。该表中的函数存在包含关系，不相加。详见 [最初的函数耗时记录](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142452_1131689/validation/current_profile.txt)。

调用链为 `continuous_solver.local_fit/assignment → evaluation → evaluate_hypothesis → rebuild_path → reflection_leg → _segment_visible`。每次位置更新都需要检查实际墙段和遮挡，不能省去这些检查来换取速度。[local_fit 中的检查](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py:254)仍然保留。

## 3. 已修复的两处问题

### 失败诊断反复建立相同墙字典

[_leg_failure](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/propagation_model.py:209)原先在每次失败诊断时遍历全部墙段建立字典。在该小规模对照中调用 9752 次，造成重复工作。

现在直接复用 `diffraction._wall_lookup(scene)` 已存在的只读缓存。缓存按不可变场景对象区分且有数量上限。本次仅增加导入并替换一行，没有新增缓存层，没有改反射点计算、遮挡容差、失败原因、路线选择或求解预算。

| 相同输入与预算，重复 3 次、不打开函数计时器 | 中位时间 |
| --- | ---: |
| 修改前 | 13.2415 秒 |
| 修改后 | 12.2056 秒 |

耗时减少 **7.82%**，速度约 **1.085 倍**。每次重复及前后结果均做完整字典相等检查，位置、偏差、选中路线、状态、备选解和诊断计数全部一致。当前只支持这一个保存输入上的结论。

### 旧 GPU 对照遗漏自动维数设置

[benchmark_gpu_localization._spectrum_arguments](/data/zhujun/differt_projects/time-bias-correct/scripts/benchmark_gpu_localization.py:316)原先没有传递 `subspace_selection`。对于当前配置，它会按固定 6 维计算，偏离生产流程自动选择维数的行为。

已补传这个现有参数。本次正式 GPU 对照中 CPU 和 GPU 均自动选择 8 维。早先两个试跑目录 `workspace_gpu_check_20260921T142452_1131689`、`workspace_gpu_check_20260921T142640_1134815` 的 MUSIC 时间来自遗漏参数的固定维数模式，保留作审查过程记录，不用于正式 GPU 数字。

## 4. GPU 实测与使用边界

硬件为 Tesla V100S-PCIE-32GB，CuPy 13.6.0；保留 `complex128/float64`。CPU 数值库限制为单线程；两个设备分别预热一次，再重复三次，GPU 每次计时结束等待计算完成。未把初始化成本计入热运行时间。

| 同一份在线 CSI 的协方差、分解、粗谱 | 中位时间 |
| --- | ---: |
| NumPy / CPU | 1.66984 秒 |
| CuPy / V100 | 0.132175 秒 |

速度为 **12.63 倍**；谱相对 L2 差异为 **1.40e-8**，通过 `rtol=1e-4, atol=1e-8` 的逐元素检查。这里没有改变噪声、读取干净 CSI 或读取真值。

可选配置使用：

```yaml
compute:
  backend: cuda
  device_id: 0
  batch_size: 4
  angle_chunk_size: 32
```

`cuda` 是代码接受的配置值，`cupy` 是所用库名。启动器只暴露选中的物理 GPU 后，进程内编号为 0。当前默认实验配置仍为 `numpy`；可选 GPU 启动应由参数覆盖并保存最终配置，不修改已有实验文件。定位进程中的 MUSIC 对象已复用，适合连续处理多个样本；单样本新进程的初始化开销可能抵消部分收益。先限制单卡 1–2 个定位进程，不把现有多个 CPU 进程数量无条件照搬到 GPU。

旧反向候选算法的 `reverse_cuda.py` 也有 GPU 实现，但当前连续定位不再走该候选生成路径，切换旧反追踪后端不能加速当前求解。

当前求解每次只优化三个未知数，配对和几何合法性决定后续分支；直接将每个小数组转到 GPU 会增加反复的数据传输和等待。本轮不据此声称 GPU 几何实现必然更慢，尚未实现并测量这部分。进一步 GPU 改造应先把多个路径段的可见性检查合批，并逐项保持墙端点、相交容差、绕射阴影和失败原因一致；它需要单独的正确性对照，不能简单把 `np` 改名为 `cp`。

## 5. 已完成检查

- 连续几何、连续求解、可见性检查共 **133 项通过，0 失败、0 跳过**：[tests.xml](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_geometry_tests_20260921_01/tests.xml)，[结束记录](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_geometry_tests_20260921_01/completion.json)。
- 真实保存输入的小规模前后求解，整份结果逐项一致；同一 CSI 的 CPU/GPU 谱通过数值对照。
- Python 语法检查、Shell 语法检查、`git diff --check` 通过。

## 6. 复查命令

脚本：[run_workspace_gpu_check.sh](/data/zhujun/differt_projects/time-bias-correct/run_workspace_gpu_check.sh)。固定参数区包含工作目录、解释器、输入、输出、GPU 编号和检查预算。默认输出使用独立目录，并通过现有 `detached_task.py` 保存实际任务 PID、启动命令、开始/结束时间和退出码。

启动新的独立对照，复用已保留的修改前代码：

```bash
cd /data/zhujun/differt_projects/time-bias-correct
REVIEW_GPU=0 REVIEW_BASELINE_MODEL=/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353/propagation_model_before.py bash /data/zhujun/differt_projects/time-bias-correct/run_workspace_gpu_check.sh
```

本次实际日志与运行记录目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353`。以下命令重新连接后可直接使用；重新启动的新运行会打印其对应绝对路径。

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353/task.log
/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py status /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353
/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py stop /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353
```

日志查看时按 `Ctrl+C` 只退出查看。停止命令先核对监督进程身份，再停止整组任务；本次已经完成，查询或停止会显示结束记录。后台机制用于抵御终端或 SSH 断开，不提供服务器重启后的自动恢复。

## 7. C02 细分与神经网络加速评估

进一步核对日期：2026-09-21。本节分析已有代码、历史计数和已保存的函数计时；神经网络方案尚未训练或实测。

### 统计口径

[原实验的阶段汇总](/data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01/timing_summary/stage_timing_summary.json)列出 1000 个样本，但 C02 有 663 条记录，其中 662 条完成、1 条仍为运行中快照；表内中位数为 632.234 秒。它不是把全部 1000 个样本都执行一次 C02 后得到的中位数。内部计数来自保存了求解文件的 639 个成功样本，不能代表失败样本。

| 内部工作量 | 639 个成功样本的统计 |
| --- | ---: |
| 尝试初值数 | 每个样本均为 96 |
| 候选传播函数数 | 中位 4096 |
| 路径几何检查次数 | 中位 515465；累计 327004597 |
| 判为不合法的检查次数 | 累计 264640869，占上述检查的 80.93% |
| 缩短步长后重试次数 | 中位 75048 |

这些是旧实验的工作量，不能当成场景修复后重新测得的统计。汇总里的 `per_evaluation_ms` 用全部 C02 时间除以成功样本的检查次数，既混入其他计算，也混用了样本范围，不能解释为一次几何检查的实测时间。

### 实际调用链

入口是 [C02_continuous_optimization](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_pipeline.py:110)，求解状态只有 `(x, y, beta)` 三个数，其中 `beta = c * b`，以米表示共享时钟偏置。

| 环节 | 实际工作 | 对应代码 |
| --- | --- | --- |
| 生成初值 | 从观测和路线组合解初值、检查和排序，再补充区域内初值 | [_make_starts](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py:483) |
| 选择路线 | 比较预测角度、时延与观测，一对一配对；选中的路线若不合法，排除后重新配对 | [assignment](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py:157) |
| 更新位置和偏置 | 固定配对，计算解析残差和导数，解三个未知数的小型有界最小二乘问题 | [local_fit](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py:207) |
| 检查新位置的物理合法性 | 重建反射、绕射路线，检查墙段、遮挡等；不合法或代价未下降就缩短步长重试 | [回退循环](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py:254) |
| 汇总各个初值的解 | 检查收敛、残差和约束是否充分，比较不同解，保留多解或失败状态 | [求解主循环](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py:665) |

中间三个环节交替重复。当前默认最多 96 个初值，每个初值最多 8 轮重新配对，每次局部求解最多 80 次迭代，每次更新最多 30 次步长尝试。实际通常会提前结束，不能把这些上限直接相乘来估计时间。C02 之后另有一次独立正向检查。

### 已有函数计时说明了什么

[修改后的函数计时](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_gpu_check_20260921T142849_1138353/validation/current_profile.txt)使用同一个保存输入：8 条观测、1238 个函数、8 个初值、每次最多 20 次迭代、2000 个初值组合。打开函数计时器时 C02 约 17.65 秒；不打开时的重复运行中位数为 12.21 秒，不能混用两种口径。

| 函数范围 | 调用次数 | 累计耗时 |
| --- | ---: | ---: |
| 整个 C02 | 1 | 17.649 秒 |
| 局部迭代 `local_fit`，包含其内部全部检查 | 14 | 16.260 秒 |
| 物理路径求值 `evaluation`，分布在配对和局部迭代中 | 35380 | 14.508 秒 |
| 局部迭代中检验提议位置的路径，属于上述两行 | 37587 次生成器调用 | 13.577 秒 |
| 遮挡检查 `_segment_visible`，属于路径求值内部 | 66846 | 5.656 秒 |
| 生成和排序初值 `_make_starts`，也包含部分路径求值 | 1 | 0.441 秒 |
| 一对一配对算法 `linear_sum_assignment` 本身 | 1296 | 0.064 秒 |

**这些时间有包含关系，不能相加，也不能按比例摊到旧实验的 632 秒中。** 小规模实测的瓶颈是反复试探位置和检查物理路线；配对算法本身仅占约 0.36%。只替换这个配对算法很难带来明显加速。解析角度、距离和导数已有直接公式，也不是优先用网络近似的对象。

### 优先尝试的神经网络方案

首先只针对修复后的固定地图和固定基站：将已有带噪 MUSIC 角度、时延观测交给一个小网络，输出 4 或 8 组 `(x, y, beta)` 初值；随后复用当前路线配对、局部优化和物理检查。未得到可靠结果时继续原有完整搜索。少量初值应覆盖不同可能位置，单一坐标回归可能在多解时输出两个位置之间的平均点。

网络输入应保留原始时延中的共同偏移，不能只使用路径间时延差而丢掉待估计的时钟偏置。角度可用正弦、余弦表示；观测数量不定、顺序无意义，先用逐条编码后汇总的小型集合网络即可。[Deep Sets](https://arxiv.org/abs/1703.06114)提供这类与输入顺序无关的结构。地图与基站先固定在模型适用范围内，跨地图推广需要另做实验。

用网络学习优化初值已有方法依据，例如 [Learning to Warm-Start Fixed-Point Optimization Algorithms](https://www.jmlr.org/papers/v25/23-1174.html)。这是本项目方案的参考，不是对当前含离散路线选择和几何边界的求解器作收敛或加速保证。

代码接入时有一个现成接口和一个限制：`initial_states` 已接受外部初值，但 `_make_starts` 仍会执行解析初值枚举，并补齐到 `max_starts`。只传入网络预测、仍保留 96 次搜索，并不能实现计划中的搜索削减；需要明确区分“先跑少量候选”和“必要时完整搜索”，复用同一个求解器。

次选是网络预测路线可行性或优先级，用于减少低价值尝试。它不能直接充当最终合法性判据；把本来合法的路线误删，会使正确位置无从恢复。直接让网络输出最终位置和偏置可以作为实验对照，但第一版不据此替换物理求解。

### 最小验证设计

| 对照 | 用途 |
| --- | --- |
| A：现有 96 初值搜索 | 精度、状态和时间的基准 |
| B：现有方式生成 4/8 个初值，必要时完整搜索 | 判断仅减少搜索能取得多少收益 |
| C：网络生成 4/8 个初值，使用与 B 相同的局部预算和回退规则 | 判断网络是否比普通初值更有效 |

固定场景版本、观测、信号设置、局部求解和物理接受条件；训练、验证、测试按位置区域分组，同一位置的不同噪声版本不能随机分散到训练和测试两边。旧地图数据不能不加区分地作为修复后地图的标签。仿真真值只用于离线训练与独立评估，旧求解器返回成功的坐标也不能直接当作无误差真值；在线输入不能带入真值、干净 CSI、注入噪声或实验设定的 SNR。

计时必须包含网络推理、短搜索以及触发的完整搜索，报告所有测试样本的中位、P90 时间，并分别列出成功、失败、多解等状态。同步比较位置误差、偏置误差、错误接受、遗漏多解、完整搜索触发率，以及几何检查和回退次数。物理检查通过不代表解唯一，少量初值遗漏竞争解的风险必须单独评估，不能仅靠网络置信度解决。

下一次完整预算复测应记录互不重复的时间：初值生成和排序本身、配对代价与配对算法、解析残差和导数及小型线性求解、物理路径检查、其余控制与最终判定；每项附调用数。几何检查再标明来自初值排序、配对还是局部步长尝试。汇总时先在每个样本内加总其全部角度分支，再跨样本取中位；不能把各环节中位数相加当成总中位数。现有记录足以确定优先方向，还不能给出 632 秒内部每项的整批中位时间，也没有 NN 加速倍数的实测结论。
