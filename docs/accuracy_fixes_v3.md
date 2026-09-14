# 定位精度修复 v3

2026-09-12 补充：计算预算退出的状态已单独修正，见 [预算与状态说明](/data/zhujun/differt_projects/time-bias-correct/docs/budget_status_fix.md)。原 v3 统计保留当时的状态标签；重分类报告在独立目录中。

代码入口：[run_accuracy_fixes.sh](/data/zhujun/differt_projects/time-bias-correct/run_accuracy_fixes.sh)。
实验配置：[diffraction_boundary_accuracy_v3.yaml](/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_accuracy_v3.yaml)。
信号及算法参数：[diffraction_boundary_generation_v3.yaml](/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_generation_v3.yaml)。

旧配置保留原算法行为，便于复现历史结果；v3 配置启用以下三个独立开关：

| 开关 | v3 值 | 行为 |
| --- | --- | --- |
| `music.path_detection.enabled` | `true` | CSI 残差检验通过的路径才进入 MC |
| `localization.candidate_bias_mode` | `full_interval` | 覆盖整个允许偏差范围的反射、绕射末段 |
| `localization.require_identifiable_solution` | `true` | 观测或局部物理约束不足时返回无法定位 |

## 路径检验

采用 [NOMP](https://arxiv.org/abs/1509.01942) 的残差检验和连续细化思路，具体顺序如下：

1. 只从观测 CSI 估计噪声。对频率轴施加周期 Hann 窗，计算时延响应，以功率中位数估计噪声底，再按窗口能量折算回 CSI 功率。
2. 对残差执行完整二维角度—时延匹配搜索。每条已接受路径的复系数参与重新拟合；每次增加路径后共同细化角度、时延和复系数。
3. 用纯模拟噪声校准**整张二维搜索面的最大统计量**。使用实际天线、子载波和网格、同一噪声估计方法，以及已拟合 CSI 列空间的投影。模拟噪声不会加到实际观测上，也不读取生成 SNR 或真实噪声。
4. 误检预算分给最多 `max_paths + 1` 次完整搜索。经验检验值为 `(1 + 模拟最大值超过观测最大值的次数) / (重复数 + 1)`；重复数不足以分辨门槛时配置直接报错。连续细化不能绕过前面的全网格检验。
5. 对 CSI 相关度超过门槛的两条候选，比较删除任一条以及合并后的模型，每个模型都重新拟合全部剩余参数。增加成分没有显著收益时，保留较简单的解释，不把同一成分算成两条观测。
6. 接受的路径数用于 MUSIC 子空间维数。随后计算观测 MUSIC 谱并构造 MC 提议分布；已经通过 CSI 验收的中心不会被局部谱最大值替换。

`num_paths` 和旧 `signal_subspace_rank` 在开启检测后不再决定在线接受数量；`max_paths` 是明确记录的计算上限。达到上限后仍有显著残差时报告未解决，不能悄悄把路径集合当作完整。

门槛是**在拟合几何和白噪声模型条件下的模拟校准**。它考虑二维搜索和多次搜索预算，但不能声称对所有真实信道都严格保证给定误检率：几何本身由数据拟合，噪声由观测估计，有色噪声、密集时延分量或阵列模型误差可能改变分布。独立噪声重复用于检查实际行为，不能把少量重复的结果当作严格概率上界。

## 整个偏差范围内的候选

每个采样角度沿反向射线逐段行进。到达某段起点前的距离为 `s`，本段长度为 `ell`，则这段在满足 `0 < c*(tau-b)-s < ell` 的偏差范围内存在。解析求出该区间与用户允许偏差区间的交集；只要交集非空，就保留该段，不要求它在 `b=0` 下有效。

反射改变下一段方向；绕射展开允许方向，并继续处理后续反射。保存完整反射墙、绕射边缘及交互顺序。每个末段有独立编号，但沿用原始来源观测，求解器依然只能为每条观测选一个解释。

进入 DBSCAN 前，同一来源、同一传播顺序的成员优先使用原参考偏差。只在其他偏差下成立的成员，根据有效区间选择共同的合法参考值；无法共用参考值的成员分组处理。每个初始点都在真实合法线段上，不把墙外的外推点当作合法 UE。

DBSCAN 的距离、密度门槛和噪点处理规则不变；同一簇的成员始终共用参考偏差。单/多代表策略只在绕射簇内不同。多代表的连续偏差覆盖检查和同一观测候选互斥机制继续保留。

“完整”只指已采样角度、绕射方向及配置反射/绕射阶数下的初始末段。有限方向采样、DBSCAN 和代表压缩仍可能造成损失，不能据此承诺连续空间或最终定位精度的完整覆盖。

## 无法定位与统计

新增正常终态 `unlocalizable`：包括可靠路径不足、聚类后独立观测不足、找不到可求解组合、以及局部物理约束不足。不会作为程序崩溃重启工作进程。

此时返回坐标为 `null`，原因和已完成阶段保存在每次结果目录下的 `localization_unavailable/<run_id>/progress.json`。已经算出的坐标可以放在明确标记为“仅供诊断”的字段里，但不能通过计时记录回流为定位精度结果。

现阶段结果判定检查局部物理约束是否足够；不宣称证明全局唯一性。正向几何检查仍作为独立诊断保留，原求解目标函数不变。

报告保留全部计划请求作为成功比例和距离达标比例的分母，分别统计无法定位、程序失败、超时和未执行。误差分位数只针对有坐标输出的样本，必须与输出比例一起看。

新增独立计时：观测噪声估计、整张二维搜索门槛校准、残差搜索、联合 CSI 拟合、重复解释检查和合法参考点分组。嵌套耗时仍按独占时间汇总，避免重复相加。

## 数据与运行

原 `outputs/diffraction_boundary_v1` 当前为空，v2 的原始 CSI 引用不可用。验收中的保存观测重放只验证几何修复，不能验证重新提取 MUSIC 的效果。

v3 预跑会在独立输出目录重新生成数据。固定一个 BS、覆盖区域内均匀随机选 30 个 UE、每个点 5 次独立噪声；单代表和多代表共用同一新 CSI 和 MC 随机种子。不得把新旧批次当成相同 CSI 的严格算法对照。

启动脚本默认后台执行，参数集中在顶部。每次必须填写 GPU 物理编号；仅 CPU 验收可填 `cpu`。`validate` 执行完整测试、独立噪声实验、CPU/GPU 一致性检查及六个旧 UE 的保存观测重放；`pilot` 执行真实 30×5 对照。

每次运行都有独立日志、GPU 对应记录、实际任务 PID、启动参数和开始/结束/退出码。启动器打印可复制的日志、状态和停止命令。查看日志时按 Ctrl+C 只退出查看，任务继续运行。失败目录保留，不自动覆盖。

## 本次验收记录（2026-09-12）

最终验收目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/accuracy_fixes_validation_20260912_02`。
运行在 08:01:04 UTC 开始，08:04:06 UTC 结束，退出码为 0。

| 检查 | 结果 | 证据文件（相对于验收目录） |
| --- | --- | --- |
| 完整测试 | 766 通过、21 跳过、0 失败 | `validation/pytest.xml` |
| 独立纯噪声 | 100/100 次接受 0 条路径 | `validation/detector_trials.json` |
| 独立单路径观测 | 30/30 次接受 1 条路径 | `validation/detector_trials.json` |
| 强路径与弱路径并存 | 30/30 次接受 2 条路径 | `validation/detector_trials.json` |
| CPU/GPU 检测对照 | 三类观测各检查一次，结果一致 | `validation/detector_trials.json` |
| 旧保存观测的几何重放 | 六个 UE 的旧候选均能重现；新候选补回合法分支 | `validation/replay_summary.json` |
| 验收期间代码一致性 | 结束时重新计算并比较源文件摘要，通过后才写入完成结果 | `validation/source_before.json`、`validation/validation_summary.json` |

跳过项主要是需要单独开启的旧反向 RT CUDA 测试，不能将其计为通过；新路径检测的 GPU 对照已执行。独立观测检验使用 v3 的检验概率、搜索次数预算和模拟校准次数，但使用较小阵列和搜索网格的合成观测；它验证实现行为，不代替真实场景复测或严格概率保证。

还修正了低幅度 CSI 的数值问题：联合拟合按输入范数缩放残差，避免 RT 信号幅度很小时优化器提前停止。这项改动及其幅度缩放回归检查已包含在上述最终验收中。

保存观测重放中，PILOT_0026 第一个来源的最近合法候选从 102.08 米降到 0.74 米；PILOT_0030 从 32.82 米降到 0.31 米。这些是**初始候选覆盖距离**，不是修复后的定位误差。

## 已启动的真实场景预跑

本次明确授权 GPU 物理编号 `0,1,2,3,4,5,6,7`。指定可用范围不表示程序会占满八张卡；定位对照按固定顺序执行，减少并发争用对阶段计时的影响。

- 输出目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3`
- 运行记录目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3/run_records/20260912T080524_2414307`
- 实际任务 PID：`2414323`；开始时间：`2026-09-12 08:05:24 UTC`。
- 配置、输入、代码摘要已冻结；每个 UE 五次噪声，两种代表策略，共 300 次定位请求。
- 新旧 UE 坐标、噪声种子和 MC 种子一致；只有 35/150 份观测文件摘要相同。逐份比较记录：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3/data_comparison_to_v2.json`。文件摘要不同的观测不应宣称为相同输入。

本次已经启动，无需重复执行启动命令。重新连接服务器后可直接执行：

```bash
# 实时查看日志；Ctrl+C 只退出查看
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3/run_records/20260912T080524_2414307/task.log

# 核对真实进程身份，并查看结束记录及退出码
bash /data/zhujun/differt_projects/time-bias-correct/run_accuracy_fixes.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3

# 核对进程身份后停止整项任务，包括子进程
bash /data/zhujun/differt_projects/time-bias-correct/run_accuracy_fixes.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3
```

后续新批次在脚本顶部参数区修改 `ACCURACY_MODE`、`ACCURACY_GPU_IDS`、`ACCURACY_CONFIG_PATH` 和 `ACCURACY_OUTPUT_ROOT`；每次使用独立输出目录。上述路径只管理本次运行。

## 预跑中的剩余问题：PILOT_0008

针对已经完成的第 4 份噪声观测，两种策略的独立阶段回溯已结束，退出码为 0。
证据目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/accuracy_fixes_v3_pilot0008_audit_01/artifacts`；
逐条证据在 `audit.jsonl`，可读表格在 `audit.csv`，真值读取只发生在这一评估脚本中。
这不是整个预跑的汇总结果。

- 单代表误差 17.92 米，多代表误差 19.97 米；两者均有选中路径未通过正向几何检查。
- 两条强路径的代表在真实偏差下分别距真值约 0.05、0.10 米，说明这两个正确解释已经进入求解器。
- 第三个、第四个来源的最近候选在聚类前约为 0.73、0.81 米，聚类后为 2.23、3.29 米，单代表后为 4.32、5.09 米。第五个来源在聚类前为 0.33 米、聚类后为 10.12 米。这些是剩余集合的最近距离，并非同一点发生位移。
- 第六个来源时延接近真实弱路径，但检测角度为 +85.02 度，真实角度为 -86.47 度。在本次 12 天线、半波长间距的阵列上，两种角度的归一化阵列响应相关度约为 0.9981。存在额外信号的证据，不等于该信号的角度已被唯一确定；只在一个角度附近做 MC 可能遗漏另一种解释。
- 单代表的真值处目标值为 160.35，错误输出处为 138.28；即使在评估侧从真值位置和真值偏差开始求解，仍然移到约 17.92 米误差的位置、偏差上界 80 ns。这说明该样本还需要处理方向不确定性、稀疏绕射候选压缩及后续评分，不能仅归因于优化初值。

这批实验保留已经冻结的 DBSCAN 和求解目标函数，避免边跑边改造成不可比较。完整复测后应结合所有 UE 的结果决定下一轮改动；现阶段不宣称定位精度已达标。
