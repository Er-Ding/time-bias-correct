# v6：观测筛选、绕射代表数量上限与 RANSAC

本版保留原始 MUSIC 直接读峰及其局部细化，不恢复逐次剔除 CSI 的残差检测。
输入继续使用 v3 保存的 30 个 UE、每点 5 次噪声观测，目标 SNR 为 35 dB。
旧结果和原始观测只读引用，不覆盖。所有参数是本轮固定的实验设置，尚未证明最优。

## 参数和含义

参数集中在 `/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_generation_v6.yaml`。
实验规模和重复次数在同目录的 `diffraction_boundary_accuracy_v6.yaml`。

| 参数 | 本轮值 | 含义 |
|---|---:|---|
| `music.observation_screen.enabled` | true | 是否排除存在近乎相同信号响应峰对的 CSI 请求 |
| `music.observation_screen.max_response_correlation` | 0.995 | 完整天线×子载波响应的归一化内积绝对值门槛 |
| `localization.diffraction_cluster_representative_max` | 16 | 每个绕射簇的代表数量上限 |
| `localization.diffraction_representative_max` | 64 | 同一条观测的全部绕射簇合计代表上限 |
| `localization.diffraction_coverage_distance_m` | 1.0 m | 补点时争取达到的成员覆盖距离 |
| `localization.solver_method` | ransac | `exhaustive` 可保留旧求解作为对照 |
| `localization.ransac.max_trials` | 2048 | 每次请求的最大抽样次数，失败抽样也计入 |
| `localization.ransac.inlier_distance_m` | 2.0 m | 一条观测支持当前位置与偏差的最大几何残差 |
| `localization.ransac.max_refinements` | 24 | 最多细化的候选解数量 |

把代表上限设为 `null` 可取消对应层的限制。`diffraction_directions_per_sample=4`
仍是反向 RT 每个采样的绕射出射方向数，不是代表数量上限。

## 观测筛选

对局部细化后的 MUSIC 峰构造现有信号模型中的天线和子载波响应，比较完整响应的
归一化内积绝对值。角度相似但时延已经分开的峰不会仅因角度而被排除。
规则只读取观测峰和公开阵列参数，不读取真实路径、真实位置、干净 CSI、真实噪声功率、
注入 SNR 或定位误差。命中规则时跳过整份 CSI 的后续 RT 和求解；不删除源文件。

状态为 `excluded_observation`，原因为 `near_identical_music_responses`，不会记为程序失败、
计算预算耗尽或物理上无法定位。峰对及相似度保存在失败步骤目录和 `trials.jsonl` 中。
这里筛选的是“观测峰解释近似重复”的输入，并非通过真值证明所有这些输入物理上无法区分。
不直接把峰合并，也不修改 MUSIC 的搜索角度区间。

在 v5 已保存的 120 份进入定位的峰集合上检查，新规则命中 43 份，涉及 9 个 UE；另外
30 份只有一条峰，不存在可比较的峰对。因此按原始 150 份输入计，预期排除 28.7%，
并非极少数。筛选门槛在本轮运行前固定，不能事后按照定位误差选择保留样本。

报告保留以下口径：

- `trials.csv`：全部原始计划请求，排除项也保留。
- `excluded_observations.csv`：排除清单、原始输入摘要、峰对、输出路径。
- `summary/precision.csv`：原始计划分母和筛选后已执行分母分别报告；观测不足及预算退出
  仍在筛选后分母内。原始输出率不会因排除而重新归一化。

完整总体结果与筛选后结果反映不同的适用范围。筛选不能证明方法在所有原始输入上有效。

## 代表点数量限制

DBSCAN、传播顺序分组和来源观测分组保持原定义。每个绕射簇从中心实际成员开始，
按最远距离补点，再补足整个合法偏差区间的覆盖；达到每簇上限就停止补点。
整个观测超出总上限时，先给各簇轮流保留中心，再按原补点顺序轮流保留额外代表。
簇数本身超过总上限时，优先保留原始成员数多的簇，并用稳定的簇标识打破并列。
这只是计算预算分配，不把成员数解释为路径真实性或独立观测数。

被删代表及被整簇删去的传播分支全部记录。每个保留代表仍保存实际方向和合法偏差区间。
触及上限且仍有未覆盖成员时，完整覆盖标志为 false；总上限删点后的偏差区间覆盖没有
重新验证时，其未覆盖长度记为未知，不继续沿用删点前的“完整覆盖”结论。
单代表对照仍为每簇一个实际成员；同一观测的绕射代表总上限对两组均生效。

## RANSAC 的求解方式

每次先均匀抽两条不同观测，再在各观测的簇中抽取候选。簇内以一半概率选中心，另一半
概率均匀选成员；随机种子来自该请求已经冻结的 MC 种子。若两条候选均为绕射，再抽第三条
不同观测。两条纯绕射观测不会因采样了出射方向而被当作足够定位的约束。

所选候选必须有共同合法的偏差区间，拟合二维位置和 `beta=c*b`，拒绝退化或明显不能相交的
组合。然后对每条观测寻找一个合法且最近的代表，2 m 内算支持，最多一票。先比较支持观测
数，再比较封顶平方残差。缺失候选与所有候选均不支持当前解时，代价相同；不会因新增一个
百米外的候选而惩罚原本的正确解。最后使用支持该解的观测重新拟合，直到选择与拟合一致。

本版仍在已有离散代表轨迹上拟合，绕射方向不是新的测量。候选解还需通过不把绕射方向算作
已知量的物理约束秩检查。协方差仍以选中的离散代表为条件，未标定，不代表全局唯一性。
不同位置的其他高分解会保存为诊断。有限抽样不承诺“99% 找到正确分支”；预算内没有合法稳定
解时记为 `solver_budget_exhausted` / `ransac_search_budget_exhausted`。

RANSAC 的随机抽样、按支持观测评分和局部重拟合参考了
[OpenCV 的 USAC 官方说明](https://docs.opencv.org/4.13.0/de/d3e/tutorial_usac.html)，
最小观测组合和残差使用本项目的位置与时间偏差模型，没有使用相机的五点法或八点法。

## 后台执行

启动脚本是 `/data/zhujun/differt_projects/time-bias-correct/run_ransac_experiment.sh`，必须显式填写
`RANSAC_GPU_IDS`。每次设置新的 `RANSAC_OUTPUT_ROOT`，脚本拒绝覆盖已有目录。
它沿用已有 `nohup + setsid` 启动器，关闭标准输入，保存实际任务 PID、日志、启动参数、
开始/结束时间和退出码。启动后会打印可直接复制的实时日志、状态和停止命令。

本次使用的运行目录：
`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_ransac_v6_20260914_01`。

```bash
# 查看本次实际日志，Ctrl+C 只退出查看，不会停止后台实验。
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_ransac_v6_20260914_01/run_records/20260914T090031_2366463/task.log

# 查询状态，以 completion.json 的退出码判断是否成功。
bash /data/zhujun/differt_projects/time-bias-correct/run_ransac_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_ransac_v6_20260914_01

# 核对进程身份后停止整项任务，包括子进程。
bash /data/zhujun/differt_projects/time-bias-correct/run_ransac_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_ransac_v6_20260914_01
```

CPU 检查脚本为 `run_ransac_check.sh`，默认后台运行完整回归；参数和日志位置在脚本的参数区。
本次完整检查 813 项通过、23 项跳过，退出码为 0；证据在
`/data/zhujun/differt_projects/time-bias-correct/outputs/ransac_v6_check_all_20260914_01/validation/validation_summary.json`。
代码检查通过不等于本轮定位精度已得到证明；定位结论以固定 CSI 复测结果为准。
