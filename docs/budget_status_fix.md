# 路径数量、候选配对上限和退出状态（2026-09-12）

`music.path_detection.max_paths=6` 限制 CSI 提取阶段接受的观测路径数。这一步先于 MC、反向 RT、聚类和代表点生成。若六条已经解释了可检测的成分，才将这些路径交给下游；若拟合六条后残差仍支持额外成分，则本次检测没有完成。

每条通过检测的观测可以产生多个代表点；联合求解时，同一观测的代表互斥，只选一个。在当前检测配置下，一个解因此最多选中六条观测各自的一个代表，但总待选代表数可以远大于六。求解器没有写死只能处理六条观测。

`localization.max_seed_pairs=100000` 限制生成初值前的跨观测候选配对总量。数量恰好等于上限时可以继续；超过上限时，在逐对计算前直接停止。当前没有随机抽取，也没有按顺序只取前十万对。配对总数在进一步检查候选的共同合法偏差区间之前计算。

本次修正范围是退出状态分类。检测、候选生成、配对规则和配置上限沿用上一版。

| 新状态 | 中文含义 | 返回位置 |
| --- | --- | --- |
| `detection_incomplete` | 路径检测未完成，例如到达路径数量预算后仍有显著残差 | 空 |
| `solver_budget_exhausted` | 候选配对数量超限，求解尚未完成 | 空 |
| `unlocalizable` | 观测不足，或当前候选/选中传播组合不能提供足够约束 | 空 |

前两个状态的输出类型为 `no_position_computation_incomplete`，不再声称位置不唯一。`unlocalizable` 仍是当前观测与候选处理下的判定，不构成连续空间所有可能解释的全局不可定位证明。

求解预算使用独立的 `SolverBudgetError`，保存配对总数、上限及“尚未开始配对搜索”的标记。工作进程会正常返回新状态并继续接受下一份任务。逐次记录使用通用 `stop_reason`；只有物理/候选约束类状态才使用 `unlocalizable_reason`。新报告分别统计两种计算未完成状态；所有请求继续保留在计划分母中。

## 验收与旧数据重分类

执行脚本：`/data/zhujun/differt_projects/time-bias-correct/run_budget_status_check.sh`。顶部参数区设置工作目录、Python 解释器、已有实验目录、独立输出目录和 CPU 线程数；后台执行，仅使用 CPU。

本次验收输出：`/data/zhujun/differt_projects/time-bias-correct/outputs/budget_status_check_20260912_01/validation`。

- 完整测试结果在 `pytest.xml`；覆盖了达到检测上限、超过候选配对上限、恰好等于配对上限、无位置结果不能带出误差、工作进程继续处理下一份请求，以及报告分母保留。
- `validation_summary.json` 保存输入摘要和重分类依据；`reclassified_trials.jsonl` 保留原状态字段并给出新的状态；`reclassified_report` 是单独生成的报告。
- 300 条旧记录中，85 条有明确预算耗尽证据：单代表 35 条检测未完成，多代表 35 条检测未完成及 15 条求解预算不足。原始坐标输出数量仍是单代表 80、多代表 70；定位误差、耗时和原始试验文件均未改动。
- 重分类之后，单代表的 `unlocalizable` 为 35 条，多代表为 30 条。它们与两个计算未完成状态分别报告。这里没有重新执行定位，不能把这份报告作为新算法精度实验。
- 修改前全部 v3 包内源码已与冻结摘要核对并保存到 `/data/zhujun/differt_projects/time-bias-correct/outputs/budget_status_fix_20260912/source_before/time_bias_localization`，便于核对历史实现。

本次运行记录目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/budget_status_check_20260912_01/run_records/20260912T132945_2896486`；实际任务 PID 为 2896496。管理命令：

```bash
# 实时日志；Ctrl+C 只退出查看
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/budget_status_check_20260912_01/run_records/20260912T132945_2896486/task.log

# 状态、结束时间与退出码
bash /data/zhujun/differt_projects/time-bias-correct/run_budget_status_check.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/budget_status_check_20260912_01

# 核对进程身份后停止整项任务（已结束时不会停止其他进程）
bash /data/zhujun/differt_projects/time-bias-correct/run_budget_status_check.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/budget_status_check_20260912_01
```

后续重新验收时，在脚本参数区配置一个新的独立输出目录；不要用旧 v3 实验目录作为验收输出目录。
