# RT 候选数量异常：BS 一侧几何检查缺失

此前墙段合并修复只消除了地图中的重复记录。五个样本仍有 2591～2835 个候选，是因为候选生成没有提前排除两类对任何 UE 位置都不成立的走法。本次已经补上这两项检查，五个样本保留 79～84 个候选；没有人为设置 100 条上限。

## 两个具体问题

1. **最后一面反射墙完全被挡住。** 例如候选写成“UE → 墙 C → 墙 B → BS”，但墙 A 把墙 B 完全挡住，则“墙 B → BS”这一段必然穿过墙 A。当前模型不包含穿墙传播，所以整条走法可以直接排除，无须知道 UE 在哪里。原程序在角度窗口内排列墙面，却没有做这项整段遮挡检查，直到后续逐点重建才拒绝路径。
2. **把反射墙两侧的点连成反射路径。** 连续两次反射时，前一面墙若整体与 BS 分处最后一面墙的两侧，则入射和出射段不满足镜面反射条件。原候选生成只检查展开后的角度窗口相交，遗漏了这一必要条件。

现在，在构造和收录候选之前检查这些条件。遮挡判据要求从 BS 到目标墙两个端点的连线都严格穿过同一挡墙；这给出整段墙处于挡墙后方凸阴影区域的充分证据。只挡住一部分、擦边、共线或数值边界情况保守保留。墙仍在地图中，也仍可参与其他合法的传播顺序；没有删除遮挡墙本身，也没有把远处墙一概删除。

实现位置：[propagation_hypotheses.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/propagation_hypotheses.py:122)。
回归检查：[test_receiver_geometry_pruning.py](/data/zhujun/differt_projects/time-bias-correct/tests/test_receiver_geometry_pruning.py)。

## 实际结果

输入地图和观测来自墙段修复后的固定记录：
`/data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_20260918_02/validation`。

| 样本 | 本次修复前候选 | 因完全遮挡排除 | 再因反射条件排除，已扣除重叠 | 修复后候选 | 在真实位置能走通的不同路径 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 000103 | 2651 | 2558 | 12 | 81 | 2 |
| 000414 | 2835 | 2736 | 15 | 84 | 2 |
| 000685 | 2591 | 2497 | 11 | 83 | 2 |
| 000748 | 2762 | 2666 | 14 | 82 | 2 |
| 000820 | 2749 | 2656 | 14 | 79 | 2 |

以 000103 为例：2651 − 2558 − 12 = 81。81 表示未知 UE 位置时仍需考虑的走法；在该样本的同一个真实位置上，其中只有 2 条能走通。这个 2 仅统计观测筛选后候选库中的有效几何路径，不是对全场景不限观测方向的完整正向 RT 计数。

两个原因可以重叠。例如 000103 总共有 321 个候选违反反射侧别条件，其中 309 个已经计入完全遮挡的 2558 个；不能把 2558 和 321 直接相加。

候选库先生成并保存，随后才读取真实位置，独立检查修复前的每一个候选。五个样本各自原本能走通的 2 条路径均被新库保留；按路径节点保留六位小数去重后，均为 2 条，没有重复。没有将真实位置反馈给筛选。新候选集合与“旧集合扣除上述两类走法”严格一致。

224 项回归检查通过，0 项失败，0 项跳过。覆盖整段遮挡、部分遮挡、擦边、端点顺序、旋转后的墙面、合法反射/绕射路径保留，以及已有连续定位与绕射检查。日志中的一次除法警告来自已有的极短交叉墙测试，测试通过，不是退出错误。

本次候选复算在原传播阶数和观测筛选条件内均完成，仍保留原 4096 条容量、50000 次枚举预算；1800 秒定位时限没有修改。没有重跑五个样本或整批的完整位置优化，因此没有据此更新定位成功率，也不报告端到端加速比。

“少于 100 条”不是所有场景的硬性物理上限。内部待验证候选与固定收发位置的有效路径应分开统计；Sionna 官方也将候选生成与路径重建、有效性检查区分：[Path Solver 技术说明](https://nvlabs.github.io/sionna/rt/tech-report/S3.html)。本次数量下降来自有明确几何证据的排除。

## 结果与复现

最终报告：[report.md](/data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_20260918_02/validation/report.md)。
逐样本统计：[samples.json](/data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_20260918_02/validation/samples.json)。
输入与代码指纹：[provenance.json](/data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_20260918_02/validation/provenance.json)。
每个样本目录内保存新候选库、每条被排除候选对应的挡墙证据、以及独立固定位置检查结果。原实验输入保持不变。

第一轮 `rt_candidate_audit_20260918_01` 保存了一次测试场景错误：纯反射测试选择了被完全遮挡、没有任何参考路径的位置，失败发生于参考路径非空断言，尚未运行真实样本审计。修正测试位置后，第二轮完整通过。第一轮记录保留，不能作为成功结果。

工作目录：`/data/zhujun/differt_projects/time-bias-correct`。
解释器：`/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python`。
参数集中在 [run_rt_candidate_audit.sh](/data/zhujun/differt_projects/time-bias-correct/run_rt_candidate_audit.sh) 顶部。

已完成运行的实际记录目录为
`/data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_20260918_02/run_records/20260918T030800_2050788`，
实际日志为该目录中的 `task.log`；实际任务 PID 为 2050908，已于 UTC 2026-09-18 03:08:41 结束，退出码 0。

重跑时同时替换下列命令中的输出目录名，已有目录拒绝覆盖。任务使用 nohup、setsid 和无缓冲 Python，脱离当前终端运行。

启动：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_rt_candidate_audit.sh start /data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_manual_01
```

实时查看日志，按 Ctrl+C 只退出查看：

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_manual_01/task.log
```

查看运行状态：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_rt_candidate_audit.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_manual_01
```

停止整项任务，管理器会先核对进程身份再停止进程组：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_rt_candidate_audit.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/rt_candidate_audit_manual_01
```
