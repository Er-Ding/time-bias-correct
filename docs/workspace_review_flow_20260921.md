# 2026-09-21 主流程与近期场景修复检查

本轮保留已有未提交修改，在现有代码上修复五类流程问题。最终局部检查为 **115 项通过、0 项失败，退出码 0**。检查涵盖连续定位、角度分支、旧采样链、墙段合并、基站侧几何排除与绕射；没有重跑 1000 个场景样本，不能把测试通过当成定位精度提升。

## 已修复的问题

| 问题 | 原来可能产生的影响 | 本次修改 |
| --- | --- | --- |
| 角度分支选择后覆盖正式峰，筛选记录仍使用选择前的编号 | 冻结回放可能越界，也可能把两个真正保留的观测再删成一个 | 同时保存选择前的峰和来源编号，回放优先使用完整输入；旧产物若已丢失峰且编号不匹配，明确拒绝错误回放 |
| 所有分支失败后统一写成 `excluded_observation`，没有候选代价时还会丢失原因 | 预算不足、几何失败和多解被混入“观测剔除”，统计分母与故障判断失真 | 保留求解器的真实状态和原因；只有观测筛选主动排除才记为观测剔除 |
| 伪峰剔除只改正式峰，旧采样链继续用全部样本 | 已删观测仍进入逆向射线追踪、聚类及求解；结果与保存的正式峰不一致 | 同步筛选样本、样本记录、局部区域与数量统计 |
| 冻结回放只读旧筛选记录，忽略关闭开关，`exclude_sample` 也会继续求解 | 同一配置从在线入口和冻结入口得到不同流程 | 根据固定接收机信息重新计算观测相似度，使用当前开关、门限和策略；不重新提取 MUSIC 峰 |
| 不同角度分支各自能定位时，无条件选最低代价并输出成功 | 两个明显不同的位置都能解释观测，却被当成唯一位置 | 复用已有代价、位置、偏差门限检查跨分支多解；不同分支同样成立时输出 `ambiguous`，不发布位置 |

代码位置：

- [pipeline.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/pipeline.py:1899)：删除观测后的样本同步；保存完整分支输入；错误状态传递。
- [continuous_pipeline.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_pipeline.py:197)：分支选择、跨分支多解检查和冻结回放。
- [test_observation_ambiguity.py](/data/zhujun/differt_projects/time-bias-correct/tests/test_observation_ambiguity.py)：在线输出再回放、旧记录拒绝、失败状态、开关与策略、跨分支多解。
- [test_spurious_peak_filter.py](/data/zhujun/differt_projects/time-bias-correct/tests/test_spurious_peak_filter.py)：直接检查送入逆向追踪与保存记录的观测编号。

## 多解判断的边界

没有增加新的判断门限。直接复用连续求解器已有的 `ambiguity_cost_tolerance`、`distinct_position_m`、`distinct_bias_m`，默认分别为 0.5、0.25 m、0.25 m。位置相同但公共距离偏差明显不同，也属于不同解。

开启幅值加权时，不同分支可能采用不同的误差尺度，不能仅凭代价较小宣布另一个有效位置错误。因此只要其它分支存在明显不同的有效解，就保守返回多解，并记录代价不可直接比较。未开启加权时，只有代价落在现有容差内的竞争解会阻止位置输出。另一个分支自身为多解时，其保存的有效候选也参与这项检查。

多解时，位置、偏差和协方差正式输出为空；各分支的状态、位置与代价继续保存在诊断中。代表分支的正向检查仍是物理检查记录，会注明它只属于诊断候选。

这会使部分历史 `success` 变为 `ambiguous`，也会使曾误写为 `excluded_observation` 的记录恢复成预算不足或几何失败。**旧实验文件未被修改，旧成功率和剔除率不能直接作为修复后结果，必须在独立输出目录重跑受影响流程。**

## 场景与真值隔离检查

已阅读墙段合并、网格切片、基站侧整段遮挡排除、反射同侧判断、绕射墙查询缓存及其调用关系。本子审查没有改动这些几何实现；相关检查覆盖真实墙缝保留、重叠墙段合并、部分遮挡、擦边、旋转、反向端点、合法反射/绕射路径保留。主任务随后单独处理的极短墙方向除零问题，以总报告及对应检查记录为准。

核心连续入口仍只接收公开地图、观测峰、基站位置和接收机配置。冻结入口的新增筛选使用输入 CSI 的公开天线/频率信息；没有 CSI 文件时使用已校验配置中的固定接收机信息，不读取真实位置、注入噪声或真实时钟偏差。既有“删除真值文件仍能完成在线定位”的检查通过。

本轮没有对全部离线分析脚本逐一完成真值使用审计，也没有重新测量新地图的整批定位精度。已有场景结果与本次代码检查应分别记录。

## 仍需保留的限制

- 连续求解使用有限初值。跨分支多解检查能阻止已发现的竞争解被忽略，仍不能证明整个地图上不存在未发现的解。
- 幅值加权是经验规则；MUSIC 谱值没有在本轮被校准成真实测量方差或路径功率。
- 角度组的组合数可能迅速增长。当前保留全部分支结果，许多歧义组会增加时间和内存。本轮未引入另一套预算参数；需要结合实际分支数量与现有工作进程时限单独评估。

## 检查记录与复现

最终运行：

- 目录：[workspace_review_flow_tests_20260921_03](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_tests_20260921_03)
- 日志：[task.log](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_tests_20260921_03/task.log)
- 结束状态：[completion.json](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_tests_20260921_03/completion.json)
- 测试明细：[tests.xml](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_tests_20260921_03/tests.xml)

结果为 115 项通过，用时 28.70 秒；实际任务 PID 为 1141578，已退出且结束记录的退出码为 0。还检查了本次修改的空白/补丁格式。前两轮日志仅记录过程，最终依据第三轮。

工作目录 `/data/zhujun/differt_projects/time-bias-correct`；解释器 `/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python`。复用本轮统一维护的 [run_tests.sh](/data/zhujun/differt_projects/time-bias-correct/run_tests.sh)，不再增加第二套后台管理器。修改脚本顶部参数区或最后的测试文件列表即可。

以下示例使用全新目录；重复运行需要同时更换四条命令中的目录名：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_tests.sh start /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_manual_01 tests/test_observation_ambiguity.py tests/test_spurious_peak_filter.py tests/test_continuous_pipeline.py tests/test_standard_music.py tests/test_scene_and_candidates.py tests/test_receiver_geometry_pruning.py tests/test_wall_segment_union.py tests/test_diffraction_workflow.py tests/test_continuous_hypothesis_search.py
```

实时查看日志；按 Ctrl+C 只退出查看，不停止后台任务：

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_manual_01/task.log
```

查看状态，及核对进程身份后停止整项任务：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_tests.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_manual_01
bash /data/zhujun/differt_projects/time-bias-correct/run_tests.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_flow_manual_01
```

实际日志和运行记录分别位于上述绝对输出目录的 `task.log` 与该目录本身，包含实际任务 PID、启动参数、开始/结束时间和退出码。

## 补充：对照实验汇总

[analyze_arm_comparison.py](/data/zhujun/differt_projects/time-bias-correct/scripts/analyze_arm_comparison.py) 另有两处统计问题，已修复：

- 原来只把已返回的结果作为成功率分母，中断实验可能显得更成功。现在按 `plan.selection` 的计划数统计，并显示已完成数、未完成数和 `pending` 状态；尚未产生第一条结果时也能生成阶段报告。
- 原来无条件输出 B0、M、W 三组，没有运行的组耗时为 `None`，格式化时会崩溃。现在只输出 `plan.arms` 指定的组，只有同时运行的基准/改动组才做配对，空耗时显示“未完成”。

补充检查 [test_arm_analysis.py](/data/zhujun/differt_projects/time-bias-correct/tests/test_arm_analysis.py) 覆盖只运行 M 组、没有结果文件、部分完成及空样本组；**2 项通过，退出码 0**。日志为 [task.log](/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_arm_analysis_tests_20260921_01/task.log)，运行记录目录为 `/data/zhujun/differt_projects/time-bias-correct/outputs/workspace_review_arm_analysis_tests_20260921_01`。核心 115 项记录和这两项补充记录分别保存。
