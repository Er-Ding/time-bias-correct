# 连续传播定位实施记录

日期：2026-09-15。仓库：`/data/zhujun/differt_projects/time-bias-correct`。

本记录状态更新至 2026-09-15 09:51 UTC：代码与全量实现检查已完成；第一份冻结观测及其报告已完成，第二份仍在后台计算。对照运行的最新状态以保存的日志、`comparison/summary.json` 和退出记录为准。

依据：[用户已确认的设计方案](/data/zhujun/differt_projects/time-bias-correct/docs/continuous_model_design_20260915.md)。旧方法和既有输出保留，新增配置明确选择连续流程。

## 1. 已实现的主流程

**CSI → 细化后的角度、时延观测 → 由地图建立连续传播函数 → 联合优化位置与公共偏差 → 逐路径检查。**

新流程不执行位置撒点、空间聚类、代表选择或旧 RANSAC，也不使用旧结果产生初值。

每个传播函数描述一种地图上的传播顺序。位置改变时，函数重新计算路径长度、基站到达角与反射点。绕射方向连续变化，不固定到几个采样方向；当前二维地图的绕射点仍为实际墙端点。

联合未知量为 `x, y, beta`，其中 `beta = c × 时钟偏差`，单位米。基本观测关系保持为：

`观测时延 = 当前路径长度 / c + 公共时钟偏差 + 观测误差`。

每条观测的角度和时延误差一起归一化、一起降权。一个观测只贡献一次，同一物理路线也不能重复分配给多个观测。无法匹配的观测支付固定代价，不能靠让路径失效免费降低目标。

## 2. 代码入口与职责

| 文件 | 职责 |
|---|---|
| [propagation_model.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/propagation_model.py) | 直射、反射、单次绕射和混合顺序的连续长度、角度、导数与路径合法性。传播顺序统一为 UE→BS。 |
| [propagation_hypotheses.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/propagation_hypotheses.py) | 根据地图、基站和原始观测建立传播函数；记录已检查、被排除和尚未检查的路线数量。 |
| [continuous_solver.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_solver.py) | 连续约束初值、允许区域内的多起点、观测与路线匹配、带边界的联合优化、物理约束检查、多解保留。 |
| [continuous_pipeline.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_pipeline.py) | CSI 和冻结 MUSIC 共用的连续入口；独立重建已选路径；保存观测、函数、搜索和结果。 |
| [continuous_config.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/continuous_config.py) | 公开参数默认值与校验；不接受真值或注入噪声参数。 |
| [raytrace2d.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/raytrace2d.py) | 将逐墙遮挡检查改为批量数组计算，缓存只读墙坐标；沿用原来的交点、端点和平行判据。 |
| [spectrum_sampling.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/spectrum_sampling.py) | 新增 `refine_music_peaks`，复用峰位细化，在随机采样之前返回；旧采样接口保留。 |
| [pipeline.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/pipeline.py) | 在生成旧候选点之前分流；连续成功结果使用第 8 版定位清单，无位置结果保留实际诊断与流程名称。 |
| [run_continuous_comparison.py](/data/zhujun/differt_projects/time-bias-correct/scripts/run_continuous_comparison.py) | 冻结旧 CSI、MUSIC、地图和观测筛选；只运行新模型；求解结束后单独评估旧位置与真值误差。 |

计划中的正向检查与约束诊断直接放在新连续模块内，避免让新接口依赖旧代表轨迹。旧 `forward_check.py`、`diffraction_diagnostics.py` 保留服务历史方法。

主入口为 `localization.solver_method: continuous`。专用配置：[continuous_model_v1.yaml](/data/zhujun/differt_projects/time-bias-correct/configs/continuous_model_v1.yaml)。原默认配置不变，以免历史配置快照被静默改写。

冻结观测接口为 `continuous_pipeline.localize_saved_music(...)`，其输入只包含公开定位配置、地图、保存的细化峰及来源文件。可选 `return_problem=True` 在新估计完成后返回同一模型供对照检查；旧位置没有进入该接口。

## 3. 接受与未通过的含义

- `success`：找到满足当前原始观测残差、合法路径和独立约束条件的解，且本次搜索未发现评分接近的不同解。
- `ambiguous`：找到评分接近而位置或偏差明显不同的有效解；保留备选，但不输出唯一位置。
- `unlocalizable`：能够证明当前函数和观测缺少足够独立约束，或完整检查后没有可用函数。
- `solver_budget_exhausted`：在给定路线、初值或局部迭代预算内没有找到可接受解；不把它解释为物理上不存在解。
- `geometry_failed`：最终独立路径重建或原始观测残差检查未通过；不输出成功位置。
- 前端的观测排除、峰数不足等状态继续保留，不悄悄从总体分母移除。

`success` 只对应本次搜索与已选路径的接受条件，不是全局唯一性的证明。保存的条件协方差仅描述固定解释附近的局部变化，不是经过标定的定位置信区间。所有结果都明确保留 `scientific_validation_status: not_validated`。

## 4. 默认参数与边界

第一版默认误差尺度为角度 1°、长度 0.75 m；这些是工程参数，不从注入信噪比或干净 CSI 得出。最多保留 4096 个传播函数，枚举检查预算 50000 条顺序，最多 96 个初值、每个初值 80 次局部迭代、8 轮匹配更新。预算和未覆盖范围均写入输出。

位置边界来自地图，公共偏差范围来自公开配置。局部更新使用有界线性子问题与合法路径回退，反射点越界、遮挡、阴影条件失效时不会继续把该路径当作有效约束。

目前支持二维、至多一次绕射及设定阶数的反射。未实现三维沿边缘移动的绕射点。大地图的传播顺序可能超过预算，因此不能宣称所有分支均已搜索。

大地图的安全预筛先计算有限墙段展开后的连续角度区间，再判断其与原始观测容差区间是否相交。两次反射要求最后墙段与前面墙段的镜像共享可行角度；不同观测轮流获得搜索机会。这是连续几何必要条件，不是对墙面或 UE 撒点。预筛计算次数和耗时单独记录，不藏在枚举预算之外。

初值组合分为两条可变角度路径、一条可变角度路径配一条绕射路径、三条绕射路径，按 2:1:1 轮流分配预算；各类内部再按观测组合和传播阶数轮换。仅有两条纯绕射距离的组合不能决定三个未知量，不消耗二元初值预算。给定有限预算仍可能遗漏解，未尝试组合会明确记录。

有限个起点只是让数值优化从几个地方开始。每一轮的位置和偏差都连续更新，最终解不必落在这些起点、固定方向或代表网格上；传播路线之间的选择与路线内部的连续优化是两件事。

第一版还保留两个实现边界：最低观测数与独立约束检查参与最终接受，匹配子问题本身没有强制最低匹配数量，因此可能拒绝某些需更高匹配代价的解释；普通 CSI 入口在正常异常时保存阶段文件，但强制终止进程可能丢失尚在内存中的阶段记录。冻结观测入口已在联合优化前落盘观测和函数，在正向验证前落盘已完成的求解诊断。

## 5. 输出与来源

每个冻结观测的新目录包含：

```text
localization/
  continuous_observations.json
  propagation_hypotheses.json
  continuous_search.json
  forward_check.json
  localization_result.json
  localization_config.json
  frozen_input_manifest.json
baseline_common_state_validation.json
evaluation/metrics.json
attempt.json
```

`continuous_search.json` 保存实际初值、路线匹配、原始观测残差、失败原因、独立约束及备选解。冻结来源清单记录输入和输出哈希。普通 CSI 主入口另保存 MUSIC 谱与细化峰，并使用原有的生成/评估绑定机制和第 8 版清单。

对照总表始终保留原计划 30 个 UE × 5 次观测。限制运行数量后，未运行项保持 `pending`；43 次历史观测排除和 30 次峰数不足也保留。旧位置只在新求解结束后，使用同一连续模型重新匹配、检查；不修正旧位置。这个检查比较同一位置的解释能力，不等同于对旧选中路线逐条复验。

真值仅由单独的误差评价函数在新定位完成后读取。冻结实验可通过 Python 入口的 `--skip-evaluation` 完全关闭真值评价。

## 6. 检查与对照记录

实现检查已通过，冻结观测对照的结果继续记录在本节。实施检查与 30×5 的科学效果验证分开记录；不能从合成检查通过或两份样本推断整批定位误差改善。

**最终全量 CPU 检查：999 项中 976 通过，23 跳过，0 失败、0 错误；退出码 0。** 包含新增的 97 项遮挡判据一致性检查，检查前后的源码与配置哈希一致。22 项跳过与 CUDA 有关，1 项依赖本机缺失的历史 Munich 产物；没有声称完成 GPU 验证。

- [最终验证汇总](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04/validation/validation_summary.json)
- [最终完整测试日志](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04/run_records/20260915T093600_299044/task.log)
- [最终退出记录](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04/run_records/20260915T093600_299044/completion.json)

搜索修复后、批量遮挡改造前的上一轮检查为 879 通过、23 跳过、0 失败（共 902 项），同样保留：

- [验证汇总](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_03/validation/validation_summary.json)
- [完整测试日志](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_03/run_records/20260915T092253_279271/task.log)
- [退出记录](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_03/run_records/20260915T092253_279271/completion.json)

覆盖内容包括：连续几何及导数、偏离旧方向网格的无噪声恢复、真实地图绕射多解、重复观测与重复距离约束、遮挡和反射点越界、角度跨越 ±180°、位置和偏差边界的最优解、连续角窗不丢合法路线、初值组合预算、CSI 主流程绕过所有旧点/聚类/代表/RANSAC 接口、删除真值后的独立定位、冻结文件篡改拒绝、失败记录保留、无真值的多解报告。

### 6.1 冻结观测的小规模检查

最终代码的对照根目录：[continuous_model_experiment_20260915_03](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03)。本次只选两份保存观测，不是完整 30×5 实验。

| 样本 | 旧位置误差 | 新位置误差 | 旧偏差绝对误差 | 新偏差绝对误差 | 新状态 |
|---|---:|---:|---:|---:|---|
| PILOT_0001 / repeat_001 | 0.049684 m | 0.011272 m | 0.135896 ns | 0.022500 ns | success |
| PILOT_0019 / repeat_001 | 5.496680 m | 待完成 | 待汇总 | 待完成 | 正在计算 |

第一份样本共三条入选路径，全部为反射路径，独立重建检查全部通过；真实约束矩阵秩为 3，条件数约 8.10。本次 96 个起点得到一个不同的可接受解，最终残差代价为 0.002292；旧位置放入同一个连续模型重新匹配后也通过，代价为 0.003654。新求解耗时约 409.38 秒（CPU 单线程，包含本次函数构造与搜索），不能宣称已经适合实时定位。

第二份样本有九条原始观测。截至上述时间已完成 16/96 个起点，尚未找到满足接受条件的结果，仍在原后台任务中继续；这个中间状态既不是最终失败，也不是物理无解的证明。尚未执行完整 30×5 效果验证。

这一份观测支持“新模型能够独立工作，且在该样本上降低原始观测残差和位置误差”。因为入选路径没有绕射，而且路线搜索方式也改变了，不能把改善单独归因于绕射方向连续化，更不能推断整批效果。

- [第一份定位结果](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03/comparison/trials/PILOT_0001/repeat_001/localization/localization_result.json)
- [第一份独立评价](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03/comparison/trials/PILOT_0001/repeat_001/evaluation/metrics.json)
- [第一份可视化报告](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_report_20260915_02/report/README.md)：真实冻结清单加载、来源校验、评价重算和 PNG/SVG 导出已实跑通过，退出码 0。

### 6.2 保留的调试记录

首轮针对性检查目录：[continuous_model_check_20260915_01](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_01)。102 项中 101 项通过，1 项涉及全路径无效时的状态分类，已修复并保留失败记录。

第二轮全量检查目录：[continuous_model_check_20260915_02](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_02)。共 884 项，860 通过、23 跳过、1 项失败；唯一失败为新增集成测试错误地期待接口抛出异常，而实际约定是返回无位置结果，已改正该测试。跳过包括 22 项 CUDA 检查及 1 项依赖本机缺失历史产物的检查。

首个大地图诊断目录：[continuous_model_experiment_20260915_01](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_01)。地图有 1440 面有限墙段；首个样本的函数库达到 4096 上限。日志中几个初值检查点均未匹配观测。只读审计确认：枚举顺序会遗漏某些合法双反射顺序，初值的首组观测对还会占满预算。该运行在修复前主动停止，退出码 143；它是未完成的诊断，不能统计成成功、物理不可定位或完整的两样本对照。

修复前源码已归档：[source_before_search_fix.tar.gz](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_01/source_before_search_fix.tar.gz)。修复依据是公开地图的连续角域必要条件及观测组合的公平预算分配，旧路线只用于解释遗漏发生在哪里，不参与正式新搜索。

第二个真实诊断运行 [continuous_model_experiment_20260915_02](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_02) 已解决前述路线覆盖问题，但 626.66 秒时仍未完成首个样本。逐墙 Python 遮挡循环成为明显的计算负担，改造前主动停止，退出码 143。源码保存在 [source_before_visibility_batching.tar.gz](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_02/source_before_visibility_batching.tar.gz)。该未完成运行同样不计入定位效果结论。

## 7. 后台执行方式

解释器固定为 `/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python`。启动脚本在顶部参数区集中设置工作目录、配置、输入、输出、线程和范围；使用 `nohup setsid`，每次新建独立运行目录，记录实际任务 PID、日志、参数、时间和退出码。

- [run_continuous_model_check.sh](/data/zhujun/differt_projects/time-bias-correct/run_continuous_model_check.sh)：实现检查。`CONTINUOUS_CHECK_TEST_SCOPE=all` 运行仓库全量回归；默认仅运行连续模型和配置检查。
- [run_continuous_model_experiment.sh](/data/zhujun/differt_projects/time-bias-correct/run_continuous_model_experiment.sh)：冻结 MUSIC 的旧新方法对照。默认先运行一份；`CONTINUOUS_EXPERIMENT_LIMIT=0` 运行所有可进入求解器的观测。
- [run_continuous_model_report.sh](/data/zhujun/differt_projects/time-bias-correct/run_continuous_model_report.sh)：读取指定的 `comparison` 或单次定位目录，导出逐次结果和连续求解步骤。必须设置 `CONTINUOUS_REPORT_INPUT_ROOT`；默认不读取评价真值，设置 `CONTINUOUS_REPORT_SKIP_EVALUATION=0` 才检查已有独立评价。

以下为本次运行的准确启动与管理命令。启动命令记录的是已使用目录，重新运行时请换新的输出目录，脚本会拒绝覆盖。查看日志时按 `Ctrl+C` 只退出查看，不会停止后台任务。

### 7.1 本次全量实现检查

```bash
CONTINUOUS_CHECK_TEST_SCOPE=all \
CONTINUOUS_CHECK_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04 \
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_check.sh

tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04/run_records/20260915T093600_299044/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_check.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_check.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04
```

实际任务 PID 为 299056；运行记录目录为 `/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_check_20260915_04/run_records/20260915T093600_299044`。此任务已完成，退出码 0。

### 7.2 本次两份冻结观测对照

```bash
CONTINUOUS_EXPERIMENT_UE_IDS=PILOT_0001,PILOT_0019 \
CONTINUOUS_EXPERIMENT_REPEAT_INDICES=1 \
CONTINUOUS_EXPERIMENT_LIMIT=2 \
CONTINUOUS_EXPERIMENT_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03 \
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_experiment.sh

tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03/run_records/20260915T093606_299258/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03
```

实际任务 PID 为 299268；运行记录目录为 `/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03/run_records/20260915T093606_299258`。原始输入来自设计文档记录的 v6 `coverage` 组，只读复用。

两份样本的历史位置误差分别约为 0.049684 m 与 5.496680 m。这个选择用于检查正常样本和既有大误差样本上的行为，不能代表整批分布。全 150 次请求仍保存于 [trials.json](/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03/comparison/trials.json)。

### 7.3 已完成的第一份结果报告

```bash
CONTINUOUS_REPORT_INPUT_KIND=run \
CONTINUOUS_REPORT_INPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_experiment_20260915_03/comparison/trials/PILOT_0001/repeat_001 \
CONTINUOUS_REPORT_SKIP_EVALUATION=0 \
CONTINUOUS_REPORT_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_report_20260915_02 \
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_report.sh

tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_report_20260915_02/run_records/20260915T094417_311336/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_report.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_report_20260915_02
bash /data/zhujun/differt_projects/time-bias-correct/run_continuous_model_report.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_report_20260915_02
```

实际任务 PID 为 311349；运行记录目录为 `/data/zhujun/differt_projects/time-bias-correct/outputs/continuous_model_report_20260915_02/run_records/20260915T094417_311336`。已完成，退出码 0；PNG 图已打开核对，SVG 与来源清单同时保留。
