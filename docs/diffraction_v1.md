# 一次绕射与簇内多代表：第一版

已接通带噪 CSI → MUSIC 细谱采样 → 反向候选点 → DBSCAN → 多代表 → 位置与公共时间偏差联合求解 → 正向几何检查。启用绕射时工作流为 `music_diffraction_cover_v4`，定位清单版本为 7；关闭时继续使用 v3。

## 这版实现的范围

固定高度的二维墙线场景，边缘视为竖直方向，最多一次绕射、最多两次镜面反射；允许反射发生在绕射前后。墙线端点生成公开边缘编号，排除共线接缝和 T 形等多墙交汇。只保留墙角阴影侧、各段无阻挡的候选，不将整个遮挡区域填满。

从 BS 到边缘的可见反射前缀由地图计算；只有其到达角与观测采样的差值在公开容差内，才从边缘继续采样方向。沿各方向走完 `c × (观测时延 − 参考偏差)` 的剩余路程，必要时继续反射。方向在不同样本间错开，以覆盖扇面。该角度容差是候选筛选参数，需要随测角误差调整，不代表 MUSIC 已经识别出传播机制。

每个候选保存从 **BS 到 UE** 的完整类型、墙/边缘编号顺序及交互点；正向检查时反转顺序，从估计 UE 重新计算路径。绕射分支保留原始采样编号 `parent_sample_id`，并使用独立分支编号。反射和绕射只是同一观测的不同地图解释，定位端不读生成路径标签、真实 UE 或真实偏差。

## 聚类和代表点

DBSCAN 的邻域、核心点、边界点、离群点规则不变。分组键扩展为“同一来源峰 + 完整传播顺序”，因此不同墙角、反射后绕射与绕射后反射不会混为一簇。

反射簇仍选一个真实成员。绕射簇先以原来的代表为起点，再选离已有代表最远的成员，直到所有成员在参考偏差下都处于覆盖距离内。随后检查各成员的整个合法偏差区间，必要时补充代表。

第二步使用两成员位置差关于偏差的一次关系，解析求解距离小于阈值的区间，并合并多个代表的覆盖区间。它没有把几个离散偏差检查点当作整个区间的保证，也不在 DBSCAN 之前建立求解器轨迹。只在各代表自己的有效区间内计入覆盖；绕射点、墙面和边界上的退化末段被排除。

一个簇可以有多个代表编号，但共用 `point_cluster_id`。成员表和簇数量只统计一次。求解器按来源峰互斥选择，每个代表权重仍为 1；增加代表数不会增加独立观测数。

覆盖距离只约束**已生成簇成员的压缩误差**。它不保证未采到的方向、被 DBSCAN 排除的离群点、在参考偏差处无效的路径也被覆盖，更不直接保证最终定位精度。检查整个偏差区间可能使代表数明显增加，极端情况下会保留全部成员。

## 参数入口

完整示例配置为 `configs/diffraction_demo.yaml`。以下字段均属于公开定位配置；生成与定位两端的绕射开关必须一致，不能直接给原来不含绕射的生成清单改一个定位开关。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `scene.max_diffractions` | 最大绕射次数，只支持 0 或 1 | 0，保持原流程 |
| `scene.max_reflections` | 整条路径上的最大反射次数 | 2 |
| `localization.diffraction_directions_per_sample` | 每个匹配的边缘前缀、每个观测采样的扇面方向数 | 4 |
| `localization.diffraction_angle_tolerance_deg` | BS 到边缘前缀与采样到达角的容差，度 | 3 |
| `localization.diffraction_coverage_distance_m` | 绕射簇成员与代表之间的最大覆盖距离，米 | 1 |
| `localization.candidate_cluster_radius_m` | DBSCAN 邻域距离，米 | 1.5 |
| `localization.candidate_cluster_min_samples` | DBSCAN 邻域点数，包含自身 | 5；示例为 3 |

前缀枚举随墙数及反射次数增长，DBSCAN 仍使用原来的组内距离矩阵。这一版尚未优化大场景的开销；求解器继续在候选对超过 `max_seed_pairs` 时明确报错，不静默丢弃某些候选来满足预算。

## 生成与验收边界

二维示例按真实路径几何计算角度和长度，但使用配置中的幅度合成 CSI，没有实现 UTD 绕射复系数。因此它验证算法和文件接口闭环，不验证真实绕射幅度、材料误差或实际场景精度。

Sionna 入口已接入绕射开关、阴影侧绕射和自由边缘开关。读取路径时分别统计反射与绕射，排除两次绕射、散射、折射和不符合固定高度的路径；绕射路径还必须由公开二维墙线重建成功。Sionna 的完整真实绕射生成流程本次尚未运行，当前验收包含该入口的交互类型筛选测试。

已选路径正向检查重建反射点和绕射点，并检查遮挡、反射定律及角度/时延残差。它没有实现全路径集合匹配或 CSI 重建评分。

结果新增 `diagnostics.diffraction_physical_constraints`：绕射的采样出射方向不作为额外测量。在二维未知位置加公共偏差的问题中，两条纯绕射距离观测不足以提供三个独立约束；诊断会给出约束秩不足。现有协方差仍是选定离散代表后的条件近似，未标定；即使局部约束秩为 3，也不保证全局只有一个位置解释。

## 本次验证记录

运行目录：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_check_20260910T093229_2454082`。

- 完整测试：624 通过，1 跳过。跳过项是本机缺少旧的固定 Munich 谱面采样文件。
- 二维遮挡墙示例：270 个初始点，6 个簇，29 个代表；其中 2 个绕射簇保留 25 个代表。4 个离群点独立记录。
- 最终选中 3 条观测，其中 1 条为绕射；所有已选路径通过正向几何检查。
- 该单例位置误差 `0.0320093428 m`，公共偏差误差 `0.0916408264 ns`。这些数值不是批量精度结论。
- 原有 UE001 / repeat_000 CSI 对照：关闭绕射，复用原公开配置及随机种子，位置和偏差变化均为 0。
- 后台任务结束记录的退出码为 0。此前测试失败和隔离环境中未完成的启动记录均保留在各自目录。

汇总：`artifacts/check_summary.json`；测试结果：`artifacts/tests.xml`；逐步图表：`artifacts/demo/step_report`；成员与多代表图位于该报告的 `samples/UE001/repeat_000/05_point_clustering/`。

## 后台运行与管理

启动器 `run_diffraction_check.sh` 的参数区集中设置项目目录、Python、配置、原 CSI 对照来源、线程数和输出目录。`DIFFRACTION_CHECK_MODE` 可设为 `tests`、`demo`、`replay` 或 `all`。每次默认创建新的目录，拒绝覆盖。任务通过 `nohup` 和 `setsid` 启动，输入接到 `/dev/null`，Python 不缓冲日志。

启动新一轮：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_diffraction_check.sh
```

启动器会直接打印该次运行可复制的日志、状态和停止命令。下面是本次已结束运行的对应命令，重新连接服务器后也可使用：

```bash
# 实时查看日志；Ctrl+C 只退出查看，不停止任务。
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_check_20260910T093229_2454082/task.log

# 查看状态与实际退出码。
/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py status /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_check_20260910T093229_2454082

# 停止该次任务；先核对 PID、进程启动标识和进程组，再向整组发送停止信号。
# 已结束的运行只显示结束记录，不发送信号。
/data/zhujun/differt_projects/time-bias-correct/.sionna-venv/bin/python /data/zhujun/differt_projects/time-bias-correct/scripts/detached_task.py stop /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_check_20260910T093229_2454082
```

记录包括 `task.log`、实际任务的 `task.pid`、带命令及启动时间的 `task.json`、带结束时间和退出码的 `completion.json`。缺少结束记录时不会将“进程已不在”解释为成功。此方式防止终端关闭导致中断，不承诺服务器重启后自动恢复。

## 对应代码

| 部分 | 文件 |
| --- | --- |
| 边缘、混合路径几何与重建 | `src/time_bias_localization/diffraction.py` |
| 观测采样生成绕射候选 | `src/time_bias_localization/diffraction_candidates.py` |
| DBSCAN、传播顺序与代表轨迹 | `src/time_bias_localization/initial_candidates.py` |
| 最远成员选取与连续偏差覆盖 | `src/time_bias_localization/representative_cover.py` |
| 独立约束诊断 | `src/time_bias_localization/diffraction_diagnostics.py` |
| 主流程与正向检查 | `src/time_bias_localization/pipeline.py`、`forward_check.py` |
| Sionna 筛选与输入一致性 | `src/time_bias_localization/sionna_generation.py`、`contracts.py` |
| 多代表图表与逐点核对 | `src/time_bias_localization/step_visualization.py` |
| 完整验证入口 | `scripts/check_diffraction_workflow.py`、`run_diffraction_check.sh` |
