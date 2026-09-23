# 同一墙面重复记录的原因与修复

本次修复的是原始三角网格转换为二维墙线时的重复覆盖，不修改定位收敛条件、多解判断或 1800 秒时限。

## 原因证据

输入网格是
`/data/zhujun/differt_projects/time-bias-correct/outputs/monte_carlo_munich_1000_20260915_01/channel_setups/20260915T142240_340230_742895/public_original_mesh.npz`。
SHA-256：`0959405af5d5b78cb674a80c667da0a07934c61e3fc5abc725e79843d902fce8`。

三角面 24960、24961 与 25357、25358 的底边占据同一位置，但法向相反；两组立面的高度分别是 19.1251888275 m 和 18.9558696747 m。
三角形划分方向、高度不同，在 z=1.5 m 截取时得到两套不同切分点。
这使同一面约 18.467 m 长的墙变成四个编号，且两两组合重复覆盖整面墙。
旧预处理只比较两个端点是否相同，因此没有识别出这类部分重叠。
“双面渲染需要”或“相邻建筑共墙”可能解释资产的建模方式，但原资产没有提供制作记录，本次不作该层面的归因。

## 改动

- 在既定范围内选择原始三角切片，再对同一对象内共线、相接或重叠的区间合并。
- 共线距离容差最多为 1e-8 m；接缝仅容许浮点舍入量，不用毫米级端点量化填补真实间隙。
- 先合并再过滤短墙，避免把三角形切出来的小片段提前删掉，造成假洞。
- 不同方向、不同对象的墙保持分开；T 形交汇和真实墙角由原几何判断继续检查。
- 已有场景的读取不隐式合并，旧墙编号、原始数据和历史结果保持可复现。

实现：[scene.py](/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/scene.py:413)。
回归检查：[test_wall_segment_union.py](/data/zhujun/differt_projects/time-bias-correct/tests/test_wall_segment_union.py)。

## 验证结果

最终结果目录：
`/data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_20260918_02/validation`。

304 项检查通过，0 项失败、0 项跳过；后台任务退出码为 0。
检查包括真实间隙、近距离平行墙、不同对象边界、T 形接点、原区域范围、三角形接缝处的反射，以及与独立完整墙面参考场景的反射/绕射路径集合一致性。

原地图 1440 个墙段，新地图 763 个墙段。逐段区间覆盖检查确认：旧墙未丢失，新图墙线均能在原始网格中找到；补回旧图遗漏的原始网格墙线约 10.089 m。
原有 562 个可供绕射使用的端点变为 493 个，此数量变化包含重复覆盖和假接缝被消除后的结果，不能仅凭数量评价绕射精度。

| 样本 | 原候选数 | 新候选数 | 新候选枚举是否完成 |
| --- | ---: | ---: | --- |
| 000103 | 4096 | 2651 | 是 |
| 000414 | 4096 | 2835 | 是 |
| 000685 | 4096 | 2591 | 是 |
| 000748 | 4096 | 2762 | 是 |
| 000820 | 4096 | 2749 | 是 |

五个样本沿用已有观测和原筛选参数，没有向候选生成传入真实位置。
000103 的两次反射加一次绕射候选由 0 个增加为 9 个。
候选枚举完成只指当前传播阶数及筛选条件内的枚举完成；候选仍需在具体位置检查，不是实际有效多径计数，也不是定位成功率结果。
本次没有重新进行整批定位，旧结果不会自动变成新结果；后续定位实验应重新生成并使用新地图。

详细报告：[report.md](/data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_20260918_02/validation/report.md)。
新地图：[scene_2d.json](/data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_20260918_02/validation/corrected_scene/scene_2d.json)。
新墙编号对应的原始三角面：[wall_sources.json](/data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_20260918_02/validation/wall_sources.json)。

第一轮目录 `wall_geometry_check_20260918_01` 是中间开发记录，已标记替代；正式结果使用第二轮。

## 执行与管理

参数集中在 [run_wall_geometry_check.sh](/data/zhujun/differt_projects/time-bias-correct/run_wall_geometry_check.sh) 顶部。
工作目录为 `/data/zhujun/differt_projects/time-bias-correct`，解释器为该目录的 `.sionna-venv/bin/python`。
每次运行均由 nohup、setsid 脱离终端，使用独立输出目录；重跑时将下面四条命令中的 `wall_geometry_check_manual_01` 同步替换为新的目录名。

启动：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_wall_geometry_check.sh start /data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_manual_01
```

实时查看日志；Ctrl+C 只退出查看，不停止任务：

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_manual_01/task.log
```

查看运行状态：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_wall_geometry_check.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_manual_01
```

停止整项任务；管理器会先核对进程身份：

```bash
bash /data/zhujun/differt_projects/time-bias-correct/run_wall_geometry_check.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_manual_01
```

实际运行记录保存在输出目录的 `run_records/时间_编号/` 中，路径同时记在 `latest_run.txt`；包含实际任务 PID、命令、日志、开始/结束时间和退出码。
本次已完成任务的实际记录目录为
`/data/zhujun/differt_projects/time-bias-correct/outputs/wall_geometry_check_20260918_02/run_records/20260918T024550_2018085`，
实际日志为该目录中的 `task.log`。
