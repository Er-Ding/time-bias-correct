# MUSIC 特征值阈值划分与谱峰验收（2026-09-14）

新配置采用：观测 CSI → 协方差与特征值 → 相对阈值划分 → MUSIC 谱和候选峰 → CSI 残差验收 → MC → 反向 RT → 聚类和定位。

MUSIC 的维数由观测特征值确定，不再由前置路径拟合的数量确定。原 v3 配置保留旧流程，便于复现；新入口配置为 `/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_generation_v4.yaml`。

## 参数与含义

```yaml
music:
  subspace_selection:
    mode: eigenvalue_threshold
    noise_reference: median
    threshold_ratio: 6.0
```

在计算分界前先去掉数值对角加载引入的特征值平移。默认基准为观测协方差全部特征值的中位数；阈值为基准乘 `threshold_ratio`，大于阈值的特征向量归入信号部分，其余归入噪声部分。`6.0` 是阈值倍数，不是路径数量。实际维数允许为零，也允许大于六；不强制补齐或截成六维。

`noise_reference` 还支持 `minimum`（最小特征值倍数规则）与 `lower_half_median`（较小一半的中位数）。MATLAB `pmusic` 支持最小特征值倍数方式，见 https://www.mathworks.com/help/signal/ref/pmusic.html 。当前滑窗协方差在有限样本下噪声特征值会分散，不能直接将“最小特征值乘一个很接近 1 的数”当作可靠通用配置。

首次回归中，较小一半的中位数乘 4 在三路径合成观测上选出了 99 维。默认值因此调整为全部特征值中位数乘 6。此选择只通过当前实现回归，尚未在固定的 30 UE × 5 次 CSI 上进行阈值校准；不能宣称最优或具有指定的整体误检概率。中位数规则仍依赖大部分特征方向主要由噪声构成的假设。

旧的 `signal_subspace_rank` 在阈值模式下不参与划分。`mode: fixed` 是旧配置复现入口。

## 谱峰数量与验收

阈值模式且启用残差验收时，MUSIC 提供整张搜索网格内经过间隔抑制的局部极大值，不按信号维数或旧 `num_paths` 数量截断。验收器根据剩余 CSI 对这些峰的支持程度选择提案，进行连续角度/时延与复系数联合拟合，保留原有合并/删除/重新拟合的重复解释检查。

门槛仍使用整张二维网格的模拟噪声最大值校准，而不是只对某一个 MUSIC 峰计算单点门槛。MUSIC 与提案使用同一份观测；噪声模型和已拟合成分条件下的校准限制继续记录，不将其称为无条件精确概率。

每轮仍检查完整残差搜索。如果残差含显著额外成分，而 MUSIC 提案已经用尽、不能通过检验或不能完成拟合，返回 `detection_incomplete`，不把“候选谱峰不足”解释成“只剩噪声”。只有完成验收的路径才能进入 MC；验收后不再反过来改变 MUSIC 子空间。

`path_detection.max_paths: 6` 仍是独立的残差验收预算，未在本次改动中取消。超过预算仍报告检测未完成。这项修改不保证原来 35 次预算退出消失；候选配对十万上限、DBSCAN 和定位目标函数也未调整。

## 可核查输出

`music_peaks.json` 的 `subspace_selection`、结果的 `diagnostics.music_subspace_selection` 保存升序特征值、基准、阈值和信号/噪声维数。阈值模式另存 `music_proposals`；`path_detection` 保留每轮整体残差与提案检验、拟合、去重和停止原因。失败或无坐标时也保留已完成阶段的记录。

零维信号会报告没有超过阈值的信号子空间；噪声基准接近数值分辨率或无法解析时，报告 `detection_incomplete`，不强行指定维数。两种判断均属于当前观测和规则下的结果，不构成全局不可定位证明。

## CPU 后台验收

脚本：`/data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh`。

固定参数区包含工作目录、解释器、输入配置、独立输出目录、CPU 线程数和测试范围。默认运行完整测试且禁止使用 GPU；`MUSIC_SUBSPACE_TEST_SCOPE=focused` 可只运行本次相关测试。使用现有 `detached_task.py` 记录实际任务 PID、启动命令、开始/结束时间和退出码，并按进程身份停止整组任务。

重新验收示例（输出目录必须尚不存在）：

```bash
MUSIC_SUBSPACE_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_recheck_20260914 \
bash /data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh

# 实时查看日志；脚本读取上述目录的 latest_run.txt，使用 tail -n 100 -F。
bash /data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh log /data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_recheck_20260914

# 查看完成记录与退出码。
bash /data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_recheck_20260914

# 核对进程身份后停止整组任务。
bash /data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_recheck_20260914
```

启动器会打印真实日志的绝对路径，形式为 `<输出目录>/run_records/<时间和启动器编号>/task.log`。日志查看时 Ctrl+C 只退出查看，不会停止后台任务。目录内 `validation/pytest.xml` 与 `validation/validation_summary.json` 是测试结果；运行记录目录内 `completion.json` 才是任务结束依据。

本次未启动 GPU 预跑，也未修改任何已有实验结果。修改前源码及摘要位于 `/data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_threshold_change_20260914_01/source_before`。

## 本次完成记录

完整 CPU 回归：790 项通过、23 项跳过、零失败；2026-09-14 02:37:51 UTC 开始，02:38:22 UTC 结束，用时 30.97 秒，退出码 0。测试期间源码和输入配置摘要保持一致。覆盖单路径停止、强路径与弱路径同时保留、幅度缩放与对角加载、不同观测不同维数、零维与超过六维、独立噪声投影公式对照，以及 MUSIC 先于残差验收、验收完成后才能进入 MC 的调用顺序。

- 测试报告：`/data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02/validation/validation_summary.json`
- 逐项测试：`/data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02/validation/pytest.xml`
- 运行记录：`/data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02/run_records/20260914T023751_1818817`
- 实际日志：`/data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02/run_records/20260914T023751_1818817/task.log`

```bash
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02/run_records/20260914T023751_1818817/task.log
bash /data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02
bash /data/zhujun/differt_projects/time-bias-correct/run_music_subspace_check.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/music_subspace_check_20260914_all_02
```

该任务已经结束；停止命令会返回结束记录，不会停止其他任务。第一次沙箱内启动没有结束记录；后续阈值和测试记录路径修正前的失败验收目录均保留，不能把它们误认为完成的定位实验。
