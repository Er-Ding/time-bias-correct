# 标准 MUSIC 路径特征提取（v5，2026-09-14）

本版把路径提取改为：**原始观测 CSI → 一次子空间划分和全局 MUSIC 谱 → 直接读峰 → 同一 MUSIC 谱的局部细化与去重 → MC → 反向 RT → 聚类 → 联合定位。**

不再逐条拟合后扣除 CSI，不再用剩余 CSI 反复验收下一条路径，也不进行 CSI 的角度、时延和复系数联合拟合。局部细化只是在已经找到的峰附近，用同一份原始 CSI 形成的子空间计算更细的网格；它不从残差中重新检测路径。

## 本版参数和路径数量

路径提取配置：`/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_generation_v5.yaml`。

```yaml
music:
  subspace_selection:
    mode: eigenvalue_threshold
    noise_reference: median
    threshold_ratio: 6.0
  path_detection:
    enabled: false
```

保留本轮已经使用的特征值阈值设置。设自动判断出的信号维数为 K，全局谱最多提取 K 个满足条件的峰，再进行局部细化和去重。峰数可以少于 K，不会为了凑数而补峰。配置中的 `num_paths: 3` 和 `signal_subspace_rank: 6` 仅供旧固定模式使用，在本配置的自动模式下不决定在线峰数。

旧的“最多验收六条路径”限制随残差验收关闭。这里没有再增加六条路径上限。没有有效峰，或只得到一条路径时，会保存停止原因并正常报告没有坐标，不会把一条路径拆成多条独立观测继续定位。

场景、无线参数、允许的时钟偏差范围、MC、反向 RT、DBSCAN、代表点方案和定位求解设置沿用 v4。求解阶段的 **10 万对候选上限仍然存在**；超过该上限时仍会停止求解。移除残差验收并不解决这一限制。

这个改动恢复了从 MUSIC 谱直接读取路径特征的流程，但尚未证明当前特征值阈值能正确识别所有路径，也未证明已经消除伪峰或提高定位精度。后续仍需用固定 CSI 统计坐标输出比例、全部请求中的误差达标比例、有坐标请求的欧式误差，以及各阶段用时。

## CPU 回归检查

执行脚本：`/data/zhujun/differt_projects/time-bias-correct/run_standard_music_check.sh`。参数集中在脚本顶部，默认运行完整 CPU 测试，不使用 GPU；也可以设置 `STANDARD_MUSIC_TEST_SCOPE=focused` 只运行本版及直接相关测试。

以下目录必须尚不存在；再次执行请更换末尾编号。

```bash
STANDARD_MUSIC_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01 \
bash /data/zhujun/differt_projects/time-bias-correct/run_standard_music_check.sh

# 实时查看日志，内部使用 tail -n 100 -F。
bash /data/zhujun/differt_projects/time-bias-correct/run_standard_music_check.sh log /data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01

# 查看运行状态、结束记录及退出码。
bash /data/zhujun/differt_projects/time-bias-correct/run_standard_music_check.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01

# 核对进程身份后停止整项任务及子进程。
bash /data/zhujun/differt_projects/time-bias-correct/run_standard_music_check.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01
```

运行记录和实际日志位于 `/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01/run_records/<启动时间_编号>/`，日志文件名为 `task.log`。启动时打印完整路径，输出目录内 `latest_run.txt` 也保存实际运行记录目录。完成后查看 `validation/validation_summary.json` 和 `validation/pytest.xml`；摘要明确保存 `path_detection_enabled: false` 和 `standard_music_direct_peaks: true`，同时记录源码、配置是否在测试过程中保持不变。测试通过只代表实现回归通过，不代表定位精度已经改善。

## 复用固定 CSI 重跑定位

执行脚本：`/data/zhujun/differt_projects/time-bias-correct/run_music_rerun.sh`。

默认实验配置为 `/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_accuracy_v5.yaml`，由它引用 v5 路径提取参数。继续只读复用 `/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v3` 中同一批 30 个 UE、每点 5 份 CSI 和随机种子，两种代表方案共运行 300 次定位请求。源数据目录必须保留。

GPU 编号必须由用户填写，脚本不会默认使用八张卡。下面的 `2,5` 仅是填写格式示例，启动前改为本次允许使用的物理 GPU 编号。

```bash
cd /data/zhujun/differt_projects/time-bias-correct
MUSIC_RERUN_GPU_IDS=2,5 \
MUSIC_RERUN_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_music_v5_20260914_01 \
bash /data/zhujun/differt_projects/time-bias-correct/run_music_rerun.sh

# 实时查看日志，先打印实际日志的完整绝对路径。
bash /data/zhujun/differt_projects/time-bias-correct/run_music_rerun.sh log /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_music_v5_20260914_01

# 查看运行状态、结束记录及退出码。
bash /data/zhujun/differt_projects/time-bias-correct/run_music_rerun.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_music_v5_20260914_01

# 核对进程身份后停止整项任务及子进程。
bash /data/zhujun/differt_projects/time-bias-correct/run_music_rerun.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_music_v5_20260914_01
```

实际日志和运行记录位于 `/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_music_v5_20260914_01/run_records/<启动时间_编号>/`，日志文件为 `task.log`。启动器打印完整路径，并在输出目录的 `latest_run.txt` 中保存实际记录目录。记录包含实际任务 PID、命令参数、开始时间、结束时间和退出码；以 `completion.json` 的退出码判定成功或失败，不能仅凭进程消失判断成功。最终报告位于 `/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_music_v5_20260914_01/pilot/report/`。

以上两个任务都通过已有 `nohup`、`setsid` 和运行监督脚本在后台启动，标准输入关闭，Python 使用及时写日志模式。关闭终端或 SSH 断线不会停止任务；查看日志时按 `Ctrl+C` 只退出查看。停止命令核对进程身份并处理整项任务的子进程。

每次修改源码或参数后都必须使用新的输出目录，避免结果、源码记录和配置记录混用。启动脚本会拒绝复用已经存在的输出目录。旧 v3/v4 配置、测试入口和历史结果保留；需要复核旧 v4 设置时，显式设置 `MUSIC_RERUN_CONFIG_PATH=/data/zhujun/differt_projects/time-bias-correct/configs/diffraction_boundary_accuracy_v4.yaml`，并提供另一个新的输出目录。这个文档不表示 GPU 实验已启动，也不预先承诺改动后的精度结果。

## 已完成的实现验证

2026-09-14 04:51:05 UTC 开始，04:51:36 UTC 结束，完整 CPU 回归用时 31.42 秒，退出码 0：**800 项通过、23 项跳过、零失败**。测试期间包内源码和 v5 配置摘要未变。新测试覆盖真实合成 CSI 的单路径、纯噪声、强弱双路径、七条路径、局部细谱去重、完整三路径定位链，以及噪声参考无法解析的早停。标准流程中的残差检测器被替换为一旦调用就失败的检查，以验证其未被执行；在线测试删除真值文件后仍能完成。

- 验证摘要：`/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01/validation/validation_summary.json`
- 逐项测试：`/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01/validation/pytest.xml`
- 实际日志：`/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01/run_records/20260914T045105_2012707/task.log`
- 结束记录：`/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_check_20260914_01/run_records/20260914T045105_2012707/completion.json`

同时使用 v5 配置通过了固定数据复用检查：30 个 UE、150 份 CSI、451 个被引用文件的摘要均核对通过，没有重新生成信号或覆盖旧结果。记录位于 `/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_change_20260914_01/reuse_check.json`。这些检查不等同于新一轮 300 次定位实验。

修改前的 41 个包内源码文件已完整保存在 `/data/zhujun/differt_projects/time-bias-correct/outputs/standard_music_change_20260914_01/source_before/time_bias_localization`，摘要与 v4 运行时冻结记录一致，便于复核历史算法。
