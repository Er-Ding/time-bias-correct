# 绕射超时修复、GPU 选择和预跑恢复（2026-09-11）

原实验 `outputs/diffraction_boundary_v1` 完成了 30 个合格 UE 的选择及 150 份噪声 CSI 生成，但第一次单代表预热在绕射反向追踪处超过 600 秒，退出码为 1，尚未产生计划定位记录。原目录和失败现场完整保留。

## 已修改

1. **绕射反向追踪**：将大量逐墙计算改为批量镜像回溯、批量遮挡检查，并缓存只依赖公开地图、BS 和反射上限的公共路径表。未减少反射次数、采样数、墙面或实验范围。首次构建耗时照常记录，后续复用单独标记；约每 5 秒输出已处理边缘数量。
2. **预热与请求失败**：预热失败后写独立记录，重建两组进程，继续固定 UE 的实际请求。不会删除失败 UE 或无限重复困难预热。请求超时、进程崩溃、初始化失败分别保留记录，并继续后续计划。
3. **公平计时**：单代表和多代表各用独立常驻进程，请求仍顺序执行。分别记录首次冷启动、已预热和复用状态；重启后只剩半对且缓存状态不对应的记录不用于成对时间差，但精度和主耗时统计继续保留。
4. **显式选择 GPU**：每次启动必须提供 `BOUNDARY_GPU_IDS`。脚本把 `nvidia-smi` 物理编号转换为 GPU 唯一编号，再设置全部子进程的 `CUDA_VISIBLE_DEVICES`。程序检查配置里的逻辑编号，避免把物理 GPU 2 错当作所选列表中的第三张卡。管理日志、状态、停止命令无需再次指定 GPU。
5. **环境与数据恢复**：启动脚本固定 Python 环境的 C++ 库和 CUDA 路径，避免依赖旧终端变量。修复后使用新输出目录，通过 `BOUNDARY_SOURCE_ROOT` 核对并引用原 CSI、真值、场景和生成清单，不重新采样或改写旧数据。生成参数必须一致；新的计算设备和代码版本独立记录。

主要入口：

- `/data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh`
- `/data/zhujun/differt_projects/time-bias-correct/scripts/select_boundary_gpus.py`
- `/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/diffraction_prefixes.py`
- `/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/boundary_reuse.py`
- `/data/zhujun/differt_projects/time-bias-correct/src/time_bias_localization/boundary_experiment.py`

## 最终验证

完整回归 **726 项通过、1 项跳过**；12 次合成场景流程请求全部完成。与修复前保留的固定小样本比较，12 次最终位置的最大差值为 0，公共偏差也一致。针对预热失败、真实子进程硬超时及强制回收、崩溃和半对恢复均有专门测试。

同一冻结 `PILOT_0001` CSI 在 CPU 上实测，完整反向追踪首次 **57.873 秒**，复用公共路径表后 **4.783 秒**。两次均为 **542 个候选**，候选全部字段的 SHA-256 相同。这是单份 CSI 的反向阶段性能，不是全体 UE 的平均端到端时间。

真实地图还独立遍历了全部 **1440 面首反射墙**，完全跳过首墙筛选：与筛选版本均得到 **373 条公共路径**，传播顺序、反射点、长度、方向等全部字段完全相同，最大差值为 0。核对只读取公开地图和 BS，没有使用 UE 真值。极窄开口和极短交叉墙的退化案例也已验证。

GPU 运行探针使用本次授权的物理 GPU **2～7**，按探针 PID 查询 `nvidia-smi`，未观察到它占用物理 0、1。Sionna/DrJit 导入会在所有授权卡建立上下文，每张约 308～340 MiB；CuPy 的计算设备逻辑 0 对应物理 GPU 2。探针结束后相关占用释放。

证据：

- 完整测试和小流程：`/data/zhujun/differt_projects/time-bias-correct/outputs/boundary_repair_validation_20260911_02/run_records/20260911T085045_406620/validation_summary.json`
- 最终反向阶段性能：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_performance_20260911T084811_403002/artifacts/performance.json`
- 真实地图完整路径表核对：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_performance_20260911T085153_408393/artifacts/equivalence.json`
- GPU 实际隔离验证：`/data/zhujun/differt_projects/time-bias-correct/outputs/gpu_runtime_check_20260911_agent_final`
- 中间版本检查也保留；其更快的初建耗时或成功退出不能替代上述最终源版本验证。

## 本次恢复命令和运行目录

以下为本次实际使用的命令。以后运行时按当时情况重新填写 GPU 编号，不默认所有卡可用。

```bash
BOUNDARY_GPU_IDS=2,3,4,5,6,7 \
BOUNDARY_SOURCE_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v1 \
BOUNDARY_OUTPUT_ROOT=/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2 \
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh
```

运行记录：`/data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2/run_records/20260911T085523_413892`。GPU 配置在其中的 `gpu_allocation.json`。此次定位采用 CUDA，所选列表中的第一张卡为物理 GPU 2；全部 300 次定位请求顺序计时，不同时铺满 6 张卡。正式 300 UE 阶段不会自动启动。

```bash
# 实时日志；Ctrl+C 只退出查看。
tail -n 100 -F /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2/run_records/20260911T085523_413892/task.log

# 查询实际进程及最终退出码。
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh status /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2

# 核对身份后停止整组任务。
bash /data/zhujun/differt_projects/time-bias-correct/run_boundary_experiment.sh stop /data/zhujun/differt_projects/time-bias-correct/outputs/diffraction_boundary_v2
```

恢复启动和实现验证不表示预跑已全部完成；当前进度请以该运行目录的日志、`pilot/trials.jsonl` 和最终 `completion.json` 为准。
