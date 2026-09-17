#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
MC_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
MC_PYTHON_BIN="${MC_PYTHON_BIN:-${MC_PROJECT_ROOT}/.sionna-venv/bin/python}"
MC_CONFIG_PATH="${MC_CONFIG_PATH:-${MC_PROJECT_ROOT}/configs/monte_carlo_continuous_munich.yaml}"
MC_OUTPUT_ROOT="${MC_OUTPUT_ROOT:-${MC_PROJECT_ROOT}/outputs/monte_carlo_munich_1000_20260915_01}"
# 留空沿用配置文件的 1000 个接受样本、4 个 CPU 定位进程；也可在此覆盖。
MC_SAMPLE_COUNT="${MC_SAMPLE_COUNT:-}"
MC_WORKERS="${MC_WORKERS:-}"
MC_GPU_ID="${MC_GPU_ID:-0}"          # nvidia-smi 中的物理 GPU 编号，仅射线追踪使用
MC_CPU_THREADS="${MC_CPU_THREADS:-1}" # 每个定位进程的数学库线程数
MC_CUDA_LIB_DIR="${MC_CUDA_LIB_DIR:-/usr/local/cuda-12.2/lib64}"
MC_MODE="${MC_MODE:-experiment}"     # experiment 或 check（CPU 测试）
MC_PREPARE_ONLY="${MC_PREPARE_ONLY:-0}" # 1：只生成样本/CSI；0：完成全部流程
# 场景、BS 坐标、35 dB 噪声、[-50,50] ns 偏置和求解预算在 MC_CONFIG_PATH 中修改。
# 同一目录再次 start 会续跑；改变已冻结参数或代码时应使用新目录。
# ====================================================================

MC_ACTION="${1:-start}"
[[ $# -le 2 ]] || { echo "用法：$0 start|log|status|stop [绝对输出目录]" >&2; exit 2; }
if [[ $# -eq 2 ]]; then MC_OUTPUT_ROOT="$2"; fi
MC_OUTPUT_ROOT="$(realpath -m -- "${MC_OUTPUT_ROOT}")"
if [[ "${MC_ACTION}" != start ]]; then
  [[ -f "${MC_OUTPUT_ROOT}/latest_run.txt" ]] || { echo "没有运行记录：${MC_OUTPUT_ROOT}" >&2; exit 1; }
  IFS= read -r MC_RUN_DIR <"${MC_OUTPUT_ROOT}/latest_run.txt"
  case "${MC_ACTION}" in
    log) exec tail -n 100 -F "${MC_RUN_DIR}/task.log" ;;
    status|stop) exec "${MC_PYTHON_BIN}" "${MC_PROJECT_ROOT}/scripts/detached_task.py" "${MC_ACTION}" "${MC_RUN_DIR}" ;;
    *) echo '动作只能为 start、log、status 或 stop。' >&2; exit 2 ;;
  esac
fi
cd "${MC_PROJECT_ROOT}"
[[ -x "${MC_PYTHON_BIN}" && -f "${MC_CONFIG_PATH}" ]] || { echo 'Python 或配置文件不存在。' >&2; exit 1; }
[[ "${MC_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo 'MC_CPU_THREADS 必须是正整数。' >&2; exit 2; }
[[ "${MC_PREPARE_ONLY}" == 0 || "${MC_PREPARE_ONLY}" == 1 ]] || { echo 'MC_PREPARE_ONLY 必须为 0 或 1。' >&2; exit 2; }
case "${MC_MODE}" in experiment|check) ;; *) echo 'MC_MODE 必须为 experiment 或 check。' >&2; exit 2;; esac
for MC_VALUE in "${MC_SAMPLE_COUNT}" "${MC_WORKERS}"; do
  [[ -z "${MC_VALUE}" || "${MC_VALUE}" =~ ^[1-9][0-9]*$ ]] || { echo '样本数和进程数必须留空或为正整数。' >&2; exit 2; }
done
command -v nohup >/dev/null
command -v setsid >/dev/null
command -v flock >/dev/null
export PYTHONPATH="${MC_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OPENBLAS_NUM_THREADS="${MC_CPU_THREADS}" OMP_NUM_THREADS="${MC_CPU_THREADS}" MKL_NUM_THREADS="${MC_CPU_THREADS}"
MC_ENV_LIB_DIR="$("${MC_PYTHON_BIN}" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix) / "lib")')"
export LD_LIBRARY_PATH="${MC_ENV_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ "${MC_MODE}" == experiment ]]; then
  [[ "${MC_GPU_ID}" =~ ^[0-9]+$ ]] || { echo 'MC_GPU_ID 必须是一张物理 GPU 的编号。' >&2; exit 2; }
  [[ -d "${MC_CUDA_LIB_DIR}" ]] || { echo "CUDA 动态库目录不存在：${MC_CUDA_LIB_DIR}" >&2; exit 1; }
  export LD_LIBRARY_PATH="${MC_CUDA_LIB_DIR}:${LD_LIBRARY_PATH}"
  export CUDA_PATH="$(dirname -- "${MC_CUDA_LIB_DIR}")" TBC_REQUIRE_CUDA=1
fi
# 先检查配置，不因拼写错误启动一项持续任务。
"${MC_PYTHON_BIN}" - "${MC_CONFIG_PATH}" "${MC_SAMPLE_COUNT}" "${MC_WORKERS}" <<'PYCHECK'
from pathlib import Path
import sys
from time_bias_localization.monte_carlo_experiment import load_settings
settings, config = load_settings(Path(sys.argv[1]), sample_count=int(sys.argv[2]) if sys.argv[2] else None,
                                 workers=int(sys.argv[3]) if sys.argv[3] else None)
print(f"配置检查通过：{settings['sample_count']} 个接受样本，{settings['workers']} 个定位进程，{config['radio']['snr_db']} dB。")
PYCHECK
mkdir -p -- "${MC_OUTPUT_ROOT}/run_records"
exec 9>"${MC_OUTPUT_ROOT}/.background_task.lock"
flock -n 9 || { echo '同一输出目录已有后台任务；请先查询状态。' >&2; exit 1; }
MC_RUN_DIR="${MC_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -- "${MC_RUN_DIR}"
if [[ "${MC_MODE}" == experiment ]]; then
  CUDA_VISIBLE_DEVICES="$("${MC_PYTHON_BIN}" "${MC_PROJECT_ROOT}/scripts/select_boundary_gpus.py" --gpu-ids "${MC_GPU_ID}" --record "${MC_RUN_DIR}/gpu_allocation.json")"
else
  CUDA_VISIBLE_DEVICES=''
fi
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLCONFIGDIR="${MC_OUTPUT_ROOT}/matplotlib_cache"
if [[ "${MC_MODE}" == check ]]; then
  MC_TASK=("${MC_PYTHON_BIN}" -u -m pytest -q tests/test_monte_carlo_experiment.py tests/test_path_policy.py
    tests/test_continuous_pipeline.py tests/test_continuous_hypothesis_search.py tests/test_continuous_solver.py
    tests/test_boundary_channel.py tests/test_boundary_experiment.py tests/test_boundary_worker.py
    tests/test_sionna_generation.py tests/test_contracts.py tests/test_config.py tests/test_compute.py
    tests/test_signal.py tests/test_spectrum_sampling.py --junitxml "${MC_RUN_DIR}/tests.xml")
else
  MC_TASK=("${MC_PYTHON_BIN}" -u -m time_bias_localization.monte_carlo_experiment
    --config "${MC_CONFIG_PATH}" --output "${MC_OUTPUT_ROOT}")
  [[ -z "${MC_SAMPLE_COUNT}" ]] || MC_TASK+=(--samples "${MC_SAMPLE_COUNT}")
  [[ -z "${MC_WORKERS}" ]] || MC_TASK+=(--workers "${MC_WORKERS}")
  [[ "${MC_PREPARE_ONLY}" == 0 ]] || MC_TASK+=(--prepare-only)
fi
nohup setsid "${MC_PYTHON_BIN}" -u "${MC_PROJECT_ROOT}/scripts/detached_task.py" run "${MC_RUN_DIR}" -- "${MC_TASK[@]}" \
  </dev/null >"${MC_RUN_DIR}/task.log" 2>&1 &
for ((MC_ATTEMPT=0; MC_ATTEMPT<50; MC_ATTEMPT++)); do
  [[ -f "${MC_RUN_DIR}/task.json" || -f "${MC_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${MC_RUN_DIR}/task.json" ]] || { cat "${MC_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${MC_RUN_DIR}" >"${MC_OUTPUT_ROOT}/latest_run.txt"
ln -sfn -- "${MC_RUN_DIR}/task.log" "${MC_OUTPUT_ROOT}/task.log"
"${MC_PYTHON_BIN}" - "${MC_RUN_DIR}" <<'PYVERIFY'
import json
from pathlib import Path
import sys
from scripts.detached_task import identity
root = Path(sys.argv[1])
record = json.loads((root / 'task.json').read_text())
if (root / 'completion.json').exists():
    result = json.loads((root / 'completion.json').read_text())
    if result['exit_code']:
        raise SystemExit(f"后台任务已失败，退出码={result['exit_code']}；日志={root / 'task.log'}")
    print('任务已经完成，退出码为 0。')
elif identity(record['task']['pid']) == record['task']:
    print(f"已核对实际任务 PID={record['task']['pid']}，正在后台运行。")
else:
    raise SystemExit('实际任务已退出，结束记录尚未写入，请查看日志和状态。')
PYVERIFY
head -n 10 "${MC_RUN_DIR}/task.log"
printf '输出目录：%s\n运行记录：%s\n实际日志：%s\n' "${MC_OUTPUT_ROOT}" "${MC_RUN_DIR}" "${MC_RUN_DIR}/task.log"
printf '实时日志：tail -n 100 -F %q\n' "${MC_OUTPUT_ROOT}/task.log"
printf '查看状态：bash %q status %q\n' "${MC_PROJECT_ROOT}/run_monte_carlo_experiment.sh" "${MC_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${MC_PROJECT_ROOT}/run_monte_carlo_experiment.sh" "${MC_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务；相同参数再次 start 可续跑。\n'
