"""实验入口的 GPU 显式选择检查；此模块不初始化 CUDA。"""
import json
import os
from pathlib import Path


def validate_gpu_selection(config, *, channel_backend):
    needs_gpu = channel_backend == 'sionna' or config.get('compute', {}).get('backend') == 'cuda'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = [] if not visible else visible.split(',')
    if needs_gpu and (not devices or any(not value.strip() or value.strip() == '-1' for value in devices)):
        raise ValueError('GPU 尚未明确指定；请通过 run_boundary_experiment.sh 填写 BOUNDARY_GPU_IDS 后启动')
    if len(set(devices)) != len(devices):
        raise ValueError('CUDA_VISIBLE_DEVICES 不允许重复设备')
    allocation_path = os.environ.get('BOUNDARY_GPU_ALLOCATION_PATH')
    if allocation_path:
        allocation = json.loads(Path(allocation_path).read_text())
        if (allocation['cuda_visible_devices'] != (visible or '')
                or allocation['cuda_device_order'] != os.environ.get('CUDA_DEVICE_ORDER')):
            raise ValueError('实际 CUDA 环境与用户 GPU 选择记录不一致')
    if config.get('compute', {}).get('backend') == 'cuda':
        device = config['compute'].get('device_id', 0)
        if type(device) is not int or not 0 <= device < len(devices):
            raise ValueError(f'compute.device_id 必须是所选 GPU 内的逻辑编号 0～{len(devices)-1}，不能填写物理编号')
