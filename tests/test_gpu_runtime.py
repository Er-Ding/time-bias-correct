import json
import pytest
from time_bias_localization.gpu_runtime import validate_gpu_selection


def test_no_gpu_selection_is_rejected_before_simulation(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('BOUNDARY_GPU_ALLOCATION_PATH', raising=False)
    with pytest.raises(ValueError, match='明确指定'):
        validate_gpu_selection({'compute': {'backend': 'numpy'}}, channel_backend='sionna')
    validate_gpu_selection({'compute': {'backend': 'numpy'}}, channel_backend='synthetic_fixture')


def test_logical_device_bounds_and_record_match(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-first,GPU-second')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    path = tmp_path / 'allocation.json'
    path.write_text(json.dumps({'cuda_visible_devices': 'GPU-first,GPU-second', 'cuda_device_order': 'PCI_BUS_ID'}))
    monkeypatch.setenv('BOUNDARY_GPU_ALLOCATION_PATH', str(path))
    validate_gpu_selection({'compute': {'backend': 'cuda', 'device_id': 0}}, channel_backend='sionna')
    with pytest.raises(ValueError, match='逻辑编号'):
        validate_gpu_selection({'compute': {'backend': 'cuda', 'device_id': 2}}, channel_backend='sionna')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-another')
    with pytest.raises(ValueError, match='选择记录'):
        validate_gpu_selection({'compute': {'backend': 'cuda', 'device_id': 0}}, channel_backend='sionna')
