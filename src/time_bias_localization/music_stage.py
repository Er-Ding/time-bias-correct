"""MUSIC 计算阶段：分批提交 CSI 扰动，保留原算法的随机序列和峰定义。"""

from functools import lru_cache
from typing import Any

import numpy as np

from .signal import MusicPeakSamples, _complex_gaussian_noise, extract_local_music_peaks


@lru_cache(maxsize=4)
def get_music_computer(backend: str, device_id: int, batch_size: int, angle_chunk_size: int):
    # 进程内复用 GPU 上下文与导向向量；不同 worker 不共享设备对象。
    from .compute import ComputeSettings, MusicComputer
    return MusicComputer(ComputeSettings(backend=backend, device_id=device_id,
                         batch_size=batch_size, angle_chunk_size=angle_chunk_size))


def estimate_batched_peak_samples(
    computer: Any, csi_observed: np.ndarray, *, noise_std: float,
    num_repetitions: int, seed: int, peaks_per_repetition: int,
    minimum_relative_height: float, minimum_separation_bins: tuple[int, int],
    batch_size: int, **spectrum_parameters: Any,
) -> MusicPeakSamples:
    """仅从在线 CSI 构造扰动，不读取真实 UE、偏差或注噪强度。

    每个重复仍分别调用原有 NumPy 噪声生成函数，CPU/GPU 和不同批大小的
    随机样本逐元素一致。只改变谱计算的提交方式，不改变噪声模型。
    """
    for name, value in (("batch_size", batch_size), ("num_repetitions", num_repetitions),
                        ("peaks_per_repetition", peaks_per_repetition)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} 必须为正整数")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std 必须非负且有限")
    observed = np.asarray(csi_observed, dtype=np.complex128)
    if observed.ndim == 2:
        observed = observed[None]
    if observed.ndim != 3 or not np.all(np.isfinite(observed)):
        raise ValueError("在线 CSI 必须为有限的 [S,M,K] 数组")
    rng = np.random.default_rng(seed)
    shape = (num_repetitions, peaks_per_repetition)
    angles, delays, values = (np.full(shape, np.nan) for _ in range(3))
    for start in range(0, num_repetitions, batch_size):
        count = min(batch_size, num_repetitions - start)
        batch = np.stack([observed + _complex_gaussian_noise(observed.shape, noise_std, rng)
                          for _ in range(count)])
        spectra = computer.spectra(batch, **spectrum_parameters)
        for index, spectrum in enumerate(spectra):
            peaks = extract_local_music_peaks(
                spectrum, aoa_grid_rad=spectrum_parameters["aoa_grid_rad"],
                delay_grid_s=spectrum_parameters["delay_grid_s"],
                max_peaks=peaks_per_repetition, minimum_relative_height=minimum_relative_height,
                minimum_separation_bins=minimum_separation_bins,
            )
            for peak_index, peak in enumerate(peaks):
                angles[start + index, peak_index] = peak.aoa_rad
                delays[start + index, peak_index] = peak.delay_s
                values[start + index, peak_index] = peak.spectrum_value
    return MusicPeakSamples(aoa_rad=angles, delay_s=delays, spectrum_value=values)
