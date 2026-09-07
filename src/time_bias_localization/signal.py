"""用于二维定位仿真的 CSI 与二维 MUSIC 信号处理。

本模块只使用 NumPy。约定 UE 为单天线发射端，BS 为均匀线阵接收端。
CSI 的最后两个维度始终为 ``(BS 天线, 子载波)``。

角度约定
--------
``aoa_rad`` 是 BS 阵列局部坐标系中的到达角，单位为弧度，取值范围为
``[-pi/2, pi/2]``。零度表示阵列正侧向（broadside）。第 ``m`` 个阵元的
位置为 ``m * antenna_spacing_m``，空间响应采用与 Sionna RT 接收阵列一致的
``exp(+j 2 pi x_m sin(aoa) / wavelength)``。全局方位角应由调用方根据 BS
朝向转换为这里的局部角度。

时延与频率约定
------------
``delay_s`` 与 ``common_delay_bias_s`` 均以秒为单位。子载波频率
``subcarrier_frequencies_hz`` 可以是绝对频率，也可以是相对载频的基带
频偏，但生成、注入偏差和 MUSIC 必须使用同一组数值。频率响应采用
``exp(-j 2 pi f_k delay)``，所以正的公共偏差会使观测时延增大。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .constants import SPEED_OF_LIGHT_M_S



@dataclass(frozen=True)
class MusicPeak2D:
    """二维 MUSIC 谱中的一个局部峰。

    ``spectrum_value`` 只是 MUSIC 伪谱值，不是概率，也不应当归一化后
    直接解释为候选路径概率。
    """

    aoa_rad: float
    delay_s: float
    spectrum_value: float
    aoa_index: int
    delay_index: int


@dataclass(frozen=True)
class MusicPeakSamples:
    """重复扰动得到的峰样本。

    三个数组形状均为 ``(num_repetitions, peaks_per_repetition)``。某次扰动
    找不到足够多的局部峰时，空位为 ``NaN``。谱值仅用于排序与诊断，不是
    概率。
    """

    aoa_rad: np.ndarray
    delay_s: np.ndarray
    spectrum_value: np.ndarray


def _as_real_vector(name: str, values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，当前形状为 {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 包含 NaN 或无穷大")
    return array


def _as_complex_vector(
    name: str, values: Sequence[complex] | np.ndarray
) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，当前形状为 {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} 不能为空")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} 包含 NaN 或无穷大")
    return array


def _validate_frequency_vector(
    subcarrier_frequencies_hz: Sequence[float] | np.ndarray,
    *,
    require_uniform_spacing: bool,
) -> np.ndarray:
    frequencies = _as_real_vector(
        "subcarrier_frequencies_hz", subcarrier_frequencies_hz
    )
    if frequencies.size < 2:
        raise ValueError("至少需要两个子载波")
    differences = np.diff(frequencies)
    if not np.all(differences > 0.0):
        raise ValueError("subcarrier_frequencies_hz 必须严格递增")
    if require_uniform_spacing:
        tolerance = max(abs(float(differences[0])) * 1e-9, 1e-9)
        if not np.allclose(differences, differences[0], rtol=1e-9, atol=tolerance):
            raise ValueError("二维 MUSIC 的子载波必须等间隔")
    return frequencies


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")
    return scalar


def ula_steering_vector(
    aoa_rad: float | np.ndarray,
    *,
    num_bs_antennas: int,
    carrier_frequency_hz: float,
    antenna_spacing_m: float | None = None,
) -> np.ndarray:
    """计算 BS 均匀线阵接收响应。

    参数
    ----
    aoa_rad:
        BS 局部坐标系到达角，弧度。标量返回 ``(M,)``，形状为 ``(...)`` 的
        数组返回 ``(M, ...)``。
    num_bs_antennas:
        BS 阵元数 ``M``。
    carrier_frequency_hz:
        用于空间相位的载频，单位 Hz。
    antenna_spacing_m:
        阵元间距，单位 m；省略时使用半波长。
    """

    if isinstance(num_bs_antennas, bool) or int(num_bs_antennas) != num_bs_antennas:
        raise ValueError("num_bs_antennas 必须是整数")
    num_antennas = int(num_bs_antennas)
    if num_antennas < 1:
        raise ValueError("num_bs_antennas 至少为 1")

    carrier_frequency = _validate_positive_scalar(
        "carrier_frequency_hz", carrier_frequency_hz
    )
    wavelength_m = SPEED_OF_LIGHT_M_S / carrier_frequency
    spacing_m = (
        wavelength_m / 2.0
        if antenna_spacing_m is None
        else _validate_positive_scalar("antenna_spacing_m", antenna_spacing_m)
    )

    angles = np.asarray(aoa_rad, dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise ValueError("aoa_rad 包含 NaN 或无穷大")
    angle_tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(angles < -np.pi / 2.0 - angle_tolerance) or np.any(
        angles > np.pi / 2.0 + angle_tolerance
    ):
        raise ValueError("aoa_rad 必须位于 [-pi/2, pi/2]")

    positions_m = np.arange(num_antennas, dtype=np.float64) * spacing_m
    phase = 2.0j * np.pi * positions_m.reshape((-1,) + (1,) * angles.ndim)
    phase = phase * np.sin(angles) / wavelength_m
    response = np.exp(phase)
    if angles.ndim == 0:
        return response.reshape(num_antennas)
    return response


def synthesize_ula_csi(
    *,
    path_aoa_rad: Sequence[float] | np.ndarray,
    path_delay_s: Sequence[float] | np.ndarray,
    path_coefficients: Sequence[complex] | np.ndarray,
    subcarrier_frequencies_hz: Sequence[float] | np.ndarray,
    num_bs_antennas: int,
    carrier_frequency_hz: float,
    antenna_spacing_m: float | None = None,
) -> np.ndarray:
    """由二维传播路径合成 UE 到 BS 的频域 CSI。

    每条路径由 BS 局部到达角、绝对传播时延和一个复系数描述。返回复数数组
    ``H``，形状为 ``(M, K)``，其中 ``M`` 为 BS 阵元数，``K`` 为子载波数：

    ``H[m, k] = sum_p alpha[p] * a[m, aoa[p]] * exp(-j 2 pi f[k] delay[p])``。

    路径系数可以包含传播损耗与不随子载波变化的公共相位。函数不做功率
    归一化，也不会移除绝对传播时延。
    """

    aoa = _as_real_vector("path_aoa_rad", path_aoa_rad)
    delays = _as_real_vector("path_delay_s", path_delay_s)
    coefficients = _as_complex_vector("path_coefficients", path_coefficients)
    if not (aoa.size == delays.size == coefficients.size):
        raise ValueError(
            "path_aoa_rad、path_delay_s 与 path_coefficients 的长度必须相同"
        )
    if np.any(delays < 0.0):
        raise ValueError("path_delay_s 必须是非负的绝对传播时延")

    frequencies = _validate_frequency_vector(
        subcarrier_frequencies_hz, require_uniform_spacing=False
    )
    spatial_response = ula_steering_vector(
        aoa,
        num_bs_antennas=num_bs_antennas,
        carrier_frequency_hz=carrier_frequency_hz,
        antenna_spacing_m=antenna_spacing_m,
    )
    frequency_response = np.exp(
        -2.0j * np.pi * delays[:, np.newaxis] * frequencies[np.newaxis, :]
    )
    return np.einsum(
        "mp,pk,p->mk", spatial_response, frequency_response, coefficients
    )


def _complex_gaussian_noise(
    shape: tuple[int, ...], noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    """生成 E[|n|^2] = noise_std^2 的圆对称复高斯噪声。"""

    if noise_std == 0.0:
        return np.zeros(shape, dtype=np.complex128)
    component_std = noise_std / np.sqrt(2.0)
    return component_std * (
        rng.standard_normal(shape) + 1.0j * rng.standard_normal(shape)
    )


def apply_common_delay_bias(
    csi_geometric: np.ndarray,
    subcarrier_frequencies_hz: Sequence[float] | np.ndarray,
    common_delay_bias_s: float,
    *,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """给 CSI 注入统一公共时延偏差，并可加入复高斯噪声。

    输入最后一维必须是子载波，常用形状为 ``(M, K)``。严格执行

    ``H_obs[..., k] = H_geo[..., k] * exp(-j 2 pi f[k] b) + noise``。

    因而 ``b > 0`` 对应观测到的路径时延统一增加。``noise_std`` 是单个
    复数 CSI 元素的均方根噪声，即 ``sqrt(E[|n|^2])``。随机数只由 ``seed``
    控制；不需要 UE 真值、干净路径或真实偏差以外的任何隐藏输入。
    """

    csi = np.asarray(csi_geometric, dtype=np.complex128)
    if csi.ndim < 1:
        raise ValueError("csi_geometric 至少需要一个维度")
    if not np.all(np.isfinite(csi.real)) or not np.all(np.isfinite(csi.imag)):
        raise ValueError("csi_geometric 包含 NaN 或无穷大")
    frequencies = _validate_frequency_vector(
        subcarrier_frequencies_hz, require_uniform_spacing=False
    )
    if csi.shape[-1] != frequencies.size:
        raise ValueError(
            "csi_geometric 最后一维必须与 subcarrier_frequencies_hz 等长"
        )

    bias_s = float(common_delay_bias_s)
    if not np.isfinite(bias_s):
        raise ValueError("common_delay_bias_s 必须是有限数")
    rms_noise = float(noise_std)
    if not np.isfinite(rms_noise) or rms_noise < 0.0:
        raise ValueError("noise_std 必须是有限非负数")

    delay_phase = np.exp(-2.0j * np.pi * frequencies * bias_s)
    biased = csi * delay_phase
    rng = np.random.default_rng(seed)
    return biased + _complex_gaussian_noise(csi.shape, rms_noise, rng)


def _prepare_music_input(csi: np.ndarray, num_subcarriers: int) -> np.ndarray:
    array = np.asarray(csi, dtype=np.complex128)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        raise ValueError("csi 必须为 (M, K) 或 (S, M, K)")
    if array.shape[-1] != num_subcarriers:
        raise ValueError("csi 的子载波维度与频率数组长度不一致")
    if array.shape[-2] < 2:
        raise ValueError("二维 MUSIC 至少需要两个 BS 阵元")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError("csi 包含 NaN 或无穷大")
    return array


def _validate_subarray_size(name: str, value: int | None, full_size: int) -> int:
    if value is None:
        return full_size // 2 + 1
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} 必须是整数")
    size = int(value)
    if size < 2 or size > full_size:
        raise ValueError(f"{name} 必须位于 [2, {full_size}]")
    return size


def _smoothed_covariance(
    csi_snapshots: np.ndarray,
    spatial_subarray_size: int,
    frequency_subarray_size: int,
) -> np.ndarray:
    """逐快照累计二维平滑协方差，避免一次堆叠全部滑窗。"""

    _, num_antennas, num_subcarriers = csi_snapshots.shape
    vector_size = spatial_subarray_size * frequency_subarray_size
    covariance = np.zeros((vector_size, vector_size), dtype=np.complex128)
    vector_count = 0
    for snapshot in csi_snapshots:
        windows = np.lib.stride_tricks.sliding_window_view(
            snapshot, (spatial_subarray_size, frequency_subarray_size)
        )
        data_matrix = windows.reshape(-1, vector_size).T
        covariance += data_matrix @ data_matrix.conj().T
        vector_count += data_matrix.shape[1]
    covariance /= vector_count
    return (covariance + covariance.conj().T) / 2.0


def music_2d_spectrum(
    csi: np.ndarray,
    *,
    subcarrier_frequencies_hz: Sequence[float] | np.ndarray,
    carrier_frequency_hz: float,
    antenna_spacing_m: float | None,
    aoa_grid_rad: Sequence[float] | np.ndarray,
    delay_grid_s: Sequence[float] | np.ndarray,
    num_sources: int = 1,
    spatial_subarray_size: int | None = None,
    frequency_subarray_size: int | None = None,
    diagonal_loading: float = 0.0,
) -> np.ndarray:
    """计算 AoA × Delay 二维 MUSIC 伪谱。

    ``csi`` 可为一个 CSI ``(M, K)``，也可为多个独立快照
    ``(S, M, K)``。函数使用二维滑动子阵列形成协方差，因此等间隔子载波是
    必需条件。输出形状为 ``(len(aoa_grid_rad), len(delay_grid_s))``。

    ``num_sources`` 是信号子空间中的路径数量。``diagonal_loading`` 是相对
    于协方差平均对角功率的非负加载系数。返回值是未做概率归一化的 MUSIC
    伪谱，数值大小只能用于同一张谱内寻找峰值。
    """

    frequencies = _validate_frequency_vector(
        subcarrier_frequencies_hz, require_uniform_spacing=True
    )
    snapshots = _prepare_music_input(csi, frequencies.size)
    aoa_grid = _as_real_vector("aoa_grid_rad", aoa_grid_rad)
    delay_grid = _as_real_vector("delay_grid_s", delay_grid_s)
    if np.any(delay_grid < 0.0):
        raise ValueError("delay_grid_s 必须为非负时延")
    if np.any(np.diff(delay_grid) <= 0.0):
        raise ValueError("delay_grid_s 必须严格递增")
    if np.any(np.diff(aoa_grid) <= 0.0):
        raise ValueError("aoa_grid_rad 必须严格递增")

    num_antennas = snapshots.shape[-2]
    spatial_size = _validate_subarray_size(
        "spatial_subarray_size", spatial_subarray_size, num_antennas
    )
    frequency_size = _validate_subarray_size(
        "frequency_subarray_size", frequency_subarray_size, frequencies.size
    )
    subspace_dimension = spatial_size * frequency_size
    if isinstance(num_sources, bool) or int(num_sources) != num_sources:
        raise ValueError("num_sources 必须是整数")
    source_count = int(num_sources)
    if source_count < 1 or source_count >= subspace_dimension:
        raise ValueError(
            f"num_sources 必须位于 [1, {subspace_dimension - 1}]"
        )
    loading = float(diagonal_loading)
    if not np.isfinite(loading) or loading < 0.0:
        raise ValueError("diagonal_loading 必须是有限非负数")

    covariance = _smoothed_covariance(snapshots, spatial_size, frequency_size)
    if loading > 0.0:
        mean_diagonal_power = float(np.trace(covariance).real / subspace_dimension)
        covariance = covariance + (
            loading * mean_diagonal_power * np.eye(subspace_dimension)
        )
    _, eigenvectors = np.linalg.eigh(covariance)
    # 对单位范数导向向量 a，有 ||E_n^H a||^2 = 1-||E_s^H a||^2。
    # 信号子空间维数通常只有路径数，使用右式可显著减少整张二维谱的计算量。
    signal_subspace = eigenvectors[:, -source_count:]

    spatial_steering = ula_steering_vector(
        aoa_grid,
        num_bs_antennas=spatial_size,
        carrier_frequency_hz=carrier_frequency_hz,
        antenna_spacing_m=antenna_spacing_m,
    )
    relative_frequencies = frequencies[:frequency_size] - frequencies[0]
    frequency_steering = np.exp(
        -2.0j
        * np.pi
        * relative_frequencies[:, np.newaxis]
        * delay_grid[np.newaxis, :]
    )

    spectrum = np.empty((aoa_grid.size, delay_grid.size), dtype=np.float64)
    numerical_floor = np.finfo(np.float64).eps
    for angle_index in range(aoa_grid.size):
        steering = (
            spatial_steering[:, angle_index, np.newaxis, np.newaxis]
            * frequency_steering[np.newaxis, :, :]
        ).reshape(subspace_dimension, delay_grid.size)
        steering /= math.sqrt(subspace_dimension)
        signal_projection = signal_subspace.conj().T @ steering
        explained_power = np.sum(np.abs(signal_projection) ** 2, axis=0)
        denominator = np.maximum(1.0 - explained_power.real, numerical_floor)
        spectrum[angle_index] = 1.0 / denominator
    return spectrum


def extract_local_music_peaks(
    spectrum: np.ndarray,
    *,
    aoa_grid_rad: Sequence[float] | np.ndarray,
    delay_grid_s: Sequence[float] | np.ndarray,
    max_peaks: int | None = None,
    minimum_relative_height: float = 0.0,
    minimum_separation_bins: tuple[int, int] = (1, 1),
) -> list[MusicPeak2D]:
    """提取二维 MUSIC 谱的局部峰并做确定性的网格非极大值抑制。

    ``minimum_relative_height`` 是相对本张谱最大值的筛选门槛，不表示概率。
    ``minimum_separation_bins=(a, d)`` 表示已经选中的峰周围 ``a`` 个角度格、
    ``d`` 个时延格内不再重复选峰。相同谱值按角度索引、时延索引排序，保证
    结果可复现。
    """

    values = np.asarray(spectrum, dtype=np.float64)
    aoa_grid = _as_real_vector("aoa_grid_rad", aoa_grid_rad)
    delay_grid = _as_real_vector("delay_grid_s", delay_grid_s)
    if values.shape != (aoa_grid.size, delay_grid.size):
        raise ValueError(
            "spectrum 形状必须为 (len(aoa_grid_rad), len(delay_grid_s))"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("spectrum 包含 NaN 或无穷大")
    if np.any(values < 0.0):
        raise ValueError("MUSIC spectrum 不应包含负数")

    if max_peaks is not None:
        if isinstance(max_peaks, bool) or int(max_peaks) != max_peaks:
            raise ValueError("max_peaks 必须是整数或 None")
        if int(max_peaks) < 1:
            raise ValueError("max_peaks 至少为 1")
        peak_limit = int(max_peaks)
    else:
        peak_limit = None

    relative_height = float(minimum_relative_height)
    if not np.isfinite(relative_height) or not 0.0 <= relative_height <= 1.0:
        raise ValueError("minimum_relative_height 必须位于 [0, 1]")
    if len(minimum_separation_bins) != 2:
        raise ValueError("minimum_separation_bins 必须包含两个整数")
    separation: list[int] = []
    for item in minimum_separation_bins:
        if isinstance(item, bool) or int(item) != item or int(item) < 0:
            raise ValueError("minimum_separation_bins 必须是两个非负整数")
        separation.append(int(item))

    threshold = float(np.max(values)) * relative_height
    candidates: list[tuple[float, int, int]] = []
    for angle_index in range(values.shape[0]):
        angle_start = max(0, angle_index - 1)
        angle_stop = min(values.shape[0], angle_index + 2)
        for delay_index in range(values.shape[1]):
            value = float(values[angle_index, delay_index])
            if value < threshold:
                continue
            delay_start = max(0, delay_index - 1)
            delay_stop = min(values.shape[1], delay_index + 2)
            neighborhood = values[
                angle_start:angle_stop, delay_start:delay_stop
            ]
            if value >= float(np.max(neighborhood)):
                candidates.append((-value, angle_index, delay_index))

    candidates.sort()
    selected: list[MusicPeak2D] = []
    for negative_value, angle_index, delay_index in candidates:
        overlaps = any(
            abs(angle_index - peak.aoa_index) <= separation[0]
            and abs(delay_index - peak.delay_index) <= separation[1]
            for peak in selected
        )
        if overlaps:
            continue
        selected.append(
            MusicPeak2D(
                aoa_rad=float(aoa_grid[angle_index]),
                delay_s=float(delay_grid[delay_index]),
                spectrum_value=-negative_value,
                aoa_index=angle_index,
                delay_index=delay_index,
            )
        )
        if peak_limit is not None and len(selected) >= peak_limit:
            break
    return selected


def estimate_music_peak_samples(
    csi_observed: np.ndarray,
    *,
    subcarrier_frequencies_hz: Sequence[float] | np.ndarray,
    carrier_frequency_hz: float,
    antenna_spacing_m: float | None,
    aoa_grid_rad: Sequence[float] | np.ndarray,
    delay_grid_s: Sequence[float] | np.ndarray,
    noise_std: float,
    num_repetitions: int,
    seed: int = 0,
    num_sources: int = 1,
    peaks_per_repetition: int = 1,
    spatial_subarray_size: int | None = None,
    frequency_subarray_size: int | None = None,
    diagonal_loading: float = 0.0,
    minimum_relative_height: float = 0.0,
    minimum_separation_bins: tuple[int, int] = (1, 1),
) -> MusicPeakSamples:
    """对观测 CSI 做可复现噪声扰动，收集二维 MUSIC 峰样本。

    这是在线可用接口：输入只有观测 CSI、阵列/频率网格和调用方给出的
    ``noise_std``。函数不会读取干净 CSI、真实路径、真实 UE 位置、请求的
    SNR 或注入时使用的真实噪声。每次在观测 CSI 上叠加一份独立的圆对称
    复高斯扰动，再重新计算 MUSIC 谱和局部峰。
    """

    rms_noise = float(noise_std)
    if not np.isfinite(rms_noise) or rms_noise < 0.0:
        raise ValueError("noise_std 必须是有限非负数")
    if (
        isinstance(num_repetitions, bool)
        or int(num_repetitions) != num_repetitions
        or int(num_repetitions) < 1
    ):
        raise ValueError("num_repetitions 必须是正整数")
    if (
        isinstance(peaks_per_repetition, bool)
        or int(peaks_per_repetition) != peaks_per_repetition
        or int(peaks_per_repetition) < 1
    ):
        raise ValueError("peaks_per_repetition 必须是正整数")

    repetitions = int(num_repetitions)
    peaks_per_repeat = int(peaks_per_repetition)
    observed = np.asarray(csi_observed, dtype=np.complex128)
    frequencies = _validate_frequency_vector(
        subcarrier_frequencies_hz, require_uniform_spacing=True
    )
    _prepare_music_input(observed, frequencies.size)
    if not np.all(np.isfinite(observed.real)) or not np.all(
        np.isfinite(observed.imag)
    ):
        raise ValueError("csi_observed 包含 NaN 或无穷大")

    aoa_samples = np.full((repetitions, peaks_per_repeat), np.nan)
    delay_samples = np.full((repetitions, peaks_per_repeat), np.nan)
    value_samples = np.full((repetitions, peaks_per_repeat), np.nan)
    rng = np.random.default_rng(seed)

    for repetition_index in range(repetitions):
        perturbed = observed + _complex_gaussian_noise(
            observed.shape, rms_noise, rng
        )
        spectrum = music_2d_spectrum(
            perturbed,
            subcarrier_frequencies_hz=frequencies,
            carrier_frequency_hz=carrier_frequency_hz,
            antenna_spacing_m=antenna_spacing_m,
            aoa_grid_rad=aoa_grid_rad,
            delay_grid_s=delay_grid_s,
            num_sources=num_sources,
            spatial_subarray_size=spatial_subarray_size,
            frequency_subarray_size=frequency_subarray_size,
            diagonal_loading=diagonal_loading,
        )
        peaks = extract_local_music_peaks(
            spectrum,
            aoa_grid_rad=aoa_grid_rad,
            delay_grid_s=delay_grid_s,
            max_peaks=peaks_per_repeat,
            minimum_relative_height=minimum_relative_height,
            minimum_separation_bins=minimum_separation_bins,
        )
        for peak_index, peak in enumerate(peaks):
            aoa_samples[repetition_index, peak_index] = peak.aoa_rad
            delay_samples[repetition_index, peak_index] = peak.delay_s
            value_samples[repetition_index, peak_index] = peak.spectrum_value

    return MusicPeakSamples(
        aoa_rad=aoa_samples,
        delay_s=delay_samples,
        spectrum_value=value_samples,
    )


__all__ = [
    "MusicPeak2D",
    "MusicPeakSamples",
    "SPEED_OF_LIGHT_M_S",
    "apply_common_delay_bias",
    "estimate_music_peak_samples",
    "extract_local_music_peaks",
    "music_2d_spectrum",
    "synthesize_ula_csi",
    "ula_steering_vector",
]
