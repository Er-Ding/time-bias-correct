"""可复用的二维 MUSIC 计算器：NumPy 参考设备与 CuPy CUDA 批量计算。

此层只接收观测 CSI、阵列参数和搜索网格，不接触真实位置或真实偏差。
批次中的每份 CSI 独立估计协方差；绝不混合不同 UE 或扰动的观测。
所有输入在主机端校验，返回值始终为 NumPy 数组，便于现有求解及绘图复用。
"""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import math
from typing import Any, Sequence

import numpy as np

from .signal import (
    _as_real_vector,
    _prepare_music_input,
    _validate_frequency_vector,
    _validate_subarray_size,
    ula_steering_vector,
)


@dataclass(frozen=True)
class ComputeSettings:
    """计算设备与内存控制；不包含任何改变定位模型的参数。"""

    backend: str = "numpy"
    device_id: int = 0
    batch_size: int = 4
    angle_chunk_size: int = 32

    def __post_init__(self) -> None:
        if self.backend not in ("numpy", "cuda"):
            raise ValueError("compute.backend 必须是 numpy 或 cuda")
        for name, minimum in (
            ("device_id", 0), ("batch_size", 1), ("angle_chunk_size", 1)
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"compute.{name} 必须是整数")
            if value < minimum:
                raise ValueError(f"compute.{name} 必须至少为 {minimum}")


class MusicComputer:
    """同一进程内复用设备和导向向量，对独立 CSI 进行批量计算。

    一个实例固定使用一个设备；多进程任务应在各自进程内创建实例。
    CUDA 被明确请求而无法使用时直接报错，避免实验悄悄在 CPU 上运行。
    """

    _CACHE_MAX_ENTRIES = 4
    _CACHE_MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, settings: ComputeSettings | None = None) -> None:
        self.settings = settings or ComputeSettings()
        self._xp: Any = np
        self._device_name = "CPU"
        self._device = None
        self._cupyx: Any = None
        self._cuda_version: int | None = None
        self._cache: OrderedDict[tuple[Any, ...], tuple[Any, Any]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_count = 0
        self._csi_count = 0
        self._largest_batch = 0
        self._covariance_count = 0
        self._eigendecomposition_count = 0
        if self.settings.backend == "cuda":
            try:
                cupy = importlib.import_module("cupy")
                count = int(cupy.cuda.runtime.getDeviceCount())
                if self.settings.device_id >= count:
                    raise RuntimeError(
                        f"指定 GPU {self.settings.device_id}，但当前只能看到 {count} 张 GPU"
                    )
                self._device = cupy.cuda.Device(self.settings.device_id)
                with self._device:
                    properties = cupy.cuda.runtime.getDeviceProperties(
                        self.settings.device_id
                    )
                    name = properties["name"]
                    self._device_name = (
                        name.decode("utf-8") if isinstance(name, bytes) else str(name)
                    )
                    # 小分配使驱动、运行时不兼容尽早明确失败。
                    cupy.empty(1, dtype=cupy.float64)
                    self._cuda_version = int(cupy.cuda.runtime.runtimeGetVersion())
                self._xp = cupy
                self._cupyx = importlib.import_module("cupyx")
            except Exception as error:
                raise RuntimeError(
                    "已要求 CUDA MUSIC，但无法初始化 GPU。请检查 CuPy、CUDA 驱动"
                    "和 compute.device_id；本次不会自动改用 CPU。"
                    f" 原因：{error}"
                ) from error

    def metadata(self) -> dict[str, Any]:
        """可写入运行记录的实际设备、精度及缓存/批量统计。"""

        return {
            "requested_backend": self.settings.backend,
            "backend": self.settings.backend,
            "array_library": "cupy" if self.settings.backend == "cuda" else "numpy",
            "array_library_version": str(self._xp.__version__),
            "device_id": int(self.settings.device_id) if self._device else None,
            "device_name": self._device_name,
            "cuda_runtime_version": self._cuda_version,
            "complex_dtype": "complex128",
            "real_dtype": "float64",
            "batch_size": int(self.settings.batch_size),
            "angle_chunk_size": int(self.settings.angle_chunk_size),
            "completed_batches": self._batch_count,
            "completed_csi": self._csi_count,
            "largest_batch": self._largest_batch,
            "covariance_count": self._covariance_count,
            "eigendecomposition_count": self._eigendecomposition_count,
            "steering_cache_entries": len(self._cache),
            "steering_cache_bytes": self._cache_bytes,
            "steering_cache_hits": self._cache_hits,
            "steering_cache_misses": self._cache_misses,
            "steering_cache_max_entries": self._CACHE_MAX_ENTRIES,
            "steering_cache_max_bytes": self._CACHE_MAX_BYTES,
        }

    def prepare(
        self, csi: np.ndarray, *,
        subcarrier_frequencies_hz: Sequence[float] | np.ndarray,
        carrier_frequency_hz: float,
        antenna_spacing_m: float | None,
        num_sources: int = 1,
        spatial_subarray_size: int | None = None,
        frequency_subarray_size: int | None = None,
        diagonal_loading: float = 0.0,
    ) -> "PreparedMusic":
        """仅对这一份观测构造一次协方差并分解，随后复用以查询任意谱坐标。"""
        frequencies = _validate_frequency_vector(
            subcarrier_frequencies_hz, require_uniform_spacing=True
        ).copy()
        observation = _prepare_music_input(csi, frequencies.size)
        spatial_size = _validate_subarray_size(
            "spatial_subarray_size", spatial_subarray_size, observation.shape[-2]
        )
        frequency_size = _validate_subarray_size(
            "frequency_subarray_size", frequency_subarray_size, frequencies.size
        )
        dimension = spatial_size * frequency_size
        if isinstance(num_sources, bool) or int(num_sources) != num_sources:
            raise ValueError("num_sources 必须是整数")
        source_count = int(num_sources)
        if source_count < 1 or source_count >= dimension:
            raise ValueError(f"num_sources 必须位于 [1, {dimension - 1}]")
        loading = float(diagonal_loading)
        if not np.isfinite(loading) or loading < 0.0:
            raise ValueError("diagonal_loading 必须是有限非负数")
        # 在进行特征分解之前验证物理阵列参数。
        ula_steering_vector(
            [0.0], num_bs_antennas=spatial_size,
            carrier_frequency_hz=carrier_frequency_hz,
            antenna_spacing_m=antenna_spacing_m,
        )
        with self._device if self._device is not None else nullcontext():
            signal_adjoint = self._prepare_subspaces(
                self._xp.asarray(observation[None]), spatial_size,
                frequency_size, source_count, loading,
            )[0]
        self._batch_count += 1
        self._csi_count += 1
        self._largest_batch = max(self._largest_batch, 1)
        return PreparedMusic(
            self, signal_adjoint, frequencies, float(carrier_frequency_hz),
            antenna_spacing_m, spatial_size, frequency_size,
        )

    def spectrum(self, csi: np.ndarray, **kwargs: Any) -> np.ndarray:
        """计算一份 ``(M,K)`` 或 ``(S,M,K)`` 观测的二维谱。"""

        array = np.asarray(csi, dtype=np.complex128)
        if array.ndim == 2:
            array = array[np.newaxis, ...]
        if array.ndim != 3:
            raise ValueError("csi 必须为 (M, K) 或 (S, M, K)")
        return self.spectra(array[np.newaxis, ...], **kwargs)[0]

    def spectra(
        self,
        csi_batch: np.ndarray,
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
        """计算 ``[B,S,M,K]`` 中每份观测的谱，返回 ``[B,A,D]``。

        ``B`` 是独立观测数，``S`` 是单份观测的快照数。按设置的批量大小
        切分 ``B``，协方差与特征分解都保留独立的批次维度。角度再分块，
        避免创建完整 ``子阵列维数 × 全部角度 × 全部时延`` 的导向矩阵。
        """

        frequencies = _validate_frequency_vector(
            subcarrier_frequencies_hz, require_uniform_spacing=True
        )
        observations = np.asarray(csi_batch, dtype=np.complex128)
        if observations.ndim != 4:
            raise ValueError("csi_batch 必须为 (B, S, M, K)")
        if observations.shape[0] < 1 or observations.shape[1] < 1:
            raise ValueError("csi_batch 的批次数与快照数必须大于零")
        # 使用原 MUSIC 的 CSI 形状和有限值校验，不改变数据约定。
        _prepare_music_input(
            observations.reshape((-1,) + observations.shape[-2:]), frequencies.size
        )
        aoa_grid = _as_real_vector("aoa_grid_rad", aoa_grid_rad)
        delay_grid = _as_real_vector("delay_grid_s", delay_grid_s)
        if np.any(delay_grid < 0.0):
            raise ValueError("delay_grid_s 必须为非负时延")
        if np.any(np.diff(delay_grid) <= 0.0):
            raise ValueError("delay_grid_s 必须严格递增")
        if np.any(np.diff(aoa_grid) <= 0.0):
            raise ValueError("aoa_grid_rad 必须严格递增")
        spatial_size = _validate_subarray_size(
            "spatial_subarray_size", spatial_subarray_size, observations.shape[-2]
        )
        frequency_size = _validate_subarray_size(
            "frequency_subarray_size", frequency_subarray_size, frequencies.size
        )
        dimension = spatial_size * frequency_size
        if isinstance(num_sources, bool) or int(num_sources) != num_sources:
            raise ValueError("num_sources 必须是整数")
        source_count = int(num_sources)
        if source_count < 1 or source_count >= dimension:
            raise ValueError(f"num_sources 必须位于 [1, {dimension - 1}]")
        loading = float(diagonal_loading)
        if not np.isfinite(loading) or loading < 0.0:
            raise ValueError("diagonal_loading 必须是有限非负数")

        result = np.empty(
            (observations.shape[0], aoa_grid.size, delay_grid.size), dtype=np.float64
        )
        with self._device if self._device is not None else nullcontext():
            spatial, frequency = self._steering(
                frequencies, aoa_grid, delay_grid, spatial_size, frequency_size,
                carrier_frequency_hz, antenna_spacing_m,
            )
            for start in range(0, observations.shape[0], self.settings.batch_size):
                stop = min(start + self.settings.batch_size, observations.shape[0])
                device_result = self._compute_batch(
                    self._xp.asarray(observations[start:stop]), spatial, frequency,
                    source_count, loading,
                )
                result[start:stop] = (
                    self._xp.asnumpy(device_result)
                    if self.settings.backend == "cuda" else device_result
                )
                self._batch_count += 1
                self._csi_count += stop - start
                self._largest_batch = max(self._largest_batch, stop - start)
        return result

    def _steering(
        self,
        frequencies: np.ndarray,
        aoa_grid: np.ndarray,
        delay_grid: np.ndarray,
        spatial_size: int,
        frequency_size: int,
        carrier_frequency: float,
        spacing: float | None,
    ) -> tuple[Any, Any]:
        key = (
            self.settings.backend, self.settings.device_id, spatial_size,
            frequency_size, float(carrier_frequency),
            None if spacing is None else float(spacing),
            frequencies.tobytes(), aoa_grid.tobytes(), delay_grid.tobytes(),
        )
        if key in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._cache_misses += 1
        # 与参考方法共用 ULA 校验及相位定义。缓存的只是固定网格参数，
        # 不缓存观测协方差或含某个 sample 信号的信息。
        spatial_host = ula_steering_vector(
            aoa_grid, num_bs_antennas=spatial_size,
            carrier_frequency_hz=carrier_frequency, antenna_spacing_m=spacing,
        )
        relative_frequencies = frequencies[:frequency_size] - frequencies[0]
        frequency_host = np.exp(
            -2.0j * np.pi * relative_frequencies[:, None] * delay_grid[None, :]
        )
        pair = self._xp.asarray(spatial_host), self._xp.asarray(frequency_host)
        size = sum(int(array.nbytes) for array in pair)
        if size <= self._CACHE_MAX_BYTES:
            while self._cache and (
                len(self._cache) >= self._CACHE_MAX_ENTRIES
                or self._cache_bytes + size > self._CACHE_MAX_BYTES
            ):
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= sum(int(array.nbytes) for array in old)
            self._cache[key] = pair
            self._cache_bytes += size
        return pair

    def _compute_batch(
        self,
        observations: Any,
        spatial: Any,
        frequency: Any,
        source_count: int,
        loading: float,
    ) -> Any:
        signal_adjoint = self._prepare_subspaces(
            observations, spatial.shape[0], frequency.shape[0], source_count, loading,
        )
        return self._evaluate_grid(signal_adjoint, spatial, frequency)

    def _prepare_subspaces(
        self, observations: Any, spatial_size: int, frequency_size: int,
        source_count: int, loading: float,
    ) -> Any:
        xp = self._xp
        batch_count, snapshot_count, antenna_count, subcarrier_count = observations.shape
        dimension = spatial_size * frequency_size
        covariance = xp.zeros((batch_count, dimension, dimension), dtype=xp.complex128)
        window_count = (
            (antenna_count - spatial_size + 1)
            * (subcarrier_count - frequency_size + 1)
        )
        # 逐快照累计，控制滑窗内存；批次维度一次交给矩阵乘法。
        # 展平顺序保持 (空间阵元, 子载波)，与 _smoothed_covariance 相同。
        for snapshot in range(snapshot_count):
            windows = xp.lib.stride_tricks.sliding_window_view(
                observations[:, snapshot], (spatial_size, frequency_size),
                axis=(-2, -1),
            )
            vectors = windows.reshape(batch_count, window_count, dimension)
            covariance += xp.matmul(vectors.swapaxes(-1, -2), vectors.conj())
        covariance /= snapshot_count * window_count
        covariance = (covariance + covariance.conj().swapaxes(-1, -2)) / 2.0
        if loading > 0.0:
            mean_power = xp.trace(covariance, axis1=-2, axis2=-1).real / dimension
            covariance += loading * mean_power[:, None, None] * xp.eye(dimension)
        # CuPy 默认忽略 cuSOLVER 的失败状态；实验中必须显式报告不收敛。
        with self._cupyx.errstate(linalg="raise") if self._cupyx else nullcontext():
            _, eigenvectors = xp.linalg.eigh(covariance)
        self._covariance_count += batch_count
        self._eigendecomposition_count += batch_count
        return eigenvectors[:, :, -source_count:].conj().swapaxes(-1, -2)

    def _evaluate_grid(self, signal_adjoint: Any, spatial: Any, frequency: Any) -> Any:
        xp = self._xp
        batch_count = signal_adjoint.shape[0]
        spatial_size, angle_count = spatial.shape
        frequency_size, delay_count = frequency.shape
        dimension = spatial_size * frequency_size
        spectrum = xp.empty((batch_count, angle_count, delay_count), dtype=xp.float64)
        for start in range(0, angle_count, self.settings.angle_chunk_size):
            stop = min(start + self.settings.angle_chunk_size, angle_count)
            steering = (
                spatial[:, None, start:stop, None] * frequency[None, :, None, :]
            ).reshape(dimension, (stop - start) * delay_count)
            steering /= math.sqrt(dimension)
            projection = xp.matmul(signal_adjoint, steering)
            explained = xp.sum(xp.abs(projection) ** 2, axis=1)
            denominator = xp.maximum(1.0 - explained.real, np.finfo(np.float64).eps)
            spectrum[:, start:stop] = (1.0 / denominator).reshape(
                batch_count, stop - start, delay_count
            )
        return spectrum


class PreparedMusic:
    """单份 CSI 的已计算子空间；查询谱面和连续坐标均不再次分解 CSI。"""

    def __init__(
        self, computer: MusicComputer, signal_adjoint: Any, frequencies: np.ndarray,
        carrier_frequency: float, spacing: float | None, spatial_size: int,
        frequency_size: int,
    ) -> None:
        self._computer = computer
        self._signal_adjoint = signal_adjoint
        self._frequencies = frequencies
        self._carrier_frequency = carrier_frequency
        self._spacing = spacing
        self._spatial_size = spatial_size
        self._frequency_size = frequency_size
        self._grid_queries = 0
        self._paired_queries = 0
        self._evaluated_coordinates = 0

    @property
    def diagnostics(self) -> dict[str, Any]:
        return self.metadata()

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self._computer.settings.backend,
            "covariance_count": 1,
            "eigendecomposition_count": 1,
            "completed_csi": 1,
            "completed_batches": 1,
            "grid_queries": self._grid_queries,
            "paired_queries": self._paired_queries,
            "evaluated_coordinates": self._evaluated_coordinates,
            "observation_reused": True,
            "added_csi_noise": False,
        }

    def spectrum(
        self, *, aoa_grid_rad: Sequence[float] | np.ndarray,
        delay_grid_s: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """在任意合法细网格上计算同一个 MUSIC 函数，返回角度×时延数组。"""
        angles = _as_real_vector("aoa_grid_rad", aoa_grid_rad)
        delays = _as_real_vector("delay_grid_s", delay_grid_s)
        if np.any(np.diff(angles) <= 0):
            raise ValueError("aoa_grid_rad 必须严格递增")
        if np.any(delays < 0) or np.any(np.diff(delays) <= 0):
            raise ValueError("delay_grid_s 必须为非负且严格递增的时延")
        computer = self._computer
        with computer._device if computer._device is not None else nullcontext():
            spatial, frequency = computer._steering(
                self._frequencies, angles, delays, self._spatial_size,
                self._frequency_size, self._carrier_frequency, self._spacing,
            )
            result = computer._evaluate_grid(
                self._signal_adjoint[None], spatial, frequency,
            )[0]
            host = computer._xp.asnumpy(result) if computer._device is not None else result
        self._grid_queries += 1
        self._evaluated_coordinates += int(angles.size * delays.size)
        return host

    def values(
        self, *, aoa_rad: Sequence[float] | np.ndarray,
        delay_s: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """逐对精确评价连续角度、时延坐标，不进行谱插值或网格取整。"""
        angles = _as_real_vector("aoa_rad", aoa_rad)
        delays = _as_real_vector("delay_s", delay_s)
        if angles.shape != delays.shape:
            raise ValueError("aoa_rad 与 delay_s 必须为等长向量")
        if np.any(delays < 0):
            raise ValueError("delay_s 必须为非负时延")
        # 与全局 MUSIC 使用同一局部角度、天线方向和相位约定。
        spatial_host = ula_steering_vector(
            angles, num_bs_antennas=self._spatial_size,
            carrier_frequency_hz=self._carrier_frequency,
            antenna_spacing_m=self._spacing,
        )
        computer = self._computer
        xp = computer._xp
        dimension = self._spatial_size * self._frequency_size
        result = np.empty(angles.size, dtype=np.float64)
        chunk_size = computer.settings.angle_chunk_size * 256
        with computer._device if computer._device is not None else nullcontext():
            frequencies = xp.asarray(
                self._frequencies[:self._frequency_size] - self._frequencies[0]
            )
            for start in range(0, angles.size, chunk_size):
                stop = min(start + chunk_size, angles.size)
                spatial = xp.asarray(spatial_host[:, start:stop])
                frequency = xp.exp(
                    -2.0j * np.pi * frequencies[:, None]
                    * xp.asarray(delays[None, start:stop])
                )
                steering = (spatial[:, None, :] * frequency[None, :, :]).reshape(
                    dimension, stop - start
                ) / math.sqrt(dimension)
                projection = self._signal_adjoint @ steering
                denominator = xp.maximum(
                    1.0 - xp.sum(xp.abs(projection) ** 2, axis=0).real,
                    np.finfo(np.float64).eps,
                )
                values = 1.0 / denominator
                result[start:stop] = (
                    xp.asnumpy(values) if computer._device is not None else values
                )
        self._paired_queries += 1
        self._evaluated_coordinates += int(angles.size)
        return result
