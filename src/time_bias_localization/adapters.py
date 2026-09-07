"""DeepMIMO V4 与 Sionna RT 的轻量适配层。

这个模块故意不在导入时加载 DeepMIMO 或 Sionna。这样，定位算法及其单元测试
可以在没有安装两套大型仿真依赖的环境中运行。只有真正调用对应后端时，才会尝试
导入依赖并给出清楚的中文错误。

适配后的数组约定如下：

* ``rx_pos``、``tx_pos``: ``[样本数, 空间维数]``，空间维数只能是 2 或 3；
* ``delay``、``aoa_az``、``aod_az``: ``[样本数, 最大路径数]``；
* ``inter``: ``[样本数, 最大路径数, 最大交互次数]``；
* ``inter_pos``: ``[样本数, 最大路径数, 最大交互次数, 空间维数]``。

``delay`` 始终表示绝对传播时延。这里不会减去每个样本的首径时延。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from types import ModuleType
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


LinkDirection = Literal["downlink", "uplink"]
AngleUnit = Literal["rad", "deg"]
PathRepresentation = Literal["cir", "cfr"]


class OptionalDependencyError(ImportError):
    """需要的可选仿真依赖未安装。"""


class AdapterContractError(ValueError):
    """输入数据不符合适配层公开的数据约定。"""


_DEFAULT_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "rx_pos": (
        "rx_pos",
        "receiver_positions",
        "receiver.position",
        "rx.position",
        "ue_pos",
        "user.position",
    ),
    "tx_pos": (
        "tx_pos",
        "transmitter_positions",
        "transmitter.position",
        "tx.position",
        "bs_pos",
        "base_station.position",
    ),
    "delay": (
        "delay",
        "delays",
        "paths.delay",
        "paths.delays",
        "path_params.delay",
        "channel_params.delay",
    ),
    "aoa_az": (
        "aoa_az",
        "paths.aoa_az",
        "path_params.aoa_az",
        "aoa.azimuth",
        "arrival_azimuth",
        "phi_r",
    ),
    "aod_az": (
        "aod_az",
        "paths.aod_az",
        "path_params.aod_az",
        "aod.azimuth",
        "departure_azimuth",
        "phi_t",
    ),
    "inter": (
        "inter",
        "interaction",
        "interactions",
        "paths.inter",
        "paths.interactions",
        "path_params.inter",
    ),
    "inter_pos": (
        "inter_pos",
        "interaction_positions",
        "paths.inter_pos",
        "paths.interaction_positions",
        "path_params.inter_pos",
    ),
    "scene": (
        "scene",
        "scenario",
        "metadata.scene",
        "metadata.scenario",
        "scene_name",
    ),
}


@dataclass(frozen=True, slots=True)
class DeepMIMOAdapterConfig:
    """DeepMIMO 数据提取设置。

    ``field_aliases`` 可为某个标准字段补充项目自己的别名。补充别名会优先于
    内置别名。角度单位只被记录，不会被悄悄转换。
    """

    link_direction: LinkDirection = "downlink"
    angle_unit: AngleUnit = "rad"
    azimuth_convention: str = "DeepMIMO/Sionna source convention"
    preserve_absolute_delays: bool = True
    require_scene: bool = True
    field_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.link_direction not in ("downlink", "uplink"):
            raise AdapterContractError(
                "link_direction 只能是 'downlink' 或 'uplink'。"
            )
        if self.angle_unit not in ("rad", "deg"):
            raise AdapterContractError("angle_unit 只能是 'rad' 或 'deg'。")
        if not self.azimuth_convention.strip():
            raise AdapterContractError("azimuth_convention 不能为空。")
        if not self.preserve_absolute_delays:
            raise AdapterContractError(
                "本项目禁止把首径时延减为零；preserve_absolute_delays 必须为 True。"
            )
        unknown_fields = set(self.field_aliases) - set(_DEFAULT_FIELD_ALIASES)
        if unknown_fields:
            names = ", ".join(sorted(unknown_fields))
            raise AdapterContractError(f"field_aliases 包含未知标准字段：{names}。")


@dataclass(frozen=True, slots=True)
class NormalizedDeepMIMOData:
    """经过严格校验的单基站 DeepMIMO 数据。

    ``aoa_az`` 总是当前链路在接收端的到达方位角，``aod_az`` 总是当前链路
    从发射端离开的方位角。它们的具体端点角色由 ``link_direction``、
    ``transmitter_role`` 与 ``receiver_role`` 明确记录。
    """

    rx_pos: NDArray[np.floating[Any]]
    tx_pos: NDArray[np.floating[Any]]
    delay: NDArray[np.floating[Any]]
    aoa_az: NDArray[np.floating[Any]]
    aod_az: NDArray[np.floating[Any]]
    inter: NDArray[Any]
    inter_pos: NDArray[np.floating[Any]]
    scene: Any = field(repr=False)
    link_direction: LinkDirection = "downlink"
    transmitter_role: Literal["bs", "ue"] = "bs"
    receiver_role: Literal["bs", "ue"] = "ue"
    aoa_az_semantics: Literal["arrival_at_receiver"] = "arrival_at_receiver"
    aod_az_semantics: Literal["departure_from_transmitter"] = (
        "departure_from_transmitter"
    )
    angle_unit: AngleUnit = "rad"
    azimuth_convention: str = "DeepMIMO/Sionna source convention"
    delays_are_absolute: Literal[True] = True

    def __post_init__(self) -> None:
        _validate_normalized_data(self)

    @property
    def valid_path_mask(self) -> NDArray[np.bool_]:
        """由有限绝对时延定义的有效路径掩码。"""

        return np.isfinite(self.delay)


@dataclass(frozen=True, slots=True)
class SionnaPathExportConfig:
    """Sionna 路径导出设置；绝对时延保护不能关闭。"""

    representation: PathRepresentation = "cir"
    preserve_absolute_delays: bool = True

    def __post_init__(self) -> None:
        if self.representation not in ("cir", "cfr"):
            raise AdapterContractError("representation 只能是 'cir' 或 'cfr'。")
        if not self.preserve_absolute_delays:
            raise AdapterContractError(
                "Sionna 路径导出必须保留绝对时延，不能把首径移动到零时延。"
            )


def load_deepmimo_module() -> ModuleType:
    """按需导入 DeepMIMO V4。"""

    try:
        return importlib.import_module("deepmimo")
    except (ModuleNotFoundError, ImportError) as exc:
        raise OptionalDependencyError(
            "缺少 DeepMIMO V4（Python 包名 deepmimo）。"
            "只有生成或转换 DeepMIMO 数据时才需要安装它。"
        ) from exc


def load_sionna_rt_module() -> ModuleType:
    """按需导入 Sionna RT。"""

    try:
        return importlib.import_module("sionna.rt")
    except (ModuleNotFoundError, ImportError) as exc:
        raise OptionalDependencyError(
            "缺少 Sionna RT（Python 包路径 sionna.rt）。"
            "只有执行射线追踪时才需要安装它。"
        ) from exc


def extract_deepmimo_dataset(
    dataset: Any,
    config: DeepMIMOAdapterConfig | None = None,
) -> NormalizedDeepMIMOData:
    """从 DeepMIMO Dataset 或相同字段的伪对象中提取标准数据。

    第一版只接收一个基站对应的数据。若输入含额外的基站维度，应先在调用端
    选定基站；适配器不会猜测应该使用哪一个基站。
    """

    config = config or DeepMIMOAdapterConfig()
    extracted: dict[str, Any] = {}
    for name in ("rx_pos", "tx_pos", "delay", "aoa_az", "aod_az", "inter", "inter_pos"):
        extracted[name] = _extract_required_field(dataset, name, config)

    scene = _extract_optional_field(dataset, "scene", config)
    if scene is _MISSING or scene is None:
        if config.require_scene:
            raise AdapterContractError(
                "DeepMIMO Dataset 中找不到 scene/scenario；"
                "请提供场景对象或通过 field_aliases 声明实际字段。"
            )
        scene = None

    rx_pos = _numeric_array(extracted["rx_pos"], "rx_pos", dtype=float)
    tx_pos = _numeric_array(extracted["tx_pos"], "tx_pos", dtype=float)
    delay = _numeric_array(extracted["delay"], "delay", dtype=float)
    aoa_az = _numeric_array(extracted["aoa_az"], "aoa_az", dtype=float)
    aod_az = _numeric_array(extracted["aod_az"], "aod_az", dtype=float)
    inter = _array(extracted["inter"], "inter")
    inter_pos = _numeric_array(extracted["inter_pos"], "inter_pos", dtype=float)

    if rx_pos.ndim != 2 or rx_pos.shape[1] not in (2, 3):
        raise AdapterContractError(
            f"rx_pos 必须是 [样本数, 2或3]，实际形状为 {rx_pos.shape}。"
        )
    sample_count, spatial_dimension = rx_pos.shape

    if tx_pos.ndim == 1:
        if tx_pos.shape != (spatial_dimension,):
            raise AdapterContractError(
                "一维 tx_pos 的长度必须等于 rx_pos 的空间维数，"
                f"实际形状为 {tx_pos.shape}。"
            )
        tx_pos = np.broadcast_to(tx_pos, (sample_count, spatial_dimension)).copy()
    elif tx_pos.ndim == 2 and tx_pos.shape == (1, spatial_dimension):
        tx_pos = np.broadcast_to(tx_pos, (sample_count, spatial_dimension)).copy()
    elif tx_pos.ndim == 2 and tx_pos.shape == (sample_count, spatial_dimension):
        tx_pos = tx_pos.copy()
    else:
        raise AdapterContractError(
            "tx_pos 必须是 [空间维数]、[1, 空间维数] 或 "
            f"[样本数, 空间维数]，实际形状为 {tx_pos.shape}。"
        )

    expected_path_shape = delay.shape
    if delay.ndim != 2 or delay.shape[0] != sample_count:
        raise AdapterContractError(
            "delay 必须是 [样本数, 最大路径数]，"
            f"实际形状为 {delay.shape}，样本数为 {sample_count}。"
        )
    for name, value in (("aoa_az", aoa_az), ("aod_az", aod_az)):
        if value.shape != expected_path_shape:
            raise AdapterContractError(
                f"{name} 必须与 delay 形状一致；"
                f"delay={expected_path_shape}，{name}={value.shape}。"
            )

    if inter.ndim == 2:
        if inter.shape != expected_path_shape:
            raise AdapterContractError(
                "二维 inter 必须与 delay 形状一致；"
                f"delay={expected_path_shape}，inter={inter.shape}。"
            )
        inter = inter[..., np.newaxis]
    elif inter.ndim != 3 or inter.shape[:2] != expected_path_shape:
        raise AdapterContractError(
            "inter 必须是 [样本数, 最大路径数] 或 "
            "[样本数, 最大路径数, 最大交互次数]，"
            f"实际形状为 {inter.shape}。"
        )

    if inter_pos.ndim == 3:
        if inter_pos.shape[:2] != expected_path_shape:
            raise AdapterContractError(
                "三维 inter_pos 的前两维必须与 delay 一致，"
                f"实际形状为 {inter_pos.shape}。"
            )
        inter_pos = inter_pos[:, :, np.newaxis, :]
    elif inter_pos.ndim != 4 or inter_pos.shape[:2] != expected_path_shape:
        raise AdapterContractError(
            "inter_pos 必须是 [样本数, 最大路径数, 空间维数] 或 "
            "[样本数, 最大路径数, 最大交互次数, 空间维数]，"
            f"实际形状为 {inter_pos.shape}。"
        )
    if inter_pos.shape[2] != inter.shape[2]:
        raise AdapterContractError(
            "inter 与 inter_pos 的最大交互次数不一致："
            f"{inter.shape[2]} != {inter_pos.shape[2]}。"
        )
    if inter_pos.shape[3] != spatial_dimension:
        raise AdapterContractError(
            "inter_pos 的空间维数必须与 rx_pos/tx_pos 一致："
            f"{inter_pos.shape[3]} != {spatial_dimension}。"
        )

    if config.link_direction == "downlink":
        transmitter_role: Literal["bs", "ue"] = "bs"
        receiver_role: Literal["bs", "ue"] = "ue"
    else:
        transmitter_role = "ue"
        receiver_role = "bs"

    return NormalizedDeepMIMOData(
        rx_pos=rx_pos.copy(),
        tx_pos=tx_pos,
        delay=delay.copy(),
        aoa_az=aoa_az.copy(),
        aod_az=aod_az.copy(),
        inter=inter.copy(),
        inter_pos=inter_pos.copy(),
        scene=scene,
        link_direction=config.link_direction,
        transmitter_role=transmitter_role,
        receiver_role=receiver_role,
        angle_unit=config.angle_unit,
        azimuth_convention=config.azimuth_convention,
    )


def absolute_delays(data: NormalizedDeepMIMOData) -> NDArray[np.floating[Any]]:
    """返回绝对传播时延副本，不做首径归零。"""

    if not data.delays_are_absolute:
        raise AdapterContractError("输入没有声明为绝对传播时延。")
    return np.array(data.delay, dtype=float, copy=True)


def absolute_delays_from_path_lengths(
    path_lengths_m: ArrayLike,
    propagation_speed_m_s: float = 299_792_458.0,
) -> NDArray[np.floating[Any]]:
    """由从发射端到接收端的完整路径长度计算绝对传播时延。"""

    lengths = _numeric_array(path_lengths_m, "path_lengths_m", dtype=float)
    if lengths.ndim != 2:
        raise AdapterContractError(
            "path_lengths_m 必须是 [样本数, 最大路径数]。"
        )
    if not np.isfinite(propagation_speed_m_s) or propagation_speed_m_s <= 0:
        raise AdapterContractError("propagation_speed_m_s 必须是有限正数。")
    finite = np.isfinite(lengths)
    if np.any(lengths[finite] < 0) or np.any(np.isinf(lengths)):
        raise AdapterContractError("路径长度不能为负数或无穷大。")
    return lengths / float(propagation_speed_m_s)


def frequency_response_from_paths(
    path_gains: ArrayLike,
    absolute_delay_s: ArrayLike,
    frequencies_hz: ArrayLike,
) -> NDArray[np.complexfloating[Any, Any]]:
    """使用绝对时延合成频域 CSI。

    ``path_gains`` 形状为 ``[样本数, ..., 最大路径数]``，路径轴必须在最后；
    ``absolute_delay_s`` 为 ``[样本数, 最大路径数]``。返回数组会用子载波轴
    替换路径轴。任何 NaN 时延都被视为填充路径，其增益不会进入求和。
    """

    gains = _numeric_array(path_gains, "path_gains", dtype=complex)
    delays = _numeric_array(absolute_delay_s, "absolute_delay_s", dtype=float)
    frequencies = _numeric_array(frequencies_hz, "frequencies_hz", dtype=float)
    if gains.ndim < 2:
        raise AdapterContractError(
            "path_gains 至少需要 [样本数, 最大路径数] 两个维度。"
        )
    if delays.ndim != 2:
        raise AdapterContractError(
            "absolute_delay_s 必须是 [样本数, 最大路径数]。"
        )
    if gains.shape[0] != delays.shape[0] or gains.shape[-1] != delays.shape[1]:
        raise AdapterContractError(
            "path_gains 的样本轴和路径轴必须与 absolute_delay_s 一致；"
            f"path_gains={gains.shape}，absolute_delay_s={delays.shape}。"
        )
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
        raise AdapterContractError("frequencies_hz 必须是一维、非空且全部有限。")

    valid = np.isfinite(delays)
    if np.any(delays[valid] < 0) or np.any(np.isinf(delays)):
        raise AdapterContractError("绝对传播时延不能为负数或无穷大。")
    finite_gain = np.isfinite(gains.real) & np.isfinite(gains.imag)
    expanded_valid = valid.reshape(
        (delays.shape[0],) + (1,) * (gains.ndim - 2) + (delays.shape[1],)
    )
    if np.any(~finite_gain & expanded_valid):
        raise AdapterContractError("有效路径的 path_gains 必须是有限复数。")

    safe_delays = np.where(valid, delays, 0.0)
    phase = np.exp(-2j * np.pi * safe_delays[..., np.newaxis] * frequencies)
    phase = phase.reshape(
        (delays.shape[0],)
        + (1,) * (gains.ndim - 2)
        + (delays.shape[1], frequencies.size)
    )
    safe_gains = np.where(expanded_valid, gains, 0.0)
    return np.sum(safe_gains[..., np.newaxis] * phase, axis=-2)


def sionna_paths_cir(paths: Any, **kwargs: Any) -> Any:
    """调用 ``paths.cir``，并强制保留绝对传播时延。"""

    call_kwargs = _absolute_delay_call_kwargs(kwargs)
    cir = getattr(paths, "cir", None)
    if not callable(cir):
        raise AdapterContractError("给定对象没有可调用的 paths.cir(...)。")
    return cir(**call_kwargs)


def sionna_paths_cfr(
    paths: Any,
    frequencies_hz: ArrayLike,
    **kwargs: Any,
) -> Any:
    """调用 ``paths.cfr``，并强制保留绝对传播时延。"""

    frequencies = _numeric_array(frequencies_hz, "frequencies_hz", dtype=float)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
        raise AdapterContractError("frequencies_hz 必须是一维、非空且全部有限。")
    if "frequencies" in kwargs:
        raise AdapterContractError(
            "请只通过 frequencies_hz 传入频率，不要在 kwargs 中重复传 frequencies。"
        )
    call_kwargs = _absolute_delay_call_kwargs(kwargs)
    cfr = getattr(paths, "cfr", None)
    if not callable(cfr):
        raise AdapterContractError("给定对象没有可调用的 paths.cfr(...)。")
    return cfr(frequencies=frequencies, **call_kwargs)


def export_sionna_paths(
    paths: Any,
    config: SionnaPathExportConfig | None = None,
    *,
    frequencies_hz: ArrayLike | None = None,
    **kwargs: Any,
) -> Any:
    """按配置导出 Sionna CIR 或 CFR。

    这是隔离 Sionna 版本变化的包装入口。无论调用者传入什么设置，接口都不允许
    ``normalize_delays=True``。
    """

    config = config or SionnaPathExportConfig()
    if config.representation == "cir":
        if frequencies_hz is not None:
            raise AdapterContractError("导出 CIR 时不应传入 frequencies_hz。")
        return sionna_paths_cir(paths, **kwargs)
    if frequencies_hz is None:
        raise AdapterContractError("导出 CFR 时必须提供 frequencies_hz。")
    return sionna_paths_cfr(paths, frequencies_hz, **kwargs)


def convert_sionna_paths_to_deepmimo(
    paths: Any,
    *,
    scene: Any = None,
    converter: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Any:
    """将 Sionna 路径交给 DeepMIMO V4 官方转换器。

    不同 DeepMIMO V4 小版本可能移动转换函数。生产代码可把该版本的官方函数
    通过 ``converter`` 显式传入；若未传入，本函数会在几个公开命名位置中查找。
    这个包装不会修改路径时延。
    """

    if converter is None:
        deepmimo = load_deepmimo_module()
        converter = _find_converter(
            deepmimo,
            (
                "from_sionna",
                "convert_sionna_paths",
                "rt.from_sionna",
                "rt.convert_sionna_paths",
            ),
        )
        if converter is None:
            raise AdapterContractError(
                "当前 DeepMIMO V4 包中没有找到已知的 Sionna 转换入口。"
                "请把该版本的官方转换函数通过 converter=... 显式传入。"
            )
    if not callable(converter):
        raise AdapterContractError("converter 必须是可调用对象。")
    call_kwargs = dict(kwargs)
    if scene is not None:
        call_kwargs["scene"] = scene
    return converter(paths, **call_kwargs)


def downlink_to_uplink_csi(
    downlink_csi: ArrayLike,
    *,
    receive_axis: int = -2,
    transmit_axis: int = -1,
) -> NDArray[np.complexfloating[Any, Any]]:
    """利用互易性把下行信道矩阵转换为上行信道矩阵。

    操作为复共轭并交换接收、发射天线轴，即 ``H_ul = H_dl^H``。频率轴、
    样本轴等其他轴保持不变。这个函数只转换 CSI；路径 AOA/AOD 的端点语义
    仍需调用端按链路方向处理。
    """

    csi = _numeric_array(downlink_csi, "downlink_csi", dtype=complex)
    if csi.ndim < 2:
        raise AdapterContractError("downlink_csi 至少需要两个天线维度。")
    rx_axis = _normalize_axis(receive_axis, csi.ndim, "receive_axis")
    tx_axis = _normalize_axis(transmit_axis, csi.ndim, "transmit_axis")
    if rx_axis == tx_axis:
        raise AdapterContractError("receive_axis 与 transmit_axis 不能相同。")
    return np.swapaxes(np.conjugate(csi), rx_axis, tx_axis)


_MISSING = object()


def _field_aliases(name: str, config: DeepMIMOAdapterConfig) -> tuple[str, ...]:
    custom = tuple(config.field_aliases.get(name, ()))
    defaults = _DEFAULT_FIELD_ALIASES[name]
    return tuple(dict.fromkeys((*custom, *defaults)))


def _extract_required_field(
    dataset: Any,
    name: str,
    config: DeepMIMOAdapterConfig,
) -> Any:
    value = _extract_optional_field(dataset, name, config)
    if value is _MISSING or value is None:
        aliases = ", ".join(_field_aliases(name, config))
        raise AdapterContractError(
            f"DeepMIMO Dataset 缺少必需字段 {name}；已尝试别名：{aliases}。"
        )
    return value


def _extract_optional_field(
    dataset: Any,
    name: str,
    config: DeepMIMOAdapterConfig,
) -> Any:
    for alias in _field_aliases(name, config):
        value = _resolve_alias(dataset, alias)
        if value is not _MISSING:
            return value
    return _MISSING


def _resolve_alias(root: Any, alias: str) -> Any:
    direct = _get_member(root, alias)
    if direct is not _MISSING:
        return direct
    current = root
    for part in alias.split("."):
        current = _get_member(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def _get_member(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping) and name in obj:
        return obj[name]
    try:
        return getattr(obj, name)
    except (AttributeError, TypeError):
        pass
    try:
        return obj[name]
    except (KeyError, IndexError, TypeError, AttributeError):
        return _MISSING


def _array(value: Any, name: str) -> NDArray[Any]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError(f"{name} 无法转换为规则数组。") from exc
    if array.dtype == object and array.ndim <= 1:
        raise AdapterContractError(f"{name} 可能是参差不齐的对象数组，无法严格校验。")
    return array


def _numeric_array(
    value: Any,
    name: str,
    *,
    dtype: type[float] | type[complex],
) -> NDArray[Any]:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError(f"{name} 必须是数值数组。") from exc
    if array.dtype == object:
        raise AdapterContractError(f"{name} 不能是对象数组。")
    return array


def _validate_normalized_data(data: NormalizedDeepMIMOData) -> None:
    if data.link_direction not in ("downlink", "uplink"):
        raise AdapterContractError("标准数据的 link_direction 非法。")
    expected_roles = ("bs", "ue") if data.link_direction == "downlink" else ("ue", "bs")
    if (data.transmitter_role, data.receiver_role) != expected_roles:
        raise AdapterContractError(
            "transmitter_role/receiver_role 与 link_direction 不一致。"
        )
    if data.aoa_az_semantics != "arrival_at_receiver":
        raise AdapterContractError("aoa_az 必须表示接收端到达方位角。")
    if data.aod_az_semantics != "departure_from_transmitter":
        raise AdapterContractError("aod_az 必须表示发射端离开方位角。")
    if data.angle_unit not in ("rad", "deg"):
        raise AdapterContractError("标准数据的 angle_unit 非法。")
    if not data.delays_are_absolute:
        raise AdapterContractError("标准数据必须明确声明 delay 是绝对传播时延。")
    if data.rx_pos.ndim != 2 or data.rx_pos.shape[1] not in (2, 3):
        raise AdapterContractError("标准数据 rx_pos 形状非法。")
    if data.tx_pos.shape != data.rx_pos.shape:
        raise AdapterContractError("标准数据 tx_pos 必须与 rx_pos 形状一致。")
    if data.delay.ndim != 2 or data.delay.shape[0] != data.rx_pos.shape[0]:
        raise AdapterContractError("标准数据 delay 形状非法。")
    if data.aoa_az.shape != data.delay.shape or data.aod_az.shape != data.delay.shape:
        raise AdapterContractError("标准数据角度数组必须与 delay 形状一致。")
    if data.inter.ndim != 3 or data.inter.shape[:2] != data.delay.shape:
        raise AdapterContractError("标准数据 inter 形状非法。")
    if (
        data.inter_pos.ndim != 4
        or data.inter_pos.shape[:3] != data.inter.shape
        or data.inter_pos.shape[3] != data.rx_pos.shape[1]
    ):
        raise AdapterContractError("标准数据 inter_pos 形状非法。")
    if not np.all(np.isfinite(data.rx_pos)) or not np.all(np.isfinite(data.tx_pos)):
        raise AdapterContractError("rx_pos 和 tx_pos 必须全部是有限坐标。")
    if np.any(np.isinf(data.delay)):
        raise AdapterContractError("delay 不能包含正负无穷大。")
    valid = np.isfinite(data.delay)
    if np.any(data.delay[valid] < 0):
        raise AdapterContractError("绝对传播时延不能为负数。")
    if np.any(~np.isfinite(data.aoa_az[valid])) or np.any(~np.isfinite(data.aod_az[valid])):
        raise AdapterContractError("有效路径的 AOA/AOD 必须是有限数值。")
    if np.any(np.isinf(data.inter_pos)):
        raise AdapterContractError("inter_pos 不能包含正负无穷大。")


def _absolute_delay_call_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    call_kwargs = dict(kwargs)
    if bool(call_kwargs.get("normalize_delays", False)):
        raise AdapterContractError(
            "禁止 normalize_delays=True：这会把首径移动到零时延，破坏绝对 TOA。"
        )
    call_kwargs["normalize_delays"] = False
    return call_kwargs


def _find_converter(module: Any, names: Sequence[str]) -> Callable[..., Any] | None:
    for name in names:
        current = module
        for part in name.split("."):
            current = getattr(current, part, None)
            if current is None:
                break
        if callable(current):
            return current
    return None


def _normalize_axis(axis: int, ndim: int, name: str) -> int:
    if not isinstance(axis, int):
        raise AdapterContractError(f"{name} 必须是整数。")
    if axis < -ndim or axis >= ndim:
        raise AdapterContractError(f"{name}={axis} 超出 {ndim} 维数组范围。")
    return axis % ndim


__all__ = [
    "AdapterContractError",
    "DeepMIMOAdapterConfig",
    "NormalizedDeepMIMOData",
    "OptionalDependencyError",
    "SionnaPathExportConfig",
    "absolute_delays",
    "absolute_delays_from_path_lengths",
    "convert_sionna_paths_to_deepmimo",
    "downlink_to_uplink_csi",
    "export_sionna_paths",
    "extract_deepmimo_dataset",
    "frequency_response_from_paths",
    "load_deepmimo_module",
    "load_sionna_rt_module",
    "sionna_paths_cfr",
    "sionna_paths_cir",
]
