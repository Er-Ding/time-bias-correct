"""三维场景到二维墙线与俯视图的预处理。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw


_GEOMETRY_EPS = 1e-9


def _stage_scene_file(path: Path, writer: Any) -> Path:
    """在正式目标同目录写好并同步临时文件；失败文件保留用于排查。"""

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f".tmp{path.suffix}",
    )
    temporary_path = Path(temporary_name)
    with os.fdopen(file_descriptor, "w+b") as handle:
        writer(handle)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary_path


def _scene_backup_path(path: Path) -> Path:
    """返回同目录、尚不存在的唯一场景备份名。"""

    while True:
        candidate = path.with_name(
            f".{path.name}.{uuid4().hex}.backup{path.suffix}"
        )
        if not (candidate.exists() or candidate.is_symlink()):
            return candidate


def _publish_scene_files(
    staged_files: Sequence[tuple[Path, Path]], *, allow_overwrite: bool
) -> None:
    """发布完整场景；中途失败时恢复发布前的整组正式目标。"""

    conflicts = [
        target
        for target, _ in staged_files
        if target.exists() or target.is_symlink()
    ]
    if conflicts and not allow_overwrite:
        joined = "、".join(str(path) for path in conflicts)
        raise FileExistsError(f"场景目标在准备期间被创建，拒绝覆盖：{joined}")

    backups: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        if allow_overwrite:
            for target, _ in staged_files:
                if not (target.exists() or target.is_symlink()):
                    continue
                backup = _scene_backup_path(target)
                os.replace(target, backup)
                backups.append((target, backup))
        for target, temporary in staged_files:
            os.replace(temporary, target)
            published.append((target, temporary))
        directory_descriptor = os.open(staged_files[0][0].parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # 先移回本轮新文件，再恢复旧文件；所有失败残留都保留下来。
        for target, temporary in reversed(published):
            if (target.exists() or target.is_symlink()) and not (
                temporary.exists() or temporary.is_symlink()
            ):
                try:
                    os.replace(target, temporary)
                except OSError:
                    pass
        for target, backup in reversed(backups):
            if (backup.exists() or backup.is_symlink()) and not (
                target.exists() or target.is_symlink()
            ):
                try:
                    os.replace(backup, target)
                except OSError:
                    pass
        raise
    else:
        for _, backup in backups:
            try:
                backup.unlink()
            except OSError:
                pass


@dataclass(frozen=True)
class WallSegment:
    """二维镜面墙段，坐标单位为米。"""

    wall_id: str
    start_m: tuple[float, float]
    end_m: tuple[float, float]
    source_object: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.wall_id, str) or not self.wall_id.strip():
            raise ValueError("墙段 wall_id 必须是非空字符串")
        if not isinstance(self.source_object, str):
            raise ValueError(f"墙段 {self.wall_id} 的 source_object 必须是字符串")
        try:
            start = np.asarray(self.start_m, dtype=float)
            end = np.asarray(self.end_m, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"墙段 {self.wall_id} 的端点必须是二维有限坐标") from error
        if start.shape != (2,) or end.shape != (2,):
            raise ValueError(f"墙段 {self.wall_id} 的端点必须严格为形状 (2,)")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError(f"墙段 {self.wall_id} 的端点必须全部为有限数")
        object.__setattr__(self, "start_m", (float(start[0]), float(start[1])))
        object.__setattr__(self, "end_m", (float(end[0]), float(end[1])))
        if not math.isfinite(self.length_m) or self.length_m <= _GEOMETRY_EPS:
            raise ValueError(f"墙段 {self.wall_id} 的长度必须大于零")

    @property
    def start(self) -> np.ndarray:
        return np.asarray(self.start_m, dtype=float)

    @property
    def end(self) -> np.ndarray:
        return np.asarray(self.end_m, dtype=float)

    @property
    def vector(self) -> np.ndarray:
        return self.end - self.start

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(np.asarray(self.end_m) - np.asarray(self.start_m)))

    @property
    def tangent(self) -> np.ndarray:
        return self.vector / self.length_m

    @property
    def normal(self) -> np.ndarray:
        tangent = self.tangent
        return np.asarray([-tangent[1], tangent[0]], dtype=float)


@dataclass(frozen=True)
class Scene2D:
    """定位模块使用的二维场景。"""

    name: str
    bounds_m: tuple[float, float, float, float]
    walls: tuple[WallSegment, ...]
    fixed_height_m: float
    bev_resolution_m: float
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("二维场景 name 必须是非空字符串")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("二维场景 source 必须是非空字符串")
        try:
            bounds = np.asarray(self.bounds_m, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("场景边界必须是四个有限数") from error
        if bounds.shape != (4,) or not np.all(np.isfinite(bounds)):
            raise ValueError("场景边界必须严格为四个有限数")
        object.__setattr__(self, "bounds_m", tuple(float(value) for value in bounds))
        x_min, x_max, y_min, y_max = self.bounds_m
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("场景边界必须满足 x_min<x_max 且 y_min<y_max")
        if not math.isfinite(float(self.fixed_height_m)):
            raise ValueError("固定高度必须是有限数")
        if not math.isfinite(float(self.bev_resolution_m)) or self.bev_resolution_m <= 0:
            raise ValueError("俯视图分辨率必须为正数")
        if not isinstance(self.walls, tuple) or not all(
            isinstance(wall, WallSegment) for wall in self.walls
        ):
            raise ValueError("二维场景 walls 必须是 WallSegment 元组")

    @property
    def width_px(self) -> int:
        return max(2, int(math.ceil((self.bounds_m[1] - self.bounds_m[0]) / self.bev_resolution_m)) + 1)

    @property
    def height_px(self) -> int:
        return max(2, int(math.ceil((self.bounds_m[3] - self.bounds_m[2]) / self.bev_resolution_m)) + 1)

    def contains(self, point_m: Sequence[float], margin_m: float = 0.0) -> bool:
        x, y = np.asarray(point_m, dtype=float)
        x_min, x_max, y_min, y_max = self.bounds_m
        return bool(
            x_min + margin_m <= x <= x_max - margin_m
            and y_min + margin_m <= y <= y_max - margin_m
        )

    def metric_to_pixel(self, point_m: Sequence[float]) -> tuple[float, float]:
        """米坐标转图像坐标；图像纵轴向下。"""

        x, y = np.asarray(point_m, dtype=float)
        x_min, _, _, y_max = self.bounds_m
        col = (x - x_min) / self.bev_resolution_m
        row = (y_max - y) / self.bev_resolution_m
        return float(col), float(row)

    def pixel_to_metric(self, pixel: Sequence[float]) -> tuple[float, float]:
        col, row = np.asarray(pixel, dtype=float)
        x_min, _, _, y_max = self.bounds_m
        x = x_min + col * self.bev_resolution_m
        y = y_max - row * self.bev_resolution_m
        return float(x), float(y)

    def rasterize(self, wall_width_px: int = 2) -> np.ndarray:
        """返回 uint8 占据图：255 是自由空间，0 是墙。"""

        image = Image.new("L", (self.width_px, self.height_px), color=255)
        draw = ImageDraw.Draw(image)
        for wall in self.walls:
            draw.line(
                [self.metric_to_pixel(wall.start_m), self.metric_to_pixel(wall.end_m)],
                fill=0,
                width=max(1, int(wall_width_px)),
            )
        return np.asarray(image, dtype=np.uint8)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bounds_m": list(self.bounds_m),
            "fixed_height_m": self.fixed_height_m,
            "bev_resolution_m": self.bev_resolution_m,
            "source": self.source,
            "image_convention": {
                "origin": "upper_left",
                "column_axis": "+x",
                "row_axis": "-y",
                "pixel_center_units": "pixel",
            },
            "walls": [asdict(wall) for wall in self.walls],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene2D":
        if not isinstance(data, Mapping):
            raise ValueError("二维场景顶层必须是键值映射")
        expected_fields = {
            "name",
            "bounds_m",
            "fixed_height_m",
            "bev_resolution_m",
            "source",
            "image_convention",
            "walls",
        }
        actual_fields = set(data)
        if actual_fields != expected_fields:
            missing = sorted(expected_fields.difference(actual_fields))
            extra = sorted(actual_fields.difference(expected_fields))
            raise ValueError(
                f"二维场景字段集合不符合契约；缺少={missing}；额外={extra}"
            )
        expected_convention = {
            "origin": "upper_left",
            "column_axis": "+x",
            "row_axis": "-y",
            "pixel_center_units": "pixel",
        }
        if data["image_convention"] != expected_convention:
            raise ValueError("二维场景 image_convention 与定位坐标约定不一致")
        raw_walls = data["walls"]
        if not isinstance(raw_walls, list):
            raise ValueError("二维场景 walls 必须是列表")
        expected_wall_fields = {"wall_id", "start_m", "end_m", "source_object"}
        walls_list: list[WallSegment] = []
        for index, wall in enumerate(raw_walls):
            if not isinstance(wall, Mapping) or set(wall) != expected_wall_fields:
                raise ValueError(f"二维场景第 {index} 条墙的字段集合不符合契约")
            walls_list.append(WallSegment(**dict(wall)))
        walls = tuple(walls_list)
        return cls(
            name=str(data["name"]),
            bounds_m=tuple(float(v) for v in data["bounds_m"]),
            walls=walls,
            fixed_height_m=float(data["fixed_height_m"]),
            bev_resolution_m=float(data["bev_resolution_m"]),
            source=str(data.get("source", "unknown")),
        )

    def save(
        self, output_dir: str | Path, *, allow_overwrite: bool = False
    ) -> dict[str, str]:
        """原子保存完整场景；已有任一正式目标时整体拒绝。"""

        directory = Path(output_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        scene_path = directory / "scene_2d.json"
        bev_path = directory / "scene_bev.png"
        occupancy_path = directory / "scene_occupancy.npy"
        if not isinstance(allow_overwrite, bool):
            raise ValueError("allow_overwrite 必须是布尔值")
        targets = (scene_path, bev_path, occupancy_path)
        conflicts = [path for path in targets if path.exists() or path.is_symlink()]
        if conflicts and not allow_overwrite:
            joined = "、".join(str(path) for path in conflicts)
            raise FileExistsError(f"场景目标已存在，拒绝覆盖整组文件：{joined}")

        scene_bytes = json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2
        ).encode("utf-8")
        occupancy = self.rasterize()
        staged_files = [
            (
                scene_path,
                _stage_scene_file(
                    scene_path, lambda handle: handle.write(scene_bytes)
                ),
            ),
            (
                bev_path,
                _stage_scene_file(
                    bev_path,
                    lambda handle: Image.fromarray(occupancy, mode="L").save(
                        handle, format="PNG"
                    ),
                ),
            ),
            (
                occupancy_path,
                _stage_scene_file(
                    occupancy_path, lambda handle: np.save(handle, occupancy)
                ),
            ),
        ]
        _publish_scene_files(staged_files, allow_overwrite=allow_overwrite)
        return {
            "scene_json": str(scene_path),
            "bev_png": str(bev_path),
            "occupancy_npy": str(occupancy_path),
        }

    @classmethod
    def load(cls, scene_json: str | Path) -> "Scene2D":
        data = json.loads(Path(scene_json).expanduser().resolve().read_text(encoding="utf-8"))
        return cls.from_dict(data)


def make_synthetic_room(
    bounds_m: Sequence[float] = (0.0, 20.0, 0.0, 14.0),
    *,
    fixed_height_m: float = 1.5,
    bev_resolution_m: float = 0.05,
) -> Scene2D:
    """创建只用于离线闭环测试的矩形房间。"""

    x_min, x_max, y_min, y_max = (float(v) for v in bounds_m)
    corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    walls = tuple(
        WallSegment(
            wall_id=f"boundary_{idx}",
            start_m=corners[idx],
            end_m=corners[(idx + 1) % 4],
            source_object="synthetic_room",
        )
        for idx in range(4)
    )
    return Scene2D(
        name="offline_room",
        bounds_m=(x_min, x_max, y_min, y_max),
        walls=walls,
        fixed_height_m=float(fixed_height_m),
        bev_resolution_m=float(bev_resolution_m),
        source="synthetic_room",
    )


def _iter_scene_objects(scene: Any) -> Iterable[Any]:
    objects = getattr(scene, "objects", None)
    if objects is None:
        raise ValueError("DeepMIMO Scene 缺少 objects 字段，无法提取二维墙线")
    return objects


def _farthest_xy_pair(vertices_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = vertices_xy[:, None, :] - vertices_xy[None, :, :]
    distances_sq = np.sum(delta * delta, axis=-1)
    i, j = np.unravel_index(int(np.argmax(distances_sq)), distances_sq.shape)
    return vertices_xy[i], vertices_xy[j]


def _canonical_segment_key(start: np.ndarray, end: np.ndarray, tolerance_m: float) -> tuple[int, ...]:
    scale = 1.0 / tolerance_m
    a = tuple(np.rint(start * scale).astype(int).tolist())
    b = tuple(np.rint(end * scale).astype(int).tolist())
    return (*a, *b) if a <= b else (*b, *a)


def _triangle_horizontal_slice(
    triangle_m: np.ndarray,
    *,
    fixed_height_m: float,
    plane_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """返回三角形与 ``z=fixed_height_m`` 平面的二维交线。"""

    signed_height = triangle_m[:, 2] - float(fixed_height_m)
    points: list[np.ndarray] = []

    def append_unique(point_xy: np.ndarray) -> None:
        if not any(
            np.linalg.norm(point_xy - existing) <= plane_tolerance_m
            for existing in points
        ):
            points.append(np.asarray(point_xy, dtype=float))

    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start = triangle_m[start_index]
        end = triangle_m[end_index]
        start_height = float(signed_height[start_index])
        end_height = float(signed_height[end_index])
        start_on_plane = abs(start_height) <= plane_tolerance_m
        end_on_plane = abs(end_height) <= plane_tolerance_m
        if start_on_plane:
            append_unique(start[:2])
        if end_on_plane:
            append_unique(end[:2])
        if start_on_plane or end_on_plane or start_height * end_height >= 0.0:
            continue
        fraction = -start_height / (end_height - start_height)
        append_unique(start[:2] + fraction * (end[:2] - start[:2]))

    if len(points) < 2:
        return None
    return _farthest_xy_pair(np.vstack(points))


def preprocess_sionna_triangle_mesh(
    vertices_m: Any,
    faces: Any,
    *,
    name: str,
    fixed_height_m: float,
    bev_resolution_m: float,
    object_vertex_ranges: Mapping[str, Sequence[int]] | None = None,
    bounds_m: Sequence[float] | None = None,
    min_wall_height_m: float = 0.5,
    min_wall_length_m: float = 0.2,
    vertical_normal_z_max: float = 0.15,
    dedup_tolerance_m: float = 1e-3,
    plane_tolerance_m: float = 1e-6,
) -> Scene2D:
    """从 Sionna 原始三角网格在固定高度截取二维墙线。

    这里直接切原始三角形，不对连通组件求凸包，也不会用两个不相邻顶点
    虚构跨越凹口或分离组件的墙。``object_vertex_ranges`` 使用
    ``sionna_exporter`` 写出的 ``对象名 -> [起始顶点, 结束顶点)`` 约定。
    """

    vertices = np.asarray(vertices_m, dtype=float)
    face_indices = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
        raise ValueError("Sionna vertices 必须是全部有限的 [顶点数, 3] 数组")
    if face_indices.ndim != 2 or face_indices.shape[1] != 3:
        raise ValueError("Sionna faces 必须是 [三角形数, 3] 的顶点编号数组")
    if not np.issubdtype(face_indices.dtype, np.integer):
        if not np.all(np.equal(face_indices, np.rint(face_indices))):
            raise ValueError("Sionna faces 的顶点编号必须是整数")
    face_indices = face_indices.astype(np.int64, copy=False)
    if face_indices.size == 0:
        raise ValueError("Sionna faces 不能为空")
    if int(np.min(face_indices)) < 0 or int(np.max(face_indices)) >= len(vertices):
        raise ValueError("Sionna faces 含有越界顶点编号")
    if plane_tolerance_m <= 0.0 or dedup_tolerance_m <= 0.0:
        raise ValueError("平面容差和去重容差必须为正数")

    object_names: list[str] = []
    vertex_owner = np.full(len(vertices), -1, dtype=np.int64)
    if object_vertex_ranges is not None:
        for object_index, (object_name, vertex_range) in enumerate(
            object_vertex_ranges.items()
        ):
            values = np.asarray(vertex_range)
            if values.shape != (2,) or not np.all(np.equal(values, np.rint(values))):
                raise ValueError(f"Sionna 对象 {object_name} 的顶点范围必须是两个整数")
            start, end = (int(values[0]), int(values[1]))
            if start < 0 or end <= start or end > len(vertices):
                raise ValueError(f"Sionna 对象 {object_name} 的顶点范围越界")
            if np.any(vertex_owner[start:end] >= 0):
                raise ValueError(f"Sionna 对象 {object_name} 的顶点范围与其他对象重叠")
            vertex_owner[start:end] = object_index
            object_names.append(str(object_name))

    walls: list[WallSegment] = []
    seen: set[tuple[int, ...]] = set()
    for face_index, vertex_indices in enumerate(face_indices):
        triangle = vertices[vertex_indices]
        if float(np.ptp(triangle[:, 2])) < min_wall_height_m:
            continue
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= _GEOMETRY_EPS:
            continue
        if abs(float(normal[2]) / normal_norm) > vertical_normal_z_max:
            continue
        segment = _triangle_horizontal_slice(
            triangle,
            fixed_height_m=fixed_height_m,
            plane_tolerance_m=plane_tolerance_m,
        )
        if segment is None:
            continue
        start, end = segment
        if float(np.linalg.norm(end - start)) < min_wall_length_m:
            continue
        key = _canonical_segment_key(start, end, dedup_tolerance_m)
        if key in seen:
            continue
        seen.add(key)

        owners = vertex_owner[vertex_indices]
        owner_index = int(owners[0]) if np.all(owners == owners[0]) else -1
        source_object = (
            object_names[owner_index]
            if 0 <= owner_index < len(object_names)
            else "sionna_mesh"
        )
        owner_token = f"{owner_index:04d}" if owner_index >= 0 else "unknown"
        walls.append(
            WallSegment(
                wall_id=f"sionna_{owner_token}_{face_index:06d}",
                start_m=(float(start[0]), float(start[1])),
                end_m=(float(end[0]), float(end[1])),
                source_object=source_object,
            )
        )

    if not walls:
        raise ValueError("Sionna 原始三角网格在指定高度没有可用竖直墙线")

    if bounds_m is None:
        all_points = np.vstack([[wall.start, wall.end] for wall in walls])
        padding = max(1.0, 5.0 * bev_resolution_m)
        inferred = (
            float(np.min(all_points[:, 0]) - padding),
            float(np.max(all_points[:, 0]) + padding),
            float(np.min(all_points[:, 1]) - padding),
            float(np.max(all_points[:, 1]) + padding),
        )
    else:
        inferred = tuple(float(value) for value in bounds_m)
        if len(inferred) != 4:
            raise ValueError("bounds_m 必须依次包含 x_min、x_max、y_min、y_max")

    x_min, x_max, y_min, y_max = inferred
    walls = [
        wall
        for wall in walls
        if max(wall.start_m[0], wall.end_m[0]) >= x_min
        and min(wall.start_m[0], wall.end_m[0]) <= x_max
        and max(wall.start_m[1], wall.end_m[1]) >= y_min
        and min(wall.start_m[1], wall.end_m[1]) <= y_max
    ]
    if not walls:
        raise ValueError("裁剪区域内没有可用的 Sionna 原始墙线")

    return Scene2D(
        name=name,
        bounds_m=inferred,
        walls=tuple(walls),
        fixed_height_m=float(fixed_height_m),
        bev_resolution_m=float(bev_resolution_m),
        source="sionna_exported_triangle_mesh",
    )


def preprocess_sionna_exported_scene(
    export_dir: str | Path,
    *,
    name: str,
    fixed_height_m: float,
    bev_resolution_m: float,
    bounds_m: Sequence[float] | None = None,
) -> Scene2D:
    """读取 ``sionna_exporter`` 的原始网格文件并生成定位二维场景。"""

    directory = Path(export_dir).expanduser().resolve()
    paths = {
        "vertices": directory / "sionna_vertices.pkl",
        "faces": directory / "sionna_faces.pkl",
        "objects": directory / "sionna_objects.pkl",
    }
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Sionna 原始场景导出不完整，缺少：{', '.join(sorted(missing))}"
        )
    with paths["vertices"].open("rb") as handle:
        vertices = pickle.load(handle)
    with paths["faces"].open("rb") as handle:
        faces = pickle.load(handle)
    with paths["objects"].open("rb") as handle:
        objects = pickle.load(handle)
    if not isinstance(objects, Mapping):
        raise ValueError("sionna_objects.pkl 必须是对象名到顶点范围的映射")
    return preprocess_sionna_triangle_mesh(
        vertices,
        faces,
        name=name,
        fixed_height_m=fixed_height_m,
        bev_resolution_m=bev_resolution_m,
        object_vertex_ranges=objects,
        bounds_m=bounds_m,
    )


def preprocess_deepmimo_scene(
    scene: Any,
    *,
    name: str,
    fixed_height_m: float,
    bev_resolution_m: float,
    bounds_m: Sequence[float] | None = None,
    min_wall_height_m: float = 0.5,
    min_wall_length_m: float = 0.2,
    vertical_normal_z_max: float = 0.15,
    dedup_tolerance_m: float = 1e-3,
) -> Scene2D:
    """从 DeepMIMO V4 Scene 的竖直面提取二维墙线。

    地面和屋顶会因法向量近似竖直而被跳过。重复三角面边会按坐标去重。
    """

    walls: list[WallSegment] = []
    seen: set[tuple[int, ...]] = set()
    for object_idx, obj in enumerate(_iter_scene_objects(scene)):
        object_name = str(getattr(obj, "name", "") or f"object_{object_idx}")
        faces = getattr(obj, "faces", None)
        if faces is None:
            continue
        for face_idx, face in enumerate(faces):
            vertices = np.asarray(getattr(face, "vertices", []), dtype=float)
            if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 3:
                continue
            if float(np.ptp(vertices[:, 2])) < min_wall_height_m:
                continue
            normal = getattr(face, "normal", None)
            if normal is not None:
                normal_arr = np.asarray(normal, dtype=float).reshape(-1)
                if normal_arr.size >= 3 and abs(float(normal_arr[2])) > vertical_normal_z_max:
                    continue
            start, end = _farthest_xy_pair(vertices[:, :2])
            if float(np.linalg.norm(end - start)) < min_wall_length_m:
                continue
            key = _canonical_segment_key(start, end, dedup_tolerance_m)
            if key in seen:
                continue
            seen.add(key)
            walls.append(
                WallSegment(
                    wall_id=f"dm_{object_idx}_{face_idx}",
                    start_m=(float(start[0]), float(start[1])),
                    end_m=(float(end[0]), float(end[1])),
                    source_object=object_name,
                )
            )
    if not walls:
        raise ValueError("场景中没有提取到可用的竖直墙面")

    if bounds_m is None:
        all_points = np.vstack([[wall.start, wall.end] for wall in walls])
        padding = max(1.0, 5.0 * bev_resolution_m)
        inferred = (
            float(np.min(all_points[:, 0]) - padding),
            float(np.max(all_points[:, 0]) + padding),
            float(np.min(all_points[:, 1]) - padding),
            float(np.max(all_points[:, 1]) + padding),
        )
    else:
        inferred = tuple(float(v) for v in bounds_m)

    x_min, x_max, y_min, y_max = inferred
    walls = [
        wall
        for wall in walls
        if max(wall.start_m[0], wall.end_m[0]) >= x_min
        and min(wall.start_m[0], wall.end_m[0]) <= x_max
        and max(wall.start_m[1], wall.end_m[1]) >= y_min
        and min(wall.start_m[1], wall.end_m[1]) <= y_max
    ]
    if not walls:
        raise ValueError("裁剪区域内没有可用墙面")

    return Scene2D(
        name=name,
        bounds_m=inferred,
        walls=tuple(walls),
        fixed_height_m=float(fixed_height_m),
        bev_resolution_m=float(bev_resolution_m),
        source="deepmimo_v4_scene",
    )


def cross_2d(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    return float(a_arr[0] * b_arr[1] - a_arr[1] * b_arr[0])


def ray_segment_intersection(
    origin: Sequence[float],
    direction: Sequence[float],
    wall: WallSegment,
    *,
    min_distance_m: float = 1e-7,
) -> tuple[float, float, np.ndarray] | None:
    """求射线与有限墙段交点，返回射线距离、墙段比例和交点。"""

    origin_arr = np.asarray(origin, dtype=float)
    direction_arr = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction_arr))
    if norm <= _GEOMETRY_EPS:
        raise ValueError("射线方向不能为零")
    direction_arr = direction_arr / norm
    wall_vec = wall.vector
    denominator = cross_2d(direction_arr, wall_vec)
    if abs(denominator) <= _GEOMETRY_EPS:
        return None
    offset = wall.start - origin_arr
    ray_distance = cross_2d(offset, wall_vec) / denominator
    wall_fraction = cross_2d(offset, direction_arr) / denominator
    if ray_distance <= min_distance_m or wall_fraction < -_GEOMETRY_EPS or wall_fraction > 1.0 + _GEOMETRY_EPS:
        return None
    point = origin_arr + ray_distance * direction_arr
    return float(ray_distance), float(wall_fraction), point


def reflect_direction(direction: Sequence[float], wall: WallSegment) -> np.ndarray:
    direction_arr = np.asarray(direction, dtype=float)
    direction_arr = direction_arr / np.linalg.norm(direction_arr)
    normal = wall.normal
    reflected = direction_arr - 2.0 * float(np.dot(direction_arr, normal)) * normal
    return reflected / np.linalg.norm(reflected)


def reflect_point(point: Sequence[float], wall: WallSegment) -> np.ndarray:
    point_arr = np.asarray(point, dtype=float)
    normal = wall.normal
    return point_arr - 2.0 * float(np.dot(point_arr - wall.start, normal)) * normal
