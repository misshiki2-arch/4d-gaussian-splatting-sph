"""Formal camera preflight and immutable values. Standard library only.

Raw metadata is checked before inheritance. P4 owns resolution; this module
applies its divisor exactly once. Neither legacy sentinels nor runtime defaults
are an authority for the effective camera.
"""
import math
import struct
from typing import NamedTuple


class RawCamera(NamedTuple):
    root_fields: tuple
    frame_fields: tuple
    source: str
    width: int
    height: int
    intrinsics: tuple
    fovs: tuple
    reader_sentinel: bool


class EffectiveCamera(NamedTuple):
    raw: RawCamera
    mode: str
    resolution: object
    frame_key: tuple
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fovx: float
    fovy: float
    tanx: float
    tany: float
    znear: float
    zfar: float
    raster_near: float = 0.2
    raster_far: object = None


def _positive(value):
    if type(value) not in (int, float):
        raise ValueError('formal_camera_number')
    try:
        value = float(value)
    except OverflowError:
        raise ValueError('formal_camera_number') from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError('formal_camera_number')
    return value


def _float32(value):
    value = _positive(value)
    try:
        narrowed = struct.unpack('f', struct.pack('f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError('formal_camera_float32') from None
    if not math.isfinite(narrowed) or narrowed <= 0:
        raise ValueError('formal_camera_float32')
    return value


def _angle(value):
    value = _positive(value)
    if value >= math.pi:
        raise ValueError('formal_camera_angle')
    return value


def _dimension(value):
    if type(value) is not int or not 0 < value <= 2147483647:
        raise ValueError('formal_camera_dimension')
    return value


def build_camera(resolution, frame_key, *, intrinsics=None, fovs=None,
                 root_fields=(), frame_fields=(), source='explicit',
                 znear=0.01, zfar=100.0):
    """Pure complete-pair API, also used by the existing JSON format adapter."""
    w0, h0 = _dimension(resolution.raw_width), _dimension(resolution.raw_height)
    d = _dimension(resolution.divisor)
    w, h = _dimension(resolution.width), _dimension(resolution.height)
    if w0 % d or h0 % d or (w, h) != (w0 // d, h0 // d):
        raise ValueError('formal_camera_resolution')
    if (intrinsics is None) == (fovs is None):
        raise ValueError('formal_camera_mode')
    znear, zfar = _positive(znear), _positive(zfar)
    if not znear < zfar or (znear, zfar) != (0.01, 100.0):
        raise ValueError('formal_camera_clip')
    if intrinsics is not None:
        if len(intrinsics) != 4:
            raise ValueError('formal_camera_partial')
        fx0, fy0, cx0, cy0 = map(_positive, intrinsics)
        if (cx0, cy0) != (w0 / 2, h0 / 2):
            raise ValueError('formal_camera_off_center')
        mode, raw_fovs = 'intrinsics', ()
    else:
        if len(fovs) != 2:
            raise ValueError('formal_camera_partial')
        ax, ay = map(_angle, fovs)
        fx0 = w0 / (2 * _positive(math.tan(ax / 2)))
        fy0 = h0 / (2 * _positive(math.tan(ay / 2)))
        cx0, cy0 = w0 / 2, h0 / 2
        mode, raw_fovs = 'fov', (ax, ay)
    fx, fy, cx, cy = (v / d for v in (fx0, fy0, cx0, cy0))
    for value in (fx, fy, cx, cy):
        _float32(value)
    if (cx, cy) != (w / 2, h / 2):
        raise ValueError('formal_camera_off_center')
    tx, ty = w / (2 * fx), h / (2 * fy)
    ax, ay = 2 * math.atan(tx), 2 * math.atan(ty)
    # Check the handoff and projection scalar domains before heavy import.
    for value in (tx, ty, ax, ay, 1 / tx if tx else 0, 1 / ty if ty else 0):
        _float32(value)
    _angle(ax)
    _angle(ay)
    raw = RawCamera(tuple(root_fields), tuple(frame_fields), source, w0, h0,
                    (fx0, fy0, cx0, cy0), raw_fovs, mode == 'intrinsics')
    return EffectiveCamera(raw, mode, resolution, tuple(frame_key), w, h,
                           fx, fy, cx, cy, ax, ay, tx, ty, znear, zfar)


def _group(mapping, names):
    present = tuple(name in mapping for name in names)
    if any(present) and not all(present):
        raise ValueError('formal_camera_partial')
    return tuple(mapping[n] for n in names) if all(present) else None


def from_metadata(root, frame, resolution, frame_key):
    """Adapt the existing root-angle-x format; do not add camera overrides."""
    keys = ('w', 'h', 'fl_x', 'fl_y', 'cx', 'cy', 'camera_angle_x')
    for mapping in (root, frame):
        for key in mapping:
            if ((key.startswith(('camera_angle_', 'fov', 'FoV', 'fl_')) and key not in keys)
                    or key in ('fx', 'fy', 'znear', 'zfar', 'near', 'far')):
                raise ValueError('formal_camera_override')
    if 'camera_angle_x' in frame:
        raise ValueError('formal_camera_override')
    dims = [_group(m, ('w', 'h')) for m in (root, frame)]
    for pair in dims:
        if pair is not None:
            pair = tuple(map(_dimension, pair))
            if pair != (resolution.raw_width, resolution.raw_height):
                raise ValueError('formal_camera_dimensions')
    if dims == [None, None]:
        raise ValueError('formal_camera_dimensions')
    groups = [_group(m, ('fl_x', 'fl_y', 'cx', 'cy')) for m in (root, frame)]
    groups = [None if g is None else tuple(map(_positive, g)) for g in groups]
    if groups[0] is not None and groups[1] is not None and groups[0] != groups[1]:
        raise ValueError('formal_camera_conflict')
    intrinsics = groups[0] if groups[0] is not None else groups[1]
    fovs = None
    if 'camera_angle_x' in root:
        ax = _angle(root['camera_angle_x'])
        if intrinsics is not None:
            raise ValueError('formal_camera_mixed')
        focal = resolution.raw_width / (2 * _positive(math.tan(ax / 2)))
        fovs = (ax, 2 * math.atan(resolution.raw_height / (2 * focal)))
    source = 'both' if all(g is not None for g in groups) else 'frame' if groups[1] is not None else 'root'
    return build_camera(resolution, frame_key, intrinsics=intrinsics, fovs=fovs,
                        root_fields=tuple((k, root[k]) for k in keys if k in root),
                        frame_fields=tuple((k, frame[k]) for k in keys if k in frame), source=source)
