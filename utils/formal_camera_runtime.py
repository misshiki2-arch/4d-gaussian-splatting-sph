"""Heavy-side camera handoff; no scene import, registry or GPU readback.

CPU construction proves numeric correspondence. Subsequent checks use scalar
values and tensor references/versions, not GPU kernels. This is an internal
handoff contract, not a security boundary against arbitrary Python mutation.
"""
from dataclasses import dataclass
import torch
from formal_camera import EffectiveCamera
from utils.graphics_utils import getProjectionMatrixCenterShift


def _attributes(camera, state, frame):
    if (not isinstance(state, EffectiveCamera) or frame is None
            or camera.formal_camera != state or camera.formal_frame != frame
            or state.frame_key != (frame.split, frame.index, frame.file_path)
            or camera.colmap_id != frame.index or camera.timestamp != frame.effective_time
            or not camera.image_path.endswith('/' + frame.file_path)
            or (camera.image_width, camera.image_height) != (state.width, state.height)
            or camera.resolution != (state.width, state.height)
            or (camera.fl_x, camera.fl_y, camera.cx, camera.cy, camera.FoVx, camera.FoVy,
                camera.znear, camera.zfar) != (state.fx, state.fy, state.cx, state.cy,
                                              state.fovx, state.fovy, state.znear, state.zfar)):
        raise ValueError('formal_camera_runtime')


_MATRICES = ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center')


@dataclass(frozen=True, eq=False)
class CameraBinding:
    camera: object
    state: EffectiveCamera
    frame: object
    tensors: tuple
    versions: tuple

    def validate(self):
        _attributes(self.camera, self.state, self.frame)
        for name, tensor, version in zip(_MATRICES, self.tensors, self.versions):
            if getattr(self.camera, name) is not tensor or tensor._version != version:
                raise ValueError('formal_camera_matrix_changed')


def bind_camera(camera, *, transferred=False):
    state, frame = camera.formal_camera, camera.formal_frame
    _attributes(camera, state, frame)
    tensors = tuple(getattr(camera, name) for name in _MATRICES)
    for tensor, shape in zip(tensors, ((4, 4), (4, 4), (4, 4), (3,))):
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError('formal_camera_matrix_type')
    if not transferred:
        if any(t.device.type != 'cpu' for t in tensors):
            raise ValueError('formal_camera_cpu_construction')
        if not all(torch.isfinite(t).all().item() for t in tensors):
            raise ValueError('formal_camera_matrix_finite')
        expected = getProjectionMatrixCenterShift(state.znear, state.zfar, state.cx, state.cy,
                    state.fx, state.fy, state.width, state.height, dtype=torch.float32).transpose(0, 1)
        if not torch.equal(tensors[1], expected) or not torch.equal(
                tensors[2], tensors[0].unsqueeze(0).bmm(expected.unsqueeze(0)).squeeze(0)):
            raise ValueError('formal_camera_projection')
    return CameraBinding(camera, state, frame, tensors, tuple(t._version for t in tensors))


def camera_binding(camera, resolution):
    binding = getattr(camera, 'formal_binding', None)
    if not isinstance(binding, CameraBinding) or binding.camera is not camera:
        raise ValueError('formal_camera_binding_missing')
    binding.validate()
    if binding.state.resolution != resolution:
        raise ValueError('formal_camera_resolution')
    return binding


def validate_settings(settings, gradient=None):
    binding = settings.camera_binding
    if not settings.formal_camera:
        if binding is not None:
            raise ValueError('formal_camera_settings_mode')
        return
    if not isinstance(binding, CameraBinding):
        raise ValueError('formal_camera_binding_missing')
    binding.validate()
    state = binding.state
    if ((settings.image_width, settings.image_height, settings.tanfovx, settings.tanfovy)
            != (state.width, state.height, state.tanx, state.tany)
            or settings.viewmatrix is not binding.tensors[0]
            or settings.projmatrix is not binding.tensors[2]
            or settings.campos is not binding.tensors[3]
            or settings.timestamp != binding.frame.effective_time):
        raise ValueError('formal_camera_settings')
    if gradient is not None and tuple(gradient.shape) != (3, state.height, state.width):
        raise ValueError('formal_camera_gradient_shape')
