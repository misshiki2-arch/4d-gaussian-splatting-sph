"""Step 10 test adapter: real config/loader/Camera, no production math oracle.

Heavy imports are local. CPU isolation belongs to the caller's test context;
the future CUDA caller uses these exact consumers without mocks.
"""
import json
import math
from types import SimpleNamespace


def make_camera(base, spec, *, split='train'):
    import numpy as np
    from PIL import Image
    from formal_camera import from_metadata, build_camera
    from formal_config import resolve_formal_config
    from formal_frame_time import FrameTime
    from scene.dataset_readers import readCamerasFromTransforms
    from scene.cameras import Camera
    from utils.camera_utils import loadCam
    from test_formal_config import fixture, encode

    source = base/'data'
    source.mkdir(exist_ok=True)
    filename = split+'.png'
    Image.new('RGBA', (spec['raw_width'], spec['raw_height']), (80, 40, 20, 255)).save(source/filename)
    value = fixture(source, base/'unused-output')
    value['dataset']['resolution']['divisor'] = spec['divisor']
    value['dataset']['time'].update(raw_interval=spec.get('raw_interval', [0, 1]),
                                  divisor=spec.get('time_divisor', 1))
    value['initialization']['time_variance_denominator'] = 5
    config = resolve_formal_config(encode(value), raw_width=spec['raw_width'], raw_height=spec['raw_height'])
    pose = np.eye(4)
    pose[:3, :3] = spec['rotation'].numpy().T @ np.diag([1., -1., -1.])
    pose[:3, 3] = spec['center'].numpy()
    frame = dict(file_path=filename, time=spec['timestamp'], transform_matrix=pose.tolist())
    root = dict(w=spec['raw_width'], h=spec['raw_height'], frames=[frame])
    timing = FrameTime(split, 0, filename,
                      spec['timestamp']*config.time_derivation.divisor, spec['timestamp'])
    key = (split, 0, filename)
    if spec['mode'] == 'pair':
        state = build_camera(config.dataset.resolution, key,
                             fovs=(2*math.atan(spec['tanx']), 2*math.atan(spec['tany'])))
        camera = Camera(0, spec['rotation'].numpy().T,
                        (-spec['rotation'] @ spec['center']).numpy(), state.fovx, state.fovy,
                        None, None, split, 0, data_device='cuda', timestamp=timing.effective_time,
                        cx=state.cx, cy=state.cy, fl_x=state.fx, fl_y=state.fy,
                        resolution=(state.width, state.height), image_path=str(source/filename),
                        meta_only=True, formal_camera=state, formal_frame=timing)
    else:
        if spec['mode'] == 'fov':
            root['camera_angle_x'] = 2*math.atan(spec['tanx'])
        else:
            root.update(fl_x=spec['raw_fx'], fl_y=spec['raw_fy'],
                        cx=spec['raw_width']/2, cy=spec['raw_height']/2)
        state = from_metadata(root, frame, config.dataset.resolution, key)
        metadata = json.dumps(root).encode()
        infos = readCamerasFromTransforms(str(source), 'unused-metadata-path.json', False,
                    extension='', dataloader=True, formal_metadata=(metadata, (timing,)),
                    formal_cameras=(state,))
        if len(infos) != 1: raise AssertionError('fixture_frame_count')
        if state.raw.reader_sentinel and (infos[0].FovX, infos[0].FovY) != (-1., -1.):
            raise AssertionError('raw_sentinel_not_retained')
        camera = loadCam(SimpleNamespace(dataloader=True, data_device='cuda'), 0, infos[0], 1.,
                         formal_resolution=config.dataset.resolution)
    if (base/'unused-output').exists(): raise AssertionError('formal_output_created')
    return camera, config


def check_camera(torch, camera, spec, rows):
    """Compare actual consumer state against independently constructed matrices."""
    import camera_projection_oracle as o
    eps = torch.finfo(torch.float32).eps
    state = camera.formal_camera
    expected_mode = 'intrinsics' if spec['mode'] == 'intrinsics' else 'fov'
    if (state.mode != expected_mode or (state.width, state.height) != (spec['width'], spec['height'])
            or (state.znear, state.zfar, state.raster_near, state.raster_far) != (.01, 100., .2, None)):
        raise AssertionError('canonical_dimensions_mode_clip')
    for name in ('fx', 'fy', 'tanx', 'tany'):
        err = abs(getattr(state, name)-spec[name])
        bound = 32*2**-52*max(1., abs(spec[name]))
        if err > bound: raise AssertionError('canonical_scalar_'+name)
        rows.append(dict(check=spec['label']+'/'+name, tolerance='camera_scalar',
                         max_abs=err, max_bound_ratio=err/bound))
    view, projection, full = o.matrices(spec)
    for name, expected in (('world_view_transform', view.T), ('projection_matrix', projection.T)):
        actual = getattr(camera, name).detach().cpu()
        if actual.dtype != torch.float32 or actual.shape != (4, 4): raise AssertionError('camera_matrix_type')
        delta = (actual.double()-expected).abs()
        bound = 8*eps*expected.abs().clamp_min(.01)
        bound = torch.where(expected == 0, torch.zeros_like(bound), bound)
        if not bool((delta <= bound).all()): raise AssertionError('camera_matrix_'+name)
        ratio = torch.where(bound == 0, torch.zeros_like(delta), delta/bound)
        rows.append(dict(check=spec['label']+'/'+name, tolerance='camera_matrix',
                         max_abs=float(delta.max()), max_bound_ratio=float(ratio.max())))
    actual = camera.full_proj_transform.detach().cpu().double()
    bound = 32*eps*torch.linalg.matrix_norm(view.T, ord=float('inf'))*torch.linalg.matrix_norm(projection.T, ord=float('inf'))
    if not bool(((actual-full.T).abs() <= bound).all()): raise AssertionError('camera_full_matrix')
    rows.append(dict(check=spec['label']+'/full_matrix', tolerance='camera_full_matrix',
                     max_abs=float((actual-full.T).abs().max()),
                     max_bound_ratio=float(((actual-full.T).abs()/bound).max())))
    if not torch.allclose(camera.camera_center.detach().cpu().double(), spec['center'], atol=8*eps, rtol=8*eps):
        raise AssertionError('camera_center')
    delta = (camera.camera_center.detach().cpu().double()-spec['center']).abs()
    rows.append(dict(check=spec['label']+'/center', tolerance='camera_center',
                     max_abs=float(delta.max()),
                     max_bound_ratio=float((delta/(8*eps+8*eps*spec['center'].abs())).max())))
    if camera.formal_binding.camera is not camera: raise AssertionError('camera_binding_identity')
    return dict(mode=state.mode, divisor=state.resolution.divisor, frame=list(state.frame_key))
