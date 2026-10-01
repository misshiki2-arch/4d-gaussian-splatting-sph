"""Step 6 CPU acceptance: actual consumers, observers only at GPU/C boundaries.

Observer images/buffers/gradients drive Python autograd, NOT CUDA correctness.
All fixtures/output are temporary; the installed CPU dependencies are real.
"""
from contextlib import contextmanager
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_formal_connection as fixtures


class CameraConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(fixtures.ConnectionTests.doClassCleanups)
        fixtures.ConnectionTests.setUpClass()
        cls.binding = importlib.import_module('gaussian_renderer.diff_gaussian_rasterization')

    def setUp(self):
        self.f = fixtures.ConnectionTests('runTest')
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()
        self.torch, self.np = self.f.torch, self.f.np
        original_to = self.torch.Tensor.to
        def cpu_to(tensor, *args, **kwargs):
            if args and str(args[0]).startswith('cuda'):
                args = ('cpu',) + args[1:]
            if str(kwargs.get('device', '')).startswith('cuda'):
                kwargs['device'] = 'cpu'
            return original_to(tensor, *args, **kwargs)
        self.f.gpu.enter_context(patch.object(self.torch.Tensor, 'to', cpu_to))

    def metadata(self, split):
        return json.loads((self.f.source / f'transforms_{split}.json').read_bytes())

    def save_metadata(self, split, root):
        self.f.store_input(f'transforms_{split}.json', json.dumps(root).encode())

    def configure(self, divisor=1, fov=False):
        f = self.f
        f.value['dataset']['resolution']['divisor'] = divisor
        angle = .3
        self.rotation = self.np.array([[math.cos(angle), -math.sin(angle), 0],
                                       [math.sin(angle), math.cos(angle), 0], [0, 0, 1.]])
        for split in ('train', 'test'):
            root = self.metadata(split)
            root['fl_y'] = 25
            if fov:
                for key in ('fl_x', 'fl_y', 'cx', 'cy'): del root[key]
                root['camera_angle_x'] = math.pi / 2
            for index, frame in enumerate(root['frames']):
                pose = self.np.eye(4)
                pose[:3, :3] = self.rotation
                pose[:3, 3] = [1 + index + (10 if split == 'test' else 0), 2, 3]
                frame['transform_matrix'] = pose.tolist()
                f.Image.new('L', (20 // divisor, 10 // divisor), 128).save(
                    f.source / frame['file_path'].replace('images/', 'masks/'))
            self.save_metadata(split, root)

    def oracle(self, divisor=1, fov=False, *, test_split=False):
        np = self.np
        fx, fy = (10., 10.) if fov else (15., 25.)
        # Independent formula, never call the production builder/projection.
        p = np.zeros((4, 4), dtype=np.float64)
        p[0, 0], p[1, 1] = 2 * fx / 20, 2 * fy / 10
        p[2, 2], p[2, 3], p[3, 2] = 100 / 99.99, -1 / 99.99, 1
        world = np.eye(4)
        rt = (self.rotation @ np.diag([1., -1., -1.])).T
        world[:3, :3] = rt
        world[:3, 3] = -rt @ np.array([11. if test_split else 1., 2., 3.])
        return dict(width=20 // divisor, height=10 // divisor,
                    fx=fx / divisor, fy=fy / divisor, tanx=20 / (2 * fx), tany=10 / (2 * fy),
                    projection=p.T, world=world.T, full=(p @ world).T)

    def check_camera(self, camera, expected):
        np = self.np
        eps = np.finfo(np.float32).eps
        self.assertEqual((camera.image_width, camera.image_height), (expected['width'], expected['height']))
        for tensor, name in ((camera.projection_matrix, 'projection'), (camera.world_view_transform, 'world')):
            target = expected[name].astype(np.float32)
            delta = np.abs(tensor.numpy() - target)
            tolerance = np.where(target == 0, 0, 8 * eps * np.maximum(np.abs(target), .01))
            self.assertTrue(np.all(delta <= tolerance), (name, delta, tolerance))
        limit = 32 * eps * np.linalg.norm(expected['world'], ord=np.inf) * np.linalg.norm(expected['projection'], ord=np.inf)
        self.assertLessEqual(float(np.abs(camera.full_proj_transform.numpy() - expected['full'].astype(np.float32)).max()), limit)
        for name, expected_name in (('fl_x', 'fx'), ('fl_y', 'fy')):
            self.assertAlmostEqual(getattr(camera, name), expected[expected_name], places=12)

    @contextmanager
    def observers(self, expected):
        torch = self.torch
        observed = dict(forward=[], backward=[], contexts=[])
        original = self.binding._RasterizeGaussians.forward
        def forward_context(ctx, *args):
            result = original(ctx, *args)
            observed['contexts'].append(ctx)
            return result
        def forward(*args):
            self.assertIs(type(args[12]), float)
            self.assertEqual(args[12], -1.0)  # PF, not camera sentinel.
            self.assertEqual((args[17], args[18]), (expected['height'], expected['width']))
            for actual, key in ((args[15], 'tanx'), (args[16], 'tany')):
                self.assertLessEqual(abs(actual - expected[key]), 32 * sys.float_info.epsilon)
            self.np.testing.assert_allclose(args[13].numpy(), expected['world'].astype('float32'), rtol=1e-6, atol=0)
            self.np.testing.assert_allclose(args[14].numpy(), expected['full'].astype('float32'), rtol=4e-6, atol=0)
            # Python -> C++ float representation, NOT actual CUDA execution.
            for size, tan, key in ((args[18], args[15], 'fx'), (args[17], args[16], 'fy')):
                self.assertAlmostEqual(size / (2 * float(self.np.float32(tan))), expected[key], delta=8*self.np.finfo('float32').eps*expected[key])
            self.assertTrue(all(a.device.type == 'cpu' for a in args if isinstance(a, torch.Tensor)))
            observed['forward'].append(args)
            n, h, w = len(args[1]), args[17], args[18]
            buffer = lambda: torch.empty(0, dtype=torch.uint8)
            return (n, torch.zeros(3, h, w), torch.zeros(2, h, w), torch.zeros(1, h, w),
                    torch.ones(1, h, w), torch.ones(n, dtype=torch.int32), buffer(), buffer(), buffer(),
                    torch.zeros(n, 6), args[1].detach().clone(), torch.empty(0), torch.empty(0))
        def backward(*args):
            fw = observed['forward'][-1]
            self.assertIs(type(args[14]), float)
            self.assertEqual(args[14], -1.0)
            self.assertIs(args[15], fw[13])
            self.assertIs(args[16], fw[14])
            self.assertEqual(args[17:19], fw[15:17])
            self.assertEqual(tuple(args[19].shape), (3, expected['height'], expected['width']))
            observed['backward'].append(args)
            return tuple(torch.zeros_like(args[i]) for i in (1, 4, 6, 1, 13, 23, 5, 7, 8, 9, 10, 11))
        boundary = types.SimpleNamespace(rasterize_gaussians=forward, rasterize_gaussians_backward=backward)
        with patch.object(self.binding, '_C', boundary), patch.object(
                self.binding._RasterizeGaussians, 'forward', staticmethod(forward_context)):
            yield observed

    def render(self, p, camera, **kwargs):
        return self.f.renderer.render(camera, p.gaussians, p.pipe, p.background,
                                      formal_config=p.inputs.config, **kwargs)

    def test_actual_train_test_loader_camera_render_autograd_at_scales(self):
        for divisor, fov in ((1, False), (2, False), (5, False), (1, True), (2, True)):
            with self.subTest(divisor=divisor, fov=fov):
                # Restore the raw fixture for the next parameter case.
                for split in ('train', 'test'):
                    root = self.metadata(split)
                    root.pop('camera_angle_x', None)
                    root.update(fl_x=15, fl_y=25, cx=10, cy=5)
                    self.save_metadata(split, root)
                self.configure(divisor, fov)
                p = self.f.prepare()
                for split, cameras in (('train', p.scene.train_cameras[1.0]), ('test', p.scene.test_cameras[1.0])):
                    expected = self.oracle(divisor, fov, test_split=split == 'test')
                    camera = next(c for c in cameras if c.formal_frame.index == 0)
                    self.assertEqual(camera.formal_camera.frame_key[0], split)
                    self.check_camera(camera, expected)
                    copied = camera.cuda()  # Real deepcopy + isolated transfer only.
                    self.assertIsNot(copied, camera)
                    self.assertEqual(copied.formal_camera, camera.formal_camera)
                    self.assertIs(copied.formal_binding.camera, copied)
                    self.assertIsNot(copied.full_proj_transform, camera.full_proj_transform)
                    self.check_camera(copied, expected)
                    with self.observers(expected) as calls:
                        result = self.render(p, copied)
                        ctx = calls['contexts'][0]
                        self.assertIs(ctx.raster_settings, ctx.forward_settings)
                        self.assertIs(ctx.raster_settings.camera_binding, copied.formal_binding)
                        result['render'].sum().backward()
                    self.assertEqual((len(calls['forward']), len(calls['backward'])), (1, 1))
                self.assertFalse(self.f.output.parent.exists())
                self.assertFalse(self.torch.cuda.is_initialized())

    def test_raw_camera_failures_before_heavy_import_and_output(self):
        self.f.verify()  # First establish a valid real snapshot/reference.
        original = self.metadata('train')
        cases = [lambda r: r['frames'][0].update(fl_x=15),
                 lambda r: r.update(fl_y=None),
                 lambda r: r.update(camera_angle_x=1.),
                 lambda r: r.update(cx=math.nextafter(10., 11.)),
                 lambda r: r['frames'][0].update(w=20),
                 lambda r: r.update(fl_x=True),
                 lambda r: r.update(fl_x=0),
                 lambda r: r.update(camera_angle_y=1.)]
        code = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
class Deny:
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('torch','numpy','PIL','scene','train','gaussian_renderer'):
   raise AssertionError('heavy import reached')
sys.meta_path.insert(0,Deny())
import formal_entry,formal_inputs
formal_inputs._reference=lambda:json.loads(Path(sys.argv[2]).read_bytes())
try: formal_entry.run(Path(sys.argv[3]))
except ValueError as e:
 assert str(e).startswith('formal_camera_'),str(e)
else: raise AssertionError('invalid camera accepted')
assert not Path(sys.argv[4]).exists()
print('camera rejected before heavy import/output')
"""
        ref, config = self.f.base / 'ref.json', self.f.base / 'config.json'
        config.write_bytes(fixtures.encode(self.f.value))
        for change in cases:
            root = json.loads(json.dumps(original))
            change(root)
            self.save_metadata('train', root)
            ref.write_text(json.dumps(self.f.ref))
            result = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', code,
                str(fixtures.ROOT), str(ref), str(config), str(self.f.output.parent)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.f.seed.assert_not_called()

    def test_frame_tuple_length_frame_key_and_loader_authority(self):
        bundle = self.f.verify()
        self.f.prepare(bundle)
        for bad in (bundle._replace(train_cameras=bundle.train_cameras[:-1]),
                    bundle._replace(train_cameras=(bundle.test_cameras[0],) + bundle.train_cameras[1:])):
            with self.assertRaisesRegex(ValueError, 'formal_(frame_count|camera_frame)'):
                self.f.prepare(bad)
        infos = self.f.readers.readCamerasFromTransforms(str(self.f.source), 'transforms_train.json', False,
                    extension='', dataloader=True, formal_metadata=(bundle.train_bytes, bundle.train_frames),
                    formal_cameras=bundle.train_cameras)
        from utils.camera_utils import loadCam
        args = self.f.legacy_args(self.f.output)
        for info in (infos[0]._replace(formal_camera=None), infos[0]._replace(formal_frame=bundle.test_frames[0]),
                     infos[0]._replace(timestamp=3), infos[0]._replace(uid=8)):
            with self.assertRaises(ValueError): loadCam(args, 0, info, 1., formal_resolution=bundle.config.dataset.resolution)
        with self.assertRaises(ValueError): loadCam(args, 0, infos[0], 1.)
        with self.assertRaises(ValueError):
            loadCam(args, 0, infos[0], 1., formal_resolution=bundle.config.dataset.resolution._replace(divisor=2))

    def test_camera_mutations_rejected_before_transfer_or_render_allocation(self):
        self.configure()
        p = self.f.prepare()
        source = next(c for c in p.scene.train_cameras[1.0] if c.formal_frame.index == 0)
        mutations = [lambda c: setattr(c, 'FoVx', -1.),
                     lambda c: setattr(c, 'fl_x', c.fl_x / 2),
                     lambda c: setattr(c, 'timestamp', 99),
                     lambda c: setattr(c, 'formal_camera', None),
                     lambda c: setattr(c, 'formal_binding', None),
                     lambda c: setattr(c, 'formal_camera', p.inputs.test_cameras[0]),
                     lambda c: setattr(c, 'projection_matrix', c.projection_matrix.clone()),
                     lambda c: c.full_proj_transform.add_(1)]
        for mutate in mutations:
            camera = source.cuda()
            mutate(camera)
            with patch.object(self.torch.Tensor, 'to', side_effect=AssertionError('transfer before guard')):
                with self.assertRaises(ValueError): camera.cuda()
            with patch.object(self.torch, 'zeros_like', side_effect=AssertionError('allocation before guard')):
                with self.assertRaises(ValueError): self.render(p, camera)
        with patch.object(self.torch, 'zeros_like', side_effect=AssertionError('allocation before guard')):
            with self.assertRaises(ValueError): self.render(p, types.SimpleNamespace())
            with self.assertRaises(ValueError):
                self.f.renderer.render(source, p.gaussians, p.pipe, p.background)

    def test_settings_and_backward_context_mutations_reject_before_c(self):
        self.configure()
        p = self.f.prepare()
        source = next(c for c in p.scene.train_cameras[1.0] if c.formal_frame.index == 0)
        expected = self.oracle()
        # Get genuine settings from a valid forward first; then drive the actual
        # nn.Module/autograd forward with exactly one changed settings field.
        with self.observers(expected) as observed:
            self.render(p, source)
        good = observed['contexts'][0].raster_settings
        rasterizer = self.f.renderer.GaussianRasterizer
        for changes in ({'tanfovx': math.tan(-.5)}, {'tanfovx': abs(math.tan(-.5))},
                        {'tanfovx': good.tanfovy}, {'image_width': good.image_width // 2},
                        {'projmatrix': good.projmatrix.clone()}, {'camera_binding': None}, {'formal_camera': False}):
            with patch.object(self.binding, '_C', types.SimpleNamespace(
                    rasterize_gaussians=lambda *a: self.fail('C reached for wrong settings'))):
                with self.assertRaises(ValueError):
                    rasterizer(good._replace(**changes))(
                        means3D=p.gaussians.get_xyz, means2D=self.torch.zeros_like(p.gaussians.get_xyz),
                        opacities=p.gaussians.get_opacity, shs=p.gaussians.get_features,
                        scales=p.gaussians.get_scaling, rotations=p.gaussians.get_rotation,
                        scales_t=p.gaussians.get_scaling_t, ts=p.gaussians.get_t,
                        rotations_r=p.gaussians.get_rotation_r, prefilter_var=-1.)
        for kind in ('replace_matrix', 'inplace_matrix', 'settings', 'gradient'):
            camera = source.cuda()
            with self.subTest(kind=kind), self.observers(expected) as calls:
                result = self.render(p, camera)
                ctx = calls['contexts'][0]
                if kind == 'replace_matrix': camera.full_proj_transform = camera.full_proj_transform.clone()
                if kind == 'inplace_matrix': camera.projection_matrix.add_(1)
                if kind == 'settings': ctx.raster_settings = ctx.raster_settings._replace(tanfovx=.1)
                with self.assertRaisesRegex(ValueError, 'formal_camera_'):
                    if kind == 'gradient':
                        # Autograd itself rejects a wrong external shape before
                        # invoking backward; call the actual backward body to
                        # additionally exercise its local C-boundary guard.
                        self.binding._RasterizeGaussians.backward(ctx, self.torch.zeros(3, 1, 1), *([None]*7))
                    else:
                        result['render'].sum().backward()
                self.assertEqual(len(calls['backward']), 0)

    def test_cpu_projection_dtype_independent_of_default_and_legacy_raw_diagnostic(self):
        self.configure()
        p = self.f.prepare()
        c = next(c for c in p.scene.train_cameras[1.0] if c.formal_frame.index == 0)
        from scene.cameras import Camera
        kwargs = dict(colmap_id=c.colmap_id, R=c.R, T=c.T, FoVx=c.FoVx, FoVy=c.FoVy,
                      image=c.image, gt_alpha_mask=c.gt_alpha_mask, image_name=c.image_name, uid=c.uid,
                      cx=c.cx, cy=c.cy, fl_x=c.fl_x, fl_y=c.fl_y, resolution=c.resolution,
                      image_path=c.image_path, timestamp=c.timestamp, meta_only=True,
                      formal_camera=c.formal_camera, formal_frame=c.formal_frame)
        old = self.torch.get_default_dtype()
        try:
            self.torch.set_default_dtype(self.torch.float64)
            actual = Camera(**kwargs)
        finally:
            self.torch.set_default_dtype(old)
        self.check_camera(actual, self.oracle())
        self.assertEqual(actual.projection_matrix.dtype, self.torch.float32)
        self.assertTrue(c.formal_camera.raw.reader_sentinel)
        # Diagnostic remains historical; formal execution never consumes it.
        self.assertGreater(c.FoVx, 0)
        self.assertGreater(c.formal_camera.tanx, 0)
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__ == '__main__':
    print('test executable:', sys.executable)
    unittest.main()
