"""Step 11 CPU acceptance. Actual CPU dependencies, GPU boundary isolated."""
import ast
from contextlib import contextmanager
import math
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import formal_time_handoff as handoff


class RepresentationTests(unittest.TestCase):
    def test_stdlib_no_heavy_import(self):
        code = """
import sys
sys.path.insert(0,sys.argv[1])
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','numpy','PIL','scene','gaussian_renderer'):
   raise AssertionError('heavy import')
sys.meta_path.insert(0,Deny())
import formal_time_handoff
print(sys.executable)
"""
        p = subprocess.run([sys.executable, '-B', '-S', '-c', code, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr); self.assertEqual(p.stdout.strip(), sys.executable)

    def test_interval_duration_and_cast_rejections(self):
        for x in (float('inf'), float('nan'), 1e40, 1e-50):
            with self.assertRaises(ValueError): handoff.binary32(x)
        self.assertGreater(handoff.binary32(2**-149), 0)  # Not a blanket subnormal ban.
        for endpoints, duration in (((1., 1.+2**-30), 2**-30), ((0., 1.), 0.), ((0., 1.), 1e-50)):
            t = types.SimpleNamespace(effective_interval=endpoints, effective_duration=duration, log_effective_scale=0.)
            with self.assertRaises(ValueError): handoff.validate_time(t)

    def test_class_order_duplicates_and_source_precision(self):
        handoff.validate_classes([0., .2, .2, 1.], [0., .1, .1, .5])
        with self.assertRaises(ValueError): handoff.validate_classes([1., 1.+2**-30], [1., 1.+2**-30])
        with self.assertRaises(ValueError): handoff.validate_classes([1., 2.], [2., 1.])
        # No cross-source raw-bit equality: PLY widening is intentionally not JSON .2.
        ply_raw = handoff.binary32(.2)
        self.assertNotEqual(ply_raw, .2)
        handoff.validate_classes([ply_raw], [ply_raw/2.5])
        handoff.validate_classes([.2], [.2/2.5])

    def test_native_static_contract_and_abi(self):
        native = ROOT/'diff-gaussian-rasterization'
        for name in ('forward.cu', 'backward.cu'):
            text = (native/'cuda_rasterizer'/name).read_text()
            self.assertIn('mask = marginal_t > 0.05;', text)
            self.assertNotIn('marginal_t > 0.05f', text)
            self.assertIn('atomicOr(temporal_status', text)
            self.assertIn('cov_t <= 0', text)
            self.assertIn('0x7fffffffU', text)
        impl = (native/'cuda_rasterizer/rasterizer_impl.cu').read_text()
        self.assertLess(impl.index('formal_time_forward_health'), impl.index('// Compute prefix sum'))
        self.assertIn('formal_time_backward_health', impl)
        self.assertIn('const int debug_preprocess_stride = 104;', (native/'rasterize_points.cu').read_text())


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import time_conditioning_oracle as o
        cls.torch, cls.o = torch, o
        cls.cases = o.cases()
        cls.guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden'))
        cls.guard.start(); cls.addClassCleanup(cls.guard.stop)

    def close(self, a, b, tol='analytic'):
        self.torch.testing.assert_close(a, b, atol=self.o.TOLERANCES[tol][0], rtol=self.o.TOLERANCES[tol][1])

    def test_conditional_analytic_vjp_full_chain(self):
        for case in self.cases[:13]:
            p, s = case['points'][0], case['camera']
            expected = self.o.sh.gradients(self.o.conditioning_loss(p, s), p)
            actual = self.o.analytic_conditioning_vjp(p, s)
            for name in self.o.PARAMETERS:
                with self.subTest(case=case['label'], name=name): self.close(actual[name], expected[name])

    def test_same_branch_two_width_cpu_fd_and_cuda_endpoint_domains(self):
        count = 0
        for case in self.cases[:13]:
            baseline = self.o.domain(case)
            gradients = self.o.gradients(case)
            for coordinate in self.o.coordinates(case):
                i, name, offset = coordinate
                for step in self.o.CPU_STEPS + self.o.CUDA_STEPS:
                    plus, minus, span = self.o.camera.perturb(case['points'], coordinate, step,
                                                            rounded=step in self.o.CUDA_STEPS)
                    self.assertEqual(self.o.domain(case, plus), baseline)
                    self.assertEqual(self.o.domain(case, minus), baseline)
                    if step in self.o.CPU_STEPS:
                        fd = (self.o.loss(case, plus)-self.o.loss(case, minus))/span
                        with self.subTest(case=case['label'], coordinate=coordinate, step=step):
                            self.close(fd, gradients[i][name].flatten()[offset], 'cpu_fd')
                    count += 2
        print('time FD endpoint checks:', count)

    def test_initial_identity_anchor_and_cull_gradients(self):
        case = self.cases[0]; p = case['points'][0]
        mean, cov, m, sigma = self.o.sh.conditional(p, case['camera'])
        self.assertEqual(float(m), 1.)
        self.close(mean, p['mean']); self.close(cov, sigma[:3, :3])
        case = self.cases[-1]; gradients = self.o.gradients(case)
        for value in gradients[1].values(): self.assertEqual(int(value.count_nonzero()), 0)
        single = dict(case, points=case['points'][:1])
        self.close(self.o.loss(case), self.o.loss(single))

    def test_threshold_marker_and_bounded_grid(self):
        record = self.o.threshold()
        self.assertFalse(record['exact_equality_reachable'])
        self.assertGreater(record['marker'], .05)
        self.assertEqual(len(list(self.o.boundary_grid())), 4096)
        self.assertFalse(self.torch.cuda.is_initialized())


class ConsumerTests(unittest.TestCase):
    # Native forward / backward / saved ctx / returned VJP indices, respectively.
    # These are boundary tests, not substitutes for CUDA gradient acceptance.
    TIME_SLOTS = {
        'means3D': (1, 1, 1, 3), 'ts': (5, 7, 10, 7),
        'scales': (6, 8, 3, 8), 'scales_t': (7, 9, 11, 9),
        'rotations': (8, 10, 4, 10), 'rotations_r': (9, 11, 12, 11),
    }

    @classmethod
    def setUpClass(cls):
        import test_formal_camera_connection as camera
        cls.camera = camera.CameraConnectionTests
        cls.camera.setUpClass(); cls.addClassCleanup(cls.camera.doClassCleanups)

    def setUp(self):
        self.c = self.camera('runTest'); self.c.setUp(); self.addCleanup(self.c.doCleanups)

    @contextmanager
    def known_vjp_observers(self, expected):
        """Reuse the real binding/ctx and camera observer, changing only C VJP.

        The row/column-distinct VJP is prescribed, NOT evaluated CUDA math.
        """
        c, torch = self.c, self.c.torch
        with c.observers(expected) as calls:
            original = c.binding._C.rasterize_gaussians_backward
            def backward(*args):
                gradients = list(original(*args))  # Existing ABI/camera checks.
                calls['known_vjp'] = {}
                for ordinal, (name, (fw, bw, _, returned)) in enumerate(self.TIME_SLOTS.items()):
                    self.assertIs(args[bw], calls['forward'][-1][fw])
                    value = args[bw]
                    weight = torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape)/32 + ordinal+1
                    calls['known_vjp'][name] = weight
                    gradients[returned] = weight
                return tuple(gradients)
            with patch.object(c.binding._C, 'rasterize_gaussians_backward', side_effect=backward):
                yield calls

    def check_saved_buffers(self, calls):
        self.assertEqual(len(calls['forward']), 1)
        ctx = calls['contexts'][0]
        for fw, _, saved, _ in self.TIME_SLOTS.values():
            self.assertIs(ctx.saved_tensors[saved], calls['forward'][0][fw])
            self.assertTrue(ctx.saved_tensors[saved].is_contiguous())

    def check_leaf_vjp(self, leaves, weights):
        """Independent exp/normalization Jacobians in NumPy float64."""
        torch, np = self.c.torch, self.c.np
        for name, leaf in leaves.items():
            self.assertIsNotNone(leaf.grad, name)
            x = leaf.detach().numpy().astype('float64')
            w = weights[name].numpy().astype('float64')
            if name in ('scales', 'scales_t'):
                expected = w*np.exp(x)
            elif name in ('rotations', 'rotations_r'):
                norm = np.sqrt(np.sum(x*x, axis=1, keepdims=True))
                expected = w/norm - x*np.sum(w*x, axis=1, keepdims=True)/(norm**3)
            else:
                expected = w
            expected = torch.tensor(expected, dtype=leaf.dtype)
            if name in ('means3D', 'ts'):
                torch.testing.assert_close(leaf.grad, expected, atol=0, rtol=0)
            else:
                eps = 16*torch.finfo(leaf.dtype).eps
                torch.testing.assert_close(leaf.grad, expected, atol=eps, rtol=eps)

    def binding_fixture(self):
        """Genuine settings/consumer args; isolate only GPU/C with existing seam."""
        c, torch = self.c, self.c.torch
        c.configure(); p = c.f.prepare()
        camera = next(cam for cam in p.scene.train_cameras[1.0] if cam.formal_frame.index == 0).cuda()
        expected = c.oracle()
        with c.observers(expected) as calls:
            c.render(p, camera)
        settings = calls['contexts'][0].raster_settings
        model = p.gaussians
        args = dict(means3D=model.get_xyz, means2D=torch.zeros_like(model.get_xyz),
            sh=model.get_features, colors_precomp=torch.empty(0), flow_2d=torch.zeros(len(model.get_xyz),2),
            opacities=model.get_opacity, ts=model.get_t, scales=model.get_scaling,
            scales_t=model.get_scaling_t, rotations=model.get_rotation, rotations_r=model.get_rotation_r,
            cov3Ds_precomp=torch.empty(0), prefilter_var=-1., raster_settings=settings)
        return args, expected

    def strided(self, value):
        # Works for N x 1 too: a size-one last dimension alone would not prove
        # noncontiguity. Keep an interleaved unused column, without detach.
        return self.c.torch.stack((value, self.c.torch.zeros_like(value)), -1)[...,0]

    def test_real_all_rows_and_sampled_layout_handoff_and_vjp(self):
        import time_conditioning_runtime as runtime
        from formal_views import ConsumerView
        from utils.sh_utils import SH2RGB
        torch, np = self.c.torch, self.c.np
        projection = np.zeros((4,4))
        projection[0,0], projection[1,1] = 1.5, 5/3
        projection[2,2], projection[2,3], projection[3,2] = 100/99.99, -1/99.99, 1
        expected_camera = dict(width=64, height=48, fx=48., fy=40.,
                               tanx=64/(2*48), tany=48/(2*40), world=np.eye(4), full=projection.T)
        for sampled in (False, True):
            with self.subTest(sampled=sampled), tempfile.TemporaryDirectory() as temp:
                inputs, model, scene, rows, indices = runtime.initialize(Path(temp), 1., sampled)
                leaves = dict(means3D=model._xyz, ts=model._t, scales=model._scaling,
                    scales_t=model._scaling_t, rotations=model._rotation, rotations_r=model._rotation_r)
                before = {k:(v, v.stride(), v.detach().clone(), v._version) for k,v in leaves.items()}
                if not sampled:
                    self.assertEqual(model._xyz.stride(), (1,16))
                    self.assertFalse(model._xyz.is_contiguous())
                    np.testing.assert_array_equal(indices, np.arange(16))
                else:
                    draws = np.random.RandomState(inputs.config.initialization.seed).randint(0,16,12)
                    np.testing.assert_array_equal(indices, draws)
                    self.assertLess(len(set(indices.tolist())), len(indices))
                xyz = np.stack([rows[k][indices] for k in ('x','y','z')], 1)
                rgb = np.stack([rows[k][indices] for k in ('red','green','blue')], 1)/255
                np.testing.assert_array_equal(model._xyz.detach().numpy(), xyz)
                np.testing.assert_array_equal(model._t.detach().numpy().ravel(), rows['time'][indices])
                np.testing.assert_allclose(SH2RGB(model._features_dc[:,0]).detach().numpy(), rgb, atol=1e-7, rtol=1e-6)
                camera = scene.getTrainCameras()[0][1].cuda()
                with self.known_vjp_observers(expected_camera) as calls:
                    result = self.c.f.renderer.render(camera, model, ConsumerView(inputs.config,'pipeline'),
                        torch.zeros(3), formal_config=inputs.config)
                    fw = calls['forward'][0]
                    np.testing.assert_array_equal(fw[1].detach().numpy(), xyz)
                    np.testing.assert_array_equal(fw[5].detach().numpy().ravel(), rows['time'][indices])
                    if not sampled: self.assertIsNot(fw[1], model._xyz)
                    self.check_saved_buffers(calls)
                    result['render'].sum().backward()
                    self.check_leaf_vjp(leaves, calls['known_vjp'])
                for name, (original, stride, value, version) in before.items():
                    attribute = {'means3D':'_xyz', 'ts':'_t', 'scales':'_scaling',
                        'scales_t':'_scaling_t', 'rotations':'_rotation', 'rotations_r':'_rotation_r'}[name]
                    self.assertIs(getattr(model,attribute), original)
                    self.assertEqual(original.stride(), stride)
                    self.assertEqual(original._version, version)
                    torch.testing.assert_close(original, value, atol=0, rtol=0)
                self.assertFalse((Path(temp)/'unused-output').exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_six_strided_tensors_preserve_copy_autograd_and_context(self):
        torch = self.c.torch
        args, expected = self.binding_fixture()
        leaves = {}
        for ordinal, (name, columns) in enumerate(zip(self.TIME_SLOTS,(3,1,3,1,4,4))):
            values = torch.arange(4*columns, dtype=torch.float32).reshape(4,columns)/16 + (ordinal+1)/8
            leaf = torch.nn.Parameter(self.strided(values))
            leaves[name] = leaf
            if name in ('scales','scales_t'):
                value = self.strided(torch.exp(leaf))
            elif name in ('rotations','rotations_r'):
                value = self.strided(torch.nn.functional.normalize(leaf, dim=1))
            else:
                value = leaf
            self.assertFalse(value.is_contiguous(), name)
            args[name] = value
        snapshots = {name:(t.stride(), t.detach().clone(), t._version) for name,t in leaves.items()}
        with self.known_vjp_observers(expected) as calls:
            output = self.c.binding.rasterize_gaussians(**args)
            for name,(fw,_,_,_) in self.TIME_SLOTS.items():
                actual, original = calls['forward'][0][fw], args[name]
                self.assertIsNot(actual, original)
                self.assertEqual((actual.shape,actual.dtype,actual.device), (original.shape,original.dtype,original.device))
                torch.testing.assert_close(actual, original, atol=0, rtol=0)
            self.check_saved_buffers(calls)
            output[0].sum().backward()
            self.check_leaf_vjp(leaves, calls['known_vjp'])
        for name,leaf in leaves.items():
            stride,value,version = snapshots[name]
            self.assertEqual(leaf.stride(),stride); self.assertEqual(leaf._version,version)
            torch.testing.assert_close(leaf,value,atol=0,rtol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_contiguous_inputs_are_noop_through_forward_and_saved_context(self):
        torch = self.c.torch
        args, expected = self.binding_fixture()
        for name in self.TIME_SLOTS: args[name] = args[name].contiguous()
        with self.known_vjp_observers(expected) as calls:
            out = self.c.binding.rasterize_gaussians(**args)
            for name,(fw,_,_,_) in self.TIME_SLOTS.items():
                self.assertIs(calls['forward'][0][fw], args[name])
            self.check_saved_buffers(calls)
            out[0].sum().backward()
        self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_attributes_not_cast_transferred_or_reshaped(self):
        # CPU metadata observation, NOT execution of native TORCH_CHECK.
        torch = self.c.torch
        args,_ = self.binding_fixture()
        class NativeObservationStop(Exception): pass
        for name,(fw,_,_,_) in self.TIME_SLOTS.items():
            good = args[name]
            invalids = dict(dtype=good.double(), rows=good[:1], columns=torch.zeros(len(good),good.shape[1]+1),
                            rank=good.unsqueeze(-1), meta_device=good.detach().to('meta'), cpu_device=good)
            for kind,value in invalids.items():
                with self.subTest(name=name,kind=kind):
                    value = self.strided(value)
                    seen = []
                    def reject(*native):
                        actual = native[fw]; seen.append(actual)
                        self.assertEqual((actual.shape,actual.dtype,actual.device),(value.shape,value.dtype,value.device))
                        if kind != 'meta_device':
                            torch.testing.assert_close(actual,value,atol=0,rtol=0)
                        if kind == 'cpu_device': self.assertFalse(actual.is_cuda)
                        raise NativeObservationStop('metadata retained for native rejection')
                    with patch.object(self.c.binding,'_C',types.SimpleNamespace(rasterize_gaussians=reject)):
                        with self.assertRaises(NativeObservationStop):
                            self.c.binding.rasterize_gaussians(**dict(args,**{name:value}))
                    self.assertEqual(len(seen),1)
        self.assertFalse(torch.cuda.is_initialized())

    def test_legacy_3d_and_4d_do_not_normalize_layout(self):
        torch = self.c.torch
        args,expected = self.binding_fixture()
        for dim in (3,4):
            with self.subTest(dim=dim):
                legacy = dict(args)
                for name in self.TIME_SLOTS: legacy[name] = self.strided(args[name])
                if dim == 3:
                    for name in ('ts','scales_t','rotations_r'): legacy[name] = torch.empty(0)
                legacy['raster_settings'] = args['raster_settings']._replace(formal_time=False, time_binding=None,
                    formal_camera=False, camera_binding=None, gaussian_dim=dim, rot_4d=dim==4)
                with self.c.observers(expected) as calls:
                    self.c.binding.rasterize_gaussians(**legacy)
                    for name,(fw,_,_,_) in self.TIME_SLOTS.items():
                        self.assertIs(calls['forward'][0][fw], legacy[name])
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_loader_scene_model_rows_and_initialization(self):
        import time_conditioning_runtime as runtime
        import numpy as np
        torch = self.c.torch
        for d in (1., 2., 2.5):
            for sampled in (False, True):
                with tempfile.TemporaryDirectory() as temp:
                    inputs, model, scene, rows, indices = runtime.initialize(Path(temp), d, sampled)
                    self.assertIs(model._formal_time, inputs.config.time_derivation)
                    expected = np.stack([rows[k][indices] for k in ('x', 'y', 'z')], axis=1)
                    np.testing.assert_array_equal(model.get_xyz.detach().numpy(), expected)
                    np.testing.assert_array_equal(model.get_t.detach().numpy().ravel(),
                                                  (rows['time'][indices].astype('float64')/d).astype('float32'))
                    expected_log = handoff.binary32(math.log(math.sqrt(8/5)/d))
                    self.assertTrue(torch.equal(model._scaling_t, torch.full_like(model._scaling_t, expected_log)))
                    torch.testing.assert_close(model.get_scaling_t.square(),
                            torch.full_like(model._scaling_t, 8/(5*d*d)), atol=2e-7, rtol=5e-6)
                    identity = torch.tensor([1., 0, 0, 0]).expand(len(indices), 4)
                    self.assertTrue(torch.equal(model.get_rotation, identity))
                    self.assertTrue(torch.equal(model.get_rotation_r, identity))
                    self.assertEqual((model.active_sh_degree, model.active_sh_degree_t), (0, 0))
                    # Decode real SH DC to verify the same sampled RGB rows.
                    from utils.sh_utils import SH2RGB
                    rgb = SH2RGB(model._features_dc[:, 0]).detach().numpy()
                    np.testing.assert_allclose(rgb, np.stack([rows[k][indices] for k in ('red','green','blue')], 1)/255,
                                               atol=1e-7, rtol=1e-6)
                    for cameras in (scene.getTrainCameras(), scene.getTestCameras()):
                        for j in range(len(cameras)):
                            _, cam = cameras[j]
                            binding = handoff.bind_time(inputs.config, model, cam.formal_binding)
                            binding.validate()
                    self.assertFalse((Path(temp)/'unused-output').exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_initialization_six_domains_independent_sampling_order(self):
        import time_conditioning_runtime as runtime
        import time_conditioning_oracle as o
        torch = self.c.torch
        for d in (1.,2.,2.5):
            for sampled in (False,True):
                with self.subTest(divisor=d,sampled=sampled), tempfile.TemporaryDirectory() as temp:
                    inputs, model, scene, data, indices = runtime.initialize(Path(temp),d,sampled)
                    case = o.initialization_case(data,indices,inputs.config.time_derivation,
                                                 scene.getTrainCameras()[0][1].timestamp,'CPU-initial')
                    record = dict(seed=inputs.config.initialization.seed,row_indices=indices.tolist())
                    o.initialization_domain(case,record)
                    self.assertEqual(record['status'],'applicable')
                    self.assertEqual(record['seed'],23)
                    # Fixed independent row mapping, not a second call to sort.
                    expected = [0,1,4,10,2,5,3,11,8,7,6] if sampled else [0,1,2,3,5,6,7,8,9,10,12,13,14,15]
                    self.assertEqual(record['order'],expected)
                    culled = [v['gaussian_id'] for v in record['nodes'] if v['reason']=='time_cull']
                    self.assertEqual(culled,[9] if sampled else [4,11])
                    self.assertEqual(o.evaluate(case)[0]['active'],expected)
                    self.assertEqual(record['row_indices'],indices.tolist())
                    self.assertGreater(record['final_transmittance'],.5)
                    if sampled:
                        self.assertEqual(indices.tolist(),[3,6,8,9,6,8,15,13,12,11,7,10])
                        gradients = o.gradients(case)
                        for a,b in ((1,4),(2,5)):
                            self.assertIsNot(case['points'][a]['features'],case['points'][b]['features'])
                            self.assertGreater(float(gradients[a]['features'][0,0]),float(gradients[b]['features'][0,0]))
                        for value in gradients[9].values(): self.assertEqual(int(value.count_nonzero()),0)
                    print('INITIAL_DOMAIN',json.dumps({k:record[k] for k in ('seed','row_indices','order','final_transmittance')}))
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_forward_backward_binding_and_mutation_rejection(self):
        c = self.c; c.configure(); p = c.f.prepare()
        cameras = p.scene.getTrainCameras()
        camera = next(cameras[i][1] for i in range(len(cameras))
                      if cameras[i][1].formal_frame.index == 0).cuda()
        with c.observers(c.oracle()) as observed:
            out = c.render(p, camera); out['render'].sum().backward()
            self.assertTrue(observed['forward'][0][-1]); self.assertTrue(observed['backward'][0][-3])
            for bad in (0., -2., 1., -1, True, float('nan')):
                p.gaussians.prefilter_var = bad
                with self.assertRaises(ValueError): c.render(p, camera)
            p.gaussians.prefilter_var = -1.0
            time = p.inputs.config.time_derivation
            for attribute, bad in (('time_duration', (0., 1.)), ('_formal_time', type(time)(*time))):
                original = getattr(p.gaussians, attribute); setattr(p.gaussians, attribute, bad)
                with self.assertRaises(ValueError): c.render(p, camera)
                setattr(p.gaussians, attribute, original)
            # Learned state updates are not forbidden by binding metadata.
            with c.torch.no_grad(): p.gaussians._t.add_(.1); p.gaussians._scaling_t.add_(.01)
            out = c.render(p, camera)
            ctx = observed['contexts'][-1]
            ctx.prefilter_var = 0.
            n = len(observed['backward'])
            with self.assertRaises(ValueError): out['render'].sum().backward()
            self.assertEqual(len(observed['backward']), n)

    def test_real_PLY_class_collapse_rejected_before_sampling(self):
        f = self.c.f; np = f.np
        f.value['dataset']['time']['divisor'] = 2.5
        f.value['initialization']['num_pts'] = 1
        loaded = f.PlyData.read(io.BytesIO((f.source/'points3d.ply').read_bytes()))
        # Adjacent raw f32 classes around 1.5 collapse after /2.5 and f32 cast.
        loaded['vertex']['time'][:] = [1.5, np.nextafter(np.float32(1.5), np.float32(2)), 2., 6.]
        stream = io.BytesIO(); loaded.write(stream); f.store_input('points3d.ply',stream.getvalue())
        with patch.object(np.random,'randint',side_effect=AssertionError('sampling before validation')) as sample:
            with self.assertRaisesRegex(ValueError,'formal_time_class_collapse'): f.prepare()
            sample.assert_not_called()


class InitializationOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import time_conditioning_oracle as o
        cls.torch,cls.o = torch,o
        spec = importlib.util.spec_from_file_location('step11_entry_cpu',ROOT/'tests/cuda/time_conditioning_cuda_acceptance.py')
        cls.entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.entry)
        guard = patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden'))
        guard.start(); cls.addClassCleanup(guard.stop)

    def fixture(self, depths=(4.,3.)):
        t,o = self.torch,self.o
        s = o.camera.camera('order-test',width=64,height=48,fx=48.,fy=40.,identity=True)
        s.update(timestamp=0.,duration=8.)
        points=[]
        for i,z in enumerate(depths):
            p=o.sh.fixture()
            with t.no_grad():
                # Away from the shared Legendre reference's polar singularity.
                # Depth disorder, opacity/color differences and leaves persist.
                p['mean'].copy_(o.camera.tensor([.04,.03,z])); p['time'].zero_()
                p['log_scale'].fill_(math.log(.15)); p['log_scale'][3]=0.
                p['left'].copy_(o.camera.tensor([1.,0,0,0])); p['right'].copy_(p['left'])
                p['opacity'].fill_(math.log((.2+.1*i)/(.8-.1*i)))
                p['features'].zero_(); p['features'][0]=o.camera.tensor([.3-i*.5,-.2+i*.4,.1])
            points.append(p)
        return dict(label='order-test',points=points,camera=s,pixels=[(32,24)],stage=(0,0),dim=4,
                    loss=(.6,-.3,.2,0.,0.,.1,.2))

    def manual(self, case, order):
        """Off-axis diagonal-3D covariance projected to a FULL 2x2 footprint.

        Identity pose/rotations, zero time offset and DC-only color are fixture
        premises, not renderer extensions. No oracle/renderer helpers are used.
        The +binary32(1e-7) belongs to homogeneous projection, NOT footprint J.
        """
        t=self.torch; trans=t.tensor(1.,dtype=t.float64); colors=[]; depths=[]
        s=case['camera']; px,py=case['pixels'][0]; fx,fy=s['fx'],s['fy']
        eps=struct.unpack('f',struct.pack('f',1e-7))[0]
        self.assertEqual(case['stage'],(0,0)); self.assertEqual(case['dim'],4)
        self.assertTrue(t.equal(s['rotation'],t.eye(3,dtype=t.float64)))
        self.assertEqual(int(s['center'].count_nonzero()),0)
        for i in order:
            p=case['points'][i]; x,y,z=p['mean'].unbind()
            self.assertEqual(float(p['time'][0]),s['timestamp'])
            for name in ('left','right'):
                self.assertTrue(t.equal(p[name],t.tensor([1.,0,0,0],dtype=t.float64)))
            self.assertEqual(int(p['features'][1:].count_nonzero()),0)
            self.assertGreater(float((x*x+y*y)/(x*x+y*y+z*z)),1e-6)
            # Strictly interior to the existing footprint clamp, away from pole.
            self.assertLess(abs(float(x/z)),s['tanx'])
            self.assertLess(abs(float(y/z)),s['tany'])
            vx,vy,vz=p['log_scale'][:3].exp().square().unbind()
            cxx=fx*fx*(vx/z.square()+x*x*vz/z**4)+.3
            cyy=fy*fy*(vy/z.square()+y*y*vz/z**4)+.3
            cxy=fx*fy*x*y*vz/z**4
            det=cxx*cyy-cxy*cxy
            dx=fx*x/(z+eps)+(s['width']-1)/2-px
            dy=fy*y/(z+eps)+(s['height']-1)/2-py
            power=-.5*(cyy*dx*dx-2*cxy*dx*dy+cxx*dy*dy)/det
            a=p['opacity'].sigmoid()[0]*t.exp(power)
            self.assertGreater(float(det),0.)
            self.assertGreater(float(a),1/255); self.assertLess(float(a),.99)
            w=trans*a
            colors.append(w*(.5+math.sqrt(1/(4*math.pi))*p['features'][0]))
            depths.append(w*z); trans=trans*(1-a)
            self.assertGreater(float(trans),1e-4)
        rgb=sum(colors)+trans*t.tensor(case['camera']['background'],dtype=t.float64)
        return t.cat((rgb,t.zeros(2,dtype=t.float64),sum(depths).view(1),(1-trans).view(1)))

    def finite_close(self, actual, expected, *, atol=1e-12, rtol=1e-12):
        self.assertTrue(bool(self.torch.isfinite(actual).all()))
        self.assertTrue(bool(self.torch.isfinite(expected).all()))
        self.torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)

    def test_non_depth_order_values_and_original_leaf_gradients(self):
        t,o=self.torch,self.o; case=self.fixture(); originals=[dict(p) for p in case['points']]
        expected=self.manual(case,[1,0]); old=self.manual(case,[0,1]); actual=o.evaluate(case)[0]['outputs']
        self.assertTrue(bool(t.isfinite(old).all()))
        self.assertGreater(float((old-expected).abs().max()),.01)
        self.finite_close(actual,expected)
        leaves=[p[k] for p in case['points'] for k in ('features','opacity','mean')]
        ref=t.autograd.grad(expected@t.tensor(case['loss'],dtype=t.float64),leaves)
        gradients=o.gradients(case)
        for i,p in enumerate(case['points']):
            for k,v in originals[i].items(): self.assertIs(p[k],v)
            for j,k in enumerate(('features','opacity','mean')):
                self.finite_close(gradients[i][k],ref[3*i+j])
        print('ORDER_REFERENCE',json.dumps(dict(old_order_max_error=float((old-expected).abs().max()),
            output_max_error=float((actual-expected).abs().max()),
            original_row_mean_gradients=[g['mean'].tolist() for g in gradients],
            gradient_max_error=max(float((gradients[i][k]-ref[3*i+j]).abs().max())
                for i in range(2) for j,k in enumerate(('features','opacity','mean'))))))

    def test_tie_distinct_leaves_and_fixed_branch_feature_fd(self):
        t,o=self.torch,self.o; case=self.fixture((3.,3.))
        case['points'][1]=o.sh.clone(case['points'][0]); points=case['points']
        self.assertEqual(o.contributor_order(case),[0,1])
        g=o.gradients(case); manual=self.manual(case,[0,1])
        self.finite_close(o.evaluate(case)[0]['outputs'],manual)
        leaves=[p[k] for p in points for k in ('features','opacity','mean')]
        expected=t.autograd.grad(manual@t.tensor(case['loss'],dtype=t.float64),leaves)
        self.assertGreater(float(g[0]['features'][0,0]),float(g[1]['features'][0,0]))
        for i in (0,1):
            for j,k in enumerate(('features','opacity','mean')):
                self.assertIsNot(points[0][k],points[1][k])
                self.finite_close(g[i][k],expected[3*i+j])
            ends=[]
            for sign in (-1,1):
                ps=[o.sh.clone(p) for p in points]
                with t.no_grad(): ps[i]['features'][0,0]+=sign*1e-5
                self.assertEqual(o.contributor_order(case,ps),[0,1])
                self.assertEqual(o.domain(case,ps),o.domain(case))
                ends.append(o.loss(case,ps))
            self.finite_close((ends[1]-ends[0])/2e-5,g[i]['features'][0,0],atol=1e-10,rtol=1e-9)
        print('TIE_REFERENCE',json.dumps(dict(order=[0,1],independent_leaves=True,
            original_row_feature_gradients=[float(v['features'][0,0]) for v in g],
            gradient_max_error=max(float((g[i][k]-expected[3*i+j]).abs().max())
                for i in range(2) for j,k in enumerate(('features','opacity','mean'))))))

    def test_initial_domain_negative_reasons_and_evidence(self):
        t,o=self.torch,self.o
        def rejected(case,reason):
            evidence=dict(seed=23,row_indices=list(range(len(case['points']))))
            with self.assertRaisesRegex(AssertionError,reason): o.initialization_domain(case,evidence)
            self.assertEqual(evidence['status'],'rejected'); self.assertEqual(evidence['reason'],reason)
            self.assertTrue(evidence['nodes']); json.dumps(evidence)
        for z in (0.,-1.,float('nan'),float('inf')):
            case=self.fixture()
            with t.no_grad(): case['points'][0]['mean'][2]=z
            rejected(case,'order_nonpositive_or_nonfinite_depth')
        case=self.fixture((3.+1e-8,3.))
        rejected(case,'order_ambiguous_rounding_tie')
        case=self.fixture((o.camera.f32(.1),3.)); rejected(case,'initial_frustum')
        case=self.fixture()
        with t.no_grad(): case['points'][0]['mean'][0]=100.
        rejected(case,'initial_target_tile')
        case=self.fixture()
        with t.no_grad(): case['points'][0]['opacity'][0]=-12.
        rejected(case,'initial_alpha_domain')
        case=self.fixture(tuple([3.]*6))
        with t.no_grad():
            for p in case['points']: p['opacity'][0]=math.log(.85/.15)
        rejected(case,'initial_early_out_domain')
        case=self.fixture()
        with t.no_grad(): case['points'][0]['time'][0]=math.sqrt(-2*math.log(.05))
        rejected(case,'initial_time_boundary')

    def diagnostic_fixture(self):
        t,o=self.torch,self.o; case=self.fixture(); record=dict(label=case['label'],seed=23,row_indices=[8,3])
        domain=dict(row_indices=record['row_indices']); record['domain']=domain
        o.initialization_domain(case,domain)
        data=t.zeros((3,32),dtype=t.float64); h=data[0]
        # Independent native buffer layout for known [1,0] contributor sequence.
        for k,v in {0:1,1:64,2:48,3:32,4:24,5:32,6:24,7:2,8:1,9:6,10:5,11:7,
                    12:2,13:2,14:2,15:2,22:-1,23:1,25:2}.items(): h[k]=v
        h[16]=domain['final_transmittance']
        for j,i in enumerate((1,0)):
            p=domain['nodes'][i]; row=data[j+1]
            for k,v in {0:j,1:5+j,2:i,3:p['depth_f32'],15:p['power'],16:p['raw_alpha'],17:p['raw_alpha'],
                        19:p['trans_before'],20:p['trans_after'],21:p['trans_after'],28:1,29:j+1}.items(): row[k]=v
        radii=t.tensor([p['radius'] for p in domain['nodes']])
        return case,record,data,radii

    def test_diagnostic_rejects_missing_truncated_unreached_or_wrong_order(self):
        t=self.torch
        case,record,data,radii=self.diagnostic_fixture()
        self.entry.initialization_diagnostics(t,data,radii,record['domain'],record,[])
        self.assertEqual(record['contributor_diagnostic']['status'],'passed')
        for name in ('missing','truncated','unreached','order','reason','radius','key'):
            with self.subTest(name=name):
                _,record,data,radii=self.diagnostic_fixture()
                if name=='missing': data=data[:0]
                if name=='truncated': data[0,13]=1
                if name=='unreached': data[0,0]=0
                if name=='order': data[1,2]=0
                if name=='reason': data[1,18]=2
                if name=='radius': radii[0]=0
                if name=='key': data[1,3]+=1
                with self.assertRaises(AssertionError):
                    self.entry.initialization_diagnostics(t,data,radii,record['domain'],record,[])
                self.assertEqual(record['contributor_diagnostic']['status'],'rejected')

    def test_failed_pixel_and_mapping_survive_result_serialization(self):
        t,o=self.torch,self.o; case,record,data,radii=self.diagnostic_fixture()
        expected=self.manual(case,[1,0]); actual=expected.detach().clone(); actual[0]+=.1
        pixel=self.entry.initialization_pixel_evidence(t,actual,expected,record)
        try:
            o.require(pixel,pixel['status']=='passed','initial_pixel_mismatch')
        except AssertionError:
            result=json.loads(json.dumps(dict(status='failed',initialization=[record])))
        else: self.fail('induced error accepted')
        saved=result['initialization'][0]
        self.assertEqual(saved['seed'],23); self.assertEqual(saved['row_indices'],[8,3])
        self.assertEqual(saved['domain']['order'],[1,0])
        self.assertTrue(all('depth_key' in p and 'source_row' in p for p in saved['domain']['nodes']))
        self.assertEqual(len(saved['pixel_comparison']['components']),7)
        self.assertEqual(saved['pixel_comparison']['worst_component'],'red')
        self.assertGreater(saved['pixel_comparison']['max_bound_ratio'],1)
        for bad in (actual[:6],actual*float('nan')):
            self.assertEqual(self.entry.initialization_pixel_evidence(t,bad,expected,{})['status'],'incomplete')


if __name__ == '__main__':
    if sys.executable != '/home/demo/miniconda3/envs/4dgs310/bin/python' or not sys.dont_write_bytecode:
        raise RuntimeError('4dgs310_B_required')
    print('executable:', sys.executable)
    unittest.main()
