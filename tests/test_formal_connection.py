"""4dgs310 CPU connection tests. Real CPU consumers, isolated GPU/JIT only.

All output/data is temporary. Fixture values are NOT approved production runs.
The training/render/update loop is intentionally not executed.
"""
import ast
from contextlib import ExitStack
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import formal_entry as entry
import formal_inputs as inputs
import formal_config_json as p2
from test_formal_config import fixture, encode


class BootstrapTests(unittest.TestCase):
    def test_lightweight_import_in_isolated_same_python(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
class Deny:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch','numpy','PIL','scene','train','gaussian_renderer'):
   raise AssertionError('heavy import before validation')
sys.meta_path.insert(0,Deny())
import formal_entry
try: formal_entry.verify_inputs(b'{}')
except ValueError: pass
else: raise AssertionError('accepted')
assert not any(n in sys.modules for n in ('torch','numpy','PIL','train'))
print(sys.executable)
"""
        result = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', code, str(ROOT)],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), sys.executable)

    def test_locator_only_cli_rejects_override_before_run(self):
        for args in ([], ['config', '--quiet'], ['config', '--seed', '7'], ['config', '--config', 'other']):
            with self.subTest(args=args), patch.object(entry, 'run') as run, self.assertRaises(SystemExit):
                entry.main(args)
            run.assert_not_called()

    def test_stdlib_static_boundary(self):
        allowed = sys.stdlib_module_names
        for name in ('formal_entry.py', 'formal_inputs.py', 'formal_views.py', 'formal_camera.py'):
            tree = ast.parse((ROOT / name).read_text())
            for node in tree.body:
                if isinstance(node, ast.Import):
                    modules = [x.name.split('.')[0] for x in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module.split('.')[0]]
                else:
                    continue
                self.assertTrue(all(m in allowed or m.startswith('formal_') for m in modules), modules)


class ConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These are actual installed dependencies, not success stubs.
        import numpy as np
        import torch
        from PIL import Image
        from plyfile import PlyData, PlyElement
        import torch.utils.cpp_extension
        cls.np, cls.torch, cls.Image = np, torch, Image
        cls.PlyData, cls.PlyElement = PlyData, PlyElement
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        def forbidden(*args, **kwargs):
            raise AssertionError('real CUDA/JIT/kernel execution forbidden')
        pointops = types.ModuleType('pointops2.functions.pointops')
        pointops.furthestsampling = pointops.knnquery = forbidden
        knn = types.ModuleType('simple_knn._C')
        knn.distCUDA2 = lambda xyz: torch.ones(len(xyz), dtype=torch.float32)
        raster = types.ModuleType('isolated_raster_extension')
        raster.__getattr__ = lambda name: forbidden
        isolated_modules = {
            'pointops2': types.ModuleType('pointops2'),
            'pointops2.functions': types.ModuleType('pointops2.functions'),
            'pointops2.functions.pointops': pointops,
            'simple_knn': types.ModuleType('simple_knn'), 'simple_knn._C': knn}
        # Restore only the GPU modules replaced here. patch.dict(sys.modules)
        # clears even newly imported real CPU dependencies at teardown; a
        # second fixture would then re-register torchvision's native operators.
        absent = object()
        def restore_module(name, previous):
            if previous is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name, module in isolated_modules.items():
            cls.stack.callback(restore_module, name, sys.modules.get(name, absent))
            sys.modules[name] = module
        cls.jit = cls.stack.enter_context(patch.object(torch.utils.cpp_extension, 'load', return_value=raster))
        cls.stack.enter_context(patch.object(torch.cuda, '_lazy_init', side_effect=forbidden))
        cls.train = importlib.import_module('train')
        cls.readers = importlib.import_module('scene.dataset_readers')
        cls.renderer = importlib.import_module('gaussian_renderer')
        cls.Scene = importlib.import_module('scene').Scene

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source, self.output = self.base / 'data', self.base / 'parent' / 'run'
        self.source.mkdir()
        (self.source / 'images').mkdir()
        (self.source / 'masks').mkdir()
        self.value = fixture(self.source, self.output)
        self.value['dataset']['resolution']['divisor'] = 1
        self.value['initialization']['time_variance_denominator'] = 5
        self.value['optimization']['batch_size'] = 2
        self.ref = dict(source_path=str(self.source), width=20, height=10, files={}, frames={})
        for split, count in (('train', 3), ('test', 1)):
            frames = []
            self.ref['frames'][split] = []
            for i in range(count):
                path, mask = f'images/{split}{i}.png', f'masks/{split}{i}.png'
                self.Image.new('RGBA', (20, 10), (100, 40, 10, 128)).save(self.source / path)
                self.Image.new('L', (20, 10), 128).save(self.source / mask)
                matrix = self.np.eye(4)
                matrix[0, 3] = i + (10 if split == 'test' else 0)
                time = (-2, 2, 6)[i]
                frames.append(dict(file_path=path, time=time, transform_matrix=matrix.tolist()))
                self.ref['frames'][split].append([path, time, mask])
            name = f'transforms_{split}.json'
            data = json.dumps(dict(w=20, h=10, fl_x=15, fl_y=15, cx=10, cy=5, frames=frames)).encode()
            self.store_input(name, data)
        vertices = self.np.zeros(4, dtype=[('x','<f4'), ('y','<f4'), ('z','<f4'),
            ('red','u1'), ('green','u1'), ('blue','u1'), ('time','<f4')])
        vertices['x'] = [0, 1, 2, 3]
        vertices['red'] = [10, 20, 30, 40]
        vertices['time'] = [-2, 0, 2, 6]
        buffer = io.BytesIO()
        self.PlyData([self.PlyElement.describe(vertices, 'vertex')], byte_order='<').write(buffer)
        self.store_input('points3d.ply', buffer.getvalue())
        self.owner = patch.object(inputs, '_reference', return_value=self.ref)
        self.owner.start()
        self.addCleanup(self.owner.stop)
        self.gpu = ExitStack()
        self.addCleanup(self.gpu.close)
        self.gpu.enter_context(patch.object(self.torch.cuda, 'is_available', return_value=True))
        self.device = self.gpu.enter_context(patch.object(self.torch.cuda, 'set_device'))
        self.gpu.enter_context(patch.object(self.torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor))
        for name in ('tensor', 'zeros', 'ones', 'empty', 'zeros_like'):
            original = getattr(self.torch, name)
            def cpu_factory(*args, _original=original, **kwargs):
                if str(kwargs.get('device', '')).startswith('cuda'):
                    kwargs['device'] = 'cpu'
                return _original(*args, **kwargs)
            self.gpu.enter_context(patch.object(self.torch, name, cpu_factory))
        self.seed = self.gpu.enter_context(patch.object(self.train, 'setup_seed', wraps=self.train.setup_seed))

    def store_input(self, name, data):
        (self.source / name).write_bytes(data)
        self.ref['files'][name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())

    def verify(self):
        return inputs.verify_inputs(encode(self.value))

    def prepare(self, bundle=None):
        return self.train.prepare_formal_runtime(self.verify() if bundle is None else bundle)

    def test_real_entry_consumers_writer_order_and_single_authority(self):
        locator = self.base / 'config.json'
        locator.write_bytes(encode(self.value))
        events, saved = [], []
        prepare = self.train.prepare_formal_runtime
        begin = self.train.begin_formal_output
        def observed_prepare(bundle):
            self.assertFalse(self.output.exists())
            events.append('prepare')
            p = prepare(bundle)
            saved.append(p)
            self.assertFalse(self.output.parent.exists())
            for view in (p.dataset, p.opt, p.pipe):
                self.assertIs(view.config, bundle.config)
                with self.assertRaises(AttributeError):
                    view.config = None
            self.assertEqual(p.scene.getTrainCameras()[0][0].shape, (3, 10, 20))
            return p
        def observed_begin(p, claim):
            self.assertTrue(self.output.is_dir())
            self.assertEqual(list(self.output.iterdir()), [])
            events.append('claimed')
            return begin(p, claim)
        def tensorboard(path):
            self.assertEqual(Path(path), self.output)
            self.assertTrue(all((self.output/n).is_file() for n in ('cfg_args','input.ply','cameras.json')))
            events.append('tensorboard')
            return None
        with patch.object(self.train, 'prepare_formal_runtime', side_effect=observed_prepare), \
                patch.object(self.train, 'begin_formal_output', side_effect=observed_begin), \
                patch.object(self.train, 'TENSORBOARD_FOUND', True), \
                patch.object(self.train, 'SummaryWriter', side_effect=tensorboard, create=True), \
                patch.object(self.train, 'continue_formal_training', side_effect=lambda p,w: events.append('loop_boundary')):
            entry.run(locator)
        self.assertEqual(events, ['prepare','claimed','tensorboard','loop_boundary'])
        self.seed.assert_called_once_with(23)
        self.device.assert_called_once_with(0)
        p, c = saved[0], saved[0].inputs.config
        self.assertEqual((p.gaussians.active_sh_degree,p.gaussians.active_sh_degree_t), (0,0))
        self.assertEqual(p.gaussians.get_max_sh_channels, 48)
        self.assertEqual(p.gaussians.time_duration, (-1,3))
        self.np.testing.assert_array_equal(p.gaussians._t.detach().numpy().ravel(), [-1,0,1,3])
        expected_log = math.log(math.sqrt(8/5)/2)
        self.torch.testing.assert_close(p.gaussians._scaling_t, self.torch.full((4,1), expected_log))
        self.torch.testing.assert_close(p.gaussians.get_scaling_t.square(), self.torch.full((4,1), 0.4))
        self.assertEqual(sorted(cam.timestamp for cam in p.scene.train_cameras[1.0]), [-1,1,3])
        rates = {g['name']:g['lr'] for g in p.gaussians.optimizer.param_groups}
        self.assertAlmostEqual(rates['t'], c.optimization.learning_rate.position_t_lr_init*p.scene.cameras_extent)
        self.assertEqual(rates['f_rest'], rates['f_dc']/20)
        self.assertEqual(rates['scaling'], rates['scaling_t'])
        self.assertEqual(rates['rotation'], rates['rotation_r'])
        self.assertEqual(p.opt.densify_until_num_points, -1)
        self.assertEqual((p.opt.densify_from_iter,p.opt.densify_until_iter), (20,2))
        self.assertIsNone(p.opt.densify_grad_t_threshold)
        self.assertEqual(p.dataloader.batch_size, 2)
        self.assertTrue(p.dataloader.drop_last)
        self.assertEqual(p.dataloader.num_workers, 0)
        self.assertEqual(len(next(iter(p.dataloader))), 2)
        stages=[]
        for i in range(5):
            p.gaussians.oneupSHdegree()
            stages.append((p.gaussians.active_sh_degree,p.gaussians.active_sh_degree_t))
        self.assertEqual(stages, [(1,0),(2,0),(3,0),(3,1),(3,2)])
        old_time_rate = rates['t']
        p.gaussians.update_learning_rate(7)
        self.assertEqual(next(g['lr'] for g in p.gaussians.optimizer.param_groups if g['name']=='t'), old_time_rate)
        self.assertEqual((self.output/'input.ply').read_bytes(), p.inputs.ply_bytes)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_same_snapshot_after_metadata_and_ply_paths_change(self):
        bundle = self.verify()
        for name in ('transforms_train.json','transforms_test.json','points3d.ply'):
            (self.source/name).write_bytes(b'changed after acquisition')
        p = self.prepare(bundle)
        self.assertEqual(len(p.scene.train_cameras[1.0]), 3)
        self.assertEqual(len(p.gaussians._t), 4)
        self.assertFalse(self.output.parent.exists())

    def test_sample_indices_shared_and_closed_endpoints_not_filtered(self):
        self.value['initialization']['num_pts'] = 2
        # RNG boundary fixture chooses both endpoints; consumer remains real.
        with patch.object(self.np.random, 'randint', return_value=self.np.array([0,3])) as draw:
            p = self.prepare()
        draw.assert_called_once_with(0, 4, 2)
        self.np.testing.assert_array_equal(p.gaussians._xyz.detach().numpy()[:,0], [0,3])
        self.np.testing.assert_array_equal(p.gaussians._t.detach().numpy()[:,0], [-1,3])

    def test_divisor_five_uses_exact_resolution_not_target_width(self):
        self.value['dataset']['resolution']['divisor'] = 5
        for split in self.ref['frames'].values():
            for _,_,mask in split:
                self.Image.new('L', (4,2), 128).save(self.source/mask)
        p = self.prepare()
        image, cam = p.scene.getTrainCameras()[0]
        self.assertEqual(tuple(image.shape), (3,2,4))
        self.assertEqual(cam.resolution, (4,2))
        self.assertEqual(cam.fl_x, 3)
        self.assertEqual(tuple(cam.gt_alpha_mask.shape), (1,2,4))

    def test_invalid_config_owner_metadata_header_ply_prevent_prepare(self):
        locator = self.base/'config.json'
        cases = ('config','owner','metadata','header','ply','debug','loss','variance')
        for case in cases:
            with self.subTest(case=case):
                original = encode(self.value)
                if case=='config':
                    locator.write_bytes(b'{}')
                else:
                    locator.write_bytes(original)
                with ExitStack() as guards:
                    if case=='owner':
                        guards.enter_context(patch.dict(self.ref, source_path=str(self.base/'wrong')))
                    if case in ('metadata','ply'):
                        name = 'points3d.ply' if case=='ply' else 'transforms_test.json'
                        guards.enter_context(patch.dict(self.ref['files'][name], sha256='0'*64))
                    if case=='header':
                        guards.enter_context(patch.object(inputs, 'png_header', return_value=(1,1,6)))
                    if case=='debug':
                        guards.enter_context(patch.dict(os.environ, STEP90_CUDA_DEBUG_PIXEL='1,1'))
                    if case=='loss':
                        value=json.loads(original); value['optimization']['loss']['lambda_rigid']=1
                        locator.write_bytes(encode(value))
                    if case=='variance':
                        value=json.loads(original); value['initialization']['time_variance_denominator']=4
                        locator.write_bytes(encode(value))
                    prepare=guards.enter_context(patch.object(self.train,'prepare_formal_runtime'))
                    with self.assertRaises(ValueError): entry.run(locator)
                    prepare.assert_not_called()
                self.assertFalse(self.output.parent.exists())
        self.seed.assert_not_called()

    def test_mask_mismatch_preclaim_and_lazy_missing_mask_rejected(self):
        self.value['dataset']['resolution']['divisor']=2
        with self.assertRaisesRegex(ValueError,'mask_resolution'): self.verify()
        self.seed.assert_not_called()
        self.assertFalse(self.output.parent.exists())
        self.value['dataset']['resolution']['divisor']=1
        bundle=self.verify()
        self.Image.new('L',(10,5),128).save(self.source/'masks/train0.png')
        with self.assertRaisesRegex(ValueError,'formal_mask_resolution'): self.prepare(bundle)
        self.assertFalse(self.output.parent.exists())

    def test_bad_unsampled_ply_time_and_activation_rejected_before_claim(self):
        bundle=self.verify()
        ply=self.PlyData.read(io.BytesIO(bundle.ply_bytes))
        ply['vertex']['time'][-1]=7
        stream=io.BytesIO(); ply.write(stream)
        bad=bundle._replace(ply_bytes=stream.getvalue())
        with patch.object(self.np.random,'randint') as draw:
            with self.assertRaisesRegex(ValueError,'formal_ply_time_range'): self.prepare(bad)
            draw.assert_not_called()
        self.assertFalse(self.output.parent.exists())
        c=bundle.config
        bad=bundle._replace(config=c._replace(time_derivation=c.time_derivation._replace(log_effective_scale=-1000)))
        with self.assertRaisesRegex(ValueError,'formal_temporal_activation'): self.prepare(bad)
        self.assertFalse(self.output.parent.exists())

    def test_prepare_failure_and_exclusive_claim_loser_never_write(self):
        locator=self.base/'config.json'; locator.write_bytes(encode(self.value))
        with patch.object(self.train,'prepare_formal_runtime',side_effect=RuntimeError('prepare_failure')), \
                patch.object(self.train,'begin_formal_output') as writer:
            with self.assertRaisesRegex(RuntimeError,'prepare_failure'): entry.run(locator)
            writer.assert_not_called()
        self.assertFalse(self.output.parent.exists())
        bundle=self.verify()
        claim=entry.OutputClaim(bundle)
        self.addCleanup(claim.close)
        with self.assertRaises((ValueError, FileExistsError)): entry.OutputClaim(bundle)
        self.assertEqual(list(self.output.iterdir()), [])
        with self.assertRaises(ValueError): self.train.prepare_output_and_logger(self.train.ConsumerView(bundle.config,'dataset'))

    def test_claim_writer_failure_keeps_leaf_no_retry(self):
        p=self.prepare()
        claim=entry.OutputClaim(p.inputs); self.addCleanup(claim.close)
        with patch.object(p.scene,'write_initial_files',side_effect=OSError('fixture_writer_failure')) as writer:
            with self.assertRaisesRegex(OSError,'fixture_writer_failure'): self.train.begin_formal_output(p,claim)
            writer.assert_called_once()
        self.assertEqual([x.name for x in self.output.iterdir()], ['cfg_args'])

    def test_renderer_pf_rejects_before_gpu_allocation(self):
        c=self.verify().config
        pc=types.SimpleNamespace(prefilter_var=-1.0)
        pipe=self.train.ConsumerView(c,'pipeline')
        for invalid in (True,-1,0.0,-2.0,float('nan'),float('inf')):
            pc.prefilter_var=invalid
            with self.subTest(invalid=invalid), patch.object(self.torch,'zeros_like') as alloc:
                with self.assertRaisesRegex(ValueError,'formal_renderer_contract'):
                    self.renderer.render(None,pc,pipe,None,scaling_modifier=1.0,override_color=None,formal_config=c)
                alloc.assert_not_called()

    def legacy_args(self, output):
        return types.SimpleNamespace(source_path=str(self.source),model_path=str(output),
            white_background=False,eval=True,extension='',num_extra_pts=0,frame_ratio=2,
            dataloader=True,resolution=1,data_device='cuda',loaded_pth='',images='images')

    def legacy_model(self):
        return self.train.GaussianModel(3,gaussian_dim=4,time_duration=(-1,3),rot_4d=True,sh_degree_t=2)

    def test_legacy_sampling_reader_camera_scene_and_logger_still_work(self):
        # Real legacy consumers. The RNG boundary chooses interior times so
        # this successful sampling case is independent of the known full-retain
        # stride failure below. Do not change legacy's time filtering policy.
        legacy=self.base/'legacy'; legacy.mkdir()
        args, model=self.legacy_args(legacy), self.legacy_model()
        scene_module=sys.modules[self.Scene.__module__]
        with patch.object(self.np.random,'randint',return_value=self.np.array([1,2,1])) as draw, \
                patch.object(scene_module,'camera_to_JSON',wraps=scene_module.camera_to_JSON) as diagnostic:
            scene=self.Scene(args,model,num_pts=3,time_duration=(-1,3),shuffle=False)
        draw.assert_called_once_with(0,4,3)
        self.assertEqual(diagnostic.call_count,4)
        self.assertEqual((legacy/'input.ply').read_bytes(),(self.source/'points3d.ply').read_bytes())
        cameras=json.loads((legacy/'cameras.json').read_text())
        self.assertEqual([c['img_name'] for c in cameras],['test0','train0','train1','train2'])
        self.assertEqual([c.timestamp for c in scene.train_cameras[1.0]],[-1,1,3])
        self.np.testing.assert_array_equal(model._xyz.detach().numpy()[:,0],[1,2,1])
        self.np.testing.assert_array_equal(model._t.detach().numpy()[:,0],[0,2,0])
        self.assertEqual(tuple(model._features_dc.shape),(3,1,3))
        image, camera=scene.getTrainCameras()[0]
        self.assertEqual(tuple(image.shape),(3,10,20))
        self.assertEqual(tuple(camera.gt_alpha_mask.shape),(1,10,20))
        # Legacy V-B keeps duration/5; formal uses raw duration / (5*d*d).
        self.torch.testing.assert_close(model.get_scaling_t.square(),self.torch.full((3,1),0.8))
        with patch.object(self.train,'TENSORBOARD_FOUND',False):
            self.assertIsNone(self.train.prepare_output_and_logger(args))
        self.assertTrue((legacy/'cfg_args').read_text().startswith('Namespace('))
        self.assertFalse(self.output.exists())
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_legacy_packed_full_retention_known_stride_failure(self):
        # Retain the previous failing condition, not an arbitrary expected
        # failure/skip. The actual Scene -> Gaussian -> torch.from_numpy must
        # reach the packed N x 1 time view and produce this specific ValueError.
        legacy=self.base/'legacy-packed'; legacy.mkdir()
        args, model=self.legacy_args(legacy), self.legacy_model()
        original=self.torch.from_numpy
        rejected=[]
        def observe_handoff(array):
            try:
                return original(array)
            except ValueError:
                rejected.append(array)
                raise
        with patch.object(self.torch,'from_numpy',side_effect=observe_handoff), \
                patch.object(model,'create_from_pcd',wraps=model.create_from_pcd) as create, \
                patch.object(self.np.random,'randint',side_effect=AssertionError('unexpected sampling')) as draw:
            with self.assertRaisesRegex(ValueError,
                    r'^given numpy array strides not a multiple of the element byte size\. Copy the numpy array to reallocate the memory\.$'):
                self.Scene(args,model,num_pts=7,time_duration=(-1,3),shuffle=False)
        create.assert_called_once()
        draw.assert_not_called()
        self.assertEqual(len(rejected),1)
        time=rejected[0]
        self.assertIs(time,create.call_args.args[0].time)
        self.assertEqual(time.shape,(4,1))
        self.assertEqual(time.dtype,self.np.dtype('float32'))
        self.assertEqual(time.strides,(19,0))
        self.assertEqual(time.dtype.itemsize,4)
        self.np.testing.assert_array_equal(time[:,0],[-2,0,2,6])
        self.assertEqual(model._xyz.numel(),0)  # Failure before parameter assignment.
        self.assertTrue((legacy/'input.ply').is_file())
        self.assertTrue((legacy/'cameras.json').is_file())
        self.assertFalse((legacy/'cfg_args').exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_legacy_loaded_iteration_skips_diagnostics_and_initial_writer(self):
        scene_module=sys.modules[self.Scene.__module__]
        # Explicit iteration and real latest-iteration directory selection.
        # Only the load boundary is replaced; no real checkpoint is read.
        # Current GaussianModel has no load_ply implementation. Explicitly
        # supply the authorized fixture boundary: this proves Scene dispatch,
        # NOT availability/correctness of real legacy checkpoint loading.
        for requested in (7,-1):
            with self.subTest(load_iteration=requested):
                legacy=self.base/f'legacy-loaded-{requested}'
                for iteration in (3,7):
                    (legacy/'point_cloud'/f'iteration_{iteration}').mkdir(parents=True)
                originals={name:('existing '+name).encode() for name in ('input.ply','cameras.json','cfg_args')}
                for name,data in originals.items(): (legacy/name).write_bytes(data)
                args, model=self.legacy_args(legacy), self.legacy_model()
                with patch.object(scene_module,'camera_to_JSON',side_effect=AssertionError('loaded diagnostics')) as diagnostic, \
                        patch.object(self.Scene,'write_initial_files',side_effect=AssertionError('loaded initial writer')) as writer, \
                        patch.object(model,'load_ply',create=True) as load, \
                        patch.object(model,'create_from_pcd',side_effect=AssertionError('loaded fresh initialization')) as create, \
                        patch.object(model,'create_from_pth',side_effect=AssertionError('wrong load branch')) as warm, \
                        patch.object(self.torch,'load',side_effect=AssertionError('real checkpoint load')):
                    scene=self.Scene(args,model,load_iteration=requested,num_pts=7,time_duration=(-1,3),shuffle=False)
                self.assertEqual(scene.loaded_iter,7)
                self.assertIs(scene.gaussians,model)
                load.assert_called_once_with(str(legacy/'point_cloud'/'iteration_7'/'point_cloud.ply'))
                diagnostic.assert_not_called()
                writer.assert_not_called()
                create.assert_not_called()
                warm.assert_not_called()
                self.assertEqual(len(scene.train_cameras[1.0]),3)
                self.assertEqual(len(scene.test_cameras[1.0]),1)
                self.assertEqual([c.timestamp for c in scene.train_cameras[1.0]],[-1,1,3])
                self.assertEqual({name:(legacy/name).read_bytes() for name in originals},originals)
                self.assertFalse((legacy/'point_cloud'/'iteration_7'/'point_cloud.ply').exists())
        self.assertFalse(self.output.exists())
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__=='__main__':
    print('test executable:',sys.executable)
    unittest.main()
