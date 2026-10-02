"""Real Step 5/6 consumers on temporary data; GPU seams only.

The linear renderer is a differentiable control input, not a raster oracle.
Save/report observers capture values at call time, never aliased references.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import test_formal_connection as connection


def cloned(value):
    if hasattr(value, 'detach'):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: cloned(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cloned(v) for v in value)
    return value


class CPUCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.addClassCleanup(connection.ConnectionTests.doClassCleanups)
        connection.ConnectionTests.setUpClass()

    def setUp(self):
        self.f = connection.ConnectionTests('runTest')
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()
        self.torch, self.train = self.f.torch, self.f.train
        threads = self.torch.get_num_threads()
        self.torch.set_num_threads(1)
        self.addCleanup(self.torch.set_num_threads, threads)
        original = self.torch.Tensor.to
        def cpu_to(tensor, *args, **kw):
            if args and str(args[0]).startswith('cuda'):
                args = ('cpu',) + args[1:]
            if str(kw.get('device', '')).startswith('cuda'):
                kw['device'] = 'cpu'
            return original(tensor, *args, **kw)
        self.f.gpu.enter_context(patch.object(self.torch.Tensor, 'to', cpu_to))
        self.f.gpu.enter_context(patch.object(self.torch.cuda, 'is_current_stream_capturing', return_value=False))
        self.f.gpu.enter_context(patch.object(self.torch.cuda, 'empty_cache'))

    def configured(self, n=3, batch=2, population=None, tests=(), saves=(), size=None):
        f = self.f
        o = f.value['optimization']
        o.update(total_updates=n, batch_size=batch)
        o['loss'].update(lambda_dssim=0, lambda_opa_mask=0)
        o['learning_rate'].update(position_lr_init=.002, position_lr_final=.001,
            position_t_lr_init=.003, feature_lr=.004, opacity_lr=.005,
            scaling_lr=0, rotation_lr=0)
        o['population'].update(densify_until_update=0, densify_from_update=0,
            densification_interval=1, opacity_reset_interval=100,
            densify_grad_threshold=.00001, percent_dense=2, thresh_opa_prune=0)
        if population:
            o['population'].update(population)
        o['sh']['increase_interval'] = 1
        f.value['reporting']['test_updates'] = list(tests)
        f.value['checkpoint']['intermediate_updates'] = list(saves)
        if size:
            f.ref.update(width=size, height=size)
            for split in ('train', 'test'):
                name = f'transforms_{split}.json'
                root = json.loads((f.source / name).read_bytes())
                root.update(w=size, h=size, fl_x=size, fl_y=size, cx=size/2, cy=size/2)
                f.store_input(name, json.dumps(root).encode())
                for frame in root['frames']:
                    path = frame['file_path']
                    f.Image.new('RGBA', (size, size), (100, 40, 10, 128)).save(f.source/path)
                    f.Image.new('L', (size, size), 128).save(f.source/path.replace('images/', 'masks/'))
        return f.prepare()

    def snapshot(self, model):
        return cloned(model.capture())

    def exercise(self, p, *, real_report=False, render=None, fail=None, writer=None):
        torch, train, model = self.torch, self.train, p.gaussians
        trace = dict(fetch=[], labels=[], forward=[], forward_parameters=[], steps=[], zero=[], saves=[], reports=[], events=[], mutations=[],
                     initial=cloned({g['name']: g['params'][0] for g in model.optimizer.param_groups}))
        self.trace = trace
        def event(name):
            trace['events'].append(name)
            if fail == name:
                raise RuntimeError('injected:' + name)
        loader = p.dataloader
        class Batches:
            def __iter__(_):
                for batch in loader:
                    event('fetch')
                    trace['fetch'].append([camera.formal_frame.index for _, camera in batch])
                    yield batch
        class Event:
            def __init__(self, **kw): pass
            def record(self): pass
            def elapsed_time(self, other): return 0.
        def surrogate(camera, gaussian, pipe, background, **kw):
            self.assertIs(kw['formal_config'], p.inputs.config)
            event('forward')
            trace['forward'].append((trace['labels'][-1] if trace['labels'] else 0,
                                     camera.formal_frame.index, gaussian.active_sh_degree, gaussian.active_sh_degree_t))
            trace['forward_parameters'].append([id(g['params'][0]) for g in gaussian.optimizer.param_groups])
            screen = torch.zeros_like(gaussian._xyz, requires_grad=True)
            signal = sum(g['params'][0].sum() * .0001 for g in gaussian.optimizer.param_groups)
            signal = signal + screen[:, :2].sum() * .001
            rgb = (.6 + signal).expand(3, camera.image_height, camera.image_width)
            if rgb.requires_grad:
                rgb.register_hook(lambda grad: (event('backward'), grad)[1])
            return dict(render=rgb, viewspace_points=screen,
                visibility_filter=torch.ones(len(screen), dtype=torch.bool),
                radii=torch.arange(1, len(screen)+1, dtype=torch.float32),
                depth=torch.linspace(1., 2., camera.image_height*camera.image_width).reshape(1, camera.image_height, camera.image_width),
                alpha=(.3 + gaussian._opacity.sum()*.0001).expand(1, camera.image_height, camera.image_width))
        original_lr = model.update_learning_rate
        def lr(k):
            event('lr'); trace['labels'].append(k)
            return original_lr(k)
        original_step, original_zero = model.optimizer.step, model.optimizer.zero_grad
        def step(*a, **kw):
            event('step')
            before = [(g['name'], id(g['params'][0]), cloned(g['params'][0].grad)) for g in model.optimizer.param_groups]
            result = original_step(*a, **kw)
            trace['steps'].append((before, self.snapshot(model)))
            return result
        def zero(*a, **kw):
            event('zero'); result = original_zero(*a, **kw)
            self.assertTrue(all(g['params'][0].grad is None for g in model.optimizer.param_groups))
            trace['zero'].append(self.snapshot(model)); return result
        def save(value, path):
            event('save'); trace['saves'].append((cloned(value), str(path)))
        report = train.training_report
        def observed_report(*a, **kw):
            k = a[1]
            if k:
                event('report')
            trace['reports'].append((k, self.snapshot(model), float(a[3]) if len(a)>3 else float(kw['loss'])))
            if real_report:
                return report(*a, **kw)
            return 0.
        with ExitStack() as stack:
            for name in ('add_densification_stats', 'add_densification_stats_grad', 'densify_and_clone',
                         'densify_and_split', 'prune_points', 'reset_opacity', 'densify_and_prune'):
                original = getattr(model, name)
                def observed(*a, _name=name, _original=original, **kw):
                    event(_name)
                    before = self.snapshot(model)
                    result = _original(*a, **kw)
                    trace['mutations'].append((_name, trace['labels'][-1], before, self.snapshot(model), cloned(kw), cloned(a)))
                    return result
                stack.enter_context(patch.object(model, name, observed))
            stack.enter_context(patch.object(torch.cuda, 'Event', Event))
            stack.enter_context(patch.object(train, 'render', render or surrogate))
            stack.enter_context(patch.object(model, 'update_learning_rate', lr))
            stack.enter_context(patch.object(model.optimizer, 'step', step))
            stack.enter_context(patch.object(model.optimizer, 'zero_grad', zero))
            stack.enter_context(patch.object(torch, 'save', save))
            stack.enter_context(patch.object(train, 'training_report', observed_report))
            trace['completed'] = train.continue_formal_training(p._replace(dataloader=Batches()), writer)
        self.assertFalse(torch.cuda.is_initialized())
        return trace
