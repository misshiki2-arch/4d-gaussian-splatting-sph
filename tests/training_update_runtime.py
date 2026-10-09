"""Step 12 synthetic inputs and pass-through observations of the REAL trainer.

No GPU substitution here. CPU callers explicitly supply a surrogate renderer
and the pre-existing CPU isolation context; CUDA callers use train.render.
"""
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import io
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

import training_update_oracle as o

CASES = (
    dict(name='G0a', n=1, batch=1, kind='initial'),
    dict(name='G0b', n=2, batch=1, kind='initial'),
    dict(name='G0c', n=3, batch=1, kind='initial'),
    dict(name='G1', n=6, batch=2, kind='visibility'),
    dict(name='G2', n=3, batch=1, kind='clone'),
    dict(name='G3', n=3, batch=2, kind='split'),
    dict(name='G4', n=3, batch=2, kind='mixed'),
    dict(name='G5', n=3, batch=2, kind='threshold'),
    dict(name='G6', n=3, batch=1, kind='reset'),
    dict(name='G7', n=4, batch=2, kind='size'),
    dict(name='G8', n=2, batch=1, kind='mixed'),
    dict(name='G9', n=2, batch=2, kind='zero_lr'),
)
MAX_ROWS, MAX_FORWARDS = 64, 500
TRAIN_CAMERA_Z = (-.5, 0., .5)


def fixture_extent():
    """Geometry-derived expectation, independent of reader/model normalization."""
    center = sum(TRAIN_CAMERA_Z)/len(TRAIN_CAMERA_Z)
    return np.float32(o.binary32(max(abs(z-center) for z in TRAIN_CAMERA_Z)*1.1))


def config_value(source, output, case):
    from test_formal_config import fixture
    value = fixture(source, output)
    value['dataset']['resolution']['divisor'] = 1
    value['initialization'].update(num_pts=16, time_variance_denominator=5)
    opt = value['optimization']
    opt.update(total_updates=case['n'], batch_size=case['batch'])
    opt['learning_rate'].update(position_lr_init=.002, position_lr_final=.001,
        position_t_lr_init=.003, feature_lr=.004, opacity_lr=.005, scaling_lr=.001, rotation_lr=.001)
    opt['sh']['increase_interval'] = 1 if case['name'] == 'G1' else 100
    pop = opt['population']
    pop.update(densify_from_update=1, densify_until_update=3, densification_interval=2,
               opacity_reset_interval=100, percent_dense=.03/.55, thresh_opa_prune=.05,
               densify_grad_threshold=0)
    kind = case['kind']
    if kind in ('initial', 'zero_lr'):
        pop.update(densify_from_update=0, densify_until_update=0)
    if kind == 'zero_lr':
        for name in opt['learning_rate']:
            if name != 'position_lr_max_steps':
                opt['learning_rate'][name] = 0
    if kind == 'visibility':
        pop.update(densify_from_update=7, densify_until_update=7)
    if kind == 'mixed':
        pop['opacity_reset_interval'] = 2
    if kind == 'threshold':
        pop.update(densify_from_update=0, densification_interval=1, densify_stop_points=16)
    if kind == 'reset':
        pop.update(densify_from_update=3, opacity_reset_interval=2)
    if kind == 'size':
        pop.update(densify_from_update=2, densify_until_update=4, densification_interval=3,
                   opacity_reset_interval=2, thresh_opa_prune=0, percent_dense=.065/.55)
    tests = {'G1': [3], 'G4': [2, 3], 'G8': [2]}.get(case['name'], [])
    value['reporting']['test_updates'] = tests
    value['checkpoint']['intermediate_updates'] = [2] if case['name'] == 'G4' else []
    return value


def prepare(base, case):
    """Real temporary reader/camera/KNN/seed/optimizer; only reference is local."""
    from PIL import Image
    from plyfile import PlyData, PlyElement
    import formal_inputs
    from test_formal_config import encode
    import train

    base.mkdir()
    source = base/'data'
    source.mkdir()
    (source/'images').mkdir()
    (source/'masks').mkdir()
    value = config_value(source, base/'output', case)
    reference = dict(source_path=str(source), width=256, height=256, files={}, frames={})

    def store(name, data):
        (source/name).write_bytes(data)
        reference['files'][name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())

    y, x = np.indices((256, 256))
    for split, stamps in (('train', (0., .2, .4)), ('test', (.1, .3))):
        frames = []
        reference['frames'][split] = []
        for i, stamp in enumerate(stamps):
            name = f'images/{split}{i}.png'
            mask = name.replace('images/', 'masks/')
            rgba = np.stack([40+(x+i*7)%130, 30+(y+3*i)%150,
                             20+(x+y)%170, np.full_like(x, 255)], axis=-1).astype('uint8')
            Image.fromarray(rgba, 'RGBA').save(source/name)
            Image.fromarray(np.where((x-128)**2+(y-128)**2 < 100**2, 255, 0).astype('uint8')).save(source/mask)
            pose = np.diag([1., -1., -1., 1.])
            pose[2, 3] = TRAIN_CAMERA_Z[i]
            frames.append(dict(file_path=name, time=stamp, transform_matrix=pose.tolist()))
            reference['frames'][split].append([name, stamp, mask])
        store(f'transforms_{split}.json', json.dumps(dict(w=256, h=256, fl_x=192., fl_y=192.,
                 cx=128., cy=128., frames=frames)).encode())
    rows = np.zeros(16, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('time', '<f4')])
    i = np.arange(16)
    rows['x'], rows['y'], rows['z'] = .16*(i%4-1.5), .12*(i//4-1.5), 3+.025*i
    rows['red'], rows['green'], rows['blue'], rows['time'] = 20+3*i, 50+2*i, 100+i, .2
    buffer = io.BytesIO()
    PlyData([PlyElement.describe(rows, 'vertex')], byte_order='<').write(buffer)
    store('points3d.ply', buffer.getvalue())
    (base/'config.json').write_bytes(encode(value))
    with patch.object(formal_inputs, '_reference', return_value=reference):
        inputs = formal_inputs.verify_inputs(encode(value))
    prepared = train.prepare_formal_runtime(inputs)
    o.require(math.isclose(prepared.scene.cameras_extent, .55, rel_tol=1e-6), 'fixture_extent')
    initialize_population(prepared, case)
    return prepared


def initialize_population(p, case):
    import torch
    m, kind = p.gaussians, case['kind']
    o.require(not m.optimizer.state and len(m._xyz) == 16, 'fixture_initial_state')
    if kind == 'initial':
        return
    with torch.no_grad():
        m._opacity.fill_(math.log(.2/.8))
        m._scaling.copy_(m._scaling.new_tensor([.01, .009, .008]).log().expand_as(m._scaling))
        m._scaling_t.fill_(math.log(.15))
        m._t.fill_(.1)
        m._features_rest.copy_(torch.arange(m._features_rest.numel(), device=m._features_rest.device,
                                           dtype=m._features_rest.dtype).reshape_as(m._features_rest)*1e-6)
        if kind in ('split', 'mixed'):
            selected = slice(None) if kind == 'split' else slice(8, None)
            m._scaling[selected] = m._scaling.new_tensor([.06, .05, .04]).log()
            m._rotation[selected] = m._rotation.new_tensor([math.cos(.07), math.sin(.07), 0, 0])
            m._rotation_r[selected] = m._rotation_r.new_tensor([math.cos(.09), 0, math.sin(.09), 0])
        if kind in ('mixed', 'threshold'):
            m._opacity[:2] = math.log(.02/.98)
        if kind == 'reset':
            m._opacity[:2] = math.log(.005/.995)
        if kind == 'visibility':
            m._t[:4, 0] = m._t.new_tensor([2., 0., .1, .2])
            m._scaling_t[:4] = math.log(.01)
        if kind == 'size':
            # Independent near screen-only, far world-only, and split-survivor domains.
            m._xyz[0] = m._xyz.new_tensor([0., 0., 1.])
            m._scaling[0] = math.log(.04)
            m._xyz[1] = m._xyz.new_tensor([0., 0., 7.])
            m._scaling[1] = math.log(.06)
            m._scaling[2:4] = math.log(.07)
    o.require(not m.optimizer.state, 'fixture_must_not_fabricate_adam_history')


def snapshot(model):
    groups = {}
    for group in model.optimizer.param_groups:
        name, param = group['name'], group['params'][0]
        o.require(param is getattr(model, o.GROUPS[name]) and param.is_leaf, 'optimizer_leaf_owner')
        state = model.optimizer.state.get(param, {})
        a = o.array(param)
        groups[name] = dict(p=a, m=o.array(state['exp_avg']) if state else np.zeros_like(a),
                            v=o.array(state['exp_avg_sq']) if state else np.zeros_like(a),
                            step=int(state['step'].item()) if state else 0)
    o.require(set(groups) == set(o.GROUPS), 'nine_groups')
    return dict(groups=groups, stats={n: o.array(getattr(model, n)) for n in o.STATS},
                sh=(model.active_sh_degree, model.active_sh_degree_t))


def equal_snapshot(a, b):
    o.require(a['sh'] == b['sh'], 'snapshot_SH')
    for n in o.GROUPS:
        for key in ('p', 'm', 'v', 'step'):
            o.compare(a['groups'][n][key], b['groups'][n][key], n+'/'+key, exact=True)
    for n in o.STATS:
        o.compare(a['stats'][n], b['stats'][n], n, exact=True)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


class Observation:
    def __init__(self, p, case, renderer, budget):
        import torch
        import train
        self.torch, self.train, self.p, self.case = torch, train, p, case
        self.model, self.renderer, self.budget = p.gaussians, renderer, budget
        self.oracle = o.Trajectory(snapshot(self.model))
        self.trace = dict(case=case['name'], status='incomplete', events=[], steps=[], reports=[], saves=[],
                          batches=[], forwards=0, population=[], comparisons=self.oracle.records)
        self.k, self.epoch = 0, 0
        self.pending, self.views, self.handles, self.retired = None, [], [], []
        self.report_mode = False
        self.context, self.split_rows, self.expected_mask = None, None, None
        self.lr_config = p.inputs.config
        self.lr_extent = fixture_extent()
        self.verified_lr = None
        self.trace['learning_rates'] = []
        self.trace.update(losses=[], adam_dispatch=[])
        self.loss_input, self.last_loss, self.evaluation, self.active_adam = None, None, None, None
        self.training_code = train.training.__code__
        # Scene shuffles storage order; a storage index is NOT a formal frame index.
        train_frames = [camera.formal_frame for camera in p.scene.train_cameras[1.0]]
        test_frames = [camera.formal_frame for camera in p.scene.test_cameras[1.0]]
        self.evaluation_frames = tuple(train_frames[i % len(train_frames)] for i in range(5, 30, 5))+tuple(test_frames)
        self.adam_module = importlib.import_module('torch.optim.adam')
        adam_source = Path(self.adam_module.__file__)
        self.trace['adam_implementation'] = dict(torch_version=torch.__version__, path=str(adam_source),
            sha256=hashlib.sha256(adam_source.read_bytes()).hexdigest())

    def event(self, name):
        self.trace['events'].append([self.k, name])

    def finish_view(self):
        if self.pending is None:
            return
        v, screen = self.pending
        o.require(v.get('loss', {}).get('backward_completed', False), 'loss_backward_missing')
        o.require(screen.grad is not None, 'screen_backward_missing')
        v['screen'] = o.array(screen.grad)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.views.append(v)
        self.pending = None
        self.loss_input = None

    def render(self, camera, model, *args, **kwargs):
        self.finish_view()
        o.require(model is self.model and kwargs['formal_config'] is self.p.inputs.config, 'render_owner')
        snapshot(model)  # Includes fresh optimizer/leaf binding after topology.
        if self.report_mode:
            equal_snapshot(self.evaluation['state'], snapshot(model))
        o.require(0 < len(model._xyz) <= MAX_ROWS, 'population_bound')
        self.budget['forwards'] += 1
        self.trace['forwards'] += 1
        o.require(self.budget['forwards'] <= MAX_FORWARDS, 'forward_bound')
        if not self.report_mode:
            expected_sh = (min(self.k, 3), min(max(self.k-3, 0), 2)) if self.case['name'] == 'G1' else (0, 0)
            o.require(snapshot(model)['sh'] == expected_sh, 'SH_before_forward')
        result = self.renderer(camera, model, *args, **kwargs)
        if self.report_mode:
            equal_snapshot(self.evaluation['state'], snapshot(model))
            # renderFunc is called directly by the existing training_report.
            frame = inspect.currentframe().f_back
            try:
                local = frame.f_locals
                o.require(local.get('viewpoint') is camera, 'metric_camera_input')
                teacher = o.array(local['gt_image'])
                o.compare(teacher, local['batch_data'][0], 'metric_teacher', exact=True)
            finally:
                del frame
            o.require(self.evaluation.get('pending') is None, 'metric_missing_previous')
            self.evaluation['pending'] = dict(frame=camera.formal_frame, teacher=teacher,
                                               image=np.clip(o.array(result['render']), 0, 1))
        depth = o.array(result['depth'])
        o.require(np.isfinite(depth).all() and depth.max() > depth.min(), 'diagnostic_depth_nonconstant')
        for name in ('render', 'alpha'):
            o.require(np.isfinite(o.array(result[name])).all(), name+'_finite')
        if not self.report_mode:
            self.event('render')
            v = dict(frame=camera.formal_frame.index, visible=o.array(result['visibility_filter']),
                     radii=o.array(result['radii']), gradients={}, calls={})
            for group in model.optimizer.param_groups:
                name = group['name']
                v['gradients'][name] = None
                v['calls'][name] = 0
                def observe(gradient, name=name):
                    v['calls'][name] += 1
                    o.require(v['calls'][name] == 1, 'duplicate_leaf_hook')
                    v['gradients'][name] = o.array(gradient)
                    # None return deliberately preserves the original VJP.
                self.handles.append(group['params'][0].register_hook(observe))
            self.pending = (v, result['viewspace_points'])
            self.loss_input = (camera, result)
        return result

    def backward(self, original, tensor, local, *args, **kwargs):
        """Read the existing backward boundary; never recompute a training graph."""
        o.require(not self.report_mode and self.pending is not None, 'loss_backward_context')
        v, _ = self.pending
        camera, result = self.loss_input
        index = len(self.views)
        o.require(local['iteration'] == self.k and local['batch_idx'] == index
                  and local['batch_size'] == self.case['batch'], 'loss_view_boundary')
        o.require(local['viewpoint_cam'] is camera and local['render_pkg'] is result
                  and local['image'] is result['render'] and tensor is local['loss'], 'loss_graph_input')
        teacher, cpu_camera = local['batch_data'][index]
        o.require(camera.formal_frame == cpu_camera.formal_frame and camera.gt_alpha_mask is not None,
                  'loss_frame_mask')
        o.compare(local['gt_image'], teacher, 'loss_teacher', exact=True)
        components = {name: self.scalar(local[field]) for name, field in
                      (('l1', 'Ll1'), ('dssim', 'Lssim'), ('mask', 'Lopa_mask'))}
        total, expected = o.optimization_loss(self.p.inputs.config, components)
        actual = self.scalar(tensor)
        o.require(actual == expected, 'loss_weight_or_batch_division')
        o.require('loss' not in v, 'loss_duplicate_backward')
        record = dict(k=self.k, view=index, frame=list(camera.formal_frame), state_k=self.k-1,
                      components=components, weighted_total=total, batch_size=self.case['batch'],
                      backward_value=actual, expected=expected, backward_completed=False)
        v['loss'] = record
        self.trace['losses'].append(record)
        # Only the final view references survive to report; no live tensors in trace.
        self.last_loss = (tensor, local['Ll1'], record)
        value = original(tensor, *args, **kwargs)
        record['backward_completed'] = True
        return value

    def scalar(self, tensor):
        o.require(tensor.dtype == self.torch.float32 and tensor.numel() == 1, 'loss_scalar_type')
        value = float(tensor.detach().item())
        o.require(math.isfinite(value), 'loss_scalar_finite')
        return value

    def metric(self, original, image, teacher, *args, **kwargs):
        value = original(image, teacher, *args, **kwargs)
        if self.report_mode:
            pending = self.evaluation['pending']
            o.require(pending is not None, 'metric_without_fresh_render')
            o.compare(image, pending['image'], 'metric_fresh_image', exact=True)
            o.compare(teacher, pending['teacher'], 'metric_teacher_input', exact=True)
            # Observe the real PSNR component; independently aggregate test views below.
            component = float(o.array(value).mean(dtype=np.float32))
            o.require(math.isfinite(component), 'metric_component_finite')
            self.evaluation['views'].append(dict(frame=list(pending['frame']), psnr=component,
                teacher_sha256=hashlib.sha256(pending['teacher'].tobytes()).hexdigest()))
            self.evaluation['pending'] = None
        return value

    def adam_function(self, original, params, *args, **kwargs):
        o.require(self.active_adam is not None, 'Adam_outside_step')
        records = self.active_adam
        groups = self.model.optimizer.param_groups
        o.require(len(records) < len(groups), 'Adam_extra_functional_call')
        group = groups[len(records)]
        expected = [p for p in group['params'] if p.grad is not None]
        o.require(len(params) == len(expected) and all(p is q for p, q in zip(params, expected)),
                  'Adam_dispatch_group_owner')
        requested = {n: group[n] for n in ('foreach', 'fused', 'amsgrad', 'maximize', 'capturable',
                                         'differentiable', 'eps', 'weight_decay')}
        o.require(all(kwargs[n] == v for n, v in requested.items()), 'Adam_dispatch_settings')
        record = dict(k=self.k, group=group['name'], requested=requested,
                      lr=float(group['lr']), betas=list(group['betas']),
                      tensor_count=len(params), devices=[str(p.device) for p in params],
                      selected=None, backend_calls=0, completed=False)
        records.append(record)
        result = original(params, *args, **kwargs)
        o.require(record['backend_calls'] == 1 and record['completed'], 'Adam_dispatch_missing')
        return result

    def adam_backend(self, path, original, *args, **kwargs):
        o.require(self.active_adam and self.active_adam[-1]['backend_calls'] == 0, 'Adam_backend_context')
        record = self.active_adam[-1]
        record.update(selected=path, backend_calls=1)
        result = original(*args, **kwargs)
        record['completed'] = True
        return result

    def lr_inputs(self):
        """Check the real handoff against config and independently known geometry."""
        c = self.lr_config.optimization.learning_rate
        o.require(self.p.inputs.config is self.lr_config and self.p.opt.config is self.lr_config,
                  'LR_config_authority')
        for name, value in c._asdict().items():
            actual = getattr(self.p.opt, name)
            o.require(type(actual) is type(value) and actual == value, 'LR_consumer/'+name)
        for actual in (self.p.scene.cameras_extent, self.model.spatial_lr_scale):
            o.require(type(actual) is np.float32 and math.isfinite(actual)
                      and float(actual) == float(self.lr_extent), 'LR_geometry_scale')
        function = self.model.xyz_scheduler_args
        o.require(inspect.isfunction(function), 'LR_scheduler_function')
        closure = inspect.getclosurevars(function).nonlocals
        o.require(set(closure) == {'lr_init', 'lr_final', 'max_steps', 'lr_delay_steps', 'lr_delay_mult'},
                  'LR_scheduler_inputs')
        endpoints = o.lr_endpoints(self.lr_config, self.lr_extent)
        for name, expected in zip(('lr_init', 'lr_final'), endpoints):
            o.require(type(closure[name]) is np.float32 and float(closure[name]) == expected,
                      'LR_endpoint/'+name)
        for name, expected in (('max_steps', c.position_lr_max_steps), ('lr_delay_steps', 0), ('lr_delay_mult', 1.0)):
            o.require(type(closure[name]) is type(expected) and closure[name] == expected, 'LR_scheduler/'+name)
        groups = self.model.optimizer.param_groups
        o.require(len(groups) == len(o.GROUPS) and {g['name'] for g in groups} == set(o.GROUPS), 'LR_groups')
        parameters = []
        for g in groups:
            o.require(len(g['params']) == 1 and g['params'][0] is getattr(self.model, o.GROUPS[g['name']])
                      and g['params'][0].is_leaf, 'LR_group_owner/'+g['name'])
            parameters.append(g['params'][0])
        o.require(len({id(p) for p in parameters}) == len(o.GROUPS), 'LR_duplicate_owner')
        return groups

    def lr(self, original, k):
        self.finish_view()
        o.require(not self.views and k == self.k+1, 'update_sequence')
        o.require(self.verified_lr is None, 'LR_unconsumed_gate')
        self.k = k
        self.event('lr')
        self.lr_inputs()
        result = original(k)
        groups = self.lr_inputs()
        rates = o.learning_rates(self.lr_config, self.lr_extent, k)
        ideal = o.learning_rates(self.lr_config, self.lr_extent, k, ideal=True)
        values, records = [], []
        for group in groups:
            name, actual, expected = group['name'], group['lr'], rates[group['name']]
            expected_type = np.float32 if name == 't' else float
            if name == 'xyz' and expected != 0:
                expected_type = np.float64
            o.require(type(actual) is expected_type and math.isfinite(actual), 'LR_type_finite/'+name)
            error = abs(float(actual)-expected)
            record = dict(group=name, actual=float(actual), expected=expected, ideal=ideal[name],
                          type=type(actual).__name__, abs_error=error,
                          relative_error=error/abs(expected) if expected else 0.)
            records.append(record)
            o.require(actual == 0 if expected == 0 else math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0),
                      f'LR_schedule/{name}: {record}')
            values.append((name, group['params'][0], type(actual), float(actual)))
        # Mint only after ALL nine independent comparisons; immutable scalar inputs.
        self.verified_lr = (k, tuple(values))
        self.trace['learning_rates'].append(dict(k=k, groups=records))
        return result

    def take_verified_lr(self):
        """At the Adam boundary: reject omission, stale k/group/owner and mutation."""
        token = self.verified_lr
        o.require(token is not None and token[0] == self.k, 'LR_gate_missing_or_stale')
        groups = self.lr_inputs()
        entries = token[1]
        o.require(len(entries) == len(o.GROUPS) and {v[0] for v in entries} == set(o.GROUPS), 'LR_snapshot_groups')
        saved = {name: (param, scalar_type, value) for name, param, scalar_type, value in entries}
        for group in groups:
            param, scalar_type, value = saved[group['name']]
            actual = group['lr']
            o.require(group['params'][0] is param and type(actual) is scalar_type
                      and math.isfinite(actual) and float(actual) == value, 'LR_after_gate/'+group['name'])
        self.verified_lr = None  # Single consumption; exceptions are not retried.
        return {name: value for name, _, _, value in entries}

    def step(self, original, *args, **kwargs):
        rates = self.take_verified_lr()
        self.finish_view()
        m, c = self.model, self.p.inputs.config
        o.require(len(self.views) == self.case['batch'], 'batch_view_count')
        gradients = {}
        for g in m.optimizer.param_groups:
            name, parameter = g['name'], g['params'][0]
            o.require(g['betas'] == (.9, .999) and g['eps'] == 1e-15 and g['weight_decay'] == 0
                      and not any(g.get(n, False) for n in ('amsgrad', 'maximize', 'capturable', 'fused')), 'Adam_configuration')
            values = [v['gradients'][name] for v in self.views if v['gradients'][name] is not None]
            gradients[name] = None if parameter.grad is None else o.array(parameter.grad)
            if not values:
                o.require(parameter.grad is None, 'unobserved_gradient')
            else:
                o.require(parameter.grad is not None, 'lost_gradient')
                o.compare(gradients[name], sum(v.astype(np.float64) for v in values),
                          f'{self.k}/{name}/batch_VJP', self.oracle.records, 'statistic')
        o.require(gradients['t'] is not None, 'time_gradient_required_by_training')
        opt = self.p.opt
        can_grow = opt.densify_until_num_points < 0 or len(m._xyz) < opt.densify_until_num_points
        h = self.oracle.add_statistics(self.views, gradients['t'], self.k < opt.densify_until_iter, can_grow)
        before = snapshot(m)
        self.oracle.check(before, f'{self.k}/before_step')
        for old in self.retired:
            o.require(old.grad is None and old not in m.optimizer.state, 'retired_leaf_updated')
        self.event('step')
        self.oracle.update(gradients, rates)
        self.active_adam = []
        try:
            result = original(*args, **kwargs)
            o.require(len(self.active_adam) == len(o.GROUPS), 'Adam_functional_call_count')
        finally:
            self.trace['adam_dispatch'].extend(self.active_adam)
            self.active_adam = None
        after = snapshot(m)
        self.oracle.check(after, f'{self.k}/after_step')
        representative = o.delta_check(before, after, gradients, rates, self.oracle.records)
        self.trace['steps'].append(dict(k=self.k, rows=len(m._xyz), sh=after['sh'], h=h.tolist(),
            gradient_kind={n: 'None' if g is None else 'zero' if not np.any(g) else 'nonzero' for n, g in gradients.items()},
            representative_rows={n: np.flatnonzero(v).tolist() for n, v in representative.items()},
            lineage=list(self.oracle.lineage), before=before, after=after))
        self.views.clear()
        return result

    def zero(self, original, *args, **kwargs):
        self.event('zero')
        result = original(*args, **kwargs)
        o.require(all(g['params'][0].grad is None for g in self.model.optimizer.param_groups), 'zero_none')
        return result

    def replacement(self, original, *args, **kwargs):
        old = [g['params'][0] for g in self.model.optimizer.param_groups]
        result = original(*args, **kwargs)
        current = [g['params'][0] for g in self.model.optimizer.param_groups]
        self.retired.extend(p for p in old if not any(p is q for q in current))
        return result

    def clone(self, original, grads, threshold, extent, *args):
        self.event('clone')
        o.compare(grads, self.oracle.gradient(), 'clone_gradient', tolerance='statistic')
        before = snapshot(self.model)
        selected = self.oracle.selection(o.array(grads), threshold, self.model.percent_dense*extent)
        rows = np.r_[np.arange(len(self.oracle.lineage)), selected]
        self.oracle.clone(selected)
        result = original(grads, threshold, extent, *args)
        after = snapshot(self.model)
        self.oracle.check(after, 'clone')
        o.exact_carry(before, after, rows, len(rows)-len(selected))
        self.trace['population'].append(dict(k=self.k, operation='clone', selected=selected.tolist()))
        return result

    def split(self, original, grads, threshold, extent, *args, **kwargs):
        self.event('split')
        o.require(kwargs.get('N', 2) == 2, 'split_N')
        before = snapshot(self.model)
        self.split_rows = self.oracle.selection(o.array(grads), threshold, self.model.percent_dense*extent, split=True)
        count = len(self.oracle.lineage)
        selected = self.split_rows.copy()
        self.expected_mask = np.r_[np.isin(np.arange(count), selected), np.zeros(2*len(selected), dtype=bool)]
        self.normal_calls = 0
        result = original(grads, threshold, extent, *args, **kwargs)
        o.require(self.normal_calls == 1, 'one_real_normal_call')
        self.split_rows = None
        after = snapshot(self.model)
        rows = np.r_[np.flatnonzero(~np.isin(np.arange(count), selected)), np.tile(selected, 2)]
        self.oracle.check(after, 'split')
        o.exact_carry(before, after, rows, count-len(selected), computed=('xyz', 't', 'scaling', 'scaling_t'))
        self.trace['population'].append(dict(k=self.k, operation='split', selected=selected.tolist()))
        return result

    def normal(self, original, *args, **kwargs):
        o.require(self.split_rows is not None and not args, 'unexpected_normal_call')
        self.normal_calls += 1
        rows = np.tile(self.split_rows, 2)
        std = np.exp(np.concatenate([self.oracle.groups[n]['p'][rows] for n in ('scaling', 'scaling_t')], axis=1))
        o.compare(kwargs['mean'], np.zeros_like(std), 'normal_mean', exact=True)
        o.compare(kwargs['std'], std, 'normal_std')
        sample = original(*args, **kwargs)
        self.oracle.split(self.split_rows, o.array(sample))
        self.trace['population'].append(dict(k=self.k, operation='normal', sample=o.array(sample)))
        return sample

    def prune(self, original, mask):
        before = snapshot(self.model)
        if self.expected_mask is not None:
            expected = self.expected_mask
            self.expected_mask = None
        else:
            o.require(self.context is not None, 'prune_context')
            expected = self.oracle.prune_mask(*self.context)
        o.compare(mask, expected, 'prune_mask', exact=True)
        self.oracle.prune(expected)
        self.event('prune')
        result = original(mask)
        after = snapshot(self.model)
        self.oracle.check(after, 'prune')
        o.exact_carry(before, after, np.flatnonzero(~expected))
        self.trace['population'].append(dict(k=self.k, operation='prune', mask=expected.tolist()))
        return result

    def population(self, original, gradient, opacity, extent, screen, *args, **kwargs):
        self.event('population')
        self.context = (opacity, extent, screen)
        result = original(gradient, opacity, extent, screen, *args, **kwargs)
        if not kwargs.get('prune_only', False):
            self.oracle.end_window()
        self.oracle.check(snapshot(self.model), 'population_end')
        self.trace['population'].append(dict(k=self.k, operation='end', prune_only=kwargs.get('prune_only', False),
                                            rows=len(self.oracle.lineage)))
        self.context = None
        return result

    def reset(self, original):
        self.event('reset')
        before = snapshot(self.model)
        self.oracle.reset()
        result = original()
        after = snapshot(self.model)
        self.oracle.check(after, 'reset')
        for n in o.GROUPS:
            if n != 'opacity':
                for field in ('p', 'm', 'v', 'step'):
                    o.compare(before['groups'][n][field], after['groups'][n][field], 'reset_unaffected', exact=True)
        for n in o.STATS:
            o.compare(before['stats'][n], after['stats'][n], 'reset_statistics', exact=True)
        o.compare(self.torch.sigmoid(self.model._opacity), 1/(1+np.exp(-self.oracle.groups['opacity']['p'])),
                  'reset_activated', self.oracle.records, 'opacity')
        self.trace['population'].append(dict(k=self.k, operation='reset'))
        return result

    def save(self, value, path):
        self.event('save')
        capture, k = value
        o.require(k == self.k and Path(path) == Path(self.p.scene.model_path)/f'chkpnt{k}.pth', 'save_label_path')
        o.require(len(capture) == 19, 'capture_boundary')
        indices = dict(xyz=1, f_dc=2, f_rest=3, scaling=4, rotation=5, opacity=6, t=13, scaling_t=14, rotation_r=15)
        snap = snapshot(self.model)
        self.oracle.check(snap, 'save_completed')
        for name, index in indices.items():
            o.compare(capture[index], snap['groups'][name]['p'], 'capture/'+name, exact=True)
        self.trace['saves'].append(dict(k=k, state=snap))
        # Terminal observer, intentionally NOT checkpoint serialization acceptance.

    def report(self, original, tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
               testing_iterations, scene, renderFunc, renderArgs, loss_dict=None):
        self.finish_view()
        k = iteration
        o.require(k == self.k, 'report_completed_label')
        if k == 0:
            o.require(self.last_loss is None and self.scalar(loss) == self.scalar(Ll1) == 0, 'report_initial_loss')
            optimization = None
        else:
            o.require(self.last_loss is not None, 'report_missing_optimization_input')
            old_loss, old_l1, optimization = self.last_loss
            o.require(loss is old_loss and Ll1 is old_l1 and optimization['k'] == k
                      and optimization['view'] == self.case['batch']-1
                      and self.scalar(loss) == optimization['backward_value']
                      and self.scalar(Ll1) == optimization['components']['l1'], 'report_final_view_input')
        self.event('report')
        snap = snapshot(self.model)
        self.oracle.check(snap, 'report_completed')
        if k in self.p.inputs.config.save_updates:
            o.require(self.trace['saves'][-1]['k'] == k, 'save_before_report')
            equal_snapshot(self.trace['saves'][-1]['state'], snap)
        self.report_mode = True
        self.evaluation = dict(state=snap, views=[], pending=None)
        start = self.trace['forwards']
        try:
            result = original(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                              testing_iterations, scene, renderFunc, renderArgs, loss_dict)
        finally:
            self.report_mode = False
        expected = 7 if k == 0 or k in self.p.inputs.config.reporting.test_updates else 0
        o.require(self.trace['forwards']-start == expected, 'report_forward_schedule')
        o.require(math.isfinite(result), 'report_metric_finite')
        views = self.evaluation['views']
        o.require(self.evaluation['pending'] is None and len(views) == expected, 'metric_view_count')
        measured = None
        if expected:
            o.require([v['frame'] for v in views] == [list(f) for f in self.evaluation_frames],
                      'metric_frame_schedule')
            measured = math.fsum(v['psnr'] for v in views if v['frame'][0] == 'test')/2
            o.compare(result, measured, 'report_completed_metric')
        else:
            o.require(result == 0, 'report_no_evaluation_sentinel')
        equal_snapshot(snap, snapshot(self.model))
        self.trace['reports'].append(dict(k=k, forwards=expected, metric=measured, returned=result, state=snap,
            kind='initial_diagnostic' if k == 0 else ('completed_evaluation' if expected else 'not_scheduled'),
            optimization=optimization, evaluation_views=views))
        self.last_loss, self.evaluation = None, None
        return result

    def run(self, writer):
        loader, outer = self.p.dataloader, self
        class Batches:
            def __iter__(self):
                outer.epoch += 1
                for batch in loader:
                    outer.event('fetch')
                    outer.trace['batches'].append(dict(epoch=outer.epoch,
                        frames=[camera.formal_frame.index for _, camera in batch]))
                    o.require(len(outer.trace['batches']) <= outer.case['n'], 'N_plus_one_fetch')
                    yield batch
        m = self.model
        with ExitStack() as stack:
            backward = self.torch.Tensor.backward
            def observed_backward(tensor, *args, **kwargs):
                frame = inspect.currentframe().f_back
                try:
                    o.require(frame.f_code is outer.training_code, 'loss_backward_caller')
                    return outer.backward(backward, tensor, frame.f_locals, *args, **kwargs)
                finally:
                    del frame
            stack.enter_context(patch.object(self.torch.Tensor, 'backward', observed_backward))
            for name, path in (('_single_tensor_adam', 'single'), ('_multi_tensor_adam', 'foreach'),
                               ('_fused_adam', 'fused')):
                original = getattr(self.adam_module, name)
                def observed_backend(*args, _original=original, _path=path, **kwargs):
                    return outer.adam_backend(_path, _original, *args, **kwargs)
                stack.enter_context(patch.object(self.adam_module, name, observed_backend))
            for owner, name, wrapper in (
                (self.train, 'psnr', self.metric), (self.adam_module, 'adam', self.adam_function),
                (m, 'update_learning_rate', self.lr), (m.optimizer, 'step', self.step),
                (m.optimizer, 'zero_grad', self.zero), (m, 'densify_and_clone', self.clone),
                (m, 'densify_and_split', self.split), (m, 'prune_points', self.prune),
                (m, 'densify_and_prune', self.population), (m, 'reset_opacity', self.reset),
                (self.torch, 'normal', self.normal), (self.train, 'training_report', self.report),
                (m, 'cat_tensors_to_optimizer', self.replacement),
                (m, '_prune_optimizer', self.replacement), (m, 'replace_tensor_to_optimizer', self.replacement)):
                original = getattr(owner, name)
                def observed(*args, _original=original, _wrapper=wrapper, **kwargs):
                    return _wrapper(_original, *args, **kwargs)
                stack.enter_context(patch.object(owner, name, observed))
            stack.enter_context(patch.object(self.train, 'render', self.render))
            stack.enter_context(patch.object(self.torch, 'save', self.save))
            try:
                completed = self.train.continue_formal_training(self.p._replace(dataloader=Batches()), writer)
            except BaseException as error:
                self.trace['error'] = f'{type(error).__name__}: {error}'
                self.trace['failure_snapshot'] = snapshot(m)
                tb = error.__traceback__
                while tb:
                    if tb.tb_frame.f_code.co_name == 'training':
                        self.trace['failed_completed'] = tb.tb_frame.f_locals.get('completed', 0)
                    tb = tb.tb_next
                raise
            finally:
                for handle in self.handles:
                    handle.remove()
                self.handles.clear()
                self.loss_input, self.last_loss, self.evaluation, self.active_adam = None, None, None, None
        self.validate_completed(completed)
        self.trace['status'] = 'passed'
        return self.trace

    def validate_completed(self, completed):
        n = self.case['n']
        o.require(completed == n and len(self.trace['batches']) == len(self.trace['steps']) == n, 'exact_N')
        o.require([s['k'] for s in self.trace['saves']] == list(self.p.inputs.config.save_updates), 'save_schedule')
        o.require([r['k'] for r in self.trace['reports']] == list(range(n+1)), 'report_schedule')
        if self.case['kind'] != 'zero_lr':
            o.require(any(rows for s in self.trace['steps'] for rows in s['representative_rows'].values()), 'representative_update_missing')
        operations = self.trace['population']
        for event in operations:
            if event['operation'] in ('clone', 'split') and event['selected'] and event['k'] < n:
                following = self.trace['steps'][event['k']]
                children = [i for i, row in enumerate(following['lineage']) if row[0] == event['operation']]
                o.require(children and any(set(children) & set(rows) for rows in following['representative_rows'].values()),
                          'next_child_update_missing')
        kind = self.case['kind']
        ends = [e for e in operations if e['operation'] == 'end']
        if kind in ('clone', 'split'):
            expected = kind
            o.require(any(e['operation'] == expected and len(e['selected']) == 16 for e in operations), 'growth_coverage')
        if kind == 'threshold':
            o.require([(e['prune_only'], e['rows']) for e in ends] == [(True, 14), (False, 28)], 'threshold_overshoot')
        if kind == 'visibility':
            o.require({0, 1, 2} <= {h for s in self.trace['steps'] for h in s['h']}, 'visibility_coverage')
        if kind in ('mixed', 'reset', 'size'):
            o.require(any(e['operation'] == 'reset' for e in operations), 'reset_coverage')
        if kind == 'mixed':
            o.require(any(e['operation'] == 'clone' and e['selected'] for e in operations)
                      and any(e['operation'] == 'split' and e['selected'] for e in operations)
                      and any(e['operation'] == 'prune' and any(e['mask']) for e in operations), 'mixed_coverage')
        if kind == 'size':
            # Final size prune must remove both original row 0 and row 1.
            final_prunes = [e for e in operations if e['operation'] == 'prune']
            o.require(final_prunes[-1]['mask'][:2] == [True, True], 'screen_world_prune_coverage')
        output = Path(self.p.scene.model_path)
        expected_png = 28*(1+len(self.p.inputs.config.reporting.test_updates))
        files = list((output/'renders_png').rglob('*.png'))
        o.require(len(files) == expected_png, 'real_PNG_count')
        from PIL import Image
        for file in files:
            with Image.open(file) as image:
                o.require(image.size == (256, 256), 'real_PNG_dimensions')
                image.verify()
        self.trace['png_count'] = len(files)


def execute_case(p, case, budget, renderer=None, observer_callback=None):
    import formal_entry
    import train
    observer = Observation(p, case, renderer or train.render, budget)
    if observer_callback:
        observer_callback(observer)
    claim = formal_entry.OutputClaim(p.inputs)
    writer = None
    try:
        writer = train.begin_formal_output(p, claim)
        return observer.run(writer)
    finally:
        if writer is not None:
            writer.close()
        claim.close()
