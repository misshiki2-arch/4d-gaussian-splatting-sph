"""Step 12 CPU evidence: real consumers/Adam/metrics, explicit GPU seams only."""
from contextlib import ExitStack
import copy
import json
import math
from pathlib import Path
import sys
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import training_update_oracle as o
import training_update_runtime as runtime
from formal_training_fixture import CPUCase


def small_snapshot():
    groups = {n: dict(p=np.arange(1, 5, dtype=np.float32).reshape(4, 1)*.1,
                     m=np.zeros((4, 1)), v=np.zeros((4, 1)), step=0) for n in o.GROUPS}
    return dict(groups=groups, stats={n: np.zeros(4 if n == 'max_radii2D' else (4, 1)) for n in o.STATS}, sh=(0, 0))


class OracleTests(unittest.TestCase):
    def test_hand_calculated_two_steps_sign_zero_none_and_zero_lr(self):
        p, m, v = (np.array([x], dtype=np.float64) for x in (1., 0., 0.))
        p1, m1, v1, s1 = o.adam(p, m, v, 0, [.2], .01)
        self.assertAlmostEqual(p1[0], .99, places=12)
        self.assertAlmostEqual(m1[0], .02, places=15)
        self.assertAlmostEqual(v1[0], .00004, places=15)
        p2, m2, v2, s2 = o.adam(p1, m1, v1, s1, [-.1], .01)
        self.assertAlmostEqual(m2[0], .008, places=15)
        self.assertAlmostEqual(v2[0], .00004996, places=15)
        self.assertAlmostEqual(p2[0], .99-.01*(.008/.19)/(math.sqrt(.00004996/.001999)+1e-15), places=14)
        skipped = o.adam(p2, m2, v2, s2, None, .01)
        for a, b in zip(skipped, (p2, m2, v2, s2)):
            np.testing.assert_array_equal(a, b)
        zero = o.adam(p2, m2, v2, s2, [0.], .01)
        self.assertEqual(zero[3], 3)
        self.assertNotEqual(zero[0][0], p2[0])
        frozen = o.adam(p2, m2, v2, s2, [.2], 0.)
        np.testing.assert_array_equal(frozen[0], p2)
        self.assertNotEqual(frozen[1][0], m2[0])

    def test_child_scalar_step_inheritance_reset_and_next_moment(self):
        trajectory = o.Trajectory(small_snapshot())
        gradients = {n: np.full((4, 1), .02) for n in o.GROUPS}
        rates = {n: .01 for n in o.GROUPS}
        trajectory.update(gradients, rates)
        trajectory.clone(np.array([1, 3]))
        self.assertEqual(trajectory.groups['xyz']['step'], 1)
        np.testing.assert_array_equal(trajectory.groups['xyz']['m'][-2:], 0)
        trajectory.reset()
        self.assertEqual(trajectory.groups['opacity']['step'], 1)
        trajectory.update({n: np.full((6, 1), .02) for n in o.GROUPS}, rates)
        self.assertAlmostEqual(trajectory.groups['xyz']['m'][-1, 0], .002)
        self.assertEqual(trajectory.groups['opacity']['step'], 2)

    def test_batch_visibility_radius_and_signed_time(self):
        views = [dict(visible=np.array([1, 1, 0], dtype=bool), radii=np.array([2, 4, 99]),
                      screen=np.array([[3., 4, 0], [0, 2, 0], [0, 0, 0]])),
                 dict(visible=np.array([1, 0, 0], dtype=bool), radii=np.array([3, 7, 100]),
                      screen=np.array([[0., 2, 0], [0, 0, 0], [0, 0, 0]]))]
        h, union, radius, screen, t = o.batch_statistics(views, [[.2], [-.3], [0.]])
        np.testing.assert_array_equal(h, [2, 1, 0])
        np.testing.assert_array_equal(radius, [3, 7, 100])
        np.testing.assert_array_equal(screen.ravel(), [7, 4, 0])
        np.testing.assert_allclose(t.ravel(), [.2, -.6, 0], atol=0)
        np.testing.assert_array_equal(union, [True, True, False])

    def test_independent_rotation_repeat_and_selection(self):
        np.testing.assert_array_equal(o.rotate(np.array([1., 2, 3, 4]), np.array([1., 0, 0, 0]),
                                               np.array([1., 0, 0, 0])), [1, 2, 3, 4])
        q = np.array([math.cos(.2), math.sin(.2), 0, 0])
        self.assertAlmostEqual(np.linalg.norm(o.rotate(np.arange(4.), q, q)), math.sqrt(14))
        tr = o.Trajectory(small_snapshot())
        tr.groups['scaling']['p'] = np.log(np.array([[.01], [.04], [.05], [.06]]))
        grads = np.array([[1], [2], [3], [0]])
        np.testing.assert_array_equal(tr.selection(grads, .5, .02), [0])
        np.testing.assert_array_equal(tr.selection(grads, .5, .02, split=True), [1, 2])
        # Repeat is parent-list repeated twice, never repeat_interleave.
        np.testing.assert_array_equal(np.tile([1, 2], 2), [1, 2, 1, 2])

    def test_negative_corrupt_moment_step_rows_and_nonfinite(self):
        initial = small_snapshot()
        trajectory = o.Trajectory(initial)
        for name, field, value in [('xyz', 'm', 1e-8), ('xyz', 'v', 1e-12), ('xyz', 'step', 1), ('f_rest', 'p', float('nan'))]:
            bad = copy.deepcopy(initial)
            if field == 'step':
                bad['groups'][name][field] = value
            else:
                bad['groups'][name][field][0] = value
            with self.subTest(field=field), self.assertRaises(AssertionError):
                trajectory.check(bad, 'negative')
        bad = copy.deepcopy(initial)
        bad['groups']['t']['p'] = bad['groups']['t']['p'][::-1]
        with self.assertRaisesRegex(AssertionError, 'mismatch'):
            o.exact_carry(initial, bad, np.arange(4))

    def test_delta_detects_missing_step_independently_of_parent_tolerance(self):
        before = small_snapshot()
        gradients = {n: np.full((4, 1), .01) for n in o.GROUPS}
        with self.assertRaisesRegex(AssertionError, 'delta_mismatch'):
            o.delta_check(before, copy.deepcopy(before), gradients, {n: .001 for n in o.GROUPS}, [])

    def test_fixture_schedule_and_geometric_margins(self):
        self.assertEqual(len(runtime.CASES), 12)
        self.assertEqual(sum(c['n'] for c in runtime.CASES), 35)
        forwards = 0
        for case in runtime.CASES:
            v = runtime.config_value(Path('/synthetic/data'), Path('/synthetic/output'), case)
            forwards += case['n']*case['batch']+7*(1+len(v['reporting']['test_updates']))
            self.assertLessEqual(case['n'], 6)
            self.assertLessEqual(case['batch'], 2)
            self.assertEqual(v['optimization']['loss']['lambda_dssim'], .25)
            self.assertEqual(v['optimization']['loss']['lambda_opa_mask'], .125)
        self.assertLessEqual(forwards, runtime.MAX_FORWARDS)
        self.assertLess(.04, .1*.55)  # Near screen-only, cloned.
        self.assertGreater(3*192*.04/.5, 20)
        self.assertGreater(.06, .1*.55)  # Far world-only, cloned.
        self.assertLess(3*192*.06/6.5, 20)
        self.assertGreater(.07, .065)  # Split parent, surviving child size.
        self.assertLess(.07/1.6, .1*.55)


class Event:
    def __init__(self, **kw):
        pass
    def record(self):
        pass
    def elapsed_time(self, other):
        return 0.


class LRConnectionTests(CPUCase):
    """Real LR consumers/CPU Adam, without invoking the training/render loop."""
    def prepared(self):
        case = runtime.CASES[10]
        p = runtime.prepare(self.f.base/'lr', case)
        return p, case

    def observer(self, p, case):
        return runtime.Observation(p, case, Mock(side_effect=AssertionError('no render')), dict(forwards=0))

    def test_schedule_rounding_endpoints_zero_and_scalar_controls(self):
        from utils.general_utils import get_expon_lr_func
        p, _ = self.prepared()
        c = p.inputs.config
        for extent in (np.float32(.55), float(.55), np.float64(.55)):
            lr = c.optimization.learning_rate
            actual = get_expon_lr_func(lr.position_lr_init*extent, lr.position_lr_final*extent,
                                       max_steps=lr.position_lr_max_steps)
            temporal = lr.position_t_lr_init*extent
            for k in (0, 1, 10, 20, 21):
                with self.subTest(type=type(extent).__name__, k=k):
                    expected = o.learning_rates(c, extent, k)
                    self.assertTrue(math.isclose(actual(k), expected['xyz'], rel_tol=1e-12, abs_tol=0))
                    self.assertEqual(expected['t'], float(temporal))
                    self.assertEqual(expected['f_rest'], expected['f_dc']/20)
                    self.assertEqual(expected['scaling'], expected['scaling_t'])
                    self.assertEqual(expected['rotation'], expected['rotation_r'])
            if type(extent) is np.float32:
                a, b, _ = o.lr_endpoints(c, extent)
                old = math.exp(.95*math.log(a)+.05*math.log(b))
                self.assertFalse(math.isclose(actual(1), old, rel_tol=1e-12, abs_tol=0))
                self.assertNotEqual(o.learning_rates(c, extent, 1)['xyz'],
                                    o.learning_rates(c, extent, 1, ideal=True)['xyz'])
            else:
                self.assertEqual(o.learning_rates(c, extent, 1), o.learning_rates(c, extent, 1, ideal=True))
        zero = lr._replace(**{name: 0. for name in lr._fields if name != 'position_lr_max_steps'})
        zc = c._replace(optimization=c.optimization._replace(learning_rate=zero))
        for k in (0, 1, 10, 20, 21):
            self.assertEqual(set(o.learning_rates(zc, np.float32(.55), k).values()), {0.})
            self.assertEqual(get_expon_lr_func(np.float32(0), np.float32(0), max_steps=20)(k), 0.)
        for extent in (True, 1, np.float16(.55), np.array(.55), float('nan'), float('inf'), 0.):
            with self.subTest(invalid_extent=repr(extent)), self.assertRaises(AssertionError):
                o.learning_rates(c, extent, 1)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_gate_config_geometry_endpoint_owner_and_schedule_negatives(self):
        from utils.general_utils import get_expon_lr_func
        p, case = self.prepared()
        m = p.gaussians
        groups = {g['name']: g for g in m.optimizer.param_groups}
        initial = {n: g['lr'] for n, g in groups.items()}
        a, b, _ = o.lr_endpoints(p.inputs.config, runtime.fixture_extent())
        mutations = ('extent', 'scale', 'config', 'max_steps', 'endpoints', 'k', 'linear',
                     'missing_update', 't_schedule', 'constant', 'shared', 'group', 'owner',
                     'nonfinite', 'unsupported_type')
        for mutation in mutations:
            with self.subTest(mutation=mutation), ExitStack() as stack:
                for name, group in groups.items():
                    stack.enter_context(patch.dict(group, lr=initial[name]))
                obs = self.observer(p, case)
                if mutation in ('extent', 'scale'):
                    obj, attr = (p.scene, 'cameras_extent') if mutation == 'extent' else (m, 'spatial_lr_scale')
                    stack.enter_context(patch.object(obj, attr, np.float32(2*float(runtime.fixture_extent()))))
                if mutation == 'config':
                    fields = p.inputs.config.optimization.learning_rate._asdict()
                    fields['position_lr_init'] *= 2
                    opt = SimpleNamespace(config=p.inputs.config, **fields)
                    obs.p = p._replace(opt=opt)
                if mutation in ('max_steps', 'endpoints'):
                    fn = get_expon_lr_func(np.float32(b if mutation == 'endpoints' else a),
                                          np.float32(a if mutation == 'endpoints' else b),
                                          max_steps=21 if mutation == 'max_steps' else 20)
                    stack.enter_context(patch.object(m, 'xyz_scheduler_args', fn))
                if mutation == 'group':
                    stack.enter_context(patch.dict(groups['f_rest'], name='f_dc'))
                if mutation == 'owner':
                    stack.enter_context(patch.dict(groups['xyz'], params=[m._t]))
                def wrong(k):
                    if mutation == 'missing_update':
                        return None
                    result = m.update_learning_rate(k+1 if mutation == 'k' else k)
                    if mutation == 'linear':
                        groups['xyz']['lr'] = np.float64(.95*a+.05*b)
                    elif mutation == 't_schedule':
                        groups['t']['lr'] = np.float32(result)
                    elif mutation == 'constant':
                        groups['f_dc']['lr'] *= 2
                    elif mutation == 'shared':
                        groups['scaling_t']['lr'] *= 2
                    elif mutation == 'nonfinite':
                        groups['xyz']['lr'] = np.float64('nan')
                    elif mutation == 'unsupported_type':
                        groups['t']['lr'] = float(groups['t']['lr'])
                    return result
                with self.assertRaisesRegex(AssertionError, 'LR_'):
                    obs.lr(wrong, 1)
                self.assertIsNone(obs.verified_lr)
                step = Mock()
                with self.assertRaisesRegex(AssertionError, 'LR_gate'):
                    obs.step(step)
                step.assert_not_called()
        self.assertFalse(m.optimizer.state)

    def test_snapshot_missing_stale_group_owner_and_post_gate_mutation(self):
        p, case = self.prepared()
        m = p.gaussians
        for mutation in ('missing', 'k', 'snapshot_group', 'group_owner', 'value', 'type'):
            with self.subTest(mutation=mutation), ExitStack() as stack:
                obs = self.observer(p, case)
                if mutation != 'missing':
                    obs.lr(m.update_learning_rate, 1)
                if mutation == 'k':
                    obs.k = 2
                elif mutation == 'snapshot_group':
                    k, entries = obs.verified_lr
                    obs.verified_lr = (k, entries[:-1])
                elif mutation == 'group_owner':
                    stack.enter_context(patch.dict(m.optimizer.param_groups[0], params=[m._t]))
                elif mutation == 'value':
                    g = m.optimizer.param_groups[0]
                    # One ULP is inside the LR relative tolerance but must not evade the snapshot.
                    stack.enter_context(patch.dict(g, lr=np.nextafter(g['lr'], np.float64('inf'))))
                elif mutation == 'type':
                    g = m.optimizer.param_groups[0]
                    stack.enter_context(patch.dict(g, lr=float(g['lr'])))
                step = Mock()
                with self.assertRaisesRegex(AssertionError, 'LR_'):
                    obs.step(step)
                step.assert_not_called()
        self.assertFalse(m.optimizer.state)

    def test_verified_rates_feed_independent_two_step_cpu_adam(self):
        p, case = self.prepared()
        m, torch = p.gaussians, self.torch
        obs = self.observer(p, case)
        trajectory = o.Trajectory(runtime.snapshot(m))
        for k in (1, 2):
            observed = m.update_learning_rate
            with patch.object(m, 'update_learning_rate', wraps=observed) as update:
                obs.lr(update, k)
                update.assert_called_once_with(k)
            token = obs.verified_lr
            self.assertIsInstance(token, tuple)
            self.assertIsInstance(token[1], tuple)
            rates = obs.take_verified_lr()
            self.assertEqual(set(rates), set(o.GROUPS))
            self.assertTrue(all(type(v) is float for v in rates.values()))
            with self.assertRaisesRegex(AssertionError, 'LR_gate'):
                obs.take_verified_lr()
            gradients = {}
            for group in m.optimizer.param_groups:
                # Controlled unit-test gradients, not renderer/CUDA evidence.
                group['params'][0].grad = torch.full_like(group['params'][0], .02 if k == 1 else -.01)
                gradients[group['name']] = o.array(group['params'][0].grad)
                self.assertEqual(rates[group['name']], float(group['lr']))
            before = runtime.snapshot(m)
            trajectory.update(gradients, rates)
            m.optimizer.step()
            trajectory.check(runtime.snapshot(m), 'verified_LR_to_Adam')
            o.delta_check(before, runtime.snapshot(m), gradients, rates, [])
            m.optimizer.zero_grad(set_to_none=True)
        self.assertFalse(torch.cuda.is_initialized())


class ConsumerTests(CPUCase):
    def renderer(self, camera, model, pipe, background, **kwargs):
        """Explicit differentiable CPU control renderer, never CUDA evidence."""
        torch = self.torch
        screen = torch.zeros_like(model._xyz, requires_grad=True)
        dt = camera.timestamp-model._t[:, 0]
        visible = (dt.abs() < 3*model._scaling_t[:, 0].exp()).detach()
        signal = screen[:, :2][visible].sum()*.001
        for group in model.optimizer.param_groups:
            p = group['params'][0]
            weights = torch.arange(1, len(p)+1, dtype=p.dtype).reshape(len(p), *([1]*(p.ndim-1)))
            signal = signal+(p*weights)[visible].sum()*.00001
        h, w = camera.image_height, camera.image_width
        ramp = torch.linspace(0., .1, h*w).reshape(h, w)
        rgb = (.5+ramp+signal).expand(3, h, w)
        depth = torch.linspace(1., 2., h*w).reshape(1, h, w)
        z = (model._xyz[:, 2]-camera.camera_center[2]).detach()
        radii = torch.ceil(3*192*model._scaling.exp().max(1).values.detach()/z)
        radii = torch.where(visible, radii, torch.zeros_like(radii))
        return dict(render=rgb, alpha=(.3+model._opacity[visible].sum()*.0001).expand(1, h, w),
                    depth=depth, radii=radii, visibility_filter=visible, viewspace_points=screen)

    def exercise_case(self, case, base=None, callback=None, budget=None):
        p = runtime.prepare(base or self.f.base/case['name'], case)
        with patch.object(self.torch.cuda, 'Event', Event):
            trace = runtime.execute_case(p, case, budget or dict(forwards=0), self.renderer, callback)
        self.assertFalse(self.torch.cuda.is_initialized())
        return p, trace

    def test_real_loop_all_cases_real_report_and_independent_trajectory(self):
        budget = dict(forwards=0)
        traces = []
        for case in runtime.CASES:
            with self.subTest(case=case['name']):
                p, trace = self.exercise_case(case, budget=budget)
                self.assertEqual(trace['status'], 'passed')
                self.assertEqual(len(trace['steps']), case['n'])
                self.assertTrue(all(not getattr(getattr(p.gaussians, name), '_backward_hooks', {}) for name in o.GROUPS.values()))
                self.assertFalse(list(Path(p.scene.model_path).glob('*.pth')))
                self.assertEqual(len(trace['losses']), case['n']*case['batch'])
                self.assertTrue(all(v['backward_completed'] and v['backward_value'] == v['expected']
                                    for v in trace['losses']))
                self.assertEqual([(v['k'], v['view']) for v in trace['losses']],
                                 [(k, j) for k in range(1, case['n']+1) for j in range(case['batch'])])
                self.assertEqual(len(trace['adam_dispatch']), 9*case['n'])
                self.assertTrue(all(d['selected'] == 'single' and d['completed'] and d['backend_calls'] == 1
                                    and d['requested']['foreach'] is None and d['requested']['fused'] is None
                                    for d in trace['adam_dispatch']))
                for r in trace['reports']:
                    if r['k'] == 0:
                        self.assertEqual(r['kind'], 'initial_diagnostic')
                        self.assertIsNone(r['optimization'])
                    else:
                        self.assertEqual(r['optimization']['view'], case['batch']-1)
                        self.assertEqual(r['optimization']['state_k'], r['k']-1)
                    if not r['forwards']:
                        self.assertEqual(r['kind'], 'not_scheduled')
                        self.assertIsNone(r['metric'])
                        self.assertEqual(r['returned'], 0)
                # Same serializer used by the future CUDA entry; no tensors/aliases in new records.
                evidence = {n: trace[n] for n in ('case', 'losses', 'adam_dispatch', 'adam_implementation')}
                evidence['reports'] = [{n: v for n, v in r.items() if n != 'state'} for r in trace['reports']]
                encoded = json.dumps(runtime.serializable(evidence), allow_nan=False)
                self.assertEqual(json.loads(encoded)['case'], case['name'])
                print('FIX2_OBSERVATIONS '+encoded)
                traces.append(trace)
        self.assertEqual(sum(len(t['steps']) for t in traces), 35)
        self.assertLessEqual(budget['forwards'], 500)
        self.assertEqual(self.f.seed.call_count, 12)

    def test_snapshot_alias_and_wrapper_noninterference(self):
        case = runtime.CASES[0]
        held = []
        p, trace = self.exercise_case(case, callback=held.append)
        snap = trace['steps'][0]['after']
        expected = copy.deepcopy(snap)
        with self.torch.no_grad():
            p.gaussians._xyz.add_(1)
        runtime.equal_snapshot(snap, expected)
        self.assertFalse(held[0].handles)
        # Comparison rejects stale/computed state; it cannot pass by aliasing.
        with self.assertRaisesRegex(AssertionError, 'mismatch'):
            runtime.equal_snapshot(snap, runtime.snapshot(p.gaussians))

    def test_step_exception_propagates_no_next_batch_hooks_removed(self):
        case = runtime.CASES[0]
        p = runtime.prepare(self.f.base/'failed-step', case)
        held = []
        with patch.object(self.torch.cuda, 'Event', Event), \
                patch.object(p.gaussians.optimizer, 'step', side_effect=RuntimeError('injected_step')) as step:
            with self.assertRaisesRegex(RuntimeError, 'injected_step'):
                runtime.execute_case(p, case, dict(forwards=0), self.renderer, held.append)
        step.assert_called_once()
        self.assertEqual(held[0].trace['failed_completed'], 0)
        self.assertEqual(len(held[0].trace['batches']), 1)
        self.assertEqual(held[0].trace['saves'], [])
        self.assertFalse(held[0].handles)
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_population_save_report_failures_preserve_completed_boundary(self):
        # Actual failure boundaries, not retry/rollback. Initial report remains real.
        for stage, completed in (('population', 1), ('save', 2), ('report', 2)):
            case = runtime.CASES[10]  # G8: simultaneous final events.
            p = runtime.prepare(self.f.base/('fail-'+stage), case)
            held = []
            def remember(obs):
                held.append(obs)
                original = getattr(obs, stage)
                def fail(*a, **kw):
                    if obs.k == 2:
                        raise RuntimeError('injected_'+stage)
                    return original(*a, **kw)
                setattr(obs, stage, fail)
            with self.subTest(stage=stage), patch.object(self.torch.cuda, 'Event', Event):
                with self.assertRaisesRegex(RuntimeError, 'injected_'+stage):
                    runtime.execute_case(p, case, dict(forwards=0), self.renderer, remember)
            self.assertEqual(held[0].trace['failed_completed'], completed)
            self.assertEqual(len(held[0].trace['batches']), 2)
            self.assertFalse(held[0].handles)


class ObservationTests(CPUCase):
    renderer = ConsumerTests.renderer
    exercise_case = ConsumerTests.exercise_case

    def test_loss_coefficient_omitted_and_double_division_refused_before_backward(self):
        case = runtime.CASES[-1]  # B=2, all three real loss branches; no training loop here.
        p = runtime.prepare(self.f.base/'loss-negative', case)
        obs = runtime.Observation(p, case, self.renderer, dict(forwards=0))
        obs.k = 1
        batch = next(iter(p.dataloader))
        teacher, cpu_camera = batch[0]
        camera = cpu_camera.cuda()
        result = obs.render(camera, p.gaussians, p.pipe, p.background, formal_config=p.inputs.config)
        try:
            l1 = self.train.l1_loss(result['render'], teacher)
            dssim = 1-self.train.ssim(result['render'], teacher)
            mask = (-(1-camera.gt_alpha_mask)*self.torch.log(1-result['alpha'].clamp(1e-6, 1-1e-6))).mean()
            total = .75*l1+.25*dssim+.125*mask
            cases = dict(coefficient=(.5*l1+.5*dssim+.125*mask)/2,
                         omitted_division=total, double_division=total/4)
            for name, loss in cases.items():
                with self.subTest(error=name):
                    local = dict(iteration=1, batch_idx=0, batch_size=2, viewpoint_cam=camera,
                                 render_pkg=result, image=result['render'], loss=loss, batch_data=batch,
                                 gt_image=teacher, Ll1=l1, Lssim=dssim, Lopa_mask=mask)
                    original = Mock()
                    with self.assertRaisesRegex(AssertionError, 'loss_weight_or_batch_division'):
                        obs.backward(original, loss, local)
                    original.assert_not_called()
            self.assertFalse(obs.trace['losses'])
            self.assertTrue(all(g['params'][0].grad is None for g in p.gaussians.optimizer.param_groups))
        finally:
            for h in obs.handles:
                h.remove()

    def test_report_wrong_final_view_loss_metric_and_stale_state_refused(self):
        for mode, label in (('input', 'report_final_view_input'), ('metric', 'report_completed_metric'),
                            ('stale', 'report_completed')):
            case = runtime.CASES[6] if mode == 'input' else runtime.CASES[10]
            held = []
            def install(obs):
                held.append(obs)
                report = obs.report
                def incorrect(original, *args, **kwargs):
                    if args[1] != 2:
                        return report(original, *args, **kwargs)
                    if mode == 'input':
                        args = list(args)
                        # The existing log is the final view, NOT the batch average.
                        args[3] = self.torch.tensor(sum(v['backward_value'] for v in obs.trace['losses'][-2:])/2)
                    elif mode == 'metric':
                        real = original
                        def original(*a, **kw):
                            real(*a, **kw)  # Preserve the real evaluation; corrupt only its handoff.
                            return float(a[3])
                    else:
                        with self.torch.no_grad():
                            obs.model._xyz[0].copy_(self.torch.from_numpy(
                                obs.trace['steps'][-1]['before']['groups']['xyz']['p'][0]))
                    return report(original, *args, **kwargs)
                obs.report = incorrect
            with self.subTest(error=mode), self.assertRaisesRegex(AssertionError, label):
                self.exercise_case(case, self.f.base/mode, install)
            self.assertFalse(held[0].handles)
            self.assertIsNone(held[0].last_loss)
            self.assertIsNone(held[0].active_adam)
            self.assertFalse(self.torch.cuda.is_initialized())

    def test_dispatch_cannot_be_inferred_from_settings_alone(self):
        p = runtime.prepare(self.f.base/'dispatch-negative', runtime.CASES[0])
        obs = runtime.Observation(p, runtime.CASES[0], self.renderer, dict(forwards=0))
        obs.active_adam = []
        g = p.gaussians.optimizer.param_groups[0]
        settings = {n: g[n] for n in ('foreach', 'fused', 'amsgrad', 'maximize', 'capturable',
                                     'differentiable', 'eps', 'weight_decay')}
        original = Mock(return_value=None)  # No actual backend call: observation MUST reject it.
        with self.assertRaisesRegex(AssertionError, 'Adam_dispatch_missing'):
            obs.adam_function(original, [], **settings)
        original.assert_called_once()
        self.assertIsNone(obs.active_adam[0]['selected'])
        self.assertFalse(obs.active_adam[0]['completed'])

    def test_new_wrappers_restore_on_backward_exception(self):
        import torch.optim.adam as adam_module
        original = (self.torch.Tensor.backward, self.train.psnr, adam_module.adam,
                    adam_module._single_tensor_adam, adam_module._multi_tensor_adam, adam_module._fused_adam)
        held = []
        def install(obs):
            held.append(obs)
            def fail(*a, **kw):
                raise RuntimeError('injected_backward_boundary')
            obs.backward = fail
        with self.assertRaisesRegex(RuntimeError, 'injected_backward_boundary'):
            self.exercise_case(runtime.CASES[0], callback=install)
        self.assertEqual(original, (self.torch.Tensor.backward, self.train.psnr, adam_module.adam,
                         adam_module._single_tensor_adam, adam_module._multi_tensor_adam, adam_module._fused_adam))
        self.assertFalse(held[0].handles)
        self.assertIsNone(held[0].loss_input)
        self.assertIsNone(held[0].last_loss)

    def test_observation_preserves_real_graph_calls_state_rng_and_snapshot_ownership(self):
        import formal_entry
        import torch.optim.adam as adam_module
        case = runtime.CASES[3]  # B=2, 6 updates; includes final-view distinction and scheduled metric.
        results = []
        for enabled in (False, True):
            p = runtime.prepare(self.f.base/str(enabled), case)
            counts = dict(render=0, backward=0, step=0, single=0, foreach=0, fused=0)
            graph = []
            original_backward = self.torch.Tensor.backward
            def backward(tensor, *a, **kw):
                counts['backward'] += 1
                graph.append((type(tensor.grad_fn).__name__, float(tensor.detach())))
                return original_backward(tensor, *a, **kw)
            def renderer(*a, **kw):
                counts['render'] += 1
                return self.renderer(*a, **kw)
            real_step = p.gaussians.optimizer.step
            def step(*a, **kw):
                counts['step'] += 1
                return real_step(*a, **kw)
            original = {n: getattr(adam_module, n) for n in
                        ('_single_tensor_adam', '_multi_tensor_adam', '_fused_adam')}
            held = []
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.torch.cuda, 'Event', Event))
                stack.enter_context(patch.object(self.torch.Tensor, 'backward', backward))
                stack.enter_context(patch.object(p.gaussians.optimizer, 'step', step))
                for name, path in (('_single_tensor_adam', 'single'), ('_multi_tensor_adam', 'foreach'),
                                   ('_fused_adam', 'fused')):
                    def counted(*a, _name=name, _path=path, **kw):
                        counts[_path] += 1
                        return original[_name](*a, **kw)
                    stack.enter_context(patch.object(adam_module, name, counted))
                if enabled:
                    trace = runtime.execute_case(p, case, dict(forwards=0), renderer, held.append)
                else:
                    stack.enter_context(patch.object(self.train, 'render', renderer))
                    stack.enter_context(patch.object(self.torch, 'save', lambda *a, **kw: None))
                    claim = formal_entry.OutputClaim(p.inputs)
                    writer = None
                    try:
                        writer = self.train.begin_formal_output(p, claim)
                        self.assertEqual(self.train.continue_formal_training(p, writer), case['n'])
                    finally:
                        if writer is not None:
                            writer.close()
                        claim.close()
            results.append(dict(state=runtime.snapshot(p.gaussians), counts=counts.copy(), graph=graph,
                                torch_rng=o.array(self.torch.get_rng_state()), numpy_rng=np.random.get_state(),
                                python_rng=random.getstate()))
            if enabled:
                saved = json.dumps(runtime.serializable(trace), allow_nan=False)
                self.assertIsNone(held[0].last_loss)
                self.assertIsNone(held[0].evaluation)
                self.assertFalse(held[0].handles)
                with self.torch.no_grad():
                    p.gaussians._xyz.add_(1)
                self.assertEqual(saved, json.dumps(runtime.serializable(trace), allow_nan=False))
        a, b = results
        runtime.equal_snapshot(a['state'], b['state'])
        self.assertEqual(a['counts'], b['counts'])
        self.assertEqual(a['graph'], b['graph'])
        np.testing.assert_array_equal(a['torch_rng'], b['torch_rng'])
        self.assertEqual(a['numpy_rng'][0], b['numpy_rng'][0])
        np.testing.assert_array_equal(a['numpy_rng'][1], b['numpy_rng'][1])
        self.assertEqual(a['numpy_rng'][2:], b['numpy_rng'][2:])
        self.assertEqual(a['python_rng'], b['python_rng'])
        print('FIX2_NONINTERFERENCE '+json.dumps(dict(counts=a['counts'], graph_and_state_exact=True, rng_exact=True)))


if __name__ == '__main__':
    print('test executable:', sys.executable)
    unittest.main()
