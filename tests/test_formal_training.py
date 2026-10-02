"""Step 7 actual-loop CPU regressions; see fixture for isolated boundaries."""
import math
import json
import importlib
import types
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).parent))
from formal_training_fixture import CPUCase, cloned


class TrainingTests(CPUCase):
    def test_exact_updates_epoch_and_independent_adam(self):
        for n in (1, 2, 3):
            with self.subTest(n=n):
                p = self.configured(n)
                t = self.exercise(p)
                self.assertEqual(t['labels'], list(range(1, n+1)))
                self.assertEqual(len(t['fetch']), n)
                self.assertEqual(len(t['steps']), n)
                self.assertEqual(t['completed'], n)
                self.assertEqual([(k, i) for k, i, _, _ in t['forward']],
                                 [(k, i) for k, ids in enumerate(t['fetch'], 1) for i in ids])
                for group in p.gaussians.optimizer.param_groups:
                    name, parameter = group['name'], group['params'][0]
                    state = p.gaussians.optimizer.state[parameter]
                    expected = t['initial'][name].double()
                    m = v = 0.
                    for k in range(1, n+1):
                        grad = .0001
                        m = .9*m + .1*grad
                        v = .999*v + .001*grad*grad
                        rate = group['lr']
                        if name == 'xyz':
                            extent = p.scene.cameras_extent
                            rate = extent*math.exp((1-k/20)*math.log(.002)+(k/20)*math.log(.001))
                        expected -= rate*(m/(1-.9**k))/(math.sqrt(v/(1-.999**k))+1e-15)
                    self.torch.testing.assert_close(parameter.double(), expected, rtol=1e-5, atol=1e-6)
                    self.torch.testing.assert_close(state['exp_avg'], self.torch.full_like(parameter, m), rtol=1e-5, atol=1e-10)
                    self.torch.testing.assert_close(state['exp_avg_sq'], self.torch.full_like(parameter, v), rtol=1e-5, atol=1e-14)
                    self.assertEqual(state['step'].item(), n)
                self.assertEqual([x[0][1] for x in t['saves']], [n])
                for k, (before, _) in enumerate(t['steps']):
                    self.assertEqual([identity for _, identity, _ in before], t['forward_parameters'][k*2])
                    self.assertTrue(all(grad is not None for _, _, grad in before))
                print('AFTER', json.dumps(dict(n=n, fetch=t['fetch'], labels=t['labels'], steps=len(t['steps']),
                    xyz_final=p.gaussians._xyz.tolist())))

    def test_sh_before_forward_saturates_and_loss_is_pre_step(self):
        p = self.configured(6, batch=1)
        t = self.exercise(p)
        self.assertEqual([(s,t) for _,_,s,t in t['forward']], [(1,0),(2,0),(3,0),(3,1),(3,2),(3,2)])
        self.assertEqual(t['reports'][0][0], 0)
        first_rgb = .6 + sum(float(x.sum())*.0001 for x in t['initial'].values())
        # RGBA over black is truncated to uint8: [50,20,5] / 255.
        expected_loss = first_rgb - (50+20+5)/(3*255)
        self.assertAlmostEqual(t['reports'][1][2], expected_loss, delta=1e-6)
        self.assertNotEqual(float(t['reports'][1][1][1][0,0]), float(t['initial']['xyz'][0,0]))

    def test_final_growth_then_reset_save_and_report_same_state(self):
        p = self.configured(1, population=dict(densify_until_update=2, opacity_reset_interval=1, thresh_opa_prune=.05), tests=(1,))
        m, torch = p.gaussians, self.torch
        with torch.no_grad(): m._opacity[0] = math.log(.04/.96)
        t = self.exercise(p)
        self.assertEqual(len(m._xyz), 6)  # Three parents + clones; no reset-before-prune loss.
        torch.testing.assert_close(m.get_opacity, torch.full_like(m._opacity, .01), rtol=1e-5, atol=1e-8)
        self.assertEqual(t['events'][-4:], ['prune_points', 'reset_opacity', 'save', 'report'])
        self.assertLess(t['events'].index('zero'), t['events'].index('densify_and_clone'))
        self.assertLess(t['events'].index('add_densification_stats_grad'), t['events'].index('step'))
        saved = t['saves'][0][0][0]; reported = t['reports'][-1][1]
        self.assertEqual(len(saved), 19)
        for idx in (1,2,3,4,5,6,7,8,9,10,13,14,15):
            self.assertTrue(torch.equal(saved[idx], reported[idx]))
        for group in m.optimizer.param_groups:
            state = m.optimizer.state[group['params'][0]]
            self.assertEqual(state['step'].item(), 1)
            if group['name'] == 'opacity':
                self.assertEqual(state['exp_avg'].count_nonzero().item(), 0)
            else:
                torch.testing.assert_close(state['exp_avg'][:3], torch.full_like(state['exp_avg'][:3], .00001), rtol=1e-5, atol=1e-10)
                self.assertEqual(state['exp_avg'][3:].count_nonzero().item(), 0)
        expected = t['steps'][0][1][1][[1,2,3,1,2,3]]
        self.assertTrue(torch.equal(m._xyz, expected))

    def test_finite_stop_prune_below_then_growth_overshoot(self):
        p = self.configured(2, population=dict(densify_stop_points=4, densify_until_update=3, thresh_opa_prune=.05))
        with self.torch.no_grad(): p.gaussians._opacity[0] = math.log(.01/.99)
        t = self.exercise(p)
        events = [x for x in t['mutations'] if x[0]=='densify_and_prune']
        self.assertEqual([x[4]['prune_only'] for x in events], [True, False])
        self.assertEqual([(len(x[2][1]),len(x[3][1])) for x in events], [(4,3),(3,6)])
        self.assertEqual(events[0][3][7].tolist(), [2,3,4])
        self.assertEqual(events[1][3][7].count_nonzero().item(), 0)

    def test_population_event_boundaries_and_reset_only(self):
        cases = [
            (dict(densify_from_update=1,densify_until_update=4,opacity_reset_interval=2), [2,3], [2], [None,20]),
            (dict(densify_from_update=4,densify_until_update=4,opacity_reset_interval=1), [], [1,2,3], []),
            (dict(densify_until_update=0,opacity_reset_interval=1), [], [], []),
            (dict(densify_until_update=5,densify_stop_points=3,opacity_reset_interval=100), [1,2,3,4], [], [None]*4),
        ]
        for policy, grows, resets, sizes in cases:
            with self.subTest(policy=policy):
                p = self.configured(4, population=policy)
                # Avoid exponential population but execute actual no-selection events.
                # A high gradient threshold is an explicit fixture config value.
                self.f.value['optimization']['population']['densify_grad_threshold'] = 10
                p = self.f.prepare()
                with self.torch.no_grad(): p.gaussians._scaling.fill_(math.log(.01))
                t = self.exercise(p)
                events = [x for x in t['mutations'] if x[0]=='densify_and_prune']
                self.assertEqual([x[1] for x in events], grows)
                self.assertEqual([x[5][3] for x in events], sizes)
                self.assertEqual([x[1] for x in t['mutations'] if x[0]=='reset_opacity'], resets)
                if not grows and resets:
                    self.assertEqual(p.gaussians.denom[:,0].tolist(), [3]*4)
                    self.assertEqual(p.gaussians.max_radii2D.tolist(), [1,2,3,4])

    def test_current_batch_stats_visibility_and_time_before_zero(self):
        torch = self.torch
        for batch in (1,2):
            with self.subTest(batch=batch):
                p = self.configured(1, batch=batch, population=dict(densify_from_update=10,densify_until_update=2))
                calls = []
                def render(camera, m, pipe, background, **kw):
                    j = len(calls); calls.append(j)
                    visibility = torch.tensor(([False,True,True,False], [False,False,True,True])[j])
                    screen = torch.zeros_like(m._xyz, requires_grad=True)
                    coeff = torch.tensor(([0.,3.,3.,0.], [0.,0.,0.,6.])[j])*.001
                    coeff_y = torch.tensor(([0.,4.,4.,0.], [0.,0.,12.,8.])[j])*.001
                    tcoeff = torch.tensor(([0.,-.002,.004,0.], [0.,0.,-.006,.008])[j])[:,None]
                    signal = (screen[:,0]*coeff+screen[:,1]*coeff_y).sum()+(m._t*tcoeff).sum()
                    signal += sum(g['params'][0].sum()*0 for g in m.optimizer.param_groups)
                    return dict(render=(.6+signal).expand(3,camera.image_height,camera.image_width), viewspace_points=screen,
                        visibility_filter=visibility, radii=torch.tensor(([90.,2.,5.,90.],[99.,99.,7.,3.])[j]),
                        depth=torch.ones(1,camera.image_height,camera.image_width),alpha=torch.zeros(1,camera.image_height,camera.image_width))
                self.exercise(p, render=render)
                expected_screen = [0,.005,.005,0] if batch==1 else [0,.005,.0085,.01]
                expected_time = [0,-.002,.004,0] if batch==1 else [0,-.002,-.001,.008]
                m = p.gaussians
                torch.testing.assert_close(m.xyz_gradient_accum[:,0], torch.tensor(expected_screen), rtol=1e-5,atol=1e-9)
                torch.testing.assert_close(m.t_gradient_accum[:,0], torch.tensor(expected_time), rtol=1e-5,atol=1e-9)
                self.assertEqual(m.denom[:,0].tolist(), [0,1,1,0] if batch==1 else [0,1,1,1])
                # Existing batch max precedes union filtering, even if the other view is invisible.
                self.assertEqual(m.max_radii2D.tolist(), [0,2,5,0] if batch==1 else [0,99,7,90])

    def test_failures_do_not_complete_or_reach_later_observers(self):
        stages = ['fetch','lr','forward','backward','add_densification_stats_grad',
                  'step','zero','densify_and_clone','densify_and_split','prune_points','reset_opacity','save','report']
        for stage in stages:
            with self.subTest(stage=stage):
                p = self.configured(2, population=dict(densify_until_update=3,opacity_reset_interval=1), saves=(1,), tests=(1,))
                try:
                    self.exercise(p, fail=stage)
                except RuntimeError as error:
                    self.assertEqual(str(error), 'injected:'+stage)
                    tb = error.__traceback__
                    while tb and tb.tb_frame.f_code.co_name != 'training': tb = tb.tb_next
                    self.assertIsNotNone(tb)
                    self.assertEqual(tb.tb_frame.f_locals['completed'], 1 if stage in ('save','report') else 0)
                else: self.fail('failure not reached')
                t = self.trace
                self.assertEqual(len(t['fetch']), 0 if stage=='fetch' else 1)
                self.assertEqual([r[0] for r in t['reports']], [0])
                self.assertEqual(len(t['saves']), 1 if stage=='report' else 0)

    def test_empty_epoch_rejects_without_update_or_output(self):
        p = self.configured(1)._replace(dataloader=[])
        with self.assertRaisesRegex(RuntimeError, 'training_empty_epoch'):
            self.exercise(p)
        self.assertEqual(self.trace['steps'], [])
        self.assertEqual(self.trace['saves'], [])

    def test_real_report_metrics_and_save_observer_with_optional_writer(self):
        for with_writer in (False, True):
            with self.subTest(writer=with_writer):
                self.f.output = self.f.base/'parent'/('report-on' if with_writer else 'report-off')
                self.f.value['output']['directory'] = str(self.f.output)
                n, test_updates = (3, (2,3)) if with_writer else (2, (1,))
                p = self.configured(n, tests=test_updates, saves=(1,), size=256)
                self.f.value['optimization']['loss'].update(lambda_dssim=.25, lambda_opa_mask=.125)
                # Background sidecar pixels exercise the actual mask loss.
                for path in (self.f.source/'masks').glob('*.png'):
                    self.f.Image.new('L', (256,256), 0).save(path)
                p = self.f.prepare()
                writer = Mock() if with_writer else None
                t = self.exercise(p, real_report=True, writer=writer)
                self.assertEqual([s[0][1] for s in t['saves']], [1,n])
                self.assertEqual([r[0] for r in t['reports']], list(range(n+1)))
                self.assertTrue((self.f.output/'renders_png/iter_000000/test/000_render.png').is_file())
                for k in range(1,n+1):
                    self.assertEqual((self.f.output/f'renders_png/iter_{k:06d}').exists(), k in test_updates)
                expected_events = ['save','report','report','save','report'] if with_writer else ['save','report','save','report']
                self.assertEqual([a for a in t['events'] if a in ('save','report')], expected_events)
                for saved,_ in t['saves']:
                    reported = t['reports'][saved[1]][1]
                    for i in (1,6,7,13): self.assertTrue(self.torch.equal(saved[0][i],reported[i]))
                self.assertFalse(self.torch.equal(t['saves'][0][0][0][1],t['saves'][-1][0][0][1]))
                self.assertTrue(all(math.isfinite(r[2]) for r in t['reports']))
                if writer:
                    tags = [c.args[0] for c in writer.add_scalar.call_args_list]
                    self.assertIn('test/loss_viewpoint - msssim',tags)
                    self.assertTrue(writer.add_images.called)

    def test_real_shared_renderer_binding_backward_to_adam(self):
        p = self.configured(2, batch=1)
        binding = importlib.import_module('gaussian_renderer.diff_gaussian_rasterization')
        torch = self.torch; calls = []
        def forward(*a):
            self.assertEqual(a[12], -1.0); self.assertIs(type(a[12]), float)
            self.assertGreater(a[15],0); self.assertGreater(a[16],0)
            calls.append(('forward',a[20],a[21]))
            n,h,w = len(a[1]),a[17],a[18]
            empty = lambda: torch.empty(0,dtype=torch.uint8)
            return (n,torch.full((3,h,w),.6),torch.zeros(2,h,w),torch.ones(1,h,w),torch.full((1,h,w),.7),
                    torch.ones(n,dtype=torch.int32),empty(),empty(),empty(),torch.zeros(n,6),a[1].clone(),torch.empty(0),torch.empty(0))
        def backward(*a):
            self.assertEqual(a[14],-1.0)
            self.assertGreater(float(a[19].abs().sum()),0)
            calls.append(('backward',a[24],a[25]))
            return tuple(torch.full_like(a[i],.01) for i in (1,4,6,1,13,23,5,7,8,9,10,11))
        with patch.object(binding,'_C',types.SimpleNamespace(rasterize_gaussians=forward,rasterize_gaussians_backward=backward)):
            t = self.exercise(p,render=self.f.renderer.render)
        self.assertEqual(calls,[('forward',1,0),('backward',1,0),('forward',2,0),('backward',2,0)])
        self.assertEqual(len(t['steps']),2)
        self.assertFalse(torch.equal(p.gaussians._xyz,t['initial']['xyz']))


if __name__ == '__main__':
    unittest.main()
