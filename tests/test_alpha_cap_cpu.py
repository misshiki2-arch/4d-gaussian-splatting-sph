"""Step 9 CPU-only independent cap mathematics, negative witnesses and domain."""
import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import alpha_cap_oracle as o

ROOT = Path(__file__).resolve().parents[1]


class AlphaCapTests(unittest.TestCase):
    metrics = dict(comparisons=0, max_abs=0., max_bound_ratio=0.)

    @classmethod
    def setUpClass(cls):
        if sys.executable != '/home/demo/miniconda3/envs/4dgs310/bin/python' or not sys.dont_write_bytecode:
            raise RuntimeError('explicit_4dgs310_B_required')
        if torch.cuda.is_initialized(): raise AssertionError('CUDA already initialized')
        guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU only'))
        guard.start(); cls.addClassCleanup(guard.stop)
        print('executable:', sys.executable)

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        assert not any(n in sys.modules for n in ('gaussian_renderer', 'scene', 'torch.utils.cpp_extension'))
        print('ALPHA_CPU_METRICS', json.dumps(cls.metrics, sort_keys=True))

    def compare(self, actual, expected, tolerance='anchor'):
        actual = torch.as_tensor(actual, dtype=torch.float64)
        expected = torch.as_tensor(expected, dtype=torch.float64)
        self.assertTrue(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        atol, rtol = o.TOLERANCES[tolerance]
        error = (actual-expected).abs()
        self.metrics['comparisons'] += 1
        self.metrics['max_abs'] = max(self.metrics['max_abs'], float(error.max()))
        self.metrics['max_bound_ratio'] = max(self.metrics['max_bound_ratio'], float((error/(atol+rtol*expected.abs())).max()))
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def core(self, values, loss='mixed'):
        raw = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        n = len(values)
        color = torch.tensor([[.5, .5, .5], [.25, .75, .5]][:n], dtype=torch.float64, requires_grad=True)
        flow = torch.tensor([[.25, -.125], [-.25, .125]][:n], dtype=torch.float64, requires_grad=True)
        depth = torch.tensor([3., 4.][:n], dtype=torch.float64, requires_grad=True)
        bg = raw.new_tensor(o.SPEC['background'])
        return (raw, color, flow, depth, bg), o.composite(raw, color, flow, depth, bg, o.LOSSES[loss])

    def test_prefix_suffix_hand_anchors(self):
        (_, colors, flows, depths, bg), r = self.core([.25, .5])
        self.compare(r['trans'], [1., .75, .375])
        self.compare(r['weights'], [.25, .375])
        self.compare(r['outputs'][:3], .25*colors[0]+.375*colors[1]+.375*bg)
        self.compare(r['outputs'][3:5], .25*flows[0]+.375*flows[1])
        self.compare(r['outputs'][5:], [2.25, .625])
        _, mask = self.core([.25, .5], 'mask')
        self.compare(mask['dr'], [.5, .75])
        _, single = self.core([.25], 'mask')
        self.compare(single['dr'], [1.])

    def test_core_all_losses_closed_derivatives_and_finite_differences(self):
        for values in ([.6], [.997], [.997, .4], [.4, .997]):
            for loss in ('rgb', 'flow', 'depth', 'mask', 'mixed'):
                inputs, r = self.core(values, loss)
                actual = torch.autograd.grad(r['loss'], inputs[:4])
                expected = (r['dr'], r['color_grad'], r['flow_grad'], r['depth_grad'])
                for a, b in zip(actual, expected): self.compare(a, b)
                for h in o.CPU_STEPS:
                    for k, v in enumerate(inputs[:4]):
                        for i in range(v.numel()):
                            plus, minus = [a.detach().clone() for a in inputs], [a.detach().clone() for a in inputs]
                            step = h*max(1., abs(float(v.flatten()[i])))
                            plus[k].view(-1)[i] += step; minus[k].view(-1)[i] -= step
                            fd = (o.composite(*plus, o.LOSSES[loss])['loss']-o.composite(*minus, o.LOSSES[loss])['loss'])/(2*step)
                            self.compare(actual[k].flatten()[i], fd, 'cpu_fd')

    def test_exact_float32_neighbors_and_selected_equality_subgradient(self):
        below, equal, above = o.neighbors()
        self.assertLess(below, equal); self.assertLess(equal, above)
        self.assertNotEqual(o.CAP, .99)
        for value, expected in zip((below, equal, above), (1., 0., 0.)):
            inputs, result = self.core([value], 'mask')
            self.assertEqual(float(result['dr'][0]), expected)
            self.assertEqual(float(torch.autograd.grad(result['loss'], inputs[0])[0]), expected)
        for h in o.CUDA_STEPS:
            _, center = self.core([equal], 'mask')
            _, left = self.core([equal-h], 'mask')
            _, right = self.core([equal+h], 'mask')
            self.assertEqual(float((center['loss']-left['loss'])/h), 1.)
            self.assertEqual(float((right['loss']-center['loss'])/h), 0.)

    def test_mutants_ungated_equality_and_background_first(self):
        _, r = self.core([.997], 'mask')
        self.assertEqual(float(r['da'][0]), 1.)
        self.assertEqual(float(r['dr'][0]), 0.)
        self.assertGreater(float(abs(r['da'][0]-r['dr'][0])), 20*o.TOLERANCES['direct'][0])
        _, eq = self.core([o.CAP], 'mask')
        self.assertEqual(float(eq['da'][0]), 1.)  # <= mutant would pass this.
        _, bg = self.core([.997], 'background')
        self.compare(bg['da'], bg['background_da'])
        self.assertEqual(float(bg['dr'][0]), 0.)
        self.assertGreater(float(bg['background_da'].abs().max()), 20*o.TOLERANCES['direct'][0])
        self.metrics['old_analytic_mask_gradient'] = float(r['da'][0])
        self.metrics['early_gate_background_gap'] = float(bg['background_da'].abs().max())

    def test_mutant_direct_value_gates_are_detected(self):
        for loss, key in (('rgb', 'color_grad'), ('flow', 'flow_grad'), ('depth', 'depth_grad')):
            _, r = self.core([.997], loss)
            self.assertEqual(float(r['dr'][0]), 0.)
            self.assertGreater(float(r[key].abs().max()), .9)
        r = o.evaluate([o.fixture()], loss=o.LOSSES['depth'])
        self.compare(r['screen'][0, :2], [0., 0.])
        self.compare(r['screen'][0, 2], o.CAP)
        self.compare(o.C0*r['color_grad'], torch.zeros((1, 3), dtype=torch.float64))

    def test_offcenter_old_gradient_and_whole_gaussian_zero_mutant(self):
        p = [o.fixture()]
        a = o.evaluate(p, loss=o.LOSSES['mask'])
        b = o.evaluate(p, (5, 4), loss=o.LOSSES['mask'])
        self.assertGreater(float(a['geometry'][0]['raw']), .992)
        self.assertLess(float(b['geometry'][0]['raw']), .985)
        self.assertEqual(float(a['opacity_grad'][0]), 0.)
        self.assertGreater(float(b['opacity_grad'][0]), .1)
        g = a['geometry'][0]
        bad_xy = g['raw']*a['da'][0]*(-g['inverse'] @ g['delta'])*4.5
        self.assertGreater(float(bad_xy.abs().max()), 20*o.TOLERANCES['direct'][0])
        selected = {c['label']: c for c in o.cases() if c['label'].startswith('P/')}
        grads = {k: o.parameter_gradients(c)[0] for k, c in selected.items()}
        for name in grads['P/sum']:
            self.compare(grads['P/sum'][name], grads['P/p0'][name]+grads['P/p1'][name])
        self.assertGreater(float(grads['P/sum']['opacity'].abs().max()), .1)

    def test_all_reviewed_fixtures_and_fd_endpoints_domain(self):
        all_cases = o.cases()
        self.assertEqual(len(all_cases), 34)
        domains = {}
        endpoints = 0
        for case in all_cases:
            base = o.domain(case)
            domains[case['label']] = base
            for coord in o.fd_coordinates(case):
                for h, rounded in [(h, False) for h in o.CPU_STEPS]+[(h, True) for h in o.CUDA_STEPS]:
                    plus, minus, _ = o.perturb(case['points'], coord, h, rounded=rounded)
                    for points in (plus, minus):
                        checked = o.domain(case, points)
                        self.assertEqual([d['topology'] for d in checked], [d['topology'] for d in base],
                                         (case['label'], coord, h, rounded))
                        endpoints += 1
        self.metrics['domain_cases'] = len(domains)
        self.metrics['domain_fd_endpoints'] = endpoints
        self.metrics['domain_raw_ranges'] = {k: [min(n['raw'] for d in v for n in d['nodes']),
                                                max(n['raw'] for d in v for n in d['nodes'])]
                                             for k, v in domains.items()}

    def test_independent_geometry_anchors_and_screen_units(self):
        g = o.geometry(o.fixture(centered=True), (4, 4))
        self.compare(g['xy'], [4., 4.])
        self.compare(g['power'], 0.)
        self.compare(g['marginal'], 1.)
        self.compare(g['color'], [.5, .5, .5])
        p = [o.fixture(.6)]
        r = o.evaluate(p, loss=o.LOSSES['mask'])
        self.compare(r['screen'][0, :2], -r['geometry'][0]['raw']*(r['geometry'][0]['inverse'] @ r['geometry'][0]['delta'])*4.5)

    def test_boundary_onesided_endpoints_preserve_geometry_domain(self):
        # Equality intentionally has different left/right cap branches. All
        # OTHER domain conditions still apply to these additional FD endpoints.
        endpoints = 0
        for case in (c for c in o.cases() if c['label'].endswith('/equal')):
            base = o.domain(case)[0]['nodes'][0]
            for h in o.CUDA_STEPS:
                for sign in (-1, 1):
                    points = o.clone(case['points'])
                    with torch.no_grad(): points[0]['opacity'][0] += sign*h
                    points = o.clone(points, rounded=True)
                    observed = o.domain(case, points)[0]['nodes'][0]
                    self.assertEqual((observed['radius'], observed['tiles'], observed['z']),
                                     (base['radius'], base['tiles'], base['z']))
                    self.assertEqual(observed['capped'], sign > 0)
                    endpoints += 1
        self.metrics['boundary_one_sided_endpoints'] = endpoints

    def test_geometry_analytic_vjp_full_graph_and_selected_fd(self):
        for case in o.cases():
            expected = o.parameter_gradients(case)
            q = o.clone(case['points'])
            loss = sum(r['loss'] for r in o.evaluate_case(case, q))
            flat = [v for p in q for v in p.values()]
            actual = iter(torch.autograd.grad(loss, flat, allow_unused=True))
            for i, p in enumerate(q):
                for name, v in p.items():
                    g = next(actual)
                    self.compare(torch.zeros_like(v) if g is None else g, expected[i][name])
            for coordinate in o.fd_coordinates(case):
                i, name, offset = coordinate
                for h in o.CPU_STEPS:
                    plus, minus, denominator = o.perturb(case['points'], coordinate, h)
                    fd = (sum(r['loss'] for r in o.evaluate_case(case, plus))
                          -sum(r['loss'] for r in o.evaluate_case(case, minus)))/denominator
                    self.compare(expected[i][name].flatten()[offset], fd, 'cpu_fd')

    def test_formal_moving_direct_sh_survives_saturation(self):
        case = next(c for c in o.cases() if c['label'] == 'F/0.997')
        r = o.evaluate_case(case)[0]
        self.assertTrue(float(r['geometry'][0]['raw']) >= .992)
        self.assertEqual(float(r['dr'][0]), 0.)
        g = o.parameter_gradients(case)[0]
        self.assertEqual(float(g['opacity'][0]), 0.)
        self.assertGreater(float(g['features'].abs().max()), .01)
        self.assertGreater(float(g['time'].abs().max()), 1e-5)
        self.assertEqual(g['flow'].count_nonzero().item(), 0)

    def test_loss_linearity_preserves_all_seven_direct_channels(self):
        p = [o.fixture(.997), o.fixture(.4, depth=4.)]
        mix = o.evaluate(p)
        separate = [o.evaluate(p, loss=tuple(float(i == j) for i in range(7))) for j in range(7)]
        for field in ('dr', 'screen', 'opacity_grad', 'color_grad', 'flow_grad', 'depth_grad'):
            self.compare(mix[field], sum(o.LOSSES['mixed'][j]*r[field] for j, r in enumerate(separate)))

    def test_local_source_gate_and_independent_import_boundary(self):
        source = (ROOT/'diff-gaussian-rasterization/cuda_rasterizer/backward.cu').read_text()
        section = source.split('const float raw_alpha = con_o.w * G;', 1)[1]
        self.assertIn('const float alpha = min(0.99f, raw_alpha);', section)
        gate = 'dL_dalpha = raw_alpha < 0.99f ? dL_dalpha : 0.0f;'
        self.assertEqual(section.count(gate), 1)
        self.assertLess(section.index('dL_dalpha += (-T_final'), section.index(gate))
        self.assertLess(section.index(gate), section.index('const float dL_dG'))
        self.assertLess(section.index('last_alpha = alpha'), section.index(gate))
        self.assertIn('dL_depth * dchannel_dcolor', section)
        self.assertNotIn('continue;', section[section.index(gate):section.index('const float dL_dG')])
        tree = ast.parse((ROOT/'tests/alpha_cap_oracle.py').read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertEqual(set(imports), {'math', 'struct', 'torch', 'sh_oracle'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
