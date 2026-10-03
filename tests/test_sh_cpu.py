"""Step 8 independent CPU mathematics/negative witnesses, no CUDA claims."""
import ast
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import sh_oracle as o

ROOT = Path(__file__).resolve().parents[1]


class SHTests(unittest.TestCase):
    metrics = {}

    @classmethod
    def setUpClass(cls):
        if sys.executable != '/home/demo/miniconda3/envs/4dgs310/bin/python':
            raise RuntimeError('4dgs310_required')
        if not sys.dont_write_bytecode: raise RuntimeError('use_python_B')
        if torch.cuda.is_initialized(): raise AssertionError('CUDA already initialized')
        cls.guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU only'))
        cls.guard.start()
        cls.addClassCleanup(cls.guard.stop)
        print('executable:', sys.executable)

    @classmethod
    def tearDownClass(cls):
        assert not torch.cuda.is_initialized()
        print('SH_CPU_METRICS', json.dumps(cls.metrics, sort_keys=True))

    def compare(self, actual, expected, label, atol=1e-8, rtol=1e-6):
        error = (actual-expected).abs()
        bound = atol+rtol*expected.abs()
        self.metrics[label] = dict(max_abs=error.max().item(), max_bound_ratio=(error/bound).max().item())
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def test_legendre_phase_order_and_addition_anchors(self):
        d = torch.tensor([.3, -.4, .5], dtype=torch.float64)
        d = d/d.norm()
        y = o.basis(d)
        c0, c1 = 1/math.sqrt(4*math.pi), math.sqrt(3/(4*math.pi))
        anchors = torch.stack((d.new_tensor(c0), -c1*d[1], c1*d[2], -c1*d[0]))
        self.compare(y[:4], anchors, 'anchors_l01', 1e-12, 1e-10)
        self.compare(y[6], math.sqrt(5/(16*math.pi))*(3*d[2]**2-1), 'anchor_l20', 1e-12, 1e-10)
        self.compare(y[12], math.sqrt(7/(16*math.pi))*(5*d[2]**3-3*d[2]), 'anchor_l30', 1e-12, 1e-10)
        for l in range(4):
            block = y[l*l:(l+1)**2]
            self.compare(block.square().sum(), d.new_tensor((2*l+1)/(4*math.pi)), f'addition_{l}', 1e-12, 1e-10)
            self.compare(o.basis(-d)[l*l:(l+1)**2], (-1)**l*block, f'parity_{l}', 1e-12, 1e-10)

    def test_hamilton_covariance_independent_anchors(self):
        p = o.fixture()
        with torch.no_grad():
            p['left'].copy_(torch.tensor([1., 0, 0, 0]))
            p['right'].copy_(p['left'])
        self.compare(o.covariance(p), torch.diag(p['log_scale'].exp().square()), 'identity_cov', 1e-12, 1e-10)
        # A pure quaternion rotation mixing x and t: independent 2x2 block.
        angle = .3
        with torch.no_grad(): p['left'].copy_(p['left'].new_tensor([math.cos(angle), 0, 0, math.sin(angle)]))
        a = torch.eye(4, dtype=torch.float64)
        a[0,0] = a[3,3] = math.cos(angle)
        a[0,3], a[3,0] = math.sin(angle), -math.sin(angle)
        a[1,1] = a[2,2] = math.cos(angle)
        a[1,2], a[2,1] = math.sin(angle), -math.sin(angle)
        expected = a @ torch.diag(p['log_scale'].exp().square()) @ a.T
        self.compare(o.covariance(p), expected, 'xt_rotation_cov', 1e-12, 1e-10)
        original = o.fixture()
        for key in ('left','right'):
            q = o.clone(original)
            with torch.no_grad(): q[key].neg_()
            self.compare(o.covariance(q), o.covariance(original), 'sign_'+key, 1e-12, 1e-10)
        for zero in ('delta','cross'):
            q = o.clone(original)
            with torch.no_grad():
                if zero == 'delta': q['time'].fill_(o.SPEC['timestamp'])
                else:
                    q['left'].copy_(q['left'].new_tensor([1.,0,0,0]))
                    q['right'].copy_(q['left'])
            self.compare(o.conditional(q)[0], q['mean'], 'mean_control_'+zero, 1e-12, 1e-10)

    def test_all_active_stages_gradients_finite_difference_and_inactive(self):
        for stage in o.STAGES:
            p = o.fixture()
            fn = lambda a: o.loss(o.color(a, stage))
            g = o.gradients(fn(p), p)
            for h in (1e-6, 5e-7):
                fd = o.finite_difference(fn, p, h)
                for name in p: self.compare(g[name], fd[name], f'fd_{stage}_{h}_{name}')
            count = (stage[0]+1)**2 if stage[1] == 0 else 16*(stage[1]+1)
            self.assertEqual(tuple(g['features'].shape), (48,3))
            self.assertEqual(g['features'][count:].count_nonzero().item(), 0)
            q = o.clone(p)
            with torch.no_grad(): q['features'][count:].fill_(99.)
            self.assertTrue(torch.equal(o.color(p, stage), o.color(q, stage)))
            for name in ('left','right'):
                self.assertAlmostEqual((g[name]*p[name]).sum().item(), 0, delta=1e-12)

    def test_time_direct_plus_conditional_chain_and_each_mode(self):
        for modes in ((1,), (2,), (1,2)):
            for timestamp in (0., .4):
                p = o.fixture(); spec = dict(o.SPEC, timestamp=timestamp)
                with torch.no_grad():
                    for mode in (1,2):
                        if mode not in modes: p['features'][mode*16:(mode+1)*16].zero_()
                mean, _, _, sigma = o.conditional(p, spec)
                scalar = o.loss(o.color_at(p, mean, (3,2), spec))
                gm, gt = torch.autograd.grad(scalar, (mean, p['time']))
                theta = 2*math.pi*(p['time'][0]-timestamp)/spec['duration']
                components = [-k*2*math.pi/spec['duration']*theta.mul(k).sin()*
                              o.loss(o.basis(mean)@p['features'][16*k:16*(k+1)]) for k in (1,2)]
                chain = -gm @ sigma[:3,3]/sigma[3,3]
                self.compare(gt[0], sum(components)+chain, f'time_chain_{modes}_{timestamp}', 1e-12, 1e-10)
                if modes == (1,2):
                    for label, mutant in (('sign', -sum(components)+chain), ('overwrite', components[1]+chain)):
                        gap = abs((gt[0]-mutant).item())
                        bound = 3e-5+3e-4*abs(gt[0].item())
                        self.assertGreater(gap, 100*bound)
                        self.metrics[f'mutant_{label}_{timestamp}'] = gap

    def test_old_coefficient_and_mean_witnesses(self):
        p = o.fixture()
        mean = o.conditional(p)[0]
        for direction in (mean, mean*mean.new_tensor([1,-1,1])):
            good = o.basis(direction)[1]*direction.new_tensor(o.WEIGHTS)
            bad = direction.new_tensor(o.WEIGHTS)/math.sqrt(4*math.pi)
            self.assertGreater((good-bad).abs().max().item(), 100*(3e-5+3e-4*good.abs().max().item()))
        good = o.color(p, (3,2))
        old = o.color_at(p, p['mean'], (3,2))
        gap = (good-old).abs().max().item()
        self.assertGreater(gap, 100*(2e-6+2e-5*good.abs().max().item()))
        self.metrics['mutant_original_mean_color_gap'] = gap

    def test_pixel_oracle_isolation_and_raw_gradient_fd(self):
        p = o.fixture()
        for stage in o.STAGES:
            a = o.pixel(p, stage)
            self.compare(o.isolate(a['rgb'], a['alpha'], o.SPEC['background']), a['color'], f'isolate_{stage}', 1e-12, 1e-10)
            g = o.gradients(o.loss(o.isolate(a['rgb'],a['alpha'],o.SPEC['background'])), p)
            direct = o.gradients(o.loss(o.color(p,stage)), p)
            for name in p: self.compare(g[name], direct[name], f'isolate_grad_{stage}_{name}', 1e-12, 1e-10)
        fn = lambda p: o.loss(o.pixel(p,(3,2))['rgb'])
        g = o.gradients(fn(p),p)
        for h in (1e-6,5e-7):
            fd = o.finite_difference(fn,p,h)
            for name in p: self.compare(g[name],fd[name],f'raw_fd_{h}_{name}')

    def test_fixture_smooth_domain_including_gpu_fd_perturbations(self):
        # CPU verifies the largest planned GPU FD perturbation, including
        # radius/tile stability. No actual CUDA code is executed here.
        p = o.fixture(dtype=torch.float32)
        p = o.clone(p)
        baseline = o.assert_safe(p)
        self.metrics['fixture_domain'] = baseline
        for name, value in p.items():
            for i in range(value.numel()):
                for sign in (-1,1):
                    q = o.clone(p)
                    with torch.no_grad(): q[name].view(-1)[i] += sign*1e-3*max(1,abs(value.view(-1)[i].item()))
                    a = o.assert_safe(q)
                    self.assertEqual((a['radius'],a['tiles']), (baseline['radius'],baseline['tiles']))

    def test_all_planned_cuda_inputs_on_cpu(self):
        cases = o.acceptance_cases()
        self.assertEqual(len(cases), 15)
        domains = {}
        for label, p, stage, spec, three_d in cases:
            p = o.clone(o.clone(p, dtype=torch.float32))
            spec['timestamp'] = float(torch.tensor(spec['timestamp'], dtype=torch.float32))
            if three_d:
                expected = o.color_at(p, p['mean'], stage, spec)
                self.assertTrue(torch.isfinite(expected).all())
                self.assertTrue(((expected > .2) & (expected < .8)).all())
            else:
                domains[label] = o.assert_safe(p, spec, stage)
        self.metrics['planned_cuda_cpu_domains'] = domains

    def test_source_contract_and_cuda_entry_syntax_without_execution(self):
        f = (ROOT/'diff-gaussian-rasterization/cuda_rasterizer/forward.cu').read_text()
        b = (ROOT/'diff-gaussian-rasterization/cuda_rasterizer/backward.cu').read_text()
        self.assertIn('computeColorFromSH_4D(idx, D, D_t, M, (glm::vec3*)out_means3D', f)
        four_d = b.split('__device__ void computeColorFromSH_4D',1)[1].split('__global__ void computeCov2DCUDA',1)[0]
        self.assertIn('dL_dsh[1] = l1m1 * dL_dRGB;',four_d)
        for k in (1,2):
            self.assertIn(f'float dt{k}_dt = -sin(',four_d)
            self.assertIn(f'dRGBdt += dt{k}_dt * (',four_d)
        ast.parse((ROOT/'tests/cuda/sh_cuda_acceptance.py').read_text())
        # This static check is supplemental, NOT compiled-kernel evidence.


if __name__ == '__main__':
    unittest.main(verbosity=2)
