"""Real GaussianModel/Adam row-state tests; independent lineage/numeric oracles."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
from formal_training_fixture import CPUCase, cloned


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return [w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v]


def rotated(sample, left, right):
    left = [x / math.sqrt(sum(v*v for v in left)) for x in left]
    right = [x / math.sqrt(sum(v*v for v in right)) for x in right]
    return multiply(multiply(left, sample[::-1]), [right[0], *[-x for x in right[1:]]])[::-1]


class PopulationTests(CPUCase):
    def model(self):
        p = self.configured(1)
        m, torch = p.gaussians, self.torch
        with torch.no_grad():
            for group in m.optimizer.param_groups:
                a = group['params'][0]
                # Distinct rows in every field; then specialize geometry.
                a.copy_(torch.arange(1, 5).reshape(4, *([1]*(a.ndim-1))).expand_as(a)*.03)
            m._xyz.copy_(torch.arange(12).reshape(4, 3)*.1)
            m._scaling.copy_(torch.tensor([.01, .04, .05, .06]).log()[:, None].expand(4, 3))
            m._scaling_t.fill_(math.log(.07))
            m._rotation.copy_(torch.tensor([[math.cos(.2), math.sin(.2), 0, 0]]).repeat(4, 1))
            m._rotation_r.copy_(torch.tensor([[math.cos(.3), 0, math.sin(.3), 0]]).repeat(4, 1))
            m._opacity.fill_(math.log(.2/.8))
        m.percent_dense = .02
        initial = {g['name']: cloned(g['params'][0]) for g in m.optimizer.param_groups}
        gradients = {}
        loss = 0
        for group in m.optimizer.param_groups:
            a = group['params'][0]
            grad = torch.arange(1, 5).reshape(4, *([1]*(a.ndim-1))).expand_as(a)*.01
            gradients[group['name']] = grad.clone()
            loss = loss + (a*grad).sum()
        loss.backward()
        m.optimizer.step()
        m.optimizer.zero_grad(set_to_none=True)
        for group in m.optimizer.param_groups:
            a, name = group['params'][0], group['name']
            grad = gradients[name].double()
            expected = initial[name].double()-group['lr']*grad/(grad.abs()+1e-15)
            torch.testing.assert_close(a.double(), expected, rtol=1e-5, atol=1e-6)
        with torch.no_grad():
            m.xyz_gradient_accum.copy_(torch.tensor([[1.], [2.], [3.], [0.]]))
            m.t_gradient_accum.copy_(torch.tensor([[11.], [12.], [13.], [14.]]))
            m.denom.fill_(1)
            m.max_radii2D.copy_(torch.tensor([21., 90., 5., 20.]))
        return m, gradients

    def assert_rows(self, m, old, gradients, rows, child_start):
        torch = self.torch
        for group in m.optimizer.param_groups:
            name, a = group['name'], group['params'][0]
            self.assertEqual(len(a), len(rows))
            self.assertIsNone(a.grad)
            state = m.optimizer.state[a]
            expected_m = gradients[name][rows]*.1
            expected_v = gradients[name][rows].square()*.001
            expected_m[child_start:] = 0
            expected_v[child_start:] = 0
            torch.testing.assert_close(state['exp_avg'], expected_m, rtol=1e-5, atol=1e-9)
            torch.testing.assert_close(state['exp_avg_sq'], expected_v, rtol=1e-5, atol=1e-12)
            self.assertEqual(state['step'].item(), 1)
            if name not in ('xyz', 't', 'scaling', 'scaling_t'):
                self.assertTrue(torch.equal(a, old[name][rows]))

    def test_mixed_growth_screen_history_repeat_and_independent_4d_split(self):
        m, gradients = self.model()
        torch = self.torch
        old = {g['name']: cloned(g['params'][0]) for g in m.optimizer.param_groups}
        masks, before_prune = [], []
        prune = m.prune_points
        def observed(mask):
            masks.append(mask.clone())
            before_prune.append((m.max_radii2D.clone(), m.xyz_gradient_accum.clone(), m.t_gradient_accum.clone(), m.denom.clone()))
            return prune(mask)
        def normal(mean, std):
            z = torch.arange(1, std.numel()+1, dtype=std.dtype).reshape_as(std)/10
            return mean + std*z
        with torch.no_grad(), patch.object(m, 'prune_points', observed), patch.object(torch, 'normal', normal):
            m.densify_and_prune(.5, .005, 1., 20)
        # Original parents 1,2 split; parent/clone 0 both screen-pruned.
        self.assertEqual(masks[0].tolist(), [False, True, True, False, False, False, False, False, False])
        self.assertEqual(masks[1].tolist(), [True, False, True, False, False, False, False])
        self.assertEqual(before_prune[1][0].tolist(), [21, 20, 21, 0, 0, 0, 0])
        self.assertEqual(before_prune[1][2].flatten().tolist(), [11, 14, 0, 0, 0, 0, 0])
        self.assertEqual(before_prune[1][3].flatten().tolist(), [1, 1, 0, 0, 0, 0, 0])
        rows = [3, 1, 2, 1, 2]
        self.assert_rows(m, old, gradients, rows, 1)
        expected_xyzt = []
        for j, row in enumerate(rows):
            point = old['xyz'][row].double().tolist()+old['t'][row].double().tolist()
            if j:
                std = old['scaling'][row].double().exp().tolist()+old['scaling_t'][row].double().exp().tolist()
                sample = [s*(4*(j-1)+k+1)/10 for k, s in enumerate(std)]
                delta = rotated(sample, old['rotation'][row].tolist(), old['rotation_r'][row].tolist())
                point = [a+b for a,b in zip(point, delta)]
            expected_xyzt.append(point)
        torch.testing.assert_close(torch.cat((m._xyz, m._t), 1).double(), torch.tensor(expected_xyzt, dtype=torch.float64), rtol=1e-5, atol=1e-6)
        for name, actual in (('scaling', m._scaling), ('scaling_t', m._scaling_t)):
            expected = old[name][rows].clone(); expected[1:] -= math.log(1.6)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for a in (m.max_radii2D, m.xyz_gradient_accum, m.t_gradient_accum, m.denom):
            self.assertEqual(a.count_nonzero().item(), 0)

    def test_clone_copies_all_post_step_rows_and_moments_are_zero(self):
        m, gradients = self.model()
        torch = self.torch
        old = {g['name']: cloned(g['params'][0]) for g in m.optimizer.param_groups}
        with torch.no_grad():
            m.densify_and_clone(torch.ones(4, 1), 1., 1., None, None)
        self.assert_rows(m, old, gradients, [0, 1, 2, 3, 0], 4)
        for group in m.optimizer.param_groups:
            self.assertTrue(torch.equal(group['params'][0], old[group['name']][[0,1,2,3,0]]))
        self.assertEqual(m.max_radii2D.tolist(), [21,90,5,20,21])
        self.assertEqual(m.denom.flatten().tolist(), [1,1,1,1,0])

    def test_prune_only_retains_sliced_window_and_reset_only_keeps_it(self):
        m, _ = self.model(); torch = self.torch
        with torch.no_grad():
            m.densify_and_prune(.5, .005, 1., 20, prune_only=True)
        self.assertEqual(len(m._xyz), 2)
        self.assertEqual(m.max_radii2D.tolist(), [5,20])
        self.assertEqual(m.xyz_gradient_accum.flatten().tolist(), [3,0])
        self.assertEqual(m.t_gradient_accum.flatten().tolist(), [13,14])
        before = cloned(m.capture())
        with torch.no_grad():
            m._opacity[0] = math.log(.001/.999)
            m.reset_opacity()
        torch.testing.assert_close(m.get_opacity[:,0], torch.tensor([.001,.01]), rtol=1e-5, atol=1e-8)
        for idx in (7,8,9,10): self.assertTrue(torch.equal(m.capture()[idx], before[idx]))
        for group in m.optimizer.param_groups:
            state = m.optimizer.state[group['params'][0]]
            if group['name'] == 'opacity':
                self.assertEqual(state['exp_avg'].count_nonzero().item(), 0)
                self.assertEqual(state['exp_avg_sq'].count_nonzero().item(), 0)
            self.assertEqual(state['step'].item(), 1)

    def test_no_selection_one_row_and_empty_result_shapes(self):
        torch = self.torch
        for survivors in (1, 0):
            with self.subTest(survivors=survivors):
                m, _ = self.model()
                with torch.no_grad():
                    m.prune_points(torch.tensor([False, True, True, True]))
                    m.densify_and_prune(100., 0 if survivors else 1., 1., None)
                for group in m.optimizer.param_groups:
                    a = group['params'][0]; self.assertEqual(a.shape[0], survivors)
                    self.assertEqual(m.optimizer.state[a]['exp_avg'].shape, a.shape)
                self.assertEqual(m._xyz.shape, (survivors, 3))
                self.assertEqual(m.denom.shape, (survivors, 1))
                self.assertEqual(m.max_radii2D.shape, (survivors,))

    def test_threshold_equality_world_size_and_zero_gradient_selection(self):
        m, _ = self.model(); torch = self.torch
        with torch.no_grad():
            m._scaling.fill_(math.log(.01))
            # Equality is clone; gradient threshold zero includes zero history.
            m.percent_dense = float(m.get_scaling[0,0])
            m.xyz_gradient_accum.zero_()
            m.densify_and_prune(0, 0, 1., None)
        self.assertEqual(len(m._xyz), 8)
        with torch.no_grad():
            m._scaling[0] = math.log(.1001)
            m.max_radii2D.fill_(20)
            m.max_radii2D[1] = 20.001
            threshold = float(m.get_opacity[2,0])
            m._opacity[3] -= .1
            # Equal opacity/radius survive; strictly greater size is removed.
            m.densify_and_prune(100, threshold, 1., 20, prune_only=True)
        self.assertEqual(len(m._xyz), 5)

    def test_seeded_actual_normal_split_smoke(self):
        m, _ = self.model(); torch = self.torch
        with torch.no_grad():
            m.densify_and_split(torch.ones(4,1), .5, 1., None, None)
        self.assertEqual(len(m._xyz), 7)
        self.assertTrue(torch.isfinite(m.get_xyzt).all())


if __name__ == '__main__':
    unittest.main()
