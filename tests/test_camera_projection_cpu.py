"""Step 10 independent float64 math and real CPU camera consumers; never CUDA."""
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).parent)]
import torch
import camera_projection_oracle as o

MEASUREMENTS = []
COVERAGE = dict(domain_centers={}, fd_cases={}, discrete_pairs={}, diagnostic_checker={})


def compare(a, e, tolerance='analytic', label=''):
    a, e = torch.as_tensor(a).detach().double(), torch.as_tensor(e).detach().double()
    assert a.shape == e.shape and bool(torch.isfinite(a).all() & torch.isfinite(e).all()), label
    atol, rtol = o.TOLERANCES[tolerance]
    delta = (a-e).abs()
    ratio = delta/(atol+rtol*e.abs())
    MEASUREMENTS.append(dict(check=label, tolerance=tolerance, max_abs=float(delta.max()), max_bound_ratio=float(ratio.max())))
    assert bool((ratio <= 1).all()), (label, float(delta.max()), float(ratio.max()))


class MathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.executable != '/home/demo/miniconda3/envs/4dgs310/bin/python' or not sys.dont_write_bytecode:
            raise RuntimeError('4dgs310_B_required')
        guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU_only'))
        guard.start(); cls.addClassCleanup(guard.stop)
        cls.cases = o.cases()
        print('executable:', sys.executable, 'camera fixture count:', len(cls.cases))

    def test_hand_anchor_old_negative_and_general_matrix_vjp(self):
        s = o.camera(width=4, height=4, fx=2., fy=2., identity=True)
        # lim=1 exactly is a mathematical anchor independent of raster constants.
        with patch.object(o, 'limits', return_value=o.tensor([1., 1.])):
            t = o.tensor([3., .25, 2.]).requires_grad_()
            sigma = torch.diag(o.tensor([1., 2., 3.])).requires_grad_()
            h = o.tensor([[1., 0.], [0., 0.]])
            g = o.footprint(t, sigma, s)
            d, ds = o.footprint_vjp(t, sigma, s, h)
            old, _ = o.footprint_vjp(t, sigma, s, h, old=True)
            compare(g['screen_cov'][0, 0], o.tensor(4.3), label='anchor/C00')
            compare(d, o.tensor([0., 0., -4.]), label='anchor/gradient')
            compare(old, o.tensor([0., 0., -7.]), label='anchor/old')
            gt, gs = torch.autograd.grad(g['screen_cov'][0, 0], (t, sigma))
            compare(d, gt, label='anchor/autograd'); compare(ds, gs, label='anchor/covariance')
        for case in self.cases:
            if case['kind'] not in ('smooth', 'mixed', 'camera', 'addition', 'diagnostic'): continue
            actual = o.gradients(case); expected = o.gradients(case, method='exact')
            for i, (a, e) in enumerate(zip(actual, expected)):
                for k in a: compare(a[k], e[k], label=case['label']+f'/analytic/{i}/{k}')

    def test_fixture_domain_signal_p2_and_old_formula_gap(self):
        for case in self.cases:
            with self.subTest(case=case['label']):
                d = o.domain(case, signal=case['kind'] in ('smooth', 'mixed', 'diagnostic'))
                COVERAGE['domain_centers'][case['label']] = d
                if case['kind'] == 'equality':
                    g = o.evaluate(case)[0]['geometry'][0]
                    axis = case['axis']
                    self.assertEqual(o.bits(g['view'][axis]), case['boundary_bits'])
                    self.assertEqual(bool(g['inside'][axis]), case['side'] != 'outside')
                if case['kind'] == 'cull':
                    self.assertEqual(d[0][0]['visible'], int(case['label'][1:]) >= 4)

    def test_same_branch_cpu_finite_difference_and_cuda_endpoint_domains(self):
        for case in self.cases:
            # The extra depth/RGB/mixed losses share S4/S7/S8 geometry. Check
            # their signals, P2 budget and CPU FD at the same endpoint sets;
            # the CUDA entry retains its original representative FD schedule.
            fd_case = dict(case, kind='smooth', label=case['label'].split('/')[0]) if case['kind'] == 'mixed' else case
            coordinates = o.fd_coordinates(fd_case)
            if not coordinates: continue
            baseline = o.topology_key(o.domain(case, signal=True))
            gradients = o.gradients(case)
            record = dict(coordinates=coordinates, steps=list(o.CPU_STEPS+o.CUDA_STEPS),
                          endpoints=0, cpu_fd_comparisons=0, min_radius_margin=1.,
                          min_covariance_signal=float('inf'), min_total_signal=float('inf'),
                          min_old_gap=None, max_p2_ratio=0.)
            COVERAGE['fd_cases'][case['label']] = record
            for coordinate in coordinates:
                for step in o.CPU_STEPS+o.CUDA_STEPS:
                    rounded = step in o.CUDA_STEPS
                    plus, minus, denominator = o.perturb(case['points'], coordinate, step, rounded)
                    with self.subTest(case=case['label'], coordinate=coordinate, step=step):
                        for points in (plus, minus):
                            endpoint = o.domain(case, points, signal=True)
                            self.assertEqual(o.topology_key(endpoint), baseline)
                            signal = endpoint[0][0]['signal']
                            record['endpoints'] += 1
                            record['min_radius_margin'] = min(record['min_radius_margin'],
                                *(abs(n['radius_float']-round(n['radius_float'])) for ns in endpoint for n in ns))
                            for name in ('covariance', 'total'):
                                record['min_'+name+'_signal'] = min(record['min_'+name+'_signal'], signal[name+'_bound_ratio'])
                            if signal['old_gap_bound_ratio'] is not None:
                                record['min_old_gap'] = min(record['min_old_gap'] or float('inf'), signal['old_gap_bound_ratio'])
                            record['max_p2_ratio'] = max(record['max_p2_ratio'], signal['p2_bound_ratio'])
                        if not rounded:
                            point, name, offset = coordinate
                            fd = (o.loss(case, plus)-o.loss(case, minus))/denominator
                            compare(fd, gradients[point][name].flatten()[offset], 'cpu_fd', case['label']+str(coordinate)+str(step))
                            record['cpu_fd_comparisons'] += 1
            self.assertEqual(record['endpoints'], len(coordinates)*8)
            self.assertEqual(record['cpu_fd_comparisons'], len(coordinates)*2)
        self.assertEqual(len(COVERAGE['fd_cases']['S7']['coordinates']), 16)

    def test_equality_selected_vjp_and_one_sided_limits(self):
        s = o.camera(identity=True)
        sigma = o.tensor([[.6, .08, .02], [.08, .9, -.04], [.02, -.04, 1.1]])
        h = o.tensor([[.3, .1], [.1, .2]])
        for axis in (0, 1):
            for sign in (-1, 1):
                t = o.tensor([.125, .09375, 1.]); t[axis] = sign*o.limits(s)[axis]; t.requires_grad_()
                selected, _ = o.footprint_vjp(t, sigma, s, h)
                auto, = torch.autograd.grad((o.footprint(t, sigma, s)['screen_cov']*h).sum(), (t,))
                compare(selected, auto, label=f'equality/{axis}/{sign}')
                outer = [True, True]; outer[axis] = False
                outside, _ = o.footprint_vjp(t, sigma, s, h, selected=outer)
                center = float((o.footprint(t, sigma, s)['screen_cov']*h).sum())
                for coordinate in (axis, 2):
                    for side, direction, expected in (('inside', -sign if coordinate == axis else 1, selected),
                                                       ('outside', sign if coordinate == axis else -1, outside)):
                        errors = []
                        for step in o.CPU_STEPS:
                            end = t.detach().clone(); end[coordinate] += direction*step
                            slope = ((o.footprint(end, sigma, s)['screen_cov']*h).sum()-center)/(direction*step)
                            errors.append(abs(float(slope-expected[coordinate])))
                        self.assertLessEqual(errors[1], errors[0]*.6+1e-7, (axis, sign, side, errors))
        t = torch.cat((o.limits(s), o.tensor([1.]))).requires_grad_()
        expected, = torch.autograd.grad((o.footprint(t, sigma, s)['screen_cov']*h).sum(), (t,))
        compare(o.footprint_vjp(t, sigma, s, h)[0], expected, label='both_equal')

    def test_projection_half_pixel_lowpass_and_clip_separation(self):
        s1, s2 = o.camera_cases()[:2]
        p = o.point(s1, [.3, .2, 3.])
        a, b = [o.geometry(p, s, (0, 0)) for s in (s1, s2)]
        compare(b['xy'], (a['xy']+.5)/2-.5, label='resolution_half_pixel')
        eye = torch.eye(2, dtype=torch.float64)*.3
        compare(b['screen_cov']-eye, (a['screen_cov']-eye)/4, label='resolution_lowpass')
        for s in o.camera_cases():
            view, projection, full = o.matrices(s)
            mu = torch.cat((o.conditional(p, s)[0].detach(), o.tensor([1.])))
            homogeneous = full @ mu
            xy = ((homogeneous[:2]/(homogeneous[3]+o.EPS)+1)*o.tensor([s['width'], s['height']])-1)/2
            compare(xy, o.geometry(p, s, (0, 0))['xy'], label='matrix_pinhole')
            for z, target in ((.01, 0.), (100., 1.)):
                v = projection @ o.tensor([0., 0., z, 1.])
                compare(v[2]/v[3], o.tensor(target), label='clip_mapping')

    def test_additivity_and_mixed_paths_not_total_z_halving(self):
        lookup = {c['label']: c for c in self.cases}
        gs = [o.gradients(lookup['M/'+s]) for s in ('p0', 'p1', 'sum')]
        for i in range(2):
            for k in gs[0][i]: compare(gs[2][i][k], gs[0][i][k]+gs[1][i][k], label='addition/'+k)
        c = lookup['S7/mixed']
        actual = o.gradients(c)
        individuals = [o.gradients(dict(c, loss=tuple(float(i == j) for i in range(7)))) for j in range(7)]
        for k in actual[0]:
            compare(actual[0][k], sum(c['loss'][j]*individuals[j][0][k] for j in range(7)), label='mixed/'+k)
        old = o.gradients(c, method='old')[0]['mean']
        self.assertGreater(float((old/2-actual[0]['mean']).abs().max()), .001)

    def test_tile_pair_crosses_post_move_rectangle(self):
        pair = [o.domain(c)[0][0] for c in self.cases if c['label'].startswith('tile/')]
        self.assertEqual(len(pair), 2)
        self.assertEqual([n['tiles'] for n in pair], [[0, 0, 2, 3], [0, 0, 3, 3]])
        self.assertEqual(pair[0]['radius'], pair[1]['radius'])
        self.assertEqual(pair[0]['clamp'], [False, True])
        self.assertEqual(pair[0]['clamp'], pair[1]['clamp'])
        COVERAGE['discrete_pairs']['tile'] = pair

    def test_radius_pair_crosses_ceil_independently(self):
        pair = [o.domain(c)[0][0] for c in self.cases if c['label'].startswith('radius/')]
        self.assertEqual(len(pair), 2)
        self.assertEqual([n['radius'] for n in pair], [30, 31])
        self.assertLess(pair[0]['radius_float'], 30)
        self.assertGreater(pair[1]['radius_float'], 30)
        self.assertEqual(pair[0]['tiles'], pair[1]['tiles'])
        self.assertEqual(pair[0]['clamp'], pair[1]['clamp'])
        COVERAGE['discrete_pairs']['radius'] = pair

    def test_single_pixel_exact_and_multi_pixel_atomic_checker(self):
        spec = importlib.util.spec_from_file_location('camera_diagnostic_checker',
            Path(__file__).parent/'cuda/camera_projection_cuda_acceptance.py')
        entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
        singles = [c for c in self.cases if len(c['points']) == len(c['pixels']) == 1]
        self.assertEqual([c['label'] for c in singles], ['D/single'])
        single = singles[0]; multi = next(c for c in self.cases if c['label'] == 'S7/mixed')
        self.assertIs(single['points'], multi['points'])
        self.assertEqual(single['pixels'], multi['pixels'][:1])
        # Synthetic values exercise the real checker, NOT a substitute render
        # or GPU gradient measurement. Signed zero proves actual bit comparison.
        actual = dict(images={k: torch.zeros(n, 1, 1) for k, n in
                              (('rgb', 3), ('flow', 2), ('depth', 1), ('mask', 1))},
                      radii=torch.tensor([83], dtype=torch.int32),
                      gradients=[{k: torch.zeros_like(v) for k, v in single['points'][0].items()}],
                      screen=torch.zeros(1, 3, dtype=torch.float64))
        self.assertEqual(entry.invariant(torch, actual, copy.deepcopy(actual), single, []), 'exact')
        rejected = []
        targets = [('image/'+k, lambda d, k=k: d['images'][k]) for k in actual['images']]
        targets += [('gradient/'+k, lambda d, k=k: d['gradients'][0][k]) for k in actual['gradients'][0]]
        targets += [('screen', lambda d: d['screen'])]
        for name, locate in targets:
            bad = copy.deepcopy(actual); locate(bad).flatten()[0] = -0.
            with self.subTest(field=name), self.assertRaisesRegex(AssertionError, 'diagnostic_'):
                entry.invariant(torch, actual, bad, single, [])
            rejected.append(name)
        bad = copy.deepcopy(actual); bad['radii'][0] += 1
        with self.assertRaisesRegex(AssertionError, 'diagnostic_radius_changed'):
            entry.invariant(torch, actual, bad, single, [])
        for key, value in (('gradients', []),):
            bad = dict(actual, **{key: value})
            with self.assertRaisesRegex(AssertionError, 'diagnostic_gradient_count'):
                entry.invariant(torch, actual, bad, single, [])
        near = copy.deepcopy(actual); near['gradients'][0]['mean'][0] = 1e-7
        near['screen'][0, 0] = 1e-7
        with self.assertRaisesRegex(AssertionError, 'diagnostic_gradient_changed'):
            entry.invariant(torch, actual, near, single, [])
        rows = []
        self.assertEqual(entry.invariant(torch, actual, near, multi, rows), 'atomic')
        far = copy.deepcopy(actual); far['gradients'][0]['mean'][0] = 1e-3
        with self.assertRaisesRegex(AssertionError, 'numerical_mismatch'):
            entry.invariant(torch, actual, far, multi, [])
        COVERAGE['diagnostic_checker'] = dict(single_case=single['label'], signed_zero_rejections=rejected,
            radius_and_count_rejection=True, sub_atomic_error_rejected_exact=True,
            atomic_small_error_accepted=True, atomic_large_error_rejected=True,
            atomic_comparisons=rows, actual_cuda_invariance_tested=False)

    def test_cpu_did_not_initialize_cuda(self):
        self.assertFalse(torch.cuda.is_initialized())


class CameraConnectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_formal_connection as fixtures
        cls.addClassCleanup(fixtures.ConnectionTests.doClassCleanups)
        fixtures.ConnectionTests.setUpClass()  # Real CPU dependencies, GPU/JIT guards only.

    def test_actual_loader_camera_all_specs_and_train_test_identity(self):
        from camera_projection_runtime import make_camera, check_camera
        with tempfile.TemporaryDirectory(prefix='camera-projection-cpu-') as temp:
            for spec in o.camera_cases()+[o.camera(identity=True)]:
                for split in ('train', 'test'):
                    with self.subTest(camera=spec['label'], split=split):
                        camera, config = make_camera(Path(temp), spec, split=split)
                        check_camera(torch, camera, spec, MEASUREMENTS)
                        self.assertEqual(camera.formal_camera.frame_key, (split, 0, split+'.png'))
                        self.assertIs(camera.formal_camera.resolution, config.dataset.resolution)
                        self.assertTrue(all(t.device.type == 'cpu' for t in
                            (camera.world_view_transform, camera.projection_matrix, camera.full_proj_transform)))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    torch.set_num_threads(1)
    program = unittest.main(exit=False)
    summary = {}
    for row in MEASUREMENTS:
        group = summary.setdefault(row.get('tolerance', 'camera_matrix'),
            dict(comparisons=0, max_abs=0., max_bound_ratio=0.))
        group['comparisons'] += 1
        group['max_abs'] = max(group['max_abs'], row['max_abs'])
        if 'max_bound_ratio' in row:
            group['max_bound_ratio'] = max(group['max_bound_ratio'], row['max_bound_ratio'])
    print('CAMERA_CPU_METRICS '+json.dumps(dict(python=sys.executable,
        cuda_initialized=torch.cuda.is_initialized(), tests=program.result.testsRun,
        failures=len(program.result.failures), errors=len(program.result.errors),
        skipped=len(program.result.skipped), comparisons=MEASUREMENTS,
        summary=summary, coverage=COVERAGE), sort_keys=True))
    sys.exit(not program.result.wasSuccessful())
