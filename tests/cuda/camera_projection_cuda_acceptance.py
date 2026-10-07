"""Step 10 separately authorized fresh-build/CUDA entry (not run by CPU tests).

Import/record/preflight use stdlib only. A reviewed content manifest, not HEAD
or tracked-clean, binds execution. Reuses Step 8/9 functions, never their mains.
No fallback, retry, source mutation or cleanup. Recording is not authorization.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT.parent/'reports/corrected-4dgs/phase1/step10'
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
SCHEMA = 'step10-camera-inputs-v1'
HELPER = 'tests/cuda/alpha_cap_cuda_acceptance.py'
EXTRA = ('tests/camera_projection_oracle.py', 'tests/camera_projection_runtime.py',
         'tests/test_camera_projection_cpu.py', 'tests/cuda/camera_projection_cuda_acceptance.py',
         'tests/test_camera_projection_source_identity.py', 'tests/test_formal_camera.py',
         'tests/test_formal_camera_connection.py', 'tests/test_formal_connection.py')
HEAVY = ('torch', 'numpy', 'PIL', 'scene', 'gaussian_renderer', 'pointops2', 'simple_knn')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def helper():
    spec = importlib.util.spec_from_file_location('step10_reviewed_alpha_helper', ROOT/HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT != ROOT: raise RuntimeError('helper_repository_mismatch')
    return module


def source_identity():
    value = helper().source_identity()
    value['schema'] = SCHEMA
    for name in EXTRA:
        path = ROOT/name
        if path.resolve() != path or not path.is_file(): raise RuntimeError('camera_input_missing_or_aliased: '+name)
        value['files'][name] = digest(path)
    value['files'] = dict(sorted(value['files'].items()))
    return value


def load_source_evidence(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise RuntimeError('duplicate_evidence_key')
            result[k] = v
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if (type(value) is not dict or set(value) != {'schema', 'repository', 'base_head', 'files'}
            or value['schema'] != SCHEMA or value['repository'] != str(ROOT)
            or not isinstance(value['base_head'], str)
            or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', value['base_head'])
            or type(value['files']) is not dict or not value['files']):
        raise RuntimeError('evidence_schema')
    for name, sha in value['files'].items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or Path(name).as_posix() != name or '..' in Path(name).parts
                or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
            raise RuntimeError('evidence_file')
    if not set(EXTRA+(HELPER, 'tests/cuda/sh_cuda_acceptance.py')) <= value['files'].keys():
        raise RuntimeError('camera_evidence_incomplete')
    return value


def verify_source(expected):
    for name, sha in expected['files'].items():
        path = ROOT/name
        if path.resolve() != path or not path.is_file() or digest(path) != sha:
            raise RuntimeError('reviewed_input_mismatch: '+name)
    actual = source_identity()  # Helpers only AFTER all reviewed bytes match.
    if any(actual[k] != expected[k] for k in ('schema', 'repository', 'files')):
        raise RuntimeError('source_inputs_mismatch')
    return actual


def alpha_subset(expected, alpha):
    verify_source(expected)
    return dict(schema=alpha.SCHEMA, repository=str(ROOT), base_head=expected['base_head'],
                files={k: expected['files'][k] for k in alpha.source_identity()['files']})


def verify_loaded_inputs(expected, alpha, sh):
    observed = alpha.verify_loaded_inputs(alpha_subset(expected, alpha), sh)
    for name in ('camera_projection_oracle', 'camera_projection_runtime'):
        module = sys.modules.get(name)
        if module is not None:
            if Path(getattr(module, '__file__', '')).resolve() != ROOT/'tests'/ (name+'.py'):
                raise RuntimeError('unexpected_import_source: '+name)
            observed[name] = module.__file__
    if Path(alpha.__file__).resolve() != ROOT/HELPER: raise RuntimeError('unexpected_alpha_helper')
    return observed


def output_path(path):
    if not path.is_absolute() or path.exists() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError('new_unaliased_absolute_output_required')
    if REPORTS not in path.parents or not path.parent.is_dir():
        raise RuntimeError('new_step10_run_directory_required')
    return path


def preflight(expected, out):
    before = verify_source(expected)
    if any(k.startswith('STEP90_CUDA_DEBUG') and v for k, v in os.environ.items()):
        raise RuntimeError('external_diagnostic_override')
    return before, output_path(out)


def compare(torch, actual, expected, label, rows, tolerance):
    from camera_projection_oracle import TOLERANCES
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != e.shape or not bool(torch.isfinite(a).all() & torch.isfinite(e).all()):
        raise AssertionError(label+': shape/nonfinite')
    atol, rtol = TOLERANCES[tolerance]
    error = (a-e).abs(); ratio = error/(atol+rtol*e.abs())
    row = dict(check=label, tolerance=tolerance, atol=atol, rtol=rtol, dtype='float32_vs_cpu_float64',
               max_abs=float(error.max()), max_bound_ratio=float(ratio.max()),
               worst_flat_index=int(ratio.flatten().argmax()))
    rows.append(row)
    if row['max_bound_ratio'] > 1: raise AssertionError(label+': numerical_mismatch')


def invoke(torch, binding, renderer, sh, case, camera, config, *, points=None,
           diagnostic=False, binding_only=False, backward=True, pixel=None, target=0):
    """Actual GPU camera/model/getters/render; no CPU success substitutes."""
    from formal_views import ConsumerView
    from formal_time_handoff import bind_time
    points = case['points'] if points is None else points
    gpu = [{k: v.detach().float().cuda().requires_grad_() for k, v in p.items()} for p in points]
    stack = lambda name: torch.stack([p[name] for p in gpu])
    s = case['camera']; bg = torch.tensor(s['background'], dtype=torch.float32, device='cuda')
    if case['dim'] == 4:
        model = sh.make_model(gpu[0], case.get('stage', (3, 2)), s, formal_time=config.time_derivation)
        # Same real model getters for multi-point fixtures (no initializer/KNN).
        model._xyz, model._t = stack('mean'), stack('time')
        model._scaling, model._scaling_t = stack('log_scale')[:, :3], stack('log_scale')[:, 3:]
        model._rotation, model._rotation_r = stack('left'), stack('right')
        model._opacity = stack('opacity')
        model._features_dc, model._features_rest = stack('features')[:, :1], stack('features')[:, 1:]
    if not binding_only and case['dim'] == 4:
        value = renderer.render(camera, model, ConsumerView(config, 'pipeline'), bg,
                                scaling_modifier=1.0, override_color=None, formal_config=config)
        images = dict(rgb=value['render'], flow=value['flow'], depth=value['depth'], mask=value['alpha'])
        radii, screen = value['radii'], value['viewspace_points']
        debug, pre = value['cuda_pixel_debug'], value['cuda_preprocess_debug']
    else:
        px = pixel or case['pixels'][0]
        settings = binding.GaussianRasterizationSettings(
            image_height=s['height'], image_width=s['width'], tanfovx=camera.formal_camera.tanx,
            tanfovy=camera.formal_camera.tany, bg=bg, scale_modifier=1.0,
            viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=case.get('stage', (3, 2))[0],
            sh_degree_t=case.get('stage', (3, 2))[1] if case['dim'] == 4 else 0,
            campos=camera.camera_center, timestamp=s['timestamp'], time_duration=s['duration'],
            rot_4d=case['dim'] == 4, gaussian_dim=case['dim'], force_sh_3d=False,
            prefiltered=False, debug=False, formal_camera=case['dim'] == 4,
            camera_binding=camera.formal_binding if case['dim'] == 4 else None,
            formal_time=case['dim'] == 4,
            time_binding=bind_time(config, model, camera.formal_binding) if case['dim'] == 4 else None,
            time_backward_debug=(torch.full((8,), -1., device='cuda') if diagnostic else None),
            debug_pixel_x=px[0] if diagnostic else -1, debug_pixel_y=px[1] if diagnostic else -1,
            debug_pixel_max_entries=len(points) if diagnostic else 0,
            debug_preprocess_target_index=target if diagnostic else -1)
        screen = torch.zeros((len(points), 3), dtype=torch.float32, device='cuda', requires_grad=True)
        if case['dim'] == 4:
            means, opacity, features = model.get_xyz, model.get_opacity, model.get_features
            scales, scales_t, times = model.get_scaling, model.get_scaling_t, model.get_t
            left, right = model.get_rotation, model.get_rotation_r
        else:
            means, opacity, features = stack('mean'), stack('opacity').sigmoid(), stack('features')
            scales, scales_t, times = stack('log_scale')[:, :3].exp(), None, None
            left, right = torch.nn.functional.normalize(stack('left'), dim=1), None
        values = binding.GaussianRasterizer(settings)(
            means3D=means, means2D=screen, opacities=opacity, shs=features,
            flow_2d=torch.zeros((len(points), 2), device='cuda'), ts=times,
            scales=scales, scales_t=scales_t, rotations=left, rotations_r=right, prefilter_var=-1.0)
        rgb, radii, depth, mask, flow, _, debug, pre = values
        images = dict(rgb=rgb, flow=flow, depth=depth, mask=mask)
    samples = [torch.cat([images[k][:, y, x] for k in ('rgb', 'flow', 'depth', 'mask')]) for x, y in case['pixels']]
    result = dict(samples=[v.detach().cpu().double() for v in samples],
                  images={k: v.detach().cpu() for k, v in images.items()}, radii=radii.detach().cpu(),
                  debug=debug.detach().cpu(), preprocess=pre.detach().cpu())
    if backward:
        loss = sum(v @ v.new_tensor(case['loss']) for v in samples)
        flat = [v for p in gpu for v in p.values()]+[screen]
        grads = torch.autograd.grad(loss, flat, allow_unused=True)
        result['screen'] = (torch.zeros_like(screen) if grads[-1] is None else grads[-1]).detach().cpu().double()
        iterator = iter(grads[:-1]); result['gradients'] = []
        for p in gpu:
            record = {}
            for k, v in p.items():
                g = next(iterator); record[k] = (torch.zeros_like(v) if g is None else g).detach().cpu().double()
            result['gradients'].append(record)
        if diagnostic and binding_only:
            result['time_backward'] = settings.time_backward_debug.detach().cpu().clone()
    torch.cuda.synchronize()
    return result


def invariant(torch, a, b, case, rows):
    def same_bits(x, y):
        return (x.shape == y.shape and x.dtype == y.dtype and
                torch.equal(x.contiguous().reshape(-1).view(torch.uint8),
                            y.contiguous().reshape(-1).view(torch.uint8)))

    for k in a['images']:
        if not same_bits(a['images'][k], b['images'][k]): raise AssertionError('diagnostic_image_changed: '+k)
    if not same_bits(a['radii'], b['radii']): raise AssertionError('diagnostic_radius_changed')
    if 'gradients' not in a: return 'forward_only'
    single = len(case['pixels']) == len(case['points']) == 1
    if len(a['gradients']) != len(b['gradients']) or len(a['gradients']) != len(case['points']):
        raise AssertionError('diagnostic_gradient_count')
    for i, (x, y) in enumerate(zip(a['gradients'], b['gradients'])):
        if x.keys() != y.keys(): raise AssertionError('diagnostic_gradient_fields')
        for k in x:
            if single:
                if not same_bits(x[k], y[k]): raise AssertionError('diagnostic_gradient_changed: '+k)
            else: compare(torch, x[k], y[k], case['label']+f'/diagnostic/{i}/{k}', rows, 'atomic')
    if single:
        if not same_bits(a['screen'], b['screen']): raise AssertionError('diagnostic_screen_changed')
    else: compare(torch, a['screen'], b['screen'], case['label']+'/diagnostic/screen', rows, 'atomic')
    return 'exact' if single else 'atomic'


def diagnostics(torch, case, actual, expected, pixel, rows, target=0):
    import camera_projection_oracle as o
    n = len(case['points']); debug = actual['debug'].double().reshape(-1, 32)
    if debug.shape != (n+1, 32) or not bool(torch.isfinite(debug).all()): raise AssertionError('pixel_debug_shape')
    head, entries = debug[0], debug[1:]
    if (head[0] != 1 or tuple(head[3:5].tolist()) != tuple(pixel) or
            any(head[i] != n for i in (12, 13, 14, 15, 25)) or
            any(head[i] != 0 for i in (21, 26, 27)) or
            entries[:, 2].tolist() != list(range(n)) or entries[:, 18].count_nonzero()):
        raise AssertionError('missing_skipped_truncated_or_reordered_contribution')
    stack = lambda k: torch.stack([g[k] for g in expected['geometry']])
    for sl, value, name in ((slice(4, 6), stack('xy'), 'pixel_screen'),
            (slice(10, 13), stack('color'), 'color'), (16, stack('raw'), 'raw'),
            (17, expected['alpha'], 'alpha'), (19, expected['trans'][:-1], 'prefix'),
            (20, expected['trans'][1:], 'transmittance')):
        compare(torch, entries[:, sl], value, case['label']+'/'+name, rows, 'pixel' if name != 'pixel_screen' else 'screen')
    if not torch.equal(entries[:, 20], entries[:, 21]): raise AssertionError('contribution_not_applied')
    g, s = expected['geometry'][target], case['camera']
    pre = actual['preprocess'].double().flatten()
    if pre.numel() != 104 or pre[0] != 1 or pre[1] != target: raise AssertionError('preprocess_debug_shape')
    a = g['jacobian']; j = torch.cat((a.T, torch.zeros(3, 1, dtype=torch.float64)), dim=1)
    w = s['rotation'].T; t = w @ j
    c = g['screen_cov']; q = g['inverse']; cov = g['covariance']
    packed = torch.stack((cov[0, 0], cov[0, 1], cov[0, 2], cov[1, 1], cov[1, 2], cov[2, 2]))
    for sl, value, name in ((slice(5, 8), g['mean'], 'conditional_mean'), (9, g['effective_opacity'], 'opacity'),
            (slice(26, 32), packed, 'world_covariance'), (slice(32, 35), g['view'], 'view'),
            (slice(35, 38), g['clamped'], 'clamped'), (slice(38, 47), j.T.flatten(), 'J'),
            (slice(47, 56), w.T.flatten(), 'W'), (slice(56, 65), t.T.flatten(), 'T'),
            (slice(65, 69), (c-.3*torch.eye(2, dtype=torch.float64)).flatten(), 'cov_before'),
            (slice(69, 73), c.flatten(), 'cov_after'), (73, torch.linalg.det(c), 'det'),
            (slice(74, 77), torch.stack((q[0, 0], q[0, 1], q[1, 1])), 'conic'),
            (slice(82, 84), g['xy'], 'screen'), (84, g['depth'], 'depth'),
            (slice(91, 93), g['delta'], 'delta'), (93, g['power'], 'power'), (94, g['raw'], 'pre_raw')):
        compare(torch, pre[sl], value, case['label']+'/'+name, rows, 'screen' if name in ('screen', 'delta') else 'geometry')
    focal = o.tensor([s['fx'], s['fy'], s['tanx'], s['tany']])
    if not bool(((pre[85:89]-focal).abs() <= 8*torch.finfo(torch.float32).eps*focal.abs()).all()):
        raise AssertionError('focal_tan_handoff')
    top = o.topology(g, s)
    if pre[77] != top['radius'] or pre[78:82].tolist() != top['tiles']: raise AssertionError('radius_tile_identity')
    if case['kind'] == 'equality':
        axis = case['axis']
        if o.bits(pre[32+axis]) != case['boundary_bits']: raise AssertionError('equality_view_bits')
        limit = o.f32(o.f32(1.3)*float(pre[87+axis]))
        ratio = o.f32(float(pre[32+axis])/float(pre[34]))
        inside = -limit <= ratio <= limit
        if inside != (case['side'] != 'outside'): raise AssertionError('equality_selected_branch')
        if o.bits(pre[35+axis]) != o.bits(o.f32(max(-limit, min(limit, ratio))*float(pre[34]))):
            raise AssertionError('equality_clamped_bits')
    return dict(pixel=list(pixel), count=n, target=target, topology=top,
                view_bits=[o.bits(x) for x in pre[32:35]], raw_alpha=entries[:, 16].tolist())


def run_camera_checks(torch, binding, renderer, sh, result):
    import camera_projection_oracle as o
    from camera_projection_runtime import make_camera, check_camera
    cases = o.cases(); rows = result['checks']
    result.update(planned_cases=[c['label'] for c in cases], cases=[], fd_forward_invocations=0)
    saved = {}
    with tempfile.TemporaryDirectory(prefix='camera-projection-cuda-input-') as temp:
        for case in cases:
            label = case['label']; result['active_case'] = label
            expected = o.evaluate(case)
            domain = o.domain(case, signal=case['kind'] in ('smooth', 'mixed', 'diagnostic'))
            cpu, config = make_camera(Path(temp), case['camera'])
            check_camera(torch, cpu, case['camera'], rows)
            camera = cpu.cuda()  # Actual device transfer, never patched here.
            for name in ('world_view_transform', 'projection_matrix', 'full_proj_transform', 'camera_center'):
                src, dst = getattr(cpu, name), getattr(camera, name)
                if dst.device.type != 'cuda' or not torch.equal(src, dst.cpu()): raise AssertionError('camera_transfer_bits')
            check_camera(torch, camera, case['camera'], rows)
            if camera is cpu or camera.formal_binding.camera is not camera: raise AssertionError('transferred_binding')
            culled = case['kind'] == 'cull' and not domain[0][0]['visible']
            differentiable = case['kind'] not in ('cull', 'discrete')
            actual = invoke(torch, binding, renderer, sh, case, camera, config, backward=differentiable or culled)
            record = dict(case=label, status='incomplete', kind=case['kind'], domain=domain,
                          camera=dict(mode=case['camera']['mode'], divisor=case['camera']['divisor']),
                          pixels=case['pixels'], diagnostic=[], invariance_modes=[])
            result['cases'].append(record)
            if culled:
                if actual['radii'].count_nonzero(): raise AssertionError('near_cull_radius')
                for k, value in actual['images'].items():
                    reference = torch.tensor(case['camera']['background']).view(3, 1, 1).expand_as(value) if k == 'rgb' else torch.zeros_like(value)
                    if not torch.equal(value, reference): raise AssertionError('culled_image_'+k)
                if any(v.count_nonzero() for p in actual['gradients'] for v in p.values()) or actual['screen'].count_nonzero():
                    raise AssertionError('culled_gradient')
            else:
                for i, ref in enumerate(expected):
                    compare(torch, actual['samples'][i], ref['outputs'], label+f'/pixel/{i}', rows, 'pixel')
                if actual['radii'].tolist() != [g['radius'] for g in domain[0]]: raise AssertionError('actual_radius')
                if differentiable:
                    exact, compat, old = [o.gradients(case, method=m) for m in ('analytic', 'compat', 'old')]
                    compare(torch, actual['screen'], sum(e['screen'] for e in expected), label+'/screen_gradient', rows, 'gradient')
                    record['position_paths'] = dict(covariance_view=o.covariance_view_gradient(case).tolist(),
                        exact=exact[0]['mean'].tolist(), compat=compat[0]['mean'].tolist(), old=old[0]['mean'].tolist())
                    for i, g in enumerate(exact):
                        for k, value in g.items():
                            compare(torch, actual['gradients'][i][k], value, label+f'/exact/{i}/{k}', rows, 'gradient')
                            compare(torch, actual['gradients'][i][k], compat[i][k], label+f'/compat/{i}/{k}', rows, 'gradient')
                normal = invoke(torch, binding, renderer, sh, case, camera, config, binding_only=True, backward=differentiable)
                record['invariance_modes'].append(invariant(torch, actual, normal, case, rows))
                for pixel, ref in zip(case['pixels'], expected):
                    for target in range(len(case['points'])):
                        checked = invoke(torch, binding, renderer, sh, case, camera, config,
                                         diagnostic=True, binding_only=True, backward=differentiable, pixel=pixel, target=target)
                        record['invariance_modes'].append(invariant(torch, normal, checked, case, rows))
                        record['diagnostic'].append(diagnostics(torch, case, checked, ref, pixel, rows, target))
                for coordinate in o.fd_coordinates(case):
                    for step in o.CUDA_STEPS:
                        plus, minus, den = o.perturb(case['points'], coordinate, step, rounded=True)
                        ends = []
                        for points in (plus, minus):
                            if o.topology_key(o.domain(case, points, signal=True)) != o.topology_key(domain): raise AssertionError('FD_domain')
                            endpoint = invoke(torch, binding, renderer, sh, case, camera, config, points=points, backward=False)
                            endpoint_ref = o.evaluate(case, points)
                            for pixel, ref in zip(case['pixels'], endpoint_ref):
                                checked = invoke(torch, binding, renderer, sh, case, camera, config, points=points,
                                    diagnostic=True, binding_only=True, backward=False, pixel=pixel)
                                invariant(torch, endpoint, checked, case, rows)
                                diagnostics(torch, case, checked, ref, pixel, rows)
                                result['fd_forward_invocations'] += 1
                            ends.append(sum(v @ v.new_tensor(case['loss']) for v in endpoint['samples']))
                            result['fd_forward_invocations'] += 1
                        i, name, offset = coordinate
                        compare(torch, actual['gradients'][i][name].flatten()[offset], (ends[0]-ends[1])/den,
                                label+f'/FD/{coordinate}/{step}', rows, 'cuda_fd')
                if case['kind'] == 'equality' and case['side'] == 'equal' and case['sign'] == 1:
                    axis = case['axis']; center = sum(v @ v.new_tensor(case['loss']) for v in actual['samples'])
                    for step in o.CUDA_STEPS:
                        plus, minus, _ = o.perturb(case['points'], (0, 'mean', axis), step, rounded=True)
                        for side, points in (('inside', minus), ('outside', plus)):
                            endpoint = invoke(torch, binding, renderer, sh, case, camera, config, points=points, backward=False)
                            h = float(points[0]['mean'][axis]-case['points'][0]['mean'][axis])
                            slope = (sum(v @ v.new_tensor(case['loss']) for v in endpoint['samples'])-center)/h
                            cpu_slope = (o.loss(case, points)-o.loss(case))/h
                            compare(torch, slope, cpu_slope, label+f'/one_sided/{side}/{step}', rows, 'cuda_fd')
                            # Diagnose the finite endpoint with its OWN intended branch.
                            end_case = dict(case, kind='one_sided', points=points)
                            if bool(o.evaluate(end_case)[0]['geometry'][0]['inside'][axis]) != (side == 'inside'):
                                raise AssertionError('one_sided_branch')
                            checked = invoke(torch, binding, renderer, sh, end_case, camera, config,
                                             diagnostic=True, binding_only=True, backward=False)
                            diagnostics(torch, end_case, checked, o.evaluate(end_case)[0], case['pixels'][0], rows)
                            result['fd_forward_invocations'] += 2
            record['status'] = 'passed'; saved[label] = actual
        for i, g in enumerate(saved['M/sum']['gradients']):
            for k, value in g.items():
                compare(torch, value, saved['M/p0']['gradients'][i][k]+saved['M/p1']['gradients'][i][k],
                        f'pixel_additivity/{i}/{k}', rows, 'atomic')
    if len(result['cases']) != len(cases) or any(r['status'] != 'passed' for r in result['cases']):
        raise AssertionError('missing_camera_case')
    result['invariance_coverage'] = {mode: sum(r['invariance_modes'].count(mode) for r in result['cases'])
                                     for mode in ('exact', 'atomic', 'forward_only')}
    if not all(result['invariance_coverage'][mode] for mode in ('exact', 'atomic')):
        raise AssertionError('missing_diagnostic_gradient_coverage')
    result['case_count'] = len(cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-source-evidence', action='store_true')
    parser.add_argument('--source-evidence', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if sys.executable != PYTHON or not sys.dont_write_bytecode: raise RuntimeError('4dgs310_B_required')
    if any(n in sys.modules for n in HEAVY): raise RuntimeError('fresh_process_required')
    if args.record_source_evidence:
        if args.source_evidence is not None or args.output_dir is not None: parser.error('record_cannot_execute')
        print(json.dumps(source_identity(), indent=2)); return
    if args.source_evidence is None or args.output_dir is None: parser.error('reviewed_source_and_new_output_required')
    expected = load_source_evidence(args.source_evidence)
    before, out = preflight(expected, args.output_dir)
    alpha = helper(); sh = alpha.helper(); git_before = alpha.git_state(sh)
    out.mkdir(); build = out/'build'; build.mkdir()
    os.environ['TORCH_EXTENSIONS_DIR'] = str(build)
    sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
    result = dict(status='failed', stage='heavy_import', python=sys.executable, source_before=before,
                  reviewed_source=expected, git_before=git_before, checks=[])
    try:
        import torch
        result['environment'] = dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), nvcc=sh.bounded_command(['nvcc', '--version']),
            compiler=sh.bounded_command(['c++', '--version']),
            driver=sh.bounded_command(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']))
        if not torch.cuda.is_available(): raise RuntimeError('CUDA_required_no_fallback')
        result['environment']['gpu'] = torch.cuda.get_device_name(0)
        verify_source(expected); result['stage'] = 'fresh_build'
        with (out/'build.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
            import gaussian_renderer as renderer
            from gaussian_renderer import diff_gaussian_rasterization as binding
        import simple_knn._C as knn
        result['simple_knn_import_only'] = dict(path=knn.__file__, sha256=digest(Path(knn.__file__)))
        result['extension'] = alpha.binary_identity(binding, build, sh)
        result['loaded_before'] = verify_loaded_inputs(expected, alpha, sh)
        result['stage'] = 'camera_matrix'; run_camera_checks(torch, binding, renderer, sh, result)
        result['stage'] = 'same_binary_alpha_regression'
        ar = dict(source_before=alpha_subset(expected, alpha), checks=[]); result['alpha_regression'] = ar
        alpha.run_alpha_checks(torch, binding, renderer, sh, ar)
        if ar['case_count'] != 34 or len(ar['checks']) != 2849: raise AssertionError('alpha_coverage')
        result['stage'] = 'same_binary_SH_regression'
        sr = dict(source_before=alpha.sh_subset(ar['source_before'], sh), checks=[]); result['sh_regression'] = sr
        sh.run_checks(torch, binding, renderer, sr)
        if sr['case_count'] != 15 or len(sr['checks']) != 309: raise AssertionError('SH_coverage')
        torch.cuda.synchronize(); result['stage'] = 'preservation'
        if alpha.binary_identity(binding, build, sh) != result['extension']: raise RuntimeError('binary_recipe_objects_changed')
        result['source_after'] = verify_source(expected)
        result['loaded_after'] = verify_loaded_inputs(expected, alpha, sh)
        result['git_after'] = alpha.git_state(sh)
        if result['git_after'] != git_before: raise RuntimeError('git_or_index_changed')
        result['status'] = 'passed'; result['stage'] = 'complete'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'[:4000]
        raise
    finally:
        if result['status'] != 'passed':
            preservation = {}
            for name, operation in (
                ('inputs', lambda: verify_source(expected)['files'] == expected['files']),
                ('git', lambda: alpha.git_state(sh) == git_before),
                ('binary', lambda: 'extension' not in result or alpha.binary_identity(binding, build, sh) == result['extension'])):
                try: preservation[name+'_unchanged'] = operation()
                except Exception as error: preservation[name+'_error'] = f'{type(error).__name__}: {error}'[:1000]
            result['failure_preservation'] = preservation
        with (out/'result.json').open('x') as stream: json.dump(result, stream, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
