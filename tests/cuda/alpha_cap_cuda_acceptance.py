"""Step 9 explicit, separately authorized fresh-build/CUDA acceptance entry.

Import and --record-source-evidence are stdlib-only, with no output directory,
toolchain/GPU probe, renderer import or JIT. Recording is not approval.
Execution requires --source-evidence REVIEWED_JSON --output-dir NEW_STEP9_RUN.
Never regenerate expected evidence in a run. No fallback, retry or cleanup.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT.parent/'reports/corrected-4dgs/phase1/step9'
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
SCHEMA = 'step9-alpha-inputs-v1'
HELPER = 'tests/cuda/sh_cuda_acceptance.py'
EXTRA = ('tests/alpha_cap_oracle.py', 'tests/test_alpha_cap_cpu.py',
         'tests/cuda/alpha_cap_cuda_acceptance.py', 'tests/test_alpha_cap_source_identity.py')
HEAVY = ('torch', 'numpy', 'gaussian_renderer', 'scene', 'pointops2', 'simple_knn')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def helper():
    spec = importlib.util.spec_from_file_location('step9_reviewed_sh_helper', ROOT/HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.ROOT != ROOT: raise RuntimeError('helper_repository_mismatch')
    return module


def source_identity():
    identity = helper().source_identity()
    identity['schema'] = SCHEMA
    for name in EXTRA:
        p = ROOT/name
        if p.resolve() != p or not p.is_file(): raise RuntimeError('alpha_input_missing_or_aliased: '+name)
        identity['files'][name] = digest(p)
    identity['files'] = dict(sorted(identity['files'].items()))
    return identity


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
    if not set(EXTRA+(HELPER,)) <= value['files'].keys(): raise RuntimeError('alpha_evidence_incomplete')
    return value


def verify_source(expected):
    # Check reviewed helper bytes BEFORE executing even its stdlib top level.
    for name, sha in expected['files'].items():
        p = ROOT/name
        if p.resolve() != p or not p.is_file() or digest(p) != sha:
            raise RuntimeError('reviewed_input_mismatch: '+name)
    actual = source_identity()
    if any(actual[k] != expected[k] for k in ('schema', 'repository', 'files')):
        raise RuntimeError('source_inputs_mismatch')
    return actual


def sh_subset(expected, sh):
    # The current enumeration chooses keys, never new expected byte identities.
    verify_source(expected)
    keys = sh.source_identity()['files'].keys()
    return dict(schema=sh.INPUT_SCHEMA, repository=str(ROOT), base_head=expected['base_head'],
                files={k: expected['files'][k] for k in keys})


def verify_loaded_inputs(expected, sh):
    observed = sh.verify_loaded_inputs(sh_subset(expected, sh))
    for name, relative in (('alpha_cap_oracle', EXTRA[0]), ('sh_oracle', 'tests/sh_oracle.py'),
                           ('test_formal_config', 'tests/test_formal_config.py')):
        module = sys.modules.get(name)
        if module is None: continue
        path = getattr(module, '__file__', None)
        if path is None or Path(path).resolve() != ROOT/relative:
            raise RuntimeError('unexpected_import_source: '+name)
        observed[name] = str(Path(path).resolve())
    if Path(sh.__file__).resolve() != ROOT/HELPER: raise RuntimeError('unexpected_helper_source')
    observed['sh_acceptance_helper'] = str(Path(sh.__file__).resolve())
    return observed


def output_path(out):
    if not out.is_absolute() or out.exists() or out.is_symlink() or out.resolve() != out:
        raise RuntimeError('new_unaliased_absolute_output_required')
    if REPORTS not in out.parents or not out.parent.is_dir():
        raise RuntimeError('new_step9_run_directory_required')
    return out


def binary_identity(binding, build, sh):
    binary = Path(binding._C.__file__).resolve()
    if build not in binary.parents: raise RuntimeError('loaded_binary_outside_fresh_build')
    if Path(binding.__file__).resolve() != ROOT/'gaussian_renderer/diff_gaussian_rasterization.py':
        raise RuntimeError('unexpected_binding')
    recipe = binary.parent/'build.ninja'
    objects = sorted(binary.parent.glob('*.o'))
    if not recipe.is_file() or len(objects) != len(sh.TRANSLATION_UNITS):
        raise RuntimeError('fresh_build_evidence_missing')
    if not all(str(ROOT/p).replace(' ', '$ ') in recipe.read_text() for p in sh.TRANSLATION_UNITS):
        raise RuntimeError('translation_units_mismatch')
    return dict(path=str(binary), sha256=digest(binary), ninja_sha256=digest(recipe),
                objects={p.name: digest(p) for p in objects}, translation_units=list(sh.TRANSLATION_UNITS))


def git_state(sh):
    index = Path(sh.git('rev-parse', '--path-format=absolute', '--git-path', 'index'))
    return dict(head=sh.git('rev-parse', 'HEAD'), branch=sh.git('branch', '--show-current'),
                status=sh.git('status', '--porcelain=v1', '--untracked-files=all'),
                index_sha256=digest(index))


def compare(torch, actual, expected, label, rows, tolerance):
    from alpha_cap_oracle import TOLERANCES
    a, e = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != e.shape or not torch.isfinite(a).all() or not torch.isfinite(e).all():
        raise AssertionError(label+': shape/nonfinite')
    atol, rtol = TOLERANCES[tolerance]
    err = (a-e).abs()
    row = dict(check=label, tolerance=tolerance, max_abs=float(err.max()),
               max_bound_ratio=float((err/(atol+rtol*e.abs())).max()))
    rows.append(row)
    if row['max_bound_ratio'] > 1: raise AssertionError(label+': numerical mismatch')


def invoke(torch, binding, renderer, sh, case, camera, config, *, points=None,
           diagnostic=False, binding_only=False, backward=True, diagnostic_pixel=None):
    """Real bindings/getters, all pixels rendered; only specified loss pixels have upstreams."""
    import alpha_cap_oracle as o
    points = case['points'] if points is None else points
    gpu = [{k: v.detach().float().cuda().requires_grad_() for k, v in p.items()} for p in points]
    bg = torch.tensor(o.SPEC['background'], dtype=torch.float32, device='cuda')
    from formal_time_handoff import bind_time
    model = None if case['dim'] != 4 else sh.make_model(
        gpu[0], (3, 2), o.SPEC, formal_time=config.time_derivation)
    if case['formal'] and not binding_only:
        from formal_views import ConsumerView
        actual = renderer.render(camera, model, ConsumerView(config, 'pipeline'), bg,
                                 scaling_modifier=1.0, override_color=None, formal_config=config)
        images = dict(rgb=actual['render'], flow=actual['flow'], depth=actual['depth'], mask=actual['alpha'])
        radii, screen, debug = actual['radii'], actual['viewspace_points'], actual['cuda_pixel_debug']
    else:
        px = diagnostic_pixel or case['pixels'][0]
        settings = binding.GaussianRasterizationSettings(
            image_height=9, image_width=9, tanfovx=camera.formal_camera.tanx,
            tanfovy=camera.formal_camera.tany, bg=bg, scale_modifier=1.0,
            viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
            sh_degree=3 if case['formal'] else 0, sh_degree_t=2 if case['formal'] else 0,
            campos=camera.camera_center, timestamp=.5, time_duration=1.,
            rot_4d=case['dim'] == 4, gaussian_dim=case['dim'], force_sh_3d=False,
            prefiltered=False, debug=False, formal_camera=case['dim'] == 4,
            camera_binding=camera.formal_binding if case['dim'] == 4 else None,
            formal_time=case['dim'] == 4,
            time_binding=bind_time(config, model, camera.formal_binding) if case['dim'] == 4 else None,
            debug_pixel_x=px[0] if diagnostic else -1, debug_pixel_y=px[1] if diagnostic else -1,
            debug_pixel_max_entries=len(points) if diagnostic else 0)
        stack = lambda name: torch.stack([p[name] for p in gpu])
        screen = torch.zeros((len(points), 3), dtype=torch.float32, device='cuda', requires_grad=True)
        scales = stack('log_scale').exp()
        opacity = stack('opacity').sigmoid() if case['formal'] else stack('opacity')
        flow = stack('flow')*0 if case['formal'] else stack('flow')
        values = binding.GaussianRasterizer(settings)(
            means3D=stack('mean'), means2D=screen, opacities=opacity,
            shs=stack('features'), colors_precomp=None, flow_2d=flow, ts=stack('time'),
            scales=scales[:, :3].contiguous(), scales_t=scales[:, 3:].contiguous(),
            rotations=torch.nn.functional.normalize(stack('left'), dim=1),
            rotations_r=torch.nn.functional.normalize(stack('right'), dim=1),
            cov3D_precomp=None, prefilter_var=-1.0)
        rgb, radii, depth, mask, flow_out, _, debug, _ = values
        images = dict(rgb=rgb, flow=flow_out, depth=depth, mask=mask)
    samples = [torch.cat([images[k][:, y, x] for k in ('rgb', 'flow', 'depth', 'mask')])
               for x, y in case['pixels']]
    result = dict(samples=[v.detach().cpu().double() for v in samples],
                  images={k: v.detach().cpu() for k, v in images.items()},
                  radii=radii.detach().cpu(), debug=debug.detach().cpu())
    if backward:
        loss = sum(v @ v.new_tensor(case['loss']) for v in samples)
        leaves = [v for p in gpu for v in p.values()]+[screen]
        gradients = list(torch.autograd.grad(loss, leaves, allow_unused=True))
        result['screen'] = (torch.zeros_like(screen) if gradients[-1] is None else gradients[-1]).detach().cpu().double()
        iterator = iter(gradients[:-1]); result['gradients'] = []
        for p in gpu:
            grad = {}
            for name, v in p.items():
                g = next(iterator)
                grad[name] = (torch.zeros_like(v) if g is None else g).detach().cpu().double()
            result['gradients'].append(grad)
    torch.cuda.synchronize()
    return result


def require_invariant(torch, a, b):
    for k in a['images']:
        if not torch.equal(a['images'][k], b['images'][k]): raise AssertionError('diagnostic_image_changed: '+k)
    if not torch.equal(a['radii'], b['radii']): raise AssertionError('diagnostic_radius_changed')
    if not torch.equal(a['screen'], b['screen']): raise AssertionError('diagnostic_screen_changed')
    for x, y in zip(a['gradients'], b['gradients']):
        for k in x:
            if not torch.equal(x[k], y[k]): raise AssertionError('diagnostic_gradient_changed: '+k)


def diagnostic_checks(torch, case, actual, expected, pixel, rows):
    import alpha_cap_oracle as o
    debug = actual['debug'].double().reshape(-1, 32)
    count = len(case['points'])
    if debug.shape != (count+1, 32) or not torch.isfinite(debug).all(): raise AssertionError('diagnostic_shape')
    h, entries = debug[0], debug[1:]
    if (h[0] != 1 or tuple(h[3:5].tolist()) != tuple(pixel) or h[12] != count
            or h[13] != count or h[14] != count or h[15] != count
            or h[21] != 0 or h[25] != count or h[26] != 0 or h[27] != 0):
        raise AssertionError('missing_truncated_or_skipped_contributor')
    if entries[:, 2].tolist() != list(range(count)) or entries[:, 18].count_nonzero():
        raise AssertionError('contributor_order_or_status')
    raw = torch.stack([g['raw'] for g in expected['geometry']])
    compare(torch, entries[:, 16], raw, case['label']+'/diagnostic_raw', rows, 'forward')
    compare(torch, entries[:, 17], expected['alpha'], case['label']+'/diagnostic_alpha', rows, 'forward')
    compare(torch, entries[:, 19], expected['trans'][:-1], case['label']+'/diagnostic_prefix', rows, 'forward')
    compare(torch, entries[:, 20], expected['trans'][1:], case['label']+'/diagnostic_T', rows, 'forward')
    if not torch.equal(entries[:, 20], entries[:, 21]): raise AssertionError('diagnostic_T_not_applied')
    if (entries[:, 16] >= o.CAP).tolist() != (raw >= o.CAP).tolist(): raise AssertionError('actual_cap_branch')
    bits = lambda v: struct.unpack('I', struct.pack('f', float(v)))[0]
    if case['boundary']:
        v = case['points'][0]['opacity'][0]
        if bits(entries[0, 16]) != bits(v) or bits(entries[0, 17]) != bits(min(float(v), o.CAP)):
            raise AssertionError('exact_boundary_bits')
    return dict(pixel=pixel, raw_alpha=entries[:, 16].tolist(), alpha=entries[:, 17].tolist(),
                raw_bits=[bits(v) for v in entries[:, 16]], status=entries[:, 18].tolist(),
                prefix=entries[:, 19].tolist(), final_T=float(h[16]))


def run_alpha_checks(torch, binding, renderer, sh, result):
    import alpha_cap_oracle as o
    from formal_config import resolve_formal_config
    from test_formal_config import fixture, encode
    cases = o.cases(); rows = result['checks']; saved = {}
    result['planned_cases'] = [c['label'] for c in cases]
    result['cases'] = []; result['fd_forward_invocations'] = 0
    with tempfile.TemporaryDirectory(prefix='alpha-config-') as temp:
        base = Path(temp); source = base/'data'; source.mkdir()
        value = fixture(source, base/'unused-output')
        value['dataset']['resolution']['divisor'] = 1
        value['dataset']['time'].update(raw_interval=[0, 1], divisor=1)
        value['initialization']['time_variance_denominator'] = 5
        config = resolve_formal_config(encode(value), raw_width=9, raw_height=9)
        camera = sh.make_camera(config, o.SPEC)
        for case in cases:
            result['active_case'] = case['label']
            domain = o.domain(case)
            expected = o.evaluate_case(case); gradient = o.parameter_gradients(case)
            actual = invoke(torch, binding, renderer, sh, case, camera, config)
            record = dict(case=case['label'], domain=domain, dim=case['dim'], formal=case['formal'],
                          loss=case['loss'], parameters=[{k: v.detach().tolist() for k, v in p.items()} for p in case['points']],
                          status='incomplete', diagnostic=[])
            result['cases'].append(record)
            for i, ref in enumerate(expected):
                compare(torch, actual['samples'][i], ref['outputs'], case['label']+f'/forward/{i}', rows, 'forward')
            compare(torch, actual['screen'], sum(e['screen'] for e in expected), case['label']+'/screen', rows, 'direct')
            for i, grad in enumerate(gradient):
                for k, v in grad.items():
                    compare(torch, actual['gradients'][i][k], v, case['label']+f'/gradient/{i}/{k}',
                            rows, 'formal' if case['formal'] else 'direct')
            radii = [n['radius'] for n in domain[0]['nodes']]
            if actual['radii'].tolist() != radii: raise AssertionError('actual_radius')
            # Same geometry through the binding supplies real diagnostic rows.
            # Formal render remains separately exercised (no success stub).
            normal = invoke(torch, binding, renderer, sh, case, camera, config, binding_only=True) if case['formal'] else actual
            if case['formal']:
                for k in actual['images']:
                    compare(torch, normal['images'][k], actual['images'][k], case['label']+'/formal_binding/'+k, rows, 'forward')
            for pixel, ref in zip(case['pixels'], expected):
                diagnosed = invoke(torch, binding, renderer, sh, case, camera, config,
                                   diagnostic=True, binding_only=True, diagnostic_pixel=pixel)
                require_invariant(torch, normal, diagnosed)
                record['diagnostic'].append(diagnostic_checks(torch, case, diagnosed, ref, pixel, rows))
            if case['boundary']:
                desired = 1. if float(case['points'][0]['opacity'][0]) < o.CAP else 0.
                if float(actual['gradients'][0]['opacity'][0]) != desired: raise AssertionError('boundary_subgradient')
            if len(case['points']) == len(case['pixels']) == 1 and float(expected[0]['geometry'][0]['raw']) >= o.CAP:
                if actual['gradients'][0]['opacity'].count_nonzero() or actual['screen'][0, :2].count_nonzero():
                    raise AssertionError('saturated_alpha_chain_not_exact_zero')
            for coordinate in o.fd_coordinates(case):
                point, name, offset = coordinate
                for h in o.CUDA_STEPS:
                    plus, minus, denominator = o.perturb(case['points'], coordinate, h, rounded=True)
                    endpoints = []
                    for points in (plus, minus):
                        checked = o.domain(case, points)
                        if [d['topology'] for d in checked] != [d['topology'] for d in domain]:
                            raise AssertionError('FD_topology_or_branch')
                        with torch.no_grad():
                            end = invoke(torch, binding, renderer, sh, case, camera, config, points=points,
                                         backward=False, diagnostic=True, binding_only=True)
                            # F finite differences must also use real model getters
                            # and formal render, not just equivalent binding inputs.
                            formal_end = (invoke(torch, binding, renderer, sh, case, camera, config,
                                                 points=points, backward=False)
                                          if case['formal'] else end)
                        if not torch.equal(end['radii'], actual['radii']): raise AssertionError('FD_radius_changed')
                        ref = o.evaluate_case(case, points)[0]
                        diagnostic_checks(torch, case, end, ref, case['pixels'][0], rows)
                        if case['formal']:
                            for k in end['images']:
                                compare(torch, formal_end['images'][k], end['images'][k],
                                        case['label']+'/FD_formal_binding/'+k, rows, 'forward')
                        endpoints.append(sum(v @ v.new_tensor(case['loss']) for v in formal_end['samples']))
                        result['fd_forward_invocations'] += 1
                        if case['formal']: result['fd_forward_invocations'] += 1
                    compare(torch, actual['gradients'][point][name].flatten()[offset],
                            (endpoints[0]-endpoints[1])/denominator,
                            case['label']+f'/cuda_fd/{coordinate}/{h}', rows, 'cuda_fd')
            # Exact boundary one-sided forward slopes, not central FD.
            if case['label'].endswith('/equal'):
                for h in o.CUDA_STEPS:
                    ends = []
                    for sign in (-1, 1):
                        points = o.clone(case['points'])
                        with torch.no_grad(): points[0]['opacity'][0] += sign*h
                        with torch.no_grad(): end = invoke(torch, binding, renderer, sh, case, camera, config, points=points, backward=False)
                        ends.append(end['samples'][0][-1])
                        result['fd_forward_invocations'] += 1
                    center = actual['samples'][0][-1]
                    if float((center-ends[0])/h) != 1. or float((ends[1]-center)/h) != 0.:
                        raise AssertionError('one_sided_boundary_slope')
            record['status'] = 'passed'; saved[case['label']] = actual
        # Per-pixel addition, not whole-Gaussian zeroing.
        for name, value in saved['P/sum']['gradients'][0].items():
            compare(torch, value, saved['P/p0']['gradients'][0][name]+saved['P/p1']['gradients'][0][name],
                    'P/additive/'+name, rows, 'direct')
        # Individual R/G/B, flow-x/y, depth and mask linearity for the same
        # representative geometry; unused pixels receive no loss.
        case = next(c for c in cases if c['label'] == 'S/4/0.997/mixed')
        individuals = [invoke(torch, binding, renderer, sh,
                             dict(case, loss=tuple(float(i == j) for i in range(7))), camera, config)
                       for j in range(7)]
        for k, v in saved[case['label']]['gradients'][0].items():
            compare(torch, v, sum(case['loss'][j]*a['gradients'][0][k] for j, a in enumerate(individuals)),
                    'S/loss_linearity/'+k, rows, 'direct')
        result['linearity_forward_backward_invocations'] = len(individuals)
        if (base/'unused-output').exists(): raise AssertionError('formal_config_output_created')
    if len(result['cases']) != 34 or any(c['status'] != 'passed' for c in result['cases']):
        raise AssertionError('missing_alpha_case')
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
    before = verify_source(expected)  # All input bytes before heavy/output/toolchain.
    if any(k.startswith('STEP90_CUDA_DEBUG') and v for k, v in os.environ.items()):
        raise RuntimeError('external_diagnostic_override')
    out = output_path(args.output_dir)
    sh = helper(); git_before = git_state(sh)
    out.mkdir(); build = out/'build'; build.mkdir()
    os.environ['TORCH_EXTENSIONS_DIR'] = str(build)
    sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
    result = dict(status='failed', stage='heavy_import', python=sys.executable,
                  source_before=before, reviewed_source=expected, git_before=git_before, checks=[])
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
        result['simple_knn_import_only'] = dict(path=knn.__file__, sha256=digest(knn.__file__))
        extension = binary_identity(binding, build, sh); result['extension'] = extension
        result['loaded_before'] = verify_loaded_inputs(expected, sh)
        result['stage'] = 'alpha_matrix'; run_alpha_checks(torch, binding, renderer, sh, result)
        result['stage'] = 'same_binary_SH_regression'
        regression = dict(source_before=sh_subset(expected, sh), checks=[])
        result['sh_regression'] = regression
        sh.run_checks(torch, binding, renderer, regression)
        if regression['case_count'] != 15 or len(regression['checks']) != 309:
            raise AssertionError('SH_case_or_comparison_count')
        torch.cuda.synchronize(); result['stage'] = 'preservation'
        if binary_identity(binding, build, sh) != extension: raise RuntimeError('binary_recipe_or_objects_changed')
        result['source_after'] = verify_source(expected)
        result['loaded_after'] = verify_loaded_inputs(expected, sh)
        result['git_after'] = git_state(sh)
        if result['git_after'] != git_before: raise RuntimeError('git_or_index_changed')
        result['stage'] = 'complete'; result['status'] = 'passed'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'[:4000]
        raise
    finally:
        # Preserve failure-stage evidence too; these are read-only checks,
        # never a second build or a retry of numerical execution.
        if result['status'] != 'passed':
            preservation = {}
            try:
                verify_source(expected)
                preservation['inputs_unchanged'] = True
            except Exception as error:
                preservation['inputs_error'] = f'{type(error).__name__}: {error}'[:1000]
            try:
                result['git_after'] = git_state(sh)
                preservation['git_unchanged'] = result['git_after'] == git_before
            except Exception as error:
                preservation['git_error'] = f'{type(error).__name__}: {error}'[:1000]
            if 'extension' in result:
                try:
                    preservation['binary_recipe_objects_unchanged'] = binary_identity(binding, build, sh) == result['extension']
                except Exception as error:
                    preservation['binary_error'] = f'{type(error).__name__}: {error}'[:1000]
            result['failure_preservation'] = preservation
        with (out/'result.json').open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
