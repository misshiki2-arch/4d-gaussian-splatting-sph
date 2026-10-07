"""Step 11 future, separately authorized two-extension fresh-build acceptance.

Import/record/preflight are stdlib only. Recording is a candidate, NOT review.
Never called by CPU tests beyond the no-output preflight. No install, retry,
cleanup, training, source write, or reuse of an installed simple-knn binary.
"""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT.parent/'reports/corrected-4dgs/phase1/step11'
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
SCHEMA = 'step11-time-inputs-v1'
HELPER = 'tests/cuda/camera_projection_cuda_acceptance.py'
EXTRA = ('tests/time_conditioning_oracle.py', 'tests/time_conditioning_runtime.py',
         'tests/test_time_conditioning_cpu.py', 'tests/test_time_conditioning_source_identity.py',
         'tests/cuda/time_conditioning_cuda_acceptance.py')
KNN = ('simple-knn/setup.py', 'simple-knn/spatial.cu', 'simple-knn/simple_knn.cu',
       'simple-knn/ext.cpp', 'simple-knn/spatial.h', 'simple-knn/simple_knn.h')
HEAVY = ('torch', 'numpy', 'PIL', 'scene', 'train', 'gaussian_renderer', 'simple_knn', 'pointops2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def helper():
    spec = importlib.util.spec_from_file_location('step11_camera_helper', ROOT/HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if module.ROOT != ROOT: raise RuntimeError('helper_repository_mismatch')
    return module


def source_identity():
    value = helper().source_identity(); value['schema'] = SCHEMA
    for name in EXTRA+KNN+('train.py',):
        path = ROOT/name
        if path.resolve() != path or not path.is_file(): raise RuntimeError('time_input_missing_or_aliased: '+name)
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
    if (type(value) is not dict or set(value) != {'schema', 'repository', 'base_head', 'files'} or
            value['schema'] != SCHEMA or value['repository'] != str(ROOT) or
            not isinstance(value['base_head'], str) or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', value['base_head']) or
            type(value['files']) is not dict or not set(EXTRA+KNN+(HELPER,)) <= value['files'].keys()):
        raise RuntimeError('time_evidence_schema')
    for name, sha in value['files'].items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts or
                Path(name).as_posix() != name or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha)):
            raise RuntimeError('time_evidence_file')
    return value


def verify_source(expected):
    # Check every byte before executing the helper hierarchy.
    for name, sha in expected['files'].items():
        path = ROOT/name
        if path.resolve() != path or not path.is_file() or digest(path) != sha:
            raise RuntimeError('reviewed_input_mismatch: '+name)
    actual = source_identity()
    if any(actual[k] != expected[k] for k in ('schema', 'repository', 'files')):
        raise RuntimeError('source_inputs_mismatch')
    return actual


def subset(expected, camera):
    return dict(schema=camera.SCHEMA, repository=str(ROOT), base_head=expected['base_head'],
                files={k: expected['files'][k] for k in camera.source_identity()['files']})


def preflight(expected, out):
    actual = verify_source(expected)
    if any(k.startswith('STEP90_CUDA_DEBUG') and v for k, v in os.environ.items()):
        raise RuntimeError('external_diagnostic_override')
    if not out.is_absolute() or out.exists() or out.is_symlink() or out.resolve() != out:
        raise RuntimeError('new_unaliased_absolute_output_required')
    if REPORTS not in out.parents or not out.parent.is_dir(): raise RuntimeError('new_step11_run_directory_required')
    return actual, out


def build_knn(out):
    """Existing recipe, three existing units, isolated lib/temp; no installation."""
    lib, temp = out/'knn-lib', out/'knn-build'
    command = [sys.executable, '-B', 'setup.py', 'build_ext', '--build-lib', str(lib), '--build-temp', str(temp)]
    with (out/'knn-build.log').open('x') as log:
        subprocess.run(command, cwd=ROOT/'simple-knn', stdout=log, stderr=subprocess.STDOUT, check=True)
    binaries = list((lib/'simple_knn').glob('_C*.so'))
    if len(binaries) != 1: raise RuntimeError('fresh_knn_binary_count')
    binary = binaries[0].resolve()
    if lib not in binary.parents or 'simple_knn._C' in sys.modules: raise RuntimeError('fresh_knn_identity')
    recipes = list(temp.rglob('build.ninja')); objects = list(temp.rglob('*.o'))
    if len(recipes) != 1 or len(objects) != 3: raise RuntimeError('fresh_knn_build_evidence')
    recipe = recipes[0].read_text()
    if not all(str(ROOT/n) in recipe for n in KNN[1:4]): raise RuntimeError('knn_translation_units')
    package = types.ModuleType('simple_knn'); package.__path__ = [str(lib/'simple_knn')]
    sys.modules['simple_knn'] = package
    spec = importlib.util.spec_from_file_location('simple_knn._C', binary)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sys.modules['simple_knn._C'] = module; package._C = module
    if Path(module.__file__).resolve() != binary: raise RuntimeError('knn_load_identity')
    return dict(path=str(binary), sha256=digest(binary), command=command,
                recipe=dict(path=str(recipes[0]), sha256=digest(recipes[0])),
                objects={str(p): digest(p) for p in objects}, sources=list(KNN))


def compare(torch, actual, expected, label, rows, tolerance):
    from time_conditioning_oracle import TOLERANCES
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != b.shape or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        raise AssertionError(label+': shape/nonfinite')
    atol, rtol = TOLERANCES[tolerance]; error = (a-b).abs(); ratio = error/(atol+rtol*b.abs())
    rows.append(dict(check=label, tolerance=tolerance, max_abs=float(error.max()), max_bound_ratio=float(ratio.max())))
    if float(ratio.max()) > 1: raise AssertionError(label+': numerical_mismatch')


def initialization_pixel_evidence(torch, actual, expected, record):
    """Persist components BEFORE any assertion; mapping is in the same record."""
    from time_conditioning_oracle import TOLERANCES
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    pixel = dict(actual=a.tolist(), expected=b.tolist(), status='incomplete', components=[])
    record['pixel_comparison'] = pixel
    if a.shape != (7,) or b.shape != (7,) or not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
        pixel['reason'] = 'shape_or_nonfinite'
        return pixel
    atol, rtol = TOLERANCES['pixel']
    for i, name in enumerate(('red','green','blue','flow_x','flow_y','depth','alpha')):
        error = float(a[i]-b[i]); bound = atol+rtol*abs(float(b[i]))
        pixel['components'].append(dict(name=name, actual=float(a[i]), expected=float(b[i]),
            signed_error=error, abs_error=abs(error), bound=bound, bound_ratio=abs(error)/bound))
    worst = max(pixel['components'], key=lambda v: v['bound_ratio'])
    pixel.update(status='passed' if worst['bound_ratio'] <= 1 else 'failed',
                 worst_component=worst['name'], max_bound_ratio=worst['bound_ratio'])
    return pixel


def initialization_diagnostics(torch, buffer, radii, domain, record, rows):
    """Step 11 mapping adapter; shared camera all-point diagnostics unchanged."""
    import time_conditioning_oracle as o
    n, order = len(domain['nodes']), domain['order']; count = len(order)
    data = buffer.detach().cpu().double()
    diagnostic = dict(status='checking', shape=list(data.shape), radii=radii.detach().cpu().tolist())
    record['contributor_diagnostic'] = diagnostic
    o.require(diagnostic, data.numel() == (n+1)*32, 'initial_diagnostic_shape')
    data = data.reshape(n+1,32)
    # Bounded by fixture row count; preserve malformed/truncated records too.
    diagnostic.update(header=data[0].tolist(), records=data[1:].tolist(), capacity=n,
                      predicted_order=order, culled_ids=[v['gaussian_id'] for v in domain['nodes'] if not v['time_active']])
    o.require(diagnostic, bool(torch.isfinite(data).all()), 'initial_diagnostic_nonfinite')
    h = data[0]; x,y = domain['pixel']; tx,ty = domain['tile']; w,height = domain['image']
    required = {0:1,1:w,2:height,3:x,4:y,5:x,6:y,7:tx,8:ty,9:ty*((w+15)//16)+tx,
                12:count,13:count,14:count,15:count,21:0,22:-1,23:1,24:0,25:count,26:0,27:0,31:0}
    o.require(diagnostic, all(float(h[k]) == value for k,value in required.items()) and
              float(h[10]) >= 0 and float(h[10]).is_integer() and h[11]-h[10] == count and count <= n,
              'initial_diagnostic_coverage')
    o.require(diagnostic, diagnostic['radii'] == [v['radius'] for v in domain['nodes']],
              'initial_diagnostic_radii')
    for j, i in enumerate(order):
        observed, expected = data[j+1], domain['nodes'][i]
        o.require(diagnostic, observed[0] == j and observed[1] == h[10]+j and observed[2] == i and
                  observed[18] == 0 and observed[28] == 1 and observed[29] == j+1,
                  'initial_diagnostic_order_or_reason')
        o.require(diagnostic, float(observed[3]) == expected['depth_f32'] and
                  o.camera.bits(float(observed[3])) == expected['depth_key'], 'initial_diagnostic_depth_key')
        values = torch.tensor([expected['power'], expected['raw_alpha'], expected['raw_alpha'],
                               expected['trans_before'], expected['trans_after'], expected['trans_after']], dtype=torch.float64)
        compare(torch, observed[[15,16,17,19,20,21]], values, record['label']+f'/contributor-{i}', rows, 'pixel')
    compare(torch, h[16], torch.tensor(domain['final_transmittance'],dtype=torch.float64),
            record['label']+'/final_T', rows, 'pixel')
    diagnostic['status'] = 'passed'


def initialization_forward(torch, binding, render, case, record, rows):
    """One real normal call, then replay identical public inputs for diagnostics.

    No native replacement, fixture alteration, inferred GPU order, or backward
    substitution. The returned normal graph is used for the real model VJP.
    """
    import time_conditioning_oracle as o
    original = binding.rasterize_gaussians; calls = []
    def observe(*args, **kwargs):
        calls.append(inspect.signature(original).bind(*args, **kwargs).arguments)
        return original(*args, **kwargs)
    with patch.object(binding, 'rasterize_gaussians', side_effect=observe): actual = render()
    o.require(record, len(calls) == 1, 'initial_normal_call_count')
    x,y = case['pixels'][0]
    sample = torch.cat([actual[k][:,y,x] for k in ('render','flow','depth','alpha')])
    pixel = initialization_pixel_evidence(torch, sample, o.evaluate(case)[0]['outputs'], record)
    args = dict(calls[0]); settings = args['raster_settings']
    args['raster_settings'] = settings._replace(debug_pixel_x=x, debug_pixel_y=y,
                                               debug_pixel_max_entries=len(case['points']))
    with torch.no_grad(): diagnostic = original(**args)
    record['noninterference'] = {name:torch.equal(actual[name], diagnostic[index])
        for name,index in (('render',0),('radii',1),('depth',2),('alpha',3),('flow',4))}
    # Store the buffer even if noninterference or the pixel comparison fails.
    initialization_diagnostics(torch, diagnostic[6], actual['radii'], record['domain'], record, rows)
    o.require(record, all(record['noninterference'].values()), 'initial_diagnostic_interference')
    o.require(pixel, pixel['status'] == 'passed', 'initial_pixel_mismatch')
    compare(torch, sample, o.evaluate(case)[0]['outputs'], record['label']+'/pixel', rows, 'pixel')
    return sample


def time_diagnostics(torch, case, actual, target, rows):
    import time_conditioning_oracle as o
    p = case['points'][target]; s = case['camera']
    mean, cov, m, sigma = o.sh.conditional(p, s)
    pre = actual['preprocess'].double()
    if pre.shape != (104,) or pre[103] != 1: raise AssertionError('time_forward_diagnostic_missing')
    compare(torch, pre[96], sigma[3, 3], case['label']+'/c', rows, 'geometry')
    compare(torch, pre[97], m, case['label']+'/m', rows, 'marginal')
    if (float(pre[98]) != -1. or float(pre[99]) != o.camera.f32(s['duration']) or
            float(pre[100]) != o.camera.f32(s['timestamp']) or bool(pre[102]) != (float(pre[97]) > .05)):
        raise AssertionError('time_diagnostic_handoff')
    if pre[102]:
        packed = torch.stack((cov[0,0], cov[0,1], cov[0,2], cov[1,1], cov[1,2], cov[2,2]))
        compare(torch, pre[5:8], mean, case['label']+'/mean', rows, 'geometry')
        compare(torch, pre[26:32], packed, case['label']+'/cov', rows, 'geometry')
    if 'time_backward' in actual:
        bw = actual['time_backward'].double()
        if actual['radii'][target] == 0:
            if not bool((bw == -1).all()): raise AssertionError('culled_backward_entered')
            if any(v.count_nonzero() for v in actual['gradients'][target].values()):
                raise AssertionError('culled_gradient_nonzero')
        else:
            if bw[7] != 1 or bw[6] != pre[102] or not torch.equal(bw[2:6], pre[98:102]):
                raise AssertionError('time_backward_handoff')
            compare(torch, bw[:2], pre[96:98], case['label']+'/backward_cm', rows, 'marginal')
    return (actual['radii'].tolist(), pre[77:82].tolist(), pre[102].item())


def run_time_checks(torch, binding, renderer, camera_entry, sh, out, result):
    import time_conditioning_oracle as o
    from camera_projection_runtime import make_camera
    rows = result['checks']; result['time_cases'] = []
    for index, case in enumerate(o.cases()):
        fixture_dir = out/f'time-fixture-{index}'; fixture_dir.mkdir()
        camera, config = make_camera(fixture_dir, case['camera']); camera = camera.cuda()
        started = time.monotonic()
        normal = camera_entry.invoke(torch, binding, renderer, sh, case, camera, config)
        call_seconds = time.monotonic()-started
        target = 1 if case['kind'] == 'time_cull' else 0
        diagnostic = camera_entry.invoke(torch, binding, renderer, sh, case, camera, config,
                                        diagnostic=True, binding_only=True, target=target)
        camera_entry.invariant(torch, normal, diagnostic, case, rows)
        topo = time_diagnostics(torch, case, diagnostic, target, rows)
        expected = o.evaluate(case); gradients = o.gradients(case)
        for j, (a, b) in enumerate(zip(normal['samples'], expected)):
            compare(torch, a, b['outputs'], case['label']+f'/pixel{j}', rows, 'pixel')
        for j, (a, b) in enumerate(zip(normal['gradients'], gradients)):
            for name in b: compare(torch, a[name], b[name], case['label']+f'/{j}/{name}', rows, 'gradient')
        if case['kind'] == 'smooth':
            reference_domain = o.domain(case)
            for coordinate in o.coordinates(case):
                point, name, offset = coordinate
                for step in o.CUDA_STEPS:
                    plus, minus, span = o.camera.perturb(case['points'], coordinate, step, rounded=True)
                    if o.domain(case, plus) != reference_domain or o.domain(case, minus) != reference_domain:
                        raise AssertionError('cuda_fd_cpu_endpoint_branch')
                    values = []
                    for endpoint in (plus, minus):
                        v = camera_entry.invoke(torch, binding, renderer, sh, case, camera, config,
                            points=endpoint, backward=False, diagnostic=True, binding_only=True)
                        ep = v['preprocess']
                        if (v['radii'].tolist(), ep[77:82].tolist(), ep[102].item()) != topo:
                            raise AssertionError('cuda_fd_actual_endpoint_branch')
                        values.append(sum(x @ x.new_tensor(case['loss']) for x in v['samples']))
                    fd = (values[0]-values[1])/span
                    compare(torch, normal['gradients'][point][name].flatten()[offset], fd,
                            case['label']+f'/CUDA_FD/{coordinate}/{step}', rows, 'cuda_fd')
        result['time_cases'].append(dict(label=case['label'], topology=topo, normal_call_seconds=call_seconds))


def run_initialization(torch, binding, renderer, out, result):
    import numpy as np
    import time_conditioning_runtime as runtime
    import time_conditioning_oracle as o
    from formal_views import ConsumerView
    result['initialization'] = []
    for divisor in (1., 2., 2.5):
        for sampled in (False, True):
            base = out/f'initial-{divisor}-{sampled}'; base.mkdir()
            inputs, model, scene, data, indices = runtime.initialize(base, divisor, sampled)
            tag = base.name; rows = result['checks']; n = len(indices)
            record = dict(label=tag, status='checking', divisor=divisor, sampled=sampled, rows=n,
                          seed=inputs.config.initialization.seed, row_indices=indices.tolist())
            result['initialization'].append(record)
            xyz = torch.tensor(np.stack([data[k][indices] for k in ('x','y','z')], 1), dtype=torch.float64)
            times = torch.tensor((data['time'][indices].astype('float64')/divisor).astype('float32'), dtype=torch.float64)
            knn = o.knn_squared(xyz)
            compare(torch, model.get_xyz, xyz, tag+'/xyz', rows, 'geometry')
            if not torch.equal(model.get_t.detach().cpu().flatten().double(), times): raise AssertionError('initial_time_bits')
            stored_log = o.camera.f32(inputs.config.time_derivation.log_effective_scale)
            if not bool((model._scaling_t.detach().cpu() == stored_log).all()): raise AssertionError('initial_log_bits')
            compare(torch, model.get_scaling_t, torch.full((n,1), math.sqrt(8/5)/divisor,dtype=torch.float64), tag+'/scale', rows, 'scale')
            compare(torch, model.get_scaling_t.square(), torch.full((n,1), 8/(5*divisor**2),dtype=torch.float64), tag+'/variance', rows, 'variance')
            compare(torch, model.get_scaling.square()[:, 0], knn, tag+'/KNN', rows, 'variance')
            identity = torch.tensor([1.,0,0,0]).expand(n,4)
            if not torch.equal(model.get_rotation.cpu(), identity) or not torch.equal(model.get_rotation_r.cpu(), identity):
                raise AssertionError('initial_rotations')
            if (model.active_sh_degree, model.active_sh_degree_t, model.prefilter_var) != (0,0,-1.):
                raise AssertionError('initial_SH_PF')
            camera = scene.getTrainCameras()[0][1].cuda()
            case = o.initialization_case(data, indices, inputs.config.time_derivation, camera.timestamp, tag)
            record['domain'] = dict(seed=record['seed'], row_indices=record['row_indices'])
            o.initialization_domain(case, record['domain'])
            parameters = {name:getattr(model,name) for name in ('_xyz','_t','_scaling','_scaling_t',
                           '_rotation','_rotation_r','_opacity','_features_dc','_features_rest')}
            before = {k:(v, v.stride(), v.detach().clone(), v._version) for k,v in parameters.items()}
            sample = initialization_forward(torch, binding,
                lambda: renderer.render(camera, model, ConsumerView(inputs.config,'pipeline'),
                    torch.tensor(case['camera']['background'], device='cuda'), formal_config=inputs.config),
                case, record, rows)
            (sample @ sample.new_tensor(case['loss'])).backward()
            expected = o.gradients(case)
            actual_grad = dict(mean=model._xyz.grad, time=model._t.grad,
                log_scale=torch.cat((model._scaling.grad,model._scaling_t.grad),1),
                left=model._rotation.grad, right=model._rotation_r.grad, opacity=model._opacity.grad,
                features=torch.cat((model._features_dc.grad,model._features_rest.grad),1))
            for name in actual_grad:
                compare(torch, actual_grad[name], torch.stack([x[name] for x in expected]), tag+'/'+name, rows, 'gradient')
            record['parameter_preservation'] = all(getattr(model,k) is p and p.stride() == stride and
                p._version == version and torch.equal(p, value) for k,(p,stride,value,version) in before.items())
            o.require(record, record['parameter_preservation'], 'initial_parameter_changed')
            record.update(status='passed', duration=case['camera']['duration'], target='actual_Scene_model_KNN_render_backward')


def run_boundary(torch, binding, renderer, camera_entry, sh, out, result):
    import time_conditioning_oracle as o
    from camera_projection_runtime import make_camera
    threshold = o.threshold(); found = {}; observations = []
    p = o.sh.clone(o.cases()[0]['points'][0])
    with torch.no_grad(): p['time'].zero_(); p['opacity'].zero_()
    base = out/'boundary-fixture'; base.mkdir()
    # All inputs are fixed before seeing a GPU result. Search is bounded even
    # when no marker is reachable. Each camera has an actual FrameTime binding.
    for index, (logscale, stamp) in enumerate(o.boundary_grid()):
        q = o.sh.clone(p)
        with torch.no_grad(): q['log_scale'][3] = logscale
        s = dict(o.cases()[0]['camera'], timestamp=stamp)
        case = dict(o.cases()[0], label=f'B/{index}', points=[q], camera=s)
        camera, config = make_camera(base, s); camera = camera.cuda()
        actual = camera_entry.invoke(torch,binding,renderer,sh,case,camera,config,
                                     diagnostic=True,binding_only=True,backward=False)
        pre = actual['preprocess']; marginal = float(pre[97]); mask = bool(pre[102])
        if pre[103] != 1 or mask != (marginal > .05): raise AssertionError('boundary_predicate')
        key = 'marker' if marginal == threshold['marker'] else ('upper' if mask else 'lower')
        observations.append(dict(index=index, logscale=logscale, timestamp=stamp, c=float(pre[96]),
                                 dt=float(pre[101]), marginal=marginal, bits=o.camera.bits(marginal), mask=mask))
        if key not in found:
            tested = camera_entry.invoke(torch,binding,renderer,sh,case,camera,config,
                                         diagnostic=True,binding_only=True)
            bw = tested['time_backward']
            if mask:
                if bw[7] != 1 or bw[6] != 1: raise AssertionError('boundary_backward_missing')
                if not any(tested['gradients'][0][k].count_nonzero() for k in ('time','log_scale')):
                    raise AssertionError('boundary_time_gradient_absent')
            else:
                if tested['radii'][0] != 0 or not bool((bw == -1).all()): raise AssertionError('boundary_cull_gate')
                if any(g.count_nonzero() for g in tested['gradients'][0].values()): raise AssertionError('boundary_cull_gradient')
                for value in tested['samples']:
                    if not torch.equal(value[:3].float(), torch.tensor(s['background'],dtype=torch.float32)):
                        raise AssertionError('boundary_cull_contribution')
            found[key] = observations[-1]
        if set(found) == {'lower','upper','marker'}: break
    result['boundary'] = dict(threshold=threshold, maximum_candidates=4096, attempted=len(observations), selected=found)
    with (out/'boundary-search.json').open('x') as stream: json.dump(observations, stream, indent=2)
    if set(found) != {'lower','upper','marker'}: raise AssertionError('bounded_boundary_coverage_unmet')


def run_health(torch, binding, renderer, camera_entry, sh, out, result):
    import time_conditioning_oracle as o
    from camera_projection_runtime import make_camera
    case = o.cases()[0]; base = out/'health-fixture'; base.mkdir()
    camera, config = make_camera(base, case['camera']); camera = camera.cuda()
    result['health'] = []
    for name, field, index, value in (('scale_zero','log_scale',3,-1000.),
            ('scale_inf','log_scale',3,1000.), ('variance_zero','log_scale',3,-60.),
            ('center_nan','time',0,float('nan')), ('rotation_nan','left',0,float('nan'))):
        p = o.sh.clone(case['points'][0])
        with torch.no_grad(): p[field].flatten()[index] = value
        for diagnostic in (False, True):
            try:
                camera_entry.invoke(torch,binding,renderer,sh,dict(case,points=[p]),camera,config,
                                    diagnostic=diagnostic,binding_only=diagnostic,backward=False)
            except RuntimeError as error:
                if 'formal_time_forward_health:' not in str(error): raise
            else: raise AssertionError('invalid_time_accepted: '+name)
            result['health'].append(dict(case=name, diagnostic=diagnostic, rejected=True))
    # Healthy positive c with marginal underflow is normal cull, not an error.
    p = o.sh.clone(case['points'][0])
    with torch.no_grad(): p['time'].fill_(1e10)
    actual = camera_entry.invoke(torch,binding,renderer,sh,dict(case,points=[p]),camera,config,
                                diagnostic=True,binding_only=True)
    if actual['preprocess'][97] != 0 or actual['radii'][0] != 0: raise AssertionError('healthy_zero_marginal')
    if not bool((actual['time_backward'] == -1).all()): raise AssertionError('zero_marginal_backward')
    result['health'].append(dict(case='healthy_zero_marginal', normal_cull=True))
    try:
        camera_entry.invoke(torch,binding,renderer,sh,
            dict(case,loss=(float('inf'),0.,0.,0.,0.,0.,0.)),camera,config)
    except RuntimeError as error:
        if 'formal_time_backward_health:' not in str(error): raise
    else: raise AssertionError('nonfinite_time_VJP_accepted')
    result['health'].append(dict(case='nonfinite_time_VJP',rejected=True))
    # Different actual binary32 inputs: either preserved on this build, or a
    # detected FTZ loss. Never infer a GPU flush from host representation alone.
    for name, stamp, center, interval in (
            ('dt_overflow',1e38,-3e38,[-1e38,1e38]),
            ('subnormal_dt',2**-149,0.,[-2,6])):
        s = dict(case['camera'],timestamp=o.camera.f32(stamp),raw_interval=interval,
                 time_divisor=1.,duration=interval[1]-interval[0])
        cam,cfg = make_camera(base,s); cam = cam.cuda()
        p = o.sh.clone(case['points'][0])
        with torch.no_grad(): p['time'].fill_(center)
        try:
            v = camera_entry.invoke(torch,binding,renderer,sh,dict(case,camera=s,points=[p]),cam,cfg,
                                    diagnostic=True,binding_only=True,backward=False)
        except RuntimeError as error:
            if 'formal_time_forward_health:1' not in str(error): raise
            result['health'].append(dict(case=name,rejected=True))
        else:
            if name == 'dt_overflow' or float(v['preprocess'][101]) == 0:
                raise AssertionError('lost_temporal_difference_accepted')
            result['health'].append(dict(case=name,actual_difference_preserved=True))

    # Mutate only the native call's metadata in this negative-test seam.
    # The actual TORCH_CHECK must reject BEFORE a kernel sees invalid storage.
    original = binding._C.rasterize_gaussians
    mutations = (
        ('shape',5,lambda t: t.repeat(1,2)), ('rows',5,lambda t: t[:0]),
        ('dtype',5,lambda t: t.double()), ('device',5,lambda t: t.cpu()),
        ('PF',12,lambda t: 0.), ('timestamp',23,lambda t: float('nan')),
        ('duration',24,lambda t: 0.))
    for name,index,transform in mutations:
        def malformed(*args):
            values = list(args); values[index] = transform(values[index]); return original(*values)
        with patch.object(binding._C,'rasterize_gaussians',side_effect=malformed):
            try: camera_entry.invoke(torch,binding,renderer,sh,case,camera,config,backward=False)
            except RuntimeError as error:
                if 'formal_time_native_' not in str(error): raise
            else: raise AssertionError('native_metadata_accepted: '+name)
        result['health'].append(dict(case='native_'+name,rejected_before_kernel=True))


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
    expected = load_source_evidence(args.source_evidence); before,out = preflight(expected,args.output_dir)
    camera = helper(); alpha = camera.helper(); sh = alpha.helper()
    git_before = alpha.git_state(sh)
    out.mkdir(); build = out/'rasterizer-build'; build.mkdir()
    os.environ['TORCH_EXTENSIONS_DIR'] = str(build)
    sys.path[:0] = [str(ROOT),str(ROOT/'tests')]
    result = dict(status='failed',stage='heavy_import',python=sys.executable,source_before=before,
                  git_before=git_before,checks=[],cpu_success_is_not_CUDA_acceptance=True)
    started = time.monotonic()
    pointops_guard = ExitStack()
    try:
        import torch
        result['environment'] = dict(python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), nvcc=sh.bounded_command(['nvcc','--version']),
            compiler=sh.bounded_command(['c++','--version']),
            driver=sh.bounded_command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']))
        if not torch.cuda.is_available(): raise RuntimeError('CUDA_required_no_fallback')
        result['environment']['gpu'] = torch.cuda.get_device_name(0)
        result['stage'] = 'fresh_simple_knn'; verify_source(expected); result['knn'] = build_knn(out)
        result['stage'] = 'fresh_rasterizer'; verify_source(expected)
        with (out/'rasterizer-build.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
            import gaussian_renderer as renderer
            from gaussian_renderer import diff_gaussian_rasterization as binding
        result['extension'] = alpha.binary_identity(binding,build,sh)
        import pointops2_cuda
        result['pointops_import_only'] = dict(path=pointops2_cuda.__file__,sha256=digest(Path(pointops2_cuda.__file__)))
        def forbidden_pointops(*args,**kwargs): raise AssertionError('pointops_kernel_out_of_scope')
        guarded = []
        for name,value in vars(pointops2_cuda).copy().items():
            if not name.startswith('_') and callable(value):
                pointops_guard.enter_context(patch.object(pointops2_cuda,name,side_effect=forbidden_pointops))
                guarded.append(name)
        result['pointops_import_only']['blocked_native_calls'] = guarded
        if not guarded: raise RuntimeError('pointops_kernel_guard_missing')
        from scene import gaussian_model
        import simple_knn._C as knn
        if (gaussian_model.distCUDA2 is not knn.distCUDA2 or knn.distCUDA2.__module__ != 'simple_knn._C'
                or Path(knn.__file__).resolve() != Path(result['knn']['path'])):
            raise RuntimeError('actual_model_KNN_binding_mismatch')
        result['stage'] = 'actual_initialization'; run_initialization(torch,binding,renderer,out,result)
        result['stage'] = 'time_matrix'; run_time_checks(torch,binding,renderer,camera,sh,out,result)
        result['stage'] = 'health'; run_health(torch,binding,renderer,camera,sh,out,result)
        result['stage'] = 'bounded_boundary'; run_boundary(torch,binding,renderer,camera,sh,out,result)
        result['stage'] = 'same_binary_controls'
        cr = dict(source_before=subset(expected,camera),checks=[]); result['camera_regression'] = cr
        camera.run_camera_checks(torch,binding,renderer,sh,cr)
        ar = dict(source_before=camera.alpha_subset(cr['source_before'],alpha),checks=[]); result['alpha_regression'] = ar
        alpha.run_alpha_checks(torch,binding,renderer,sh,ar)
        sr = dict(source_before=alpha.sh_subset(ar['source_before'],sh),checks=[]); result['sh_regression'] = sr
        sh.run_checks(torch,binding,renderer,sr)
        if cr['case_count'] != 54 or ar['case_count'] != 34 or sr['case_count'] != 15:
            raise AssertionError('same_binary_control_coverage')
        if alpha.binary_identity(binding,build,sh) != result['extension']: raise RuntimeError('rasterizer_binary_changed')
        if digest(Path(result['knn']['path'])) != result['knn']['sha256']: raise RuntimeError('KNN_binary_changed')
        result['loaded_inputs'] = camera.verify_loaded_inputs(cr['source_before'],alpha,sh)
        for name in ('time_conditioning_runtime','time_conditioning_oracle'):
            if Path(sys.modules[name].__file__).resolve() != ROOT/'tests'/(name+'.py'):
                raise RuntimeError('unexpected_time_input')
        result['source_after'] = verify_source(expected)
        if alpha.git_state(sh) != git_before: raise RuntimeError('git_index_changed')
        result['status'],result['stage'] = 'passed','complete'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'[:4000]
        raise
    finally:
        pointops_guard.close()
        result['elapsed_seconds'] = time.monotonic()-started
        result['preservation'] = {}
        for name, check in (('source',lambda: verify_source(expected)['files'] == expected['files']),
                            ('git_index',lambda: alpha.git_state(sh) == git_before),
                            ('rasterizer',lambda: 'extension' not in result or alpha.binary_identity(binding,build,sh) == result['extension']),
                            ('KNN',lambda: 'knn' not in result or
                             (digest(Path(result['knn']['path'])) == result['knn']['sha256'] and
                              digest(Path(result['knn']['recipe']['path'])) == result['knn']['recipe']['sha256'] and
                              all(digest(Path(p)) == h for p,h in result['knn']['objects'].items())))):
            try: result['preservation'][name] = check()
            except Exception as error: result['preservation'][name] = str(error)[:1000]
        failed_preservation = any(value is not True for value in result['preservation'].values())
        was_passed = result['status'] == 'passed'
        if failed_preservation:
            result['status'] = 'failed'
            result['preservation_error'] = 'input_or_binary_or_index_changed'
        with (out/'result.json').open('x') as stream: json.dump(result,stream,ensure_ascii=False,indent=2)
        if was_passed and failed_preservation: raise RuntimeError('acceptance_preservation_failed')


if __name__ == '__main__': main()
