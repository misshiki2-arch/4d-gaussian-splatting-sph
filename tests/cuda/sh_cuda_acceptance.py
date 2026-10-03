"""Explicit Step 8 live acceptance entry; importing it never starts CUDA.

Requires separately approved build/CUDA execution, a fresh process, -B,
4dgs310, and independently reviewed actual-input evidence (not HEAD alone).
--record-source-evidence prints a stdlib-only candidate record and exits;
recording is NOT approval. Execution requires that separately reviewed file.
Reviewed uncommitted/new inputs are allowed. No Git writes, installs or training.
--output-dir is a NEW directory outside the repository: result.json,
build.log and build/ (fresh TORCH_EXTENSIONS_DIR). Existing output is refused.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
INPUT_SCHEMA = 'step8-sh-inputs-v1'
NATIVE = 'diff-gaussian-rasterization'
TRANSLATION_UNITS = tuple(NATIVE+'/'+p for p in (
    'cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
    'cuda_rasterizer/backward.cu', 'rasterize_points.cu', 'ext.cpp'))
PYTHON_PACKAGES = ('gaussian_renderer', 'scene', 'utils', 'arguments')
EXPLICIT_INPUTS = (
    'tests/cuda/sh_cuda_acceptance.py', 'tests/sh_oracle.py',
    'tests/test_sh_cpu.py', 'tests/test_formal_config.py',
    'pointops2/__init__.py', 'pointops2/functions/__init__.py',
    'pointops2/functions/pointops.py',
    'gaussian_renderer/diff_gaussian_rasterization.py',
    'gaussian_renderer/__init__.py', 'scene/gaussian_model.py',
    'scene/cameras.py', 'formal_config.py', 'formal_camera.py',
    NATIVE+'/rasterize_points.h', NATIVE+'/third_party/glm/glm/glm.hpp')


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    # Bounded Step-8 inputs, from the actual loader/import paths, NOT Git's
    # tracking set. Docs, datasets, unrelated tools/tests and build caches are
    # not inputs. Package-level inclusion catches newly added local modules.
    paths = {ROOT/p for p in EXPLICIT_INPUTS + TRANSLATION_UNITS}
    paths.update(ROOT.glob('formal*.py'))
    for name in PYTHON_PACKAGES:
        paths.update((ROOT/name).rglob('*.py'))
    for folder in (ROOT/NATIVE, ROOT/NATIVE/'cuda_rasterizer'):
        paths.update(p for p in folder.iterdir() if p.suffix in {'.cu','.cpp','.h','.cuh'})
    glm = ROOT/NATIVE/'third_party/glm/glm'
    paths.update(p for p in glm.rglob('*') if p.suffix in {'.h','.hpp','.inl'})
    files = {}
    for path in sorted(paths):
        # No outside-tree or aliased input; do not silently follow a symlink
        # to a different source tree. Missing required files fail here.
        if path.resolve() != path or not path.is_file():
            raise RuntimeError('source_input_missing_or_aliased: '+str(path.relative_to(ROOT)))
        files[path.relative_to(ROOT).as_posix()] = digest(path)
    return dict(schema=INPUT_SCHEMA, repository=str(ROOT), base_head=git('rev-parse','HEAD'),
                files=files)


def load_source_evidence(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise RuntimeError('source_evidence_duplicate_key')
            result[key] = value
        return result
    expected = json.loads(path.read_text(), object_pairs_hook=unique)
    if (type(expected) is not dict or set(expected) != {'schema','repository','base_head','files'}
            or expected['schema'] != INPUT_SCHEMA or expected['repository'] != str(ROOT)
            or not isinstance(expected['base_head'],str)
            or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',expected['base_head']) is None
            or type(expected['files']) is not dict or not expected['files']):
        raise RuntimeError('source_evidence_schema')
    for name, value in expected['files'].items():
        if (not isinstance(name,str) or Path(name).is_absolute()
                or Path(name).as_posix() != name or '..' in Path(name).parts
                or not isinstance(value,str) or re.fullmatch(r'[0-9a-f]{64}',value) is None):
            raise RuntimeError('source_evidence_file')
    return expected


def verify_source(expected):
    actual = source_identity()
    # HEAD is contextual provenance, not the identity of uncommitted bytes.
    # A docs-only commit cannot invalidate otherwise identical execution inputs.
    if any(actual[k] != expected[k] for k in ('schema','repository','files')):
        raise RuntimeError('source_inputs_mismatch')
    return actual


def verify_loaded_inputs(identity):
    """Check local module resolution without importing anything for this check."""
    observed = {}
    for path in identity['files']:
        if not path.endswith('.py'): continue
        if path.startswith('tests/'):
            if path not in ('tests/sh_oracle.py','tests/test_formal_config.py'): continue
            name = Path(path).stem
        elif path.startswith(NATIVE+'/'):
            continue
        else:
            name = path[:-3].replace('/','.')
            if name.endswith('.__init__'): name = name[:-9]
        module = sys.modules.get(name)
        if module is None: continue
        location = getattr(module,'__file__',None)
        if location is None or Path(location).resolve() != ROOT/path:
            raise RuntimeError('unexpected_import_source: '+name)
        observed[name] = str(Path(location).resolve())
    return observed


def bounded_command(command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if p.returncode: raise RuntimeError('toolchain_command_failed: '+command[0])
    return (p.stdout+p.stderr)[:8000]


def compare(torch, actual, expected, label, rows, gradient=False):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    atol, rtol = (3e-5, 3e-4) if gradient else (2e-6, 2e-5)
    error = (actual-expected).abs()
    rows.append(dict(check=label, max_abs=error.max().item(),
                     max_bound_ratio=(error/(atol+rtol*expected.abs())).max().item()))
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(label+': nonfinite')
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=label)


def make_camera(config, spec):
    import numpy as np
    from formal_camera import build_camera
    from formal_frame_time import FrameTime
    from scene.cameras import Camera
    state = build_camera(config.dataset.resolution, ('train',0,'synthetic.png'),
                         intrinsics=(spec['fx'],spec['fy'],spec['width']/2,spec['height']/2))
    frame = FrameTime('train',0,'synthetic.png',spec['timestamp'],spec['timestamp'])
    return Camera(0, np.eye(3), np.zeros(3), state.fovx, state.fovy, None, None,
                  'synthetic', 0, data_device='cuda', timestamp=frame.effective_time,
                  cx=state.cx, cy=state.cy, fl_x=state.fx, fl_y=state.fy,
                  resolution=(state.width,state.height), image_path='/synthetic/synthetic.png',
                  meta_only=True, formal_camera=state, formal_frame=frame).cuda()


def make_model(p, stage, spec):
    from scene.gaussian_model import GaussianModel
    model = GaussianModel(3, gaussian_dim=4, time_duration=[0.,spec['duration']],
                          rot_4d=True, force_sh_3d=False, sh_degree_t=2, prefilter_var=-1.0)
    model._xyz = p['mean'].view(1,3)
    model._t = p['time'].view(1,1)
    model._scaling = p['log_scale'][:3].view(1,3)
    model._scaling_t = p['log_scale'][3:].view(1,1)
    model._rotation = p['left'].view(1,4)
    model._rotation_r = p['right'].view(1,4)
    model._opacity = p['opacity'].view(1,1)
    model._features_dc = p['features'][:1].view(1,1,3)
    model._features_rest = p['features'][1:].view(1,47,3)
    for expected in ((0,0),(1,0),(2,0),(3,0),(3,1),(3,2)):
        assert (model.active_sh_degree,model.active_sh_degree_t) == expected
        if expected == stage: return model
        model.oneupSHdegree()
    raise AssertionError('active_stage')


def invoke(torch, binding, renderer, p, stage, camera, config, spec, *, integrated=False,
           diagnostic=False, three_d=False):
    from formal_views import ConsumerView
    bg = torch.tensor(spec['background'],device='cuda',dtype=torch.float32)
    if integrated:
        model = make_model(p,stage,spec)
        result = renderer.render(camera,model,ConsumerView(config,'pipeline'),bg,
                                 scaling_modifier=1.0,override_color=None,formal_config=config)
        rgb, alpha = result['render'],result['alpha']
        assert result['radii'].min().item() > 0
    else:
        settings = binding.GaussianRasterizationSettings(
            image_height=spec['height'],image_width=spec['width'],
            tanfovx=camera.formal_camera.tanx,tanfovy=camera.formal_camera.tany,
            bg=bg,scale_modifier=1.0,viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,sh_degree=stage[0],sh_degree_t=stage[1],
            campos=camera.camera_center,timestamp=spec['timestamp'],time_duration=spec['duration'],
            rot_4d=not three_d,gaussian_dim=3 if three_d else 4,force_sh_3d=False,
            prefiltered=False,debug=False,formal_camera=not three_d,
            camera_binding=None if three_d else camera.formal_binding,
            debug_pixel_x=spec['pixel'][0] if diagnostic else -1,
            debug_pixel_y=spec['pixel'][1] if diagnostic else -1,
            debug_pixel_max_entries=1 if diagnostic else 0,
            debug_preprocess_target_index=0 if diagnostic else -1)
        result = binding.GaussianRasterizer(settings)(
            means3D=p['mean'].view(1,3),means2D=torch.zeros((1,3),device='cuda'),
            shs=p['features'].view(1,48,3),colors_precomp=None,
            flow_2d=torch.zeros((1,2),device='cuda'),opacities=p['opacity'].sigmoid().view(1,1),
            ts=p['time'].view(1,1),scales=p['log_scale'][:3].exp().view(1,3),
            scales_t=p['log_scale'][3:].exp().view(1,1),
            rotations=torch.nn.functional.normalize(p['left'],dim=0).view(1,4),
            rotations_r=torch.nn.functional.normalize(p['right'],dim=0).view(1,4),
            cov3D_precomp=None,prefilter_var=-1.0)
        rgb, radii, _, alpha = result[:4]
        assert radii.min().item() > 0
        if diagnostic:
            assert result[-2].numel() and result[-1].numel()
            assert torch.isfinite(result[-2]).all() and torch.isfinite(result[-1]).all()
    torch.cuda.synchronize()
    x,y = spec['pixel']
    return rgb[:,y,x],alpha[0,y,x]


def run_checks(torch, binding, renderer, result):
    import sh_oracle as o
    from formal_config import resolve_formal_config
    from test_formal_config import fixture, encode
    verify_source(result['source_before'])
    result['loaded_inputs_before_checks'] = verify_loaded_inputs(result['source_before'])
    rows = result['checks']
    # Reuse ONLY the pure semantic fixture, never ConnectionTests GPU patches.
    with tempfile.TemporaryDirectory(prefix='sh-config-') as temp:
        base = Path(temp); source=base/'data'; source.mkdir()
        value=fixture(source,base/'unused-output')
        value['dataset']['resolution']['divisor']=1
        value['dataset']['time'].update(raw_interval=[0,1],divisor=1)
        value['initialization']['time_variance_denominator']=5
        config=resolve_formal_config(encode(value),raw_width=64,raw_height=48)
        cases=o.acceptance_cases()
        result['inputs']=[]
        for label,p,stage,spec,three_d in cases:
            spec['timestamp']=float(torch.tensor(spec['timestamp'],dtype=torch.float32))
            # Both sides start from the same representable raw inputs.
            cpu=o.clone(o.clone(p,dtype=torch.float32))
            result['inputs'].append(dict(case=label,stage=stage,spec=spec,three_d=three_d,
                                         parameters={k:v.detach().tolist() for k,v in cpu.items()}))
            if not three_d: o.assert_safe(cpu,spec,stage)
            camera=make_camera(config,spec)
            expected = (lambda a: o.color_at(a,a['mean'],stage,spec)) if three_d else (lambda a: o.color(a,stage,spec))
            for integrated in ((False,) if three_d else (False,True)):
                gpu=o.clone(cpu,dtype=torch.float32,device='cuda')
                rgb,alpha=invoke(torch,binding,renderer,gpu,stage,camera,config,spec,integrated=integrated,three_d=three_d)
                assert .05 < alpha.item() < .5
                isolated=o.isolate(rgb,alpha,spec['background'])
                tag=f'{label}/'+('render' if integrated else 'binding')
                compare(torch,isolated,expected(cpu),tag+'/color',rows)
                actual_g=o.gradients(o.loss(isolated),gpu)
                expected_g=o.gradients(o.loss(expected(cpu)),cpu)
                for name in cpu: compare(torch,actual_g[name],expected_g[name],tag+'/'+name,rows,True)
                count=(stage[0]+1)**2 if not stage[1] else 16*(stage[1]+1)
                assert actual_g['features'][count:].count_nonzero().item()==0
                if not three_d:
                    a=o.pixel(cpu,stage,spec)
                    compare(torch,rgb,a['rgb'],tag+'/raw_rgb',rows)
                    compare(torch,alpha,a['alpha'],tag+'/alpha',rows)
                if label=='(3, 2)' and integrated:
                    raw_gpu=o.clone(cpu,dtype=torch.float32,device='cuda')
                    raw,aa=invoke(torch,binding,renderer,raw_gpu,stage,camera,config,spec,integrated=True)
                    grad=o.gradients(o.loss(raw),raw_gpu)
                    ref=o.gradients(o.loss(o.pixel(cpu,stage,spec)['rgb']),cpu)
                    for name in cpu: compare(torch,grad[name],ref[name],tag+'/raw_grad/'+name,rows,True)
                    # Actual CUDA-forward FD supplements the independent CPU
                    # comparison. It cannot excuse failure against that oracle.
                    def forward_loss(a):
                        with torch.no_grad():
                            g=o.clone(a,dtype=torch.float32,device='cuda')
                            c,_=invoke(torch,binding,renderer,g,stage,camera,config,spec,integrated=True)
                            return o.loss(c).cpu()
                    for h in (1e-3,5e-4):
                        fd=o.finite_difference(forward_loss,cpu,h)
                        for name in cpu:
                            ag=grad[name].detach().cpu().double()
                            torch.testing.assert_close(ag,fd[name],atol=1e-3,rtol=5e-3,msg=f'CUDA FD {h} {name}')
                            rows.append(dict(check=f'cuda_fd/{h}/{name}',max_abs=(ag-fd[name]).abs().max().item()))
                if not integrated and not three_d:
                    # Inactive slots must not influence the executed forward.
                    changed=o.clone(cpu,dtype=torch.float32,device='cuda')
                    with torch.no_grad(): changed['features'][count:].fill_(99.)
                    c,a=invoke(torch,binding,renderer,changed,stage,camera,config,spec)
                    assert torch.equal(c,rgb.detach()) and torch.equal(a,alpha.detach())
            if label=='(3, 2)':
                g=o.clone(cpu,dtype=torch.float32,device='cuda')
                normal=invoke(torch,binding,renderer,g,stage,camera,config,spec)
                diagnostic=invoke(torch,binding,renderer,g,stage,camera,config,spec,diagnostic=True)
                assert all(torch.equal(a,b) for a,b in zip(normal,diagnostic))
        assert not (base/'unused-output').exists()
    result['case_count']=len(cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-source-evidence',action='store_true')
    parser.add_argument('--source-evidence',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if sys.executable!=PYTHON or not sys.dont_write_bytecode: raise RuntimeError('4dgs310_B_required')
    if any(n in sys.modules for n in ('torch','gaussian_renderer','scene')): raise RuntimeError('fresh_process_required')
    if args.record_source_evidence:
        if args.source_evidence is not None or args.output_dir is not None:
            parser.error('record mode cannot execute or consume approval')
        print(json.dumps(source_identity(),indent=2))
        return
    if args.source_evidence is None or args.output_dir is None:
        parser.error('execution requires --source-evidence and --output-dir')
    expected=load_source_evidence(args.source_evidence)
    before=verify_source(expected)  # Before any output, toolchain command or heavy import.
    out=args.output_dir
    if not out.is_absolute() or out.exists() or out.is_symlink(): raise RuntimeError('new_absolute_output_required')
    out=out.resolve()
    if out==ROOT or ROOT in out.parents or out in ROOT.parents: raise RuntimeError('output_overlaps_repository')
    out.mkdir()  # Parent must already be explicitly prepared by the caller.
    build=out/'build'; build.mkdir()
    os.environ['TORCH_EXTENSIONS_DIR']=str(build)
    sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
    result=dict(status='failed',source_before=before,expected_source=expected,
                python=sys.executable,checks=[])
    try:
        import torch
        result['environment']=dict(python_version=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
            nvcc=bounded_command(['nvcc','--version']),compiler=bounded_command(['c++','--version']),
            driver=bounded_command(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']))
        if not torch.cuda.is_available(): raise RuntimeError('CUDA_required_no_fallback')
        result['environment']['gpu']=torch.cuda.get_device_name(0)
        verify_source(before)  # Recheck immediately before the loader/build.
        # This is the actual production loader, with an initially empty cache.
        with (out/'build.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
            import gaussian_renderer as renderer
            from gaussian_renderer import diff_gaussian_rasterization as binding
        import simple_knn._C as knn
        result['simple_knn_import_only']=dict(path=knn.__file__,sha256=digest(knn.__file__))
        binary=Path(binding._C.__file__).resolve()
        if build not in binary.parents: raise RuntimeError('loaded_binary_outside_fresh_build')
        if Path(binding.__file__).resolve()!=ROOT/'gaussian_renderer/diff_gaussian_rasterization.py': raise RuntimeError('wrong_binding')
        ninja=binary.parent/'build.ninja'
        if not ninja.is_file() or not list(binary.parent.glob('*.o')): raise RuntimeError('fresh_build_evidence_missing')
        if not all(str(ROOT/p).replace(' ','$ ') in ninja.read_text() for p in TRANSLATION_UNITS):
            raise RuntimeError('build_translation_units_mismatch')
        result['extension']=dict(path=str(binary),sha256=digest(binary),ninja_sha256=digest(ninja),
                                 objects=[p.name for p in binary.parent.glob('*.o')],
                                 translation_units=list(TRANSLATION_UNITS))
        result['loaded_inputs_after_build']=verify_loaded_inputs(before)
        run_checks(torch,binding,renderer,result)
        torch.cuda.synchronize()
        if digest(binary)!=result['extension']['sha256']: raise RuntimeError('binary_changed')
        if digest(ninja)!=result['extension']['ninja_sha256']: raise RuntimeError('build_recipe_changed')
        result['source_after']=verify_source(before)
        result['loaded_inputs_after_checks']=verify_loaded_inputs(before)
        result['status']='passed'
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'[:4000]
        raise
    finally:
        with (out/'result.json').open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)


if __name__=='__main__':
    main()
