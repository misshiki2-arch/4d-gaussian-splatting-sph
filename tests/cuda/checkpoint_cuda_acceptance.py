"""Step 13 future one-run CUDA acceptance. record/preflight are stdlib only.

Executing run requires separate authorization. Never invoke from CPU tests.
G0a uses the real ordinary entry and production fresh-build bootstrap. The
reader imports those exact binaries without rebuilding, then creates new models.
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from formal_checkpoint_identity import (NativeEvidence,canonical,file_identity,require,
    source_snapshot,strict_json,validate_native,RASTER_UNITS)
REPORTS=ROOT.parent/'reports/corrected-4dgs/phase1/step13'
PYTHON='/home/demo/miniconda3/envs/4dgs310/bin/python'
EXTRA=('tests/cuda/checkpoint_cuda_acceptance.py','tests/training_update_runtime.py',
       'tests/training_update_oracle.py','tests/test_formal_checkpoint.py',
       'tests/formal_training_fixture.py','tests/test_formal_connection.py',
       'tests/checkpoint_native_fixture.py','tests/checkpoint_entry_cpu.py',
       'tests/test_formal_config.py','tests/test_training_update_cpu.py',
       'tests/test_checkpoint_cuda_preflight.py')
CASES=('G0a','G1','G4','G5','G8')


def candidate():
    files=source_snapshot()
    files.update({n:file_identity(ROOT/n) for n in EXTRA})
    head=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    return dict(schema='step13-checkpoint-inputs-v1',repository=str(ROOT),base_head=head,files=files)


def verify_candidate(expected):
    actual=candidate()
    require(set(expected)==set(actual) and all(expected[k]==actual[k] for k in
        ('schema','repository','files')), 'reviewed_inputs')
    return actual


def write_json(path,value):
    data=canonical(value)
    with path.open('xb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    require(strict_json(path.read_bytes(),64*1024**2)==value,'result_readback')
    return file_identity(path)


def preflight(expected,out):
    verify_candidate(expected)
    require(sys.executable==PYTHON,'python')
    require(out.is_absolute() and out.resolve()==out and not out.exists() and
            REPORTS in out.parents and out.parent.is_dir(),'new_report_output')
    require(not any(k.startswith('STEP90_CUDA_DEBUG') and v for k,v in os.environ.items()),'debug_override')
    require(not any(n in sys.modules for n in ('torch','train','scene','gaussian_renderer')),'preloaded')


def guard_pointops(stack):
    import pointops2_cuda
    def forbidden(*a,**kw):raise AssertionError('pointops_kernel_out_of_scope')
    names=[]
    for name,value in vars(pointops2_cuda).copy().items():
        if not name.startswith('_') and callable(value):
            stack.enter_context(patch.object(pointops2_cuda,name,forbidden));names.append(name)
    require(bool(names),'pointops_guard')
    return dict(binary=file_identity(Path(pointops2_cuda.__file__).resolve()),blocked=names)


def camera_values(camera):
    tensors=('world_view_transform','projection_matrix','full_proj_transform','camera_center')
    return dict(key=list((camera.formal_frame.split,camera.formal_frame.index,camera.formal_frame.file_path)),
        timestamp=camera.timestamp,width=camera.image_width,height=camera.image_height,
        fx=camera.fl_x,fy=camera.fl_y,cx=camera.cx,cy=camera.cy,
        **{n:getattr(camera,n).detach().cpu().tolist() for n in tensors})


def buffer(torch,renderer,camera,model,config,budget):
    from formal_views import ConsumerView
    require(len(model._xyz)<=64,'row_budget')
    budget['forwards']+=1;require(budget['forwards']<=500,'forward_budget')
    with torch.no_grad():
        value=renderer(camera.cuda(),model,ConsumerView(config,'pipeline'),
                       torch.zeros(3,device='cuda'),formal_config=config)
    result={n:value[n].detach().cpu().contiguous().clone() for n in
            ('render','alpha','depth','flow','radii','visibility_filter')}
    require(all(bool(torch.isfinite(v).all()) for v in result.values()),'render_finite')
    require(bool(result['visibility_filter'].any()) and bool((result['alpha']>0).any()),'nonempty_render')
    return result


def gpu_identity(torch):
    require(torch.cuda.is_available(),'CUDA_required')
    return dict(name=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)),
                torch=str(torch.__version__),cuda=str(torch.version.cuda),python=sys.version,executable=sys.executable)


def worker(out,expected):
    import torch
    import formal_entry,formal_inputs,formal_native_bootstrap as native
    import formal_checkpoint
    import training_update_runtime as runtime
    from test_formal_checkpoint import independent,assert_expected
    from formal_checkpoint_state import equal_state
    import unittest
    test=unittest.TestCase();torch.set_num_threads(1)
    result=dict(schema='step13-worker-v1',status='running',stage='preflight',cases=[],checkpoints=[],
                environment=gpu_identity(torch))
    budget=dict(forwards=0);held={};events=[]
    try:
        verify_candidate(expected)
        with ExitStack() as stack:
            result['pointops']=guard_pointops(stack)
            cases=[next(c for c in runtime.CASES if c['name']==n) for n in CASES]
            def observe(p,case,writer,continuation):
                obs=runtime.Observation(p,case,held['render'],budget)
                def save(real_save,k):
                    original=independent(p.gaussians)
                    require(all(g['state'] for g in original[1].values()),'nine_initialized_groups')
                    rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
                    views=[]
                    for split,index in (('train',0),('test',0),('test',1)):
                        cameras=(p.scene.train_cameras if split=='train' else p.scene.test_cameras)[1.0]
                        cam=next(c for c in cameras if c.formal_frame.split==split and c.formal_frame.index==index)
                        views.append(dict(camera=camera_values(cam),output=buffer(torch,held['render'],
                                     cam,p.gaussians,p.inputs.config,budget)))
                    # Extra observation cannot perturb state, RNG or training.
                    equal_state(original,independent(p.gaussians))
                    require(torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'observation_rng')
                    receipt=real_save(k)
                    directory=Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k))
                    loaded=formal_checkpoint.read_checkpoint(directory)
                    assert_expected(test,original,loaded)
                    require(loaded['metadata']['base']['native']==held['native'].value(),'entry_evidence_handoff')
                    path=out/(case['name']+'-'+str(k)+'-expected.pt')
                    with path.open('xb') as f:torch.save(dict(attrs=original[0],groups=original[1],views=views),f)
                    result['checkpoints'].append(dict(case=case['name'],k=k,path=str(directory),
                        marker=loaded['marker_identity'],expected=str(path),expected_identity=file_identity(path),
                        reference=json.loads(p.inputs.reference_bytes)))
                    return receipt
                try:trace=obs.run(writer,checkpoint_observer=save,continue_call=continuation)
                finally:
                    # Failure trace is preserved even if no checkpoint was published.
                    write_json(out/(case['name']+'-trace.json'),runtime.serializable(obs.trace))
                result['cases'].append(dict(name=case['name'],steps=len(trace['steps']),status=trace['status']))
                return case['n']
            first=cases[0];_,ref=runtime.create_fixture(out/'G0a',first)
            original_bootstrap=native.bootstrap
            def bootstrap(source):
                events.append('bootstrap');train,evidence=original_bootstrap(source)
                held.update(runtime=train,native=evidence,render=train.render)
                prepare,begin,continuation=train.prepare_formal_runtime,train.begin_formal_output,train.continue_formal_training
                def observed_prepare(inputs,**kw):
                    events.append('prepare');require(kw['native_evidence'] is evidence,'same_native_object')
                    p=prepare(inputs,**kw);held['g0a']=p;return p
                def observed_begin(p,claim):
                    events.append('claim');writer=begin(p,claim)
                    require(p.scene._checkpoint_writer.context is p.checkpoint_context,'same_context')
                    return writer
                def observed_continue(p,writer):
                    events.append('continue');return observe(p,first,writer,continuation)
                # Scoped wrappers delegate; no prepared object/receipt injection.
                stack.enter_context(patch.object(train,'prepare_formal_runtime',observed_prepare))
                stack.enter_context(patch.object(train,'begin_formal_output',observed_begin))
                stack.enter_context(patch.object(train,'continue_formal_training',observed_continue))
                held['prepare'],held['begin'],held['continue']=prepare,begin,continuation
                return train,evidence
            result['stage']='G0a/ordinary-entry'
            with patch.object(formal_inputs,'_reference',return_value=ref),patch.object(native,'bootstrap',bootstrap):
                require(formal_entry.run(out/'G0a/config.json')==1,'ordinary_entry_N')
            require(events==['bootstrap','prepare','claim','continue'],'ordinary_entry_order')
            result['entry_events']=events;result['native']=held['native'].value()
            for case in cases[1:]:
                result['stage']=case['name']
                value,ref=runtime.create_fixture(out/case['name'],case)
                if case['name'] in ('G1','G5'):
                    value['checkpoint']['intermediate_updates']=list(range(1,case['n']))
                from test_formal_config import encode
                with patch.object(formal_inputs,'_reference',return_value=ref):
                    inputs=formal_inputs.verify_inputs(encode(value))
                    p=held['prepare'](inputs,native_evidence=held['native'])
                    runtime.initialize_population(p,case)
                    claim=formal_entry.OutputClaim(inputs);writer=None
                    try:
                        writer=held['begin'](p,claim)
                        observe(p,case,writer,held['continue'])
                    finally:
                        if writer is not None:writer.close()
                        claim.close()
            require(len(result['checkpoints'])==13 and sum(c['steps'] for c in result['cases'])==15,'coverage_total')
            require([r['k'] for r in result['checkpoints'] if r['case']=='G1']==[1,2,3,4,5,6],'SH_stages')
            verify_candidate(expected);result.update(status='passed',stage='completed',forwards=budget['forwards'])
    except BaseException as error:
        result.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:write_json(out/'worker-result.json',result)


def diagnostic_camera(c,frame_key):
    """Rebuild the reader's CPU camera using the saved, verified dataset config."""
    import formal_inputs,formal_camera
    from formal_views import ConsumerView
    from scene.dataset_readers import readCamerasFromTransforms
    from utils.camera_utils import loadCam
    split,index,_=frame_key;source=Path(c.paths.source_path)
    raw=(source/('transforms_'+split+'.json')).read_bytes();data=json.loads(raw)
    frames=tuple(formal_inputs.Frame(split,i,f['file_path'],float(f['time']),float(f['time'])/c.time_derivation.divisor)
                 for i,f in enumerate(data['frames']))
    cameras=tuple(formal_camera.from_metadata(data,f,c.dataset.resolution,(split,i,f['file_path']))
                  for i,f in enumerate(data['frames']))
    infos=readCamerasFromTransforms(str(source),'transforms_'+split+'.json',False,
          extension=c.dataset.extension,dataloader=True,formal_metadata=(raw,frames),formal_cameras=cameras)
    return loadCam(ConsumerView(c,'dataset'),index,infos[index],1.,formal_resolution=c.dataset.resolution)


def reader(out,expected):
    import formal_checkpoint as cp,formal_inputs,formal_native_bootstrap as native
    worker_result=strict_json((out/'worker-result.json').read_bytes(),64*1024**2)
    require(worker_result['status']=='passed','worker_required')
    verify_candidate(expected);evidence=NativeEvidence(canonical(worker_result['native']))
    validate_native(evidence)
    import torch
    import torch.utils.cpp_extension as ext
    from formal_checkpoint_state import restore_diagnostic,equal_state
    from test_formal_checkpoint import assert_expected,SLOTS
    import unittest
    test=unittest.TestCase();budget=dict(forwards=worker_result['forwards']);rows=[]
    require(gpu_identity(torch)==worker_result['environment'],'same_GPU_environment')
    with ExitStack() as stack:
        guard_pointops(stack);native._load_knn(Path(worker_result['native']['knn']['binary']['path']))
        count=[]
        def exact_load(*a,**kw):
            require(not a and kw==dict(name='diff_gaussian_rasterization',
                sources=[str(ROOT/n) for n in RASTER_UNITS],
                extra_cuda_cflags=['-I '+str(ROOT/'diff-gaussian-rasterization/third_party/glm')+'/', '-g'],verbose=True), 'reader_load_request')
            require(not count,'reader_load_once');count.append(1)
            path=worker_result['native']['raster']['binary']['path']
            spec=importlib.util.spec_from_file_location('diff_gaussian_rasterization',path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
        stack.enter_context(patch.object(ext,'load',exact_load))
        import gaussian_renderer as renderer
        from scene.gaussian_model import GaussianModel
        def forbidden(*a,**kw):raise AssertionError('diagnostic_training_or_KNN_forbidden')
        stack.enter_context(patch.object(GaussianModel,'training_setup',forbidden))
        stack.enter_context(patch.object(torch.optim.Adam,'load_state_dict',forbidden))
        stack.enter_context(patch.object(sys.modules['scene.gaussian_model'],'distCUDA2',forbidden))
        stack.enter_context(patch.object(sys.modules['simple_knn._C'],'distCUDA2',forbidden))
        for row in worker_result['checkpoints']:
            require(file_identity(Path(row['expected']))==row['expected_identity'],'expected_snapshot_identity')
            expected_state=torch.load(row['expected'],map_location='cpu',weights_only=True)
            with patch.object(formal_inputs,'_reference',return_value=row['reference']):
                result=cp.read_checkpoint(Path(row['path']))
                require(result['marker_identity']==row['marker'],'marker_identity')
                assert_expected(test,(expected_state['attrs'],expected_state['groups']),result)
                model=restore_diagnostic(result,'cuda');c=result['config']
                for slot,name in SLOTS.items():equal_state(getattr(model,name).detach().cpu(),expected_state['attrs'][slot])
                for view in expected_state['views']:
                    cam=diagnostic_camera(c,view['camera']['key'])
                    require(camera_values(cam)==view['camera'],'camera_exact')
                    actual=buffer(torch,renderer.render,cam,model,c,budget)
                    equal_state(view['output'],actual)
                    rows.append(dict(case=row['case'],k=row['k'],frame=view['camera']['key'],max_abs=0.,max_bound_ratio=0.,exact=True))
        require(count==[1] and len(rows)==39 and budget['forwards']<=500,'reader_coverage')
        validate_native(evidence,live=True);verify_candidate(expected)
        write_json(out/'reader-result.json',dict(schema='step13-reader-v1',status='passed',comparisons=rows,forwards=budget['forwards']))


def child(mode,out,timeout):
    with (out/(mode+'.stdout.log')).open('xb') as stdout,(out/(mode+'.stderr.log')).open('xb') as stderr:
        p=subprocess.Popen([PYTHON,'-B',str(Path(__file__).resolve()),mode,'--out',str(out)],
                           stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    write_json(out/(mode+'-exit.json'),dict(mode=mode,exit_code=code))
    require(code==0,'child_failed_'+mode)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('record','run','worker','reader'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--candidate',type=Path)
    args=parser.parse_args(argv)
    require(sys.executable==PYTHON,'python')
    if args.operation=='record':
        write_json(args.out,candidate());return
    if args.operation in ('worker','reader'):
        expected=strict_json((args.out/'inputs.json').read_bytes())
        (worker if args.operation=='worker' else reader)(args.out,expected);return
    require(args.candidate is not None,'candidate_required')
    expected=strict_json(args.candidate.read_bytes());preflight(expected,args.out)
    args.out.mkdir(mode=0o700);write_json(args.out/'inputs.json',expected)
    import time
    start=time.monotonic();result=dict(schema='step13-entry-v1',status='running',stage='worker')
    try:
        child('worker',args.out,1800)
        result['stage']='reader';remaining=1800-(time.monotonic()-start)
        require(remaining>0,'wall_budget');child('reader',args.out,remaining)
        for name in ('worker','reader'):
            require(strict_json((args.out/(name+'-result.json')).read_bytes(),64*1024**2)['status']=='passed','child_result')
        verify_candidate(expected)
        result.update(status='passed',stage='completed',seconds=time.monotonic()-start,
            files={p.name:file_identity(p) for p in args.out.iterdir() if p.is_file()})
    except BaseException as error:
        result.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:
        receipt=write_json(args.out/'result.json',result)
        write_json(args.out/'receipt.json',dict(result=receipt,status=result['status']))


if __name__=='__main__':main()
