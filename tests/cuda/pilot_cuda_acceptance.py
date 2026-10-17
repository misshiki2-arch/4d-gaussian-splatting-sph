"""Step 16 pilot: record is CPU/read-only; run requires separate authorization.

run calls the ordinary formal entry once, then one independent same-binary
reader. No synthetic run, resume, retry, alternate configuration or cleanup.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).parent)]
import pilot_contract as contract
from formal_checkpoint_identity import (NativeEvidence, canonical, file_identity, require,
    strict_json, validate_native, RASTER_UNITS)
from real_input_cuda_acceptance import environment, camera_values, render_buffers


def resources(out, *, initial):
    """Only called in the authorized live worker, after formal input preflight."""
    import torch
    from torch.utils.cpp_extension import get_default_build_root
    mem={s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    disks={}
    for name,path in (('evidence',out),('output',Path(contract.CONFIG).parent),('cache',Path(get_default_build_root()))):
        if name=='output': path=Path(strict_json(contract.CONFIG.read_bytes())['output']['directory'])
        while not path.exists(): path=path.parent
        disks[name]=shutil.disk_usage(path).free
    values=dict(host_available=mem['MemAvailable'],disks=disks)
    if initial: values['gpu_free']=torch.cuda.mem_get_info(0)[0]
    else: values.update(rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved())
    contract.check_resources(values,initial=initial)
    return values


def worker(out, expected):
    require(not any(n in sys.modules for n in ('torch','numpy','train','scene','gaussian_renderer')), 'worker_preloaded')
    contract.verify_candidate(expected)
    import formal_entry, formal_native_bootstrap as native
    result=dict(status='running',stage='entry',entry_calls=0,writer=dict(present=False,flush=0,close=0),resources=[])
    held={}
    try:
        with ExitStack() as stack:
            original=native.bootstrap
            def bootstrap(source):
                result['resources_before_build']=resources(out,initial=True)
                result['environment']=environment()
                train,evidence=original(source)
                import torch
                from pilot_runtime import Observation
                from checkpoint_cuda_acceptance import guard_pointops
                result['pointops']=guard_pointops(stack); result['native']=evidence.value(); held['evidence']=evidence
                torch.cuda.reset_peak_memory_stats()
                begin=train.begin_formal_output; continuation=train.continue_formal_training
                def observed_begin(p,claim):
                    writer=begin(p,claim); held['prepared']=p
                    result['writer']['present']=writer is not None
                    if writer is not None:
                        for name in ('flush','close'):
                            original_method=getattr(writer,name)
                            def observed(*a,_name=name,_method=original_method,**kw):
                                value=_method(*a,**kw); result['writer'][_name]+=1; return value
                            stack.enter_context(patch.object(writer,name,observed))
                    return writer
                def observe(p,writer):
                    def check(k,stage): result['resources'].append(dict(k=k,stage=stage,values=resources(out,initial=False)))
                    check(0,'prepared')
                    obs=Observation(p,train.render,out,expected['schedule'],expected['representatives'],
                                    expected['probes'],expected['fixed_updates'],resource_check=check)
                    held['observer']=obs
                    returned=obs.run(continuation,writer)
                    result.update(spec=obs.spec,trace=obs.trace,consumed=obs.consumed.finish(),
                                  expected_identity=obs.expected_identity,learning=obs.summary(),pilot_diagnostics=obs.diagnostics)
                    return returned
                stack.enter_context(patch.object(train,'begin_formal_output',observed_begin))
                stack.enter_context(patch.object(train,'continue_formal_training',observe))
                return train,evidence
            stack.enter_context(patch.object(native,'bootstrap',bootstrap))
            result['entry_calls']+=1; result['returned']=formal_entry.run(contract.CONFIG)
            p=held['prepared']; result['diagnostic']=asdict(p.health_run)
            require(p.health_run.stage=='normal_return', 'entry_return_stage')
            result['stage']=p.health_run.stage
            validate_native(held['evidence'],live=True); contract.verify_candidate(expected)
            result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__)
        if 'prepared' in held: result['diagnostic']=asdict(held['prepared'].health_run)
        if 'observer' in held:
            result['trace']=held['observer'].trace
            result['pilot_diagnostics']=held['observer'].diagnostics
        raise
    finally: contract.write_json(out/'worker-result.json',result)


def read_expected(out, row, expected_identity):
    """Public M -> T first, independent live expectation second. CPU-testable."""
    import formal_checkpoint as cp
    loaded=cp.read_checkpoint(Path(row['path']))
    require(loaded['marker_identity']==row['marker'] and loaded['metadata']['k']==row['k'], 'marker_preserved')
    import torch
    from test_formal_checkpoint import assert_expected
    import unittest
    require(file_identity(out/'expected.pt')==expected_identity, 'expected_snapshot_identity')
    saved=torch.load(out/'expected.pt',map_location='cpu',weights_only=True)
    assert_expected(unittest.TestCase(),(saved['attrs'],saved['groups']),loaded)
    return loaded,saved


def reader(out, expected):
    require(not any(n in sys.modules for n in ('torch','numpy','scene','gaussian_renderer')), 'reader_preloaded')
    contract.verify_candidate(expected)
    w=strict_json((out/'worker-result.json').read_bytes(),64*1024**2)
    require(w['status']=='passed', 'worker_required')
    row=w['trace']['checkpoints'][-1]; result=dict(status='running',stage='M_T')
    try:
        loaded,saved=read_expected(out,row,w['expected_identity'])
        require(loaded['metadata']['n']==expected['schedule']['updates']==row['k'], 'final_only')
        import torch.utils.cpp_extension as ext
        import formal_native_bootstrap as native
        from formal_checkpoint_state import restore_diagnostic,equal_state
        from checkpoint_cuda_acceptance import diagnostic_camera,guard_pointops
        from training_health_cuda_acceptance import guard_reader_training
        from test_formal_checkpoint import SLOTS
        require(environment()==w['environment'], 'same_environment')
        evidence=NativeEvidence(canonical(w['native'])); validate_native(evidence)
        loads=[]
        with ExitStack() as stack:
            guard_pointops(stack); native._load_knn(Path(w['native']['knn']['binary']['path']))
            def exact_load(*a,**kw):
                require(not a and kw==dict(name='diff_gaussian_rasterization',sources=[str(ROOT/u) for u in RASTER_UNITS],
                    extra_cuda_cflags=['-I '+str(ROOT/'diff-gaussian-rasterization/third_party/glm')+'/', '-g'],verbose=True), 'reader_load_request')
                require(not loads, 'load_once'); loads.append(1)
                spec=importlib.util.spec_from_file_location('diff_gaussian_rasterization',w['native']['raster']['binary']['path'])
                module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
            stack.enter_context(patch.object(ext,'load',exact_load))
            import gaussian_renderer
            guard_reader_training(stack)
            model=restore_diagnostic(loaded,'cuda'); require(model.optimizer is None, 'no_resume')
            for slot,attr in SLOTS.items(): equal_state(getattr(model,attr).detach().cpu(),saved['attrs'][slot])
            require(len(saved['views'])==len(expected['representatives']), 'representative_count')
            for view,rep in zip(saved['views'],expected['representatives']):
                split,index,path,stamp=rep; cam=diagnostic_camera(loaded['config'],(split,index,path))
                require(camera_values(cam)==view['camera'] and cam.formal_frame.raw_time==stamp, 'representative_camera')
                equal_state(view['output'],render_buffers(gaussian_renderer.render,cam,model,loaded['config']))
            require(loads==[1], 'reader_load_count'); validate_native(evidence,live=True)
        contract.verify_candidate(expected)
        result.update(status='passed',stage='completed',checkpoint=row,representatives=expected['representatives'],
            parameter_groups=sorted(contract.TOLERANCES_GROUPS),exact=True,reader_builds=0,forwards=len(saved['views']))
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__); raise
    finally: contract.write_json(out/'reader-result.json',result)


def child(mode,out,timeout):
    command=[contract.PYTHON,'-B',str(Path(__file__).resolve()),mode,'--out',str(out)]
    timed_out=False; start=time.monotonic()
    with (out/(mode+'.stdout.log')).open('xb') as stdout,(out/(mode+'.stderr.log')).open('xb') as stderr:
        p=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
        try: p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
    contract.write_json(out/(mode+'-exit.json'),dict(mode=mode,argv=command,cwd=str(ROOT),exit_code=p.returncode,
        timed_out=timed_out,seconds=time.monotonic()-start,logs={s:file_identity(out/(mode+'.'+s+'.log')) for s in ('stdout','stderr')}))
    require(not timed_out and p.returncode==0, 'child_failed_'+mode)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('operation',choices=('record','run','worker','reader','verify'))
    parser.add_argument('--out',required=True,type=Path); parser.add_argument('--candidate',type=Path)
    args=parser.parse_args(argv); require(sys.executable==contract.PYTHON, 'python')
    if args.operation=='record': contract.write_json(args.out,contract.candidate()); return
    if args.operation=='verify': contract.verify_completion(args.out); print('STEP16_SAVED_RESULT_VERIFIED'); return
    if args.operation in ('worker','reader'):
        expected=strict_json((args.out/'inputs.json').read_bytes(),64*1024**2)
        require(args.out==Path(expected['evidence_output']), 'child_destination')
        (worker if args.operation=='worker' else reader)(args.out,expected); return
    require(args.candidate is not None, 'candidate_required')
    expected=strict_json(args.candidate.read_bytes(),64*1024**2); contract.preflight(expected,args.out)
    args.out.mkdir(mode=0o700); contract.write_json(args.out/'inputs.json',expected)
    start=time.monotonic(); result=dict(status='running',stage='worker')
    try:
        child('worker',args.out,contract.RESOURCES['worker_seconds']); result['stage']='reader'
        remaining=contract.RESOURCES['wall_seconds']-(time.monotonic()-start)
        require(remaining>0, 'wall_budget')
        child('reader',args.out,min(remaining,contract.RESOURCES['reader_seconds']))
        w,_=contract.validate_results(args.out)
        result.update(status='passed',stage='completed',seconds=time.monotonic()-start,learning=w['learning'])
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,seconds=time.monotonic()-start)
        try: contract.finalize(args.out,result)
        except BaseException as secondary: print('failure_record_error='+type(secondary).__name__,file=sys.stderr)
        raise
    contract.finalize(args.out,result)
    print('STEP16_RESULT_COMPLETE')


if __name__=='__main__': main()
