"""Step 15 real-input acceptance. Only record is authorized during preparation.

run requires separate user authorization. It invokes one ordinary formal entry
and a separate diagnostic reader, never a synthetic case or a resume.
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
import real_input_contract as contract
from formal_checkpoint_identity import (NativeEvidence,canonical,file_identity,require,strict_json,
                                         validate_native,RASTER_UNITS)


def resources(out, *, gpu=False):
    result={}
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    result['host_available']=mem['MemAvailable']
    require(result['host_available']>=contract.RESOURCES['host_available'],'host_memory_budget')
    for name,path in (('evidence',out),('output',Path(strict_json(contract.CONFIG.read_bytes())['output']['directory']).parent)):
        while not path.exists():path=path.parent
        result[name+'_disk_free']=shutil.disk_usage(path).free
        require(result[name+'_disk_free']>=contract.RESOURCES['disk_free'],'disk_budget')
    if gpu:
        import torch
        from torch.utils.cpp_extension import get_default_build_root
        cache=Path(get_default_build_root()).resolve()
        while not cache.exists():cache=cache.parent
        result['cache_disk_free']=shutil.disk_usage(cache).free
        require(result['cache_disk_free']>=contract.RESOURCES['disk_free'],'cache_disk_budget')
        free,total=torch.cuda.mem_get_info(0);result.update(vram_free=free,vram_total=total)
        require(free>=contract.RESOURCES['vram_free'],'vram_budget')
    return result


def environment():
    import torch
    import importlib.metadata
    result=dict(executable=sys.executable,python=sys.version,torch=str(torch.__version__),
        cuda=str(torch.version.cuda),cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),threads=torch.get_num_threads(),packages={})
    for name in ('numpy','Pillow','torchvision','torchaudio','omegaconf'):
        result['packages'][name]=importlib.metadata.version(name)
    for name,args in (('nvcc',['nvcc','--version']),('driver',['nvidia-smi','--query-gpu=driver_version,name,memory.total','--format=csv,noheader'])):
        command=shutil.which(args[0]);require(command is not None,'environment_command')
        p=subprocess.run([command,*args[1:]],capture_output=True,text=True,timeout=20)
        require(p.returncode==0 and len(p.stdout)+len(p.stderr)<16384,'environment_record')
        result[name]=dict(command=command,stdout=p.stdout,stderr=p.stderr)
    return result


def camera_values(camera):
    from checkpoint_cuda_acceptance import camera_values as existing
    return existing(camera)


def render_buffers(renderer,camera,model,c):
    import torch
    from formal_views import ConsumerView
    with torch.no_grad():
        result=renderer(camera.cuda(),model,ConsumerView(c,'pipeline'),torch.zeros(3,device='cuda'),formal_config=c)
    fields=('render','alpha','depth','radii','visibility_filter')
    value={k:result[k].detach().cpu().contiguous().clone() for k in fields}
    require(all(bool(torch.isfinite(v).all()) for v in value.values()),'representative_finite')
    return value


def worker(out,expected):
    require(not any(n in sys.modules for n in ('torch','numpy','train','scene','gaussian_renderer')),'worker_preloaded')
    contract.verify_candidate(expected)
    import formal_entry,formal_native_bootstrap as native
    result=dict(status='running',stage='entry',entry_calls=0,writer=dict(present=False,flush=0,close=0))
    held={}
    try:
        with ExitStack() as stack:
            real_bootstrap=native.bootstrap
            def bootstrap(source):
                # Called only after real formal_input validation in this process.
                result['resources_before_build']=resources(out,gpu=True)
                result['environment']=environment()
                train,evidence=real_bootstrap(source)
                import torch
                from real_input_runtime import Observation,rng_state,state
                from formal_checkpoint_state import equal_state
                from checkpoint_cuda_acceptance import diagnostic_camera,guard_pointops
                result['pointops']=guard_pointops(stack)
                held['evidence']=evidence;result['native']=evidence.value()
                torch.cuda.reset_peak_memory_stats()
                real_continue=train.continue_formal_training;real_begin=train.begin_formal_output
                def begin(p,claim):
                    writer=real_begin(p,claim);held['prepared']=p
                    result['writer']['present']=writer is not None
                    if writer is not None:
                        for name in ('flush','close'):
                            original=getattr(writer,name)
                            def observe(*a,_name=name,_original=original,**kw):
                                value=_original(*a,**kw);result['writer'][_name]+=1;return value
                            stack.enter_context(patch.object(writer,name,observe))
                    return writer
                def continuation(p,writer):
                    def save_observer(k,independent_state):
                        if k!=p.inputs.config.optimization.total_updates:return
                        require('expected_identity' not in result,'expected_once')
                        before=rng_state(p.gaussians);views=[]
                        for split,index,path,stamp in expected['representatives']:
                            cam=diagnostic_camera(p.inputs.config,(split,index,path))
                            require(cam.formal_frame.raw_time==stamp,'representative_time')
                            views.append(dict(camera=camera_values(cam),output=render_buffers(train_render,cam,p.gaussians,p.inputs.config)))
                        equal_state(before,rng_state(p.gaussians));equal_state(independent_state,state(p.gaussians))
                        with (out/'expected.pt').open('xb') as f:
                            torch.save(dict(attrs=independent_state[0],groups=independent_state[1],views=views),f)
                            f.flush();os.fsync(f.fileno())
                        result['expected_identity']=file_identity(out/'expected.pt')
                        result['representative_forwards']=len(views)
                    train_render=train.render
                    obs=Observation(p,train_render,save_observer=save_observer);held['observer']=obs
                    returned=obs.run(real_continue,writer)
                    result.update(spec=obs.spec,trace=obs.trace,consumed=obs.consumed.finish())
                    return returned
                stack.enter_context(patch.object(train,'begin_formal_output',begin))
                stack.enter_context(patch.object(train,'continue_formal_training',continuation))
                return train,evidence
            stack.enter_context(patch.object(native,'bootstrap',bootstrap))
            result['entry_calls']+=1
            result['returned']=formal_entry.run(contract.CONFIG)
            import torch
            p=held['prepared'];result['diagnostic']=asdict(p.health_run)
            require(p.health_run.stage=='normal_return','entry_return_stage')
            result.update(representatives=expected['representatives'],stage=p.health_run.stage,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved())
            validate_native(held['evidence'],live=True);contract.verify_candidate(expected)
            result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__)
        if 'prepared' in held:result['diagnostic']=asdict(held['prepared'].health_run)
        if 'observer' in held:result['trace']=held['observer'].trace
        raise
    finally:contract.write_json(out/'worker-result.json',result)


def reader(out,expected):
    # Public reader finishes M before importing Torch for T. No preloaded heavy
    # helper/environment lookup before it; live native check occurs after load.
    require(not any(n in sys.modules for n in ('torch','numpy','scene','gaussian_renderer')),'reader_preloaded')
    contract.verify_candidate(expected)
    w=strict_json((out/'worker-result.json').read_bytes(),64*1024**2)
    require(w['status']=='passed','worker_required')
    row=w['trace']['checkpoints'][-1]
    import formal_checkpoint as cp
    result=dict(status='running',stage='M_T')
    try:
        loaded=cp.read_checkpoint(Path(row['path']))
        require(loaded['marker_identity']==row['marker'],'marker_preserved')
        import torch
        import torch.utils.cpp_extension as ext
        import formal_native_bootstrap as native
        from formal_checkpoint_state import restore_diagnostic,equal_state
        from checkpoint_cuda_acceptance import diagnostic_camera,guard_pointops
        from training_health_cuda_acceptance import guard_reader_training
        from test_formal_checkpoint import assert_expected,SLOTS
        import unittest
        require(environment()==w['environment'],'same_environment')
        evidence=NativeEvidence(canonical(w['native']));validate_native(evidence)
        require(file_identity(out/'expected.pt')==w['expected_identity'],'expected_snapshot_identity')
        saved=torch.load(out/'expected.pt',map_location='cpu',weights_only=True)
        assert_expected(unittest.TestCase(),(saved['attrs'],saved['groups']),loaded)
        count=[]
        with ExitStack() as stack:
            guard_pointops(stack);native._load_knn(Path(w['native']['knn']['binary']['path']))
            def exact_load(*a,**kw):
                require(not a and kw==dict(name='diff_gaussian_rasterization',sources=[str(ROOT/u) for u in RASTER_UNITS],
                    extra_cuda_cflags=['-I '+str(ROOT/'diff-gaussian-rasterization/third_party/glm')+'/', '-g'],verbose=True),'reader_load_request')
                require(not count,'load_once');count.append(1)
                spec=importlib.util.spec_from_file_location('diff_gaussian_rasterization',w['native']['raster']['binary']['path'])
                mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
            stack.enter_context(patch.object(ext,'load',exact_load))
            import gaussian_renderer
            guard_reader_training(stack)
            model=restore_diagnostic(loaded,'cuda');require(model.optimizer is None,'no_resume')
            for slot,attr in SLOTS.items():equal_state(getattr(model,attr).detach().cpu(),saved['attrs'][slot])
            require(len(saved['views'])==len(expected['representatives'])==3,'representative_count')
            for view,rep in zip(saved['views'],expected['representatives']):
                split,index,path,stamp=rep;cam=diagnostic_camera(loaded['config'],(split,index,path))
                require(camera_values(cam)==view['camera'] and cam.formal_frame.raw_time==stamp,'representative_camera')
                equal_state(view['output'],render_buffers(gaussian_renderer.render,cam,model,loaded['config']))
            require(count==[1],'reader_load_count');validate_native(evidence,live=True)
        contract.verify_candidate(expected)
        result.update(status='passed',stage='completed',checkpoint=row,representatives=expected['representatives'],
            parameter_groups=sorted(contract.TOLERANCES_GROUPS),exact=True,reader_builds=0,forwards=3)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:contract.write_json(out/'reader-result.json',result)


def child(mode,out,timeout):
    with (out/(mode+'.stdout.log')).open('xb') as stdout,(out/(mode+'.stderr.log')).open('xb') as stderr:
        p=subprocess.Popen([contract.PYTHON,'-B',str(Path(__file__).resolve()),mode,'--out',str(out)],
                           stdout=stdout,stderr=stderr,start_new_session=True)
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.wait()
            contract.write_json(out/(mode+'-exit.json'),dict(mode=mode,exit_code=p.returncode,timed_out=True))
            raise
    contract.write_json(out/(mode+'-exit.json'),dict(mode=mode,exit_code=code))
    require(code==0,'child_failed_'+mode)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('operation',choices=('record','run','worker','reader'))
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--candidate',type=Path)
    args=parser.parse_args(argv);require(sys.executable==contract.PYTHON,'python')
    if args.operation=='record':contract.write_json(args.out,contract.candidate());return
    if args.operation in ('worker','reader'):
        expected=strict_json((args.out/'inputs.json').read_bytes())
        require(args.out==Path(expected['evidence_output']),'child_destination')
        (worker if args.operation=='worker' else reader)(args.out,expected);return
    require(args.candidate is not None,'candidate_required')
    expected=strict_json(args.candidate.read_bytes());contract.preflight(expected,args.out)
    resources(args.out);args.out.mkdir(mode=0o700);contract.write_json(args.out/'inputs.json',expected)
    start=time.monotonic();result=dict(status='running',stage='worker')
    try:
        child('worker',args.out,contract.RESOURCES['worker_seconds']);result['stage']='reader'
        remaining=contract.RESOURCES['wall_seconds']-(time.monotonic()-start)
        require(remaining>0,'wall_budget');child('reader',args.out,min(remaining,contract.RESOURCES['reader_seconds']))
        contract.validate_results(args.out)
        result.update(status='passed',stage='completed',seconds=time.monotonic()-start)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:contract.finalize(args.out,result)


if __name__=='__main__':main()
