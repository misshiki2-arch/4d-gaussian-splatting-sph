"""Future Step 14 acceptance. record/preflight are stdlib-only, not GPU runs.

run requires separate authorization and a new output. One ordinary entry
builds both extensions; subsequent cases and the reader use those binaries.
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
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(Path(__file__).parent)]
import checkpoint_cuda_acceptance as base
from training_update_cuda_acceptance import result_payload
from formal_checkpoint_identity import (NativeEvidence,canonical,file_identity,require,
    strict_json,validate_native,RASTER_UNITS,GROUPS)
PYTHON=base.PYTHON
REPORTS=ROOT.parent/'reports/corrected-4dgs/phase1/step14'
EXTRA=('tests/cuda/training_health_cuda_acceptance.py','tests/training_health_runtime.py',
       'tests/test_training_health.py','tests/test_training_health_preflight.py',
       'tests/test_training_health_reader.py',
       'tests/cuda/training_update_cuda_acceptance.py')
CASES=('ordinary','population','invisible','loss','gradient','post_adam','empty_population','evaluation')


def write_json(path,value):
    # Reuse Step 12's lossless normalization and Step 13's exclusive/fsync
    # writing. Canonical comparison detects changed leaf types as well as values.
    payload=result_payload(value)
    with path.open('xb') as stream:
        stream.write(canonical(payload));stream.flush();os.fsync(stream.fileno())
    actual=strict_json(path.read_bytes(),64*1024**2)
    require(canonical(actual)==canonical(payload),'result_readback')
    return file_identity(path)


def compare_parameters(model,capture):
    from formal_checkpoint_state import equal_state
    require(len(GROUPS)==9,'required_parameter_groups')
    compared=[]
    for name,(slot,attr) in GROUPS.items():
        equal_state(getattr(model,attr).detach().cpu(),capture[slot])
        compared.append(name)
    require(len(compared)==9 and set(compared)==set(GROUPS),'parameter_coverage')
    require(model.optimizer is None,'reader_not_training')
    return sorted(compared)


def guard_reader_training(stack):
    import torch
    from scene.gaussian_model import GaussianModel
    def forbidden(*a,**kw):raise AssertionError('reader_training_or_KNN')
    stack.enter_context(patch.object(GaussianModel,'training_setup',forbidden))
    stack.enter_context(patch.object(torch.optim.Adam,'load_state_dict',forbidden))
    # Preserve the required function identity even while forbidding calls.
    stack.enter_context(patch.object(sys.modules['scene.gaussian_model'],'distCUDA2',forbidden))
    stack.enter_context(patch.object(sys.modules['simple_knn._C'],'distCUDA2',forbidden))


def candidate():
    result=base.candidate();result['schema']='step14-health-inputs-v1'
    result['files'].update({n:file_identity(ROOT/n) for n in EXTRA})
    return result


def verify_candidate(expected):
    require(expected==candidate(),'reviewed_inputs')


def preflight(expected,out):
    verify_candidate(expected)
    require(sys.executable==PYTHON,'python')
    require(out.is_absolute() and out.resolve()==out and not out.exists() and
            REPORTS in out.parents and out.parent.is_dir(),'new_report_output')
    require(not any(n in sys.modules for n in ('torch','train','scene','gaussian_renderer')),'preloaded')
    require(not any(k.startswith('STEP90_CUDA_DEBUG') and v for k,v in os.environ.items()),'debug_override')


def worker(out,expected):
    import torch
    import formal_entry,formal_inputs,formal_native_bootstrap as native,formal_checkpoint as cp
    import training_update_runtime as runtime
    import training_health_runtime as hr
    from test_formal_config import encode
    torch.set_num_threads(1)
    result=dict(status='running',stage='ordinary',cases=[],checkpoints=[],environment=base.gpu_identity(torch))
    held={};budget=dict(forwards=0)
    try:
        verify_candidate(expected)
        with ExitStack() as stack:
            result['pointops']=base.guard_pointops(stack)
            real_bootstrap=native.bootstrap
            def bootstrap(source):
                train,evidence=real_bootstrap(source)
                held.update(train=train,evidence=evidence,render=train.render,
                            continuation=train.continue_formal_training)
                def continuation(p,writer):
                    row=hr.exercise(p,held['render'],'normal',writer,held['continuation'],budget)
                    row['name']='ordinary';result['cases'].append(row)
                    result['checkpoints'].extend(row['checkpoints'])
                    return row['outcome']['returned']
                stack.enter_context(patch.object(train,'continue_formal_training',continuation))
                return train,evidence
            first=next(c for c in runtime.CASES if c['name']=='G0a')
            _,ref=runtime.create_fixture(out/'ordinary',first)
            with patch.object(formal_inputs,'_reference',return_value=ref),patch.object(native,'bootstrap',bootstrap):
                require(formal_entry.run(out/'ordinary/config.json')==1,'ordinary_return')
            result['native']=held['evidence'].value()
            train=held['train']
            for name in CASES[1:]:
                result['stage']=name
                if name=='population':case=next(c for c in runtime.CASES if c['name']=='G4')
                else:case=dict(name=name,n=3,batch=2,kind='mixed' if name=='empty_population' else 'initial')
                value,ref=runtime.create_fixture(out/name,case)
                value['checkpoint']['intermediate_updates']=[1,2]
                value['reporting']['test_updates']=[2,3]
                with patch.object(formal_inputs,'_reference',return_value=ref):
                    p=train.prepare_formal_runtime(formal_inputs.verify_inputs(encode(value)),native_evidence=held['evidence'])
                    runtime.initialize_population(p,case)
                    if name=='invisible':
                        with torch.no_grad():p.gaussians._xyz[:,2]=-10.
                    claim=formal_entry.OutputClaim(p.inputs);writer=None
                    try:
                        writer=train.begin_formal_output(p,claim)
                        if name=='population':
                            obs=runtime.Observation(p,case,held['render'],budget)
                            def save(real,k):
                                from test_formal_checkpoint import independent,assert_expected
                                import unittest
                                before=independent(p.gaussians);receipt=real(k)
                                path=Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k))
                                loaded=cp.read_checkpoint(path);assert_expected(unittest.TestCase(),before,loaded)
                                result['checkpoints'].append(dict(k=k,path=str(path),marker=loaded['marker_identity'],reference=ref))
                                return receipt
                            try:trace=obs.run(writer,checkpoint_observer=save,continue_call=held['continuation'])
                            finally:write_json(out/'population-trace.json',runtime.serializable(obs.trace))
                            row=dict(name=name,status='passed',outcome=dict(run='returned',returned=3),steps=len(trace['steps']))
                        else:
                            row=hr.exercise(p,held['render'],name,writer,held['continuation'],budget)
                            row['name']=name;result['checkpoints'].extend(row['checkpoints'])
                        result['cases'].append(row)
                        if writer is not None:writer.flush()
                    finally:
                        if writer is not None:writer.close()
                        claim.close()
            require([r['name'] for r in result['cases']]==list(CASES),'coverage')
            validate_native(held['evidence'],live=True);verify_candidate(expected)
            result.update(status='passed',stage='completed',forwards=budget['forwards'])
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:write_json(out/'worker-result.json',result)


def reader(out,expected):
    """Public reader in a separate process, exact saved extensions, no build."""
    import formal_checkpoint as cp,formal_inputs,formal_native_bootstrap as native
    result=strict_json((out/'worker-result.json').read_bytes(),64*1024**2)
    require(result['status']=='passed','worker_required');verify_candidate(expected)
    # Validate bytes/recipe first; live module identity is only meaningful
    # after the saved binaries have been loaded in this fresh reader process.
    evidence=NativeEvidence(canonical(result['native']));validate_native(evidence)
    import torch
    import torch.utils.cpp_extension as ext
    require(base.gpu_identity(torch)==result['environment'],'same_environment')
    count=[];rows=[]
    with ExitStack() as stack:
        base.guard_pointops(stack);native._load_knn(Path(result['native']['knn']['binary']['path']))
        def exact_load(*a,**kw):
            require(not a and kw==dict(name='diff_gaussian_rasterization',sources=[str(ROOT/n) for n in RASTER_UNITS],
                extra_cuda_cflags=['-I '+str(ROOT/'diff-gaussian-rasterization/third_party/glm')+'/', '-g'],verbose=True), 'reader_load_request')
            require(not count,'reader_once');count.append(1)
            spec=importlib.util.spec_from_file_location('diff_gaussian_rasterization',result['native']['raster']['binary']['path'])
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
        stack.enter_context(patch.object(ext,'load',exact_load))
        import gaussian_renderer
        from formal_checkpoint_state import restore_diagnostic
        guard_reader_training(stack)
        for row in result['checkpoints']:
            with patch.object(formal_inputs,'_reference',return_value=row['reference']):
                read=cp.read_checkpoint(Path(row['path']))
                require(read['marker_identity']==row['marker'],'marker_preserved')
                model=restore_diagnostic(read,'cuda')
                compared=compare_parameters(model,read['state']['capture'])
                rows.append(dict(path=row['path'],k=row['k'],marker=read['marker_identity'],parameters=compared))
        require(count==[1] and rows,'reader_coverage')
    validate_native(evidence,live=True);verify_candidate(expected)
    write_json(out/'reader-result.json',dict(status='passed',checkpoints=rows))


def validate_results(out):
    worker_result=strict_json((out/'worker-result.json').read_bytes(),64*1024**2)
    reader_result=strict_json((out/'reader-result.json').read_bytes(),64*1024**2)
    require(worker_result['status']==reader_result['status']=='passed','child_status')
    require([r['name'] for r in worker_result['cases']]==list(CASES) and
            all(r['status']=='passed' for r in worker_result['cases']),'case_coverage')
    for row in worker_result['cases']:
        require(row['outcome']['run']==('returned' if row['name'] in CASES[:3] else 'failed_expected'),'run_outcome')
    expected=[dict({k:r[k] for k in ('path','k','marker')},parameters=sorted(GROUPS))
              for r in worker_result['checkpoints']]
    require(expected and canonical(reader_result['checkpoints'])==canonical(expected),'reader_coverage')


def finalize_result(out,result):
    """A checks-passed file alone is not completion. No retry or replacement."""
    payload=dict(result,status='checks_passed' if result['status']=='passed' else 'failed',
                 finalization='requires_verified_receipt_and_exit_zero')
    receipt=write_json(out/'result.json',payload)
    write_json(out/'receipt.json',dict(result=receipt,status=result['status']))
    verify_completion(out,allow_failure=result['status']!='passed')


def verify_completion(out,*,allow_failure=False):
    payload=strict_json((out/'result.json').read_bytes(),64*1024**2)
    receipt=strict_json((out/'receipt.json').read_bytes())
    require(set(receipt)=={'result','status'} and receipt['result']==file_identity(out/'result.json'),'receipt_binding')
    require(payload.get('finalization')=='requires_verified_receipt_and_exit_zero','finalization')
    if allow_failure:
        require(payload['status']==receipt['status']=='failed','failure_receipt')
    else:
        require(payload['status']=='checks_passed' and receipt['status']=='passed','completion_status')
        validate_results(out)
        for name in ('worker','reader'):
            require(strict_json((out/(name+'-exit.json')).read_bytes())==dict(mode=name,exit_code=0),'child_exit')
    return payload


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
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--candidate',type=Path)
    args=parser.parse_args(argv);require(sys.executable==PYTHON,'python')
    if args.operation=='record':write_json(args.out,candidate());return
    if args.operation in ('worker','reader'):
        expected=strict_json((args.out/'inputs.json').read_bytes())
        (worker if args.operation=='worker' else reader)(args.out,expected);return
    require(args.candidate is not None,'candidate_required')
    expected=strict_json(args.candidate.read_bytes());preflight(expected,args.out)
    args.out.mkdir(mode=0o700);write_json(args.out/'inputs.json',expected)
    start=time.monotonic();result=dict(status='running',stage='worker')
    try:
        child('worker',args.out,1800);result['stage']='reader'
        remaining=1800-(time.monotonic()-start);require(remaining>0,'wall_budget')
        child('reader',args.out,remaining);validate_results(args.out);verify_candidate(expected)
        result.update(status='passed',stage='completed',seconds=time.monotonic()-start,
            files={p.name:file_identity(p) for p in args.out.iterdir() if p.is_file()})
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__);raise
    finally:
        finalize_result(args.out,result)


if __name__=='__main__':main()
