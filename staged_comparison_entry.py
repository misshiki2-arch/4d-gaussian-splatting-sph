"""Fixed four-stage comparison parent; preparation is not live authorization."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import sys
import time

if __name__=='__main__':
    # The follow-up invokes this fixed source file with -I. Make only its own
    # verified repository importable and retain the ordinary module identity.
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    sys.modules['staged_comparison_entry']=sys.modules[__name__]

from formal_checkpoint_identity import file_identity, require, source_snapshot, verify_source
from comparison_profile import STAGES, stage_config
import staged_comparison_checkpoint as cp
import staged_comparison_results as results
import staged_comparison_support as s

CANDIDATE=s.HERE/'step18-fix3-cuda-input-candidate.json'
PROGRAM=s.HERE/'four-stage-run'
ENTRY=s.HERE/'step18-fix3-entry.py'
A_PROGRAM=s.OLD/'four-stage-run'


def followup(value):
    from staged_comparison_followup import SCHEMA
    return value.get('schema')==SCHEMA


def stage_paths(program=PROGRAM):
    return [dict(name=name,run=str(program/name/'run'),output=str(program/name/'output')) for name in STAGES]


def a_bindings():
    program=cp._read_record(A_PROGRAM/'completion/result.json')
    require(program['status']=='checks_passed_requires_quality_review' and
        [v['stage'] for v in program['stages']]==list(STAGES),'A_program')
    files={str(A_PROGRAM/'completion/result.json'):file_identity(A_PROGRAM/'completion/result.json')}
    for item in program['stages']:
        run=A_PROGRAM/item['stage']/'run'
        for name,identity in (('completion/result.json',item['result']),('verify-exit.json',item['verify'])):
            require(file_identity(run/name)==identity,'A_completion_identity');files[str(run/name)]=identity
        result=cp._read_record(run/'completion/result.json')
        require(result['status']=='checks_passed' and result['stage']['name']==item['stage'],'A_result')
        for mode,expected in result['evidence'].items():
            p=run/(mode+'-result.json');require(file_identity(p)==expected,'A_child_identity');files[str(p)]=expected
        config=s.OLD/('step17-fix3-'+item['stage'].replace('_','-')+'-config.json')
        files[str(config)]=file_identity(config)
    return files


def initial_binding():
    # Reuse saved stat/hash evidence without reloading or rehashing a huge init.
    evidence=cp._read_record(s.REPORTS/'step17/step17-design2/step17-design2-design-evidence.json')
    rows=[r for r in evidence['material']['selected_existing_artifacts'] if r['path']==cp.INITIAL_PATH]
    require(len(rows)==1,'initial_evidence');row=rows[0];st=Path(cp.INITIAL_PATH).stat()
    require(dict(size=st.st_size,mtime_ns=st.st_mtime_ns,ino=st.st_ino,mode=st.st_mode)==row['stat'],'initial_stat')
    old=row['prior_tensor_identity']
    return dict(kind='initial_dict',path=cp.INITIAL_PATH,identity=dict(size=old['bytes'],sha256=old['sha256']),
                stage=None,completed=0,verification=None)


def plan():
    from formal_inputs import _reference
    ref=_reference();origin=cp._trusted_native()
    prov=dict(native_origin=origin,connection_source=source_snapshot(),environment=origin['environment'])
    cp.provenance(prov)
    return dict(schema='step18-four-stage-v1',execution_authorized=False,executable=False,review_required=True,
        program=str(PROGRAM),stages=stage_paths(),initial=initial_binding(),resources=s.RESOURCES,
        provenance=prov,execution_files=s.execution_files(),a_files=a_bindings(),
        representatives=[['train',i,ref['frames']['train'][i][0],ref['frames']['train'][i][1]] for i in s.VIEWS],
        profiles=[cp.contract(stage_config(x['name'],x['output'])) for x in stage_paths()],
        environment='4dgs310',unverified=['live capacity','exact native load','CUDA','four-stage completion'],
        counts=dict(A_optimizer=65992,B_optimizer=66000),step18_accepted=False,gate_b_passed=False)


def authorized(value):
    require(value.get('execution_authorized') is True and value.get('executable') is True,
            'separate_live_authorization_required')


def bind(value, *, absent=False):
    authorized(value)
    from staged_comparison_followup import CANDIDATE as NEXT, PROGRAM as DEST, paths, check
    continued=followup(value)
    fixed=cp._read_record(NEXT if continued else CANDIDATE)
    flags=('execution_authorized','executable')
    require({k:v for k,v in value.items() if k not in flags}==
            {k:v for k,v in fixed.items() if k not in flags},'fixed_candidate')
    program=DEST if continued else PROGRAM
    stages=paths() if continued else stage_paths()
    require(value['program']==str(program) and value['stages']==stages,'fixed_paths')
    s.check_execution_files(value['execution_files']);cp.provenance(value['provenance'])
    require(value['a_files']==a_bindings() and value['resources']==s.RESOURCES,'bound_A_resources')
    require(value['profiles']==[cp.contract(stage_config(x['name'],x['output'])) for x in stages],'fixed_profiles')
    if continued:check(value)
    if absent:require(not program.exists() and not program.is_symlink(),'exclusive_program')
    else:require(cp._read_record(program/'authorized-input.json')==value,'authorized_input')


def parent_binding(value,index):
    if followup(value):
        from staged_comparison_followup import previous
        return previous(value,index)
    if index==0:
        cp.parent_binding(value['initial'],STAGES[0]);return value['initial']
    run=Path(value['stages'][index-1]['run'])
    return results.parent_from_receipt(results.reference(run/'completion/result.json'),STAGES[index])


@contextmanager
def user_stop():
    def stop(signum,frame):raise RuntimeError('user_stop_signal:'+str(signum))
    previous={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)


def child_command(mode,input_path,index, *, continued=False):
    path=s.ROOT/'staged_comparison_entry.py' if continued else ENTRY
    return [s.PYTHON,'-I','-B',str(path),mode,str(input_path),str(index)]


def run(value,input_path):
    authorized(value);bind(value,absent=True)
    program=Path(value['program']);program.mkdir();phase='input';dispatched=[];start=time.monotonic()
    try:
        with user_stop():
            s.write_json(program/'authorized-input.json',value)
            phase='start_resources';observed=s.resources().start(program)
            s.write_json(program/'start-resources.json',observed)
            require(observed['status']=='passed','start_resources')
            if followup(value):
                from staged_comparison_followup import reused
                receipts=reused(value)
            else:receipts=[]
            for i,stage in enumerate(value['stages']):
                phase=stage['name']+':parent';parent=parent_binding(value,i)
                out=Path(stage['run']);out.mkdir(parents=True)
                s.write_json(out/'parent.json',parent)
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                for key in list(env):
                    if key.startswith('GIT_') or key in ('PYTHONHOME','PYTHONPATH','TORCH_EXTENSIONS_DIR'):env.pop(key)
                for mode in ('worker','reader','verify'):
                    phase=stage['name']+':'+mode;dispatched.append(phase)
                    # The reviewed monitor owns session identity, resources,
                    # stdout/stderr, exit collection and stop records.
                    s.monitor().monitored_child(child_command(mode,input_path,i,continued=followup(value)),cwd=s.ROOT,env=env,
                        out=out,mode=mode,filesystem=program)
                    s.process_record(out,mode)
                phase=stage['name']+':comparison'
                from staged_comparison_images import compare
                a_run=A_PROGRAM/stage['name']/'run';a=cp._read_record(a_run/'worker-result.json')
                w=cp._read_record(out/'worker-result.json')
                s.resources().memory(out,'comparison',w['final_points'])
                s.resources().disk(out,'comparison',w['final_points'])
                phase=stage['name']+':finalize'
                receipt=results.finalize(out,lambda worker,dest:compare(worker,dest,a,a_run,
                    value['profiles'][i]['paths']['source_path']))
                receipts.append(receipt)
            phase='program_finalization';pending=program/'.pending-result';pending.mkdir()
            if followup(value):
                # Revalidate the reused publication; never relabel its failed
                # old four-stage program, nor count only three new receipts.
                require(reused(value)==receipts[:1] and len(receipts)==4,'followup_completion')
            for stage in value['stages']:results.verify_publication(Path(stage['run']))
            s.write_json(pending/'result.json',dict(status='checks_passed_requires_quality_review',stages=receipts,
                seconds=time.monotonic()-start,counts=value['counts'],step18_accepted=False,gate_b_passed=False))
            s.write_json(pending/'receipt.json',dict(result=file_identity(pending/'result.json')))
            pending.rename(program/'completion')
    except BaseException as error:
        record=dict(status='failed',phase=phase,error_type=type(error).__name__,reason=str(error)[:512],
                    dispatched=dispatched,seconds=time.monotonic()-start,downstream_not_run=True)
        try:s.write_json(program/'failure.json',record)
        except BaseException as secondary:print('FAILURE_RECORD_UNAVAILABLE',record,type(secondary).__name__,file=sys.stderr)
        raise


def main():
    require(sys.executable==s.PYTHON,'python')
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('mode',choices=('run','worker','reader','verify'))
    parser.add_argument('input',type=Path);parser.add_argument('index',nargs='?',type=int)
    args=parser.parse_args();value=cp._read_record(args.input)
    authorized(value)  # Before resource queries, imports, output or worker spawn.
    if args.mode=='run':
        require(args.index is None,'run_index');return run(value,args.input.resolve())
    bind(value);require(type(args.index) is int and 0<=args.index<len(value['stages']),'stage_index')
    stage=dict(value['stages'][args.index]);parent=parent_binding(value,args.index)
    require(cp._read_record(Path(stage['run'])/'parent.json')==parent,'parent_record')
    stage['initial_points']=1326947 if parent['kind']=='initial_dict' else results.checkpoint_marker(
        dict(path=parent['path'],identity=parent['identity']),parent['stage'])['rows']
    if args.mode=='verify':return results.verify_child(Path(stage['run']))
    require(not any(n in sys.modules for n in ('torch','train','scene','gaussian_renderer')),'cold_child')
    sys.path[:0]=[str(s.ROOT/'tests'),str(s.ROOT/'tests/cuda')]
    from staged_comparison_worker import worker,reader
    with user_stop():
        (worker if args.mode=='worker' else reader)(value,stage,parent)
    bind(value)


if __name__=='__main__':main()
