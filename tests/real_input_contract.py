"""Step 15 bounded evidence contract. Stdlib only; no training authority."""
import math
import os
from pathlib import Path
import subprocess
import sys

from formal_checkpoint_identity import canonical, file_identity, require, source_snapshot, strict_json

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
STEP = ROOT.parent / 'reports/corrected-4dgs/phase2/step15'
CONFIG = STEP / 'step15-design1/step15-design1-design-run-config-candidate.json'
RUN = STEP / 'step15-validation1/step15-validation1-run'
EXTRA = ('tests/real_input_contract.py', 'tests/real_input_runtime.py',
    'tests/test_real_input_acceptance.py', 'tests/cuda/real_input_cuda_acceptance.py',
    'tests/training_update_oracle.py', 'tests/test_formal_checkpoint.py',
    'tests/formal_training_fixture.py', 'tests/test_formal_connection.py', 'tests/test_formal_config.py',
    'tests/cuda/checkpoint_cuda_acceptance.py', 'tests/cuda/training_health_cuda_acceptance.py',
    'tests/cuda/training_update_cuda_acceptance.py')
REPRESENTATIVES = (('train', 0), ('test', 83), ('train', 5145))
RESOURCES = dict(host_available=8*1024**3, vram_free=6*1024**3, disk_free=8*1024**3,
                 wall_seconds=7200, worker_seconds=6600, reader_seconds=600)
TOLERANCES = dict(parameter=[1e-6,1e-5], moment=[1e-10,1e-5], variance=[1e-14,1e-5],
                  statistic=[1e-9,1e-5], opacity=[1e-8,1e-5])


def candidate():
    c = strict_json(CONFIG.read_bytes())
    ref = strict_json((ROOT/'formal_dataset_reference.json').read_bytes())
    files = source_snapshot()
    files.update({name:file_identity(ROOT/name) for name in EXTRA})
    return dict(schema='step15-real-input-v1', repository=str(ROOT),
        base_head=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        files=files, config=dict(path=str(CONFIG),identity=file_identity(CONFIG)),
        output=c['output']['directory'], evidence_output=str(RUN), resources=RESOURCES,
        representatives=[[split,index,ref['frames'][split][index][0],ref['frames'][split][index][1]]
                         for split,index in REPRESENTATIVES],
        budgets=dict(rows=c['initialization']['num_pts'],updates=c['optimization']['total_updates'],
                     forwards=351,backwards=3), tolerances=TOLERANCES)


def verify_candidate(expected):
    actual=candidate()
    require(set(expected)==set(actual) and all(expected[k]==actual[k] for k in actual if k!='base_head'),
            'reviewed_real_inputs')
    return actual


def preflight(expected, out):
    verify_candidate(expected)
    require(sys.executable==PYTHON,'python')
    require(not any(n in sys.modules for n in ('torch','numpy','PIL','train','scene','gaussian_renderer')),'preloaded')
    require(not any(k.startswith('STEP90_CUDA_DEBUG') and v for k,v in os.environ.items()),'debug_override')
    for path in (out,Path(expected['output'])):
        require(path.is_absolute() and path.resolve()==path and not os.path.lexists(path),'new_output')
    require(out==Path(expected['evidence_output']) and out.parent.is_dir(),'evidence_destination')
    # Public resolver, not a substitute config. Read-only P6; no dataset decode.
    import formal_config
    c=formal_config.resolve_formal_config(CONFIG.read_bytes(),raw_width=1280,raw_height=720)
    require(c.paths.output_directory==expected['output'],'output_binding')
    return c


def frame_key(camera):
    f=camera.formal_frame
    return [f.split,f.index,f.file_path]


def expected_events(n, saves, tests):
    result=[['report_begin',0],['report_end',0]]
    for k in range(1,n+1):
        result.extend([[name,k] for name in ('fetch','lr','render','backward','statistics','step','zero')])
        if k in saves:result.extend([['save_begin',k],['save_end',k]])
        result.extend([['report_begin',k],['report_end',k]])
    return result


def validate_trace(trace, spec):
    """Evidence, not an exit code or a worker's passed flag, determines coverage."""
    n=spec['updates']
    require(trace['events']==expected_events(n,spec['saves'],spec['tests']),'event_sequence')
    require(trace['returned']==n and trace['reports']==list(range(n+1)),'return_reports')
    require([s['k'] for s in trace['steps']]==list(range(1,n+1)),'steps')
    for row in trace['steps']:
        require(row['visible']>0 and row['nonzero_gradients']>0 and row['changed_positive_lr']>0,'real_update')
        require(set(row['lr'])==set(TOLERANCES_GROUPS) and row['rows']==spec['rows'],'groups_rows')
        require(len(row['comparisons'])==36 and
                {c['check'] for c in row['comparisons']}=={n+'/'+k for n in TOLERANCES_GROUPS for k in ('p','m','v','step')} and
                all(math.isfinite(c['max_bound_ratio']) and
                0<=c['max_bound_ratio']<=1 for c in row['comparisons']),'adam_comparisons')
    require(len(trace['batches'])==n and all(len(b)==1 and b[0] in spec['train'] for b in trace['batches']), 'train_only_fetch')
    require([r['k'] for r in trace['evaluations']]==[0,*spec['tests']],'evaluation_schedule')
    for evaluation in trace['evaluations']:
        require([r['frame'] for r in evaluation['views']]==spec['evaluation'],'evaluation_frames')
        for view in evaluation['views']:
            require(set(view['metrics'])=={'l1','psnr','ssim','msssim'},'metrics_coverage')
            for name, values in view['metrics'].items():
                require(type(values) is list and len(values)==(3 if name=='psnr' else 1),'metric_shape')
                for value in values:
                    if type(value) is dict:
                        require(name=='psnr' and value==dict(positive_infinity=True,zero_mse=True),'metric_infinity')
                    else:require(type(value) in (float,int) and math.isfinite(value),'metric_finite')
    require([r['k'] for r in trace['checkpoints']]==spec['saves'],'checkpoint_coverage')
    require(trace['forwards']==n+len(spec['evaluation'])*(1+len(spec['tests'])),'forward_coverage')
    return trace


TOLERANCES_GROUPS=('xyz','f_dc','f_rest','opacity','scaling','rotation','t','scaling_t','rotation_r')


def validate_results(out):
    w=strict_json((out/'worker-result.json').read_bytes(),64*1024**2)
    r=strict_json((out/'reader-result.json').read_bytes(),64*1024**2)
    require(w['status']==r['status']=='passed','child_status')
    validate_trace(w['trace'],w['spec'])
    expected=strict_json((out/'inputs.json').read_bytes())
    verify_candidate(expected)
    ref=strict_json((ROOT/'formal_dataset_reference.json').read_bytes())
    spec=w['spec']
    train=[[s,i,f[0]] for s in ('train',) for i,f in enumerate(ref['frames'][s])]
    test=[[s,i,f[0]] for s in ('test',) for i,f in enumerate(ref['frames'][s])]
    require(spec['updates']==3 and spec['rows']==expected['budgets']['rows'] and spec['saves']==[1,3]
            and spec['tests']==[3] and spec['train']==train,'real_spec')
    ev=spec['evaluation']
    require(len(ev)==171 and len({tuple(v) for v in ev[:5]})==5 and all(v in train for v in ev[:5])
            and sorted(ev[5:])==sorted(test),'real_evaluation_population')
    require(w['entry_calls']==1 and w['returned']==3 and w['stage']=='normal_return','entry_completion')
    require(w['writer']==dict(present=w['writer']['present'],flush=1 if w['writer']['present'] else 0,
                             close=1 if w['writer']['present'] else 0),'writer_completion')
    require(w['diagnostic']['completed']==w['diagnostic']['attempted']==3 and
            w['diagnostic']['saves']==2 and w['diagnostic']['tests']==1 and
            w['diagnostic']['last_confirmed_save']==3,'run_completion')
    require(w['representatives']==expected['representatives'] and r['representatives']==w['representatives']
            and r['checkpoint']==w['trace']['checkpoints'][-1] and r['parameter_groups']==sorted(TOLERANCES_GROUPS)
            and r['exact'] is True and r['reader_builds']==0 and r['forwards']==3,'reader_coverage')
    require(w['trace']['forwards']+w['representative_forwards']+r['forwards']==351,'total_forwards')
    require(file_identity(out/'expected.pt')==w['expected_identity'],'expected_identity')
    keys={tuple(v) for v in spec['evaluation']}
    keys.update(tuple(v) for batch in w['trace']['batches'] for v in batch)
    paths={str(Path(ref['source_path'])/path) for split,i,_ in keys for path in ref['frames'][split][i][::2]}
    require(set(w['consumed'])==paths,'consumed_coverage')
    for path,identity in w['consumed'].items():require(file_identity(Path(path))==identity,'consumed_changed')
    for row in w['trace']['checkpoints']:
        require(row['path']==str(Path(expected['output'])/'checkpoints'/('completed-'+str(row['k']))) and
                set(row['files'])=={'complete.json','metadata.json','state.pt'} and
                row['marker']==row['files']['complete.json'],'checkpoint_identity')
        for name,identity in row['files'].items():require(file_identity(Path(row['path'])/name)==identity,'checkpoint_changed')
    return w,r


def write_json(path,value):
    # Same lossless normalizer and exclusive/fsync/read-back as Steps 12–14.
    from training_update_cuda_acceptance import result_payload
    data=canonical(result_payload(value))
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    require(canonical(strict_json(Path(path).read_bytes(),64*1024**2))==data,'result_readback')
    return file_identity(Path(path))


def finalize(out,result):
    payload=dict(result,status='checks_passed' if result['status']=='passed' else 'failed',
                 finalization='requires_verified_receipt_and_exit_zero')
    receipt=write_json(out/'result.json',payload)
    write_json(out/'receipt.json',dict(result=receipt,status=result['status']))
    if result['status']=='passed':verify_completion(out)


def verify_completion(out):
    result=strict_json((out/'result.json').read_bytes(),64*1024**2)
    receipt=strict_json((out/'receipt.json').read_bytes())
    require(receipt==dict(result=file_identity(out/'result.json'),status='passed') and
            result['status']=='checks_passed' and result['finalization']=='requires_verified_receipt_and_exit_zero','receipt')
    for name in ('worker','reader'):
        require(strict_json((out/(name+'-exit.json')).read_bytes())==dict(mode=name,exit_code=0),'child_exit')
    validate_results(out)
    return result
