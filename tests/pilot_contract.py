"""Run-local Step 16 evidence, not configuration authority. Stdlib only."""
import math
import os
from pathlib import Path
import subprocess
import sys

from formal_checkpoint_identity import canonical, file_identity, require, source_snapshot, strict_json
from real_input_contract import frame_key, write_json, TOLERANCES_GROUPS

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
STEP = ROOT.parent / 'reports/corrected-4dgs/phase3/step16'
CONFIG = STEP / 'step16-design2/step16-design2-design-run-config-candidate.json'
RUN = STEP / 'step16-validation2/step16-validation2-run'
BASELINE = STEP / 'step16-validation1/step16-validation1-run'
SOURCE_CHANGE = STEP / 'step16-fix2/step16-fix2-source-change.json'
COMPARE = STEP / 'step16-fix2/step16-fix2-compare.py'
DIAGNOSTICS = dict(initial=0, reset=3000, prune=3100)
REPRESENTATIVES = (0, 15, 30, 2573, 2588, 2603, 5115, 5130, 5145)
PROBES = (1, 601, 1001, 2001, 3001, 4001, 4901, 5001, 6000)
GIB = 1024 ** 3
RESOURCES = dict(start_host=36*GIB, start_gpu=12*GIB, start_disk=20*GIB,
                 rss_limit=32*GIB, cuda_limit=13*GIB, host_floor=4*GIB,
                 disk_floor=8*GIB, interval=100, wall_seconds=28800,
                 worker_seconds=27000, reader_seconds=1200)


def schedule(raw):
    """Expected major calls from the sole JSON; no training implementation."""
    o = raw['optimization']; p = o['population']; n = o['total_updates']
    sh = []; spatial = temporal = 0
    for k in range(o['sh']['increase_interval'], n+1, o['sh']['increase_interval']):
        before = [spatial, temporal]
        if spatial < raw['model']['spatial_sh_degree']: spatial += 1
        elif temporal < raw['model']['temporal_sh_degree']: temporal += 1
        sh.append(dict(k=k, before=before, after=[spatial, temporal]))
    population = [k for k in range(1, n+1) if p['densify_from_update'] < k < p['densify_until_update']
                  and k % p['densification_interval'] == 0]
    reset = [k for k in range(1, n+1) if k < p['densify_until_update'] and
             (k % p['opacity_reset_interval'] == 0 or
              (raw['dataset']['white_background'] and k == p['densify_from_update']))]
    return dict(updates=n, batch=o['batch_size'], sh=sh, population=population, reset=reset,
                saves=sorted(set(raw['checkpoint']['intermediate_updates']+[n])),
                tests=raw['reporting']['test_updates'],
                growth_threshold=p['densify_stop_points'], reset_interval=p['opacity_reset_interval'])


def condition_changes(base, new, path=''):
    require(type(base) is type(new), 'comparison_field_type')
    if type(base) is dict:
        require(base.keys()==new.keys(), 'comparison_keys')
        return [v for k in base for v in condition_changes(base[k],new[k],path+'/'+k)]
    if type(base) is list:
        require(len(base)==len(new), 'comparison_length')
        return [v for i,(a,b) in enumerate(zip(base,new)) for v in condition_changes(a,b,path+'/'+str(i))]
    return [] if base==new else [dict(path=path,before=base,after=new)]


def verify_comparison_source(files):
    """One fixed reviewed consumer/test delta; no broad production exemption."""
    record=strict_json(SOURCE_CHANGE.read_bytes())
    require(set(record)=={'files','instruction'} and set(record['files'])==
            {'formal_inputs.py','tests/test_formal_connection.py'}, 'comparison_source_membership')
    require(record['instruction']==file_identity(STEP/'step16-fix2/step16-fix2-instruction.txt'), 'comparison_source_authority')
    adjusted={}
    for name,change in record['files'].items():
        path=STEP/'step16-fix2'/('step16-fix2-source-diff-'+name.replace('/','--')+'.txt')
        require(set(change)=={'before','after','diff'} and change['before']==files[name] and
                change['after']==file_identity(ROOT/name) and change['before']!=change['after'] and
                change['diff']==dict(path=str(path),identity=file_identity(path)), 'comparison_source_delta')
        adjusted[name]=change['after']
    mutable={'tests/pilot_contract.py','tests/pilot_runtime.py','tests/cuda/pilot_cuda_acceptance.py','tests/test_pilot_acceptance.py'}
    require(all(file_identity(ROOT/p)==adjusted.get(p,i) for p,i in files.items() if p not in mutable), 'comparison_production_source')


def comparison_binding(raw):
    old=strict_json((BASELINE/'inputs.json').read_bytes(),64*1024**2)
    config=Path(old['config']['path'])
    require(file_identity(config)==old['config']['identity'], 'baseline_config_identity')
    changes=condition_changes(strict_json(config.read_bytes()),raw)
    require(changes==[dict(path='/initialization/time_variance_denominator',before=5.0,after=33.0),
        dict(path='/output/directory',before=old['output'],after=str(ROOT.parent/'outputs/sph_scene_4dgs_corrected_step16_timewidth33_pilot1'))], 'comparison_two_leaves')
    require(old['resources']==RESOURCES and old['schedule']==schedule(raw), 'comparison_budget_schedule')
    verify_comparison_source(old['files'])
    result=strict_json((BASELINE/'result.json').read_bytes())
    require(result['status']=='checks_passed' and strict_json((BASELINE/'receipt.json').read_bytes())==
            dict(status='passed',result=file_identity(BASELINE/'result.json')), 'baseline_receipt')
    paths=[BASELINE/n for n in ('inputs.json','result.json','receipt.json','worker-result.json','reader-result.json')]
    paths += [STEP/'step16-investigation1'/('step16-investigation1-'+n+'.json') for n in ('bindings','tensors','images')]
    paths += [COMPARE,SOURCE_CHANGE]
    return dict(records={str(p):file_identity(p) for p in paths},changes=changes,diagnostics=DIAGNOSTICS)


def candidate():
    raw = strict_json(CONFIG.read_bytes())
    ref = strict_json((ROOT/'formal_dataset_reference.json').read_bytes())
    files = source_snapshot()
    # Includes actual new WIP and transitive test-helper dependencies, not HEAD alone.
    files.update({str(p.relative_to(ROOT)): file_identity(p) for p in sorted((ROOT/'tests').rglob('*.py'))})
    return dict(schema='step16-pilot-v1', repository=str(ROOT),
        base_head=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        files=files, config=dict(path=str(CONFIG), identity=file_identity(CONFIG)),
        reference=file_identity(ROOT/'formal_dataset_reference.json'),
        output=raw['output']['directory'], evidence_output=str(RUN), resources=RESOURCES,
        schedule=schedule(raw), probes=list(PROBES), fixed_updates=[0,3000,5000,6000],
        representatives=[['train', i, ref['frames']['train'][i][0], ref['frames']['train'][i][1]] for i in REPRESENTATIVES],
        comparisons=dict(state='dtype_shape_bytes_exact', render='same_binary_exact',
                         quality='train_only_review_no_numeric_threshold', forward_estimate_not_gate=6738,
                         **comparison_binding(raw)))


def verify_candidate(expected):
    actual = candidate()
    require(set(expected)==set(actual) and all(expected[k]==actual[k] for k in actual if k!='base_head'),
            'reviewed_pilot_inputs')
    return actual


def preflight(expected, out):
    verify_candidate(expected)
    require(sys.executable==PYTHON, 'python')
    require(not any(n in sys.modules for n in ('torch','numpy','PIL','train','scene','gaussian_renderer')), 'preloaded')
    require(not any(k.startswith('STEP90_CUDA_DEBUG') and v for k,v in os.environ.items()), 'debug_override')
    for p in (out, Path(expected['output'])):
        require(p.is_absolute() and p.resolve()==p and not os.path.lexists(p), 'new_output')
    require(out==Path(expected['evidence_output']) and out.parent.is_dir(), 'evidence_destination')
    import formal_inputs
    c = formal_inputs.verify_inputs(CONFIG.read_bytes()).config
    require(c.paths.output_directory==expected['output'], 'output_binding')
    return c


def check_resources(values, *, initial):
    """Injected CPU values and future real measurements use the same budget."""
    r = RESOURCES
    require(values['host_available'] >= (r['start_host'] if initial else r['host_floor']), 'host_budget')
    require(bool(values['disks']) and all(v >= (r['start_disk'] if initial else r['disk_floor'])
                                        for v in values['disks'].values()), 'disk_budget')
    if initial:
        require(values['gpu_free'] >= r['start_gpu'], 'gpu_budget')
    else:
        require(values['rss'] <= r['rss_limit'] and values['cuda_allocated'] <= r['cuda_limit']
                and values['cuda_reserved'] <= r['cuda_limit'], 'worker_memory_budget')
    return values


def metrics(values, names=('l1','psnr','ssim','msssim')):
    require(set(values)==set(names), 'metrics_coverage')
    for name, numbers in values.items():
        require(type(numbers) is list and len(numbers)==(3 if name=='psnr' else 1), 'metric_shape')
        for value in numbers:
            if type(value) is dict:
                require(name=='psnr' and value==dict(positive_infinity=True, zero_mse=True), 'metric_infinity')
            else:
                require(type(value) in (int,float) and math.isfinite(value), 'metric_finite')


def validate_trace(t, spec):
    n = spec['updates']
    require(t['returned']==n and t['fetches']==n, 'completed_fetches')
    require(sum(v['count'] for v in t['visits'])==n*spec['batch'] and
            len({tuple(v['frame']) for v in t['visits']})==len(t['visits']) and
            all(v['frame'] in spec['train'] and type(v['count']) is int and v['count']>0 for v in t['visits']), 'train_visits')
    require(t['sh']==spec['sh'], 'SH_events')
    require([v['k'] for v in t['population']]==spec['population'], 'population_events')
    for v in t['population']:
        threshold = spec['growth_threshold']
        require(v['before']>0 and v['after']>0 and type(v['prune_only']) is bool and
                v['prune_only']==(threshold!='unlimited' and v['before']>=threshold) and
                v['size_threshold']==(20 if v['k']>spec['reset_interval'] else None), 'population_observation')
    require([v['k'] for v in t['reset']]==spec['reset'], 'reset_events')
    require([v['k'] for v in t['checkpoints']]==spec['saves'], 'save_events')
    require([v['k'] for v in t['evaluations']]==[0,*spec['tests']], 'evaluation_events')
    for v in t['evaluations']:
        require([x['frame'] for x in v['views']]==spec['evaluation'], 'evaluation_frames')
        for x in v['views']: metrics(x['metrics'])
    require([v['k'] for v in t['fixed']]==spec['fixed_updates'], 'fixed_events')
    for v in t['fixed']:
        require([x['frame'] for x in v['views']]==[r[:3] for r in spec['representatives']], 'fixed_frames')
        for x in v['views']:
            metrics(x['metrics'], ('l1','psnr','ssim'))
            require(set(x['images'])=={'gt','render','alpha','depth'}, 'fixed_images')
    require([v['k'] for v in t['probes']]==spec['probes'], 'probe_events')
    for v in t['probes']:
        require(set(v['groups'])==set(TOLERANCES_GROUPS), 'probe_groups')
        require(all(g['nonzero_gradient']>=0 and g['changed']>=0 and math.isfinite(g['lr'])
                    for g in v['groups'].values()), 'probe_values')
    # Major boundaries only. No all-update LR/backward/Adam/zero FSM.
    events = []
    for k in range(n+1):
        if any(v['k']==k for v in spec['sh']): events.append(['sh',k])
        if k in spec['population']: events.append(['population',k])
        if k in spec['reset']: events.append(['reset',k])
        if k==n: events.append(['fixed',k])
        if k in spec['saves']: events.append(['save',k])
        if k==0 or k in spec['tests']: events.append(['evaluation',k])
        if k in spec['fixed_updates'] and k!=n: events.append(['fixed',k])
    require(t['events']==events, 'major_event_order')
    return t


def learning_summary(trace):
    """No test-based selection, fixed threshold, early stop or acceptance claim."""
    first, last = trace['fixed'][0]['views'], trace['fixed'][-1]['views']
    deltas = [dict(frame=b['frame'], l1=b['metrics']['l1'][0]-a['metrics']['l1'][0],
                   ssim=b['metrics']['ssim'][0]-a['metrics']['ssim'][0]) for a,b in zip(first,last)]
    changed = sum(g['changed'] for v in trace['probes'] for g in v['groups'].values())
    concerns=[]
    if not changed: concerns.append('no_change_in_sampled_updates')
    if not any(v['l1']<0 or v['ssim']>0 for v in deltas): concerns.append('no_fixed_train_improvement_evidence')
    if not any(v['visible'] for v in last): concerns.append('final_fixed_views_all_invisible')
    if all(v['rgb_max']==0 for v in last): concerns.append('final_fixed_views_all_black')
    return dict(status='requires_advisor_review', concerns=concerns, per_view_delta=deltas,
                probe_zero_updates=[v['k'] for v in trace['probes'] if not any(g['changed'] for g in v['groups'].values())],
                test_used_for_selection=False, selected_final=trace['returned'])


def validate_diagnostics(d, spec, trace, out):
    require(type(d) is dict and set(d)=={'initial','reset','prune'}, 'diagnostic_membership')
    initial,reset,prune=(d[k] for k in ('initial','reset','prune'))
    require(initial['k']==0 and initial['stage']=='before_first_fetch' and type(initial['rows']) is int and initial['rows']>0, 'initial_diagnostic')
    widths=initial['widths']
    require(set(widths)=={'log_scale','scale','variance','sigma_tt','rotated_sigma'}, 'width_fields')
    for name,s in widths.items():
        require(s['count']==s['finite']==initial['rows'] and set(s)=={'count','finite','mean','quantiles'}, 'width_count')
        require(set(s['quantiles'])=={'min','p01','p10','p25','median','p75','p90','p99','max'}, 'width_quantiles')
        require(all(type(v) in (int,float) and math.isfinite(v) and v==(0 if name=='log_scale' else 1)
                    for v in [s['mean'],*s['quantiles'].values()]), 'initial_width_value')
    k=DIAGNOSTICS['reset']
    require(reset['attempted_update']==k and reset['completed']==k-1 and reset['stage']=='after_population_before_reset'
            and reset['after_fixed']==k and k in spec['reset'], 'reset_diagnostic')
    after=next(v for v in trace['fixed'] if v['k']==k)
    require([v['frame'] for v in reset['views']]==[r[:3] for r in spec['representatives']], 'reset_frames')
    for view,post,rep in zip(reset['views'],after['views'],spec['representatives']):
        require(view['raw_time']==post['raw_time']==rep[3] and view['images']['gt']==post['images']['gt'], 'reset_correspondence')
        metrics(view['metrics'],('l1','psnr','ssim'))
        require(set(view['images'])=={'gt','render','alpha','depth'}, 'reset_images')
        for name,i in view['images'].items():
            require(file_identity(out/'diagnostics'/('reset-'+str(k)+'-before')/(str(rep[1])+'-'+name+'.png'))==i,'reset_image_identity')
    k=DIAGNOSTICS['prune'];row=next(v for v in trace['population'] if v['k']==k)
    require(prune['attempted_update']==k and prune['stage']=='final_mask_after_growth' and prune['prune_only'] is row['prune_only']
            and prune['before']==row['before'] and prune['after']==row['after'] and prune['net']==row['after']-row['before'], 'prune_diagnostic')
    require(prune['split_parent_calls']==int(not prune['prune_only']), 'prune_parent_separation')
    bins=prune['bins'];require(type(bins) is list and len(bins)==8 and all(type(n) is int and n>=0 for n in bins), 'prune_bins')
    require(sum(bins)==prune['rows'] and prune['union']==sum(bins[1:]) and prune['kept']==bins[0]==prune['after'], 'prune_counts')
    require(prune['inclusive']=={name:sum(n for i,n in enumerate(bins) if i&bit) for name,bit in [('opacity',1),('screen',2),('world',4)]}, 'prune_inclusive')
    require(prune['overlaps']=={name:sum(n for i,n in enumerate(bins) if i&bit==bit) for name,bit in [('opacity_screen',3),('opacity_world',5),('screen_world',6),('triple',7)]}, 'prune_overlaps')
    thresholds=prune['thresholds']
    raw=strict_json(CONFIG.read_bytes())
    require(set(thresholds)=={'opacity','extent','screen'} and thresholds['opacity']==raw['optimization']['population']['thresh_opa_prune']
            and type(thresholds['extent']) in (int,float) and math.isfinite(thresholds['extent']) and thresholds['extent']>0
            and thresholds['screen']==row['size_threshold'], 'prune_thresholds')
    return d


def validate_results(out):
    expected = strict_json((out/'inputs.json').read_bytes(), 64*1024**2)
    verify_candidate(expected)
    w = strict_json((out/'worker-result.json').read_bytes(), 64*1024**2)
    r = strict_json((out/'reader-result.json').read_bytes(), 64*1024**2)
    require(w['status']==r['status']=='passed', 'child_status')
    spec = w['spec']; t = w['trace']; n = expected['schedule']['updates']
    require(all(spec[k]==v for k,v in expected['schedule'].items()) and
            all(spec[k]==expected[k] for k in ('probes','fixed_updates','representatives')), 'bound_spec')
    ref = strict_json((ROOT/'formal_dataset_reference.json').read_bytes())
    train = [['train',i,f[0]] for i,f in enumerate(ref['frames']['train'])]
    test = [['test',i,f[0]] for i,f in enumerate(ref['frames']['test'])]
    require(spec['train']==train and len(spec['evaluation'])==5+len(test) and
            len({tuple(v) for v in spec['evaluation'][:5]})==5 and
            all(v in train for v in spec['evaluation'][:5]) and sorted(spec['evaluation'][5:])==sorted(test), 'evaluation_population')
    validate_trace(t,spec)
    validate_diagnostics(w['pilot_diagnostics'],spec,t,out)
    d=w['diagnostic']; writer=w['writer']
    require(w['entry_calls']==1 and w['returned']==n and w['stage']=='normal_return' and
            d['completed']==d['attempted']==n and d['saves']==len(spec['saves']) and
            d['tests']==len(spec['tests']) and d['last_confirmed_save']==n, 'normal_completion')
    require(writer==dict(present=writer['present'],flush=int(writer['present']),close=int(writer['present'])), 'writer_completion')
    require(r['checkpoint']==t['checkpoints'][-1] and r['representatives']==spec['representatives'] and
            r['parameter_groups']==sorted(TOLERANCES_GROUPS) and r['exact'] is True and
            r['reader_builds']==0 and r['forwards']==len(spec['representatives']), 'reader_coverage')
    require(file_identity(out/'expected.pt')==w['expected_identity'], 'expected_identity')
    keys={tuple(v['frame']) for v in t['visits']}
    keys.update(tuple(v) for v in spec['evaluation'])
    keys.update(tuple(v[:3]) for v in spec['representatives'])
    paths={str(Path(ref['source_path'])/path) for split,i,_ in keys for path in ref['frames'][split][i][::2]}
    require(set(w['consumed'])==paths, 'consumed_coverage')
    for path,identity in w['consumed'].items(): require(file_identity(Path(path))==identity, 'consumed_changed')
    for row in t['checkpoints']:
        require(row['path']==str(Path(expected['output'])/'checkpoints'/('completed-'+str(row['k']))) and
                set(row['files'])=={'complete.json','metadata.json','state.pt'} and row['marker']==row['files']['complete.json'], 'checkpoint_identity')
        for name,identity in row['files'].items(): require(file_identity(Path(row['path'])/name)==identity, 'checkpoint_changed')
    for fixed in t['fixed']:
        for view in fixed['views']:
            for name,identity in view['images'].items():
                path=out/'fixed'/str(fixed['k'])/(str(view['frame'][1])+'-'+name+'.png')
                require(file_identity(path)==identity, 'fixed_image_identity')
    return w,r


def finalize(out, result):
    payload=dict(result, status='checks_passed' if result['status']=='passed' else 'failed',
                 finalization='requires_verified_receipt_and_exit_zero')
    if result['status']=='passed':
        payload['evidence']={n:file_identity(out/n) for n in ('inputs.json','worker-result.json','reader-result.json')}
    receipt=write_json(out/'result.json',payload)
    write_json(out/'receipt.json',dict(result=receipt,status=result['status']))
    if result['status']=='passed': verify_completion(out)


def verify_completion(out):
    result=strict_json((out/'result.json').read_bytes(),64*1024**2)
    receipt=strict_json((out/'receipt.json').read_bytes())
    require(receipt==dict(result=file_identity(out/'result.json'),status='passed') and
            result['status']=='checks_passed' and result['finalization']=='requires_verified_receipt_and_exit_zero', 'receipt')
    for name in ('worker','reader'):
        value=strict_json((out/(name+'-exit.json')).read_bytes())
        require(value['mode']==name and value['exit_code']==0 and value['timed_out'] is False, 'child_exit')
        logs=value.get('logs')
        require(type(logs) is dict and set(logs)=={'stdout','stderr'}, 'process_log_membership')
        for stream,identity in logs.items():
            require(file_identity(out/(name+'.'+stream+'.log'))==identity, 'process_log_identity')
    require(result['evidence']=={n:file_identity(out/n) for n in ('inputs.json','worker-result.json','reader-result.json')}, 'result_evidence_identity')
    validate_results(out)
    return result
