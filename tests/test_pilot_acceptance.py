"""Focused Step 16 CPU checks. Actual CPU consumers; only GPU seams isolated."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'tests/cuda')]
import pilot_contract as c
import pilot_cuda_acceptance as entry


def small_trace():
    raw=c.strict_json(c.CONFIG.read_bytes())
    raw['optimization'].update(total_updates=2,batch_size=1)
    raw['optimization']['population'].update(densify_from_update=0,densify_until_update=0)
    raw['optimization']['sh']['increase_interval']=1000
    raw['checkpoint']['intermediate_updates']=[]; raw['reporting']['test_updates']=[2]
    f=['train',0,'images/train0.png']
    spec=dict(c.schedule(raw),train=[f],evaluation=[f,['test',0,'images/test0.png']],
              representatives=[f+[0.]],probes=[1,2],fixed_updates=[0,2])
    def view(frame,fixed=False):
        row=dict(frame=frame,metrics=dict(l1=[.1],psnr=[10.,11.,12.],ssim=[.9]))
        if fixed:row.update(images={n:{} for n in ('gt','render','alpha','depth')},visible=1,rgb_max=.5)
        else:row['metrics']['msssim']=[.8]
        return row
    trace=dict(returned=2,fetches=2,visits=[dict(frame=f,count=2)],sh=[],population=[],reset=[],
        checkpoints=[dict(k=2)],evaluations=[dict(k=k,views=[view(f) for f in spec['evaluation']]) for k in (0,2)],
        fixed=[dict(k=k,views=[view(f,True)]) for k in (0,2)],
        probes=[dict(k=k,visible=0,groups={n:dict(lr=.01,nonzero_gradient=0,changed=0) for n in c.TOLERANCES_GROUPS}) for k in (1,2)],
        events=[['evaluation',0],['fixed',0],['fixed',2],['save',2],['evaluation',2]])
    return trace,spec


class ContractTests(unittest.TestCase):
    def test_sole_json_real_resolver_and_major_boundaries(self):
        import formal_config
        raw=c.CONFIG.read_bytes(); config=formal_config.resolve_formal_config(raw,raw_width=1280,raw_height=720)
        s=c.schedule(c.strict_json(raw))
        self.assertEqual(config.optimization.total_updates,6000)
        self.assertEqual(config.save_updates,(3000,5000,6000))
        self.assertEqual(s['population'],list(range(600,5000,100)))
        self.assertEqual(s['reset'],[3000]); self.assertEqual(len(s['sh']),6)
        self.assertEqual(s['sh'][4]['after'],[3,2]); self.assertEqual(s['sh'][5]['before'],s['sh'][5]['after'])
        self.assertFalse(Path(config.paths.output_directory).exists()); self.assertEqual(c.CONFIG.read_bytes(),raw)

    def test_cold_entry_candidate_and_preflight_no_gpu_import(self):
        code="""
import sys,importlib.util,tempfile,os
from pathlib import Path
from unittest.mock import patch
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','numpy','PIL','train','scene','gaussian_renderer','simple_knn'):
   raise AssertionError('heavy before validation')
sys.meta_path.insert(0,Deny())
s=importlib.util.spec_from_file_location('e',sys.argv[1]);e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
value=e.contract.candidate();e.contract.verify_candidate(value)
assert len(value['representatives'])==9
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp)/'evidence';value['evidence_output']=str(out)
 with patch.object(e.contract,'verify_candidate'):
  verified=e.contract.preflight(value,out)
  assert verified.initialization.time_variance_denominator==33.
  assert verified.time_derivation.effective_variance==1.
  assert verified.time_derivation.log_effective_scale==0.
  assert not out.exists() and not Path(value['output']).exists()
  out.mkdir()
  try:e.contract.preflight(value,out)
  except ValueError:pass
  else:raise AssertionError('reuse accepted')
 try:e.main(['run','--out',str(out),'--seed','7'])
 except SystemExit as ex:assert ex.code==2
 else:raise AssertionError('override accepted')
import formal_inputs
old=e.strict_json((e.contract.BASELINE/'inputs.json').read_bytes(),64*1024**2)
# Historical baseline output is occupied: it must NOT become a new run.
# Valid coefficient-5 connection is covered by the real temporary CPU fixture.
try:formal_inputs.verify_inputs(Path(old['config']['path']).read_bytes())
except ValueError as error:assert str(error)=='p6_output_occupied'
else:raise AssertionError('historical output accepted as a new run')
assert not any(n in sys.modules for n in ('formal_native_bootstrap','scene','gaussian_renderer'))
assert not any(n in sys.modules for n in ('torch','numpy','PIL','train','scene'))
print(sys.executable)
"""
        result=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(ROOT/'tests/cuda/pilot_cuda_acceptance.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(result.stdout.strip(),c.PYTHON)

    def test_candidate_binds_actual_wip_and_review_not_base_head(self):
        value=c.candidate(); value['base_head']='provenance only'; c.verify_candidate(value)
        for field in value:
            if field=='base_head':continue
            bad=copy.deepcopy(value); bad[field]=None
            with self.subTest(field=field),self.assertRaises(ValueError): c.verify_candidate(bad)
        self.assertIn('tests/pilot_runtime.py',value['files'])
        self.assertEqual([r[1] for r in value['representatives']],list(c.REPRESENTATIVES))
        self.assertEqual([r[3] for r in value['representatives']],[0.]*3+[16.6]*3+[33.]*3)

    def test_major_coverage_negative_not_total_forward_or_zero_probe_gate(self):
        trace,spec=small_trace(); c.validate_trace(trace,spec)
        trace['forwards']=123456  # Not a second independent acceptance condition.
        c.validate_trace(trace,spec)
        self.assertIn('no_change_in_sampled_updates',c.learning_summary(trace)['concerns'])
        for mode in ('extra_fetch','wrong_train','order','missing_fixed','missing_probe','extra_SH','wrong_save','wrong_evaluation','nan','false_inf'):
            bad=copy.deepcopy(trace)
            if mode=='extra_fetch':bad['fetches']+=1
            elif mode=='wrong_train':bad['visits'][0]['frame'][0]='test'
            elif mode=='order':bad['events'][-2:]=list(reversed(bad['events'][-2:]))
            elif mode=='missing_fixed':bad['fixed'].pop()
            elif mode=='missing_probe':bad['probes'].pop()
            elif mode=='extra_SH':bad['sh']=[dict(k=1)]
            elif mode=='wrong_save':bad['checkpoints'][0]['k']=1
            elif mode=='wrong_evaluation':bad['evaluations'][1]['views'].pop()
            elif mode=='nan':bad['fixed'][0]['views'][0]['metrics']['l1']=[float('nan')]
            else:bad['fixed'][0]['views'][0]['metrics']['psnr'][0]=dict(positive_infinity=True,zero_mse=False)
            with self.subTest(mode=mode),self.assertRaises(ValueError):c.validate_trace(bad,spec)

    def test_resource_budget_edges_injected_cpu(self):
        initial=dict(host_available=36*c.GIB,gpu_free=12*c.GIB,disks=dict(a=20*c.GIB,b=20*c.GIB))
        ongoing=dict(host_available=4*c.GIB,rss=32*c.GIB,cuda_allocated=13*c.GIB,cuda_reserved=13*c.GIB,disks=dict(a=8*c.GIB))
        c.check_resources(initial,initial=True);c.check_resources(ongoing,initial=False)
        for base,start,fields in ((initial,True,('host_available','gpu_free','disks')),
                                 (ongoing,False,('host_available','rss','cuda_allocated','cuda_reserved','disks'))):
            for field in fields:
                bad=copy.deepcopy(base)
                if field=='disks':bad[field]['a']-=1
                else:bad[field]+=1 if field in ('rss','cuda_allocated','cuda_reserved') else -1
                with self.subTest(field=field,initial=start),self.assertRaises(ValueError): c.check_resources(bad,initial=start)


class IOTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.out=Path(self.temp.name)

    def test_json_normalization_exclusive_and_incomplete_receipt(self):
        import numpy as np
        # Existing normalizer deliberately rejects NumPy objects. Observers
        # explicitly project metric scalars to Python float before persistence.
        value=dict(a=float(np.float32(.1)),b=(True,3.,dict(positive_infinity=True,zero_mse=True)))
        c.write_json(self.out/'value.json',value)
        read=c.strict_json((self.out/'value.json').read_bytes())
        self.assertEqual(read['a'],float(value['a']));self.assertIs(read['b'][0],True)
        with self.assertRaises(FileExistsError):c.write_json(self.out/'value.json',value)
        for i,v in enumerate((float('nan'),float('inf'),object(),np.float32(.1))):
            with self.subTest(i=i),self.assertRaises((ValueError,TypeError)):c.write_json(self.out/f'invalid{i}.json',v)
        real=c.write_json
        for name in ('inputs.json','worker-result.json','reader-result.json'):(self.out/name).write_bytes(b'CPU evidence')
        def fail(path,value):
            if path.name=='receipt.json':raise OSError('injected receipt')
            return real(path,value)
        with patch.object(c,'write_json',side_effect=fail),self.assertRaises(OSError):c.finalize(self.out,dict(status='passed'))
        self.assertTrue((self.out/'result.json').exists())
        with self.assertRaises(FileNotFoundError):c.verify_completion(self.out)

    def test_child_logs_nonzero_timeout_and_preserved_files(self):
        popen=entry.subprocess.Popen
        for mode,code,timeout in (('worker',"import sys;print('o'*12000);print('e'*12000,file=sys.stderr);sys.exit(4)",20),
                                  ('reader','import time;time.sleep(20)',.05)):
            def cpu(command,**kw):
                self.assertTrue(kw['start_new_session']);self.assertEqual(command[0],sys.executable)
                return popen([sys.executable,'-B','-c',code],**kw)
            with patch.object(entry.subprocess,'Popen',side_effect=cpu),self.assertRaisesRegex(ValueError,'child_failed'):
                entry.child(mode,self.out,timeout)
            record=c.strict_json((self.out/(mode+'-exit.json')).read_bytes())
            self.assertNotEqual(record['exit_code'],0);self.assertEqual(record['timed_out'],mode=='reader')
        self.assertEqual((self.out/'worker.stdout.log').read_text(),'o'*12000+'\n')
        self.assertEqual((self.out/'worker.stderr.log').read_text(),'e'*12000+'\n')
        self.assertTrue((self.out/'worker-exit.json').exists())

    def test_receipt_requires_exit_and_log_identity(self):
        for name in ('inputs.json','worker-result.json','reader-result.json'):(self.out/name).write_bytes(b'CPU evidence')
        for name in ('worker','reader'):
            logs={}
            for stream in ('stdout','stderr'):
                path=self.out/(name+'.'+stream+'.log');path.write_bytes(b'CPU fixture');logs[stream]=c.file_identity(path)
            c.write_json(self.out/(name+'-exit.json'),dict(mode=name,exit_code=0,timed_out=False,logs=logs))
        with patch.object(c,'validate_results') as validate:
            c.finalize(self.out,dict(status='passed'));c.verify_completion(self.out)
            self.assertEqual(validate.call_count,2)
            (self.out/'worker.stdout.log').write_bytes(b'corrupted CPU fixture')
            with self.assertRaisesRegex(ValueError,'process_log'):c.verify_completion(self.out)

    def test_cold_public_reader_rejects_M_before_heavy(self):
        (self.out/'checkpoint').mkdir()
        code="""
import importlib.util,sys
from pathlib import Path
s=importlib.util.spec_from_file_location('e',sys.argv[1]);e=importlib.util.module_from_spec(s);s.loader.exec_module(e)
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','numpy','PIL','scene','train','gaussian_renderer'):
   raise AssertionError('heavy before M')
sys.meta_path.insert(0,Deny())
try:e.read_expected(Path(sys.argv[2]),dict(path=sys.argv[2]+'/checkpoint',marker={},k=1),{})
except (ValueError,FileNotFoundError):pass
else:raise AssertionError('missing checkpoint accepted')
assert 'torch' not in sys.modules
"""
        p=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(ROOT/'tests/cuda/pilot_cuda_acceptance.py'),str(self.out)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_result_aggregator_independent_coverage_and_identity(self):
        # Encoded evidence test, not a serializer/render substitute: those are
        # exercised with real consumers separately below. Never write dataset.
        ref=c.strict_json((ROOT/'formal_dataset_reference.json').read_bytes())
        trace,spec=small_trace();expected=c.candidate();expected['output']=str(self.out/'output')
        expected['schedule']={k:spec[k] for k in expected['schedule']}
        expected.update(probes=spec['probes'],fixed_updates=spec['fixed_updates'])
        spec['representatives']=expected['representatives']
        spec['train']=[['train',i,f[0]] for i,f in enumerate(ref['frames']['train'])]
        spec['evaluation']=[spec['train'][i] for i in (5,10,15,20,25)]+[['test',i,f[0]] for i,f in enumerate(ref['frames']['test'])]
        trace['visits']=[dict(frame=spec['train'][0],count=2)]
        for row in trace['evaluations']:
            v=row['views'][0];row['views']=[dict(v,frame=f) for f in spec['evaluation']]
        for fixed in trace['fixed']:
            template=fixed['views'][0];fixed['views']=[]
            folder=self.out/'fixed'/str(fixed['k']);folder.mkdir(parents=True)
            for rep in spec['representatives']:
                view=copy.deepcopy(template);view['frame']=rep[:3]
                for name in view['images']:
                    path=folder/(str(rep[1])+'-'+name+'.png');path.write_bytes(b'encoded identity fixture')
                    view['images'][name]=c.file_identity(path)
                fixed['views'].append(view)
        for row in trace['checkpoints']:
            path=self.out/'output/checkpoints'/('completed-'+str(row['k']));path.mkdir(parents=True)
            for name in ('complete.json','metadata.json','state.pt'):(path/name).write_bytes(b'encoded identity fixture')
            row.update(path=str(path),files={n:c.file_identity(path/n) for n in ('complete.json','metadata.json','state.pt')})
            row['marker']=row['files']['complete.json']
        (self.out/'expected.pt').write_bytes(b'encoded independent expectation')
        frames=spec['evaluation']+[spec['train'][0]]+[r[:3] for r in spec['representatives']]
        consumed={str(Path(ref['source_path'])/p):c.file_identity(self.out/'expected.pt')
                  for split,i,_ in frames for p in ref['frames'][split][i][::2]}
        w=dict(status='passed',spec=spec,trace=trace,entry_calls=1,returned=2,stage='normal_return',
               writer=dict(present=False,flush=0,close=0),diagnostic=dict(completed=2,attempted=2,saves=1,tests=1,last_confirmed_save=2),
               expected_identity=c.file_identity(self.out/'expected.pt'),consumed=consumed)
        r=dict(status='passed',checkpoint=trace['checkpoints'][-1],representatives=spec['representatives'],
               parameter_groups=sorted(c.TOLERANCES_GROUPS),exact=True,reader_builds=0,forwards=9)
        for name,value in (('inputs',expected),('worker-result',w),('reader-result',r)):c.write_json(self.out/(name+'.json'),value)
        identity=c.file_identity;read=c.strict_json
        def file(path):return consumed[str(path)] if str(path) in consumed else identity(path)
        # Diagnostic semantic validation has its own real saved-file cases.
        w['pilot_diagnostics']={}
        (self.out/'worker-result.json').write_bytes(c.canonical(w))
        with patch.object(c,'verify_candidate'),patch.object(c,'file_identity',side_effect=file),patch.object(c,'validate_diagnostics') as diagnostic:
            c.validate_results(self.out)
            diagnostic.assert_called_once()
            for mode in ('reader_missing','reader_rebuild','marker','writer','consumed','expected','missing_view','schedule','nonzero_exit_status'):
                def change(data,*a,**kw):
                    v=read(data,*a,**kw)
                    if 'parameter_groups' in v:
                        if mode=='reader_missing':v['parameter_groups'].pop()
                        elif mode=='reader_rebuild':v['reader_builds']=1
                        elif mode=='marker':v['checkpoint']['k']=1
                    if 'entry_calls' in v:
                        if mode=='writer':v['writer']['flush']=1
                        elif mode=='consumed':v['consumed'].pop(next(iter(v['consumed'])))
                        elif mode=='expected':v['expected_identity']['size']+=1
                        elif mode=='missing_view':v['trace']['fixed'][0]['views'].pop()
                        elif mode=='schedule':v['spec']['updates']=3
                        elif mode=='nonzero_exit_status':v['status']='failed'
                    return v
                with self.subTest(mode=mode),patch.object(c,'strict_json',side_effect=change),self.assertRaises(ValueError):c.validate_results(self.out)


class CompletionTests(unittest.TestCase):
    """Real saved receipt/exit/log IO; isolate only the semantic aggregator.

    These small fixtures are not evidence of a real training or CUDA reader run.
    The separate IOTests aggregator regression exercises semantic validation.
    """
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)

    def completion(self, label, *, mode=None, change=None, payload=b'CPU log\n'):
        out=self.out/label;out.mkdir()
        for name in ('inputs.json','worker-result.json','reader-result.json'):(out/name).write_bytes(b'CPU evidence')
        for name in ('worker','reader'):
            logs={}
            for stream in ('stdout','stderr'):
                path=out/(name+'.'+stream+'.log');path.write_bytes(payload)
                logs[stream]=c.file_identity(path)
            record=dict(mode=name,exit_code=0,timed_out=False,logs=logs)
            if name==mode:change(record)
            c.write_json(out/(name+'-exit.json'),record)
        result=dict(status='checks_passed',finalization='requires_verified_receipt_and_exit_zero',
                    evidence={n:c.file_identity(out/n) for n in ('inputs.json','worker-result.json','reader-result.json')})
        identity=c.write_json(out/'result.json',result)
        c.write_json(out/'receipt.json',dict(status='passed',result=identity))
        return out,result

    def test_log_membership_rejects_missing_extra_and_invalid_shapes(self):
        changes={
            'missing':lambda r:r.pop('logs'),
            'empty':lambda r:r.update(logs={}),
            'no_stdout':lambda r:r['logs'].pop('stdout'),
            'no_stderr':lambda r:r['logs'].pop('stderr'),
            'extra':lambda r:r['logs'].update(other=r['logs']['stdout']),
        }
        for label,value in (('null',None),('array',[]),('pairs',[['stdout',{}],['stderr',{}]]),
                            ('text','stdout stderr'),('boolean',False),('number',2)):
            changes[label]=lambda r,v=value:r.update(logs=v)
        for mode in ('worker','reader'):
            for label,change in changes.items():
                with self.subTest(mode=mode,case=label):
                    out,_=self.completion(mode+'-'+label,mode=mode,change=change)
                    with patch.object(c,'validate_results') as semantic:
                        with self.assertRaisesRegex(ValueError,'process_log_membership'):c.verify_completion(out)
                        semantic.assert_not_called()

    def test_log_files_required_for_both_processes_and_streams(self):
        for mode in ('worker','reader'):
            for stream in ('stdout','stderr'):
                with self.subTest(mode=mode,stream=stream):
                    out,_=self.completion(mode+'-'+stream,payload=b'')
                    (out/(mode+'.'+stream+'.log')).unlink()
                    with patch.object(c,'validate_results') as semantic:
                        with self.assertRaises(FileNotFoundError):c.verify_completion(out)
                        semantic.assert_not_called()

    def test_each_log_size_hash_and_identity_shape_are_checked(self):
        for mode in ('worker','reader'):
            for stream in ('stdout','stderr'):
                for field in ('size','hash','missing_identity','invalid_identity'):
                    def change(record):
                        value=record['logs'][stream]
                        if field=='size':value['size']+=1
                        elif field=='hash':value['sha256']=c.file_identity(Path(__file__))['sha256']
                        elif field=='missing_identity':record['logs'][stream]={}
                        else:record['logs'][stream]=None
                    with self.subTest(mode=mode,stream=stream,field=field):
                        out,_=self.completion(mode+'-'+stream+'-'+field,mode=mode,change=change)
                        with patch.object(c,'validate_results') as semantic:
                            with self.assertRaisesRegex(ValueError,'process_log_identity'):c.verify_completion(out)
                            semantic.assert_not_called()

    def test_two_matching_logs_allow_empty_contents(self):
        for label,payload in (('nonempty',b'CPU log\n'),('empty',b'')):
            with self.subTest(case=label):
                out,result=self.completion(label,payload=payload)
                with patch.object(c,'validate_results') as semantic:
                    self.assertEqual(c.verify_completion(out),result)
                    semantic.assert_called_once_with(out)

    def test_receipt_binding_and_status_still_required(self):
        for label in ('result_bytes','receipt_status','result_status','finalization'):
            with self.subTest(case=label):
                out,result=self.completion(label)
                if label=='result_bytes':
                    with (out/'result.json').open('ab') as f:f.write(b' ')
                else:
                    receipt=c.strict_json((out/'receipt.json').read_bytes())
                    if label=='receipt_status':receipt['status']='failed'
                    else:
                        result['status' if label=='result_status' else 'finalization']='failed'
                        (out/'result.json').write_bytes(c.canonical(result))
                        receipt['result']=c.file_identity(out/'result.json')
                    (out/'receipt.json').write_bytes(c.canonical(receipt))
                with patch.object(c,'validate_results') as semantic:
                    with self.assertRaisesRegex(ValueError,'receipt'):c.verify_completion(out)
                    semantic.assert_not_called()

    def test_child_exit_timeout_and_mode_still_required(self):
        for mode in ('worker','reader'):
            for field,value in (('exit_code',4),('timed_out',True),('mode','wrong')):
                with self.subTest(mode=mode,field=field):
                    out,_=self.completion(mode+'-'+field,mode=mode,change=lambda r:r.update({field:value}))
                    with patch.object(c,'validate_results') as semantic:
                        with self.assertRaisesRegex(ValueError,'child_exit'):c.verify_completion(out)
                        semantic.assert_not_called()

    def test_matching_logs_do_not_bypass_semantic_failure(self):
        out,_=self.completion('semantic')
        with patch.object(c,'validate_results',side_effect=ValueError('semantic fixture failure')) as semantic:
            with self.assertRaisesRegex(ValueError,'semantic fixture failure'):c.verify_completion(out)
            semantic.assert_called_once_with(out)


from formal_training_fixture import CPUCase


class ConsumerTests(CPUCase):
    def prepare(self,name, *, kind='clone', initial_unit=False):
        import training_update_runtime as fixture
        import formal_inputs
        from test_formal_config import encode
        case=dict(name=name,n=6,batch=1,kind=kind)
        value,ref=fixture.create_fixture(self.f.base/name,case)
        value['optimization']['sh']['increase_interval']=1
        value['optimization']['population'].update(densify_from_update=0,densify_until_update=5,
            densification_interval=2,opacity_reset_interval=3,thresh_opa_prune=0,densify_stop_points=24)
        value['checkpoint']['intermediate_updates']=[3,5];value['reporting']['test_updates']=[3,5,6]
        if initial_unit:value['initialization']['time_variance_denominator']=2
        owner=patch.object(formal_inputs,'_reference',return_value=ref);owner.start();self.addCleanup(owner.stop)
        p=self.train.prepare_formal_runtime(formal_inputs.verify_inputs(encode(value)),native_evidence=self.f.checkpoint_evidence())
        fixture.initialize_population(p,case)
        out=self.f.base/(name+'-evidence');out.mkdir()
        return p,value,ref,out

    def renderer(self,camera,model,pipe,background,**kw):
        """Explicit differentiable GPU seam only; real model/CPU consumers elsewhere."""
        import training_update_oracle as oracle
        t=self.torch;screen=t.zeros_like(model._xyz,requires_grad=True)
        signal=screen[:,:2].sum()*.001
        for attr in oracle.GROUPS.values():
            p=getattr(model,attr);signal=signal+p.sum()*.00001
        h,w=camera.image_height,camera.image_width
        ramp=t.linspace(0,.1,h*w).reshape(h,w)
        return dict(render=(.5+ramp+signal).expand(3,h,w),alpha=(.3+model._opacity.sum()*.0001).expand(1,h,w),
            depth=(1+ramp).reshape(1,h,w),radii=t.ones(len(model._xyz)),
            visibility_filter=t.ones(len(model._xyz),dtype=t.bool),viewspace_points=screen)

    def observer(self,p,value,ref,out,**kw):
        from pilot_runtime import Observation
        reps=[['train',i,*ref['frames']['train'][i][:2]] for i in range(3)]
        kw.setdefault('diagnostic_updates',None)  # Existing CPU regressions unchanged.
        return Observation(p,self.renderer,out,c.schedule(value),reps,[1,3,5,6],[0,3,5,6],**kw)

    def test_added_observations_same_coefficient_noninterference_and_reset(self):
        from pilot_runtime import state,rng_state
        from formal_checkpoint_state import equal_state
        p0,v0,r0,o0=self.prepare('extra-off');off=self.observer(p0,v0,r0,o0)
        self.execute(p0,off);before=state(p0.gaussians);rng=rng_state(p0.gaussians)
        p,v,r,out=self.prepare('extra-on');obs=self.observer(p,v,r,out,diagnostic_updates=(3,4))
        self.execute(p,obs)
        equal_state(before,state(p.gaussians));equal_state(rng,rng_state(p.gaussians))
        for attr in ('_xyz','_opacity','_scaling_t'):self.assertIsNone(getattr(p.gaussians,attr).grad)
        for name in ('events','visits','sh','population','reset','fetches'):self.assertEqual(off.trace[name],obs.trace[name])
        self.assertEqual(set(obs.diagnostics),{'initial','reset','prune'})
        self.assertAlmostEqual(obs.diagnostics['initial']['widths']['scale']['mean'],.15,places=7)
        reset=obs.diagnostics['reset'];self.assertEqual((reset['completed'],reset['after_fixed']),(2,3))
        self.assertEqual(len(list((out/'diagnostics/reset-3-before').glob('*.png'))),12)
        self.assertEqual([r['k'] for r in obs.trace['fixed']],[0,3,5,6])
        self.assertIsNone(obs.reset_after);self.assertIsNotNone(obs.expected_identity)
        self.assertTrue(obs.diagnostics['prune']['prune_only']);self.assertEqual(obs.diagnostics['prune']['split_parent_calls'],0)
        for pre,post in zip(reset['views'],obs.trace['fixed'][1]['views']):self.assertEqual(pre['images']['gt'],post['images']['gt'])
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_real_initialization_unit_width_and_nonidentity(self):
        import pilot_diagnostics as d
        p,v,r,out=self.prepare('unit-width',kind='initial',initial_unit=True);m=p.gaussians
        summary=d.widths(m._scaling,m._scaling_t,m._rotation,m._rotation_r,m.get_scaling_t)
        for name,s in summary.items():self.assertEqual(s['mean'],0. if name=='log_scale' else 1.)
        # Quarter-turn quaternion mixes temporal and one spatial axis. Literal
        # column-major GLM multiplication is independent of the observer formula.
        import numpy as np
        q=self.torch.tensor([[.8,.6,0.,0.]])
        l=q;r=self.torch.tensor([[1.,0.,0.,0.]])
        log_s=self.torch.tensor([[.2,.4,.6]]);log_t=self.torch.tensor([[.8]])
        _,_,variance=d.temporal_arrays(log_s,log_t,l,r)
        aa,bb,cc,dd=l.double().numpy()[0]/np.linalg.norm(l.double().numpy()[0])
        ml=np.array([aa,bb,-cc,dd,-bb,aa,dd,cc,cc,-dd,aa,bb,-dd,-cc,-bb,aa]).reshape(4,4).T
        scales=np.exp(np.concatenate((log_s.double().numpy()[0],log_t.double().numpy()[0])))
        mat=scales[:,None]*ml
        self.assertAlmostEqual(variance[0],(mat.T@mat)[3,3],places=12)

    def test_growth_prune_final_rows_excludes_split_parent(self):
        p,v,r,out=self.prepare('growth-reasons',kind='mixed');m=p.gaussians
        obs=self.observer(p,v,r,out,diagnostic_updates=(3,2))
        p.health_run.completed=1;p.health_run.stage='population'
        with self.torch.no_grad():
            m.xyz_gradient_accum.fill_(1);m.denom.fill_(1)
            obs.population(m.densify_and_prune,0.,.05,p.scene.cameras_extent,20,None,prune_only=False)
        row=obs.diagnostics['prune']
        self.assertEqual((row['before'],row['rows'],row['after'],row['union']),(16,32,28,4))
        self.assertEqual(row['split_parent_calls'],1);self.assertEqual(row['bins'],[28,4,0,0,0,0,0,0])
        self.assertEqual(m.prune_points.__func__.__name__,'prune_points')

    def test_added_guard_rejects_state_rng_parameter_and_mutation(self):
        import pilot_diagnostics as d
        for mode in ('rng','parameter','grad','stats','adam','health','loader','step'):
            with self.subTest(mode=mode):
                p,_,_,_=self.prepare('guard-'+mode);m=p.gaussians
                p.dataloader.generator=self.torch.Generator().manual_seed(5)
                with self.assertRaises((ValueError,AssertionError)):
                    with d.unchanged(p):
                        if mode=='rng':self.torch.rand(1)
                        elif mode=='parameter':m._xyz=self.torch.nn.Parameter(m._xyz.detach().clone())
                        elif mode=='grad':m._xyz.grad=self.torch.ones_like(m._xyz)
                        elif mode=='stats':m.denom.add_(1)
                        elif mode=='adam':m.optimizer.param_groups[0]['lr']+=.1
                        elif mode=='health':p.health_run.completed+=1
                        elif mode=='loader':self.torch.rand(1,generator=p.dataloader.generator)
                        else:m.optimizer.step()

    def test_added_capture_failure_unwinds_and_does_not_reset(self):
        p,v,r,out=self.prepare('extra-failure');obs=self.observer(p,v,r,out,diagnostic_updates=(3,4))
        p.health_run.completed=2;p.health_run.stage='reset'
        original=p.gaussians.reset_opacity
        def broken(*a,**kw):raise ValueError('capture fixture')
        with patch.object(obs,'capture',side_effect=broken),patch.object(p.gaussians,'reset_opacity',wraps=original) as reset:
            with self.assertRaisesRegex(ValueError,'capture fixture'):obs.reset(reset)
            reset.assert_not_called()
        self.assertEqual(p.gaussians.reset_opacity.__func__,original.__func__)

    def test_comparison_real_three_checkpoint_cpu_reader_and_view_pairing(self):
        import importlib.util
        import formal_checkpoint as cp
        spec=importlib.util.spec_from_file_location('pilot_compare',c.COMPARE)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        p,v,r,out=self.prepare('comparison');obs=self.observer(p,v,r,out)
        self.execute(p,obs)
        summaries=[]
        for row in obs.trace['checkpoints']:
            loaded=cp.read_checkpoint(Path(row['path']))
            summaries.append(helper.checkpoint_widths(loaded,obs.spec['representatives']))
            del loaded
        self.assertEqual([s['k'] for s in summaries],[3,5,6])
        for s in summaries:
            self.assertEqual([v['raw_time'] for v in s['temporal_gate']],[0.,.2,.4])
            self.assertEqual([v['time'] for v in s['temporal_gate']],[0.,.1,.2])
            self.assertTrue(all(0<=v['temporal_gate_count']<=s['rows'] for v in s['temporal_gate']))
        c.write_json(out/'widths.json',summaries)
        self.assertEqual(c.strict_json((out/'widths.json').read_bytes()),summaries)
        self.assertEqual([s['rows'] for s in summaries],[row['after'] for row in [obs.trace['population'][0],obs.trace['population'][1],obs.trace['population'][1]]])
        row=obs.trace['fixed'][-1]
        def pair(b):return helper.paired_views(b,row,obs.spec['representatives'],out,out,obs.consumed.finish(),Path(r['source_path']))
        result=pair(row)
        self.assertEqual(result['equal_view_means']['baseline'],result['equal_view_means']['new'])
        self.assertEqual(len(result['by_time']),3)
        for mode in ('order','time','gt','test'):
            bad=copy.deepcopy(row)
            if mode=='order':bad['views'].reverse()
            elif mode=='time':bad['views'][0]['raw_time']+=.1
            elif mode=='gt':bad['views'][0]['images']['gt']={}
            else:bad['views'][0]['frame'][0]='test'
            with self.subTest(mode=mode),self.assertRaises(ValueError):pair(bad)
        payload=dict(status='comparison_recorded_requires_review',selected_final=6000,cuda_initialized=False,
                     records={str(out/'expected.pt'):c.file_identity(out/'expected.pt')},views=result)
        target=out/'comparison.json';c.write_json(target,payload)
        c.write_json(target.with_suffix('.receipt.json'),dict(result=c.file_identity(target),status='recorded'))
        self.assertEqual(helper.verify(target),payload)
        with (out/'expected.pt').open('ab') as f:f.write(b'x')
        with self.assertRaisesRegex(ValueError,'comparison_records'):helper.verify(target)

    def execute(self,p,obs=None,visits=None):
        from test_training_update_cpu import Event
        import formal_entry
        claim=formal_entry.OutputClaim(p.inputs);self.addCleanup(claim.close)
        with patch.object(self.train,'TENSORBOARD_FOUND',False),patch.object(self.torch.cuda,'Event',Event),patch.object(self.train,'render',self.renderer):
            writer=self.train.begin_formal_output(p,claim)
            if obs is not None:returned=obs.run(self.train.continue_formal_training,writer)
            else:
                loader=p.dataloader
                class Batches:
                    def __iter__(self):
                        for b in loader:
                            if visits is not None:visits.extend(tuple(c.frame_key(cam)) for _,cam in b)
                            yield b
                returned=self.train.continue_formal_training(p._replace(dataloader=Batches()),writer)
            p.health_run.finish_updates(p.inputs.config,returned)
        return returned

    def test_real_loop_observation_noninterference_events_save_and_reader(self):
        from pilot_runtime import state,rng_state
        from formal_checkpoint_state import equal_state,restore_diagnostic
        p0,_,_,_=self.prepare('baseline');visits=[];self.execute(p0,visits=visits)
        baseline=state(p0.gaussians);baseline_rng=rng_state(p0.gaussians)
        p,value,ref,out=self.prepare('observed');obs=self.observer(p,value,ref,out)
        self.assertEqual(self.execute(p,obs),6)
        equal_state(baseline,state(p.gaussians));equal_state(baseline_rng,rng_state(p.gaussians))
        self.assertEqual(obs.visits,{key:visits.count(key) for key in set(visits)})
        self.assertEqual([(r['k'],r['prune_only']) for r in obs.trace['population']],[(2,False),(4,True)])
        self.assertEqual([r['k'] for r in obs.trace['reset']],[3]);self.assertEqual(obs.trace['sh'][-1]['before'],[3,2])
        self.assertEqual(len(obs.trace['fixed']),4);self.assertEqual(len(obs.trace['probes']),4)
        self.assertEqual(len(obs.consumed.finish()),10)
        loaded,saved=entry.read_expected(out,obs.trace['checkpoints'][-1],obs.expected_identity)
        with patch.object(type(p.gaussians),'training_setup',side_effect=AssertionError('reader training')):
            model=restore_diagnostic(loaded,'cpu')
        self.assertIsNone(model.optimizer)
        from checkpoint_cuda_acceptance import diagnostic_camera,camera_values
        for view,rep in zip(saved['views'],obs.spec['representatives']):
            cam=diagnostic_camera(loaded['config'],rep[:3]);self.assertEqual(camera_values(cam),view['camera'])
            equal_state(view['output'],entry.render_buffers(self.renderer,cam,model,loaded['config']))
        row=obs.trace['fixed'][0]['views'][0]
        path=out/'fixed/0'/(str(row['frame'][1])+'-render.png')
        with self.f.Image.open(path) as image:self.assertEqual(image.format,'PNG');self.assertEqual(image.size,(256,256))
        self.assertFalse(self.torch.cuda.is_initialized())
        with self.assertRaisesRegex(ValueError,'expected_snapshot_identity'):entry.read_expected(out,obs.trace['checkpoints'][-1],{})
        saved['attrs'][1][0,0]+=1
        from test_formal_checkpoint import assert_expected
        with self.assertRaises(AssertionError):assert_expected(self,(saved['attrs'],saved['groups']),loaded)

    def test_original_delegate_and_exception_and_zero_probe(self):
        p,value,ref,out=self.prepare('delegation');obs=self.observer(p,value,ref,out)
        calls=[];result=object()
        def delegate(*a,**kw):calls.append((a,kw));return result
        self.assertIs(obs.step(delegate,7,example=True),result)
        self.assertEqual(calls,[((7,),dict(example=True))])
        self.assertTrue(all(g['changed']==0 for g in obs.trace['probes'][0]['groups'].values()))
        error=RuntimeError('original error')
        def fail(*a,**kw):raise error
        with self.assertRaises(RuntimeError) as got:obs.reset(fail)
        self.assertIs(got.exception,error);self.assertFalse(obs.trace['reset'])

    def test_fixed_identity_metric_type_and_rng_interference_rejected(self):
        from pilot_runtime import metric_values
        p,value,ref,out=self.prepare('invariants');obs=self.observer(p,value,ref,out)
        image=self.torch.ones(3,2,2)
        self.assertEqual(metric_values('l1',.25,image,image),[.25])
        self.assertEqual(metric_values('psnr',self.torch.full((3,1),float('inf')),image,image)[0],dict(positive_infinity=True,zero_mse=True))
        with self.assertRaisesRegex(ValueError,'PSNR_zero_mse'):
            metric_values('psnr',self.torch.full((3,1),float('inf')),image,image*0)
        def changes_rng(*a,**kw):self.torch.rand(1);return self.renderer(*a,**kw)
        obs.renderer=changes_rng
        with self.assertRaises((ValueError,AssertionError)):obs.fixed(0)
        self.assertFalse(obs.trace['fixed']);self.assertTrue(list((out/'fixed/0').glob('*.png')))

    def test_resource_failure_preserves_confirmed_checkpoint_and_stops(self):
        import formal_checkpoint as cp
        p,value,ref,out=self.prepare('resource_failure');held={}
        def check(k,stage):
            if k==3 and stage=='before_evaluation':
                directory=Path(p.scene.model_path)/'checkpoints/completed-3'
                held.update({str(f):c.file_identity(f) for f in directory.iterdir()})
                raise ValueError('injected resource ceiling')
        obs=self.observer(p,value,ref,out,resource_check=check)
        with self.assertRaisesRegex(ValueError,'injected resource'):self.execute(p,obs)
        self.assertEqual(p.health_run.completed,3);self.assertEqual(p.health_run.last_confirmed_save,3)
        self.assertIsNone(obs.trace['returned']);self.assertFalse((out/'expected.pt').exists())
        self.assertTrue(held)
        for path,identity in held.items():self.assertEqual(c.file_identity(Path(path)),identity)
        self.assertEqual(cp.read_checkpoint(Path(p.scene.model_path)/'checkpoints/completed-3')['metadata']['k'],3)
        self.assertFalse((Path(p.scene.model_path)/'checkpoints/completed-5').exists())

    def test_ordinary_entry_once_real_prepare_claim_training_and_return(self):
        import formal_entry,formal_native_bootstrap
        import training_update_runtime as fixture
        from test_formal_config import encode
        from test_training_update_cpu import Event
        # GPU bootstrap is the only execution substitute. Input verification,
        # preparation, claim, output, training, writer and termination are real.
        p,value,ref,out=self.prepare('entry')
        locator=self.f.base/'entry-config.json';locator.write_bytes(encode(value))
        real_prepare=self.train.prepare_formal_runtime;real_continue=self.train.continue_formal_training
        held={};calls=[]
        def prepare(*a,**kw):
            calls.append('prepare');prepared=real_prepare(*a,**kw)
            fixture.initialize_population(prepared,dict(kind='clone'))
            held['p']=prepared;return prepared
        def continuation(prepared,writer):
            calls.append('continue');obs=self.observer(prepared,value,ref,out);held['obs']=obs
            return obs.run(real_continue,writer)
        with patch.object(formal_native_bootstrap,'bootstrap',return_value=(self.train,self.f.checkpoint_evidence())) as bootstrap, \
                patch.object(self.train,'prepare_formal_runtime',side_effect=prepare), \
                patch.object(self.train,'continue_formal_training',side_effect=continuation), \
                patch.object(self.train,'TENSORBOARD_FOUND',False),patch.object(self.torch.cuda,'Event',Event):
            self.assertEqual(formal_entry.run(locator),6)
        self.assertEqual(bootstrap.call_count,1);self.assertEqual(calls,['prepare','continue'])
        self.assertEqual(held['p'].health_run.stage,'normal_return')
        self.assertEqual(held['p'].health_run.saves,3);self.assertEqual(held['p'].health_run.tests,3)
        self.assertIsNotNone(held['obs'].expected_identity);self.assertFalse(self.torch.cuda.is_initialized())


if __name__=='__main__':
    if sys.executable!=c.PYTHON:raise SystemExit('4dgs310 required')
    unittest.main(failfast=True)
