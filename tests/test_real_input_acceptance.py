"""Step 15 CPU tests. Real CPU consumers, temporary files, isolated GPU only."""
from contextlib import ExitStack
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'tests/cuda')]
import real_input_contract as c
import real_input_cuda_acceptance as entry


def small_trace():
    train=[['train',0,'images/train0.png']]
    spec=dict(updates=3,rows=16,saves=[1,3],tests=[3],train=train,
              evaluation=train+[['test',0,'images/test0.png']])
    events=[['report_begin',0],['report_end',0]]
    for k in (1,2,3):
        events.extend([[name,k] for name in ('fetch','lr','render','backward','statistics','step','zero')])
        if k in (1,3):events.extend([['save_begin',k],['save_end',k]])
        events.extend([['report_begin',k],['report_end',k]])
    comparisons=[dict(check=n+'/'+key,max_bound_ratio=0.) for n in c.TOLERANCES_GROUPS for key in ('p','m','v','step')]
    trace=dict(events=events,returned=3,reports=[0,1,2,3],batches=[train,train,train],
        steps=[dict(k=k,rows=16,visible=4,nonzero_gradients=2,changed_positive_lr=1,
                    lr={n:.001 for n in c.TOLERANCES_GROUPS},comparisons=copy.deepcopy(comparisons)) for k in (1,2,3)],
        checkpoints=[dict(k=1),dict(k=3)],forwards=7,
        evaluations=[dict(k=k,views=[dict(frame=frame,metrics=dict(l1=[.1],psnr=[10.,11.,12.],ssim=[.9],msssim=[.8]))
                                    for frame in spec['evaluation']]) for k in (0,3)])
    return trace,spec


class ContractTests(unittest.TestCase):
    def test_adopted_json_public_resolver_read_only(self):
        import formal_config
        raw=c.CONFIG.read_bytes()
        value=formal_config.resolve_formal_config(raw,raw_width=1280,raw_height=720)
        self.assertEqual(value.optimization.total_updates,3)
        self.assertEqual(value.save_updates,(1,3));self.assertEqual(value.reporting.test_updates,(3,))
        self.assertEqual(value.initialization.num_pts,16384)
        self.assertFalse(Path(value.paths.output_directory).exists())
        self.assertEqual(c.CONFIG.read_bytes(),raw)

    def test_stdlib_entry_candidate_no_heavy_import(self):
        code="""
import sys,importlib.util
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','numpy','PIL','scene','train','gaussian_renderer','simple_knn'):
   raise AssertionError('preloaded heavy dependency')
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location('entry',sys.argv[1]);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
value=e.contract.candidate();e.contract.verify_candidate(value)
assert 'tests/real_input_runtime.py' in value['files']
print(sys.executable)
"""
        p=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(ROOT/'tests/cuda/real_input_cuda_acceptance.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip(),sys.executable)

    def test_candidate_rejects_changes_all_technical_fields_not_base_head(self):
        value=c.candidate();changed=copy.deepcopy(value);changed['base_head']='historical provenance'
        c.verify_candidate(changed)
        for field in ('files','config','output','evidence_output','resources','representatives','tolerances','budgets'):
            bad=copy.deepcopy(value);bad[field]=None
            with self.subTest(field=field),self.assertRaises(ValueError):c.verify_candidate(bad)

    def test_cold_preflight_output_import_debug_and_cli_boundaries(self):
        code="""
import importlib.util,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('entry',sys.argv[1]);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
value=e.contract.candidate()
with tempfile.TemporaryDirectory() as tmp:
 out=Path(tmp)/'evidence';value['evidence_output']=str(out)
 # Only the reviewed destination is substituted at this unit-test boundary;
 # the real adopted JSON still passes through the public pure resolver.
 with patch.object(e.contract,'verify_candidate'):
  e.contract.preflight(value,out)
  assert not out.exists() and not Path(value['output']).exists()
  for mode in ('existing','preloaded','debug','alias'):
   with patch.dict(sys.modules,{'torch':object()} if mode=='preloaded' else {}),patch.dict(os.environ,{'STEP90_CUDA_DEBUG_TEST':'1'} if mode=='debug' else {}):
    if mode=='existing':out.mkdir()
    target=out/'..'/'evidence' if mode=='alias' else out
    try:e.contract.preflight(value,target)
    except ValueError:pass
    else:raise AssertionError(mode+' accepted')
    if mode=='existing':out.rmdir()
 try:e.main(['run','--out',str(out),'--seed','7'])
 except SystemExit as error:assert error.code==2
 else:raise AssertionError('CLI override accepted')
 assert not out.exists() and not Path(value['output']).exists()
assert not any(n in sys.modules for n in ('torch','numpy','PIL','train','scene'))
print(sys.executable)
"""
        p=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(ROOT/'tests/cuda/real_input_cuda_acceptance.py')],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip(),sys.executable)

    def test_trace_normal(self):
        trace,spec=small_trace();self.assertIs(c.validate_trace(trace,spec),trace)

    def test_missing_extra_reordered_events(self):
        trace,spec=small_trace()
        for mode in ('missing','extra','order','zero','N_plus_one','save_after_test'):
            bad=copy.deepcopy(trace)
            if mode=='missing':bad['events'].pop(6)
            elif mode=='extra':bad['events'].append(['zero',3])
            elif mode=='order':bad['events'][6],bad['events'][7]=bad['events'][7],bad['events'][6]
            elif mode=='zero':bad['events'].insert(0,['step',0])
            elif mode=='N_plus_one':bad['events'].append(['fetch',4])
            else:bad['events'][-4:]=bad['events'][-2:]+bad['events'][-4:-2]
            with self.subTest(mode=mode),self.assertRaises(ValueError):c.validate_trace(bad,spec)

    def test_updates_gradients_visibility_group_coverage(self):
        trace,spec=small_trace()
        for field in ('visible','nonzero_gradients','changed_positive_lr'):
            bad=copy.deepcopy(trace);bad['steps'][2][field]=0
            with self.subTest(field=field),self.assertRaises(ValueError):c.validate_trace(bad,spec)
        for mode in ('step','group','comparison','duplicate','nan'):
            bad=copy.deepcopy(trace)
            if mode=='step':bad['steps'].pop()
            elif mode=='group':bad['steps'][0]['lr'].pop('t')
            elif mode=='comparison':bad['steps'][0]['comparisons'].pop()
            elif mode=='duplicate':bad['steps'][0]['comparisons'][0]=bad['steps'][0]['comparisons'][1]
            else:bad['steps'][0]['comparisons'][0]['max_bound_ratio']=float('nan')
            with self.subTest(mode=mode),self.assertRaises(ValueError):c.validate_trace(bad,spec)

    def test_evaluation_schedule_frames_metrics_and_training_mixing(self):
        trace,spec=small_trace()
        for mode in ('missing','extra','wrong','metric','nan','test_train','forwards','save','returned'):
            bad=copy.deepcopy(trace)
            if mode=='missing':bad['evaluations'][1]['views'].pop()
            elif mode=='extra':bad['evaluations'][1]['views'].append(bad['evaluations'][1]['views'][0])
            elif mode=='wrong':bad['evaluations'][1]['views'][0]['frame']=['train',2,'other']
            elif mode=='metric':bad['evaluations'][1]['views'][0]['metrics'].pop('ssim')
            elif mode=='nan':bad['evaluations'][1]['views'][0]['metrics']['l1']=[float('nan')]
            elif mode=='test_train':bad['batches'][1]=[spec['evaluation'][-1]]
            elif mode=='forwards':bad['forwards']+=1
            elif mode=='save':bad['checkpoints'][-1]['k']=2
            else:bad['returned']=2
            with self.subTest(mode=mode),self.assertRaises(ValueError):c.validate_trace(bad,spec)

    def test_only_justified_psnr_infinity(self):
        trace,spec=small_trace();v=trace['evaluations'][0]['views'][0]
        v['metrics']['psnr'][0]=dict(positive_infinity=True,zero_mse=True)
        c.validate_trace(trace,spec)
        for key in ('zero_mse','positive_infinity'):
            bad=copy.deepcopy(trace);bad['evaluations'][0]['views'][0]['metrics']['psnr'][0][key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):c.validate_trace(bad,spec)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.out=Path(self.temp.name)

    def test_lossless_exclusive_json_and_readback_failure(self):
        original=dict(value=(True,1,1.,'x',dict(positive_infinity=True,zero_mse=True)))
        before=copy.deepcopy(original);path=self.out/'value.json'
        c.write_json(path,original);self.assertEqual(original,before)
        with self.assertRaises(FileExistsError):c.write_json(path,original)
        read=c.strict_json
        def altered(*a,**kw):
            value=read(*a,**kw);value['value'][0]=1;return value
        with patch.object(c,'strict_json',side_effect=altered),self.assertRaisesRegex(ValueError,'readback'):
            c.write_json(self.out/'wrong.json',original)
        for i,value in enumerate((float('inf'),float('nan'),object(),{1:2})):
            with self.subTest(i=i),self.assertRaises((ValueError,TypeError)):c.write_json(self.out/f'bad{i}.json',value)

    def test_receipt_exit_codes_and_incomplete_write(self):
        for name in ('worker','reader'):c.write_json(self.out/(name+'-exit.json'),dict(mode=name,exit_code=0))
        with patch.object(c,'validate_results') as validate:
            c.finalize(self.out,dict(status='passed'));c.verify_completion(self.out);self.assertEqual(validate.call_count,2)
        read=c.strict_json
        def nonzero(*a,**kw):
            value=read(*a,**kw)
            if value.get('mode')=='reader':value['exit_code']=1
            return value
        with patch.object(c,'strict_json',side_effect=nonzero),self.assertRaisesRegex(ValueError,'child_exit'):c.verify_completion(self.out)
        (self.out/'receipt.json').unlink()
        with self.assertRaises(FileNotFoundError):c.verify_completion(self.out)

    def test_finalization_error_never_completes(self):
        real=c.write_json
        def fail(path,value):
            if path.name=='receipt.json':raise OSError('injected receipt')
            return real(path,value)
        with patch.object(c,'write_json',side_effect=fail),self.assertRaises(OSError):c.finalize(self.out,dict(status='passed'))
        self.assertTrue((self.out/'result.json').exists());self.assertFalse((self.out/'receipt.json').exists())

    def test_child_logs_original_nonzero_exit(self):
        popen=entry.subprocess.Popen
        def cpu(command,**kw):
            self.assertEqual(command[0],sys.executable)
            return popen([sys.executable,'-B','-c',"import sys;print('o'*12000);print('e'*12000,file=sys.stderr);sys.exit(4)"],**kw)
        with patch.object(entry.subprocess,'Popen',side_effect=cpu),self.assertRaisesRegex(ValueError,'child_failed'):
            entry.child('worker',self.out,20)
        self.assertEqual((self.out/'worker.stdout.log').read_text(),'o'*12000+'\n')
        self.assertEqual((self.out/'worker.stderr.log').read_text(),'e'*12000+'\n')
        self.assertEqual(c.strict_json((self.out/'worker-exit.json').read_bytes())['exit_code'],4)

    def test_full_result_aggregator_rejects_reader_identity_and_missing_evidence(self):
        # Controlled evidence only. No actual dataset files are read/written;
        # their encoded identities are mocked at this unit-test boundary.
        ref=c.strict_json((ROOT/'formal_dataset_reference.json').read_bytes())
        candidate=c.candidate();candidate['output']=str(self.out/'output')
        trace,spec=small_trace();spec.update(rows=16384)
        spec['train']=[['train',i,f[0]] for i,f in enumerate(ref['frames']['train'])]
        spec['evaluation']=[spec['train'][i] for i in (5,10,15,20,25)]+[
            ['test',i,f[0]] for i,f in enumerate(ref['frames']['test'])]
        trace['batches']=[[spec['train'][0]] for _ in range(3)]
        for row in trace['steps']:row['rows']=16384
        trace['forwards']=345
        trace['evaluations']=[dict(k=k,views=[dict(frame=f,metrics=dict(l1=[.1],psnr=[10.,11.,12.],ssim=[.9],msssim=[.8]))
                                         for f in spec['evaluation']]) for k in (0,3)]
        for row in trace['checkpoints']:
            directory=self.out/'output/checkpoints'/('completed-'+str(row['k']));directory.mkdir(parents=True)
            for name in ('complete.json','state.pt','metadata.json'):(directory/name).write_bytes(b'CPU identity fixture')
            row.update(path=str(directory),files={n:c.file_identity(directory/n) for n in ('complete.json','state.pt','metadata.json')})
            row['marker']=row['files']['complete.json']
        (self.out/'expected.pt').write_bytes(b'CPU expected identity fixture')
        consumed={str(Path(ref['source_path'])/path):c.file_identity(self.out/'expected.pt')
                  for split,index,_ in spec['evaluation']+[spec['train'][0]] for path in ref['frames'][split][index][::2]}
        worker=dict(status='passed',trace=trace,spec=spec,entry_calls=1,returned=3,stage='normal_return',
            writer=dict(present=False,flush=0,close=0),diagnostic=dict(attempted=3,completed=3,saves=2,tests=1,last_confirmed_save=3),
            representatives=candidate['representatives'],expected_identity=c.file_identity(self.out/'expected.pt'),
            representative_forwards=3,consumed=consumed)
        reader=dict(status='passed',representatives=worker['representatives'],checkpoint=trace['checkpoints'][-1],
            parameter_groups=sorted(c.TOLERANCES_GROUPS),exact=True,reader_builds=0,forwards=3)
        for name,value in (('inputs',candidate),('worker-result',worker),('reader-result',reader)):
            c.write_json(self.out/(name+'.json'),value)
        identity=c.file_identity;parse=c.strict_json
        def mocked_identity(path):return consumed[str(path)] if str(path) in consumed else identity(path)
        with patch.object(c,'verify_candidate'),patch.object(c,'file_identity',side_effect=mocked_identity):
            c.validate_results(self.out)
            for mode in ('reader_missing','reader_extra','wrong_frame','wrong_marker','writer','consumed','expected','extra_test'):
                def mutate(data,*a,**kw):
                    value=parse(data,*a,**kw)
                    if 'parameter_groups' in value:
                        if mode=='reader_missing':value['parameter_groups'].pop()
                        elif mode=='reader_extra':value['parameter_groups'].append('xyz')
                        elif mode=='wrong_frame':value['representatives'][0][1]=1
                        elif mode=='wrong_marker':value['checkpoint']['k']=2
                    if 'entry_calls' in value:
                        if mode=='writer':value['writer']['flush']=1
                        elif mode=='consumed':value['consumed'].pop(next(iter(value['consumed'])))
                        elif mode=='expected':value['expected_identity']['size']+=1
                        elif mode=='extra_test':value['spec']['evaluation'].append(value['spec']['evaluation'][-1])
                    return value
                with self.subTest(mode=mode),patch.object(c,'strict_json',side_effect=mutate),self.assertRaises(ValueError):
                    c.validate_results(self.out)


from formal_training_fixture import CPUCase


class ConsumerTests(CPUCase):
    def prepare(self):
        import training_update_runtime as fixture
        import formal_inputs
        from test_formal_config import encode
        case=dict(name='real_input_control',n=3,batch=1,kind='initial')
        value,ref=fixture.create_fixture(self.f.base/'control',case)
        value['optimization']['population'].update(densify_from_update=500,densify_until_update=15000,
            densification_interval=100,opacity_reset_interval=3000)
        value['optimization']['sh']['increase_interval']=1000
        value['checkpoint']['intermediate_updates']=[1];value['reporting']['test_updates']=[3]
        self.owner=patch.object(formal_inputs,'_reference',return_value=ref);self.owner.start();self.addCleanup(self.owner.stop)
        self.p=self.train.prepare_formal_runtime(formal_inputs.verify_inputs(encode(value)),native_evidence=self.f.checkpoint_evidence())
        return self.p

    def test_real_consumers_observer_three_updates_save_report_and_cpu_reader(self):
        from real_input_runtime import Observation,state,rng_state
        from test_training_update_cpu import ConsumerTests as Control,Event
        from formal_checkpoint_state import restore_diagnostic,equal_state
        import formal_entry,formal_checkpoint as cp
        from test_formal_checkpoint import assert_expected
        p=self.prepare();control=Control('runTest');control.torch=self.torch
        captures={}
        def save(k,expected):captures[k]=expected
        claim=formal_entry.OutputClaim(p.inputs);self.addCleanup(claim.close)
        with patch.object(self.train,'TENSORBOARD_FOUND',False),patch.object(self.torch.cuda,'Event',Event):
            writer=self.train.begin_formal_output(p,claim)
            obs=Observation(p,control.renderer,save_observer=save)
            returned=obs.run(self.train.continue_formal_training,writer)
        self.assertEqual(returned,3);self.assertEqual(sorted(captures),[1,3]);self.assertEqual(len(obs.consumed.identities),10)
        c.validate_trace(obs.trace,obs.spec)
        read=cp.read_checkpoint(Path(p.scene.model_path)/'checkpoints/completed-3')
        assert_expected(self,captures[3],read)
        with patch.object(type(p.gaussians),'training_setup',side_effect=AssertionError('reader training')):
            model=restore_diagnostic(read,'cpu')
        self.assertIsNone(model.optimizer)
        from test_formal_checkpoint import SLOTS
        for slot,attr in SLOTS.items():equal_state(getattr(model,attr),captures[3][0][slot])
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_getitem_single_delegate_identity_and_exception(self):
        from real_input_runtime import ConsumedFiles
        p=self.prepare();dataset=p.scene.getTrainCameras();original=type(dataset).__getitem__
        recorder=ConsumedFiles();calls=[];held=[]
        def delegate(*a):calls.append(1);value=original(*a);held.append(value);return value
        result=recorder.getitem(delegate,dataset,0)
        self.assertIs(result,held[0]);self.assertEqual(calls,[1]);self.assertEqual(len(recorder.finish()),2)
        error=RuntimeError('original failure')
        def fail(*a):raise error
        with self.assertRaises(RuntimeError) as got:recorder.getitem(fail,dataset,0)
        self.assertIs(got.exception,error)
        Path(result[1].image_path).write_bytes(b'changed temporary fixture')
        with self.assertRaisesRegex(ValueError,'consumed'):recorder.finish()

    def test_metric_return_and_positive_infinity_evidence(self):
        from real_input_runtime import Observation
        p=self.prepare();obs=Observation(p,lambda *a:None)
        obs.mode=0;obs.current_view=dict(metrics={})
        image=self.torch.ones(3,2,2);values=self.torch.tensor([[float('inf')],[1.],[2.]])
        self.assertIs(obs.metric('psnr',lambda *a:values,image,image),values)
        self.assertEqual(obs.current_view['metrics']['psnr'][0],dict(positive_infinity=True,zero_mse=True))
        obs.current_view=dict(metrics={})
        with self.assertRaisesRegex(ValueError,'zero_mse'):
            obs.metric('psnr',lambda *a:values,image,image*0)


if __name__=='__main__':
    if sys.executable!=c.PYTHON:raise SystemExit('4dgs310 required')
    unittest.main()
