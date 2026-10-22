"""4dgs310 CPU-only. Real serializer, camera, PNG, M/T and publication.

Synthetic inputs and native boundary only; never load the production tensors.
"""
import copy
from contextlib import ExitStack,contextmanager
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests'),str(ROOT/'tests/cuda')]
import staged_comparison_adopted_input as ai
import adopted_reference_bundle as b
import adopted_reference_runtime as rt
import staged_comparison_checkpoint as cp
import staged_comparison_support as support
from formal_checkpoint_identity import canonical,file_identity,byte_identity
from formal_training_fixture import CPUCase
import test_comparison_pipeline as pipeline_fixture


class InputTests(unittest.TestCase):
    def test_stdlib_gate_rejects_before_heavy_or_output(self):
        code='''
import sys
sys.path.insert(0,sys.argv[1])
class Deny:
 def find_spec(self,name,*a):
  if name.split('.')[0] in ('torch','numpy','PIL','train','scene','gaussian_renderer'):
   raise AssertionError('heavy before validation')
sys.meta_path.insert(0,Deny())
import staged_comparison_adopted_input as i
try:i.verify({})
except ValueError:pass
else:raise AssertionError('accepted')
print(sys.executable)
'''
        result=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(ROOT)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),support.PYTHON)

    def test_historical_and_current_identity_bridge(self):
        with tempfile.TemporaryDirectory() as tmp:
            here=Path(tmp);(here/'before').mkdir()
            old_bytes=b'preserved reader';(here/'before/staged_comparison_checkpoint.py').write_bytes(old_bytes)
            old={'staged_comparison_checkpoint.py':byte_identity(old_bytes),'train.py':byte_identity(b'train')}
            current=dict(old,**{'staged_comparison_checkpoint.py':byte_identity(b'new reader'),
                               ai.ADDED_SOURCE:byte_identity(b'new adopted helper')})
            execution={'previous':byte_identity(b'old')}
            new_execution=dict(execution,**{str(ROOT/n):byte_identity(n.encode()) for n in ai.LOCAL if n.startswith('tests/')})
            origin={'environment':{'python':'synthetic'}}
            meta=dict(base=dict(provenance=dict(connection_source=old,native_origin=origin),execution=execution))
            with patch.object(ai,'HERE',here),patch.object(ai,'source_snapshot',return_value=current),\
                 patch.object(support,'execution_files',return_value=new_execution),\
                 patch.object(cp,'_trusted_native',return_value=origin),\
                 patch.object(cp,'provenance',side_effect=lambda v:v) as native_boundary,\
                 patch.object(ai,'current_files',return_value={'reviewed-current':byte_identity(b'current')}):
                result=ai.bridge(meta)
                self.assertEqual(result['historical_source'],old)
                self.assertEqual(result['current_provenance']['connection_source'],current)
                native_boundary.assert_called_once()
                for mutation in ('changed','extra','missing'):
                    bad=dict(current)
                    if mutation=='changed':bad['train.py']=byte_identity(b'changed')
                    if mutation=='extra':bad['other.py']=byte_identity(b'extra')
                    if mutation=='missing':del bad['train.py']
                    with self.subTest(mutation=mutation),patch.object(ai,'source_snapshot',return_value=bad):
                        with self.assertRaises(ValueError):ai.bridge(meta)
                with patch.object(support,'execution_files',return_value=execution):
                    with self.assertRaisesRegex(ValueError,'execution_delta'):ai.bridge(meta)
                (here/'before/staged_comparison_checkpoint.py').write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError,'before_bytes'):ai.bridge(meta)

    def test_selected_real_small_records_and_current_binding_only(self):
        # This is metadata/source/binary FILE hashing only; no tensor/native load.
        value=ai.plan();proof=ai.verify(value)
        self.assertEqual(ai.checked_proof(proof),value['selected'])
        self.assertFalse(value['execution_authorized']);self.assertFalse(value['executable'])
        with self.assertRaisesRegex(ValueError,'separate_live_authorization'):ai.verify(value,authorized=True)
        for field in ('selected','bridge'):
            bad=copy.deepcopy(value);bad[field]={}
            with self.subTest(field=field),self.assertRaises(ValueError):ai.verify(bad)
        with self.assertRaisesRegex(ValueError,'proof_type'):cp.read_adopted_checkpoint(True)


class ActualCPU(CPUCase):
    initial=pipeline_fixture.FinalConnectionTests.initial
    bundle=pipeline_fixture.FinalConnectionTests.bundle
    provenance_fixture=pipeline_fixture.FinalConnectionTests.provenance_fixture
    final_fixture=pipeline_fixture.FinalConnectionTests.final_fixture

    def capture(self,n=3):
        t=self.torch;a=[None]*19
        for slot,shape in ((1,(3,)),(2,(1,3)),(3,(15,3)),(4,(3,)),(5,(4,)),
                           (6,(1,)),(13,(1,)),(14,(1,)),(15,(4,))):
            count=n
            for width in shape:count*=width
            a[slot]=(t.arange(count,dtype=t.float32).reshape(n,*shape)%97+slot)/100
        a[4]=-a[4];a[14]=-a[14];a[6]=-a[6];a[1][0,0]=-0.
        a[0]=a[16]=0
        return a

    def test_serializer_independent_all_rows_and_negative_correspondence(self):
        a=self.capture(8201);path=self.f.base/'population.splat4d'
        module=b.serializer(ai.SERIALIZER,file_identity(ai.SERIALIZER))
        result=b.export_spl4(path,a,module)
        self.assertEqual(result['identity']['size'],128+8201*260)
        self.assertEqual(result['ranges'][1]['start'],4)
        saved=path.read_bytes()
        self.assertEqual(struct.unpack('<f',saved[128:132]),(-0.,))
        for problem in ('short','extra','flag','row','left-right','sh','scale','opacity','nan'):
            data=bytearray(saved)
            if problem=='short':data=data[:-1]
            elif problem=='extra':data.extend(b'x')
            elif problem=='flag':data[22]=0
            elif problem=='row':data[128:388],data[388:648]=data[388:648],data[128:388]
            elif problem=='left-right':data[140:156],data[156:172]=data[156:172],data[140:156]
            else:
                offset={'sh':17,'scale':11,'opacity':62,'nan':0}[problem]
                struct.pack_into('<f',data,128+4*offset,float('nan') if problem=='nan' else .999)
            path.write_bytes(data)
            with self.subTest(problem=problem),self.assertRaises((ValueError,OSError)):b.check_spl4(path,a)
        path.write_bytes(saved)

    def test_native_observation_actual_renderer_and_camera_copy(self):
        from formal_views import ConsumerView
        from checkpoint_cuda_acceptance import diagnostic_camera,camera_values
        from real_input_cuda_acceptance import render_buffers
        bundle=self.bundle();p=self.train.prepare_comparison_runtime(bundle);m=p.gaussians
        m.env_map=None
        c=bundle.config;rep=['train',1,self.f.ref['frames']['train'][1][0],16.5]
        cam=diagnostic_camera(c,tuple(rep[:3]));original_camera=camera_values(cam)
        native=self.f.renderer.diff_gaussian_rasterization._C
        calls=[];returned=[];observations=[]
        t=self.torch
        def kernel(*args):
            calls.append(args);h,w=args[17:19];n=len(args[1])
            result=(7,t.zeros(3,h,w),t.zeros(2,h,w),t.ones(1,h,w),t.zeros(1,h,w),
                    t.ones(n,dtype=t.int32),t.zeros(2,dtype=t.uint8),t.zeros(3,dtype=t.uint8),
                    t.zeros(4,dtype=t.uint8),t.zeros(n,6),t.zeros(n,3),t.empty(0),t.empty(0))
            returned.append(result);return result
        def observed_renderer(device,*args,**kw):
            self.assertIsNot(device,cam)
            with rt.observe_native(native,device,len(m._xyz)) as records:
                value=self.f.renderer.render(device,*args,**kw)
            observations.extend(records);return value
        with patch.object(native,'rasterize_gaussians',kernel,create=True):
            actual=render_buffers(observed_renderer,cam,m,c)
            self.assertIs(native.rasterize_gaussians,kernel)
        self.assertEqual(len(calls),1);self.assertEqual(observations[0]['tile_references'],7)
        self.assertEqual(observations[0]['population_rows'],2)
        self.assertEqual(camera_values(cam),original_camera)
        self.assertEqual(actual['render'].shape,(3,10,20))
        device=cam.cuda();args=list(calls[0]);args[13]=device.world_view_transform
        args[14]=device.full_proj_transform;args[22]=device.camera_center
        error=RuntimeError('native-boundary-failure')
        def broken(*a):raise error
        with patch.object(native,'rasterize_gaussians',broken,create=True):
            try:
                with rt.observe_native(native,device,2):native.rasterize_gaussians(*args)
            except RuntimeError as found:self.assertIs(found,error)
            else:self.fail('exception swallowed')
            self.assertIs(native.rasterize_gaussians,broken)
        with patch.object(native,'rasterize_gaussians',kernel,create=True):
            with self.assertRaisesRegex(ValueError,'missing_native_call'):
                with rt.observe_native(native,device,2):pass
            with self.assertRaisesRegex(ValueError,'multiple_native_calls'):
                with rt.observe_native(native,device,2):
                    self.assertIs(native.rasterize_gaussians(*args),returned[-1])
                    native.rasterize_gaussians(*args)

    def test_actual_camera_gt_raw_png_and_independent_png_check(self):
        from real_input_runtime import ConsumedFiles
        from checkpoint_cuda_acceptance import camera_values
        from PIL import Image
        import numpy as np
        c=self.bundle().config;rep=['train',1,self.f.ref['frames']['train'][1][0],16.5]
        paths=[str(self.f.source/rep[2]),str(self.f.source/rep[2]).replace('/images/','/masks/')]
        old=dict(consumed={p:file_identity(p) for p in paths})
        consumed=ConsumedFiles();cam,gt=rt.get_view(c,rep,consumed,old);consumed.finish()
        independent=rt.independent_gt(c,rep,old['consumed'])
        rt.exact_arrays({'gt':gt},{'gt':independent})
        self.assertEqual(gt.shape,(3,10,20));self.assertTrue(bool((cam.gt_alpha_mask==1).all()))
        self.assertAlmostEqual(float(gt[0,0,0]),50/255,places=6)
        t=self.torch
        raw=dict(render=t.full((3,10,20),.5),alpha=t.full((1,10,20),.5),
            depth=t.full((1,10,20),7.),radii=t.ones(2,dtype=t.int32),
            visibility_filter=t.ones(2,dtype=t.bool),gt=gt)
        record=dict(camera=rt.camera_record(cam),native={'population_rows':2})
        root=self.f.base/'view';rt.save_view(root,raw,record)
        with Image.open(root/'render.png') as image:self.assertTrue(np.all(np.asarray(image)==127))
        with Image.open(root/'depth.png') as image:self.assertFalse(np.asarray(image).any())
        prior=self.f.base/'prior';directory=prior/'fixed/12000';directory.mkdir(parents=True)
        # Independent expected pixels, not copied output PNGs.
        arrays=dict(render=np.full((10,20,3),127,np.uint8),alpha=np.full((10,20,3),127,np.uint8),
            depth=np.zeros((10,20,3),np.uint8),gt=np.tile(np.array([50,20,5],dtype=np.uint8),(10,20,1)))
        for name,a in arrays.items():Image.fromarray(a).save(directory/(str(rep[1])+'-'+name+'.png'))
        row=dict(frame=rep[:3],images={name:file_identity(directory/(str(rep[1])+'-'+name+'.png')) for name in arrays})
        old['trace']=dict(fixed=[dict(views=[row])])
        with patch.object(ai,'RUN',prior):
            rt.check_png(root,{n:rt.array(v) for n,v in raw.items()},old,rep)
            # A raw corruption invisible to 8-bit PNG must still be rejected.
            changed={n:rt.array(v).copy() for n,v in raw.items()}
            changed['gt'][0,0,0]+=np.float32(.00001)
            rt.check_png(root,changed,old,rep)
            with self.assertRaisesRegex(ValueError,'raw_exact:gt'):
                rt.exact_arrays({'gt':changed['gt']},{'gt':rt.independent_gt(c,rep,old['consumed'])})
            Image.fromarray(np.full((10,20,3),128,np.uint8)).save(root/'render.png')
            with self.assertRaisesRegex(ValueError,'png_quantization'):
                rt.check_png(root,{n:rt.array(v) for n,v in raw.items()},old,rep)
        bad=dict(old,consumed={})
        with self.assertRaises(KeyError):rt.get_view(c,rep,ConsumedFiles(),bad)

    def test_selected_reader_reuses_real_M_T_and_rejects_wrong_binding(self):
        run,receipt,loaded,attrs,groups=self.final_fixture()
        from staged_comparison_observation import expected_equal
        checkpoint=Path(loaded['config'].paths.output_directory)/'checkpoints/completed-12000'
        meta=cp._read_record(checkpoint/'metadata.json')
        selected=dict(checkpoint=str(checkpoint),marker=file_identity(checkpoint/'complete.json'),
                      metadata=file_identity(checkpoint/'metadata.json'))
        # Pin a tiny selected fixture in place of the fixed production authority;
        # real metadata/serializer/state/dataset/native tests are not stubbed.
        value=dict(schema=ai.SCHEMA,execution_authorized=False,executable=False,review_required=True,
            selected=selected,bridge={'fixture':'reviewed'},output=str(ai.OUTPUT),python=support.PYTHON,
            entry=str(ROOT/'tests/cuda/adopted_reference_acceptance.py'),resources=support.RESOURCES,
            unverified=[],gate_c_passed=False)
        with patch.object(ai,'CHECKPOINT',checkpoint),patch.object(ai,'selected',return_value=(meta,selected,{})),\
             patch.object(ai,'bridge',return_value=value['bridge']):
            proof=ai.verify(value);actual=cp.read_adopted_checkpoint(proof)
            expected_equal((attrs,groups),actual)
            bad=copy.deepcopy(meta);bad['k']-=1
            with self.assertRaisesRegex(ValueError,'metadata_binding'):
                cp.validate_metadata(bad,checkpoint.parent.parent,_adopted=proof)
            bad_proof=ai._Proof(canonical(dict(value,selected=dict(selected,checkpoint='/wrong'))))
            with patch.object(self.torch,'load',side_effect=AssertionError('T reached')):
                with self.assertRaises(ValueError):cp.read_adopted_checkpoint(bad_proof)

    def test_producer_separate_reader_and_nonoverwrite_publication(self):
        from checkpoint_cuda_acceptance import diagnostic_camera,camera_values
        from tools.export_adopted_reference import parent
        import numpy as np
        run,_,loaded,attrs,groups=self.final_fixture()
        checkpoint=Path(loaded['config'].paths.output_directory)/'checkpoints/completed-12000'
        reps=[['train',i%3,*self.f.ref['frames']['train'][i%3][:2]] for i in range(14)]
        expected=[];pictures=[];consumed={};oldrun=self.f.base/'reference-old';directory=oldrun/'fixed/12000'
        directory.mkdir(parents=True)
        for rep in reps:
            cam=diagnostic_camera(loaded['config'],tuple(rep[:3]));t=self.torch
            expected.append(dict(camera=camera_values(cam),output=dict(render=t.zeros(3,10,20),
                alpha=t.ones(1,10,20),depth=t.ones(1,10,20),radii=t.ones(2,dtype=t.int32),
                visibility_filter=t.ones(2,dtype=t.bool))))
            pixels=dict(render=np.zeros((10,20,3),np.uint8),alpha=np.full((10,20,3),255,np.uint8),
                depth=np.zeros((10,20,3),np.uint8),gt=np.tile(np.array([50,20,5],np.uint8),(10,20,1)))
            for name,a in pixels.items():self.f.Image.fromarray(a).save(directory/(str(rep[1])+'-'+name+'.png'))
            pictures.append(dict(frame=rep[:3],images={n:file_identity(directory/(str(rep[1])+'-'+n+'.png')) for n in pixels}))
            for p in (cam.image_path,cam.image_path.replace('/images/','/masks/')):consumed[p]=file_identity(p)
        exp=self.f.base/'reference-expected.pt';t.save(dict(attrs=attrs,groups=groups,views=expected),exp)
        b.write_json(oldrun/'worker-result.json',dict(consumed=consumed,trace=dict(fixed=[dict(views=pictures)])))
        meta=cp._read_record(checkpoint/'metadata.json')
        selected=dict(checkpoint=str(checkpoint),marker=file_identity(checkpoint/'complete.json'),
            metadata=file_identity(checkpoint/'metadata.json'),expected=dict(path=str(exp),identity=file_identity(exp)),representatives=reps)
        bridge=dict(current_provenance=self.provenance_fixture(),current_files={str(ai.SERIALIZER):file_identity(ai.SERIALIZER)})
        output=self.f.base/'reference-output';candidate=self.f.base/'reference-candidate.json'
        value=dict(schema=ai.SCHEMA,execution_authorized=True,executable=True,review_required=True,
            selected=selected,bridge=bridge,output=str(output),python=support.PYTHON,
            entry=str(ROOT/'tests/cuda/adopted_reference_acceptance.py'),resources=support.RESOURCES,
            unverified=['CPU synthetic fixture only'],gate_c_passed=False)
        b.write_json(candidate,value)
        control=dict(value=value,reference=self.f.ref,checkpoint=str(checkpoint),run=str(oldrun),candidate=str(candidate),
            initial=meta['base']['parent']['path'],owner=str(self.f.base))
        controlpath=self.f.base/'reference-control.json';b.write_json(controlpath,control)
        monitor=support.monitor();launch=monitor.monitored_child
        def child(command,**kw):
            self.assertEqual(command[-2],kw['mode'])
            command=[support.PYTHON,'-I','-B',str(Path(__file__).resolve()),'--cpu-child',str(controlpath),kw['mode']]
            kw.update(interval=.005,sample_fn=lambda *a:dict(host_available=10**15,disk_free=10**15,
                rss_current=1,rss_high_water=1,swap=0))
            return launch(command,**kw)
        with cpu_authority(control),patch.object(monitor,'monitored_child',side_effect=child):
            try:parent(value)
            except BaseException:
                # Preserve useful child output in this test's captured report log.
                for p in sorted(output.glob('*.log')):print(p.name,p.read_text(),flush=True)
                raise
            receipt=(output/'receipt.json').read_bytes()
            with self.assertRaises(ValueError):parent(value)
            self.assertEqual((output/'receipt.json').read_bytes(),receipt)
        self.assertFalse((output/'.pending-bundle').exists());self.assertFalse((output/'failure.json').exists())
        self.assertEqual(b.read_json(output/'result.json')['status'],'verified_requires_review')
        verified=b.read_json(output/'bundle/verification.json')
        self.assertTrue(verified['gt14_raw_exact'] and verified['raw14_exact'] and verified['all_rows'])
        pids=[b.read_json(output/(m+'-cpu-process.json'))['pid'] for m in ('worker','reader')]
        self.assertEqual(len(set(pids+[os.getpid()])),3)
        for mode in ('worker','reader'):support.process_record(output,mode)
        b.check_index(output/'bundle',b.read_json(output/'bundle/manifest.json'))


class ResourceTests(unittest.TestCase):
    def test_incremental_host_gpu_and_unchanged_margins(self):
        n=3248901;before=rt.incremental_demand('cpu_load',n);after=rt.incremental_demand('load',n)
        view=rt.incremental_demand('render',n)
        self.assertIsNone(before['gpu_needed'])
        self.assertEqual(before['host_payload'],support.demand('reader_load',n)['host_payload'])
        self.assertEqual(after['host_payload'],128*1280*720+16*1024**2)
        self.assertLess(after['host_needed'],before['host_needed'])
        self.assertEqual(after['gpu_payload'],2*(260*n+16*n)+128*1280*720)
        self.assertEqual(view['gpu_payload'],128*1280*720+5*n)
        judge=support.resources().judge
        for d in (before,after,view):
            self.assertEqual(d['host_margin'],64*1024**2+(d['host_payload']+3)//4)
            kw={} if d['gpu_needed'] is None else dict(free=d['gpu_needed'])
            self.assertEqual(judge(d,d['host_needed'],**kw)['status'],'passed')
            self.assertEqual(judge(d,d['host_needed']-1,**kw)['status'],'failed')
            if kw:self.assertEqual(judge(d,d['host_needed'],free=d['gpu_needed']-1)['status'],'failed')

    def test_remaining_payload_headers_records_and_reserve(self):
        n=3248901;wh=1280*720;png=4*(3*wh+720+(3*wh+720+99)//100+65536)
        perview=32*wh+5*n+png+6*256;record=16*1024**2
        full=128+260*n+14*perview+record
        self.assertEqual(rt.remaining_disk('bundle',n),full)
        for i in range(14):
            self.assertEqual(rt.remaining_disk('view-'+str(i),n,123),(14-i)*perview+record-123)
        self.assertEqual(rt.remaining_disk('records',n,123),record-123)
        for bad in ('view-14','view--1','arbitrary'):
            with self.assertRaises(ValueError):rt.remaining_disk(bad,n)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'.pending-bundle').mkdir()
            (p/'.pending-bundle/population.json').write_bytes(b'{}')
            need=rt.remaining_disk('view-0',n,2);reserve=support.RESOURCES['disk_reserve_bytes']
            res=rt.resource_adapter()
            with patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=need+reserve)):
                self.assertEqual(res.disk(p,'view-0',n)['new_write_estimate'],need)
            with patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=need+reserve-1)):
                with self.assertRaisesRegex(ValueError,'save_disk_budget'):res.disk(p,'view-0',n)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)

    def fixture(self,name):
        root=self.base/name
        claim=b.Claim(root);self.addCleanup(claim.close)
        p=root/'.pending-bundle';p.mkdir()
        b.write(p/'population.splat4d',lambda f:f.write(b'fixture'))
        b.write_json(p/'population.json',{'fixture':True})
        manifest=dict(schema=b.SCHEMA,version=1,input={},provenance={},population={},references=[],
            encoding=b.ENCODING,limitations=b.LIMITATIONS,verification='verification.json')
        verification=dict(status='verified_requires_review')
        b.write_json(p/'manifest.json',manifest);b.write_json(p/'verification.json',verification)
        b.index_bundle(p,manifest)
        return root,manifest,verification

    def test_exclusive_atomic_publish_readback_and_no_self_hash_cycle(self):
        root,m,v=self.fixture('good');receipt=b.publish(root,m,v)
        self.assertEqual(receipt['index'],file_identity(root/'bundle/index.json'))
        self.assertFalse((root/'.pending-bundle').exists())
        self.assertNotIn('index.json',b.read_json(root/'bundle/index.json')['members'])
        with self.assertRaises(FileExistsError):b.Claim(root)
        with self.assertRaises(FileNotFoundError):b.publish(root,m,v)

    def test_reject_missing_extra_symlink_hash_and_unknown_manifest(self):
        for problem in ('missing','extra','symlink','hash','schema'):
            root,m,v=self.fixture(problem);p=root/'.pending-bundle'
            if problem=='missing':(p/'population.json').unlink()
            if problem=='extra':(p/'extra').write_text('unlisted')
            if problem=='symlink':(p/'population.json').unlink();(p/'population.json').symlink_to(p/'manifest.json')
            if problem=='hash':(p/'population.json').write_text('{}')
            if problem=='schema':(p/'manifest.json').write_bytes(canonical(dict(m,unknown=True)))
            with self.subTest(problem=problem),self.assertRaises((ValueError,OSError)):b.publish(root,m,v)
            self.assertFalse((root/'receipt.json').exists());self.assertTrue(p.exists())

    def test_existing_destination_receipt_failure_and_preserved_partial(self):
        root,m,v=self.fixture('collision');(root/'bundle').mkdir()
        with self.assertRaises(FileExistsError):b.publish(root,m,v)
        self.assertTrue((root/'.pending-bundle').exists());self.assertFalse((root/'receipt.json').exists())
        root,m,v=self.fixture('receipt-failure');write=b.write_json
        def fail(path,value):
            if Path(path).name=='.pending-receipt.json':raise OSError('injected receipt write')
            return write(path,value)
        with patch.object(b,'write_json',side_effect=fail):
            with self.assertRaises(OSError):b.publish(root,m,v)
        b.failure(root,'receipt',OSError('injected receipt write'))
        self.assertTrue((root/'bundle').exists());self.assertFalse((root/'receipt.json').exists())
        self.assertEqual(b.read_json(root/'failure.json')['status'],'incomplete')


@contextmanager
def cpu_authority(control):
    """Fixed tiny authority instead of production inputs, not reader stubs."""
    import comparison_inputs
    import formal_inputs
    value=control['value'];ck=Path(control['checkpoint']);meta=cp._read_record(ck/'metadata.json')
    with ExitStack() as stack:
        for name,val in (('CHECKPOINT',ck),('RUN',Path(control['run'])),('OUTPUT',Path(value['output'])),
                         ('CANDIDATE',Path(control['candidate']))):stack.enter_context(patch.object(ai,name,val))
        stack.enter_context(patch.object(ai,'selected',return_value=(meta,value['selected'],{})))
        stack.enter_context(patch.object(ai,'bridge',return_value=value['bridge']))
        stack.enter_context(patch.object(formal_inputs,'_reference',return_value=control['reference']))
        stack.enter_context(patch.object(cp,'INITIAL_PATH',control['initial']))
        stack.enter_context(patch.object(comparison_inputs,'OUTPUT_OWNER',Path(control['owner'])))
        yield


def cpu_child(controlpath,mode):
    """Fresh producer/reader interpreters; real CPU processing and publication."""
    from checkpoint_native_fixture import cpu_device_queries
    assert sys.executable==support.PYTHON and mode in ('worker','reader')
    def audit(event,args):
        if event=='subprocess.Popen' and any(Path(str(w)).name in ('nvidia-smi','nvcc','ninja','make','cmake')
                for w in (args[1] if isinstance(args[1],(list,tuple)) else [args[1]])):
            raise AssertionError('CPU build/device query forbidden')
    sys.addaudithook(audit)
    control=b.read_json(controlpath);value=control['value']
    with cpu_device_queries(),ExitStack() as stack:
        import torch
        torch.set_num_threads(1)
        if mode=='worker':
            CPUCase.setUpClass();stack.callback(CPUCase.doClassCleanups)
            fixture=CPUCase('runTest');fixture.setUp();stack.callback(fixture.doCleanups)
            import staged_comparison_worker as worker
            native=fixture.f.renderer.diff_gaussian_rasterization._C
            def kernel(*args):
                h,w=args[17:19];n=len(args[1]);t=torch
                return (7,t.zeros(3,h,w),t.zeros(2,h,w),t.ones(1,h,w),t.zeros(1,h,w),
                    t.ones(n,dtype=t.int32),t.zeros(2,dtype=t.uint8),t.zeros(3,dtype=t.uint8),
                    t.zeros(4,dtype=t.uint8),t.zeros(n,6),t.zeros(n,3),t.empty(0),t.empty(0))
            @contextmanager
            def native_boundary(origin):yield fixture.train,fixture.f.renderer
            stack.enter_context(patch.object(worker,'native',native_boundary))
            stack.enter_context(patch.object(native,'rasterize_gaussians',kernel,create=True))
            # Represent isolated CUDA tensors at the device-description seam;
            # all real shape/stride/scalar/camera checks remain in the reader.
            describe=rt.tensor_description
            stack.enter_context(patch.object(rt,'tensor_description',side_effect=lambda t:
                dict(describe(t),device='cuda:0' if t.numel() else 'cpu')))
            stack.enter_context(patch.object(torch.cuda,'mem_get_info',return_value=(10**15,10**15)))
            stack.enter_context(patch.object(torch.cuda,'memory_stats',return_value={'inactive_split_bytes.all.current':0}))
            stack.enter_context(patch.object(torch.cuda,'memory_allocated',return_value=0))
            stack.enter_context(patch.object(torch.cuda,'memory_reserved',return_value=0))
        else:
            class DenyNative:
                def find_spec(self,name,*args):
                    if name.split('.')[0] in ('scene','train','gaussian_renderer','pointops2','simple_knn'):
                        raise AssertionError('native dependency in independent CPU reader:'+name)
            deny=DenyNative();sys.meta_path.insert(0,deny);stack.callback(sys.meta_path.remove,deny)
        resources=support.resources()
        stack.enter_context(patch.dict(resources.memory.__globals__,collectors=lambda:{'host_available':lambda:10**15}))
        stack.enter_context(patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)))
        with cpu_authority(control):
            (rt.produce if mode=='worker' else rt.independent)(value)
        assert not torch.cuda.is_initialized()
        b.write_json(Path(value['output'])/(mode+'-cpu-process.json'),dict(pid=os.getpid(),python=sys.executable,
            cuda_initialized=False,mode=mode,native_boundary='synthetic' if mode=='worker' else 'denied'))


if __name__=='__main__':
    if sys.argv[1:2]==['--cpu-child']:cpu_child(*sys.argv[2:])
    else:unittest.main()
