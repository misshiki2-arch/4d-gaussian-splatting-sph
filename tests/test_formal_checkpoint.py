"""Real CPU file/serialization acceptance; synthetic native boundary only.

No actual build, CUDA or dataset/run mutation. Independent expectations do not
use capture() or the checkpoint decoder. All files live in temporary folders.
"""
from contextlib import ExitStack
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from formal_training_fixture import CPUCase
import formal_checkpoint as cp
import formal_checkpoint_identity as identity
import formal_entry


# Independent slot/attribute specification, deliberately not imported from writer.
SLOTS = {1:'_xyz', 2:'_features_dc', 3:'_features_rest', 4:'_scaling',
         5:'_rotation', 6:'_opacity', 7:'max_radii2D', 8:'xyz_gradient_accum',
         9:'t_gradient_accum', 10:'denom', 13:'_t', 14:'_scaling_t', 15:'_rotation_r'}


def independent(model):
    def snap(v):
        if hasattr(v, 'detach'): return v.detach().cpu().contiguous().clone()
        if isinstance(v, dict): return {k:snap(x) for k,x in v.items()}
        if isinstance(v, (tuple,list)): return type(v)(snap(x) for x in v)
        return v.item() if hasattr(v, 'item') else v
    attrs = {slot:snap(getattr(model,name)) for slot,name in SLOTS.items()}
    attrs.update({0:model.active_sh_degree, 12:float(model.spatial_lr_scale),
                  16:model.rot_4d, 17:model.env_map, 18:model.active_sh_degree_t})
    groups = {}
    for g in model.optimizer.param_groups:
        p = g['params'][0]
        groups[g['name']] = dict(settings=snap({k:v for k,v in g.items() if k!='params'}),
                                 state=snap(model.optimizer.state.get(p,{})))
    return attrs, groups


def assert_expected(test, expected, result):
    attrs, groups = expected
    a = result['state']['capture']
    test.assertEqual(len(a),19)
    for slot,value in attrs.items():
        if hasattr(value, 'dtype'):
            test.assertEqual(value.dtype,a[slot].dtype)
            test.assertEqual(tuple(value.shape),tuple(a[slot].shape))
            test.assertEqual(value.numpy().tobytes(),a[slot].numpy().tobytes())
        else: test.assertEqual(value,a[slot])
    test.assertEqual(len(a[11]['param_groups']),9)
    for g in a[11]['param_groups']:
        test.assertEqual({k:v for k,v in g.items() if k!='params'},groups[g['name']]['settings'])
        expected_state=groups[g['name']]['state']; actual=a[11]['state'].get(g['params'][0],{})
        test.assertEqual(actual.keys(),expected_state.keys())
        for name,v in expected_state.items():
            test.assertEqual(v.dtype,actual[name].dtype)
            test.assertEqual(v.numpy().tobytes(),actual[name].numpy().tobytes())


class CheckpointTests(CPUCase):
    def test_private_readback_released_before_published_readback(self):
        import weakref
        self.setup_writer(); self.update(1)
        refs=[]; pair=cp._read_pair; public=cp.read_checkpoint
        def observed_pair(*a,**kw):
            value=pair(*a,**kw)
            if not refs: refs.append(weakref.ref(value[1]['capture'][1]))
            return value
        def observed_public(*a,**kw):
            self.assertEqual(len(refs),1)
            self.assertIsNone(refs[0](), 'private state overlaps published read-back')
            return public(*a,**kw)
        expected=independent(self.m)
        with patch.object(cp,'_read_pair',observed_pair),patch.object(cp,'read_checkpoint',observed_public):
            self.p.scene.save(1)
        assert_expected(self,expected,cp.read_checkpoint(self.directory()))
        self.assertFalse(self.writer.failed)

    def test_diagnostic_camera_real_png_extension_frame_and_time(self):
        import ast
        import inspect
        from test_checkpoint_cuda_preflight import entry
        bundle=self.f.verify();c=bundle.config
        self.assertEqual(c.dataset.extension,'')
        # The actual CUDA reader must use this same reconstruction function.
        calls=[node for node in ast.walk(ast.parse(inspect.getsource(entry.reader)))
               if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
               and node.func.id=='diagnostic_camera']
        self.assertEqual(len(calls),1)
        with patch.object(self.f.readers,'readCamerasFromTransforms',
                          wraps=self.f.readers.readCamerasFromTransforms) as read, \
                patch.object(self.f.readers.imagesize,'get',wraps=self.f.readers.imagesize.get) as size:
            for split,count in (('train',3),('test',1)):
                for index in range(count):
                    key=(split,index,f'images/{split}{index}.png')
                    with self.subTest(key=key):
                        cam=entry.diagnostic_camera(c,key)
                        self.assertEqual(read.call_args.kwargs['extension'],'')
                        self.assertEqual((cam.formal_frame.split,cam.formal_frame.index,
                                          cam.formal_frame.file_path),key)
                        self.assertEqual(cam.formal_camera,getattr(bundle,split+'_cameras')[index])
                        self.assertEqual((cam.uid,cam.colmap_id),(index,index))
                        self.assertEqual(Path(cam.image_path),self.f.source/key[2])
                        with self.f.Image.open(cam.image_path) as image:
                            self.assertEqual(image.format,'PNG')
                            self.assertEqual(image.size,(20,10))
                            self.assertEqual(image.getpixel((0,0)),(100,40,10,128))
                        self.assertEqual((cam.image_width,cam.image_height),(20,10))
                        self.assertEqual((cam.fl_x,cam.fl_y,cam.cx,cam.cy),(15.,15.,10.,5.))
                        raw_time=(-2,2,6)[index]
                        self.assertEqual(cam.formal_frame.raw_time,float(raw_time))
                        self.assertEqual(cam.timestamp,raw_time/c.time_derivation.divisor)
                        x=index+(10 if split=='test' else 0)
                        self.assertEqual(cam.camera_center.tolist(),[float(x),0.,0.])
                        world=self.f.np.diag([1.,-1.,-1.,1.]);world[0,3]=-x
                        self.f.np.testing.assert_array_equal(cam.world_view_transform.numpy(),world.T)
                        for name in ('world_view_transform','projection_matrix','full_proj_transform','camera_center'):
                            self.assertEqual(getattr(cam,name).device.type,'cpu')
            self.assertEqual(read.call_count,4)
            self.assertEqual(size.call_count,10)  # Full split: 3 train reads + 1 test read.
            self.assertTrue(all(Path(call.args[0]).is_file() for call in size.call_args_list))
        self.assertFalse(self.f.output.parent.exists())
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_diagnostic_camera_missing_image_is_rejected(self):
        from test_checkpoint_cuda_preflight import entry
        c=self.f.verify().config
        path=self.f.source/'images/train0.png'
        path.rename(path.with_suffix('.missing'))
        with self.assertRaisesRegex(ValueError,'^formal_camera_dimensions$'):
            entry.diagnostic_camera(c,('train',0,'images/train0.png'))
        self.assertFalse(self.f.output.parent.exists())
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_diagnostic_camera_wrong_image_dimensions_are_rejected(self):
        from test_checkpoint_cuda_preflight import entry
        c=self.f.verify().config
        self.f.Image.new('RGBA',(21,10),(100,40,10,128)).save(self.f.source/'images/train0.png')
        with self.assertRaisesRegex(ValueError,'^formal_camera_dimensions$'):
            entry.diagnostic_camera(c,('train',0,'images/train0.png'))
        self.assertFalse(self.f.output.parent.exists())
        self.assertFalse(self.torch.cuda.is_initialized())

    def setup_writer(self, *, n=2):
        f = self.f
        f.value['optimization'].update(total_updates=n,batch_size=1)
        f.value['optimization']['sh']['increase_interval']=1
        f.value['checkpoint']['intermediate_updates']=list(range(1,n))
        f.value['reporting']['test_updates']=[]
        self.p=self.train.prepare_formal_runtime(f.verify(),native_evidence=f.checkpoint_evidence())
        self.m=self.p.gaussians
        self.m.env_map=None  # Completed training sets this; no manufactured history.
        with self.torch.no_grad():
            for j,(slot,name) in enumerate(SLOTS.items()):
                a=getattr(self.m,name)
                if slot in (7,8,9,10): continue
                a.copy_(self.torch.arange(a.numel()).reshape(a.shape)*.0001+(j+1)*.01)
            self.m.max_radii2D.copy_(self.torch.arange(len(self.m._xyz)))
            self.m.xyz_gradient_accum.fill_(.125)
            self.m.t_gradient_accum[:,0]=self.torch.tensor([-.25,.5,-.75,1.])
            self.m.denom[:,0]=self.torch.tensor([0.,1.,2.,3.])
        self.claim=formal_entry.OutputClaim(self.p.inputs); self.addCleanup(self.claim.close)
        with patch.object(self.train,'TENSORBOARD_FOUND',False):
            self.train.begin_formal_output(self.p,self.claim)
        self.writer=self.p.scene._checkpoint_writer
        return self.p

    def update(self,k):
        self.m.oneupSHdegree()
        self.m.update_learning_rate(k)
        for j,g in enumerate(self.m.optimizer.param_groups):
            g['params'][0].grad=self.torch.full_like(g['params'][0],(j+1)*.01)
        self.m.optimizer.step(); self.m.optimizer.zero_grad(set_to_none=True)

    def directory(self,k=1): return self.f.output/'checkpoints'/('completed-'+str(k))

    def hashes(self,directory):
        return {str(p.relative_to(directory)):identity.file_identity(p) for p in directory.rglob('*') if p.is_file()}

    def modify_bundle(self, *, metadata=None, payload=None):
        d=self.directory()
        if metadata is not None: (d/'metadata.json').write_bytes(identity.canonical(metadata))
        if payload is not None: self.torch.save(payload,d/'state.pt')
        marker=json.loads((d/'complete.json').read_bytes())
        marker['files']={n:identity.file_identity(d/n) for n in ('metadata.json','state.pt')}
        (d/'complete.json').write_bytes(identity.canonical(marker))

    def fresh_reader_rejection(self, label, stage, error):
        """Public reader, real files and validators; observe imports/load/writes."""
        reference=self.f.base/'reader-reference.json'
        reference.write_bytes(identity.canonical(self.f.ref))
        before=self.hashes(self.f.output)
        marker=json.loads((self.directory()/'complete.json').read_bytes())
        self.assertEqual(marker['files'],{n:identity.file_identity(self.directory()/n)
                                         for n in ('metadata.json','state.pt')})
        code="""
import sys,json,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
directory=Path(sys.argv[2]);stage,label,error=sys.argv[4:7]
root=directory.parent.parent
seen=dict(metadata_passed=False,torch_imports=0,torch_loads=0,forbidden_imports=0,output_writes=0)
class Observe:
 def find_spec(self,n,*a):
  top=n.split('.')[0]
  if top in ('scene','train','gaussian_renderer','simple_knn') or (stage=='M' and top in ('torch','numpy')):
   seen['forbidden_imports']+=1
   raise AssertionError('forbidden import '+n)
  if n=='torch':
   assert seen['metadata_passed'], 'Torch import before M completion'
   seen['torch_imports']+=1
sys.meta_path.insert(0,Observe())
def audit(event,args):
 paths=()
 if event=='open':
  path,mode,flags=args
  if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):paths=(path,)
 elif event in ('os.mkdir','os.remove','os.rmdir'):paths=args[:1]
 elif event in ('os.rename','os.link','os.symlink'):paths=args[:2]
 for p in paths:
  if isinstance(p,(str,bytes)):
   p=Path(os.fsdecode(p))
   if not p.is_absolute() or p==root or root in p.parents:
    seen['output_writes']+=1
    raise AssertionError('reader output write')
sys.addaudithook(audit)
def observe_call(frame,event,arg):
 if event=='call' and frame.f_globals.get('__name__')=='torch.serialization' and frame.f_code.co_name=='load':
  seen['torch_loads']+=1
  assert seen['metadata_passed'] and frame.f_locals['map_location']=='cpu'
  assert frame.f_locals['weights_only'] is True
 if event=='call' and frame.f_globals.get('__name__')=='torch.cuda' and frame.f_code.co_name=='_lazy_init':
  raise AssertionError('CUDA initialization')
sys.setprofile(observe_call)
import formal_inputs,formal_checkpoint
original=formal_checkpoint.validate_metadata
def observe_metadata(*a,**kw):
 result=original(*a,**kw)
 seen['metadata_passed']=True
 return result
with patch.object(formal_inputs,'_reference',return_value=json.loads(Path(sys.argv[3]).read_bytes())), \
     patch.object(formal_checkpoint,'validate_metadata',observe_metadata):
 try:formal_checkpoint.read_checkpoint(directory)
 except ValueError as e:assert str(e)==error,(str(e),error)
 else:raise AssertionError('bad checkpoint accepted')
sys.setprofile(None)
assert seen['forbidden_imports']==seen['output_writes']==0
assert not any(n in sys.modules for n in ('scene','train','gaussian_renderer','simple_knn'))
if stage=='M':
 assert not seen['metadata_passed'] and seen['torch_loads']==seen['torch_imports']==0
 assert not any(n in sys.modules for n in ('torch','numpy','formal_checkpoint_state'))
else:
 assert seen['metadata_passed'] and seen['torch_imports']==seen['torch_loads']==1
 assert not sys.modules['torch'].cuda.is_initialized()
print('READER_BOUNDARY '+json.dumps(dict(case=label,stage=stage,error=error,**seen)),flush=True)
"""
        command=[sys.executable,'-I','-B']+(['-S'] if stage=='M' else [])
        r=subprocess.run(command+['-c',code,str(identity.ROOT),str(self.directory()),str(reference),stage,label,error],
                         capture_output=True,text=True,timeout=120)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('READER_BOUNDARY ',r.stdout)
        print(r.stdout,end='',flush=True)
        self.assertEqual(before,self.hashes(self.f.output))

    def test_optimizer_metadata_rejected_in_M_fresh_process_matrix(self):
        self.setup_writer();self.update(1);self.p.scene.save(1)
        good=json.loads((self.directory()/'metadata.json').read_bytes())
        cases=[('unknown-field',lambda m:m['optimizer']['xyz'].__setitem__('unexpected_field',True),'keys'),
               ('missing-group',lambda m:m['optimizer'].pop('xyz'),'optimizer_descriptor'),
               ('unknown-group',lambda m:m['optimizer'].__setitem__('other',{}),'optimizer_descriptor'),
               ('descriptor-type',lambda m:m.__setitem__('optimizer',[]),'optimizer_descriptor'),
               ('group-type',lambda m:m['optimizer'].__setitem__('xyz',[]),'keys'),
               ('duplicate-id',lambda m:m['optimizer']['f_dc'].__setitem__('id',m['optimizer']['xyz']['id']),'optimizer_id')]
        for key in ('slot','id','initialized','step_dtype'):
            cases.append(('missing-'+key,lambda m,k=key:m['optimizer']['xyz'].pop(k),'keys'))
        for key,values,error in (
            ('slot',(True,1.0,'1',2,None),'optimizer_binding'),
            ('id',(True,0.0,'0',-1,None),'optimizer_id'),
            ('initialized',(0,1,'true',None),'optimizer_binding'),
            ('step_dtype',(None,0,[],{},'torch.float16','float32'),'adam_step')):
            for i,value in enumerate(values):
                cases.append((key+'-type-value-'+str(i),
                              lambda m,k=key,v=value:m['optimizer']['xyz'].__setitem__(k,v),error))
        cases.append(('uninitialized-with-dtype',lambda m:m['optimizer']['xyz'].__setitem__('initialized',False),
                      'uninitialized_step'))
        for label,change,error in cases:
            with self.subTest(case=label):
                m=copy.deepcopy(good);change(m)
                self.modify_bundle(metadata=m)  # Rebind hashes: reject semantics, not bytes.
                if label=='unknown-field':
                    # Retain the exact saved reproduction with the real load spy.
                    with patch.object(self.torch,'load',wraps=self.torch.load) as load:
                        with self.assertRaisesRegex(ValueError,'checkpoint_keys'):cp.read_checkpoint(self.directory())
                        load.assert_not_called()
                self.fresh_reader_rejection(label,'M','checkpoint_'+error)

    def test_valid_metadata_bad_payload_rejected_in_T_fresh_process(self):
        self.setup_writer();self.update(1);self.p.scene.save(1)
        good=cp.read_checkpoint(self.directory())
        cases=[('moment',lambda p:p['capture'][11]['state'][0]['exp_avg_sq'].fill_(-1),'second_moment'),
               ('missing-state',lambda p:p['capture'][11]['state'].pop(0),'optimizer_binding'),
               ('bad-lr',lambda p:p['capture'][11]['param_groups'][0].__setitem__('lr',9.),'optimizer_settings'),
               ('actual-step-dtype',lambda p:p['capture'][11]['state'][0].__setitem__('step',self.torch.tensor(1,dtype=self.torch.int64)),
                'adam_step'),
               ('actual-id',lambda p:p['capture'][11]['param_groups'][0].__setitem__('params',[999]),'optimizer_binding')]
        for label,change,error in cases:
            with self.subTest(case=label):
                p=copy.deepcopy(good['state']);change(p)
                self.modify_bundle(payload=p)
                self.fresh_reader_rejection(label,'T','checkpoint_'+error)

    def test_all_admitted_step_dtypes_and_nonsequential_ids(self):
        self.setup_writer();self.update(1);self.p.scene.save(1)
        good=cp.read_checkpoint(self.directory())
        for dtype in (self.torch.float32,self.torch.float64,self.torch.int64):
            with self.subTest(dtype=str(dtype)):
                p=copy.deepcopy(good['state']);m=copy.deepcopy(good['metadata'])
                opt=p['capture'][11];states={}
                for group in opt['param_groups']:
                    old=group['params'][0];new=old*7+100
                    state=opt['state'][old];state['step']=state['step'].to(dtype)
                    states[new]=state;group['params']=[new]
                    m['optimizer'][group['name']].update(id=new,step_dtype=str(dtype))
                opt['state']=states
                self.modify_bundle(metadata=m,payload=p)
                actual=cp.read_checkpoint(self.directory())
                self.assertEqual(actual['metadata']['optimizer'],m['optimizer'])
                self.assertTrue(all(s['step'].dtype==dtype for s in actual['state']['capture'][11]['state'].values()))

    def test_real_files_independent_expectation_and_new_model(self):
        from formal_checkpoint_state import restore_diagnostic
        self.setup_writer(); self.update(1)
        expected=independent(self.m)
        self.p.scene.save(1)
        result=cp.read_checkpoint(self.directory()); assert_expected(self,expected,result)
        self.assertTrue(all(d['initialized'] for d in result['metadata']['optimizer'].values()))
        self.assertEqual(result['metadata']['base']['effective'],identity.projection(self.p.inputs.config))
        self.assertEqual(result['metadata']['base']['native'],self.p.checkpoint_context.native.value())
        before=self.hashes(self.directory())
        # cfg_args is never an authority for the reader.
        (self.f.output/'cfg_args').write_text('untrusted legacy representation')
        with patch.object(type(self.m),'training_setup',side_effect=AssertionError('resume forbidden')), \
                patch.object(self.torch.optim.Adam,'load_state_dict',side_effect=AssertionError('optimizer load forbidden')), \
                patch.object(self.train,'setup_seed',side_effect=AssertionError('seed forbidden')), \
                patch.object(sys.modules['scene.gaussian_model'],'distCUDA2',side_effect=AssertionError('KNN forbidden')) as knn:
            # Native function identity stays required; use the same fail guard in
            # its module, too. Neither real function nor guard may be called.
            with patch.object(sys.modules['simple_knn._C'],'distCUDA2',knn):
                restored=restore_diagnostic(result,'cpu')
        for slot,name in SLOTS.items():
            actual=getattr(restored,name)
            self.assertEqual(actual.numpy().tobytes(),expected[0][slot].numpy().tobytes())
            self.assertNotEqual(actual.data_ptr(),result['state']['capture'][slot].data_ptr())
        self.assertIsNone(restored.optimizer)
        self.assertEqual(before,self.hashes(self.directory()))
        self.update(2); self.p.scene.save(2)
        self.assertEqual(before,self.hashes(self.directory()))
        self.assertTrue(cp.read_checkpoint(self.directory(2))['metadata']['final'])
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_fresh_process_cpu_reader_and_actual_diagnostic_constructor(self):
        self.setup_writer();self.update(1);self.p.scene.save(1)
        reference=self.f.base/'reference.json';reference.write_bytes(identity.canonical(self.f.ref))
        code="""
import sys,json
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'tests')]
import formal_inputs,formal_checkpoint
with patch.object(formal_inputs,'_reference',return_value=json.loads(Path(sys.argv[3]).read_bytes())):
 result=formal_checkpoint.read_checkpoint(Path(sys.argv[2]))
assert 'torch' in sys.modules
assert not any(n in sys.modules for n in ('train','scene','gaussian_renderer'))
# Only now explicitly enter the heavy diagnostic model phase, using real CPU
# deps and constructor/restore but the same synthetic native/GPU boundaries.
from test_formal_connection import ConnectionTests
from contextlib import ExitStack
ConnectionTests.setUpClass()
try:
 import torch,scene.gaussian_model as gm
 e=result['metadata']['base']['native']
 sys.modules['simple_knn._C'].__file__=e['knn']['binary']['path']
 sys.modules['gaussian_renderer.diff_gaussian_rasterization']._C.__file__=e['raster']['binary']['path']
 def forbidden(*a,**kw):raise AssertionError('training/KNN forbidden')
 with ExitStack() as guards:
  guards.enter_context(patch.object(gm,'distCUDA2',forbidden))
  guards.enter_context(patch.object(sys.modules['simple_knn._C'],'distCUDA2',forbidden))
  guards.enter_context(patch.object(gm.GaussianModel,'training_setup',forbidden))
  guards.enter_context(patch.object(torch.optim.Adam,'load_state_dict',forbidden))
  guards.enter_context(patch.object(torch.Tensor,'cuda',lambda t,*a,**kw:t))
  from formal_checkpoint_state import restore_diagnostic
  model=restore_diagnostic(result,'cpu')
  assert model.optimizer is None and not torch.cuda.is_initialized()
 print('CPU_READER_AND_MODEL_PASSED')
finally:ConnectionTests.doClassCleanups()
"""
        r=subprocess.run([sys.executable,'-B','-c',code,str(identity.ROOT),str(self.directory()),str(reference)],
                         text=True,capture_output=True,timeout=120)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertIn('CPU_READER_AND_MODEL_PASSED',r.stdout)

    def test_independent_hand_assembled_bundle_and_payload_rejections(self):
        from formal_checkpoint_state import validate_state
        self.setup_writer(); self.update(1); self.p.scene.save(1)
        r=cp.read_checkpoint(self.directory()); expected=independent(self.m)
        # Independently reconstruct all slots, optimizer bindings and bytes.
        slots=[None]*19
        for i,value in expected[0].items(): slots[i]=value
        groups=[]; states={}
        for i,(name,g) in enumerate(expected[1].items()):
            groups.append(dict(g['settings'],params=[i]))
            states[i]=g['state']
        slots[11]=dict(state=states,param_groups=groups)
        manual=dict(schema=identity.SCHEMA,layout=1,k=1,capture=tuple(slots))
        self.modify_bundle(payload=manual)
        assert_expected(self,expected,cp.read_checkpoint(self.directory()))
        cases=[]
        for slot in SLOTS:
            cases += [('nan-'+str(slot),lambda p,m,s=slot:p['capture'][s].reshape(-1).__setitem__(0,float('nan'))),
                      ('inf-'+str(slot),lambda p,m,s=slot:p['capture'][s].reshape(-1).__setitem__(0,float('inf')))]
        cases += [('quaternion',lambda p,m:p['capture'][5][0].zero_()),
                  ('underflow',lambda p,m:p['capture'][14].fill_(-1000)),
                  ('overflow',lambda p,m:p['capture'][4].fill_(1000)),
                  ('radius',lambda p,m:p['capture'][7].fill_(-1)),
                  ('denom',lambda p,m:p['capture'][10].fill_(.25)),
                  ('moment',lambda p,m:p['capture'][11]['state'][0]['exp_avg_sq'].fill_(-1)),
                  ('duplicate-id',lambda p,m:p['capture'][11]['param_groups'][1].__setitem__('params',[0])),
                  ('missing-state',lambda p,m:p['capture'][11]['state'].pop(0)),
                  ('bad-lr',lambda p,m:p['capture'][11]['param_groups'][0].__setitem__('lr',9.)),
                  ('k-bool',lambda p,m:p.__setitem__('k',True))]
        for label,change in cases:
            with self.subTest(case=label):
                p=copy.deepcopy(manual); m=copy.deepcopy(r['metadata']); change(p,m)
                self.modify_bundle(metadata=m,payload=p)
                with self.assertRaises(ValueError): cp.read_checkpoint(self.directory())
        for slot,replacement in ((3,manual['capture'][3][:,:46]),(1,manual['capture'][1].double()),
                                 (2,manual['capture'][2][:-1]),(0,False),(16,False),(17,self.torch.tensor([]))):
            with self.subTest(slot=slot):
                p=copy.deepcopy(manual); a=list(p['capture']);a[slot]=replacement;p['capture']=tuple(a)
                with self.assertRaises(ValueError): validate_state(p,r['metadata'],r['config'])
        # Row/slot corruption can remain schema-valid: independent oracle must catch it.
        p=copy.deepcopy(manual); p['capture'][3][0,40,2]+=1
        with self.assertRaises(AssertionError): assert_expected(self,expected,dict(state=p))
        p=copy.deepcopy(manual); p['capture'][1][:]=p['capture'][1].flip(0)
        with self.assertRaises(AssertionError): assert_expected(self,expected,dict(state=p))

    def test_metadata_failures_precede_torch_load(self):
        self.setup_writer(); self.update(1); self.p.scene.save(1)
        good=json.loads((self.directory()/'metadata.json').read_bytes())
        cases=[('version',lambda m:m.__setitem__('schema','other')),
               ('unknown',lambda m:m.__setitem__('extra',0)),
               ('missing',lambda m:m.pop('rows')),
               ('bool',lambda m:m.__setitem__('k',True)),
               ('final',lambda m:m.__setitem__('final',True)),
               ('zero',lambda m:m.__setitem__('k',0)),
               ('beyond',lambda m:m.__setitem__('k',3)),
               ('rows',lambda m:m.__setitem__('rows',0)),
               ('config',lambda m:m['base'].__setitem__('raw_config','{}')),
               ('effective',lambda m:m['base']['effective'].__setitem__('maximum_sh',None)),
               ('camera',lambda m:m['base']['dataset'].__setitem__('cameras',[])),
               ('source',lambda m:m['base']['native'].__setitem__('source',{})),
               ('binary',lambda m:m['base']['native']['knn']['binary'].__setitem__('path','/tmp/unknown.so'))]
        for label,change in cases:
            with self.subTest(case=label):
                m=copy.deepcopy(good); change(m); self.modify_bundle(metadata=m)
                with patch.object(self.torch,'load',side_effect=AssertionError('stage T reached')) as load:
                    with self.assertRaises((ValueError,FileNotFoundError)): cp.read_checkpoint(self.directory())
                    load.assert_not_called()
        # Fresh process proves M does not import Torch/Scene, not only no load.
        code="""
import sys
sys.path.insert(0,sys.argv[1])
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','scene','train','numpy','gaussian_renderer'):
   raise AssertionError('heavy import in M')
sys.meta_path.insert(0,Deny())
import formal_checkpoint
try: formal_checkpoint.read_checkpoint(sys.argv[2])
except ValueError: pass
else: raise AssertionError('bad metadata accepted')
assert 'torch' not in sys.modules
"""
        bad=copy.deepcopy(good);bad['schema']='bad';self.modify_bundle(metadata=bad)
        r=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(identity.ROOT),str(self.directory())],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_file_format_corruption_and_no_jit(self):
        self.setup_writer(); self.update(1); self.p.scene.save(1)
        d=self.directory(); original=(d/'state.pt').read_bytes()
        formats=[b'broken',original[:50]]
        fake=io.BytesIO()
        with zipfile.ZipFile(fake,'w') as z:
            z.writestr('archive/data.pkl',b'x');z.writestr('archive/constants.pkl',b'x')
        formats.append(fake.getvalue())
        for content in formats:
            (d/'state.pt').write_bytes(content);self.modify_bundle()
            with patch.object(self.torch.jit,'load',side_effect=AssertionError('JIT load forbidden')):
                with self.assertRaises(ValueError): cp.read_checkpoint(d)
        (d/'state.pt').write_bytes(original);self.modify_bundle()
        (d/'state.pt').write_bytes(original+b'changed')
        with patch.object(self.torch,'load',side_effect=AssertionError('hash failure must precede T')):
            with self.assertRaisesRegex(ValueError,'component_hash'): cp.read_checkpoint(d)
        (d/'state.pt').unlink();(d/'state.pt').symlink_to(d/'metadata.json')
        with self.assertRaises(OSError):cp.read_checkpoint(d)
        (d/'complete.json').unlink()
        with self.assertRaises(FileNotFoundError):cp.read_checkpoint(d)

    def test_publication_failures_keep_previous_and_are_terminal(self):
        self.setup_writer();self.update(1);self.p.scene.save(1)
        prior=self.hashes(self.directory());self.update(2)
        with patch.object(cp.os,'link',side_effect=OSError('injected_link')):
            with self.assertRaisesRegex(OSError,'injected_link'): self.p.scene.save(2)
        self.assertFalse((self.directory(2)/'complete.json').exists())
        self.assertTrue((self.directory(2)/'.complete.pending').exists())
        self.assertEqual(prior,self.hashes(self.directory()))
        with self.assertRaisesRegex(ValueError,'writer_failed'):self.p.scene.save(2)

    def test_same_k_collision_and_missing_context(self):
        self.setup_writer();self.update(1);self.p.scene.save(1)
        prior=self.hashes(self.directory())
        rival=cp.CheckpointWriter(self.p.checkpoint_context,self.claim,self.p.scene)
        with self.assertRaises(FileExistsError):rival.save(1)
        self.assertEqual(prior,self.hashes(self.directory()))
        with self.assertRaisesRegex(ValueError,'context_claim'):cp.CheckpointWriter(None,self.claim,self.p.scene)

    def test_fsync_failure_after_publication_reports_failure_not_success(self):
        self.setup_writer();self.update(1)
        original=cp.os.fsync
        import stat
        def fail_directory(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):raise OSError('directory_fsync')
            return original(fd)
        with patch.object(cp.os,'fsync',fail_directory):
            with self.assertRaisesRegex(OSError,'directory_fsync'):self.p.scene.save(1)
        self.assertTrue(self.writer.failed)
        self.assertTrue((self.directory()/'complete.json').is_file())
        self.assertEqual(cp.read_checkpoint(self.directory())['metadata']['k'],1)

    def test_write_publication_fault_matrix_and_namespace_conflicts(self):
        from formal_checkpoint_state import snapshot
        cases=('snapshot','metadata_write','file_fsync','close','readback','pending',
               'link_competitor','published_read','existing_empty','symlink')
        for case in cases:
            with self.subTest(case=case),ExitStack() as stack:
                self.f.output=self.f.base/('fault-'+case)
                self.f.value['output']['directory']=str(self.f.output)
                self.setup_writer();self.update(1);self.p.scene.save(1)
                before=self.hashes(self.directory());self.update(2)
                def fail(*a,**kw):raise OSError('injected_'+case)
                if case=='snapshot':stack.enter_context(patch('formal_checkpoint_state.snapshot',fail))
                elif case in ('metadata_write','close','pending'):
                    original=cp._write
                    def write(fd,name,writer):
                        if (case=='metadata_write' and name=='metadata.json') or (case=='pending' and name=='.complete.pending'):fail()
                        original(fd,name,writer)
                        if case=='close' and name=='state.pt':fail()
                    stack.enter_context(patch.object(cp,'_write',write))
                elif case=='file_fsync':stack.enter_context(patch.object(cp.os,'fsync',fail))
                elif case=='readback':stack.enter_context(patch.object(cp,'_read_pair',fail))
                elif case=='published_read':stack.enter_context(patch.object(cp,'read_checkpoint',fail))
                elif case=='link_competitor':
                    original=cp.os.link
                    def link(src,dst,**kw):
                        fd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=kw['dst_dir_fd'])
                        with os.fdopen(fd,'wb') as f:f.write(b'competing publication')
                        return original(src,dst,**kw)
                    stack.enter_context(patch.object(cp.os,'link',link))
                elif case=='existing_empty':self.directory(2).mkdir()
                elif case=='symlink':self.directory(2).symlink_to(self.directory(),target_is_directory=True)
                with self.assertRaises(OSError):self.p.scene.save(2)
                self.assertEqual(before,self.hashes(self.directory()))
                self.assertTrue(self.writer.failed)
                with self.assertRaisesRegex(ValueError,'writer_failed'):self.p.scene.save(2)
                if case=='link_competitor':self.assertEqual((self.directory(2)/'complete.json').read_bytes(),b'competing publication')
                elif case not in ('published_read','symlink'):
                    self.assertFalse((self.directory(2)/'complete.json').exists())

    def test_short_write_readback_failure_never_publishes(self):
        self.setup_writer();self.update(1)
        original=cp._write
        def short(fd,name,writer):
            return original(fd,name,(lambda f:f.write(b'x')) if name=='state.pt' else writer)
        with patch.object(cp,'_write',short):
            with self.assertRaises(ValueError):self.p.scene.save(1)
        self.assertFalse((self.directory()/'complete.json').exists())
        self.assertTrue(self.writer.failed)

    def test_uninitialized_state_signed_stats_and_outside_time_allowed(self):
        from formal_checkpoint_state import snapshot
        self.setup_writer();self.m.oneupSHdegree();self.m.update_learning_rate(1)
        with self.torch.no_grad():self.m._t.fill_(100)
        state,desc=snapshot(self.m,self.p.inputs.config,1)
        self.assertTrue(all(not d['initialized'] for d in desc.values()))
        self.assertFalse(state['capture'][11]['state'])
        self.p.scene.save(1)
        self.assertTrue(bool((cp.read_checkpoint(self.directory())['state']['capture'][13]==100).all()))

    def test_real_cpu_loop_topology_reset_and_all_SH_checkpoint_states(self):
        import formal_inputs
        import training_update_runtime as runtime
        from test_formal_config import encode
        from test_training_update_cpu import ConsumerTests,Event
        renderer=ConsumerTests('runTest');renderer.torch=self.torch
        evidence=self.f.checkpoint_evidence();budget=dict(forwards=0);saved=[]
        for name in ('G1','G4','G5','G8'):
            with self.subTest(case=name):
                case=next(c for c in runtime.CASES if c['name']==name)
                value,reference=runtime.create_fixture(self.f.base/name,case)
                if name in ('G1','G5'):value['checkpoint']['intermediate_updates']=list(range(1,case['n']))
                with patch.object(formal_inputs,'_reference',return_value=reference), \
                        patch.object(self.torch.cuda,'Event',Event):
                    inputs=formal_inputs.verify_inputs(encode(value))
                    p=self.train.prepare_formal_runtime(inputs,native_evidence=evidence)
                    runtime.initialize_population(p,case)
                    claim=formal_entry.OutputClaim(inputs);writer=None
                    try:
                        writer=self.train.begin_formal_output(p,claim)
                        obs=runtime.Observation(p,case,renderer.renderer,budget)
                        def save(real_save,k):
                            expected=independent(p.gaussians)
                            self.assertTrue(all(g['state'] for g in expected[1].values()))
                            real_save(k)
                            result=cp.read_checkpoint(Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k)))
                            assert_expected(self,expected,result)
                            saved.append((name,k,len(result['state']['capture'][1])))
                        trace=obs.run(writer,checkpoint_observer=save)
                        self.assertEqual(trace['status'],'passed')
                    finally:
                        if writer is not None:writer.close()
                        claim.close()
        self.assertEqual(len(saved),12)
        self.assertEqual([k for n,k,m in saved if n=='G1'],[1,2,3,4,5,6])
        self.assertEqual([m for n,k,m in saved if n=='G5'],[14,28,28])
        self.assertFalse(self.torch.cuda.is_initialized())
        print('CHECKPOINT_CPU_STATE_COVERAGE '+json.dumps(saved))


class EntryProcessTests(unittest.TestCase):
    def test_compiler_and_recipe_failures_cannot_reach_prepare_or_claim(self):
        code="""
import sys,tempfile,subprocess
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[sys.argv[1],str(Path(sys.argv[1])/'tests')]
import torch
import formal_entry,formal_inputs,formal_native_bootstrap as native
import training_update_runtime as runtime
import checkpoint_native_fixture as fixture
with tempfile.TemporaryDirectory() as d:
 base=Path(d);case=next(c for c in runtime.CASES if c['name']=='G0a')
 _,ref=runtime.create_fixture(base/'case',case)
 with patch.object(formal_inputs,'_reference',return_value=ref),fixture.compiler_boundaries(base):
  def fail(*a,**kw):raise subprocess.CalledProcessError(1,['isolated-compiler'])
  original=fixture.products
  def corrupt(directory,units,binary):
   original(directory,units,binary)
   (directory/'build.ninja').write_text('wrong translation units')
  target=(patch.object(native.subprocess,'run',fail) if sys.argv[2]=='compiler'
          else patch.object(fixture,'products',corrupt))
  with target:
   try:formal_entry.run(base/'case/config.json')
   except (subprocess.CalledProcessError,ValueError):pass
   else:raise AssertionError('failure accepted')
  assert not (base/'case/output').exists()
  assert not any(n in sys.modules for n in ('train','scene','gaussian_renderer'))
  assert not torch.cuda.is_initialized()
"""
        for case in ('compiler','recipe'):
            with self.subTest(case=case):
                r=subprocess.run([sys.executable,'-B','-c',code,str(identity.ROOT),case],capture_output=True,text=True,timeout=120)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)

    def test_missing_source_and_preload_stop_before_torch(self):
        code="""
import sys,types
sys.path.insert(0,sys.argv[1])
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','numpy','scene','train','gaussian_renderer'):
   raise AssertionError('heavy import before provenance')
sys.meta_path.insert(0,Deny())
import formal_native_bootstrap as n
for preloaded in (False,True):
 if preloaded:sys.modules['train']=types.ModuleType('train')
 try:n.bootstrap({})
 except ValueError as e:assert str(e)=='checkpoint_'+('native_preloaded' if preloaded else 'source_changed')
 else:raise AssertionError('accepted')
assert 'torch' not in sys.modules
"""
        r=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(identity.ROOT)],text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_normal_entry_real_bootstrap_collectors_and_context(self):
        result=subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('checkpoint_entry_cpu.py'))],
                              capture_output=True,text=True,timeout=240)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('ENTRY_CPU_PASSED',result.stdout)
        print(result.stdout)


if __name__=='__main__':
    print('CPU_EXECUTABLE',sys.executable,flush=True)
    unittest.main(verbosity=2)
