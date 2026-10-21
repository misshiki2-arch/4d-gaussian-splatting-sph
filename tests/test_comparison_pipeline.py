"""Small CPU consumers/files/subprocesses; no real native or device access."""
import copy
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).resolve().parents[1])]
import staged_comparison_support as support
import staged_comparison_results as results
import staged_comparison_entry as entry
import staged_comparison_checkpoint as cp
import comparison_profile as profile
from formal_checkpoint_identity import file_identity,canonical
from formal_training_fixture import CPUCase
import test_comparison_connection as connection


def process(root,mode,code='print("CPU child")'):
    return support.monitor().monitored_child([support.PYTHON,'-I','-B','-c',code],
        cwd=root,env=dict(os.environ),out=root,mode=mode,filesystem=root,interval=.005)


class FinalConnectionTests(CPUCase):
    initial=connection.ComparisonConnectionTests.initial
    bundle=connection.ComparisonConnectionTests.bundle
    provenance_fixture=connection.ComparisonConnectionTests.provenance_fixture

    def final_fixture(self):
        from staged_comparison_runtime import ComparisonClaim,begin_output
        from test_formal_checkpoint import independent
        from staged_comparison_observation import expected_equal
        from staged_comparison_state import restore_diagnostic
        import comparison_inputs
        bundle=self.bundle();p=self.train.prepare_comparison_runtime(bundle);m=p.gaussians;m.env_map=None
        self.f.output.parent.mkdir();run=self.f.base/'run';run.mkdir()
        initial=self.f.base/'initial.pt';self.torch.save(self.initial(),initial)
        parent=dict(kind='initial_dict',path=str(initial),identity=file_identity(initial),stage=None,completed=0,verification=None)
        self.f.gpu.enter_context(patch.object(cp,'INITIAL_PATH',str(initial)))
        self.f.gpu.enter_context(patch.object(comparison_inputs,'OUTPUT_OWNER',self.f.base))
        claim=ComparisonClaim(bundle);self.addCleanup(claim.close)
        with patch.object(self.train,'TENSORBOARD_FOUND',False):begin_output(p,claim,self.provenance_fixture(),parent)
        n=bundle.config.optimization.total_updates;m.update_learning_rate(n)
        sum(g['params'][0].sum() for g in m.optimizer.param_groups).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        attrs,groups=independent(m)
        self.torch.save(dict(attrs=attrs,groups=groups),run/'expected.pt')
        p.scene.save(n);ck=self.f.output/'checkpoints'/('completed-'+str(n))
        # Real independent expected vs real serializer/readback, then no-Adam
        # restoration. Only rendering/native is a tiny CPU boundary fixture.
        loaded=cp.read_checkpoint(ck);expected_equal((attrs,groups),loaded)
        with patch.object(self.torch.optim.Adam,'__init__',side_effect=AssertionError('reader Adam')):
            restored=restore_diagnostic(loaded,'cpu')
        self.assertIsNone(restored.optimizer)
        self.assertTrue(self.torch.equal(restored._xyz,m._xyz))
        w=dict(status='passed',stage='stabilization',parent=parent,completed=n,optimizer_calls=n,
            checkpoint=results.reference(ck/'complete.json'),expected=results.reference(run/'expected.pt'))
        support.write_json(run/'worker-result.json',w);process(run,'worker')
        r=dict(status='passed',stage=w['stage'],parent=parent,worker=results.reference(run/'worker-result.json'),
            checkpoint=w['checkpoint'],expected=w['expected'],independent_state=True,raw14_exact=True,
            forwards=14,optimizer_created=False,builds=0)
        support.write_json(run/'reader-result.json',r);process(run,'reader')
        results.verify_child(run);process(run,'verify')
        def comparison(worker,dest):
            dest.mkdir();support.write_json(dest/'comparison.json',dict(stage=worker['stage'],
                status='recorded_requires_image_review',pictures={}))
        receipt=results.finalize(run,comparison)
        return run,receipt,loaded,attrs,groups

    def test_confirmed_parent_actual_readback_fresh_consumer(self):
        from staged_comparison_state import Initializer
        from comparison_inputs import _acquire_dataset
        run,receipt,loaded,attrs,groups=self.final_fixture()
        c=profile.stage_config('growth',str(self.f.base/'next'))
        parent=results.parent_from_receipt(receipt,'growth')
        init=Initializer.from_checkpoint(parent,c)
        for value,slot in zip(init.parameters,(1,2,3,4,5,6,13,14,15)):
            self.assertTrue(self.torch.equal(value,attrs[slot]))
        prepared=self.train.prepare_comparison_runtime(_acquire_dataset(c,init))
        self.assertEqual(init.parameters,());self.assertTrue(init.consumed)
        self.assertEqual(prepared.gaussians.optimizer.state,{})
        for name in ('max_radii2D','xyz_gradient_accum','t_gradient_accum','denom'):
            self.assertFalse(getattr(prepared.gaussians,name).any())
        self.assertEqual((prepared.gaussians.active_sh_degree,prepared.gaussians.active_sh_degree_t),(0,0))
        self.assertTrue(all(g['params'][0].grad is None for g in prepared.gaussians.optimizer.param_groups))
        with self.assertRaises(ValueError):Initializer.from_checkpoint(dict(loaded,verified_final=True),c)
        with self.assertRaises(ValueError):results.parent_from_receipt(receipt,'e2_like')
        for name in ('reader-result.json','verify-result.json','verify-exit.json','completion/result.json','completion/receipt.json'):
            p=run/name;original=p.read_bytes();p.write_bytes(original+b'x')
            with self.subTest(name=name),self.assertRaises(ValueError):results.parent_from_receipt(receipt,'growth')
            p.write_bytes(original)
        (run/'failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'stage_failed'):results.parent_from_receipt(receipt,'growth')

    def test_independent_expected_rejects_state_and_adam_mutation(self):
        from staged_comparison_observation import expected_equal
        _,_,loaded,attrs,groups=self.final_fixture()
        for target in ('parameter','moment','radius'):
            changed=copy.deepcopy(loaded)
            a=changed['state']['capture']
            if target=='parameter':a[1][0,0]+=1
            if target=='radius':a[7][0]+=1
            if target=='moment':next(iter(a[11]['state'].values()))['exp_avg'].flatten()[0]+=1
            with self.subTest(target=target),self.assertRaises(ValueError):expected_equal((attrs,groups),changed)

    def test_actual_mask_copy_and_chunk_independence(self):
        from staged_comparison_observation import observation_class,local_dataset
        from utils.formal_camera_runtime import camera_binding
        from real_input_runtime import rng_state
        p=self.train.prepare_comparison_runtime(self.bundle());p.gaussians.env_map=None
        cls=observation_class();cam=p.dataloader.dataset.viewpoint_stack[0]
        original=copy.copy(cam);before=rng_state(p.gaussians)
        mem=support.memory_helper();local=local_dataset(p.dataloader.dataset,0)
        image,camera=local[0]
        self.assertIsNot(camera,cam);self.assertEqual(cam.__dict__.keys(),original.__dict__.keys())
        self.assertIsNone(cam.gt_alpha_mask);self.assertIsNotNone(camera.gt_alpha_mask)
        self.assertIs(cam.formal_binding,original.formal_binding)
        self.assertIs(camera_binding(camera,p.inputs.config.dataset.resolution).camera,camera)
        transferred=camera.cuda()
        self.assertIs(camera_binding(transferred,p.inputs.config.dataset.resolution).camera,transferred)
        self.assertIs(camera_binding(cam,p.inputs.config.dataset.resolution).camera,cam)
        import gc,weakref
        ref=weakref.ref(camera.gt_alpha_mask)
        del transferred,camera,local
        gc.collect();self.assertIsNone(ref());self.assertIsNone(cam.gt_alpha_mask)
        mem.equal_live(rng_state(p.gaussians),before)
        a=self.torch.arange(60,dtype=self.torch.float32).reshape(3,4,5).transpose(0,2)
        mem.equal_live(a,a.clone());b=a.clone();b[0,0,0]+=1
        with self.assertRaises(ValueError):mem.equal_live(a,b)
        from staged_comparison_observation import schedule
        self.assertEqual(schedule(p.inputs.config)['sh'],[])

    def test_original_camera_corruption_not_repaired_by_copy(self):
        from staged_comparison_observation import local_dataset
        from utils.formal_camera_runtime import CameraBinding
        p=self.train.prepare_comparison_runtime(self.bundle());dataset=p.dataloader.dataset
        original=dataset.viewpoint_stack[0]
        cases=(lambda c:setattr(c,'formal_binding',None),
               lambda c:setattr(c,'formal_binding',original.formal_binding),
               lambda c:setattr(c,'timestamp',99),
               lambda c:setattr(c,'formal_frame',None),
               lambda c:setattr(c,'projection_matrix',c.projection_matrix.clone()),
               lambda c:c.full_proj_transform.add_(1))
        for mutation in cases:
            c=original.cuda();mutation(c);dataset.viewpoint_stack[0]=c
            with self.assertRaises(ValueError):local_dataset(dataset,0)
        # Even forged version stamps cannot skip new CPU numeric validation.
        c=original.cuda();c.projection_matrix[0,0]=float('nan')
        b=c.formal_binding
        c.formal_binding=CameraBinding(c,b.state,b.frame,b.tensors,tuple(t._version for t in b.tensors))
        dataset.viewpoint_stack[0]=c
        with self.assertRaisesRegex(ValueError,'matrix_finite'):local_dataset(dataset,0)
        dataset.viewpoint_stack[0]=original

    def test_selected_clone_split_budget_precedes_allocations(self):
        from staged_comparison_observation import observation_class,schedule
        p=self.train.prepare_comparison_runtime(self.bundle());m=p.gaussians;m.env_map=None;t=self.torch
        sum(g['params'][0].sum() for g in m.optimizer.param_groups).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        with t.no_grad():m._scaling[0].fill_(-9);m._scaling[1].fill_(0)
        m.xyz_gradient_accum.fill_(1);m.denom.fill_(1)
        obs=observation_class()(p,None,self.f.base,schedule(p.inputs.config),[],[],[],
                                diagnostic_updates=None)
        decisions=[];allocations=[];original=m.densification_postfix
        def budget(directory,n,added):decisions.append((n,added));self.assertEqual(len(allocations),len(decisions)-1)
        def allocated(*a):
            allocations.append((len(m._xyz),len(a[0])))
            self.assertEqual(allocations[-1],decisions[-1]);return original(*a)
        with patch.object(support,'topology',side_effect=budget),patch.object(m,'densification_postfix',side_effect=allocated):
            obs.population(m.densify_and_prune,.5,0.,1.,None,prune_only=False)
        self.assertEqual(decisions,[(2,1),(3,2)]);self.assertEqual(len(m._xyz),4)
        self.assertEqual(allocations,decisions)
        # A rejecting decision happens before the target's clone/split allocation.
        before=m._xyz
        with patch.object(support,'topology',side_effect=ValueError('budget')),\
             patch.object(m,'densification_postfix',side_effect=AssertionError('allocation reached')):
            with self.assertRaisesRegex(ValueError,'budget'):
                obs.population(m.densify_and_prune,.5,0.,1.,None,prune_only=False)
        self.assertIs(m._xyz,before)

    def test_storage_rejection_before_observation_and_save_producers(self):
        from types import SimpleNamespace
        from staged_comparison_observation import observation_class,schedule
        import staged_comparison_storage as disk
        from pilot_runtime import Observation as Pilot
        p=self.train.prepare_comparison_runtime(self.bundle());p.gaussians.env_map=None
        # No real output here: sentinels are used only to prove non-entry.
        p.scene.model_path=str(self.f.base)
        obs=observation_class()(p,None,self.f.base,schedule(p.inputs.config),[],[],[],diagnostic_updates=None)
        callbacks=[]
        obs._fixed=lambda k:callbacks.append('fixed')
        obs._report=lambda *a,**kw:callbacks.append('ordinary')
        obs._save=lambda writer,k:writer(k)
        with patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            for kind,invoke in (
                ('fixed',lambda:obs.fixed(0)),
                ('ordinary',lambda:obs.report(lambda *a:None,None,0)),
                ('checkpoint',lambda:obs.save(lambda k:callbacks.append('checkpoint'),1))):
                with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'save_disk_budget:'+kind):invoke()
            # The actual final expected opener is after capture returns. Model
            # producer is isolated only in this pre-write ordering negative.
            with patch.object(Pilot,'capture',return_value=([],[])):
                with self.assertRaisesRegex(ValueError,'save_disk_budget:expected'):
                    obs.capture(self.f.base/'fixed',final=True)
        self.assertEqual(callbacks,[]);self.assertFalse((self.f.base/'expected.pt').exists())
        rows=[json.loads(line) for line in (self.f.base/'resource-decisions.jsonl').read_text().splitlines()]
        self.assertEqual([r['boundary'] for r in rows],['fixed','ordinary','checkpoint','expected'])
        self.assertTrue(all(r['status']=='failed' and r['points']==2 for r in rows))

    def test_worker_reader_real_cpu_consumers_and_raw14(self):
        import staged_comparison_worker as runtime
        import comparison_inputs
        from types import SimpleNamespace
        self.bundle()
        # Fourteen small real files/cameras; retain actual NumPy/Pillow readers,
        # losses, Adam, observer, writer, decoder and independent render reader.
        f=self.f;t=self.torch;f.ref.update(width=256,height=256)
        for split,count in (('train',14),('test',1)):
            old=json.loads((f.source/('transforms_'+split+'.json')).read_text())
            old.update(w=256,h=256,fl_x=256,fl_y=256,cx=128,cy=128);frames=[];f.ref['frames'][split]=[]
            for i in range(count):
                row=copy.deepcopy(old['frames'][0]);row.update(file_path='images/'+split+str(i)+'.png',time=float(i))
                row['transform_matrix'][0][3]=i;frames.append(row)
                mask=row['file_path'].replace('images/','masks/')
                f.Image.new('RGBA',(256,256),(100,40,10,128)).save(f.source/row['file_path'])
                f.Image.new('L',(256,256),128).save(f.source/mask)
                f.ref['frames'][split].append([row['file_path'],row['time'],mask])
            old['frames']=frames;f.store_input('transforms_'+split+'.json',json.dumps(old).encode())
        run=f.base/'worker-run';run.mkdir();f.output.parent.mkdir()
        initial=f.base/'init.pt';t.save(self.initial(),initial)
        parent=dict(kind='initial_dict',path=str(initial),identity=file_identity(initial),stage=None,completed=0,verification=None)
        stage=dict(name='stabilization',run=str(run),output=str(f.output),initial_points=2)
        candidate=dict(provenance=self.provenance_fixture(),representatives=[['train',i,*row[:2]] for i,row in enumerate(f.ref['frames']['train'])])
        def render(cam,m,pipe,bg,**kw):
            screen=t.zeros_like(m._xyz,requires_grad=True)
            signal=sum(getattr(m,attr).sum()*.0001 for attr in ('_xyz','_features_dc','_features_rest','_scaling',
                '_rotation','_opacity','_t','_scaling_t','_rotation_r'))+screen[:,:2].sum()*.001
            return dict(render=(.6+signal).expand(3,cam.image_height,cam.image_width),viewspace_points=screen,
                visibility_filter=t.ones(len(m._xyz),dtype=t.bool),radii=t.ones(len(m._xyz)),
                depth=t.ones(1,cam.image_height,cam.image_width),alpha=(.3+m._opacity.sum()*.0001).expand(1,cam.image_height,cam.image_width))
        @contextmanager
        def native(value):yield self.train,SimpleNamespace(render=render)
        class Event:
            def __init__(self,**kw):pass
            def record(self):pass
            def elapsed_time(self,other):return 0.
        small=((4,*profile._VALUES[0][1:]),*profile._VALUES[1:])
        budgets=support.resources()
        with patch.object(profile,'_VALUES',small),patch.object(cp,'INITIAL_PATH',str(initial)),\
             patch.object(comparison_inputs,'OUTPUT_OWNER',f.base),patch.object(runtime,'native',native),\
             patch.object(budgets,'memory'),patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=10**15)),\
             patch.object(t.cuda,'Event',Event),\
             patch.object(self.train,'TENSORBOARD_FOUND',False):
            runtime.worker(candidate,stage,parent)
            w=cp._read_record(run/'worker-result.json')
            self.assertEqual(w['completed'],4);self.assertEqual(w['optimizer_calls'],4)
            self.assertEqual([x['k'] for x in w['trace']['fixed']],[0,2,4])
            self.assertEqual([x['k'] for x in w['trace']['checkpoints']],[2,4])
            self.assertTrue(all(len(x['views'])==14 for x in w['trace']['fixed']))
            runtime.reader(candidate,stage,parent)
            r=cp._read_record(run/'reader-result.json');self.assertTrue(r['raw14_exact']);self.assertEqual(r['forwards'],14)
            # Independent A images from prescribed fixture pixels, not a copy of
            # the B saved result. Real compare / publication / receipt / bridge.
            import staged_comparison_images as images
            from staged_comparison_state import Initializer
            aroot=f.base/'a';aroot.mkdir();fixed=[]
            for k in (0,2,4):
                area=aroot/'fixed'/str(k);area.mkdir(parents=True);views=[]
                for i,row in enumerate(f.ref['frames']['train']):
                    refs={}
                    for kind,color in (('gt',(50,20,5)),('render',(80,60,40)),('alpha',(100,100,100))):
                        path=area/(str(i)+'-'+kind+'.png');f.Image.new('RGB',(256,256),color).save(path)
                        refs[kind]=file_identity(path)
                    views.append(dict(frame=['train',i,row[0]],raw_time=row[1],images=refs,metrics=dict(fixture=True)))
                fixed.append(dict(label=k,views=views))
            a=dict(stage=dict(name='stabilization'),calls=dict(optimizer_calls=2),fixed=fixed,
                   consumed={str(f.source/row[2]):file_identity(f.source/row[2]) for row in f.ref['frames']['train']},evaluations=[])
            with patch.object(support,'VIEWS',tuple(range(14))),patch.object(support,'NINE',tuple(range(9))),\
                 patch.object(support,'HISTORY',tuple(range(9,14))):
                process(run,'worker');process(run,'reader');results.verify_child(run);process(run,'verify')
                receipt=results.finalize(run,lambda worker,dest:images.compare(worker,dest,a,aroot,f.source))
                self.assertEqual(len(cp._read_record(run/'completion/comparison/comparison.json')['pictures']),84)
                parent_next=results.parent_from_receipt(receipt,'growth')
                c=profile.stage_config('growth',str(f.base/'next'))
                init=Initializer.from_checkpoint(parent_next,c)
                expected=tuple(v.clone() for v in init.parameters)
                prepared=self.train.prepare_comparison_runtime(comparison_inputs._acquire_dataset(c,init))
                self.assertEqual(prepared.gaussians.optimizer.state,{})
                for name in ('max_radii2D','xyz_gradient_accum','t_gradient_accum','denom'):
                    self.assertFalse(getattr(prepared.gaussians,name).any())
                from formal_checkpoint_identity import GROUPS
                from staged_comparison_state import SLOTS
                for value,slot in zip(expected,SLOTS):
                    attr=next(attr for index,attr in GROUPS.values() if index==slot)
                    self.assertTrue(t.equal(value,getattr(prepared.gaussians,attr)))
                for error in ('frame','time','missing_image'):
                    bad=copy.deepcopy(a)
                    if error=='frame':bad['fixed'][0]['views'][0]['frame'][2]='images/other.png'
                    elif error=='time':bad['fixed'][0]['views'][0]['raw_time']=9.
                    else:bad['fixed'][0]['views'][0]['images'].pop('render')
                    with self.subTest(error=error),self.assertRaises((ValueError,KeyError)):
                        images.compare(w,f.base/error,bad,aroot,f.source)
            decisions=[json.loads(line) for line in (run/'resource-decisions.jsonl').read_text().splitlines()]
            self.assertEqual([d['boundary'] for d in decisions if d['boundary'] in ('fixed','expected')],
                             ['fixed','fixed','fixed','expected'])


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_unapproved_entry_cold_no_output(self):
        candidate=self.root/'candidate.json';candidate.write_bytes(canonical(dict(executable=False,execution_authorized=False)))
        code="import sys;sys.path.insert(0,"+repr(str(support.ROOT))+");import staged_comparison_entry as e\n"\
             "def guard(event,args):\n if event=='import' and args[0].split('.')[0] in ('torch','numpy','scene','train','gaussian_renderer'):raise AssertionError('heavy import')\n"\
             "sys.addaudithook(guard);sys.argv=['entry','run',"+repr(str(candidate))+"];e.main()"
        result=subprocess.run([support.PYTHON,'-I','-B','-S','-c',code],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('separate_live_authorization_required',result.stderr)
        self.assertNotIn('heavy import',result.stderr);self.assertEqual(list(self.root.iterdir()),[candidate])

    def test_real_monitor_success_nonzero_and_resource_stop(self):
        for name,code in (('ok','print("ok")'),('failed','raise SystemExit(3)'),('stop','import time;time.sleep(2)')):
            root=self.root/name;root.mkdir()
            if name=='ok':process(root,'worker',code);support.process_record(root,'worker')
            elif name=='failed':
                with self.assertRaises(support.monitor().ChildFailure):process(root,'worker',code)
                self.assertEqual(cp._read_record(root/'worker-exit.json')['exit_code'],3)
            else:
                with self.assertRaises(support.monitor().ChildFailure):
                    support.monitor().monitored_child([support.PYTHON,'-B','-c',code],cwd=root,
                        env=dict(os.environ),out=root,mode='worker',filesystem=root,interval=.005,grace=.1,kill_wait=.1,
                        sample_fn=lambda *a:dict(host_available=0,disk_free=10**12,rss_current=1,rss_high_water=1,swap=0))
                self.assertTrue((root/'worker-stop-request.json').is_file())
                self.assertTrue(cp._read_record(root/'worker-exit.json')['stop_confirmed'])

    def test_b_resource_shapes_boundaries_unavailable_distinction(self):
        n=1326947;p=support.payload(n)
        self.assertEqual(p['parameter'],65*4*n);self.assertEqual(p['capture'],796*n+72)
        d=support.demand('reader_load',n);self.assertGreaterEqual(d['host_payload'],2*p['capture']+p['raw'])
        self.assertGreaterEqual(support.demand('save',n)['host_payload'],2*p['capture'])
        self.assertGreater(support.demand('topology',n)['gpu_payload'],p['training'])
        res=support.resources();kw=dict(free=d['gpu_needed'],allocated=0,reserved=0,inactive_split=0)
        self.assertEqual(res.judge(d,d['host_needed'],**kw)['status'],'passed')
        self.assertEqual(res.judge(d,d['host_needed']-1,**kw)['violations'],['host_available'])
        with patch.dict(res.start.__globals__,collectors=lambda:dict(host_available=lambda:10**15,
            gpu_free=lambda:(_ for _ in ()).throw(RuntimeError('query unavailable')))):
            value=res.start(self.root)
        self.assertEqual(value['status'],'failed');self.assertIn('unavailable',value['meaning'])
        self.assertEqual(res.parent_violations(dict(host_available=10**15,disk_free=10**15,
            rss_current=1,rss_high_water=10**16,swap=0),'worker'),[])

    def test_b_disk_collection_independent_arithmetic_and_threshold(self):
        import staged_comparison_storage as disk
        from types import SimpleNamespace
        w,h,n=1280,720,3204870
        def serial(size):return size+(size+99)//100+1048576
        def image(panels):
            size=3*w*h*panels+h
            return size+(size+99)//100+65536
        record=16777216
        tb_scanlines=3*(2*w+6)*(2*h+6)+(2*h+6)
        tb_image=tb_scanlines+(tb_scanlines+99)//100+65536
        expected=dict(initial=123+record,ordinary=40*image(1)+10*tb_image+record,
            fixed=56*image(1)+record,expected=serial(796*n+72+14*(20*w*h+5*n))+record,
            checkpoint=serial(796*n+72)+record,comparison=84*image(3)+record,record=record)
        for kind,needed in expected.items():
            self.assertEqual(disk.collection(kind,n,initial_bytes=123)['bytes'],needed)
            for delta in (0,-1):
                with patch.object(support.shutil,'disk_usage',return_value=SimpleNamespace(free=needed+8*1024**3+delta)) as probe:
                    if delta:
                        with self.assertRaisesRegex(ValueError,'save_disk_budget'):
                            disk.check(self.root,kind,n,initial_bytes=123)
                    else:disk.check(self.root,kind,n,initial_bytes=123)
                    probe.assert_called_once_with(self.root)
                row=json.loads((self.root/'resource-decisions.jsonl').read_text().splitlines()[-1])
                self.assertEqual(row['new_write_estimate'],needed)
                self.assertEqual(row['status'],'failed' if delta else 'passed')
                self.assertEqual(row['filesystem']['device'],self.root.stat().st_dev)
                self.assertEqual(row['collection']['bytes'],needed)
        total=expected['initial']+3*expected['ordinary']+3*expected['fixed']+expected['expected']+2*expected['checkpoint']+expected['comparison']
        self.assertEqual(disk.collection('stage_initial',n,initial_bytes=123)['bytes'],total)
        self.assertEqual(disk.collection('ordinary',n,tests=1,tensorboard=False)['bytes'],24*image(1)+record)
        self.assertGreater(disk.disk_need('expected',n+100),disk.disk_need('expected',n))
        # Private/public hardlink or rename does not duplicate the tensor/images.
        self.assertEqual(disk.disk_need('checkpoint',n),serial(796*n+72)+record)
        self.assertEqual(disk.disk_need('record',n),record)
        for kind in ('final','middle','unexpected'):
            with self.assertRaises(ValueError):disk.disk_need(kind,n)

    def test_four_stage_sequence_and_failure_downstream_suppression(self):
        from types import SimpleNamespace
        for failure in (None,'start','worker','reader','verify','finalize','program_finalization'):
            area=self.root/(failure or 'success');area.mkdir();program=area/'program'
            value=dict(executable=True,execution_authorized=True,program=str(program),stages=entry.stage_paths(program),
                profiles=[dict(paths=dict(source_path=str(area)))]*4,counts=dict(A_optimizer=65992,B_optimizer=66000))
            events=[]
            def start(*a):return dict(status='failed' if failure=='start' else 'passed')
            def monitored(command,**kw):
                mode=kw['mode'];events.append(mode)
                if mode==failure:raise RuntimeError('injected_'+mode)
                return process(kw['out'],mode)
            def finalize(out,compare):
                events.append('finalize')
                if failure=='finalize':raise RuntimeError('injected_finalization')
                return dict(path=str(out/'completion/result.json'),identity={})
            reads=cp._read_record
            def record(path,*a):
                if str(path).endswith('worker-result.json'):return dict(final_points=2)
                return reads(path,*a)
            def publication(*a):
                if failure=='program_finalization':raise RuntimeError('injected_program_finalization')
            with patch.object(entry,'bind'),patch.object(entry,'parent_binding',return_value={}),\
                 patch.object(cp,'_read_record',side_effect=record),patch.object(entry.results,'finalize',side_effect=finalize),\
                 patch.object(entry.results,'verify_publication',side_effect=publication),\
                 patch.object(support,'resources',return_value=SimpleNamespace(start=start,memory=lambda *a:None,disk=lambda *a:None)),\
                 patch.object(support,'monitor',return_value=SimpleNamespace(monitored_child=monitored)):
                # The actual monitor is used by process(); avoid recursive mock.
                real_monitor=ControlTests.real_monitor
                def local_process(root,mode):
                    return real_monitor.monitored_child([support.PYTHON,'-I','-B','-c','print("cpu")'],cwd=root,
                        env=dict(os.environ),out=root,mode=mode,filesystem=root,interval=.005,
                        sample_fn=lambda *a:dict(host_available=10**15,disk_free=10**15,rss_current=1,rss_high_water=1,swap=0))
                with patch(__name__+'.process',side_effect=local_process),patch.object(support,'process_record'):
                    if failure:
                        with self.assertRaises((RuntimeError,ValueError)):entry.run(value,area/'input')
                        recorded=json.loads((program/'failure.json').read_text())
                        self.assertTrue(recorded['downstream_not_run']);self.assertFalse((program/'completion').exists())
                    else:
                        entry.run(value,area/'input');self.assertTrue((program/'completion/result.json').exists())
            if failure not in (None,'program_finalization'):self.assertLessEqual(len(events),4)
            if failure is None:self.assertEqual(events,['worker','reader','verify','finalize']*4)

    def test_dependency_membership_and_changed_entry_rejected(self):
        files=support.execution_files();support.check_execution_files(files)
        self.assertIn(str(support.HERE/'step18-fix3-entry.py'),files)
        self.assertIn(str(support.OLD/'step17-fix3-monitor.py'),files)
        helper=support.ROOT/'staged_comparison_monitor.py'
        self.assertIn(str(helper),files)
        from formal_checkpoint_identity import source_snapshot
        self.assertEqual(source_snapshot()[helper.name],file_identity(helper))
        for mutation in ('missing','changed'):
            bad=copy.deepcopy(files);key=next(iter(bad))
            if mutation=='missing':bad.pop(key)
            else:bad[key]={'size':0,'sha256':'0'*64}
            with self.assertRaises(ValueError):support.check_execution_files(bad)


class MonitorObservationTests(unittest.TestCase):
    """Only self-spawned small CPU children; fault injection at observation seams."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        if hasattr(support.monitor,'cached'):del support.monitor.cached
        self.m=support.monitor();self.g=self.m.monitored_child.__globals__

    def child(self, *, sample=None, exit_code=0, seconds=.08, candidate_errno=None,
              mutate=None, write_failure=None, journal_failure=False):
        area=self.root/str(len(list(self.root.iterdir())));area.mkdir()
        m=self.m;original=m.proc_identity;children=[];sent=[];injected=[]
        code=f'import time;time.sleep({seconds});raise SystemExit({exit_code})'
        argv=[support.PYTHON,'-I','-B','-c',code]
        def spawn(command,**kw):
            self.assertEqual(command,argv)
            p=subprocess.Popen(command,**kw)
            owner=original(p.pid);children.append((p,owner))
            return p
        test=self
        class SafeOS:
            def __getattr__(self,key):return getattr(os,key)
            def killpg(self,pgid,sig):
                p,owner=children[0]
                test.assertEqual(pgid,owner['pgid'])
                test.assertTrue(m.same_process(owner,original(p.pid)))
                events=[json.loads(s)['event'] for s in (area/'worker-monitor.jsonl').read_text().splitlines()]
                test.assertIn('stop_request',events)
                test.assertEqual(events[-1],'signal_request')
                sent.append(sig.name)
                os.killpg(pgid,sig)
            def fsync(self,fd):
                if journal_failure and not injected:
                    injected.append('journal')
                    raise OSError(errno.EIO,'journal_fault')
                return os.fsync(fd)
        class MissingStat:
            def read_text(self):
                injected.append(candidate_errno)
                raise OSError(candidate_errno,'candidate_fault')
        class Candidate:
            name='999999999'
            def __truediv__(self,key):test.assertEqual(key,'stat');return MissingStat()
        class Proc:
            def __truediv__(self,key):return Path('/proc')/key
            def iterdir(self):
                if candidate_errno is not None and not injected:yield Candidate()
                for p,_ in children:yield Path('/proc')/str(p.pid)
        def path(*args):
            p=Path(*args)
            return Proc() if p==Path('/proc') else p
        def sampled(owner,fs):
            if sample:return sample(children[0][0],owner,fs)
            return m.resource_sample(owner,fs)
        def writer(p,value):
            if write_failure and str(p).endswith(write_failure):
                raise OSError(errno.ENOSPC,'write_fault')
            return support.write_json(p,value)
        patches=dict(Path=path,os=SafeOS(),subprocess=SimpleNamespace(
            Popen=spawn,TimeoutExpired=subprocess.TimeoutExpired),write_json=writer)
        if mutate:patches.update(mutate(children,original))
        record=None
        try:
            with patch.dict(self.g,patches):
                try:
                    record=m.monitored_child(argv,cwd=area,env=dict(os.environ),out=area,
                        mode='worker',filesystem=area,interval=.005,grace=.1,kill_wait=.5,sample_fn=sampled)
                except m.ChildFailure as error:record=error.record
        finally:
            for p,owner in children:
                if p.poll() is None:
                    self.assertTrue(m.same_process(owner,original(p.pid)))
                    os.killpg(owner['pgid'],signal.SIGTERM)
                    p.wait(timeout=3)
                    self.fail('CPU child needed harness recovery')
                p.wait(timeout=3)
        self.assertTrue(children)
        self.assertTrue(all(not Path('/proc',str(p.pid)).exists() for p,_ in children))
        self.assertIsNotNone(record)
        if (area/'worker-exit.json').exists():
            self.assertEqual(json.loads((area/'worker-exit.json').read_text()),record)
        if record['primary'] is None:
            support.process_record(area,'worker')
        print('MONITOR_CASE',json.dumps(dict(exit=record['exit_code'],primary=record['primary'],
            secondary=record['secondary_errors'],signals=sent,reaped=True)),flush=True)
        return record,area,sent

    def test_unassigned_disappearance_success_and_nonzero(self):
        for number in (errno.ENOENT,errno.ESRCH):
            for code in (0,7):
                with self.subTest(errno=number,code=code):
                    row,_,sent=self.child(candidate_errno=number,exit_code=code)
                    self.assertEqual(row['exit_code'],code);self.assertFalse(sent)
                    self.assertTrue(row['stop_confirmed'])
                    self.assertEqual(row['primary']['kind'] if code else row['primary'],
                                     'child_exit' if code else None)

    def test_target_read_disappearance_requires_actual_exit(self):
        for number in (errno.ENOENT,errno.ESRCH):
            for code in (0,7):
                def sample(p,*args):
                    p.wait(timeout=3)
                    raise OSError(number,'target_status_disappeared')
                with self.subTest(errno=number,code=code):
                    row,area,sent=self.child(sample=sample,exit_code=code)
                    self.assertEqual(row['exit_code'],code);self.assertFalse(sent)
                    self.assertEqual(row['primary']['kind'] if code else row['primary'],
                                     'child_exit' if code else None)
                    events=[json.loads(s) for s in (area/'worker-monitor.jsonl').read_text().splitlines()]
                    self.assertTrue(any(e['event']=='terminal_reconciled' for e in events))
                    self.assertEqual(row['samples'],0);self.assertIsNone(row['last_sample'])
        def alive(*args):raise ProcessLookupError(errno.ESRCH,'live_missing_sample')
        row,_,sent=self.child(sample=alive,seconds=2)
        self.assertEqual(row['primary']['kind'],'monitor_unavailable')
        self.assertEqual(row['primary']['operation'],'resource_sample')
        self.assertEqual(sent,['SIGTERM']);self.assertTrue(row['stop_confirmed'])

    def test_same_child_zombie_and_readtime_leader_exit(self):
        def zombie(p,owner,fs):
            deadline=time.monotonic()+3
            while self.m.proc_identity(p.pid)['state']!='Z':
                self.assertLess(time.monotonic(),deadline);time.sleep(.002)
            return self.m.resource_sample(owner,fs)
        row,_,sent=self.child(sample=zombie)
        self.assertIsNone(row['primary']);self.assertFalse(sent)
        def mutate(children,original):
            count=0
            def proc(pid):
                nonlocal count
                count+=1
                if count==2:
                    children[0][0].wait(timeout=3)
                    raise ProcessLookupError(errno.ESRCH,'leader_read_exit')
                return original(pid)
            return dict(proc_identity=proc)
        row,_,sent=self.child(mutate=mutate)
        self.assertIsNone(row['primary']);self.assertFalse(sent)

    def test_real_errors_remain_failures_with_bounded_cause(self):
        for number in (errno.EIO,errno.EACCES):
            row,_,sent=self.child(candidate_errno=number,seconds=2)
            self.assertEqual(row['primary']['errno'],number)
            self.assertEqual(row['primary']['role'],'unassigned')
            self.assertEqual(row['primary']['operation'],'candidate_stat')
            self.assertTrue(row['primary']['location'])
            self.assertLess(len(json.dumps(row['primary'])),1800)
            self.assertEqual(sent,['SIGTERM']);self.assertTrue(row['stop_confirmed'])
        row,_,sent=self.child(sample=lambda *a:{'host_available':1},seconds=2)
        self.assertEqual(row['primary']['operation'],'resource_predicate')
        self.assertEqual(sent,['SIGTERM'])

    def test_original_exit_and_secondary_record_failures_retained(self):
        def failed(p,*args):
            p.wait(timeout=3);raise OSError(errno.EIO,'read_failed_after_child')
        row,_,sent=self.child(sample=failed,exit_code=7,write_failure='-stop-request.json')
        self.assertEqual(row['primary']['errno'],errno.EIO)
        self.assertEqual(row['primary']['child_exit_code'],7)
        self.assertEqual(row['exit_code'],7);self.assertFalse(sent)
        self.assertEqual(row['secondary_errors'][0]['errno'],errno.ENOSPC)
        for kwargs,kind in ((dict(write_failure='-exit.json'),'exit_record_failure'),
                            (dict(journal_failure=True),'monitor_unavailable')):
            row,_,_=self.child(**kwargs)
            self.assertEqual(row['primary']['kind'],kind)
            self.assertTrue(row['secondary_errors'])

    def test_identity_failure_prevents_signal_and_retains_stop_failure(self):
        def mutate(children,original):
            def proc(pid):
                row=original(pid)
                if children:row['start_ticks']+=1
                return row
            # spawn's original identity is captured first, but monitor's initial
            # identity must stay real; corrupt only later group observations.
            count=0
            def after_spawn(pid):
                nonlocal count
                count+=1
                return original(pid) if count==1 else proc(pid)
            return dict(proc_identity=after_spawn)
        row,_,sent=self.child(mutate=mutate)
        self.assertEqual(row['primary']['operation'],'identity_match')
        self.assertFalse(sent);self.assertFalse(row['stop_confirmed'])
        self.assertTrue(any(e['stage']=='stop_identity_or_signal' for e in row['secondary_errors']))

    def test_known_members_absence_residue_reuse_and_unknown_orphan(self):
        owner=dict(boot_id='test-boot',pid=101,start_ticks=10,pgid=101,sid=101,uid=123,state='S')
        member=dict(owner,pid=102,start_ticks=11)
        known={101:owner,102:member}
        class Proc:
            def iterdir(self):return iter(())
        for number in (errno.ENOENT,errno.ESRCH):
            reads=[]
            def proc(pid):
                reads.append(pid)
                if pid==102:raise OSError(number,'member_disappeared')
                return owner
            with patch.dict(self.g,boot_id=lambda:'test-boot',proc_identity=proc,Path=lambda *a:Proc() if a==('/proc',) else Path(*a)):
                self.assertEqual(self.m.group_members(owner,dict(known)),[owner])
                self.assertEqual(reads.count(102),2)
        def orphan(pid):
            if pid==101:raise FileNotFoundError(errno.ENOENT,'leader_reaped')
            return member
        with patch.dict(self.g,boot_id=lambda:'test-boot',proc_identity=orphan,Path=lambda *a:Proc() if a==('/proc',) else Path(*a)):
            self.assertEqual(self.m.group_members(owner,dict(known)),[member])
        for key in ('start_ticks','pgid','sid','uid'):
            def reused(pid):
                return owner if pid==101 else dict(member,**{key:member[key]+1})
            with patch.dict(self.g,boot_id=lambda:'test-boot',proc_identity=reused,Path=lambda *a:Proc() if a==('/proc',) else Path(*a)):
                with self.assertRaises(self.m.ObservationError):self.m.group_members(owner,dict(known))
        # Unknown survivor cannot become proven merely by matching pgid/sid.
        fields=['S','1','101','101']+['0']*15+['11']
        class Candidate:
            name='102'
            def __truediv__(self,key):return SimpleNamespace(read_text=lambda:'102 (x) '+' '.join(fields))
        with patch.dict(self.g,boot_id=lambda:'test-boot',proc_identity=orphan,
                        Path=lambda *a:SimpleNamespace(iterdir=lambda:iter([Candidate()])) if a==('/proc',) else Path(*a)):
            with self.assertRaisesRegex(self.m.ObservationError,'unproven_remaining_member'):
                self.m.group_members(owner,{101:owner})

    def test_remaining_member_blocks_success_and_stop_error_keeps_primary(self):
        def mutate(children,original):
            calls=0
            def group(owner,known):
                nonlocal calls
                calls+=1
                children[0][0].wait(timeout=3)
                # Synthetic residue only: never a real signal target. The
                # subsequent pre-signal check sees it gone, so no killpg.
                return [dict(owner,pid=999999999,state='S')] if calls<=2 else []
            return dict(group_members=group)
        row,_,sent=self.child(mutate=mutate)
        self.assertEqual(row['primary']['kind'],'remaining_children')
        self.assertEqual(row['exit_code'],0);self.assertFalse(sent)
        self.assertEqual(row['remaining'],[]);self.assertTrue(row['stop_confirmed'])
        def failed(p,*args):
            p.wait(timeout=3)
            raise OSError(errno.EIO,'primary_sample_fault')
        def recovery(children,original):
            real=self.m.group_members;calls=0
            def group(owner,known):
                nonlocal calls
                calls+=1
                if calls>1:
                    raise PermissionError(errno.EACCES,'recovery_membership_fault')
                return real(owner,known)
            return dict(group_members=group)
        row,_,sent=self.child(sample=failed,exit_code=7,mutate=recovery)
        self.assertEqual(row['primary']['errno'],errno.EIO)
        self.assertEqual(row['primary']['child_exit_code'],7)
        self.assertFalse(sent);self.assertFalse(row['stop_confirmed'])
        secondary=next(e for e in row['secondary_errors'] if e['stage']=='stop_identity_or_signal')
        self.assertEqual(secondary['errno'],errno.EACCES)
        self.assertTrue(secondary['location'])


ControlTests.real_monitor=support.monitor()

if __name__=='__main__':unittest.main(verbosity=2)
