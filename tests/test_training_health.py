"""Step 14: real Torch CPU/loop/Adam/checkpoint/PNG, GPU seams only."""
from contextlib import ExitStack
from dataclasses import asdict
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))
import formal_training_fixture as fixture
import training_health_runtime as hr
import training_update_runtime as runtime
import test_training_update_cpu as update_cpu
from test_formal_config import encode
import formal_entry
import formal_inputs
import formal_native_bootstrap
import formal_training_health as health
from formal_run_diagnostic import RunDiagnostic


def setUpModule():
    # One GPU-isolation module lifetime for all classes in this module. The
    # real model retains the very same KNN function installed at first import.
    fixture.connection.ConnectionTests.setUpClass()


def tearDownModule():
    fixture.connection.ConnectionTests.doClassCleanups()


class HealthCPUCase(fixture.CPUCase):
    @classmethod
    def setUpClass(cls):
        pass  # Owned by setUpModule, not recreated for each test class.


class NumericalTests(HealthCPUCase):
    def test_real_metric_consumer_types_and_invalid_loss(self):
        t=self.torch
        image=t.linspace(0.,1.,256*256).reshape(1,1,256,256).expand(1,3,256,256)
        before=image.clone()
        value=self.train.msssim(image,image)
        self.assertIs(type(value),float)
        self.assertAlmostEqual(value,1.,places=6)
        for valid in (value,0, -2., t.tensor(value), t.tensor(0)):
            health.metric(valid,'metric/test')
        self.assertTrue(t.equal(image,before))
        for bad in (True,False,[value],complex(1,0),float('nan'),float('inf'),float('-inf'),
                    t.tensor([value]),t.tensor(True),t.tensor(1+0j),t.tensor(float('nan')),
                    t.tensor(float('inf')),t.ones(2)):
            with self.subTest(type=type(bad).__name__),self.assertRaises(health.HealthError):
                health.metric(bad,'metric/test')
        loss=t.tensor(.25,requires_grad=True)
        health.scalar(loss,'loss/test');loss.backward()
        self.assertEqual(loss.grad.item(),1.)
        for bad in (value,float('nan'),t.tensor(float('nan')),t.tensor(float('inf'))):
            with self.assertRaises(health.HealthError):health.scalar(bad,'loss/test')

    def test_initial_normal_exceptions_and_input_unchanged(self):
        p = self.configured();m = p.gaussians;t = self.torch
        with t.no_grad():
            m._t.fill_(100)
            m._opacity[:2] = m._opacity.new_tensor([[-1000.], [1000.]])
            m.denom[0] = 1;m.t_gradient_accum[0] = -1
        before = self.snapshot(m)
        health.model_state(m,p.inputs.config,0)
        from formal_checkpoint_state import equal_state
        equal_state(before,self.snapshot(m))
        self.assertFalse(m.optimizer.state)
        self.assertTrue(all(g['params'][0].grad is None for g in m.optimizer.param_groups))
        screen=t.zeros_like(m._xyz,requires_grad=True);screen.grad=t.zeros_like(screen)
        m._t.grad=t.zeros_like(m._t)
        health.gradients(m,screen)

    def test_parameter_and_activation_negative_matrix(self):
        p = self.configured();m=p.gaussians;t=self.torch
        for attr, value in (('_xyz',float('nan')),('_t',float('inf')),('_features_rest',float('-inf')),
                            ('_scaling',1000.),('_scaling',-1000.),('_scaling_t',1000.),
                            ('_rotation',0.),('_rotation_r',0.)):
            with self.subTest(attr=attr,value=str(value)), t.no_grad():
                parameter=getattr(m,attr);original=parameter.clone();parameter.fill_(value)
                with self.assertRaises(health.HealthError):health.model_state(m,p.inputs.config,0)
                parameter.copy_(original)
        for alteration in ('shape','dtype','binding','empty'):
            with self.subTest(alteration=alteration):
                original=m._xyz
                value=original.detach().clone()
                if alteration=='shape':value=value[:,:2]
                elif alteration=='dtype':value=value.double()
                elif alteration=='empty':value=value[:0]
                m._xyz=t.nn.Parameter(value)
                with self.assertRaises(health.HealthError):health.model_state(m,p.inputs.config,0)
                m._xyz=original

    def test_statistics_normal_and_reject_without_repair(self):
        p=self.configured();m=p.gaussians;t=self.torch
        health.statistics(m)
        for name,value in (('denom',-.1),('denom',.5),('denom',float('inf')),
                            ('xyz_gradient_accum',-1),('xyz_gradient_accum',1),
                            ('t_gradient_accum',-1),('max_radii2D',-1)):
            with self.subTest(name=name,value=value):
                a=getattr(m,name);a[0]=value;before=a.clone()
                with self.assertRaises(health.HealthError):health.statistics(m)
                self.assertTrue(t.equal(a,before));a.zero_()
        m.denom[0]=1;m.t_gradient_accum[0]=-2;health.statistics(m)
        m.xyz_gradient_accum[0]=float('nan')
        with self.assertRaises(health.HealthError):health.statistics(m)
        self.assertTrue(t.isnan(m.xyz_gradient_accum[0]))

    def test_adam_state_and_binding(self):
        p=self.configured();m=p.gaussians;t=self.torch
        m.env_map=None;m.update_learning_rate(1)
        for g in m.optimizer.param_groups:g['params'][0].grad=t.zeros_like(g['params'][0])
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        health.model_state(m,p.inputs.config,1)
        for field,value in (('step',2.),('step',.5),('exp_avg',float('nan')),('exp_avg_sq',-1.)):
            s=m.optimizer.state[m._xyz];old=s[field].clone();s[field].fill_(value)
            with self.subTest(field=field,value=str(value)),self.assertRaises(health.HealthError):
                health.model_state(m,p.inputs.config,1)
            s[field].copy_(old)
        m.optimizer.state[t.nn.Parameter(t.zeros(1))]={}
        with self.assertRaisesRegex(health.HealthError,'extra_state'):health.model_state(m,p.inputs.config,1)

    def test_raw_outputs_gradients_and_scalar_matrix(self):
        p=self.configured();m=p.gaussians;t=self.torch
        gt=t.zeros(3,2,2)
        raw=dict(render=t.full_like(gt,-2),alpha=t.full((1,2,2),2.),depth=t.zeros(1,2,2),
            radii=t.zeros(len(m._xyz),dtype=t.int32),visibility_filter=t.zeros(len(m._xyz),dtype=t.bool),
            viewspace_points=t.zeros_like(m._xyz))
        health.raw_render(raw,gt,m)
        for key in ('render','alpha','depth','viewspace_points'):
            bad={**raw,key:t.full_like(raw[key],float('inf'))}
            with self.subTest(key=key),self.assertRaises(health.HealthError):health.raw_render(bad,gt,m)
        for bad in ({**raw,'radii':-t.ones_like(raw['radii'])},
                    {**raw,'visibility_filter':t.zeros(len(m._xyz))}):
            with self.assertRaises(health.HealthError):health.raw_render(bad,gt,m)
        with self.assertRaisesRegex(health.HealthError,'required_gradient'):health.gradients(m,raw['viewspace_points'])
        for bad in (t.tensor(float('nan')),t.tensor(float('inf')),t.ones(2)):
            with self.assertRaises(health.HealthError):health.scalar(bad,'test')

    def test_depth_and_channel_psnr(self):
        from utils.image_utils import easy_cmap,psnr
        t=self.torch
        for c in (0.,2.,-3.):
            a=t.full((2,3),c);before=a.clone();b=easy_cmap(a)
            self.assertEqual(tuple(b.shape),(3,2,3));self.assertTrue(t.equal(b,t.zeros_like(b)))
            self.assertTrue(t.equal(a,before))
        a=t.tensor([[-1.,0.],[1.,3.]])
        t.testing.assert_close(easy_cmap(a),t.tensor([[0.,.25],[.5,1.]]).expand(3,2,2),rtol=0,atol=0)
        for a in (t.zeros(0,1),t.tensor([[float('nan')]]),t.tensor([[float('inf')]]),
                  t.tensor([[-3e38,3e38]])):
            with self.assertRaises(ValueError):easy_cmap(a)
        gt=t.zeros(3,2,2)
        for image in (gt.clone(),t.stack([gt[0],t.ones_like(gt[0]),2*t.ones_like(gt[0])]),t.full_like(gt,2.)):
            before=image.clone();health.psnr_channels(psnr(image,gt),image,gt)
            self.assertTrue(t.equal(before,image))
        self.assertTrue(bool((psnr(t.full_like(gt,2),gt)<0).all()))
        for value in (float('nan'),float('-inf'),float('inf')):
            with self.assertRaises(health.HealthError):
                health.psnr_channels(t.full((3,1),value),t.ones_like(gt),gt)
        tiny=t.full_like(gt,1e-30)
        with self.assertRaises(health.HealthError):health.psnr_channels(psnr(tiny,gt),tiny,gt)


class LoopTests(HealthCPUCase):
    def renderer(self,*a,**kw):
        return update_cpu.ConsumerTests.renderer(self,*a,**kw)

    def prepare_case(self,mode):
        case=dict(name='health',n=3,batch=2,kind='mixed' if mode in ('empty_population','reset') else 'initial')
        value,reference=runtime.create_fixture(self.f.base/mode,case)
        value['checkpoint']['intermediate_updates']=[1,2]
        value['reporting']['test_updates']=[2,3]
        if mode=='statistics':
            value['optimization']['population'].update(densify_from_update=3,densify_until_update=4)
        with patch.object(formal_inputs,'_reference',return_value=reference):
            p=self.train.prepare_formal_runtime(formal_inputs.verify_inputs(encode(value)),
                                                native_evidence=self.f.checkpoint_evidence())
        runtime.initialize_population(p,case)
        return p,reference

    def test_real_boundary_matrix_and_prior_checkpoint_preservation(self):
        for mode in ('normal','loss','gradient','statistics','post_adam','empty_population','reset',
                     'evaluation','metric','save','png','writer'):
            with self.subTest(mode=mode):
                p,reference=self.prepare_case(mode)
                with ExitStack() as stack:
                    stack.enter_context(patch.object(formal_inputs,'_reference',return_value=reference))
                    stack.enter_context(patch.object(self.torch.cuda,'Event',update_cpu.Event))
                    stack.enter_context(patch.object(self.train,'TENSORBOARD_FOUND',False))
                    claim=formal_entry.OutputClaim(p.inputs);stack.callback(claim.close)
                    self.train.begin_formal_output(p,claim)
                    result=hr.exercise(p,self.renderer,mode,Mock() if mode=='writer' else None)
                self.assertEqual(result['status'],'passed')
                if mode=='normal':self.assertEqual(result['reports'],[0,1,2,3])
                print('HEALTH_CPU_BOUNDARY',mode,result['outcome'],result['counts'])
        self.assertFalse(self.torch.cuda.is_initialized())

    def test_invisible_zero_grad_constant_depth_real_png(self):
        p,reference=self.prepare_case('invisible');t=self.torch
        def invisible(camera,model,*a,**kw):
            screen=t.zeros_like(model._xyz,requires_grad=True)
            zero=screen.sum()*0+sum(g['params'][0].sum()*0 for g in model.optimizer.param_groups)
            h,w=camera.image_height,camera.image_width
            return dict(render=zero.expand(3,h,w),alpha=zero.expand(1,h,w),depth=t.zeros(1,h,w),
                radii=t.zeros(len(screen),dtype=t.int32),visibility_filter=t.zeros(len(screen),dtype=t.bool),viewspace_points=screen)
        with patch.object(formal_inputs,'_reference',return_value=reference),patch.object(t.cuda,'Event',update_cpu.Event), \
                patch.object(self.train,'TENSORBOARD_FOUND',False):
            claim=formal_entry.OutputClaim(p.inputs)
            try:
                self.train.begin_formal_output(p,claim)
                result=hr.exercise(p,invisible,'invisible')
            finally:claim.close()
        pngs=list(Path(p.scene.model_path).rglob('*_depth.png'));self.assertTrue(pngs)
        for path in pngs:
            with self.f.Image.open(path) as image:self.assertEqual(image.getextrema(),((0,0),(0,0),(0,0)))
        self.assertEqual(result['outcome']['returned'],3)

    def test_entry_writer_termination_and_primary_exception(self):
        for mode in ('normal','absent','flush','close','primary'):
            with self.subTest(mode=mode):
                case=dict(name='G0a',n=1,batch=1,kind='initial')
                _,reference=runtime.create_fixture(self.f.base/('entry-'+mode),case)
                writer=Mock();failure=OSError('primary-io')
                if mode=='flush':writer.flush.side_effect=failure
                if mode in ('close','primary'):writer.close.side_effect=OSError('secondary-close')
                if mode=='primary':writer.add_scalar.side_effect=failure
                real_prepare=self.train.prepare_formal_runtime;held={}
                def prepare(*a,**kw):
                    p=real_prepare(*a,**kw);held['p']=p
                    real_save=p.scene.save
                    def save(k):
                        result=real_save(k)
                        directory=Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k))
                        held['before_termination']=hr.files(directory)
                        return result
                    p.scene.save=save
                    return p
                with patch.object(formal_inputs,'_reference',return_value=reference), \
                        patch.object(self.torch.cuda,'Event',update_cpu.Event),patch.object(self.train,'render',self.renderer), \
                        patch.object(formal_native_bootstrap,'bootstrap',return_value=(self.train,self.f.checkpoint_evidence())), \
                        patch.object(self.train,'prepare_formal_runtime',prepare), \
                        patch.object(self.train,'TENSORBOARD_FOUND',mode!='absent'), \
                        patch.object(self.train,'SummaryWriter',return_value=writer,create=True), \
                        patch.object(sys,'stderr',io.StringIO()) as stderr:
                    locator=self.f.base/('entry-'+mode)/'config.json'
                    if mode in ('normal','absent'):
                        self.assertEqual(formal_entry.run(locator),1)
                    else:
                        with self.assertRaises(OSError) as caught:formal_entry.run(locator)
                        if mode!='close':self.assertIs(caught.exception,failure)
                        self.assertIn('formal_failure',stderr.getvalue())
                    if mode!='primary':
                        import formal_checkpoint
                        directory=Path(held['p'].scene.model_path)/'checkpoints/completed-1'
                        self.assertEqual(hr.files(directory),held['before_termination'])
                        self.assertEqual(formal_checkpoint.read_checkpoint(directory)['metadata']['k'],1)
                self.assertEqual(writer.close.call_count,0 if mode=='absent' else 1)
                self.assertEqual(held['p'].health_run.completed,0 if mode=='primary' else 1)


class DiagnosticTests(unittest.TestCase):
    def test_only_local_test_classes_collected(self):
        def leaves(suite):
            for item in suite:
                if isinstance(item,unittest.TestSuite):yield from leaves(item)
                else:yield item
        found=list(leaves(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])))
        self.assertTrue(found)
        self.assertTrue(all(type(t).__module__==__name__ for t in found))

    def test_broken_stderr_does_not_mask_primary(self):
        r=RunDiagnostic()
        with patch.object(sys,'stderr',Mock(write=Mock(side_effect=OSError))):r.failure(RuntimeError())
        self.assertEqual(asdict(r)['completed'],0)


if __name__=='__main__':unittest.main()
