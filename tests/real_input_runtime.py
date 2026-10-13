"""Real Step 15 observers. Import only AFTER ordinary native bootstrap.

No fixture creation, reference overrides, surrogate rendering or retry here.
CPU tests install their GPU isolation outside this module.
"""
from contextlib import ExitStack
import inspect
import math
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import torch

import training_update_oracle as oracle
from formal_checkpoint_identity import file_identity, require
from formal_checkpoint_state import equal_state
from real_input_contract import frame_key, validate_trace
from test_formal_checkpoint import independent, assert_expected


def rng_state(model):
    n=np.random.get_state()
    return (random.getstate(),(n[0],n[1].tobytes(),n[2:]),torch.get_rng_state().clone(),
            torch.cuda.get_rng_state(model._xyz.device).clone() if model._xyz.is_cuda else None)


def state(model):
    return independent(model)


def group_arrays(model):
    result={}
    for g in model.optimizer.param_groups:
        p=g['params'][0];name=g['name']
        require(p is getattr(model,oracle.GROUPS[name]) and p.is_leaf,'leaf_owner')
        a=oracle.array(p);s=model.optimizer.state.get(p,{})
        result[name]=dict(p=a,m=oracle.array(s['exp_avg']) if s else np.zeros_like(a),
            v=oracle.array(s['exp_avg_sq']) if s else np.zeros_like(a),step=int(s['step'].item()) if s else 0)
    require(set(result)==set(oracle.GROUPS) and len(model.optimizer.param_groups)==9,'group_coverage')
    return result


class ConsumedFiles:
    """Hash only files actually read by the real CameraDataset consumer."""
    def __init__(self):self.identities={};self.metadata={}

    def before(self,path):
        path=Path(path);s=path.stat();stamp=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        name=str(path)
        if name not in self.identities:
            self.identities[name]=file_identity(path);self.metadata[name]=stamp
        require(self.metadata[name]==stamp,'consumed_metadata_changed')

    def finish(self):
        for path,value in self.identities.items():
            self.before(path);require(file_identity(Path(path))==value,'consumed_bytes_changed')
        return dict(self.identities)

    def getitem(self,original,dataset,index):
        camera=dataset.viewpoint_stack[index]
        paths=(camera.image_path,camera.image_path.replace('/images/','/masks/'))
        for p in paths:self.before(p)
        result=original(dataset,index)
        for p in paths:self.before(p)
        require(result[1] is camera,'dataset_camera_return')
        return result


class Observation:
    def __init__(self,p,renderer,*,save_observer=None):
        import train
        self.train,self.p,self.model,self.renderer=train,p,p.gaussians,renderer
        self.c=p.inputs.config;self.k=0;self.mode=None;self.current_view=None;self.lr_token=None
        self.save_observer=save_observer;self.consumed=ConsumedFiles()
        self.trace=dict(events=[],reports=[],steps=[],batches=[],evaluations=[],checkpoints=[],forwards=0,returned=None)
        cams=p.scene.train_cameras[1.0];tests=p.scene.test_cameras[1.0]
        self.spec=dict(updates=self.c.optimization.total_updates,rows=len(self.model._xyz),
            saves=list(self.c.save_updates),tests=list(self.c.reporting.test_updates),
            train=[frame_key(c) for c in sorted(cams,key=lambda c:c.formal_frame.index)],
            evaluation=[frame_key(cams[i%len(cams)]) for i in range(5,30,5)]+[frame_key(c) for c in tests])
        self.visible=0;self.handles=[];self.evaluation=None

    def event(self,name):self.trace['events'].append([name,self.k])

    def lr(self,original,k):
        require(k==self.k+1 and self.lr_token is None,'lr_sequence');self.k=k
        result=original(k);self.event('lr')
        require(self.model.spatial_lr_scale==self.p.scene.cameras_extent,'extent_handoff')
        rates=oracle.learning_rates(self.c,self.model.spatial_lr_scale,k)
        token={}
        for g in self.model.optimizer.param_groups:
            require(math.isclose(float(g['lr']),rates[g['name']],rel_tol=1e-12,abs_tol=0),'independent_lr')
            token[g['name']]=(g['params'][0],type(g['lr']),float(g['lr']))
        require(set(token)==set(oracle.GROUPS),'lr_groups');self.lr_token=(k,token)
        return result

    def render(self,camera,model,*a,**kw):
        require(model is self.model and kw['formal_config'] is self.c,'render_owner')
        require(len(model._xyz)==self.spec['rows'],'rows_changed')
        self.trace['forwards']+=1
        limit=self.spec['updates']+len(self.spec['evaluation'])*(1+len(self.spec['tests']))
        require(self.trace['forwards']<=limit,'forward_budget')
        result=self.renderer(camera,model,*a,**kw)
        if self.mode is None:
            require((model.active_sh_degree,model.active_sh_degree_t)==(0,0),'active_SH')
            self.event('render');self.visible=int(result['visibility_filter'].sum().item())
            require(self.visible>0 and bool((result['alpha']>0).any()),'visible_update')
        else:
            row=dict(frame=frame_key(camera),metrics={})
            self.evaluation['views'].append(row);self.current_view=row
        return result

    def backward(self,original,tensor,*a,**kw):
        frame=inspect.currentframe().f_back
        # The patch's small forwarding lambda adds one frame.
        if frame.f_code is not self.train.training.__code__:frame=frame.f_back
        try:
            require(self.mode is None and frame.f_code is self.train.training.__code__,'backward_owner')
            local=frame.f_locals
            require(local['iteration']==self.k and tensor is local['loss'],'backward_loss')
            loss=float(tensor.detach().item());require(math.isfinite(loss),'input_loss_finite')
        finally:del frame
        result=original(tensor,*a,**kw);self.event('backward')
        self.last_loss=loss
        return result

    def statistics(self,original,*a,**kw):
        result=original(*a,**kw);self.event('statistics');return result

    def step(self,original,*a,**kw):
        require(self.mode is None and self.lr_token is not None and self.lr_token[0]==self.k,'lr_gate')
        token=self.lr_token[1];self.lr_token=None
        before=group_arrays(self.model);gradients={};rates={}
        for g in self.model.optimizer.param_groups:
            name=g['name'];p=g['params'][0]
            require(token[name]==(p,type(g['lr']),float(g['lr'])),'lr_after_gate')
            gradients[name]=None if p.grad is None else oracle.array(p.grad)
            rates[name]=float(g['lr'])
        nonzero=sum(int(np.count_nonzero(g)) for g in gradients.values() if g is not None)
        require(nonzero>0,'nonzero_gradient')
        expected={name:oracle.adam(v['p'].astype(np.float64),v['m'].astype(np.float64),
            v['v'].astype(np.float64),v['step'],gradients[name],rates[name]) for name,v in before.items()}
        result=original(*a,**kw);self.event('step');after=group_arrays(self.model)
        records=[]
        for name,v in after.items():
            for key,want,tolerance in zip(('p','m','v','step'),expected[name],('parameter','moment','variance','parameter')):
                oracle.compare(v[key],want,name+'/'+key,records,tolerance,exact=key=='step')
        changed=sum(int(np.count_nonzero(after[n]['p']!=v['p'])) for n,v in before.items() if rates[n]>0)
        require(changed>0,'positive_lr_parameter_delta')
        self.trace['steps'].append(dict(k=self.k,rows=len(self.model._xyz),visible=self.visible,
            nonzero_gradients=nonzero,changed_positive_lr=changed,lr=rates,comparisons=records,input_loss=self.last_loss))
        return result

    def zero(self,original,*a,**kw):
        result=original(*a,**kw)
        require(all(g['params'][0].grad is None for g in self.model.optimizer.param_groups),'zero_grad')
        self.event('zero');return result

    def metric(self,name,original,*a,**kw):
        value=original(*a,**kw)
        if self.mode is not None:
            row=self.current_view
            require(row is not None and name not in row['metrics'],'metric_context')
            array=oracle.array(value).reshape(-1);values=[]
            for i,scalar in enumerate(array):
                x=float(scalar)
                if math.isinf(x) and x>0 and name=='psnr':
                    image,gt=(oracle.array(v).reshape(3,-1) for v in a[:2])
                    require(i<3 and np.array_equal(image[i],gt[i]),'PSNR_zero_mse')
                    values.append(dict(positive_infinity=True,zero_mse=True))
                else:require(math.isfinite(x),'metric_finite');values.append(x)
            row['metrics'][name]=values
        return value

    def save(self,original,k):
        require(k==self.k==self.p.health_run.completed,'save_completed')
        self.event('save_begin')
        expected=state(self.model)
        if self.save_observer is not None:self.save_observer(k,expected)
        result=original(k)
        import formal_checkpoint as cp
        import unittest
        path=Path(self.p.scene.model_path)/'checkpoints'/('completed-'+str(k))
        loaded=cp.read_checkpoint(path);assert_expected(unittest.TestCase(),expected,loaded)
        require(loaded['metadata']['k']==k and loaded['metadata']['n']==self.spec['updates'],'saved_labels')
        self.trace['checkpoints'].append(dict(k=k,path=str(path),marker=loaded['marker_identity'],
            files={name:file_identity(path/name) for name in ('complete.json','metadata.json','state.pt')}))
        self.event('save_end');return result

    def report(self,original,*a,**kw):
        k=a[1];require(k==self.k==self.p.health_run.completed,'report_completed')
        before=state(self.model);rng=rng_state(self.model)
        self.event('report_begin')
        active=k==0 or k in self.spec['tests']
        self.mode=k if active else None;self.evaluation=dict(k=k,views=[])
        try:result=original(*a,**kw)
        finally:self.mode=None;self.current_view=None
        equal_state(before,state(self.model));equal_state(rng,rng_state(self.model))
        if active:self.trace['evaluations'].append(self.evaluation)
        self.trace['reports'].append(k);self.event('report_end');return result

    def run(self,continuation,writer):
        from utils.data_utils import CameraDataset
        obs=self;loader=self.p.dataloader
        class Batches:
            def __iter__(self):
                for batch in loader:
                    obs.trace['events'].append(['fetch',obs.k+1])
                    obs.trace['batches'].append([frame_key(cam) for _,cam in batch])
                    require(len(obs.trace['batches'])<=obs.spec['updates'],'N_plus_one_fetch')
                    yield batch
        m=self.model;t=self.train
        def forbidden(*a,**kw):raise AssertionError('unexpected_population_or_SH_event')
        with ExitStack() as stack:
            def wrap(owner,name,method):
                original=getattr(owner,name)
                stack.enter_context(patch.object(owner,name,lambda *a,**kw:method(original,*a,**kw)))
            stack.enter_context(patch.object(t,'render',self.render))
            wrap(m,'update_learning_rate',self.lr);wrap(m.optimizer,'step',self.step)
            wrap(m.optimizer,'zero_grad',self.zero);wrap(m,'add_densification_stats',self.statistics)
            wrap(self.p.scene,'save',self.save);wrap(t,'training_report',self.report)
            wrap(torch.Tensor,'backward',self.backward);wrap(CameraDataset,'__getitem__',self.consumed.getitem)
            for name in ('densify_and_prune','reset_opacity','oneupSHdegree'):
                stack.enter_context(patch.object(m,name,forbidden))
            for name,field in (('l1_loss','l1'),('psnr','psnr'),('ssim','ssim'),('msssim','msssim')):
                original=getattr(t,name)
                stack.enter_context(patch.object(t,name,lambda *a,_field=field,_original=original,**kw:
                    self.metric(_field,_original,*a,**kw)))
            self.trace['returned']=continuation(self.p._replace(dataloader=Batches()),writer)
        self.consumed.finish();validate_trace(self.trace,self.spec)
        return self.trace['returned']
