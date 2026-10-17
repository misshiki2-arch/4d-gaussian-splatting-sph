"""Step 16 pass-through observations, imported AFTER native bootstrap only.

No fixture, substitute renderer, alternate trainer or GPU isolation lives here.
Full snapshots are limited to scheduled evaluation/fixed/final-save boundaries.
"""
from contextlib import ExitStack
import math
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from formal_checkpoint_identity import file_identity, require
from formal_checkpoint_state import equal_state
from formal_views import ConsumerView
import formal_training_health as health
from real_input_runtime import ConsumedFiles, rng_state, state
from pilot_contract import frame_key, validate_trace, learning_summary
from checkpoint_cuda_acceptance import camera_values
import pilot_diagnostics as diagnostics


def metric_values(name, value, image, gt):
    a = value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)
    numbers=[]
    for i,v in enumerate(a.reshape(-1)):
        x=float(v)
        if name=='psnr' and math.isinf(x) and x>0:
            require(i<3 and torch.equal(image[i],gt[i]), 'PSNR_zero_mse')
            numbers.append(dict(positive_infinity=True,zero_mse=True))
        else:
            require(math.isfinite(x), 'metric_finite'); numbers.append(x)
    return numbers


def save_png(path, tensor):
    array=(tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    with path.open('xb') as stream:
        Image.fromarray(array).save(stream,format='PNG'); stream.flush(); os.fsync(stream.fileno())
    return file_identity(path)


class Observation:
    def __init__(self, prepared, renderer, out, schedule, representatives, probes, fixed_updates, *, resource_check=None,
                 diagnostic_updates=(3000,3100)):
        import train
        self.train=train; self.p=prepared; self.model=prepared.gaussians
        self.c=prepared.inputs.config; self.renderer=renderer; self.out=Path(out)
        self.resource_check=resource_check; self.consumed=ConsumedFiles()
        self.mode=None; self.view=None; self.evaluation=None; self.visible=0
        self.visits={}; self.final_state=None; self.expected_identity=None
        self.diagnostic_updates=diagnostic_updates
        self.diagnostics={}; self.reset_after=None
        cams=prepared.scene.train_cameras[1.0]; tests=prepared.scene.test_cameras[1.0]
        self.spec=dict(schedule, representatives=representatives, probes=list(probes), fixed_updates=list(fixed_updates),
            train=[frame_key(c) for c in sorted(cams,key=lambda c:c.formal_frame.index)],
            evaluation=[frame_key(cams[i%len(cams)]) for i in range(5,30,5)]+[frame_key(c) for c in tests])
        self.trace=dict(events=[],sh=[],population=[],reset=[],checkpoints=[],evaluations=[],fixed=[],probes=[],
                        visits=[],fetches=0,returned=None)
        self.fixed_indices={tuple(frame_key(cam)):i for i,cam in enumerate(prepared.dataloader.dataset.viewpoint_stack)}
        require(len(self.fixed_indices)==len(cams), 'train_camera_identity')
        for rep in representatives:
            require(rep[0]=='train' and tuple(rep[:3]) in self.fixed_indices, 'fixed_train_only')
            cam=prepared.dataloader.dataset.viewpoint_stack[self.fixed_indices[tuple(rep[:3])]]
            require(cam.formal_frame.raw_time==rep[3], 'fixed_time')

    @property
    def k(self): return self.p.health_run.completed+1

    def event(self,name,k): self.trace['events'].append([name,k])

    def resources(self,k,stage):
        if self.resource_check is not None: self.resource_check(k,stage)

    def sh(self, original, *a, **kw):
        before=[self.model.active_sh_degree,self.model.active_sh_degree_t]
        result=original(*a,**kw)
        self.trace['sh'].append(dict(k=self.k,before=before,after=[self.model.active_sh_degree,self.model.active_sh_degree_t]))
        self.event('sh',self.k); return result

    def population(self,original,*a,**kw):
        before=len(self.model._xyz)
        if self.diagnostic_updates is None or self.k!=self.diagnostic_updates[1]:
            result=original(*a,**kw)
        else:
            require('prune' not in self.diagnostics and self.p.health_run.stage=='population', 'prune_stage')
            depth=0; rows=[]; parents=[]; m=self.model
            split=m.densify_and_split; prune=m.prune_points
            def observed_split(*args,**kwargs):
                nonlocal depth
                depth+=1
                try:return split(*args,**kwargs)
                finally:depth-=1
            def observed_prune(mask):
                if depth:
                    parents.append(1);return prune(mask)
                require(not rows, 'prune_once')
                with diagnostics.unchanged(self.p):row=diagnostics.prune_summary(m,mask,a[1],a[2],a[3])
                rows.append(row)
                return prune(mask)
            with patch.object(m,'densify_and_split',observed_split),patch.object(m,'prune_points',observed_prune):
                result=original(*a,**kw)
            require(len(rows)==1 and len(m._xyz)==rows[0]['kept'], 'prune_rows')
            self.diagnostics['prune']=dict(rows[0],attempted_update=self.k,stage='final_mask_after_growth',
                before=before,after=len(m._xyz),net=len(m._xyz)-before,prune_only=kw['prune_only'],split_parent_calls=len(parents))
        self.trace['population'].append(dict(k=self.k,before=before,after=len(self.model._xyz),
            prune_only=kw['prune_only'],size_threshold=a[3]))
        self.event('population',self.k); return result

    def reset(self,original,*a,**kw):
        before=len(self.model._xyz)
        active=self.diagnostic_updates is not None and self.k==self.diagnostic_updates[0]
        if active:
            require('reset' not in self.diagnostics and self.p.health_run.stage=='reset', 'reset_stage')
            with diagnostics.unchanged(self.p):
                views,_=self.capture(self.out/'diagnostics'/('reset-'+str(self.k)+'-before'))
            self.diagnostics['reset']=dict(attempted_update=self.k,completed=self.p.health_run.completed,
                stage='after_population_before_reset',views=views,after_fixed=None)
        result=original(*a,**kw)
        if active:self.reset_after=diagnostics.snapshot(self.p,health=False)
        self.trace['reset'].append(dict(k=self.k,before=before,after=len(self.model._xyz)))
        self.event('reset',self.k); return result

    def step(self,original,*a,**kw):
        if self.k not in self.spec['probes']: return original(*a,**kw)
        groups=self.model.optimizer.param_groups
        before={g['name']:g['params'][0].detach().clone() for g in groups}
        rows={g['name']:dict(lr=float(g['lr']), nonzero_gradient=(0 if g['params'][0].grad is None else
              int(torch.count_nonzero(g['params'][0].grad).item()))) for g in groups}
        result=original(*a,**kw)
        for g in groups:
            rows[g['name']]['changed']=int(torch.count_nonzero(g['params'][0].detach()!=before[g['name']]).item())
        self.trace['probes'].append(dict(k=self.k,rows=len(self.model._xyz),visible=self.visible,groups=rows))
        return result

    def render(self,camera,model,*a,**kw):
        require(model is self.model and kw['formal_config'] is self.c, 'render_owner')
        value=self.renderer(camera,model,*a,**kw)
        if self.mode is not None:
            self.view=dict(frame=frame_key(camera),metrics={}); self.evaluation['views'].append(self.view)
        elif self.k in self.spec['probes']:
            self.visible=int(value['visibility_filter'].sum().item())
        return value

    def metric(self,name,original,*a,**kw):
        value=original(*a,**kw)
        if self.mode is not None:
            require(self.view is not None and name not in self.view['metrics'], 'metric_context')
            self.view['metrics'][name]=metric_values(name,value,*a[:2])
        return value

    def capture(self,directory, *, final=False):
        views=[]; raw_views=[]; directory.mkdir(parents=True)
        dataset=self.p.dataloader.dataset
        with torch.no_grad():
            for rep in self.spec['representatives']:
                image,cam=dataset[self.fixed_indices[tuple(rep[:3])]]
                gt=image.cuda(); device_cam=cam.cuda()
                value=self.renderer(device_cam,self.model,ConsumerView(self.c,'pipeline'),
                    torch.zeros(3,device=self.model._xyz.device),formal_config=self.c)
                health.raw_render(value,gt,self.model)
                rgb=value['render'].clamp(0,1)
                ms={n:metric_values(n,f(rgb,gt),rgb,gt) for n,f in
                    (('l1',self.train.l1_loss),('psnr',self.train.psnr),('ssim',self.train.ssim))}
                alpha=value['alpha']; mask=device_cam.gt_alpha_mask
                require(mask is not None and mask.shape==alpha.shape, 'fixed_mask')
                summary={}
                for name,selected in (('foreground',mask>=.5),('background',mask<.5)):
                    v=alpha[selected]
                    summary[name]=dict(pixels=v.numel(),mean=float(v.mean()) if v.numel() else None)
                images={n:save_png(directory/(str(rep[1])+'-'+n+'.png'),v) for n,v in
                    (('gt',gt),('render',rgb),('alpha',alpha.clamp(0,1).repeat(3,1,1)),
                     ('depth',self.train.easy_cmap(value['depth'][0])))}
                views.append(dict(frame=rep[:3],raw_time=rep[3],metrics=ms,images=images,alpha=summary,
                    visible=int(value['visibility_filter'].sum()),rgb_max=float(rgb.max()),
                    radii_max=float(value['radii'].max()),depth_min=float(value['depth'].min()),depth_max=float(value['depth'].max())))
                if final:
                    raw_views.append(dict(camera=camera_values(cam),output={n:value[n].detach().cpu().contiguous().clone()
                        for n in ('render','alpha','depth','radii','visibility_filter')}))
        return views,raw_views

    def fixed(self,k):
        before=state(self.model); rng=rng_state(self.model)
        if self.reset_after is not None and k==self.diagnostic_updates[0]:
            equal_state(self.reset_after,diagnostics.snapshot(self.p,health=False))
            self.reset_after=None; self.diagnostics['reset']['after_fixed']=k
        views,raw_views=self.capture(self.out/'fixed'/str(k),final=k==self.spec['updates'])
        equal_state(before,state(self.model)); equal_state(rng,rng_state(self.model))
        self.trace['fixed'].append(dict(k=k,views=views)); self.event('fixed',k)
        if k==self.spec['updates']:
            require(self.expected_identity is None, 'final_expected_once')
            self.final_state=before
            with (self.out/'expected.pt').open('xb') as f:
                torch.save(dict(attrs=before[0],groups=before[1],views=raw_views),f); f.flush(); os.fsync(f.fileno())
            self.expected_identity=file_identity(self.out/'expected.pt')

    def save(self,original,k):
        require(k==self.p.health_run.completed, 'save_completed')
        self.resources(k,'before_save')
        if k==self.spec['updates']: self.fixed(k)
        rng=rng_state(self.model); result=original(k)
        if k==self.spec['updates']: equal_state(self.final_state,state(self.model))
        equal_state(rng,rng_state(self.model))
        path=Path(self.p.scene.model_path)/'checkpoints'/('completed-'+str(k))
        # The real writer already validates/reads back each intermediate save.
        files={n:file_identity(path/n) for n in ('complete.json','metadata.json','state.pt')}
        self.trace['checkpoints'].append(dict(k=k,path=str(path),marker=files['complete.json'],files=files))
        self.event('save',k); self.resources(k,'after_save'); return result

    def report(self,original,*a,**kw):
        k=a[1]; active=k==0 or k in self.spec['tests']
        if k==0 and self.diagnostic_updates is not None:
            require(not self.diagnostics and self.trace['fetches']==0 and self.p.health_run.completed==0, 'initial_stage')
            with diagnostics.unchanged(self.p):
                m=self.model
                self.diagnostics['initial']=dict(k=0,stage='before_first_fetch',rows=len(m._xyz),
                    widths=diagnostics.widths(m._scaling,m._scaling_t,m._rotation,m._rotation_r,m.get_scaling_t))
        if not active:
            value=original(*a,**kw)
            if k%100==0: self.resources(k,'periodic')
            return value
        self.resources(k,'before_evaluation')
        before=state(self.model); rng=rng_state(self.model)
        self.mode=k; self.evaluation=dict(k=k,views=[])
        try: value=original(*a,**kw)
        finally: self.mode=None; self.view=None
        equal_state(before,state(self.model)); equal_state(rng,rng_state(self.model))
        self.trace['evaluations'].append(self.evaluation); self.event('evaluation',k)
        if k in self.spec['fixed_updates'] and k!=self.spec['updates']: self.fixed(k)
        self.resources(k,'after_evaluation'); return value

    def run(self,continuation,writer):
        from utils.data_utils import CameraDataset
        obs=self; loader=self.p.dataloader
        class Batches:
            def __iter__(self):
                for batch in loader:
                    obs.trace['fetches']+=1
                    require(obs.trace['fetches']<=obs.spec['updates'] and len(batch)==obs.spec['batch'], 'fetch_budget')
                    for _,cam in batch:
                        key=tuple(frame_key(cam)); require(list(key) in obs.spec['train'], 'train_only')
                        obs.visits[key]=obs.visits.get(key,0)+1
                    yield batch
        m=self.model; t=self.train
        try:
            with ExitStack() as stack:
                def wrap(owner,name,method):
                    original=getattr(owner,name)
                    stack.enter_context(patch.object(owner,name,lambda *a,**kw:method(original,*a,**kw)))
                stack.enter_context(patch.object(t,'render',self.render))
                wrap(m.optimizer,'step',self.step)
                wrap(m,'oneupSHdegree',self.sh); wrap(m,'densify_and_prune',self.population); wrap(m,'reset_opacity',self.reset)
                wrap(self.p.scene,'save',self.save); wrap(t,'training_report',self.report)
                wrap(CameraDataset,'__getitem__',self.consumed.getitem)
                for name,field in (('l1_loss','l1'),('psnr','psnr'),('ssim','ssim'),('msssim','msssim')):
                    original=getattr(t,name)
                    stack.enter_context(patch.object(t,name,lambda *a,_f=field,_o=original,**kw:self.metric(_f,_o,*a,**kw)))
                self.trace['returned']=continuation(self.p._replace(dataloader=Batches()),writer)
        finally:
            self.trace['visits']=[dict(frame=list(k),count=v) for k,v in sorted(self.visits.items())]
        self.consumed.finish(); validate_trace(self.trace,self.spec)
        return self.trace['returned']

    def summary(self): return learning_summary(self.trace)
