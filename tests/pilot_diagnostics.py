"""Three bounded pilot observations; heavy imports only after bootstrap."""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from unittest.mock import patch
import copy
import numpy as np
import torch
from formal_checkpoint_identity import require
from formal_checkpoint_state import equal_state
from real_input_runtime import state, rng_state
from training_update_oracle import GROUPS


def summary(a):
    a=np.asarray(a).reshape(-1)
    require(len(a)>0 and np.isfinite(a).all(), 'width_finite')
    return dict(count=len(a),finite=len(a),mean=float(a.mean()),quantiles=dict(zip(
        ('min','p01','p10','p25','median','p75','p90','p99','max'),
        map(float,np.quantile(a,[0,.01,.1,.25,.5,.75,.9,.99,1])))))


def temporal_arrays(log_spatial, log_time, left, right):
    """Independent NumPy covariance from Investigation1, no native helper."""
    arrays=[v.detach().cpu().numpy().astype('float64') for v in (log_spatial,log_time,left,right)]
    n=len(arrays[0])
    require(n>0 and all(a.shape==(n,d) and np.isfinite(a).all() for a,d in zip(arrays,(3,1,4,4))), 'width_layout')
    spatial,temporal,l,r=arrays
    scales=np.exp(np.concatenate((spatial,temporal),axis=1));variance=np.empty(n)
    require(np.isfinite(scales).all() and (scales>0).all(), 'width_activation')
    for offset in range(0,n,32768):
        a=l[offset:offset+32768].copy();b=r[offset:offset+32768].copy()
        an=np.linalg.norm(a,axis=1,keepdims=True);bn=np.linalg.norm(b,axis=1,keepdims=True)
        require((an>0).all() and (bn>0).all(), 'width_rotation')
        a/=an;b/=bn
        aa,bb,cc,dd=a.T;p,q,rr,ss=b.T
        ml=np.stack((aa,-bb,-cc,-dd,bb,aa,-dd,cc,cc,dd,aa,-bb,dd,-cc,bb,aa),axis=1).reshape(-1,4,4)
        mr=np.stack((p,q,rr,ss,-q,p,-ss,rr,-rr,ss,p,-q,-ss,-rr,q,p),axis=1).reshape(-1,4,4)
        rotation=(ml@mr)[:,::-1,::-1]
        variance[offset:offset+len(a)]=np.sum((rotation[:,3,:]*scales[offset:offset+len(a)])**2,axis=1)
    require(np.isfinite(variance).all() and (variance>0).all(), 'width_covariance')
    return temporal[:,0],scales[:,3],variance


def widths(log_spatial,log_time,left,right,activated=None):
    ell,scale,variance=temporal_arrays(log_spatial,log_time,left,right)
    if activated is not None:
        actual=activated.detach().cpu().numpy().reshape(-1)
        require(actual.shape==scale.shape and np.allclose(actual,scale,rtol=2e-7,atol=0), 'width_actual_activation')
        scale=actual
    return dict(log_scale=summary(ell),scale=summary(scale),variance=summary(scale.astype('float64')**2),
                sigma_tt=summary(variance),rotated_sigma=summary(np.sqrt(variance)))


def snapshot(p, *, health=True):
    m=p.gaussians
    params={name:(id(getattr(m,attr)),None if getattr(m,attr).grad is None else
                  getattr(m,attr).grad.detach().cpu().clone()) for name,attr in GROUPS.items()}
    generators={}
    for name,owner in (('loader',p.dataloader),('sampler',p.dataloader.sampler)):
        gen=getattr(owner,'generator',None)
        if gen is not None:generators[name]=(id(gen),gen.get_state().clone())
    return dict(values=state(m),params=params,bindings=[(g['name'],[id(v) for v in g['params']]) for g in m.optimizer.param_groups],
        rng=rng_state(m),generators=generators,health=copy.deepcopy(asdict(p.health_run)) if health else None,
        time_duration=tuple(m.time_duration))


@contextmanager
def unchanged(p):
    before=snapshot(p)
    def forbidden(*a,**kw):raise ValueError('diagnostic_training_mutation')
    with ExitStack() as stack, torch.no_grad():
        for owner,name in ((torch.Tensor,'backward'),(torch.autograd,'backward'),(p.gaussians.optimizer,'step'),
                           (p.gaussians,'densify_and_prune'),(p.gaussians,'reset_opacity')):
            stack.enter_context(patch.object(owner,name,forbidden))
        yield
    equal_state(before,snapshot(p))


def prune_summary(model,mask,min_opacity,extent,max_screen_size):
    """Use current rows and original mask; never replace the delegated object."""
    with torch.no_grad():
        o=(model.get_opacity<min_opacity).squeeze(-1)
        s=model.max_radii2D>max_screen_size if max_screen_size else torch.zeros_like(o)
        w=model.get_scaling.max(dim=1).values>.1*extent if max_screen_size else torch.zeros_like(o)
        require(mask.dtype==torch.bool and mask.shape==o.shape and torch.equal(mask,o|s|w), 'prune_original_mask')
        code=o.to(torch.int64)+2*s.to(torch.int64)+4*w.to(torch.int64)
        bins=torch.bincount(code,minlength=8).cpu().tolist()
    return dict(rows=len(mask),thresholds=dict(opacity=float(min_opacity),extent=float(extent),screen=max_screen_size),
        bins=bins,inclusive=dict(opacity=int(o.sum()),screen=int(s.sum()),world=int(w.sum())),
        overlaps=dict(opacity_screen=int((o&s).sum()),opacity_world=int((o&w).sum()),screen_world=int((s&w).sum()),triple=int((o&s&w).sum())),
        union=int(mask.sum()),kept=int((~mask).sum()))
