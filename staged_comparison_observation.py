"""B-only observation adapter; reuse real training and independent CPU snapshots."""
from contextlib import ExitStack
import types
from unittest.mock import patch

from formal_checkpoint_identity import require
import staged_comparison_support as support
import staged_comparison_storage as storage


def local_dataset(dataset, index):
    """Validate the original before making a new CPU-validated camera owner."""
    from utils.formal_camera_runtime import camera_binding, bind_camera
    camera = dataset.viewpoint_stack[index]
    camera_binding(camera, dataset.formal_resolution)
    local = support.memory_helper().local_dataset(dataset, index)
    copied = local.viewpoint_stack[index]
    copied.formal_binding = bind_camera(copied)
    return local


def observation_class():
    # Delayed until the caller has validated provenance and loaded exact native.
    import torch
    import pilot_runtime
    from real_input_runtime import ConsumedFiles
    old=support.module('_step18_old_observation',support.IMPL/'comparison_observation.py')
    import comparison_state
    mem=support.memory_helper()
    # Local method globals; old helper files/modules are not modified.
    def clone(method, **changes):
        return types.FunctionType(method.__code__,dict(method.__globals__,**changes),
                                  method.__name__,method.__defaults__,method.__closure__)
    class Guard(comparison_state.ModelGuard):
        check=clone(comparison_state.ModelGuard.check,equal_live=mem.equal_live)
    class Consumed(ConsumedFiles):
        def getitem(self, original, dataset, index):
            return super().getitem(original,local_dataset(dataset,index),index)
    class Observation(old.Observation):
        _fixed=clone(old.Observation.fixed,equal_live=mem.equal_live,ModelGuard=Guard)
        _save=clone(old.Observation.save,equal_live=mem.equal_live,ModelGuard=Guard)
        _report=clone(old.Observation.report,equal_live=mem.equal_live,ModelGuard=Guard)
        run=clone(pilot_runtime.Observation.run,validate_trace=validate_trace)
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.consumed=Consumed();self.optimizer_calls=0
            self.trace['population_checks']=[];self.trace['reset_checks']=[]
        def fixed(self,k):
            self.resources(k,'before_evaluation')
            storage.check(self.out,'fixed',len(self.model._xyz),**self.storage_shape())
            return self._fixed(k)
        def storage_shape(self):
            camera=self.p.dataloader.dataset.viewpoint_stack[0]
            return dict(width=camera.image_width,height=camera.image_height,
                        tests=len(self.p.scene.test_cameras[1.0]))
        def capture(self,directory,*,final=False):
            result=super().capture(directory,final=final)
            if final:
                # The fixed PNGs now occupy disk. Only the still-unwritten
                # independent expected/raw is counted before opening expected.pt.
                storage.check(self.out,'expected',len(self.model._xyz),**self.storage_shape())
            return result
        def save(self,original,k):
            def checked(label):
                # Old save captures final expected first. Recheck AFTER it and
                # immediately before the actual checkpoint transaction.
                storage.check(self.p.scene.model_path,'checkpoint',len(self.model._xyz))
                return original(label)
            return self._save(checked,k)
        def report(self,original,*a,**kw):
            k=a[1]
            if k==0 or k in self.spec['tests']:
                storage.check(self.p.scene.model_path,'ordinary',len(self.model._xyz),
                              tensorboard=a[0] is not None,**self.storage_shape())
            return self._report(original,*a,**kw)
        def step(self,original,*a,**kw):
            # Called after the existing population eligibility decision, before
            # mutation. Record actual row count, not a forecast of growth.
            n=len(self.model._xyz);o=self.c.optimization.population;k=self.k
            active=k<o.densify_until_update and (o.densify_stop_points=='unlimited' or n<o.densify_stop_points)
            if k>o.densify_from_update and k%o.densification_interval==0:
                self.trace['population_checks'].append(dict(k=k,rows=n,active=active))
            if k%o.opacity_reset_interval==0:
                self.trace['reset_checks'].append(dict(k=k,active=active))
            result=original(*a,**kw);self.optimizer_calls+=1
            return result
        def population(self,original,*a,**kw):
            m=self.model
            self.resources(self.k,'periodic')  # Selection-array allowance before observation.
            with ExitStack() as stack:
                for name,split in (('densify_and_clone',False),('densify_and_split',True)):
                    target=getattr(m,name)
                    def checked(grads,threshold,extent,*rest,_target=target,_split=split,**options):
                        # Read-only projection of the existing selection. No
                        # RNG, mask replacement, parameter or gradient mutation.
                        n=len(m._xyz)
                        if _split:
                            score=grads.new_zeros(n);score[:len(grads)]=grads.squeeze()
                            selected=(score>=threshold)&(m.get_scaling.max(dim=1).values>m.percent_dense*extent)
                            added=int(selected.sum())*options.get('N',2)
                        else:
                            selected=(torch.norm(grads,dim=-1)>=threshold)&(m.get_scaling.max(dim=1).values<=m.percent_dense*extent)
                            added=int(selected.sum())
                        del selected
                        support.topology(self.out,n,added)
                        return _target(grads,threshold,extent,*rest,**options)
                    stack.enter_context(patch.object(m,name,checked))
                return super().population(original,*a,**kw)
    return Observation


def schedule(c):
    p=c.optimization.population;n=c.optimization.total_updates
    return dict(updates=n,batch=c.optimization.batch_size,sh=[],population=[],reset=[],
        saves=list(c.save_updates),tests=list(c.reporting.test_updates),
        growth_threshold=p.densify_stop_points,reset_interval=p.opacity_reset_interval)


def validate_trace(trace,spec):
    # Reuse the full event/order/frame/metric validator. Only the comparison
    # policy changes: threshold stops the whole block instead of prune-only.
    import pilot_contract
    expected=dict(spec,population=[v['k'] for v in trace['population_checks'] if v['active']],
                  reset=[v['k'] for v in trace['reset_checks'] if v['active']],
                  growth_threshold='unlimited')
    # Actual calls must correspond to the pre-Adam decision and row identity.
    checks={v['k']:v for v in trace['population_checks']}
    for row in trace['population']:
        require(row['k'] in checks and checks[row['k']]['active'] and
                row['before']==checks[row['k']]['rows'] and row['prune_only'] is False,'comparison_population')
    return pilot_contract.validate_trace(trace,expected)


def expected_equal(expected, loaded):
    """Independent attribute/group representation, chunked comparison, not capture()."""
    mem=support.memory_helper();attrs,groups=expected
    a=loaded['state']['capture']
    require(len(a)==19,'capture_arity')
    for slot,value in attrs.items():mem.equal_live(value,a[slot])
    require(len(a[11]['param_groups'])==len(groups)==9,'group_count')
    for group in a[11]['param_groups']:
        saved=groups[group['name']]
        mem.equal_live(saved['settings'],{k:v for k,v in group.items() if k!='params'})
        mem.equal_live(saved['state'],a[11]['state'].get(group['params'][0],{}))
