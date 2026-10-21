"""Authorized cold-child adapters. Never a second trainer or build fallback."""
from contextlib import contextmanager, closing
import importlib.util
from pathlib import Path
import sys
import json
from unittest.mock import patch

from formal_checkpoint_identity import file_identity, require, RASTER_UNITS
import staged_comparison_support as s
import staged_comparison_checkpoint as cp
import staged_comparison_storage as storage
from staged_comparison_results import reference


@contextmanager
def native(value):
    cp.provenance(value)
    require(not any(n in sys.modules for n in ('train','scene','gaussian_renderer','simple_knn._C')),
            'cold_native_required')
    import torch
    import torch.utils.cpp_extension as ext
    import formal_native_bootstrap as bootstrap
    origin=value['native_origin']
    bootstrap._load_knn(Path(origin['knn']['binary']['path']))
    loads=[]
    def exact(*a,**kw):
        expected=dict(name='diff_gaussian_rasterization',sources=[str(s.ROOT/u) for u in RASTER_UNITS],
            extra_cuda_cflags=['-I '+str(s.ROOT/'diff-gaussian-rasterization/third_party/glm')+'/', '-g'],verbose=True)
        require(not a and kw==expected and not loads,'comparison_native_request')
        loads.append(1)
        spec=importlib.util.spec_from_file_location('diff_gaussian_rasterization',origin['raster']['binary']['path'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        sys.modules[spec.name]=module;return module
    # Keep the guard throughout the child; no other JIT request can build.
    with patch.object(ext,'load',exact):
        import gaussian_renderer
        import train
        require(loads==[1],'comparison_native_load_count');cp.provenance(value,live=True)
        yield train,gaussian_renderer
        cp.provenance(value,live=True)


def worker(candidate, stage, parent):
    import torch
    from comparison_profile import stage_config
    from comparison_inputs import acquire
    from staged_comparison_state import Initializer
    from staged_comparison_runtime import ComparisonClaim,begin_output
    from staged_comparison_observation import observation_class,schedule
    run=Path(stage['run']);config=stage_config(stage['name'],stage['output'])
    cp.parent_binding(parent,config.stage)
    n=stage['initial_points'];res=s.resources()
    res.memory(run,'worker_load',n,initial=config.stage=='stabilization')
    res.disk(run,'stage_initial' if config.stage=='stabilization' else 'stage_capture',n)
    if config.stage=='stabilization':
        initial=torch.load(parent['path'],map_location='cpu',weights_only=True)
        require(file_identity(parent['path'])==parent['identity'],'initial_changed')
        initializer=Initializer.from_initial(initial);del initial
    else:initializer=Initializer.from_checkpoint(parent,config)
    inherited=initializer.parameters
    bundle=acquire(config.stage,stage['output'],initializer)
    with native(candidate['provenance']) as (train,renderer):
        prepared=train.prepare_comparison_runtime(bundle)
        require(initializer.consumed and initializer.parameters==(),'initial_storage_released')
        require(not prepared.gaussians.optimizer.state,'fresh_adam')
        from formal_checkpoint_identity import GROUPS
        from staged_comparison_state import SLOTS
        for tensor,slot in zip(inherited,SLOTS):
            attr=next(attr for index,attr in GROUPS.values() if index==slot)
            s.memory_helper().equal_live(tensor,getattr(prepared.gaussians,attr))
        require((prepared.gaussians.active_sh_degree,prepared.gaussians.active_sh_degree_t)==initializer.active,
                'initial_active_exact')
        del inherited  # No CPU parent state retained into observations/training.
        prepared.gaussians.env_map=None
        initial_bytes=len(bundle.ply_bytes)+len(json.dumps(prepared.scene._initial_cameras).encode('utf-8'))
        storage.check(Path(stage['output']).parent,'initial',len(prepared.gaussians._xyz),
                      initial_bytes=initial_bytes)
        with closing(ComparisonClaim(bundle)) as claim:
            writer=begin_output(prepared,claim,candidate['provenance'],parent)
            def resource_check(k, boundary):
                points=len(prepared.gaussians._xyz)
                kind={'before_save':'save','before_evaluation':'observe',
                      'topology':'topology'}.get(boundary,'continue')
                res.memory(run,kind,points)
            obs=observation_class()(prepared,renderer.render,run,schedule(config),
                candidate['representatives'],[],[0,*config.save_updates],
                resource_check=resource_check,diagnostic_updates=None)
            try:
                obs.run(train.continue_comparison_training,writer)
            finally:
                if writer is not None:writer.flush();writer.close()
            require(obs.trace['returned']==config.optimization.total_updates==obs.optimizer_calls,'comparison_updates')
            final=Path(stage['output'])/'checkpoints'/('completed-'+str(config.optimization.total_updates))
            result=dict(status='passed',stage=config.stage,parent=parent,completed=obs.trace['returned'],
                optimizer_calls=obs.optimizer_calls,checkpoint=reference(final/'complete.json'),
                expected=reference(run/'expected.pt'),final_points=len(prepared.gaussians._xyz),
                trace=obs.trace,spec=obs.spec,consumed=obs.consumed.finish(),provenance=candidate['provenance'])
            s.write_json(run/'worker-result.json',result)


def reader(candidate,stage,parent):
    from staged_comparison_observation import expected_equal
    from staged_comparison_state import restore_diagnostic
    run=Path(stage['run']);w=cp._read_record(run/'worker-result.json')
    require(w['status']=='passed' and w['stage']==stage['name'] and w['parent']==parent,'reader_binding')
    s.resources().memory(run,'reader_load',w['final_points'])
    loaded=cp.read_checkpoint(Path(w['checkpoint']['path']).parent)
    require(loaded['marker_identity']==w['checkpoint']['identity'],'reader_marker')
    import torch
    saved=torch.load(w['expected']['path'],map_location='cpu',weights_only=True)
    require(file_identity(w['expected']['path'])==w['expected']['identity'],'expected_identity')
    expected_equal((saved['attrs'],saved['groups']),loaded)
    # Raw views survive model restore; independent full expected does not.
    raw=saved.pop('views');del saved
    with native(candidate['provenance']) as (_,renderer):
        from checkpoint_cuda_acceptance import diagnostic_camera,camera_values
        from real_input_cuda_acceptance import render_buffers
        with patch.object(torch.optim.Adam,'__init__',side_effect=AssertionError('reader_optimizer_forbidden')):
            model=restore_diagnostic(loaded,'cuda');config=loaded['config'];del loaded
            require(model.optimizer is None and len(raw)==len(candidate['representatives'])==14,'reader_model')
            for expected,rep in zip(raw,candidate['representatives']):
                s.resources().memory(run,'reader_view',len(model._xyz))
                cam=diagnostic_camera(config,tuple(rep[:3]))
                require(camera_values(cam)==expected['camera'],'reader_camera')
                actual=render_buffers(renderer.render,cam,model,config)
                s.memory_helper().equal_live(expected['output'],actual)
                del actual,cam
    s.write_json(run/'reader-result.json',dict(status='passed',stage=stage['name'],parent=parent,
        worker=reference(run/'worker-result.json'),checkpoint=w['checkpoint'],expected=w['expected'],
        independent_state=True,raw14_exact=True,forwards=14,optimizer_created=False,builds=0))
