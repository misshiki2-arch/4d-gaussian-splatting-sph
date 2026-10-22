"""Export/diagnostic-only orchestration, imported after the lightweight gate.

No training, optimizer, build fallback, checkpoint search or alternate renderer.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

import adopted_reference_bundle as bundle
import staged_comparison_adopted_input as inputs
import staged_comparison_checkpoint as cp
import staged_comparison_support as support
from formal_checkpoint_identity import (require, keys, file_identity, canonical,
    open_directory, checked_file, stream_identity)


def test_paths():
    for p in (support.ROOT/'tests',support.ROOT/'tests/cuda'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))


def incremental_demand(kind,n,**kw):
    require(kind in ('cpu_load','load','render'),'reference_memory_boundary')
    if kind=='cpu_load':
        d=support.demand('reader_load',n)
        d.update(gpu_payload=None,gpu_margin=None,gpu_needed=None)
    else:
        # Checkpoint + expected are already resident. Only chunk/frame scratch
        # is additional on host. The GPU has no restored model at 'load'; retain
        # the existing restore copies estimate there, but not at each view.
        d=support.demand('reader_view',n)
        if kind=='load':
            gpu=support.demand('reader_load',n)
            d.update({k:gpu[k] for k in ('gpu_payload','gpu_margin','gpu_needed')})
    return dict(d,boundary=kind)


def remaining_disk(kind,n,records_written=0):
    support.payload(n)  # Same positive point-count validation.
    record=support.RESOURCES['record_allowance_bytes']
    require(type(records_written) is int and 0<=records_written<=record,'reference_record_budget')
    if kind=='bundle':
        require(records_written==0,'reference_initial_records')
        population=128+260*n;views=14
    elif kind in tuple('view-'+str(i) for i in range(14)):
        population=0;views=14-int(kind.split('-')[1])
    else:
        require(kind=='records','reference_disk_boundary')
        population=views=0
    wh=1280*720;png=4*(3*wh+720+(3*wh+720+99)//100+65536)
    # Raw six NPYs: RGB/GT/alpha/depth + int radii + bool visibility.
    # Include header allowance; exclusive streaming writes have no duplicate
    # payload temporary. rename publication also does not duplicate the bundle.
    return population+views*(32*wh+5*n+png+6*256)+record-records_written


def resource_adapter():
    """Reuse exact existing margin/reserve predicates and failure journals."""
    existing=support.resources()
    def clone(fn,**overrides):
        scope=dict(fn.__globals__,**overrides)
        result=FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__)
        result.__kwdefaults__=fn.__kwdefaults__;return result
    def disk(directory,kind,n):
        pending=Path(directory)/'.pending-bundle'
        paths=[pending/'population.json',pending/'manifest.json',pending/'verification.json',pending/'index.json']
        paths.extend(pending/'references'/f'{i:02d}'/'runtime.json' for i in range(14))
        written=0
        for path in paths:
            require(not path.is_symlink(),'reference_record_symlink')
            if path.exists():
                require(path.is_file(),'reference_record_type');written+=path.stat().st_size
        return clone(existing.disk,disk_need=lambda k,rows:remaining_disk(k,rows,written))(directory,kind,n)
    return SimpleNamespace(memory=clone(existing.memory,demand=incremental_demand),disk=disk)


def load(proof):
    selected=inputs.checked_proof(proof)
    loaded=cp.read_adopted_checkpoint(proof)
    import torch
    from staged_comparison_observation import expected_equal
    ref=selected['expected'];path=Path(ref['path']);fd=open_directory(str(path.parent))
    try:
        stream,before=checked_file(fd,path.name,ref['identity']['size'])
        with stream:
            require(stream_identity(stream,before)==ref['identity'],'adopted_expected_identity')
            saved=torch.load(stream,map_location='cpu',weights_only=True)
            require(stream_identity(stream,before)==ref['identity'],'adopted_expected_changed')
    finally:os.close(fd)
    keys(saved,'attrs groups views')
    expected_equal((saved['attrs'],saved['groups']),loaded)
    views=saved.pop('views');del saved
    require(len(views)==len(selected['representatives'])==14,'adopted_views')
    return loaded,views


def tensor_description(tensor):
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),stride=list(tensor.stride()),
                device=str(tensor.device))


def invocation(args,camera,n):
    """Bounded scalars/matrices only, no retained population tensor references."""
    require(len(args)==35,'reference_native_arity')
    a=args
    require(tuple(a[1].shape)==(n,3) and tuple(a[19].shape)==(n,16,3) and
            a[2].numel()==a[11].numel()==0,'reference_population')
    expected={10:1.0,12:-1.0,15:camera.formal_camera.tanx,16:camera.formal_camera.tany,
        17:camera.image_height,18:camera.image_width,20:0,21:0,23:camera.timestamp,24:33.0,
        25:True,26:4,27:False,28:False,29:False,30:-1,31:-1,32:0,33:-1,34:True}
    require(all(type(a[i]) is type(v) and a[i]==v for i,v in expected.items()),'reference_effective_settings')
    require(tuple(a[0].shape)==(3,) and not bool(a[0].any()),'reference_background')
    for i,attr in ((13,'world_view_transform'),(14,'full_proj_transform'),(22,'camera_center')):
        require(a[i] is getattr(camera,attr),'reference_camera_arguments')
    tensors={str(i):tensor_description(a[i]) for i in (0,1,2,3,4,5,6,7,8,9,11,13,14,19,22)}
    return dict(call_count=1,population_rows=n,scalars={str(i):v for i,v in expected.items()},
        tensors=tensors,matrices={str(i):a[i].detach().cpu().tolist() for i in (0,13,14,22)})


@contextmanager
def observe_native(native,camera,n):
    original=native.rasterize_gaussians;records=[]
    def observed(*args):
        require(not records,'reference_multiple_native_calls')
        record=invocation(args,camera,n);records.append(record)
        result=original(*args)  # Exactly once; exceptions and return identity pass through.
        require(isinstance(result,tuple) and len(result)==13 and
                type(result[0]) is int and result[0]>=0,'reference_native_result')
        record.update(tile_references=result[0],buffer_bytes={name:result[i].numel()*result[i].element_size()
            for name,i in (('geometry',6),('binning',7),('image',8))})
        return result
    native.rasterize_gaussians=observed
    try:
        yield records
        require(len(records)==1 and 'tile_references' in records[0],'reference_missing_native_call')
    finally:native.rasterize_gaussians=original


def camera_record(camera):
    from checkpoint_cuda_acceptance import camera_values
    return dict(values=camera_values(camera),effective=cp.contract(camera.formal_camera),
        frame=cp.contract(camera.formal_frame),matrix_layout='row-major storage of transposed transforms',
        time_divisor=1,time_duration=33.0)


def get_view(c,rep,consumed,saved_worker):
    from checkpoint_cuda_acceptance import diagnostic_camera
    from utils.data_utils import CameraDataset
    from utils.formal_camera_runtime import camera_binding
    cam=diagnostic_camera(c,tuple(rep[:3]));camera_binding(cam,c.dataset.resolution)
    require(list((cam.formal_frame.split,cam.formal_frame.index,cam.formal_frame.file_path))==rep[:3] and
            cam.formal_frame.raw_time==rep[3],'reference_frame')
    paths=(cam.image_path,cam.image_path.replace('/images/','/masks/'))
    for p in paths:
        consumed.before(p)
        require(consumed.identities[str(p)]==saved_worker['consumed'][str(p)],'reference_historical_image')
    dataset=CameraDataset([cam],False,formal_resolution=c.dataset.resolution)
    gt,returned=consumed.getitem(CameraDataset.__getitem__,dataset,0)
    require(returned is cam,'reference_camera_owner')
    return cam,gt


def array(value):
    return value.detach().cpu().numpy()


def independent_gt(c,rep,identities):
    """Read original RGBA/mask through the real CPU consumer, without Scene.

    This image-only descriptor cannot supply cached image pixels. Camera/time
    identity is checked separately from metadata by the independent reader.
    """
    import numpy as np
    import torch
    from PIL import Image
    # general_utils imports pointops at module scope. Reuse its unchanged
    # PILtoTorch and CameraDataset definitions through the established scoped
    # definitions loader, without loading unused GPU imports in this reader.
    scope=dict(torch=torch,np=np,Image=Image,os=os,Dataset=torch.utils.data.Dataset)
    support.definitions(support.ROOT/'utils/general_utils.py',scope)
    support.definitions(support.ROOT/'utils/data_utils.py',scope)
    CameraDataset=scope['CameraDataset']
    image=str(Path(c.paths.source_path)/rep[2]);mask=image.replace('/images/','/masks/')
    require(set(identities)=={image,mask} and image!=mask,'reference_gt_paths')
    def check():
        require(all(file_identity(n)==v for n,v in identities.items()),'reference_gt_input_changed')
    check()
    r=c.dataset.resolution
    descriptor=SimpleNamespace(image_path=image,meta_only=True,resolution=(r.width,r.height),
                               image_width=r.width,image_height=r.height)
    dataset=CameraDataset([descriptor],False,formal_resolution=r)
    gt,returned=dataset[0]
    require(returned is descriptor,'reference_gt_owner')
    check()
    return gt


def exact_arrays(actual,expected):
    import numpy as np
    require(set(actual)==set(expected),'reference_raw_fields')
    for name,a in actual.items():
        b=expected[name]
        av=array(a) if hasattr(a,'detach') else a
        bv=array(b) if hasattr(b,'detach') else b
        require(av.shape==bv.shape and av.dtype==bv.dtype and av.tobytes()==bv.tobytes(),
                'reference_raw_exact:'+name)


def raw_contract(raw,n,h,w):
    import numpy as np
    require(set(raw)==set(bundle.RAW),'reference_saved_fields')
    for name,a in raw.items():
        shape=(n,) if name in ('radii','visibility_filter') else (3 if name in ('render','gt') else 1,h,w)
        dtype=np.dtype('int32' if name=='radii' else 'bool' if name=='visibility_filter' else 'float32')
        require(a.shape==shape and a.dtype==dtype and np.isfinite(a).all(),'reference_raw_shape')


def save_view(root,raw,record):
    import numpy as np
    import torch
    from pilot_runtime import save_png
    from utils.image_utils import easy_cmap
    root=Path(root);root.mkdir()
    raw_contract({k:array(v) for k,v in raw.items()},record['native']['population_rows'],
                 record['camera']['values']['height'],record['camera']['values']['width'])
    for name,value in raw.items():
        bundle.write(root/(name+'.npy'),lambda f,v=value:np.save(f,array(v),allow_pickle=False))
    for name,value in (('render',raw['render']),('gt',raw['gt']),
                       ('alpha',raw['alpha'].repeat(3,1,1)),('depth',easy_cmap(raw['depth'][0]))):
        save_png(root/(name+'.png'),value)
    bundle.write_json(root/'runtime.json',record)


def check_png(root,raw,old,rep):
    """Independent NumPy quantization, not the producer's PNG helper."""
    import numpy as np
    from PIL import Image
    depth=raw['depth'][0];lo,hi=depth.min(),depth.max()
    normalized=np.zeros_like(depth) if hi==lo else (depth-lo)/(hi-lo)
    pictures=dict(render=raw['render'],gt=raw['gt'],alpha=np.repeat(raw['alpha'],3,axis=0),
                  depth=np.repeat(normalized[None],3,axis=0))
    old_row=next(v for v in old['trace']['fixed'][-1]['views'] if v['frame']==rep[:3])
    for name,value in pictures.items():
        expected=(np.clip(value,0,1).transpose(1,2,0)*255).astype(np.uint8)
        with Image.open(Path(root)/(name+'.png')) as image:
            require(image.mode=='RGB' and np.array_equal(np.asarray(image),expected),'reference_png_quantization')
        previous=inputs.RUN/'fixed/12000'/(str(rep[1])+'-'+name+'.png')
        require(file_identity(previous)==old_row['images'][name],'reference_prior_png_identity')
        with Image.open(previous) as image:
            require(image.mode=='RGB' and np.array_equal(np.asarray(image),expected),'reference_prior_pixels')


def manifest(value,meta,population,references,environment):
    return dict(schema=bundle.SCHEMA,version=1,
        input=dict(selected=value['selected'],profile=meta['base']['profile'],dataset=meta['base']['dataset'],
                   stage_updates=12000,cumulative_updates=66000),
        provenance=dict(bridge=value['bridge'],runtime_environment=environment),population=population,
        references=references,encoding=bundle.ENCODING,limitations=bundle.LIMITATIONS,
        verification='verification.json')


def produce(value):
    proof=inputs.verify(value,authorized=True);inputs.absent_output(Path(value['output'])/'.pending-bundle')
    test_paths();root=Path(value['output']);stage='load';res=resource_adapter()
    try:
        res.memory(root,'cpu_load',3248901);res.disk(root,'bundle',3248901)
        loaded,expected=load(proof)
        import torch
        from PIL import __version__ as pillow_version
        import numpy as np
        from staged_comparison_worker import native
        from staged_comparison_state import restore_diagnostic
        from real_input_runtime import ConsumedFiles
        from checkpoint_cuda_acceptance import camera_values
        from real_input_cuda_acceptance import render_buffers
        from unittest.mock import patch
        origin=value['bridge']['current_provenance']
        require(origin['environment']==dict(executable=sys.executable,python=sys.version,
            torch=str(torch.__version__),torch_cuda=str(torch.version.cuda)),'reference_environment')
        environment=dict(origin['environment'],numpy=np.__version__,pillow=pillow_version)
        stage='native'
        with native(origin) as (_,renderer):
            res.memory(root,'load',3248901)
            pending=root/'.pending-bundle';pending.mkdir();(pending/'references').mkdir()
            stage='export'
            module=bundle.serializer(inputs.SERIALIZER,value['bridge']['current_files'][str(inputs.SERIALIZER)])
            population=bundle.export_spl4(pending/'population.splat4d',loaded['state']['capture'],module)
            bundle.write_json(pending/'population.json',population)
            meta,c=loaded['metadata'],loaded['config']
            with patch.object(torch.optim.Adam,'__init__',side_effect=AssertionError('reference_optimizer_forbidden')):
                model=restore_diagnostic(loaded,'cuda')
            require(model.optimizer is None,'reference_no_optimizer');del loaded
            old=cp._read_record(inputs.RUN/'worker-result.json');consumed=ConsumedFiles();references=[]
            for i,(rep,wanted) in enumerate(zip(value['selected']['representatives'],expected)):
                stage='render:'+str(i);res.memory(root,'render',len(model._xyz))
                cam,gt=get_view(c,rep,consumed,old)
                require(camera_values(cam)==wanted['camera'],'reference_saved_camera')
                native_module=renderer.diff_gaussian_rasterization._C
                observations=[]
                def observed_render(device_cam,*args,**kw):
                    # The existing helper makes a bound GPU camera copy. Observe
                    # that exact owner, never the original CPU camera's tensors.
                    with observe_native(native_module,device_cam,len(model._xyz)) as records:
                        result=renderer.render(device_cam,*args,**kw)
                    observations.extend(records)
                    return result
                raw=render_buffers(observed_render,cam,model,c)
                require(len(observations)==1,'reference_forward_count')
                exact_arrays(raw,wanted['output']);raw['gt']=gt
                record=dict(schema='adopted-view-runtime-v1',invocation=i,frame=rep,camera=camera_record(cam),
                    native=observations[0],depth_min=float(raw['depth'].min()),depth_max=float(raw['depth'].max()),
                    files={str(p):consumed.identities[str(p)] for p in
                        (cam.image_path,cam.image_path.replace('/images/','/masks/'))})
                res.disk(root,'view-'+str(i),len(model._xyz))
                save_view(pending/'references'/f'{i:02d}',raw,record)
                check_png(pending/'references'/f'{i:02d}',{k:array(v) for k,v in raw.items()},old,rep)
                references.append(record);del raw,cam,gt
            consumed.finish();del model,expected
        inputs.verify(value)
        res.disk(root,'records',meta['rows'])
        bundle.write_json(pending/'manifest.json',manifest(value,meta,population,references,environment))
        bundle.write_json(root/'producer-result.json',dict(status='produced_requires_independent_check',
            forwards=14,optimizer_created=False,builds=0,gate_c_passed=False))
    except BaseException as error:
        if not (root/'failure.json').exists():bundle.failure(root,stage,error)
        raise


def independent(value):
    """New CPU process: no renderer/native load, no second GPU forward."""
    proof=inputs.verify(value,authorized=True);test_paths();root=Path(value['output']);pending=root/'.pending-bundle'
    stage='independent_load'
    try:
        resource_adapter().memory(root,'cpu_load',3248901)
        loaded,expected=load(proof);c=loaded['config'];meta=loaded['metadata']
        population=bundle.check_spl4(pending/'population.splat4d',loaded['state']['capture'])
        require(bundle.read_json(pending/'population.json')==population,'population_record');del loaded
        import numpy as np
        import torch
        from PIL import Image,__version__ as pillow_version
        from formal_inputs import Frame
        import formal_camera
        old=cp._read_record(inputs.RUN/'worker-result.json');refs=[]
        for i,(rep,wanted) in enumerate(zip(value['selected']['representatives'],expected)):
            stage='independent_view:'+str(i);p=pending/'references'/f'{i:02d}'
            raw={n:np.load(p/(n+'.npy'),allow_pickle=False) for n in bundle.RAW}
            raw_contract(raw,meta['rows'],c.dataset.resolution.height,c.dataset.resolution.width)
            exact_arrays({n:raw[n] for n in wanted['output']},wanted['output'])
            record=bundle.read_json(p/'runtime.json')
            keys(record,'schema invocation frame camera native depth_min depth_max files')
            require(record['schema']=='adopted-view-runtime-v1' and record['invocation']==i and record['frame']==rep and
                    record['camera']['values']==wanted['camera'],'reference_runtime_binding')
            transforms=cp._read_record(Path(c.paths.source_path)/('transforms_'+rep[0]+'.json'))
            frame=Frame(rep[0],rep[1],rep[2],float(rep[3]),float(rep[3]))
            effective=formal_camera.from_metadata(transforms,transforms['frames'][rep[1]],c.dataset.resolution,tuple(rep[:3]))
            require(record['camera']==dict(values=wanted['camera'],effective=cp.contract(effective),frame=cp.contract(frame),
                matrix_layout='row-major storage of transposed transforms',time_divisor=1,time_duration=33.0),
                'reference_camera_record')
            require(record['depth_min']==float(raw['depth'].min()) and record['depth_max']==float(raw['depth'].max()),
                    'reference_depth_record')
            expected_paths=[str(Path(c.paths.source_path)/rep[2]),str(Path(c.paths.source_path)/rep[2]).replace('/images/','/masks/')]
            require(record['files']=={n:old['consumed'][n] for n in expected_paths} and
                    all(file_identity(n)==v for n,v in record['files'].items()),'reference_image_record')
            gt=independent_gt(c,rep,record['files'])
            exact_arrays({'gt':raw['gt']},{'gt':gt});del gt
            check_invocation_record(record['native'],wanted['camera'],effective,meta['rows'])
            check_png(p,raw,old,rep);refs.append(record);del raw
        env=dict(value['bridge']['current_provenance']['environment'],numpy=np.__version__,pillow=pillow_version)
        expected_manifest=manifest(value,meta,population,refs,env)
        bundle.check_manifest(bundle.read_json(pending/'manifest.json'),expected_manifest)
        inputs.verify(value)
        verification=dict(schema='adopted-reference-verification-v1',status='verified_requires_review',
            selected=value['selected'],all_rows=True,raw14_exact=True,png14_exact=True,
            gt14_raw_exact=True,separate_cpu_process=True,gate_c_passed=False)
        resource_adapter().disk(root,'records',meta['rows'])
        bundle.write_json(pending/'verification.json',verification)
        bundle.index_bundle(pending,expected_manifest)
    except BaseException as error:
        if not (root/'failure.json').exists():bundle.failure(root,stage,error)
        raise


def check_invocation_record(row,camera,effective,n):
    keys(row,'call_count population_rows scalars tensors matrices tile_references buffer_bytes')
    expected={10:1.0,12:-1.0,15:effective.tanx,16:effective.tany,17:camera['height'],18:camera['width'],
        20:0,21:0,23:camera['timestamp'],24:33.0,25:True,26:4,27:False,28:False,29:False,
        30:-1,31:-1,32:0,33:-1,34:True}
    require(row['call_count']==1 and row['population_rows']==n and
            canonical(row['scalars'])==canonical({str(i):v for i,v in expected.items()}) and
            row['matrices']=={'0':[0.,0.,0.],'13':camera['world_view_transform'],
                              '14':camera['full_proj_transform'],'22':camera['camera_center']} and
            type(row['tile_references']) is int and row['tile_references']>=0,'reference_invocation_record')
    shapes={0:[3],1:[n,3],2:[0],3:[n,2],4:[n,1],5:[n,1],6:[n,3],7:[n,1],8:[n,4],9:[n,4],
            11:[0],13:[4,4],14:[4,4],19:[n,16,3],22:[3]}
    require(set(row['tensors'])=={str(i) for i in shapes},'reference_tensor_members')
    for i,shape in shapes.items():
        v=row['tensors'][str(i)];keys(v,'shape dtype stride device')
        device_ok=v['device']=='cpu' if i in (2,11) else v['device'].startswith('cuda')
        require(v['shape']==shape and v['dtype']=='torch.float32' and device_ok and
                len(v['stride'])==len(shape) and all(type(s) is int and s>=0 for s in v['stride']),
                'reference_tensor_record')
    keys(row['buffer_bytes'],'geometry binning image')
    require(all(type(v) is int and v>=0 for v in row['buffer_bytes'].values()),'reference_buffer_record')
