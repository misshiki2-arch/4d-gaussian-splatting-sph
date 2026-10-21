"""Comparison domain M/T reader and shared atomic checkpoint publication.

Formal readers never accept this schema. Metadata binds current Python to the
unchanged native-origin record, not a fabricated fresh-build claim.
"""
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
import os
from pathlib import Path

import formal_checkpoint as cp
import formal_checkpoint_identity as identity
from formal_checkpoint_identity import (canonical, checked_file, keys, require,
    open_directory, strict_json, stream_identity, validate_identity, file_identity)
from comparison_profile import DOMAIN, STAGES, stage_config, validate_config, state_byte_budget

INITIAL_PATH = '/home/demo/work/data/4dgs_sph_scene/init_from_3dgs_35_200_aligned_sigmaT0p05.pth'
NATIVE_ANCHOR = identity.ROOT/'staged_comparison_native_origin.json'


def _read_record(path, expected=None):
    """Bounded, no-symlink, same-fd hash and parse; no Torch/native imports."""
    path = Path(path)
    fd = open_directory(str(path.parent))
    try:
        stream, before = checked_file(fd, path.name, 8*1024**2)
        with stream:
            found = stream_identity(stream, before)
            if expected is not None:
                validate_identity(expected)
                require(found == expected, 'comparison_native_record')
            value = strict_json(stream.read())
            require(stream_identity(stream, before) == found, 'readback_changed')
            return value
    finally:
        os.close(fd)


def _trusted_native():
    # Machine-transferred from the reviewed record identity, not supplied by
    # candidate metadata. The anchor itself belongs to current source identity.
    anchor = _read_record(NATIVE_ANCHOR)
    keys(anchor, 'path identity')
    record = _read_record(anchor['path'], anchor['identity'])
    require(type(record) is dict and type(record.get('native')) is dict,
            'comparison_native_record')
    return record['native']


def contract(value):
    if is_dataclass(value):
        return {f.name:contract(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value,tuple) and hasattr(value,'_fields'):
        return {name:contract(getattr(value,name)) for name in value._fields}
    if isinstance(value,(list,tuple)):return [contract(x) for x in value]
    if type(value) is dict:return {k:contract(v) for k,v in value.items()}
    require(type(value) in (str,int,float,bool,type(None)), 'comparison_contract_type')
    return value


def provenance(value, *, live=False):
    keys(value, 'native_origin connection_source environment')
    origin=value['native_origin']
    identity.verify_source(value['connection_source'])
    # Shared build validation obtains its root only from the fixed saved record,
    # never an unverified root supplied by this caller or candidate metadata.
    checked = identity._validate_native_build(identity.NativeEvidence(canonical(origin)),
                                             live=live, comparison=True)
    current=value['connection_source']
    prefixes = ('simple-knn/', 'diff-gaussian-rasterization/', 'gaussian_renderer/')
    # The current source inventory is independently enumerated and checked
    # above. Do not derive required membership from the candidate origin.
    native_names = {n for n in current if n.startswith(prefixes)}
    require(type(origin.get('source')) is dict and native_names and
            {n for n in origin['source'] if n.startswith(prefixes)} == native_names and
            all(current[n] == origin['source'][n] for n in native_names),
            'comparison_native_source')
    source_root = Path(checked['repository'])
    require(all(file_identity(source_root/n) == origin['source'][n] for n in native_names),
            'comparison_native_source')
    keys(value['environment'],'python executable torch torch_cuda')
    require(value['environment']==origin['environment'], 'comparison_environment')
    return value


def parent_binding(value, stage):
    keys(value,'kind path identity stage completed verification')
    validate_identity(value['identity'])
    index=STAGES.index(stage)
    require(type(value['path']) is str and Path(value['path']).is_absolute(), 'comparison_parent_path')
    if index==0:
        require(value['kind']=='initial_dict' and value['stage'] is None and
                value['completed']==0 and value['verification'] is None and
                value['path']==INITIAL_PATH,
                'comparison_initial_parent')
    else:
        require(value['kind']=='comparison_final' and value['stage']==STAGES[index-1] and
                value['completed']==stage_config(STAGES[index-1],'/unused').optimization.total_updates,
                'comparison_parent_stage')
        from staged_comparison_results import parent_from_receipt
        require(value==parent_from_receipt(value['verification'],stage), 'comparison_parent_unverified')
    require(file_identity(value['path'])==value['identity'], 'comparison_parent_identity')


@dataclass(frozen=True)
class Context:
    inputs: object
    base: bytes


def prepare_context(inputs, native, parent):
    from staged_comparison_support import execution_files
    validate_config(inputs.config);provenance(native,live=True)
    parent_binding(parent,inputs.config.stage)
    return Context(inputs,canonical(dict(profile=contract(inputs.config),dataset=cp._dataset(inputs),
        provenance=native,parent=parent,execution=execution_files())))


def validate_metadata(meta,root):
    keys(meta,'schema layout boundary k n final rows base optimizer')
    require(meta['schema']==DOMAIN and type(meta['layout']) is int and meta['layout']==1 and
            meta['boundary']=='completed-after-topology-reset','comparison_metadata_domain')
    base=meta['base'];keys(base,'profile dataset provenance parent execution')
    from staged_comparison_support import check_execution_files
    from staged_comparison_followup import historical_metadata
    historical=historical_metadata(meta,root)
    if not historical:check_execution_files(base['execution'])
    require(type(base['profile']) is dict and 'stage' in base['profile'],'comparison_profile')
    c=stage_config(base['profile']['stage'],str(root))
    require(canonical(base['profile'])==canonical(contract(c)), 'comparison_profile')
    import formal_inputs
    ref=formal_inputs._reference()
    dataset=base['dataset'];keys(dataset,'reference files frames cameras')
    require(dataset['reference']==ref and dataset['files']==ref['files'],'comparison_dataset')
    for name,expected in dataset['files'].items():
        require(file_identity(Path(c.paths.source_path)/name)==expected,'comparison_dataset_changed')
    # Frame/camera projection comes from the same independently acquired metadata.
    # Existing output is intentional for a reader. Reconstruct data without
    # invoking the new-output preflight; see the shared acquisition below.
    from comparison_inputs import _acquire_dataset
    rebound=_acquire_dataset(c,None)
    require(canonical(cp._dataset(rebound))==canonical(dataset),'comparison_dataset_projection')
    if not historical:provenance(base['provenance'])
    parent_binding(base['parent'],c.stage)
    k,n,m=meta['k'],meta['n'],meta['rows']
    require(type(k) is type(n) is type(m) is int and m>0 and n==c.optimization.total_updates and
            k in c.save_updates and type(meta['final']) is bool and meta['final']==(k==n),
            'comparison_completed')
    identity.validate_optimizer_descriptor(meta['optimizer'])
    return c


def _read_pair(fd,root,marker=None):
    with ExitStack() as stack:
        ms,mb=checked_file(fd,'metadata.json',cp.MAX_METADATA);stack.enter_context(ms)
        mi=stream_identity(ms,mb)
        if marker is not None:require(mi==marker['files']['metadata.json'],'component_hash')
        meta=strict_json(ms.read(),cp.MAX_METADATA)
        c=validate_metadata(meta,root)  # All M validation before Torch/serializer.
        ss,sb=checked_file(fd,'state.pt',state_byte_budget(meta['rows']));stack.enter_context(ss)
        si=stream_identity(ss,sb)
        if marker is not None:
            require(si==marker['files']['state.pt'] and marker['k']==meta['k'],'component_hash')
        from staged_comparison_state import load_state
        payload=load_state(ss,meta,c)
        require(stream_identity(ms,mb)==mi and stream_identity(ss,sb)==si,'readback_changed')
        return meta,payload,c,{'metadata.json':mi,'state.pt':si}


def read_checkpoint(directory):
    directory=Path(directory)
    require(directory.parent.name=='checkpoints','checkpoint_directory')
    fd=open_directory(str(directory))
    try:
        stream,before=checked_file(fd,'complete.json',4096)
        with stream:
            marker_identity=stream_identity(stream,before);marker=strict_json(stream.read(),4096)
        keys(marker,'schema k files');require(marker['schema']==DOMAIN,'comparison_marker_domain')
        keys(marker['files'],'metadata.json state.pt')
        for value in marker['files'].values():validate_identity(value)
        require(type(marker['k']) is int and directory.name=='completed-'+str(marker['k']),'directory_label')
        meta,state,c,_=_read_pair(fd,directory.parent.parent,marker)
        return dict(config=c,metadata=meta,state=state,marker_identity=marker_identity)
    finally:os.close(fd)


class CheckpointWriter:
    def __init__(self,context,claim,scene):
        require(type(context) is Context and scene._formal_inputs is context.inputs and
                claim.path==context.inputs.config.paths.output_directory,'comparison_context')
        self.context,self.claim,self.scene=context,claim,scene
        self.failed=False

    def save(self,k):
        require(not self.failed,'writer_failed');self.failed=True;self.claim.check()
        base=strict_json(self.context.base);provenance(base['provenance'],live=True)
        from staged_comparison_state import snapshot
        c=self.context.inputs.config
        state,desc=snapshot(self.scene.gaussians,c,k)
        meta=dict(schema=DOMAIN,layout=1,boundary='completed-after-topology-reset',k=k,
            n=c.optimization.total_updates,final=k==c.optimization.total_updates,
            rows=len(state['capture'][1]),base=base,optimizer=desc)
        result=cp._publish_snapshot(self.claim,meta,state,k,DOMAIN,_read_pair,read_checkpoint)
        self.failed=False
        return result
