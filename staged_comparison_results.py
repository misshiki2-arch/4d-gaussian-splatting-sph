"""Persisted comparison-final proof; no Torch import before M verification.

A final is inherited only through a published receipt tied to real child exits,
reader/state/raw checks, expected bytes, checkpoint and comparison publication.
"""
from pathlib import Path
from formal_checkpoint_identity import file_identity, keys, require
from comparison_profile import STAGES, stage_config, DOMAIN
from staged_comparison_checkpoint import _read_record
from staged_comparison_support import write_json, process_record

SCHEMA='step18-confirmed-final-v1'


def reference(path):
    return dict(path=str(path),identity=file_identity(path))


def record(ref):
    keys(ref,'path identity')
    return _read_record(ref['path'],ref['identity'])


def checkpoint_marker(ref, stage):
    marker=record(ref); keys(marker,'schema k files')
    n=stage_config(stage,'/unused').optimization.total_updates
    require(marker['schema']==DOMAIN and type(marker['k']) is int and marker['k']==n,
            'comparison_parent_final')
    path=Path(ref['path']);require(path.name=='complete.json' and
        path.parent.name=='completed-'+str(n) and path.parent.parent.name=='checkpoints','final_path')
    keys(marker['files'],'metadata.json state.pt')
    for name,expected in marker['files'].items():
        require(file_identity(path.parent/name)==expected,'component_hash')
    metadata=_read_record(path.parent/'metadata.json',marker['files']['metadata.json'])
    require(metadata['final'] is True and metadata['k']==metadata['n']==n and
        metadata['base']['profile']['stage']==stage,'comparison_parent_final')
    return metadata


def children(run):
    run=Path(run)
    require(not (run/'failure.json').exists(),'stage_failed')
    for mode in ('worker','reader'):process_record(run,mode)
    w,r=[_read_record(run/(mode+'-result.json')) for mode in ('worker','reader')]
    n=stage_config(w['stage'],'/unused').optimization.total_updates
    require(w['status']==r['status']=='passed' and w['stage']==r['stage'] and
        w['parent']==r['parent'] and w['completed']==n and w['optimizer_calls']==n,
        'comparison_child_binding')
    require(r['worker']==reference(run/'worker-result.json') and
        r['checkpoint']==w['checkpoint'] and r['expected']==w['expected'] and
        r['independent_state'] is True and r['raw14_exact'] is True and
        r['forwards']==14 and r['optimizer_created'] is False and r['builds']==0,
        'comparison_reader_proof')
    meta=checkpoint_marker(w['checkpoint'],w['stage'])
    require(meta['base']['parent']==w['parent'],'comparison_parent_binding')
    require(file_identity(w['expected']['path'])==w['expected']['identity'],'expected_identity')
    return w,r


def verify_child(run):
    w,r=children(run)
    value=dict(schema=SCHEMA,stage=w['stage'],checkpoint=w['checkpoint'],
        parent=w['parent'],expected=w['expected'],
        worker=reference(Path(run)/'worker-result.json'),reader=reference(Path(run)/'reader-result.json'),
        independent_state=True,raw14_exact=True,status='passed')
    write_json(Path(run)/'verify-result.json',value)
    return value


def verify_publication(run, publication=None):
    run=Path(run);pub=Path(publication) if publication is not None else run/'completion'
    w,r=children(run);process_record(run,'verify')
    v=_read_record(run/'verify-result.json')
    require(v==dict(schema=SCHEMA,stage=w['stage'],checkpoint=w['checkpoint'],
        parent=w['parent'],expected=w['expected'],worker=reference(run/'worker-result.json'),
        reader=reference(run/'reader-result.json'),independent_state=True,raw14_exact=True,status='passed'),
        'comparison_verify_binding')
    receipt=_read_record(pub/'receipt.json')
    require(receipt==dict(result=file_identity(pub/'result.json')),'comparison_completion_receipt')
    value=_read_record(pub/'result.json')
    require(value==dict(schema=SCHEMA,stage=w['stage'],completed=w['completed'],parent=w['parent'],
        checkpoint=w['checkpoint'],verification=reference(run/'verify-result.json'),
        evidence={name:reference(run/name) for name in ('worker-result.json','reader-result.json',
            'worker-exit.json','reader-exit.json','verify-exit.json')},
        comparison=file_identity(pub/'comparison/comparison.json'),
        status='checks_passed_requires_quality_review',step18_accepted=False,gate_b_passed=False),
        'comparison_publication')
    compare=_read_record(pub/'comparison/comparison.json')
    require(compare['stage']==w['stage'] and compare['status']=='recorded_requires_image_review',
            'comparison_result_stage')
    actual={str(p.relative_to(pub/'comparison')) for p in (pub/'comparison').rglob('*') if p.is_file()}
    require(actual=={'comparison.json',*compare['pictures']},'comparison_picture_members')
    for name,expected in compare['pictures'].items():
        p=pub/'comparison'/name
        require(p.resolve().is_relative_to((pub/'comparison').resolve()) and
                file_identity(p)==expected,'comparison_picture_identity')
    require({p.name for p in pub.iterdir()}=={'result.json','receipt.json','comparison'},'completion_members')
    return value


def finalize(run, compare):
    run=Path(run);w,r=children(run);process_record(run,'verify')
    import staged_comparison_storage as storage
    # All comparison images are new in private publication. Rename publishes
    # those same bytes, not a second full copy. Existing checkpoints stay out.
    rows=checkpoint_marker(w['checkpoint'],w['stage'])['rows']
    storage.check(run,'comparison',rows)
    pending=run/'.pending-result';published=run/'completion'
    require(not pending.exists() and not published.exists(),'exclusive_completion')
    pending.mkdir()
    compare(w,pending/'comparison')
    storage.check(run,'record',rows)  # Same filesystem; journal is not a completion member.
    value=dict(schema=SCHEMA,stage=w['stage'],completed=w['completed'],parent=w['parent'],
        checkpoint=w['checkpoint'],verification=reference(run/'verify-result.json'),
        evidence={name:reference(run/name) for name in ('worker-result.json','reader-result.json',
            'worker-exit.json','reader-exit.json','verify-exit.json')},
        comparison=file_identity(pending/'comparison/comparison.json'),
        status='checks_passed_requires_quality_review',step18_accepted=False,gate_b_passed=False)
    write_json(pending/'result.json',value)
    write_json(pending/'receipt.json',dict(result=file_identity(pending/'result.json')))
    require(verify_publication(run,pending)==value,'private_publication_readback')
    pending.rename(published)
    return reference(published/'result.json')


def parent_from_receipt(ref, next_stage):
    value=record(ref);path=Path(ref['path'])
    require(path.name=='result.json' and path.parent.name=='completion','published_parent_required')
    require(value==verify_publication(path.parent.parent),'comparison_parent_receipt')
    i=STAGES.index(next_stage)
    require(i>0 and value['stage']==STAGES[i-1],'comparison_parent_stage')
    ck=value['checkpoint']
    return dict(kind='comparison_final',path=ck['path'],identity=ck['identity'],stage=value['stage'],
        completed=value['completed'],verification=ref)


def read_confirmed(parent, next_config):
    """Verify immutable persisted chain before invoking the shared M/T reader."""
    from staged_comparison_checkpoint import read_checkpoint
    require(parent==parent_from_receipt(parent['verification'],next_config.stage),'comparison_parent_receipt')
    result=read_checkpoint(Path(parent['path']).parent)
    require(result['marker_identity']==parent['identity'],'comparison_parent_identity')
    return result
