"""One selected, confirmed stabilization final -> growth/E2/E3, never resume.

Historical metadata remains immutable. Only this selected parent can cross the
reviewed Python-monitor/connection change; formal and ordinary readers retain
their current-source checks.
"""
from pathlib import Path

from formal_checkpoint_identity import canonical, file_identity, require, source_snapshot
from comparison_profile import STAGES, stage_config
import staged_comparison_checkpoint as cp
import staged_comparison_results as results
import staged_comparison_support as s

HERE=s.REPORTS/'step18/step18-implement3'
AUTHORITY=HERE/'step18-implement3-instruction-evidence.json'
CANDIDATE=HERE/'step18-implement3-cuda-input-candidate.json'
PROGRAM=HERE/'three-stage-run'
SCHEMA='step18-selected-stabilization-followup-v1'
OLD_PROGRAM=s.HERE/'four-stage-run'
PARENT_RUN=OLD_PROGRAM/'stabilization/run'
PARENT_OUTPUT=OLD_PROGRAM/'stabilization/output'
PUBLICATION=PARENT_RUN/'completion/result.json'
MARKER=PARENT_OUTPUT/'checkpoints/completed-12000/complete.json'
BEFORE=s.REPORTS/'step18/step18-fix4/step18-fix4-before.json'
AFTER=s.REPORTS/'step18/step18-fix4/step18-fix4-after.json'
# Only reviewed monitoring/connection code can differ from the saved parent.
# Candidate binding still checks every current file's exact identity; these are
# not runtime bypasses for arbitrary changes to these files after preparation.
SOURCE_DELTA={'staged_comparison_entry.py','staged_comparison_checkpoint.py',
              'staged_comparison_support.py','staged_comparison_followup.py'}
EXECUTION_DELTA={str(s.ROOT/'staged_comparison_followup.py'),str(AUTHORITY),
                 str(s.ROOT/'tests/test_comparison_followup.py')}


def paths():
    return [dict(name=n,run=str(PROGRAM/n/'run'),output=str(PROGRAM/n/'output')) for n in STAGES[1:]]


def pinned(path, authority):
    expected=authority['references'][str(path)]
    return cp._read_record(path,dict(size=expected['bytes'],sha256=expected['sha256']))


def check_delta(old, current, allowed):
    require(set(old)<=set(current), 'followup_removed_dependency')
    changed={p for p in old.keys()|current.keys() if old.get(p)!=current.get(p)}
    require(changed<=allowed,'followup_unreviewed_difference')


def compatible(meta, authority):
    before,after=pinned(BEFORE,authority),pinned(AFTER,authority)
    base=meta['base']
    require(base['provenance']['connection_source']==before['source'] and
            base['execution']==before['execution'],'followup_historical_inventory')
    current=source_snapshot();execution=s.execution_files()
    check_delta(after['source'],current,SOURCE_DELTA)
    check_delta(after['execution'],execution,EXECUTION_DELTA)
    # Accepted Fix4 itself is pinned by the instruction, not reclassified as
    # arbitrary compatible code. Native origin and all computing/data code stay
    # under the existing strict validation.
    origin=cp._trusted_native()
    require(canonical(base['provenance']['native_origin'])==canonical(origin) and
            base['provenance']['environment']==origin['environment'],'followup_native_origin')
    cp.provenance(dict(native_origin=origin,connection_source=current,environment=origin['environment']))
    return dict(historical_source=base['provenance']['connection_source'],
                historical_execution=base['execution'],
                current_source=current,current_execution=execution,native_origin=origin)


def selected():
    authority=cp._read_record(AUTHORITY)
    value=pinned(PUBLICATION,authority)
    pinned(PARENT_RUN/'completion/receipt.json',authority)
    pinned(MARKER,authority)
    metadata=pinned(MARKER.parent/'metadata.json',authority)
    require(value['stage']=='stabilization' and value['completed']==12000 and
            value['checkpoint']==results.reference(MARKER),'followup_selected_final')
    parent=results.parent_from_receipt(results.reference(PUBLICATION),'growth')
    require(parent['path']==str(MARKER) and parent['completed']==12000 and
            metadata['final'] is True and metadata['k']==metadata['n']==12000,
            'followup_selected_final')
    bridge=compatible(metadata,authority)
    return parent,metadata,bridge


def historical_metadata(meta, root):
    if Path(root)!=PARENT_OUTPUT:
        return False
    _,saved,_=selected()
    require(canonical(meta)==canonical(saved),'followup_selected_metadata')
    return True


def plan():
    # Reuse the ordinary current-source/native/A/dataset/resource plan builder;
    # only the fixed origin, output paths and stage slice are different.
    import staged_comparison_entry as entry
    value=entry.plan()
    parent,_,bridge=selected()
    value.update(schema=SCHEMA,program=str(PROGRAM),stages=paths(),initial=parent,
        profiles=[cp.contract(stage_config(x['name'],x['output'])) for x in paths()],
        selected_publication=results.reference(PUBLICATION),origin_bridge=bridge,
        counts=dict(A_optimizer=65992,B_optimizer=66000,reused_B_optimizer=12000,new_B_optimizer=54000),
        unverified=['live capacity','exact native load','CUDA parent T read/transfer',
                    'growth/e2/e3 learning, independent readback and image quality'])
    return value


def check(value):
    parent,_,bridge=selected()
    require(value['initial']==parent and value['selected_publication']==results.reference(PUBLICATION)
            and value['origin_bridge']==bridge,'followup_selected_binding')
    require(value['provenance']['connection_source']==bridge['current_source'] and
            value['execution_files']==bridge['current_execution'],'followup_current_binding')


def previous(value, index):
    if index==0:
        check(value)
        return value['initial']
    run=Path(value['stages'][index-1]['run'])
    return results.parent_from_receipt(results.reference(run/'completion/result.json'),
                                       value['stages'][index]['name'])


def reused(value):
    check(value)
    return [value['selected_publication']]
