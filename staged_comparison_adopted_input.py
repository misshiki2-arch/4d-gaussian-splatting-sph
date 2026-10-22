"""Stdlib-only, selected E3 -> export authority. Never a generic resume bridge."""
from dataclasses import dataclass
from pathlib import Path
import os
import sys

from formal_checkpoint_identity import (ROOT, canonical, strict_json, file_identity,
    byte_identity, keys, require, source_snapshot, open_directory)
import staged_comparison_checkpoint as cp
import staged_comparison_support as support

HERE=ROOT.parent/'reports/corrected-4dgs/phase4/step20/step20-implement1'
AUTHORITY=HERE/'step20-implement1-instruction-evidence.json'
DESIGN=HERE.parent/'step20-design1/step20-design1-design-evidence.json'
COMPLETION=ROOT.parent/'reports/corrected-4dgs/phase3/step19/step19-completion-evidence.json'
RUN=ROOT.parent/'reports/corrected-4dgs/phase3/step18/step18-implement3/three-stage-run/e3_like/run'
CHECKPOINT=RUN.parent/'output/checkpoints/completed-12000'
SERIALIZER=ROOT.parent/'GaussianSplats3D/converter/export_splat4d_from_ckpt.py'
CANDIDATE=HERE.parent/'step20-fix1/step20-fix1-cuda-input-candidate.json'
OUTPUT=HERE/'reference-run'
ADDED_SOURCE='staged_comparison_adopted_input.py'
LOCAL=(ADDED_SOURCE,'adopted_reference_bundle.py','adopted_reference_runtime.py',
       'tools/export_adopted_reference.py','tests/test_adopted_reference.py',
       'tests/cuda/adopted_reference_acceptance.py')
SCHEMA='adopted-b-reference-input-v1'


def pinned(path, references):
    v=references[str(path)]
    expected=dict(size=v.get('size',v.get('bytes')),sha256=v['sha256'])
    return cp._read_record(path,expected)


def current_files():
    """Independent membership, not candidate-supplied dependency selection."""
    files={str(ROOT/n):v for n,v in source_snapshot().items()}
    files.update(support.execution_files())
    for path in [*(ROOT/n for n in LOCAL),SERIALIZER,AUTHORITY,DESIGN,COMPLETION,
                 HERE/'before/staged_comparison_checkpoint.py']:
        files[str(path)]=file_identity(path)
    return files


def selected():
    authority=cp._read_record(AUTHORITY)
    design=pinned(DESIGN,authority['references'])
    completion=pinned(COMPLETION,authority['references'])
    require(completion['user_accepted_formal_baseline']==
            'saved B E3 final with executed profile and stated use limits' and
            completion['gate_B']=='passed by explicit user adoption',
            'adopted_decision')
    check=design['input_check']
    require(check['path']==str(CHECKPOINT),'adopted_fixed_path')
    marker=cp._read_record(CHECKPOINT/'complete.json',check['marker'])
    meta=cp._read_record(CHECKPOINT/'metadata.json',check['metadata'])
    require(marker['files']['metadata.json']==check['metadata'] and meta['final'] is True and
            meta['k']==meta['n']==12000 and meta['rows']==3248901 and
            meta['base']['profile']['stage']=='e3_like','adopted_selected_final')
    refs=design['additional_read_reference_identity']
    records={n:pinned(RUN/n,refs) for n in ('worker-result.json','reader-result.json',
        'verify-result.json','completion/result.json','completion/receipt.json')}
    w,r=records['worker-result.json'],records['reader-result.json']
    require(w['status']==r['status']=='passed' and w['checkpoint']==
            dict(path=str(CHECKPOINT/'complete.json'),identity=check['marker']) and
            r['checkpoint']==w['checkpoint'] and r['expected']==w['expected'] and
            r['raw14_exact'] is True and r['independent_state'] is True and
            w['spec']['representatives']==check['representatives'],'adopted_saved_proof')
    require(records['completion/receipt.json']['result']==file_identity(RUN/'completion/result.json'),
            'adopted_receipt')
    # No huge state/expected load or hash during candidate preparation. M/T and
    # independent expected bytes are checked at the separately authorized run.
    for path,expected in ((CHECKPOINT/'state.pt',marker['files']['state.pt']),
                          (Path(w['expected']['path']),w['expected']['identity'])):
        require(not path.is_symlink() and path.is_file() and path.stat().st_size==expected['size'],
                'adopted_component_size')
    return meta,dict(checkpoint=str(CHECKPOINT),marker=check['marker'],metadata=check['metadata'],
        state=marker['files']['state.pt'],expected=w['expected'],records={str(RUN/n):file_identity(RUN/n)
            for n in records},representatives=w['spec']['representatives'],
        adoption=file_identity(COMPLETION),design=file_identity(DESIGN)),w


def bridge(meta):
    old=meta['base']['provenance']['connection_source']; now=source_snapshot()
    require(ADDED_SOURCE not in old and set(now)==set(old)|{ADDED_SOURCE},'adopted_source_members')
    preserved=HERE/'before/staged_comparison_checkpoint.py'
    require(file_identity(preserved)==old['staged_comparison_checkpoint.py'],'adopted_before_bytes')
    require(all(old[n]==now[n] for n in old if n!='staged_comparison_checkpoint.py'),
            'adopted_unreviewed_source')
    previous=meta['base']['execution']; current=support.execution_files()
    extra={str(ROOT/n) for n in LOCAL if n.startswith('tests/')}
    require(set(current)==set(previous)|extra and all(current[n]==v for n,v in previous.items()),
            'adopted_execution_delta')
    origin=cp._trusted_native()
    require(meta['base']['provenance']['native_origin']==origin,'adopted_native_origin')
    prov=dict(native_origin=origin,connection_source=now,environment=origin['environment'])
    cp.provenance(prov)
    return dict(historical_source=old,historical_execution=previous,current_provenance=prov,
                current_files=current_files())


def absent_output(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and not p.exists() and not p.is_symlink(),
            'adopted_output_exists')
    fd=open_directory(str(p.parent));os.close(fd)


def plan():
    require(sys.executable==support.PYTHON,'adopted_python')
    meta,selected_input,_=selected();b=bridge(meta)
    absent_output(OUTPUT)
    return dict(schema=SCHEMA,execution_authorized=False,executable=False,review_required=True,
        selected=selected_input,bridge=b,output=str(OUTPUT),python=support.PYTHON,
        entry=str(ROOT/'tests/cuda/adopted_reference_acceptance.py'),
        resources=support.RESOURCES,unverified=['live capacity','native load','CUDA raw14 exact',
            'full SPL4 and independent bundle verification'],gate_c_passed=False)


@dataclass(frozen=True)
class _Proof:
    data: bytes


def verify(value,*,authorized=False):
    keys(value,'schema execution_authorized executable review_required selected bridge output python entry '
               'resources unverified gate_c_passed')
    require(value['schema']==SCHEMA and value['python']==sys.executable==support.PYTHON and
            value['entry']==str(ROOT/'tests/cuda/adopted_reference_acceptance.py') and
            value['output']==str(OUTPUT) and value['review_required'] is True and
            value['gate_c_passed'] is False and value['resources']==support.RESOURCES,'adopted_contract')
    require(type(value['execution_authorized']) is type(value['executable']) is bool,'adopted_flags')
    if authorized:
        require(value['execution_authorized'] is True and value['executable'] is True,
                'separate_live_authorization_required')
        fixed=cp._read_record(CANDIDATE)
        flags=('execution_authorized','executable')
        require({k:v for k,v in value.items() if k not in flags}==
                {k:v for k,v in fixed.items() if k not in flags},'adopted_fixed_candidate')
    meta,actual,_=selected()
    require(value['selected']==actual and value['bridge']==bridge(meta),'adopted_input_changed')
    return _Proof(canonical(value))


def checked_proof(proof):
    require(type(proof) is _Proof,'adopted_proof_type')
    value=strict_json(proof.data)
    verify(value)
    return value['selected']


def check_metadata(proof,meta,root):
    selected_input=checked_proof(proof)
    require(Path(root)==CHECKPOINT.parent.parent and
            byte_identity(canonical(meta))==byte_identity(canonical(
                cp._read_record(CHECKPOINT/'metadata.json',selected_input['metadata']))),
            'adopted_metadata_binding')
