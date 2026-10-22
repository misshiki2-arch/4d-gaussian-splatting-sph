"""Locator-only selected B exporter. Preparation does not authorize execution."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import staged_comparison_adopted_input as inputs
import adopted_reference_bundle as bundle
import staged_comparison_checkpoint as cp
import staged_comparison_support as support
from formal_checkpoint_identity import require


def parent(value):
    inputs.verify(value,authorized=True);inputs.absent_output(value['output'])
    root=Path(value['output'])
    # The audit root can exist on failure. The producer opens its payload area
    # only after native preparation; no bundle/receipt exists before validation.
    claim=bundle.Claim(root)
    try:
        bundle.write_json(root/'inputs.json',value)
        for mode in ('worker','reader'):
            inputs.verify(value,authorized=True);claim.check()
            command=[support.PYTHON,'-I','-B',str(ROOT/'tools/export_adopted_reference.py'),mode,str(root/'inputs.json')]
            support.monitor().monitored_child(command,cwd=str(ROOT),env=dict(os.environ),out=root,
                mode=mode,filesystem=root,interval=support.RESOURCES['poll_seconds'])
            support.process_record(root,mode)
        require(not (root/'failure.json').exists(),'reference_failed')
        inputs.verify(value,authorized=True)
        pending=root/'.pending-bundle'
        manifest=bundle.read_json(pending/'manifest.json');verification=bundle.read_json(pending/'verification.json')
        receipt=bundle.publish(root,manifest,verification)
        bundle.write_json(root/'result.json',dict(status='verified_requires_review',receipt=receipt,
            worker_exit=support.file_identity(root/'worker-exit.json'),
            reader_exit=support.file_identity(root/'reader-exit.json'),gate_c_passed=False))
    except BaseException as error:
        if not (root/'failure.json').exists():bundle.failure(root,'parent',error)
        raise
    finally:claim.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','run','worker','reader'))
    parser.add_argument('input',nargs='?')
    args=parser.parse_args(argv)
    if args.operation=='prepare':
        require(args.input is None,'prepare_has_no_override')
        import json
        print(json.dumps(inputs.plan(),ensure_ascii=False,sort_keys=True));return
    require(args.input is not None,'input_required')
    value=cp._read_record(args.input);inputs.verify(value,authorized=True)
    if args.operation=='run':return parent(value)
    root=Path(value['output'])
    require(Path(args.input)==root/'inputs.json' and not (root/'failure.json').exists(),
            'reference_child_binding')
    from adopted_reference_runtime import produce,independent
    return (produce if args.operation=='worker' else independent)(value)


if __name__=='__main__':main()
