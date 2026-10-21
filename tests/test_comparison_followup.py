"""Selected-parent metadata/control and small CPU M/T fixtures, never learning."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).parent),str(Path(__file__).resolve().parents[1])]
import staged_comparison_followup as f
import staged_comparison_entry as e
import staged_comparison_support as s
import staged_comparison_checkpoint as cp
import staged_comparison_results as results
from formal_checkpoint_identity import canonical, file_identity
from test_comparison_pipeline import FinalConnectionTests
from formal_training_fixture import CPUCase


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(canonical(value))


def authority_for(paths):
    return dict(references={str(p):dict(bytes=file_identity(p)['size'],
        sha256=file_identity(p)['sha256']) for p in paths})


class FollowupControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_direct_isolated_entry_rejects_unapproved_input_before_output(self):
        candidate=self.root/'input.json'
        put(candidate,dict(schema=f.SCHEMA,execution_authorized=False,executable=False))
        command=e.child_command('worker',candidate,0,continued=True)
        result=subprocess.run(command,cwd=self.root,env=dict(os.environ),
                              capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('separate_live_authorization_required',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)
        self.assertEqual(list(self.root.iterdir()),[candidate])

    def test_historical_and_current_inventory_separation_rejects_unreviewed_delta(self):
        # Isolate inventory policy. Native verifier itself is covered by the
        # real selected-artifact check and existing native-provenance tests.
        before=self.root/'before.json';after=self.root/'after.json'
        old={'train.py':{'v':1},'staged_comparison_support.py':{'v':1}}
        accepted=copy.deepcopy(old);accepted['staged_comparison_support.py']={'v':2}
        current=copy.deepcopy(accepted);current['staged_comparison_entry.py']={'v':3}
        prior_exec={'old-helper':{'v':1}}
        current_exec=dict(prior_exec,**{str(f.AUTHORITY):{'v':2}})
        put(before,dict(source=old,execution=prior_exec))
        put(after,dict(source=accepted,execution=prior_exec))
        authority=authority_for([before,after])
        origin=dict(environment={'python':'fixture'})
        meta=dict(base=dict(provenance=dict(connection_source=old,native_origin=origin,
             environment=origin['environment']),execution=prior_exec))
        with patch.object(f,'BEFORE',before),patch.object(f,'AFTER',after),\
             patch.object(f,'source_snapshot',return_value=current),\
             patch.object(s,'execution_files',return_value=current_exec),\
             patch.object(cp,'_trusted_native',return_value=origin),\
             patch.object(cp,'provenance',side_effect=lambda v: v) as validate:
            bridge=f.compatible(meta,authority)
            self.assertNotEqual(bridge['historical_source'],bridge['current_source'])
            self.assertEqual(bridge['historical_source'],old)
            self.assertEqual(bridge['current_execution'],current_exec)
            validate.assert_called_once_with(dict(native_origin=origin,connection_source=current,
                                                   environment=origin['environment']))
            for mutation in ('removed','calculation','new_unknown'):
                bad=copy.deepcopy(current)
                if mutation=='removed':del bad['train.py']
                elif mutation=='calculation':bad['train.py']={'v':2}
                else:bad['unapproved.py']={'v':1}
                with self.subTest(mutation=mutation),patch.object(f,'source_snapshot',return_value=bad):
                    with self.assertRaises(ValueError):f.compatible(meta,authority)
            for field in ('connection_source','native_origin','environment'):
                bad=copy.deepcopy(meta);bad['base']['provenance'][field]={}
                with self.subTest(field=field),self.assertRaises(ValueError):f.compatible(bad,authority)
            bad=copy.deepcopy(meta);bad['base']['execution']={}
            with self.assertRaises(ValueError):f.compatible(bad,authority)
            with patch.object(s,'execution_files',return_value={}):
                with self.assertRaises(ValueError):f.compatible(meta,authority)
            before.write_bytes(before.read_bytes()+b' ')
            with self.assertRaises(ValueError):f.compatible(meta,authority)

    def test_fixed_candidate_no_permission_no_output_and_strict_binding(self):
        candidate=self.root/'candidate.json';program=self.root/'new-run'
        stages=[dict(name=n,run=str(program/n/'run'),output=str(program/n/'output')) for n in f.STAGES[1:]]
        value=dict(schema=f.SCHEMA,execution_authorized=False,executable=False,
            program=str(program),stages=stages,initial={'selected':True},execution_files={},
            provenance={},a_files={},resources=s.RESOURCES,
            profiles=[cp.contract(f.stage_config(x['name'],x['output'])) for x in stages])
        put(candidate,value)
        with patch.object(f,'CANDIDATE',candidate),patch.object(f,'PROGRAM',program),\
             patch.object(f,'check') as selected,patch.object(cp,'provenance'),\
             patch.object(s,'check_execution_files'),patch.object(e,'a_bindings',return_value={}):
            with self.assertRaisesRegex(ValueError,'separate_live_authorization'):e.run(value,candidate)
            selected.assert_not_called();self.assertFalse(program.exists())
            live=dict(value,executable=True,execution_authorized=True)
            e.bind(live,absent=True);selected.assert_called_once_with(live)
            for field in ('initial','provenance','execution_files','stages','profiles','program'):
                bad=copy.deepcopy(live);bad[field]=None
                with self.subTest(field=field),self.assertRaises(ValueError):e.bind(bad,absent=True)
            program.mkdir()
            with self.assertRaisesRegex(ValueError,'exclusive_program'):e.bind(live,absent=True)

    def test_three_stages_inherit_only_verified_predecessor_and_suppress_downstream(self):
        for failure in (None,'worker','reader','verify','comparison','finalization','origin'):
            with self.subTest(failure=failure):
                area=self.root/(failure or 'ok');area.mkdir();program=area/'new-program'
                stages=[dict(name=n,run=str(program/n/'run'),output=str(program/n/'output')) for n in f.STAGES[1:]]
                old={'path':'selected-stabilization-result','identity':{}}
                initial=dict(kind='comparison_final',path='selected-marker',stage='stabilization')
                value=dict(schema=f.SCHEMA,execution_authorized=True,executable=True,
                    program=str(program),stages=stages,selected_publication=old,initial=initial,
                    profiles=[dict(paths=dict(source_path=str(area)))]*3,
                    counts=dict(reused_B_optimizer=12000,new_B_optimizer=54000))
                events=[];complete={};parents=[]
                def checked(v):
                    if failure=='origin':raise ValueError('selected_parent_changed')
                def from_receipt(ref,next_stage):
                    prev=f.STAGES[f.STAGES.index(next_stage)-1]
                    self.assertEqual(ref['path'],str(program/prev/'run/completion/result.json'))
                    self.assertIn(prev,complete)
                    return dict(kind='comparison_final',path=prev+'-marker',stage=prev)
                def monitored(command,**kw):
                    stage=kw['out'].parent.name;mode=kw['mode']
                    self.assertEqual(Path(command[3]),s.ROOT/'staged_comparison_entry.py')
                    self.assertEqual(int(command[-1]),[x['name'] for x in stages].index(stage))
                    parent=cp._read_record(kw['out']/'parent.json')
                    if mode=='worker':parents.append(parent['stage'])
                    events.append((stage,mode))
                    if mode==failure:raise RuntimeError('injected_'+mode)
                def finalize(run,comparison):
                    stage=run.parent.name;events.append((stage,'comparison'))
                    if failure=='comparison':raise RuntimeError('comparison_failure')
                    complete[stage]=True
                    dest=run/'completion/result.json';put(dest,dict(stage=stage))
                    return results.reference(dest)
                def verify(run):
                    self.assertIn(run.parent.name,complete)
                    if failure=='finalization':raise RuntimeError('final_readback_failure')
                read=cp._read_record
                def record(path,*a):
                    if str(path).endswith('worker-result.json'):return dict(final_points=2)
                    return read(path,*a)
                # The controller is real; compute/reader/compare are event seams.
                # Parent publication validation has independent real-fixture tests.
                with patch.object(e,'bind'),patch.object(f,'check',side_effect=checked),\
                     patch.object(results,'parent_from_receipt',side_effect=from_receipt),\
                     patch.object(results,'reference',wraps=results.reference),\
                     patch.object(cp,'_read_record',side_effect=record),\
                     patch.object(results,'finalize',side_effect=finalize),\
                     patch.object(results,'verify_publication',side_effect=verify),\
                     patch.object(s,'resources',return_value=SimpleNamespace(start=lambda *a:dict(status='passed'),
                         memory=lambda *a:None,disk=lambda *a:None)),\
                     patch.object(s,'monitor',return_value=SimpleNamespace(monitored_child=monitored)),\
                     patch.object(s,'process_record'):
                    if failure:
                        with self.assertRaises((RuntimeError,ValueError)):e.run(value,area/'input.json')
                        row=read(program/'failure.json')
                        self.assertTrue(row['downstream_not_run'])
                        self.assertFalse((program/'completion').exists())
                    else:
                        e.run(value,area/'input.json')
                        row=read(program/'completion/result.json')
                        self.assertEqual(row['stages'][0],old);self.assertEqual(len(row['stages']),4)
                        self.assertEqual(parents,['stabilization','growth','e2_like'])
                        self.assertEqual(events,[(stage,mode) for stage in f.STAGES[1:]
                            for mode in ('worker','reader','verify','comparison')])
                    if failure not in (None,'finalization'):
                        self.assertTrue(all(stage=='growth' for stage,mode in events))
                self.assertFalse((program/'stabilization').exists())


class SelectedParentCPU(CPUCase):
    # Reuse existing tiny actual serializer/state/initializer fixtures, not the
    # production checkpoint or a CPU training loop.
    initial=FinalConnectionTests.initial
    bundle=FinalConnectionTests.bundle
    provenance_fixture=FinalConnectionTests.provenance_fixture
    final_fixture=FinalConnectionTests.final_fixture

    def test_selected_metadata_real_M_T_inheritance_and_negative_proofs(self):
        from staged_comparison_state import Initializer
        from comparison_inputs import _acquire_dataset
        run,receipt,loaded,attrs,groups=self.final_fixture()
        marker=Path(loaded['config'].paths.output_directory)/'checkpoints/completed-12000/complete.json'
        before=self.f.base/'before.json';after=self.f.base/'after.json';authority=self.f.base/'authority.json'
        meta=cp._read_record(marker.parent/'metadata.json')
        saved=dict(source=meta['base']['provenance']['connection_source'],execution=meta['base']['execution'])
        put(before,saved);put(after,saved)
        files=[before,after,run/'completion/result.json',run/'completion/receipt.json',
               marker,marker.parent/'metadata.json']
        put(authority,authority_for(files))
        # Same real M/T fixtures; independent metadata tests above cover the
        # old/new inventory difference, and the real-artifact preflight covers
        # the actual Fix4-before -> current delta.
        with patch.multiple(f,AUTHORITY=authority,BEFORE=before,AFTER=after,
            PARENT_RUN=run,PARENT_OUTPUT=marker.parent.parent.parent,
            PUBLICATION=run/'completion/result.json',MARKER=marker):
            parent,checked,bridge=f.selected()
            self.assertEqual(parent['verification'],receipt)
            self.assertEqual(checked,meta)
            c=f.stage_config('growth',str(self.f.base/'growth'))
            init=Initializer.from_checkpoint(parent,c)
            for value,slot in zip(init.parameters,(1,2,3,4,5,6,13,14,15)):
                self.assertTrue(self.torch.equal(value,attrs[slot]))
            prepared=self.train.prepare_comparison_runtime(_acquire_dataset(c,init))
            self.assertEqual(prepared.gaussians.optimizer.state,{})
            self.assertEqual(init.parameters,())
            self.assertEqual((prepared.gaussians.active_sh_degree,prepared.gaussians.active_sh_degree_t),(0,0))
            self.assertTrue(all(g['params'][0].grad is None for g in prepared.gaussians.optimizer.param_groups))
            for attr in ('max_radii2D','xyz_gradient_accum','t_gradient_accum','denom'):
                self.assertFalse(getattr(prepared.gaussians,attr).any())
            for field in ('schema','k','n','final'):
                bad=copy.deepcopy(meta);bad[field]=None
                with self.subTest(field=field),self.assertRaises(ValueError):
                    cp.validate_metadata(bad,marker.parent.parent.parent)
            with self.assertRaises(ValueError):Initializer.from_checkpoint(parent,f.stage_config('e2_like',str(self.f.base/'bad')))
            for name in ('reader-result.json','verify-result.json','verify-exit.json','completion/receipt.json'):
                p=run/name;original=p.read_bytes();p.write_bytes(original+b'x')
                try:
                    with self.subTest(name=name),self.assertRaises(ValueError):f.selected()
                finally:p.write_bytes(original)
            put(run/'failure.json',{})
            with self.assertRaises(ValueError):f.selected()


if __name__=='__main__':unittest.main(verbosity=2)
