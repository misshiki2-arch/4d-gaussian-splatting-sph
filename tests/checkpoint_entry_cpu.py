"""Fresh-process real formal entry, substituting only compiler/native/GPU."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    import torch
    import training_update_runtime as runtime
    import formal_inputs
    import formal_entry
    import formal_native_bootstrap as native
    import formal_checkpoint
    from checkpoint_native_fixture import compiler_boundaries
    from test_training_update_cpu import ConsumerTests,Event
    from test_formal_checkpoint import independent,assert_expected
    import unittest
    torch.set_num_threads(1)
    test=unittest.TestCase(); control=ConsumerTests('runTest');control.torch=torch
    for name in ('train','scene','gaussian_renderer','simple_knn._C'):
        test.assertNotIn(name,sys.modules)
    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        base=Path(tmp);case=next(c for c in runtime.CASES if c['name']=='G0a')
        _,reference=runtime.create_fixture(base/'case',case)
        stack.enter_context(patch.object(formal_inputs,'_reference',return_value=reference))
        stack.enter_context(compiler_boundaries(base))
        stack.enter_context(patch.object(torch.cuda,'empty_cache'))
        original=native.bootstrap
        events=[];saved={}
        def bootstrap(source):
            events.append('bootstrap')
            train,evidence=original(source)
            stack.enter_context(patch.object(torch.cuda,'Event',Event))
            realprepare=train.prepare_formal_runtime
            realbegin=train.begin_formal_output
            realcontinue=train.continue_formal_training
            def prepare(inputs,**kw):
                events.append('prepare');test.assertIs(kw['native_evidence'],evidence)
                p=realprepare(inputs,**kw);saved['prepared']=p
                test.assertFalse(Path(p.scene.model_path).exists());return p
            def begin(p,claim):
                events.append('claim');result=realbegin(p,claim)
                test.assertIs(p.scene._checkpoint_writer.context,p.checkpoint_context)
                return result
            def continuation(p,writer):
                events.append('continue')
                originalsave=p.scene.save
                def save(k):
                    events.append('save');expected=independent(p.gaussians)
                    receipt=originalsave(k)
                    result=formal_checkpoint.read_checkpoint(Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k)))
                    assert_expected(test,expected,result)
                    test.assertEqual(result['metadata']['base']['native'],evidence.value())
                    return receipt
                with patch.object(p.scene,'save',save):return realcontinue(p,writer)
            stack.enter_context(patch.object(train,'prepare_formal_runtime',prepare))
            stack.enter_context(patch.object(train,'begin_formal_output',begin))
            stack.enter_context(patch.object(train,'continue_formal_training',continuation))
            stack.enter_context(patch.object(train,'render',control.renderer))
            stack.enter_context(patch.object(train,'TENSORBOARD_FOUND',False))
            return train,evidence
        stack.enter_context(patch.object(native,'bootstrap',bootstrap))
        test.assertEqual(formal_entry.run(base/'case/config.json'),1)
        test.assertEqual(events,['bootstrap','prepare','claim','continue','save'])
        test.assertFalse(torch.cuda.is_initialized())
        print('ENTRY_CPU_PASSED '+json.dumps(dict(executable=sys.executable,events=events,
              real_collector=True,real_prepare=True,real_writer=True,real_load=True,cuda=False)))


if __name__=='__main__':main()
