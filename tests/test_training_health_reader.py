"""CPU public reader to the very same mandatory parameter comparator as CUDA."""
import importlib.util
from contextlib import ExitStack
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parent))
from formal_training_fixture import CPUCase
import formal_checkpoint as cp
import formal_entry
from formal_checkpoint_identity import GROUPS,validate_native
from formal_checkpoint_state import restore_diagnostic
PATH=Path(__file__).parent/'cuda/training_health_cuda_acceptance.py'
spec=importlib.util.spec_from_file_location('health_reader_entry',PATH)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class ReaderTests(CPUCase):
    def test_real_save_load_restore_all_nine_parameters_and_each_mismatch(self):
        first=self.configured(n=1,batch=1)
        p=self.train.prepare_formal_runtime(first.inputs,native_evidence=self.f.checkpoint_evidence())
        m=p.gaussians;m.env_map=None;m.update_learning_rate(1);m.oneupSHdegree()
        sum(g['params'][0].sum() for g in m.optimizer.param_groups).backward()
        m.optimizer.step();m.optimizer.zero_grad(set_to_none=True)
        with patch.object(self.train,'TENSORBOARD_FOUND',False):
            claim=formal_entry.OutputClaim(p.inputs);self.addCleanup(claim.close)
            self.train.begin_formal_output(p,claim);p.scene.save(1)
        loaded=cp.read_checkpoint(Path(p.scene.model_path)/'checkpoints/completed-1')
        with ExitStack() as stack:
            entry.guard_reader_training(stack)
            validate_native(self.f.checkpoint_evidence(),live=True)
            restored=restore_diagnostic(loaded,'cpu')
            with self.assertRaisesRegex(AssertionError,'reader_training_or_KNN'):
                sys.modules['scene.gaussian_model'].distCUDA2(None)
            with self.assertRaisesRegex(AssertionError,'reader_training_or_KNN'):
                restored.training_setup(None)
        compared=entry.compare_parameters(restored,loaded['state']['capture'])
        self.assertEqual(set(compared),{'xyz','f_dc','f_rest','scaling','rotation','opacity','t','scaling_t','rotation_r'})
        self.assertEqual(len(compared),9)
        for name,(_,attr) in GROUPS.items():
            parameter=getattr(restored,attr)
            with self.torch.no_grad():
                before=parameter.clone();parameter.add_(1.)
                with self.subTest(group=name),self.assertRaises(ValueError):
                    entry.compare_parameters(restored,loaded['state']['capture'])
                parameter.copy_(before)
        with patch.object(entry,'GROUPS',{}):
            with self.assertRaisesRegex(ValueError,'required_parameter_groups'):
                entry.compare_parameters(restored,loaded['state']['capture'])
        self.assertFalse(self.torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
