"""CPU/stdlib checks of the future CUDA launcher, never call run/worker/reader."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'tests/cuda/checkpoint_cuda_acceptance.py'
spec=importlib.util.spec_from_file_location('checkpoint_acceptance',PATH)
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class PreflightTests(unittest.TestCase):
    def test_stdlib_only_candidate_includes_new_production_and_test_inputs(self):
        code="""
import sys,importlib.util
class Deny:
 def find_spec(self,n,*a):
  if n.split('.')[0] in ('torch','numpy','PIL','train','scene','gaussian_renderer','simple_knn'):
   raise AssertionError('heavy import in candidate/preflight')
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location('e',sys.argv[1]);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
v=e.candidate();e.verify_candidate(v)
assert {'formal_checkpoint.py','formal_checkpoint_state.py','formal_native_bootstrap.py',
        'tests/cuda/checkpoint_cuda_acceptance.py'} <= v['files'].keys()
assert 'torch' not in sys.modules
"""
        r=subprocess.run([sys.executable,'-I','-B','-S','-c',code,str(PATH)],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_input_mismatch_and_existing_output_refused_before_heavy(self):
        value=entry.candidate();entry.verify_candidate(value)
        value['files'].pop('train.py')
        with self.assertRaises(ValueError):entry.verify_candidate(value)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):entry.preflight(entry.candidate(),Path(d))
        self.assertNotIn('torch',sys.modules)

    def test_strict_result_readback_and_exclusive_creation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'result.json'
            result=entry.write_json(p,dict(status='passed',comparisons=[dict(max_abs=0.,exact=True)]))
            self.assertGreater(result['size'],0)
            with self.assertRaises(FileExistsError):entry.write_json(p,dict(status='passed'))
            with self.assertRaises(ValueError):entry.write_json(Path(d)/'nan.json',dict(value=float('nan')))


if __name__=='__main__':
    print('CPU_EXECUTABLE',sys.executable,flush=True)
    unittest.main(verbosity=2)
