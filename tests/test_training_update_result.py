"""Step 12 result finalization only: stdlib CPU, no build/import/CUDA work."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT/'tests/cuda/training_update_cuda_acceptance.py'
spec = importlib.util.spec_from_file_location('step12_result_entry', ENTRY)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def sample():
    return dict(status='checks_passed', stage='checks_complete',
                preservation=dict(source=True, git_content=True, rasterizer=True, KNN=True),
                cases=[dict(case='synthetic', steps=3)], identity='synthetic-identity',
                flags=dict(enabled=True), count=1, scalar=1.0,
                SH_regression=dict(inputs=[dict(stage=(3, 2), spec=dict(
                    pixel=(37, 26), background=(.07, .11, .03), timestamp=.4))]))


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()

    def finish(self, result):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return entry.finalize_result(self.out, result)

    def failure(self):
        self.assertNotIn('STEP12_RESULT_COMPLETE', self.stdout.getvalue())
        value = json.loads((self.out/'failure.json').read_text())
        self.assertEqual(value['status'], 'failed')
        self.assertIn('STEP12_RESULT_FAILED', self.stderr.getvalue())
        return value

    def test_nested_SH_tuple_roundtrip_nonmutating_and_receipt(self):
        value = sample()
        before = copy.deepcopy(value)
        self.assertTrue(self.finish(value))
        actual = json.loads((self.out/'result.json').read_text())
        expected = copy.deepcopy(before)
        item = expected['SH_regression']['inputs'][0]
        item['stage'] = [3, 2]
        item['spec']['pixel'] = [37, 26]
        item['spec']['background'] = [.07, .11, .03]
        expected['finalization'] = 'requires_verified_process_receipt'
        self.assertEqual(actual, expected)
        self.assertEqual(value, before)
        self.assertIsInstance(value['SH_regression']['inputs'][0]['stage'], tuple)
        receipt = json.loads(self.stdout.getvalue().split('STEP12_RESULT_COMPLETE ', 1)[1])
        self.assertEqual(receipt['status'], 'passed')
        self.assertEqual(receipt['stage'], 'complete')
        self.assertEqual(receipt['result_sha256'], hashlib.sha256((self.out/'result.json').read_bytes()).hexdigest())
        self.assertEqual(actual['status'], 'checks_passed')
        self.assertNotEqual(actual['stage'], 'complete')
        self.assertFalse((self.out/'failure.json').exists())
        self.assertEqual(self.stderr.getvalue(), '')

    def test_saved_validation1_full_values_with_real_SH_input_shape(self):
        old = ROOT.parent/'reports/corrected-4dgs/phase1/step12/step12-validation1/step12-validation1-run/result.json'
        raw = old.read_bytes()
        value = json.loads(raw)
        value.update(status='checks_passed', stage='checks_complete')
        expected = copy.deepcopy(value)
        for item in value['SH_regression']['inputs']:
            item['stage'] = tuple(item['stage'])
            for key in ('pixel', 'background'):
                item['spec'][key] = tuple(item['spec'][key])
        self.assertTrue(self.finish(value))
        expected['finalization'] = 'requires_verified_process_receipt'
        self.assertEqual(json.loads((self.out/'result.json').read_text()), expected)
        self.assertEqual(old.read_bytes(), raw)

    def test_str_subclass_and_shared_containers_keep_values(self):
        class Version(str):
            pass
        common = [Version('2.3.1'), None, False, 5, -0.0]
        value = dict(a=common, b=common)
        actual = entry.result_payload(value)
        self.assertEqual(actual, value)
        self.assertIsNot(actual['a'], actual['b'])
        self.assertIs(type(actual['a'][0]), str)

    def test_nonfinite_unsupported_keys_and_cycles_fail_closed(self):
        cycle = []; cycle.append(cycle)
        for bad in (float('nan'), float('inf'), -float('inf'), object(), b'x', {1}, {1: 'key'}, cycle):
            with self.subTest(kind=type(bad).__name__), tempfile.TemporaryDirectory() as temp:
                self.out = Path(temp)
                value = sample(); value['bad'] = bad
                with self.assertRaises((TypeError, ValueError)):
                    self.finish(value)
                self.failure()
                self.assertFalse((self.out/'result.json').exists())

    def test_missing_modified_and_changed_leaf_types_rejected(self):
        original = entry.read_result
        def changed(path):
            value = original(path)
            if mode == 'missing': del value['cases']
            if mode == 'value': value['SH_regression']['inputs'][0]['spec']['pixel'][0] += 1
            if mode == 'bool_int': value['flags']['enabled'] = 1
            if mode == 'int_float': value['count'] = 1.0
            return value
        for mode in ('missing', 'value', 'bool_int', 'int_float'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                self.out = Path(temp)
                with patch.object(entry, 'read_result', side_effect=changed):
                    with self.assertRaisesRegex(RuntimeError, 'result_readback_mismatch'):
                        self.finish(sample())
                self.failure()
                self.assertEqual(json.loads((self.out/'result.json').read_text())['status'], 'checks_passed')

    def test_invalid_truncated_duplicate_and_nonfinite_saved_JSON(self):
        original = entry.read_result
        for damaged in ('not-json', '{"x":', '{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            def tamper(path):
                path.write_text(damaged)
                return original(path)
            with self.subTest(damaged=damaged), tempfile.TemporaryDirectory() as temp:
                self.out = Path(temp)
                with patch.object(entry, 'read_result', side_effect=tamper):
                    with self.assertRaises(ValueError):
                        self.finish(sample())
                self.failure()

    def test_write_failure_and_failure_record_unavailable_emit_stderr(self):
        with patch.object(Path, 'open', side_effect=PermissionError('CPU injected write denial')):
            with self.assertRaises(PermissionError):
                self.finish(sample())
        self.assertIn('failure_record_error', self.stderr.getvalue())
        self.assertIn('PermissionError', self.stderr.getvalue())
        self.assertNotIn('STEP12_RESULT_COMPLETE', self.stdout.getvalue())
        self.assertFalse(list(self.out.iterdir()))

    def test_partial_write_and_close_failure_never_save_final_success(self):
        original = Path.open
        for mode in ('write', 'close'):
            class BrokenStream:
                def __init__(self, stream): self.stream = stream
                def __enter__(self): return self
                def write(self, text):
                    if mode == 'write':
                        self.stream.write(text[:len(text)//2])
                        raise OSError('CPU partial write')
                    return self.stream.write(text)
                def __exit__(self, *args):
                    self.stream.close()
                    if mode == 'close': raise OSError('CPU close failure')
            def opened(path, *a, **kw):
                stream = original(path, *a, **kw)
                return BrokenStream(stream) if path == self.out/'result.json' else stream
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                self.out = Path(temp)
                with patch.object(Path, 'open', opened), self.assertRaises(OSError):
                    self.finish(sample())
                self.failure()
                if mode == 'close':
                    saved = json.loads((self.out/'result.json').read_text())
                    self.assertEqual(saved['status'], 'checks_passed')
                    self.assertNotEqual(saved['stage'], 'complete')

    def test_read_failure_retains_primary_and_preservation_causes(self):
        value = sample()
        value.update(status='failed', stage='training/G1', error='RuntimeError: CPU primary failure')
        value['preservation']['source'] = 'CPU preservation failure'
        with patch.object(entry, 'read_result', side_effect=OSError('CPU read failure')):
            with self.assertRaises(OSError): self.finish(value)
        failure = self.failure()
        self.assertEqual(failure['primary_error'], value['error'])
        self.assertIn('CPU read failure', failure['finalization_error'])
        self.assertEqual(failure['preservation_errors'], {'source': 'CPU preservation failure'})
        self.assertEqual(json.loads((self.out/'result.json').read_text())['error'], value['error'])

    def test_prior_execution_or_preservation_failure_not_promoted(self):
        for mode in ('execution', 'preservation', 'nonboolean', 'missing', 'unfinished'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                self.out = Path(temp); value = sample()
                if mode == 'execution': value.update(status='failed', error='CPU prior error')
                if mode == 'preservation': value['preservation']['KNN'] = False
                if mode == 'nonboolean': value['preservation']['source'] = 1
                if mode == 'missing': del value['preservation']['source']
                if mode == 'unfinished': value['stage'] = 'training/G1'
                self.assertFalse(self.finish(value))
                self.assertIsNone(self.failure()['finalization_error'])

    def test_existing_result_is_not_replaced_or_retried(self):
        path = self.out/'result.json'; path.write_text('protected preexisting result')
        with self.assertRaises(FileExistsError): self.finish(sample())
        self.assertEqual(path.read_text(), 'protected preexisting result')
        self.failure()

    def test_failure_record_collision_preserves_old_record(self):
        path = self.out/'failure.json'; path.write_text('protected failure')
        with patch.object(entry, 'read_result', side_effect=OSError('CPU read failure')):
            with self.assertRaises(OSError): self.finish(sample())
        self.assertEqual(path.read_text(), 'protected failure')
        self.assertIn('failure_record_error', self.stderr.getvalue())
        self.assertNotIn('STEP12_RESULT_COMPLETE', self.stdout.getvalue())

    def test_receipt_output_failure_does_not_leave_passed_file(self):
        class BrokenOutput(io.StringIO):
            def write(self, text): raise OSError('CPU stdout failure')
        self.stdout = BrokenOutput()
        with self.assertRaises(OSError): self.finish(sample())
        self.failure()
        self.assertEqual(json.loads((self.out/'result.json').read_text())['status'], 'checks_passed')

    def test_failure_text_bounded_and_no_unknown_payload_stringification(self):
        class Unknown:
            def __str__(self): raise AssertionError('unknown payload stringified')
        value = sample(); value['unknown'] = Unknown(); value['error'] = 'x'*8000
        with self.assertRaises(TypeError): self.finish(value)
        self.assertEqual(len(self.failure()['primary_error']), 4000)

    def test_premature_passed_result_is_rejected(self):
        value = sample(); value.update(status='passed', stage='complete')
        with self.assertRaisesRegex(ValueError, 'nonterminal_result_required'): self.finish(value)
        self.failure()
        self.assertFalse((self.out/'result.json').exists())

    def test_real_main_finally_connected_before_any_heavy_import(self):
        code = '''
import importlib.util,json,pathlib,sys,tempfile
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('entry',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Deny:
 def find_spec(self,name,*a):
  if name.split('.')[0] in m.HEAVY: raise RuntimeError('CPU blocked heavy import: '+name)
sys.meta_path.insert(0,Deny())
with tempfile.TemporaryDirectory() as temp:
 base=pathlib.Path(temp);m.REPORTS=base
 source=base/'input.json';source.write_text(json.dumps(m.source_identity()))
 original=m.read_result
 def read(path):
  if sys.argv[2]=='read_failure':raise OSError('CPU finalization read failure')
  return original(path)
 with patch.object(m,'read_result',read):
  try:m.main(['--source-evidence',str(source),'--output-dir',str(base/'run')])
  except (RuntimeError,OSError):pass
  else:raise AssertionError('real main incorrectly succeeded')
 result=json.loads((base/'run/result.json').read_text())
 failure=json.loads((base/'run/failure.json').read_text())
 assert result['status']=='failed' and result['stage']=='heavy_import'
 assert 'CPU blocked heavy import: torch' in result['error']
 assert failure['primary_error']==result['error']
 assert (failure['finalization_error'] is not None)==(sys.argv[2]=='read_failure')
 assert not any(n in sys.modules for n in m.HEAVY)
 assert not list((base/'run/rasterizer-build').iterdir())
print(sys.executable)
'''
        for mode in ('saved_failure', 'read_failure'):
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', code, str(ENTRY), mode],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), entry.PYTHON)
                self.assertIn('STEP12_RESULT_FAILED', result.stderr)
                self.assertNotIn('STEP12_RESULT_COMPLETE', result.stdout)


if __name__ == '__main__':
    print('test executable:', sys.executable)
    unittest.main()
