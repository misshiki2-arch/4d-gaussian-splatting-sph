"""Strict stdlib-only Step 12 candidate/preflight tests, never CUDA execution."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'tests/cuda/training_update_cuda_acceptance.py'
spec = importlib.util.spec_from_file_location('step12_entry', PATH)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = entry.source_identity()

    def test_candidate_contains_all_new_uncommitted_and_build_inputs(self):
        for name in entry.EXTRA+(entry.HELPER, 'train.py', 'simple-knn/setup.py',
                                   'scene/gaussian_model.py', 'diff-gaussian-rasterization/cuda_rasterizer/backward.cu'):
            self.assertIn(name, self.expected['files'])
        self.assertEqual(entry.verify_source(self.expected), self.expected)

    def test_missing_hash_changed_and_added_input_fail_before_helper(self):
        name = entry.EXTRA[0]
        for mode in ('missing', 'changed', 'outside'):
            v = copy.deepcopy(self.expected)
            if mode == 'missing':
                del v['files'][name]
            elif mode == 'changed':
                v['files'][name] = '0'*64  # Deliberately invalid fixture, not authority.
            else:
                v['files']['../outside.py'] = v['files'][name]
            with self.subTest(mode=mode), patch.object(entry, 'helper') as helper:
                with self.assertRaises(RuntimeError):
                    entry.verify_source(v)
                helper.assert_not_called()

    def test_preflight_no_output_existing_alias_and_debug_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            out = base/'run'
            with patch.object(entry, 'REPORTS', base):
                entry.preflight(self.expected, out)
                self.assertFalse(out.exists())
                out.mkdir()
                with self.assertRaisesRegex(RuntimeError, 'new_unaliased'):
                    entry.preflight(self.expected, out)
                alias = base/'alias'
                alias.symlink_to(out, target_is_directory=True)
                with self.assertRaisesRegex(RuntimeError, 'new_unaliased'):
                    entry.preflight(self.expected, alias/'new')
                with patch.dict(os.environ, {'STEP90_CUDA_DEBUG': '1'}):
                    with self.assertRaisesRegex(RuntimeError, 'external_diagnostic'):
                        entry.preflight(self.expected, base/'other')
                self.assertFalse((base/'other').exists())

    def test_duplicate_schema_and_loaded_module_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'input.json'
            path.write_text('{"schema":1,"schema":2}')
            with self.assertRaisesRegex(RuntimeError, 'duplicate_evidence'):
                entry.load_source_evidence(path)
        fake = types.ModuleType('training_update_runtime')
        fake.__file__ = '/outside/training_update_runtime.py'
        with patch.dict(sys.modules, {'training_update_runtime': fake}):
            with self.assertRaisesRegex(RuntimeError, 'unexpected_import'):
                entry.loaded_test_inputs(self.expected)

    def test_raw_index_bytes_not_substitute_for_entries_flags(self):
        a = dict(head='fixture', branch='fixture', status='fixture', entries='same', flags='same', raw_index='old')
        b = dict(a, raw_index='new')
        self.assertTrue(entry.same_git(a, b))
        self.assertFalse(entry.same_git(a, dict(b, entries='different')))
        self.assertFalse(entry.same_git(a, dict(b, flags='different')))

    def test_isolated_fresh_interpreter_no_heavy_import_no_output(self):
        code = '''
import importlib.util, pathlib, sys, tempfile
class Deny:
 def find_spec(self, name, *a):
  if name.split('.')[0] in ('torch','numpy','PIL','scene','train','gaussian_renderer','simple_knn','pointops2'):
   raise AssertionError('heavy import in preflight')
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location('entry',sys.argv[1])
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
v=m.source_identity()
with tempfile.TemporaryDirectory() as t:
 m.REPORTS=pathlib.Path(t)
 out=m.REPORTS/'new'
 m.preflight(v,out)
 assert not out.exists()
try:m.main([])
except SystemExit as e:assert e.code==2
else:raise AssertionError('missing execution arguments accepted')
assert not any(n in sys.modules for n in m.HEAVY)
print(sys.executable)
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', code, str(PATH)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), entry.PYTHON)


if __name__ == '__main__':
    print('test executable:', sys.executable)
    unittest.main()
