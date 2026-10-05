"""Stdlib-only Step 10 preflight tests; actual files, no build/output/GPU."""
import ast
import builtins
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HEAVY = ('torch', 'numpy', 'PIL', 'scene', 'gaussian_renderer', 'pointops2', 'simple_knn')
ORIGINAL_IMPORT = builtins.__import__


def deny_heavy(name, *args, **kwargs):
    if name.split('.')[0] in HEAVY: raise AssertionError('heavy_import_attempt: '+name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


with patch('builtins.__import__', side_effect=deny_heavy):
    spec = importlib.util.spec_from_file_location('camera_entry_cpu', ROOT/'tests/cuda/camera_projection_cuda_acceptance.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.executable != entry.PYTHON or not sys.dont_write_bytecode:
            raise RuntimeError('4dgs310_B_required')
        guard = patch('builtins.__import__', side_effect=deny_heavy)
        guard.start(); cls.addClassCleanup(guard.stop)
        cls.actual = entry.source_identity()
        print('executable:', sys.executable, 'input count:', len(cls.actual['files']))

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='step10-source-cpu-')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name); self.root = self.base/'research'; self.root.mkdir()
        for name in self.actual['files']:
            path = self.root/name; path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, path)
        self.reports = self.base/'step10'; self.reports.mkdir()
        self.output = self.reports/'future-run'
        for name, value in (('ROOT', self.root), ('REPORTS', self.reports)):
            guard = patch.object(entry, name, value); guard.start(); self.addCleanup(guard.stop)
        self.alpha = entry.helper(); self.sh = self.alpha.helper()
        self.sh.git = lambda *args: self.actual['base_head']
        for owner, name, value in ((self.alpha, 'helper', lambda: self.sh), (entry, 'helper', lambda: self.alpha)):
            guard = patch.object(owner, name, value); guard.start(); self.addCleanup(guard.stop)
        self.expected = dict(self.actual, repository=str(self.root))
        self.evidence = self.base/'reviewed.json'; self.write(self.expected)

    def write(self, value):
        self.evidence.write_text(json.dumps(value))

    def main(self):
        entry.main(['--source-evidence', str(self.evidence), '--output-dir', str(self.output)])

    def reject(self):
        with self.assertRaises((RuntimeError, FileNotFoundError)): self.main()
        self.assertFalse(self.output.exists())
        self.assertFalse(any(k in sys.modules for k in HEAVY))

    def test_wip_and_new_files_are_inputs_not_clean_head_gate(self):
        self.assertEqual(entry.verify_source(entry.load_source_evidence(self.evidence)), self.expected)
        self.assertTrue(set(entry.EXTRA) <= self.expected['files'].keys())
        self.assertTrue(set(self.sh.TRANSLATION_UNITS) <= self.expected['files'].keys())
        with patch.object(self.alpha, 'git_state', return_value={'status': ' M approved_WIP'}), \
                patch.object(Path, 'mkdir', side_effect=RuntimeError('output_boundary')):
            with self.assertRaisesRegex(RuntimeError, 'output_boundary'): self.main()

    def test_every_new_file_missing_changed_or_alias_rejected(self):
        for name in entry.EXTRA:
            path = self.root/name; original = path.read_bytes()
            for mutation in ('missing', 'changed', 'alias'):
                with self.subTest(file=name, mutation=mutation):
                    path.unlink()
                    if mutation == 'changed': path.write_bytes(original+b'\n# unexpected edit\n')
                    if mutation == 'alias': path.symlink_to(ROOT/name)
                    self.reject()
                    if path.exists() or path.is_symlink(): path.unlink()
                    path.write_bytes(original)

    def test_changed_helper_rejected_before_executing_any_helper(self):
        for name in (entry.HELPER, 'tests/cuda/sh_cuda_acceptance.py'):
            path = self.root/name; original = path.read_bytes()
            path.write_bytes(original+b'\n# changed helper\n')
            with patch.object(entry, 'helper', side_effect=AssertionError('unreviewed_helper_executed')):
                self.reject()
            path.write_bytes(original)

    def test_production_header_binding_consumer_and_fixture_closure(self):
        for name in ('diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h',
                     'gaussian_renderer/diff_gaussian_rasterization.py', 'scene/dataset_readers.py',
                     'utils/camera_utils.py', 'tests/sh_oracle.py', 'tests/alpha_cap_oracle.py',
                     'tests/test_formal_config.py'):
            path = self.root/name; original = path.read_bytes()
            path.write_bytes(original+b'\n# changed input\n'); self.reject(); path.write_bytes(original)
        (self.root/'utils/new_input.py').write_text('# added input')
        self.reject()

    def test_context_head_and_unrelated_docs_do_not_authorize_byte_changes(self):
        changed = copy.deepcopy(self.expected)
        head = changed['base_head']; changed['base_head'] = ('0' if head[0] != '0' else '1')+head[1:]
        (self.root/'README.md').write_text('unrelated')
        self.write(changed)
        self.assertEqual(entry.verify_source(entry.load_source_evidence(self.evidence))['files'], changed['files'])
        (self.root/entry.EXTRA[0]).write_text('# modified')
        self.reject()

    def test_schema_duplicates_missing_extra_and_path_escape(self):
        self.evidence.write_text('{"schema":1,"schema":2}'); self.reject()
        values = [dict(self.expected, schema=self.alpha.SCHEMA), dict(self.expected, extra=True)]
        missing = copy.deepcopy(self.expected); del missing['files'][entry.EXTRA[1]]; values.append(missing)
        for name in ('../escape.py', '/absolute.py', './alias.py', 'tests/../escape.py'):
            value = copy.deepcopy(self.expected)
            value['files'][name] = next(iter(value['files'].values())); values.append(value)
        for value in values:
            self.write(value); self.reject()

    def test_output_new_absolute_under_step10_and_no_alias(self):
        good = self.output
        for value in (Path('relative'), self.base/'outside', self.root/'new', self.reports/'missing/new'):
            self.output = value; self.reject()
        self.output = good; good.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'new_unaliased_absolute_output_required'): self.main()
        self.assertEqual(list(good.iterdir()), []); good.rmdir()
        alias = self.reports/'alias'; alias.symlink_to(self.base, target_is_directory=True)
        self.output = alias/'new'; self.reject()

    def test_record_candidate_is_stdlib_only_and_cannot_execute(self):
        stream = io.StringIO()
        with redirect_stdout(stream): entry.main(['--record-source-evidence'])
        self.assertEqual(json.loads(stream.getvalue()), self.expected)
        self.assertFalse(self.output.exists())
        for option in ('--source-evidence', '--output-dir'):
            with self.assertRaises(SystemExit): entry.main(['--record-source-evidence', option, str(self.output)])

    def test_subsets_keep_reviewed_hashes_and_old_schemas(self):
        subset = entry.alpha_subset(self.expected, self.alpha)
        self.assertEqual(subset['schema'], self.alpha.SCHEMA)
        self.assertEqual(self.alpha.verify_source(subset), subset)
        subsh = self.alpha.sh_subset(subset, self.sh)
        self.assertEqual(subsh['schema'], self.sh.INPUT_SCHEMA)
        for k, v in subsh['files'].items(): self.assertEqual(v, self.expected['files'][k])
        (self.root/entry.EXTRA[1]).write_text('# changed after snapshot')
        with self.assertRaisesRegex(RuntimeError, 'reviewed_input_mismatch'): entry.alpha_subset(self.expected, self.alpha)

    def test_loaded_oracle_runtime_shadowing_and_external_debug_rejected(self):
        for name in ('camera_projection_oracle', 'camera_projection_runtime'):
            module = types.ModuleType(name); module.__file__ = str(self.root/'tests'/(name+'.py'))
            with patch.dict(sys.modules, {name: module}):
                self.assertIn(name, entry.verify_loaded_inputs(self.expected, self.alpha, self.sh))
                module.__file__ = str(self.base/'wrong.py')
                with self.assertRaisesRegex(RuntimeError, 'unexpected_import_source'):
                    entry.verify_loaded_inputs(self.expected, self.alpha, self.sh)
        with patch.dict(os.environ, STEP90_CUDA_DEBUG_PIXEL='1,1'): self.reject()

    def test_syntax_and_same_binary_regression_functions_not_mains(self):
        for name in entry.EXTRA:
            ast.parse((self.root/name).read_text(), filename=name)
        tree = ast.parse((self.root/'tests/cuda/camera_projection_cuda_acceptance.py').read_text())
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        attributes = [c.attr for c in calls if isinstance(c, ast.Attribute)]
        self.assertIn('run_alpha_checks', attributes); self.assertIn('run_checks', attributes)
        self.assertNotIn('main', attributes)

    def test_reused_binary_identity_rejects_cache_recipe_and_object_errors(self):
        build = self.base/'future-build-fixture'; build.mkdir()
        directory = build/'module'; directory.mkdir()
        binary = directory/'module.so'; binary.write_bytes(b'CPU test data, not loadable')
        recipe = directory/'build.ninja'
        recipe.write_text('\n'.join(str(self.root/p) for p in self.sh.TRANSLATION_UNITS))
        for i in range(5): (directory/f'{i}.o').write_bytes(bytes([i]))
        binding = types.SimpleNamespace(_C=types.SimpleNamespace(__file__=str(binary)),
                   __file__=str(self.root/'gaussian_renderer/diff_gaussian_rasterization.py'))
        self.assertEqual(len(self.alpha.binary_identity(binding, build, self.sh)['objects']), 5)
        binding._C.__file__ = str(self.base/'unrelated-cache.so')
        with self.assertRaisesRegex(RuntimeError, 'loaded_binary_outside_fresh_build'):
            self.alpha.binary_identity(binding, build, self.sh)
        binding._C.__file__ = str(binary); recipe.write_text('missing translation units')
        with self.assertRaisesRegex(RuntimeError, 'translation_units_mismatch'):
            self.alpha.binary_identity(binding, build, self.sh)
        (directory/'4.o').unlink()
        with self.assertRaisesRegex(RuntimeError, 'fresh_build_evidence_missing'):
            self.alpha.binary_identity(binding, build, self.sh)


if __name__ == '__main__':
    unittest.main()
