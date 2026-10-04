"""Step 9 stdlib-only input/output/binary boundary checks; no renderer/JIT/CUDA.

Only Git context and binary metadata are test doubles; file inventories,
digests, schema rejection and path checks use real temporary files.
"""
import ast
import builtins
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_IMPORT = builtins.__import__
HEAVY = ('torch', 'numpy', 'gaussian_renderer', 'scene', 'pointops2', 'simple_knn')


def blocked_import(name, *args, **kwargs):
    if name.split('.')[0] in HEAVY: raise AssertionError('heavy import attempted: '+name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


with patch('builtins.__import__', side_effect=blocked_import):
    spec = importlib.util.spec_from_file_location('alpha_entry_cpu', ROOT/'tests/cuda/alpha_cap_cuda_acceptance.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.executable != entry.PYTHON or not sys.dont_write_bytecode:
            raise RuntimeError('explicit_4dgs310_B_required')
        guard = patch('builtins.__import__', side_effect=blocked_import)
        guard.start(); cls.addClassCleanup(guard.stop)
        cls.actual = entry.source_identity()
        print('executable:', sys.executable)
        print('actual input count:', len(cls.actual['files']))

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='step9-input-test-')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name); self.root = self.base/'research'; self.root.mkdir()
        for name in self.actual['files']:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        self.reports = self.base/'reports'; self.reports.mkdir()
        self.output = self.reports/'new-run'
        p = patch.object(entry, 'ROOT', self.root); p.start(); self.addCleanup(p.stop)
        p = patch.object(entry, 'REPORTS', self.reports); p.start(); self.addCleanup(p.stop)
        self.sh = entry.helper()
        def git(*args):
            self.assertEqual(args, ('rev-parse', 'HEAD'))
            return self.actual['base_head']
        self.sh.git = git
        p = patch.object(entry, 'helper', return_value=self.sh); p.start(); self.addCleanup(p.stop)
        p = patch.object(entry, 'git_state', return_value={'context': 'test_only'}); p.start(); self.addCleanup(p.stop)
        self.expected = dict(self.actual, repository=str(self.root))
        self.evidence = self.base/'reviewed.json'; self.write(self.expected)

    def write(self, value):
        self.evidence.write_text(json.dumps(value))

    def main(self):
        entry.main(['--source-evidence', str(self.evidence), '--output-dir', str(self.output)])

    def rejection(self):
        with self.assertRaises((RuntimeError, FileNotFoundError)): self.main()
        self.assertFalse(self.output.exists())
        self.assertFalse(any(n in sys.modules for n in HEAVY))

    def test_reviewed_actual_input_wip_is_allowed_not_head_only(self):
        self.assertEqual(entry.verify_source(entry.load_source_evidence(self.evidence)), self.expected)
        self.assertTrue(set(entry.EXTRA) <= self.expected['files'].keys())
        self.assertTrue(set(self.sh.TRANSLATION_UNITS) <= self.expected['files'].keys())
        with patch.object(Path, 'mkdir', side_effect=RuntimeError('output_boundary_reached')):
            with self.assertRaisesRegex(RuntimeError, 'output_boundary_reached'): self.main()

    def test_all_new_inputs_missing_modified_and_aliased_reject(self):
        for name in entry.EXTRA:
            p = self.root/name; original = p.read_bytes()
            for change in ('missing', 'changed', 'aliased'):
                with self.subTest(name=name, change=change):
                    p.unlink()
                    if change == 'changed': p.write_bytes(original+b'\n# unexpected\n')
                    if change == 'aliased': p.symlink_to(ROOT/name)
                    self.rejection()
                    if p.exists() or p.is_symlink(): p.unlink()
                    p.write_bytes(original)

    def test_helper_bytes_reject_before_helper_execution(self):
        p = self.root/entry.HELPER; p.write_bytes(p.read_bytes()+b'\n# unexpected helper\n')
        with patch.object(entry, 'helper', side_effect=AssertionError('helper must not execute')):
            self.rejection()

    def test_production_headers_binding_and_oracle_changes_reject(self):
        for name in (self.sh.TRANSLATION_UNITS[2], 'gaussian_renderer/diff_gaussian_rasterization.py',
                     'diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h', 'tests/sh_oracle.py'):
            p = self.root/name; original = p.read_bytes()
            with self.subTest(name=name):
                p.write_bytes(original+b'\n# unexpected\n'); self.rejection(); p.write_bytes(original)

    def test_added_local_input_rejects_and_documents_do_not(self):
        (self.root/'README.md').write_text('unrelated document')
        self.assertEqual(entry.verify_source(self.expected), self.expected)
        (self.root/'utils/unexpected.py').write_text('# input')
        self.rejection()

    def test_context_head_can_change_but_content_cannot(self):
        changed = copy.deepcopy(self.expected)
        head = changed['base_head']; changed['base_head'] = ('0' if head[0] != '0' else '1')+head[1:]
        self.write(changed)
        self.assertEqual(entry.verify_source(entry.load_source_evidence(self.evidence))['files'], changed['files'])
        (self.root/entry.EXTRA[0]).write_text('# changed')
        self.rejection()

    def test_schema_duplicates_missing_extra_and_paths_reject(self):
        self.evidence.write_text('{"schema":1,"schema":2}'); self.rejection()
        values = [dict(self.expected, schema=self.sh.INPUT_SCHEMA), dict(self.expected, extra=1),
                  dict(repository=str(self.root), head=self.actual['base_head'])]
        missing = copy.deepcopy(self.expected); del missing['files'][entry.EXTRA[0]]; values.append(missing)
        for name in ('../escape.py', '/absolute.py', 'tests/../escape.py', './file.py'):
            value = copy.deepcopy(self.expected); value['files'][name] = next(iter(value['files'].values())); values.append(value)
        for value in values:
            with self.subTest(schema=value.get('schema')):
                self.write(value); self.rejection()

    def test_existing_output_overlap_missing_parent_and_alias_reject(self):
        self.output.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'new_unaliased_absolute_output_required'):
            self.main()
        self.assertTrue(self.output.is_dir())
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertFalse(any(n in sys.modules for n in HEAVY))
        self.output.rmdir()
        for p in (self.root/'new', self.base/'outside', self.reports/'missing/new', Path('relative')):
            self.output = p; self.rejection()
        alias = self.reports/'alias'; alias.symlink_to(self.base, target_is_directory=True)
        self.output = alias/'new'; self.rejection()

    def test_record_mode_is_candidate_only_no_heavy_or_output(self):
        stream = io.StringIO()
        with redirect_stdout(stream): entry.main(['--record-source-evidence'])
        self.assertEqual(json.loads(stream.getvalue()), self.expected)
        self.assertFalse(self.output.exists())
        for extra in ('--source-evidence', '--output-dir'):
            with self.assertRaises(SystemExit): entry.main(['--record-source-evidence', extra, str(self.output)])

    def test_sh_subset_uses_reviewed_bytes_and_keeps_old_schema(self):
        subset = entry.sh_subset(self.expected, self.sh)
        self.assertEqual(subset['schema'], self.sh.INPUT_SCHEMA)
        self.assertEqual(self.sh.verify_source(subset), subset)
        self.assertFalse(set(entry.EXTRA) & subset['files'].keys())
        for k, v in subset['files'].items(): self.assertEqual(v, self.expected['files'][k])
        (self.root/entry.EXTRA[0]).write_text('# changed during run')
        with self.assertRaisesRegex(RuntimeError, 'reviewed_input_mismatch'): entry.sh_subset(self.expected, self.sh)

    def test_loaded_alpha_and_sh_helpers_require_reviewed_locations(self):
        module = types.ModuleType('alpha_cap_oracle'); module.__file__ = str(self.root/entry.EXTRA[0])
        with patch.dict(sys.modules, {'alpha_cap_oracle': module}):
            self.assertIn('alpha_cap_oracle', entry.verify_loaded_inputs(self.expected, self.sh))
            module.__file__ = str(self.base/'wrong_oracle.py')
            with self.assertRaisesRegex(RuntimeError, 'unexpected_import_source'):
                entry.verify_loaded_inputs(self.expected, self.sh)
        self.sh.__file__ = str(self.base/'wrong_helper.py')
        with self.assertRaisesRegex(RuntimeError, 'unexpected_helper_source'):
            entry.verify_loaded_inputs(self.expected, self.sh)

    def test_binary_must_be_new_recipe_must_bind_all_translation_units(self):
        build = self.reports/'test-build'; folder = build/'extension'; folder.mkdir(parents=True)
        binary = folder/'synthetic.so'; binary.write_bytes(b'test metadata, not executable')
        recipe = folder/'build.ninja'
        recipe.write_text('\n'.join(str(self.root/p) for p in self.sh.TRANSLATION_UNITS))
        for i in range(5): (folder/f'{i}.o').write_bytes(b'test metadata')
        binding = types.SimpleNamespace(__file__=str(self.root/'gaussian_renderer/diff_gaussian_rasterization.py'),
                                        _C=types.SimpleNamespace(__file__=str(binary)))
        before = entry.binary_identity(binding, build, self.sh)
        self.assertEqual(len(before['objects']), 5)
        binary.write_bytes(b'changed')
        self.assertNotEqual(before, entry.binary_identity(binding, build, self.sh))
        binding._C.__file__ = str(self.base/'old.so')
        with self.assertRaisesRegex(RuntimeError, 'outside_fresh_build'): entry.binary_identity(binding, build, self.sh)
        binding._C.__file__ = str(binary); recipe.write_text('missing source')
        with self.assertRaisesRegex(RuntimeError, 'translation_units_mismatch'): entry.binary_identity(binding, build, self.sh)

    def test_entry_syntax_and_no_top_level_heavy_import_or_execution(self):
        source = (ROOT/entry.EXTRA[2]).read_text()
        tree = ast.parse(source); compile(tree, entry.EXTRA[2], 'exec')
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(a.name.split('.')[0] in HEAVY for a in node.names))
            if isinstance(node, ast.ImportFrom): self.assertNotIn(node.module.split('.')[0], HEAVY)
        self.assertFalse(any(n in sys.modules for n in HEAVY))
        # Static supplements, not GPU mathematical acceptance.
        self.assertIn('sh.run_checks(torch, binding, renderer, regression)', source)
        self.assertIn("regression['case_count'] != 15", source)
        self.assertIn("len(regression['checks']) != 309", source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
