"""CPU-only Step 8 input-evidence regression, with real temporary files.

No temporary Git repository, renderer import, torch import or build. Git's
base revision alone is mocked; content enumeration and hashes are real.
"""
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
ENTRY = ROOT/'tests/cuda/sh_cuda_acceptance.py'
HEAVY = ('torch', 'numpy', 'gaussian_renderer', 'scene', 'pointops2', 'simple_knn')


def blocked_import(name, *args, **kwargs):
    if name.split('.')[0] in HEAVY:
        raise AssertionError('heavy import attempted: '+name)
    return ORIGINAL_IMPORT(name, *args, **kwargs)


ORIGINAL_IMPORT = builtins.__import__
# Guard collection/import itself, not just test execution.
with patch('builtins.__import__', side_effect=blocked_import):
    spec = importlib.util.spec_from_file_location('sh_cuda_acceptance_cpu', ENTRY)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.executable != entry.PYTHON or not sys.dont_write_bytecode:
            raise RuntimeError('explicit_4dgs310_B_required')
        print('executable:', sys.executable)
        cls.guard = patch('builtins.__import__', side_effect=blocked_import)
        cls.guard.start()
        cls.addClassCleanup(cls.guard.stop)
        cls.actual = entry.source_identity()
        print('actual input count:', len(cls.actual['files']))

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='step8-input-test-')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base/'research'
        self.root.mkdir()
        for name in self.actual['files']:
            target = self.root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, target)
        root_patch = patch.object(entry, 'ROOT', self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)
        def git(*args):
            self.assertEqual(args, ('rev-parse','HEAD'))  # No status/ls-files gate.
            return self.actual['base_head']
        git_patch = patch.object(entry, 'git', side_effect=git)
        git_patch.start(); self.addCleanup(git_patch.stop)
        self.expected = dict(self.actual, repository=str(self.root))
        self.evidence = self.base/'reviewed.json'
        self.write_evidence(self.expected)
        self.output = self.base/'new-output'

    def write_evidence(self, value):
        self.evidence.write_text(json.dumps(value))

    def main(self, *extra):
        argv = ['sh_cuda_acceptance.py', '--source-evidence',str(self.evidence),
                '--output-dir',str(self.output), *extra]
        with patch.object(sys,'argv',argv): entry.main()

    def assert_prebuild_rejection(self):
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.main()
        self.assertFalse(self.output.exists())
        self.assertFalse(any(name in sys.modules for name in HEAVY))

    def test_reviewed_wip_and_new_files_pass_without_git_tracking(self):
        # The actual approved starting tree contains changed CUDA and new
        # untracked tests; neither presence nor Git metadata substitutes bytes.
        checked = entry.verify_source(entry.load_source_evidence(self.evidence))
        self.assertEqual(checked, self.expected)
        for name in ('tests/sh_oracle.py','tests/test_sh_cpu.py','tests/cuda/sh_cuda_acceptance.py'):
            self.assertIn(name, checked['files'])
        self.assertEqual(set(entry.TRANSLATION_UNITS) - checked['files'].keys(),set())
        with patch.object(Path,'mkdir',side_effect=RuntimeError('reached_prebuild_output_boundary')):
            with self.assertRaisesRegex(RuntimeError,'reached_prebuild_output_boundary'):
                self.main()

    def test_same_head_changed_cuda_rejects_before_build(self):
        p = self.root/entry.TRANSLATION_UNITS[1]
        p.write_bytes(p.read_bytes()+b'\n// changed after review\n')
        self.assert_prebuild_rejection()

    def test_required_new_inputs_missing_or_changed_reject(self):
        for name in ('tests/sh_oracle.py','tests/test_sh_cpu.py','tests/cuda/sh_cuda_acceptance.py'):
            p=self.root/name; original=p.read_bytes()
            with self.subTest(name=name,change='missing'):
                p.unlink(); self.assert_prebuild_rejection(); p.write_bytes(original)
            with self.subTest(name=name,change='bytes'):
                p.write_bytes(original+b'\n# changed\n')
                self.assert_prebuild_rejection(); p.write_bytes(original)

    def test_headers_glm_and_binding_are_bound(self):
        for name in ('diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h',
                     'diff-gaussian-rasterization/third_party/glm/glm/detail/type_vec3.inl',
                     'gaussian_renderer/diff_gaussian_rasterization.py'):
            p=self.root/name; original=p.read_bytes()
            with self.subTest(name=name):
                p.write_bytes(original+b'\n// modified input\n')
                self.assert_prebuild_rejection(); p.write_bytes(original)

    def test_old_head_only_and_incomplete_evidence_reject(self):
        candidates=[dict(repository=str(self.root),head=self.actual['base_head'])]
        missing=copy.deepcopy(self.expected)
        del missing['files']['tests/sh_oracle.py']
        candidates.append(missing)
        for candidate in candidates:
            with self.subTest(keys=list(candidate)):
                self.write_evidence(candidate); self.assert_prebuild_rejection()

    def test_new_local_input_is_not_silently_ignored(self):
        (self.root/'utils/new_module.py').write_text('# unexpected input\n')
        self.assert_prebuild_rejection()

    def test_before_after_change_detected_but_docs_ignored(self):
        before=entry.verify_source(self.expected)
        (self.root/'README.md').write_text('unrelated documentation\n')
        (self.root/'docs').mkdir()
        (self.root/'docs/plan.md').write_text('reviewed document WIP\n')
        self.assertEqual(entry.verify_source(before),before)
        (self.root/'scene/cameras.py').write_text('# changed during execution\n')
        with self.assertRaisesRegex(RuntimeError,'source_inputs_mismatch'):
            entry.verify_source(before)

    def test_base_head_is_context_not_content(self):
        different=copy.deepcopy(self.expected)
        head=different['base_head']
        different['base_head']=('0' if head[0]!='0' else '1')+head[1:]  # Synthetic test only.
        self.write_evidence(different)
        self.assertEqual(entry.verify_source(entry.load_source_evidence(self.evidence))['files'],different['files'])

    def test_aliased_input_rejects(self):
        p=self.root/'tests/sh_oracle.py'; p.unlink(); p.symlink_to(ROOT/'tests/sh_oracle.py')
        self.assert_prebuild_rejection()

    def test_evidence_schema_duplicates_and_paths_reject(self):
        self.evidence.write_text('{"schema":1,"schema":2}')
        self.assert_prebuild_rejection()
        for name in ('../escape.py','/absolute.py','tests/../oracle.py'):
            value=copy.deepcopy(self.expected)
            value['files'][name]=next(iter(value['files'].values()))
            self.write_evidence(value); self.assert_prebuild_rejection()
        value=copy.deepcopy(self.expected); value['files']['tests/sh_oracle.py']='not-a-digest'
        self.write_evidence(value); self.assert_prebuild_rejection()

    def test_loaded_local_module_must_match_reviewed_path(self):
        module=types.ModuleType('sh_oracle'); module.__file__=str(self.root/'tests/sh_oracle.py')
        with patch.dict(sys.modules,{'sh_oracle':module}):
            self.assertIn('sh_oracle',entry.verify_loaded_inputs(self.expected))
            module.__file__=str(self.base/'other/sh_oracle.py')
            with self.assertRaisesRegex(RuntimeError,'unexpected_import_source'):
                entry.verify_loaded_inputs(self.expected)

    def test_record_mode_is_lightweight_and_not_execution(self):
        stream=io.StringIO()
        with patch.object(sys,'argv',['entry','--record-source-evidence']), redirect_stdout(stream):
            entry.main()
        self.assertEqual(json.loads(stream.getvalue()),self.expected)
        self.assertFalse(self.output.exists())
        self.assertFalse(any(name in sys.modules for name in HEAVY))
        with patch.object(sys,'argv',['entry','--record-source-evidence','--output-dir',str(self.output)]):
            with self.assertRaises(SystemExit): entry.main()


if __name__ == '__main__':
    unittest.main(verbosity=2)
