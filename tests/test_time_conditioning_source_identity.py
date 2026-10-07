"""CPU-only input/preflight/ABI checks. No compiler, heavy import or CUDA."""
import ast
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_IMPORT = builtins.__import__
HEAVY = ('torch','numpy','PIL','train','scene','gaussian_renderer','simple_knn','pointops2')


def deny(name, *args, **kwargs):
    if name.split('.')[0] in HEAVY: raise AssertionError('heavy_import: '+name)
    return ORIGINAL_IMPORT(name,*args,**kwargs)


spec = importlib.util.spec_from_file_location('time_entry_cpu',ROOT/'tests/cuda/time_conditioning_cuda_acceptance.py')
entry = importlib.util.module_from_spec(spec)
with patch('builtins.__import__',side_effect=deny): spec.loader.exec_module(entry)


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.executable != entry.PYTHON or not sys.dont_write_bytecode: raise RuntimeError('4dgs310_B_required')
        cls.guard = patch('builtins.__import__',side_effect=deny); cls.guard.start(); cls.addClassCleanup(cls.guard.stop)
        cls.actual = entry.source_identity()
        print('executable:',sys.executable,'time input count:',len(cls.actual['files']))

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.base = Path(temp.name)
        self.root = self.base/'research'; self.root.mkdir()
        for name in self.actual['files']:
            p = self.root/name; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/name,p)
        self.reports = self.base/'reports'; self.reports.mkdir(); self.output = self.reports/'new-run'
        for name,value in (('ROOT',self.root),('REPORTS',self.reports)):
            guard = patch.object(entry,name,value); guard.start(); self.addCleanup(guard.stop)
        camera = entry.helper(); alpha = camera.helper(); sh = alpha.helper()
        sh.git = lambda *args: self.actual['base_head']
        for owner,value in ((alpha,sh),(camera,alpha),(entry,camera)):
            guard = patch.object(owner,'helper',return_value=value); guard.start(); self.addCleanup(guard.stop)
        self.expected = dict(self.actual,repository=str(self.root))
        self.evidence = self.base/'inputs.json'; self.evidence.write_text(json.dumps(self.expected))

    def test_actual_wip_new_source_and_knn_bound_without_build(self):
        actual,out = entry.preflight(entry.load_source_evidence(self.evidence),self.output)
        self.assertEqual(actual,self.expected); self.assertEqual(out,self.output)
        self.assertFalse(out.exists())
        self.assertTrue(set(entry.KNN+entry.EXTRA) <= actual['files'].keys())
        self.assertIn('formal_time_handoff.py',actual['files'])
        self.assertFalse(any(n in sys.modules for n in HEAVY))

    def test_every_time_and_knn_input_changed_missing_or_alias_rejected(self):
        for name in entry.EXTRA+entry.KNN+('formal_time_handoff.py',):
            p = self.root/name; original = p.read_bytes()
            for kind in ('changed','missing','alias'):
                with self.subTest(name=name,kind=kind):
                    p.unlink()
                    if kind == 'changed': p.write_bytes(original+b'\n# changed\n')
                    if kind == 'alias': p.symlink_to(ROOT/name)
                    with self.assertRaises((RuntimeError,FileNotFoundError)): entry.preflight(self.expected,self.output)
                    self.assertFalse(self.output.exists())
                    if p.exists() or p.is_symlink(): p.unlink()
                    p.write_bytes(original)

    def test_evidence_rejects_schema_omission_unknown_and_duplicate(self):
        for mutation in ('schema','missing','unknown'):
            v = copy.deepcopy(self.expected)
            if mutation == 'schema': v['schema'] = 'old'
            if mutation == 'missing': del v['files'][entry.KNN[0]]
            if mutation == 'unknown': v['unknown'] = True
            self.evidence.write_text(json.dumps(v))
            with self.assertRaises(RuntimeError): entry.load_source_evidence(self.evidence)
        self.evidence.write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(RuntimeError,'duplicate'): entry.load_source_evidence(self.evidence)

    def test_no_output_on_wrong_locator_or_external_override(self):
        for out in (Path('relative'),self.root/'outside',self.reports):
            with self.assertRaises(RuntimeError): entry.preflight(self.expected,out)
        with patch.dict('os.environ',{'STEP90_CUDA_DEBUG_PIXEL':'1,1'}):
            with self.assertRaisesRegex(RuntimeError,'override'): entry.preflight(self.expected,self.output)
        self.assertFalse(self.output.exists())

    def test_entry_syntax_and_bounded_search_no_old_main_calls(self):
        tree = ast.parse((ROOT/entry.EXTRA[-1]).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(ast.unparse(node.func),('camera.main','alpha.main','sh.main'))
        text = (ROOT/entry.EXTRA[-1]).read_text()
        self.assertIn("'build_ext', '--build-lib'",text)
        self.assertIn("'--build-temp'",text)
        self.assertNotIn('pip install',text)
        self.assertIn("'simple_knn._C' in sys.modules",text)
        for path in (ROOT/'tests').glob('time_conditioning*.py'): ast.parse(path.read_text())


class SignatureTests(unittest.TestCase):
    def test_native_declarations_match_definitions(self):
        native = ROOT/'diff-gaussian-rasterization'
        pairs = [('rasterize_points.h','rasterize_points.cu','RasterizeGaussiansCUDA','RasterizeGaussiansCUDA'),
                 ('rasterize_points.h','rasterize_points.cu','RasterizeGaussiansBackwardCUDA','RasterizeGaussiansBackwardCUDA'),
                 ('cuda_rasterizer/forward.h','cuda_rasterizer/forward.cu','preprocess','FORWARD::preprocess'),
                 ('cuda_rasterizer/backward.h','cuda_rasterizer/backward.cu','preprocess','BACKWARD::preprocess')]
        def parameters(text,name):
            start = text.index(name+'(')+len(name)+1; depth = 1; end = start
            while depth:
                if text[end] == '(': depth += 1
                if text[end] == ')': depth -= 1
                end += 1
            return [re.sub(r'\s+',' ',s.strip()) for s in text[start:end-1].split(',')]
        for header,source,declaration,definition in pairs:
            a = parameters((native/header).read_text(),declaration)
            b = parameters((native/source).read_text(),definition)
            # Parameter names already differ in upstream declarations. ABI is
            # the ordered types, not local identifier spelling.
            types_only = lambda items: [re.sub(r'[A-Za-z_][A-Za-z_0-9]*$', '', x).strip() for x in items]
            with self.subTest(source=source): self.assertEqual(types_only(a),types_only(b))
        tree = ast.parse((ROOT/'gaussian_renderer/diff_gaussian_rasterization.py').read_text())
        counts = [len(node.value.elts) for node in ast.walk(tree) if isinstance(node,ast.Assign)
                  and isinstance(node.value,ast.Tuple) and any(isinstance(t,ast.Name) and t.id == 'args' for t in node.targets)]
        self.assertEqual(counts,[35,40])
        self.assertEqual(len(parameters((native/'rasterize_points.h').read_text(),'RasterizeGaussiansCUDA')),35)
        self.assertEqual(len(parameters((native/'rasterize_points.h').read_text(),'RasterizeGaussiansBackwardCUDA')),40)


if __name__ == '__main__': unittest.main()
