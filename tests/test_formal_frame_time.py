"""Issue 48 CPU partial tests, including the two expressly approved JSON files.

Real fixture access is test-only: one bounded read per file per suite, same
bytes for hashing/decoding, tied to Issue 32 evidence AND Issue 48 preflight.
Missing evidence/data is a failure, never a skip. No image/PLY/camera-source
read, production reader, runnable formal config, or actual run-value adoption.
"""

import ast
import copy
from dataclasses import FrozenInstanceError, fields
from fractions import Fraction
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal_config_json as p2
import formal_config_time as time_part
import formal_frame_time as part


def config(a=2, b=22, d=2):
    # Deliberately partial; later-owner objects do NOT establish full validity.
    value = {
        "schema": "corrected_4dgs_training_config_v1", "run_mode": "from_scratch",
        "dataset": {"kind": "nerf_transforms", "eval": True,
                    "time": {"raw_interval": [a, b], "divisor": d}},
        "model": {"gaussian_dim": 4, "spatial_sh_degree": 3,
                  "temporal_sh_degree": 2, "rot_4d": True,
                  "force_sh_3d": False, "sh_evaluation": "conditional_mean"},
        "renderer": {"compute_cov3D_python": False, "convert_SHs_python": False,
                     "scaling_modifier": 1, "env_map_res": 0,
                     "override_color": None, "temporal_prefilter": "disabled"},
        "initialization": {"time_variance_denominator": 5},
        "optimization": {}, "reporting": {}, "checkpoint": {}, "output": {},
    }
    return p2.parse_json_bytes(json.dumps(value, allow_nan=False).encode())


def record(path, time):
    return {"file_path": path, "time": p2.NumberToken(str(time))}


def small_reference():
    # Independent literal expectation, never derived from a mutated candidate.
    return {"train": [record("./train/A", 2), record("./train/B", 2),
                      record("./train/C", 12)],
            "test": [record("./test/D", 22)]}


def identity_tree(value):
    if type(value) is dict:
        return (id(value), [(k, identity_tree(v)) for k, v in value.items()])
    if type(value) in (list, tuple):
        return (id(value), type(value), [identity_tree(v) for v in value])
    return (id(value), type(value), value)


def mutations():
    """Independent single faults; the separately kept reference never changes."""
    return (
        ("missing_time", lambda c: c['train'][-1].pop('time'), 'frame_missing_input'),
        ("missing_path", lambda c: c['train'][-1].pop('file_path'), 'frame_missing_input'),
        ("bool_time", lambda c: c['train'][-1].update(time=True), 'num_type'),
        ("string_time", lambda c: c['train'][-1].update(time='2'), 'num_type'),
        ("null_time", lambda c: c['train'][-1].update(time=None), 'num_type'),
        ("float_time", lambda c: c['train'][-1].update(time=2.0), 'num_type'),
        ("int_time", lambda c: c['train'][-1].update(time=2), 'num_type'),
        ("num_overflow", lambda c: c['train'][-1].update(time=p2.NumberToken('1e309')), 'num_overflow'),
        ("num_underflow", lambda c: c['train'][-1].update(time=p2.NumberToken('1e-999')), 'num_underflow'),
        ("below", lambda c: c['train'][-1].update(time=p2.NumberToken('-100')), 'frame_time_range'),
        ("above", lambda c: c['train'][-1].update(time=p2.NumberToken('100')), 'frame_time_range'),
        ("time_changed", lambda c: c['train'][-1].update(time=p2.NumberToken('3')), 'frame_correspondence'),
        ("path_bool", lambda c: c['train'][-1].update(file_path=False), 'frame_path_type'),
        ("path_null", lambda c: c['train'][-1].update(file_path=None), 'frame_path_type'),
        ("path_number", lambda c: c['train'][-1].update(file_path=p2.NumberToken('2')), 'frame_path_type'),
        ("path_case", lambda c: c['train'][-1].update(file_path=c['train'][-1]['file_path'].swapcase()), 'frame_correspondence'),
        ("path_spelling", lambda c: c['train'][-1].update(file_path=c['train'][-1]['file_path']+'.png'), 'frame_correspondence'),
        ("path_normalization", lambda c: c['train'][-1].update(file_path='./'+c['train'][-1]['file_path']), 'frame_correspondence'),
        ("missing_record", lambda c: c['train'].pop(), 'frame_count'),
        ("extra_record", lambda c: c['train'].append(record('new', 3)), 'frame_count'),
        ("duplicate", lambda c: c['train'].__setitem__(-1, copy.deepcopy(c['train'][0])), 'frame_duplicate_path'),
        ("replacement", lambda c: c['train'].__setitem__(-1, record('other', 3)), 'frame_correspondence'),
        ("reorder", lambda c: c['train'].__setitem__(slice(0, 2), c['train'][0:2][::-1]), 'frame_correspondence'),
        ("split_move", lambda c: c['test'].append(c['train'].pop()), 'frame_count'),
        ("intersection", lambda c: c['test'].__setitem__(0, copy.deepcopy(c['train'][0])), 'frame_duplicate_path'),
        ("record_type", lambda c: c['train'].__setitem__(-1, []), 'frame_record_type'),
    )


class Assertions:
    def checked(self, parsed, candidate, reference, error=None):
        inputs = [parsed.root if type(parsed) is p2.ParsedJSON else parsed,
                  candidate, reference]
        before, identities = copy.deepcopy(inputs), identity_tree(inputs)
        result = None
        try:
            if error is None:
                result = part.derive_frame_times(parsed, candidate['train'],
                                                candidate['test'], reference)
            else:
                with self.assertRaises(p2.JSONInputError) as caught:
                    part.derive_frame_times(parsed, candidate['train'],
                                            candidate['test'], reference)
                self.assertEqual(caught.exception.code, error)
                self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual(inputs, before)
            self.assertEqual(identity_tree(inputs), identities)
        return result

    def complete_match(self, result, reference, divisor):
        # Exact rational division of already-decoded binary64 inputs, then one
        # rounding to binary64. No production helper supplies expected results.
        for split in ('train', 'test'):
            self.assertEqual(len(getattr(result, split)), len(reference[split]))
            for index, (actual, expected) in enumerate(zip(getattr(result, split), reference[split])):
                raw = float(expected['time'].raw)
                effective = float(Fraction.from_float(raw) / Fraction.from_float(float(divisor)))
                self.assertEqual((actual.split, actual.index, actual.file_path,
                                  actual.raw_time, actual.effective_time),
                                 (split, index, expected['file_path'], raw, effective))


class FrameTests(Assertions, unittest.TestCase):
    def test_full_retention_order_endpoints_multiview_and_one_division(self):
        ref = small_reference()
        result = self.checked(config(), copy.deepcopy(ref), ref)
        self.complete_match(result, ref, 2)
        self.assertEqual([r.effective_time for r in result.train], [1, 1, 6])
        self.assertEqual(result.test[0].effective_time, 11)
        self.assertEqual(result.time.raw_interval, (2, 22))

    def test_d_one_fractional_divisor_and_nonzero_signed_origin(self):
        for a, b, d in ((2, 22, 1), (2, 22, 2.5), (-7, 13, 1.1)):
            with self.subTest(a=a, b=b, d=d):
                ref = {'train': [record('a', a), record('b', a)], 'test': [record('c', b)]}
                self.complete_match(self.checked(config(a, b, d), copy.deepcopy(ref), ref), ref, d)

    def test_independent_negative_matrix(self):
        for name, mutate, error in mutations():
            with self.subTest(name=name):
                ref = small_reference()
                candidate = copy.deepcopy(ref)
                mutate(candidate)
                self.checked(config(), candidate, ref, error)

    def test_same_count_cross_split_swap(self):
        ref = small_reference()
        candidate = copy.deepcopy(ref)
        candidate['train'][0], candidate['test'][0] = candidate['test'][0], candidate['train'][0]
        self.checked(config(), candidate, ref, 'frame_correspondence')

    def test_nonzero_to_zero_rejected_no_rounding_repair(self):
        ref = {'train': [record('a', '5e-324')], 'test': [record('b', 1)]}
        self.checked(config(0, 1, 2), copy.deepcopy(ref), ref, 'frame_time_conversion')

    def test_positive_subnormal_is_not_rejected_merely_for_being_small(self):
        ref = {'train': [record('a', '1e-323')], 'test': [record('b', 1)]}
        result = self.checked(config(0, 1, 2), copy.deepcopy(ref), ref)
        self.assertEqual(result.train[0].effective_time, math.ulp(0.0))

    def test_signed_true_zero_and_num_spelling(self):
        ref = {'train': [record('a', '-0.0')], 'test': [record('b', 1)]}
        candidate = copy.deepcopy(ref)
        candidate['test'][0]['time'] = p2.NumberToken('1.0e0')
        result = self.checked(config(0, 1, 2), candidate, ref)
        self.assertEqual(math.copysign(1, result.train[0].effective_time), -1)

    def test_array_and_reference_shapes(self):
        ref = small_reference()
        for bad in (None, (), {}, True):
            with self.subTest(bad=type(bad).__name__):
                candidate = copy.deepcopy(ref)
                candidate['train'] = bad
                self.checked(config(), candidate, ref, 'frame_array_type')
        for bad in (None, [], {'train': []}, dict(ref, extra=[])):
            with self.subTest(reference=type(bad).__name__):
                self.checked(config(), copy.deepcopy(ref), bad, 'frame_reference_shape')
        bad = copy.deepcopy(ref)
        bad['test'] = tuple(bad['test'])
        self.checked(config(), copy.deepcopy(ref), bad, 'frame_array_type')

    def test_reference_is_also_checked_not_a_trusted_typed_token(self):
        original = small_reference()
        for value, error in ((None, 'num_type'), (p2.NumberToken('100'), 'frame_time_range')):
            ref = copy.deepcopy(original)
            ref['test'][0]['time'] = value
            self.checked(config(), copy.deepcopy(original), ref, error)
        ref = copy.deepcopy(original)
        ref['test'][0] = copy.deepcopy(ref['train'][0])
        self.checked(config(), copy.deepcopy(original), ref, 'frame_duplicate_path')

    def test_config_rechecked_each_call_and_time_math_reused(self):
        ref, parsed = small_reference(), config()
        with patch.object(time_part, 'derive_time_interval_and_variance',
                          wraps=time_part.derive_time_interval_and_variance) as derive:
            result = self.checked(parsed, copy.deepcopy(ref), ref)
            derive.assert_called_once_with(parsed)
        self.assertEqual(result.time, time_part.derive_time_interval_and_variance(parsed))
        parsed.root['dataset']['time']['divisor'] = p2.NumberToken('0')
        self.checked(parsed, copy.deepcopy(ref), ref, 'time_divisor_domain')
        parsed = config()
        parsed.root['dataset']['eval'] = False
        self.checked(parsed, copy.deepcopy(ref), ref, 'p1_fixed_value')
        parsed = config()
        del parsed.root['dataset']['time']['raw_interval']
        self.checked(parsed, copy.deepcopy(ref), ref, 'time_missing_input')

    def test_derived_results_are_not_raw_or_verified_inputs(self):
        ref = small_reference()
        result = self.checked(config(), copy.deepcopy(ref), ref)
        self.checked(result.time, copy.deepcopy(ref), ref, 'p1_parsed_input')
        self.checked(config(), {'train': result.train, 'test': result.test}, ref, 'frame_array_type')
        candidate = copy.deepcopy(ref)
        for split in ('train', 'test'):
            for frame, derived in zip(candidate[split], getattr(result, split)):
                frame['time'] = p2.NumberToken(str(derived.effective_time))
        self.checked(config(), candidate, ref, 'frame_time_range')

    def test_result_is_immutable_and_detached(self):
        ref, parsed = small_reference(), config()
        candidate = copy.deepcopy(ref)
        result = self.checked(parsed, candidate, ref)
        saved = copy.deepcopy(result)
        for obj in (result, result.time, result.train[0]):
            for field in fields(obj):
                with self.assertRaises(FrozenInstanceError):
                    setattr(obj, field.name, None)
        with self.assertRaises(TypeError):
            result.train[0] = result.test[0]
        candidate['train'].clear()
        ref['test'][0]['time'] = p2.NumberToken('3')
        parsed.root.clear()
        self.assertEqual(result, saved)

    def test_unknown_camera_metadata_not_consumed_or_certified(self):
        ref = small_reference()
        candidate = copy.deepcopy(ref)
        candidate['train'][0]['unknown_camera_field'] = {'time': None}
        self.complete_match(self.checked(config(), candidate, ref), ref, 2)

    def test_late_error_returns_no_partial_result_or_generator(self):
        self.assertFalse(inspect.isgeneratorfunction(part.derive_frame_times))
        ref = small_reference()
        candidate = copy.deepcopy(ref)
        candidate['test'][-1].pop('time')
        with patch.object(part, 'FrameTimes', wraps=part.FrameTimes) as constructor:
            self.checked(config(), candidate, ref, 'frame_missing_input')
            constructor.assert_not_called()

    def test_nonfinite_conversion_guard_fault_injection(self):
        # Finite raw and d>=1 cannot naturally overflow; test the final guard,
        # without claiming a valid-input runtime overflow was observed.
        ref = small_reference()
        with patch.object(part, 'math') as math_guard:
            math_guard.isfinite.return_value = False
            self.checked(config(), copy.deepcopy(ref), ref, 'frame_time_conversion')


def decode_metadata(data):
    """Limited fixture decoder only. Never route metadata through P2's reader."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_metadata_key')
            result[key] = value
        return result
    def nonfinite(unused):
        raise ValueError('nonstandard_metadata_number')
    return json.loads(data.decode('utf-8', errors='strict'), object_pairs_hook=pairs,
                      parse_int=p2.NumberToken, parse_float=p2.NumberToken,
                      parse_constant=nonfinite)


def stat_identity(value):
    return {name: getattr(value, name) for name in (
        'st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}


class MetadataDecoderTests(unittest.TestCase):
    def test_token_lexemes_and_duplicate_nonfinite_rejection(self):
        value = decode_metadata(b'{"time":1.0e1,"integer":2}')
        self.assertEqual(value['time'].raw, '1.0e1')
        self.assertEqual(value['integer'].raw, '2')
        for raw in (b'{"time":1,"ti\\u006de":2}', b'{"time":NaN}',
                    b'{"time":Infinity}', b'{"time":-Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_metadata(raw)


class RealMetadataTests(Assertions, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        reports = ROOT.parent / 'reports/corrected-4dgs'
        preflight = json.loads((reports/'issue-48/issue-48-preflight.json').read_text())
        evidence = json.loads((reports/'issue-32/issue-32-validation-evidence.json').read_text())['transforms']
        if evidence['status'] != 'pass' or not evidence['frame_path_split_time_order_match']:
            raise AssertionError('reference evidence not successful')
        pre = {item['path']: item for item in preflight['metadata_reference_files']}
        historic = {item['path']: item for item in evidence['files']}
        cls.original, cls.reference, cls.read_evidence = {}, {}, []
        for split, count in (('train', 5146), ('test', 166)):
            # Only these two real data paths may be opened. Never follow paths
            # from the larger historical evidence to images/PLY/J-drive files.
            path = ROOT.parent / ('data/4dgs_sph_scene/transforms_'+split+'.json')
            expected, prior = pre[str(path)], historic[str(path)]
            if (expected['sha256'] != prior['sha256'] or
                    expected['bytes'] != prior['bytes_read'] or expected['frames'] != count):
                raise AssertionError('reference authorities disagree')
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_size > 16 * 1024 * 1024:
                raise AssertionError('metadata file type/size')
            with path.open('rb') as stream:
                fd_before = os.fstat(stream.fileno())
                raw = stream.read(16 * 1024 * 1024 + 1)  # exactly one read/file
                fd_after = os.fstat(stream.fileno())
            after = path.lstat()
            identities = [stat_identity(s) for s in (before, fd_before, fd_after, after)]
            if any(s != identities[0] for s in identities[1:]):
                raise AssertionError('metadata changed during read')
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != expected['bytes'] or digest != expected['sha256']:
                raise AssertionError('metadata content identity mismatch')
            decoded = decode_metadata(raw)
            frames = decoded['frames']
            if type(frames) is not list or len(frames) != count:
                raise AssertionError('metadata count/type')
            cls.original[split] = frames
            # Preserve the approved mapping before ANY candidate exists. Only
            # consumed fields are copied; image/camera metadata is not certified.
            cls.reference[split] = copy.deepcopy([
                {'file_path': f['file_path'], 'time': f['time']} for f in frames])
            cls.read_evidence.append(dict(path=str(path), bytes=len(raw), sha256=digest,
                                          frames=count, reads=1, stat=identities[0],
                                          issue32_match=True, preflight_match=True))
        # Also compare every expected raw time/index against Issue 32's mapping,
        # without recomputing time from filenames, FPS or source camera indices.
        visited = {'train': set(), 'test': set()}
        for mapping in evidence['frames']:
            for split in ('train', 'test'):
                for index in mapping[split+'_output_indices']:
                    if index in visited[split]:
                        raise AssertionError('reference index duplicated')
                    visited[split].add(index)
                    if float(cls.reference[split][index]['time'].raw) != mapping['time_binary64']:
                        raise AssertionError('reference raw time differs from Issue 32')
        for split in ('train', 'test'):
            if visited[split] != set(range(len(cls.reference[split]))):
                raise AssertionError('reference index coverage incomplete')
        cls.reference_before = copy.deepcopy(cls.reference)
        cls.reference_identity = identity_tree(cls.reference)

    @classmethod
    def tearDownClass(cls):
        if cls.reference != cls.reference_before or identity_tree(cls.reference) != cls.reference_identity:
            raise AssertionError('reference changed')
        for item in cls.read_evidence:
            if stat_identity(Path(item['path']).lstat()) != item['stat']:
                raise AssertionError('metadata changed after suite')
        print('METADATA_EVIDENCE '+json.dumps(cls.read_evidence, sort_keys=True))

    def test_all_5312_d_one_and_d_two_with_independent_full_correspondence(self):
        for d in (1, 2):
            with self.subTest(divisor=d):
                candidate = copy.deepcopy(self.original)
                result = self.checked(config(0, 33, d), candidate, self.reference)
                self.complete_match(result, self.reference, d)
                self.assertEqual((len(result.train), len(result.test)), (5146, 166))
                self.assertEqual(sum(r.raw_time == 0 for r in result.train+result.test), 32)
                self.assertEqual(sum(r.raw_time == 33 for r in result.train+result.test), 32)
                self.assertEqual(len({r.raw_time for r in result.train+result.test}), 166)
                self.assertEqual(len({r.file_path for r in result.train+result.test}), 5312)

    def test_all_records_negative_matrix_reference_unchanged(self):
        for name, mutate, error in mutations():
            with self.subTest(name=name):
                # All 5312 consumed correspondences, not a small sampled subset.
                candidate = copy.deepcopy(self.reference)
                mutate(candidate)
                self.checked(config(0, 33, 2), candidate, self.reference, error)

    def test_declared_closed_interval_does_not_filter_real_endpoints(self):
        self.checked(config(0.1, 33, 2), copy.deepcopy(self.reference),
                     self.reference, 'frame_time_range')
        self.checked(config(0, 32.9, 2), copy.deepcopy(self.reference),
                     self.reference, 'frame_time_range')

    def test_same_count_split_swap_rejected(self):
        candidate = copy.deepcopy(self.reference)
        candidate['train'][0], candidate['test'][0] = candidate['test'][0], candidate['train'][0]
        self.checked(config(0, 33, 2), candidate, self.reference, 'frame_correspondence')


class IsolationTests(unittest.TestCase):
    def test_import_allowlist_and_no_yield_or_io_in_production(self):
        for path in (ROOT/'formal_frame_time.py', Path(__file__)):
            tree = ast.parse(path.read_text())
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_time', 'formal_frame_time'})
            if path.name == 'formal_frame_time.py':
                self.assertEqual(names, {'dataclasses', 'math', 'formal_config_json', 'formal_config_time'})
                self.assertFalse(any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(tree)))
                calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
                self.assertTrue(calls.isdisjoint({'open', 'eval', 'exec', '__import__'}))

    def test_isolated_success_and_late_rejection_without_heavy_import_or_writer(self):
        script = r'''
import sys, os
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_time', 'formal_frame_time'}:
            raise AssertionError('non-stdlib import attempted')
sys.meta_path.insert(0, OnlyStdlib())
def audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (mode and any(c in mode for c in 'wax+')) or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('writer attempted')
    if event.startswith(('subprocess.', 'socket.')) or event in (
            'os.mkdir', 'os.remove', 'os.rename', 'os.system', 'os.rmdir', 'ctypes.dlopen'):
        raise AssertionError('runtime side effect attempted')
sys.addaudithook(audit)
import formal_config_json as p2
import formal_frame_time as part
parsed = p2.parse_json_bytes(sys.argv[2].encode())
candidate = {'train':[{'file_path':'a','time':p2.NumberToken('2')}],
             'test':[{'file_path':'b','time':p2.NumberToken('22')}]}
reference = {'train':[{'file_path':'a','time':p2.NumberToken('2')}],
             'test':[{'file_path':'b','time':p2.NumberToken('22')}]}
result = part.derive_frame_times(parsed, candidate['train'], candidate['test'], reference)
assert result.train[0].effective_time == 1 and result.test[0].effective_time == 11
candidate['test'][0].pop('time')
try:
    part.derive_frame_times(parsed, candidate['train'], candidate['test'], reference)
except p2.JSONInputError as error:
    assert error.code == 'frame_missing_input'
else:
    raise AssertionError('late error accepted')
assert not any(x.split('.')[0] in {'torch','scene','gaussian_renderer','train'} for x in sys.modules)
print('pure-success-and-rejection')
'''
        raw = json.dumps(config().root, default=lambda n: int(n.raw) if n.is_integer else float(n.raw))
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            run = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', script, str(ROOT), raw],
                                 cwd=directory, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), 'pure-success-and-rejection')
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
