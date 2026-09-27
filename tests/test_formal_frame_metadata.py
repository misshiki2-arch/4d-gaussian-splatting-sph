"""CPU partial reader tests. Temporary fixture writes are outside reader calls.

Only the two explicitly authorized real transforms are read. Real references
are hash-bound to Issue 32 and Issue 49 before candidate use. No skip fallback,
image/PLY access, package dependencies, full config or dataset authorization.
"""

import ast
import copy
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal_config_json as p2
import formal_config_p6 as p6
import formal_frame_time as frame
import formal_frame_metadata as part


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(',', ':')).encode()


def config(source, output, a=2, b=22, d=2):
    return p2.parse_json_bytes(encode({
        'schema': 'corrected_4dgs_training_config_v1', 'run_mode': 'from_scratch',
        'dataset': {'kind': 'nerf_transforms', 'eval': True, 'source_path': str(source),
                    'time': {'raw_interval': [a, b], 'divisor': d}},
        'model': {'gaussian_dim': 4, 'spatial_sh_degree': 3, 'temporal_sh_degree': 2,
                  'rot_4d': True, 'force_sh_3d': False, 'sh_evaluation': 'conditional_mean'},
        'renderer': {'compute_cov3D_python': False, 'convert_SHs_python': False,
                     'scaling_modifier': 1, 'env_map_res': 0, 'override_color': None,
                     'temporal_prefilter': 'disabled'},
        'initialization': {'time_variance_denominator': 5},
        'optimization': {}, 'reporting': {}, 'checkpoint': {},
        'output': {'directory': str(output)},
    }))


def identity(data):
    return part.FileIdentity(len(data), hashlib.sha256(data).hexdigest())


def identities(value):
    if is_dataclass(value):
        return (id(value), type(value), [(f.name, identities(getattr(value, f.name))) for f in fields(value)])
    if type(value) is dict:
        return (id(value), [(k, identities(v)) for k, v in value.items()])
    if type(value) in (tuple, list):
        return (id(value), type(value), [identities(v) for v in value])
    return (id(value), type(value), value)


def stat_copy(s, **changes):
    values = {name: getattr(s, name) for name in (
        'st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_size', 'st_mtime_ns', 'st_ctime_ns')}
    values.update(changes)
    return SimpleNamespace(**values)


class Assertions:
    def checked(self, parsed, reference, error=None):
        before = copy.deepcopy((parsed, reference))
        before_ids = identities((parsed, reference))[2]
        try:
            if error is None:
                return part.read_frame_metadata(parsed, reference)
            with self.assertRaises(p2.JSONInputError) as caught:
                part.read_frame_metadata(parsed, reference)
            self.assertEqual(caught.exception.code, error)
            self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual((parsed, reference), before)
            self.assertEqual(identities((parsed, reference))[2], before_ids)

    def full_match(self, result, reference, divisor):
        for split in ('train', 'test'):
            self.assertEqual(len(getattr(result, split)), len(reference.frames[split]))
            for index, (actual, expected) in enumerate(zip(getattr(result, split), reference.frames[split])):
                raw = float(expected['time'].raw)
                effective = float(Fraction.from_float(raw)/Fraction.from_float(float(divisor)))
                self.assertEqual((actual.split, actual.index, actual.file_path, actual.raw_time, actual.effective_time),
                                 (split, index, expected['file_path'], raw, effective))


class ReaderTests(Assertions, unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.source = self.base/'data'
        self.source.mkdir()
        self.output = self.base/'unused-output'
        self.parsed = config(self.source, self.output)
        self.documents = {
            'train': {'unknown_camera': None, 'frames': [
                {'file_path': './a', 'time': 2}, {'file_path': './b', 'time': 2},
                {'file_path': './c', 'time': 12}]},
            'test': {'frames': [{'file_path': './d', 'time': 20}, {'file_path': './e', 'time': 22}]},
        }
        raw = {s: encode(v) for s, v in self.documents.items()}
        # Explicit fixture approval before any negative candidate modification.
        mapping = {s: [{'file_path': f['file_path'], 'time': p2.NumberToken(str(f['time']))}
                       for f in v['frames']] for s, v in self.documents.items()}
        self.reference = part.MetadataReference(identity(raw['train']), identity(raw['test']), mapping)
        for s, data in raw.items():
            self.path(s).write_bytes(data)

    def path(self, split):
        return self.source/('transforms_'+split+'.json')

    def fixture_reference(self, split, data):
        # Deliberately approve bytes ONLY to test later decoder/frame failures.
        # The independent original correspondence is never regenerated.
        self.path(split).write_bytes(data)
        return replace(self.reference, **{split: identity(data)})

    def test_pipeline_once_per_file_no_output_and_immutable_detached_result(self):
        original_open, original_read = os.open, os.read
        with patch.object(part.os, 'open', wraps=original_open) as opens, \
                patch.object(part.os, 'read', wraps=original_read) as reads, \
                patch.object(p6, 'preflight_p6_paths', wraps=p6.preflight_p6_paths) as preflight, \
                patch.object(frame, 'derive_frame_times', wraps=frame.derive_frame_times) as handoff:
            result = self.checked(self.parsed, self.reference)
            self.assertEqual([c.args[0] for c in opens.call_args_list],
                             [str(self.source), 'transforms_train.json', 'transforms_test.json'])
            self.assertEqual([c.args[1] for c in reads.call_args_list],
                             [self.reference.train.size+1, self.reference.test.size+1])
            self.assertEqual(handoff.call_count, 1)
            preflight.assert_called_once_with(self.parsed)
        self.full_match(result, self.reference, 2)
        self.assertFalse(self.output.exists())
        with self.assertRaises(FrozenInstanceError):
            result.train[0].raw_time = 0
        saved = copy.deepcopy(result)
        self.reference.frames['train'].clear()
        self.parsed.root.clear()
        self.assertEqual(result, saved)

    def test_p6_config_failure_before_any_open_and_rechecked(self):
        self.checked(self.parsed, self.reference)
        self.parsed.root['dataset']['eval'] = False
        with patch.object(part.os, 'open') as opens:
            self.checked(self.parsed, self.reference, 'p1_fixed_value')
            opens.assert_not_called()
        self.parsed = config(self.source, self.output)
        self.parsed.root['output']['directory'] = str(self.source)
        with patch.object(part.os, 'open') as opens:
            self.checked(self.parsed, self.reference, 'p6_output_occupied')
            opens.assert_not_called()
        paths = p6.P6PathsPartial(str(self.source), str(self.output))
        self.checked(paths, self.reference, 'p1_parsed_input')

    def test_p6_source_symlink_resolution_retained(self):
        alias = self.base/'alias'
        alias.symlink_to(self.source, target_is_directory=True)
        self.full_match(self.checked(config(alias, self.output), self.reference), self.reference, 2)

    def test_reference_type_missing_fields_size_and_digest_format_before_open(self):
        cases = [(None, 'metadata_reference_type'), ({}, 'metadata_reference_type'),
                 (replace(self.reference, train=None), 'metadata_identity_type'),
                 (replace(self.reference, frames={}), 'metadata_reference_frames'),
                 (replace(self.reference, frames={'train': [], 'test': ()}), 'metadata_reference_frames')]
        for bad in (True, 0, -1, 1.0, '10', None):
            cases.append((replace(self.reference, train=part.FileIdentity(bad, self.reference.train.sha256)),
                          'metadata_reference_size'))
        for bad in (None, b'0'*64, '0'*63, 'g'*64, self.reference.train.sha256.upper()):
            cases.append((replace(self.reference, train=part.FileIdentity(self.reference.train.size, bad)),
                          'metadata_reference_hash'))
        for ref, error in cases:
            with self.subTest(error=error), patch.object(part.os, 'open') as opens:
                self.checked(self.parsed, ref, error)
                opens.assert_not_called()

    def test_missing_file_no_fallback(self):
        self.path('train').rename(self.source/'transforms_val.json')
        self.checked(self.parsed, self.reference, 'metadata_io')

    def test_directory_fifo_and_symlink_rejected_without_content_read(self):
        path = self.path('train')
        original = path.read_bytes()
        for kind in ('directory', 'fifo', 'symlink'):
            with self.subTest(kind=kind):
                path.unlink()
                if kind == 'directory': path.mkdir()
                elif kind == 'fifo': os.mkfifo(path)
                else: path.symlink_to(self.path('test'))
                with patch.object(part.os, 'read') as read:
                    self.checked(self.parsed, self.reference, 'metadata_not_regular')
                    read.assert_not_called()
                if kind == 'directory': path.rmdir()
                else: path.unlink()
                path.write_bytes(original)

    def test_short_long_and_same_size_mismatch_reject_before_decode(self):
        original = self.path('test').read_bytes()
        for data, error in ((original[:-1], 'metadata_size'), (original+b' ', 'metadata_size'),
                            (original.replace(b'20', b'21'), 'metadata_hash')):
            with self.subTest(error=error):
                self.path('test').write_bytes(data)
                with patch.object(part, '_decode') as decode, patch.object(frame, 'derive_frame_times') as handoff:
                    self.checked(self.parsed, self.reference, error)
                    decode.assert_not_called()
                    handoff.assert_not_called()

    def test_read_short_or_extra_bytes_no_retry(self):
        data = self.path('train').read_bytes()
        for fake in (data[:-1], data+b' '):
            with patch.object(part.os, 'read', return_value=fake) as read:
                self.checked(self.parsed, self.reference, 'metadata_size')
                self.assertEqual(read.call_count, 1)

    def test_read_error_is_bounded_and_closes_descriptors(self):
        with patch.object(part.os, 'read', side_effect=OSError('PRIVATE-CONTENT')) as read, \
                patch.object(part.os, 'close', wraps=os.close) as close:
            self.checked(self.parsed, self.reference, 'metadata_io')
            self.assertEqual(read.call_count, 1)
            self.assertEqual(close.call_count, 2)

    def test_fd_change_before_and_after_read_rejected(self):
        real_fstat = os.fstat
        for target_call in (2, 3):  # directory, file-open, file-after-read
            calls = []
            def changed(fd):
                value = real_fstat(fd)
                calls.append(fd)
                return stat_copy(value, st_ino=value.st_ino+1) if len(calls) == target_call else value
            with self.subTest(stage=target_call), patch.object(part.os, 'fstat', side_effect=changed), \
                    patch.object(part, '_decode') as decode:
                self.checked(self.parsed, self.reference, 'metadata_identity_changed')
                decode.assert_not_called()

    def test_named_file_replacement_after_read_rejected(self):
        real_stat = os.stat
        calls = []
        def changed(path, *args, **kwargs):
            value = real_stat(path, *args, **kwargs)
            if path == 'transforms_train.json':
                calls.append(path)
                if len(calls) == 2:
                    return stat_copy(value, st_ino=value.st_ino+1)
            return value
        with patch.object(part.os, 'stat', side_effect=changed):
            self.checked(self.parsed, self.reference, 'metadata_identity_changed')

    def test_resolved_directory_replaced_before_open_or_after_reads(self):
        real_fstat = os.fstat
        for target in (1, 6):  # root open / root after both file acquisitions
            calls = []
            def changed(fd):
                value = real_fstat(fd)
                calls.append(fd)
                return stat_copy(value, st_ino=value.st_ino+1) if len(calls) == target else value
            with self.subTest(stage=target), patch.object(part.os, 'fstat', side_effect=changed):
                self.checked(self.parsed, self.reference, 'metadata_directory_changed')

    def test_strict_decoder_with_explicit_temporary_byte_references(self):
        cases = [(b'\xef\xbb\xbf{}', 'metadata_bom'), (b'{"x":"\xff"}', 'metadata_utf8'),
                 (b'{"frames":[],"fr\\u0061mes":[]}', 'metadata_duplicate_key'),
                 (b'{"x":{"a":1,"a":2},"frames":[]}', 'metadata_duplicate_key'),
                 (b'{"x":NaN}', 'metadata_nonfinite'), (b'{"x":Infinity}', 'metadata_nonfinite'),
                 (b'{"x":-Infinity}', 'metadata_nonfinite'), (b'{', 'metadata_json'),
                 (b'{} {}', 'metadata_json'), (b'[]', 'metadata_root'), (b'null', 'metadata_root'),
                 (b'{}', 'metadata_frames'), (b'{"frames":null}', 'metadata_frames'),
                 (b'{"frames":{}}', 'metadata_frames')]
        for data, code in cases:
            with self.subTest(code=code):
                ref = self.fixture_reference('test', data)
                with patch.object(frame, 'derive_frame_times') as handoff:
                    self.checked(self.parsed, ref, code)
                    handoff.assert_not_called()

    def test_reader_handoff_negative_time_mapping_and_late_test_error(self):
        cases = [('missing', 'frame_missing_input'), ('range', 'frame_time_range'),
                 ('path', 'frame_correspondence'), ('bool', 'num_type'), ('order', 'frame_correspondence')]
        for fault, code in cases:
            with self.subTest(fault=fault):
                data = copy.deepcopy(self.documents['test'])
                last = data['frames'][-1]
                if fault == 'missing': last.pop('time')
                elif fault == 'range': last['time'] = 100
                elif fault == 'path': last['file_path'] = './other'
                elif fault == 'bool': last['time'] = True
                else: data['frames'].reverse()
                ref = self.fixture_reference('test', encode(data))
                with patch.object(frame, 'FrameTimes', wraps=frame.FrameTimes) as result:
                    self.checked(self.parsed, ref, code)
                    result.assert_not_called()

    def test_time_config_is_rechecked_by_existing_handoff(self):
        self.parsed.root['dataset']['time']['divisor'] = p2.NumberToken('0')
        self.checked(self.parsed, self.reference, 'time_divisor_domain')

    def test_decoder_preserves_numeric_lexemes_unknown_metadata_unchecked(self):
        data = b'{"frames":[{"file_path":"./d","time":2.0e1},{"file_path":"./e","time":22}],"camera":null}'
        ref = self.fixture_reference('test', data)
        with patch.object(frame, 'derive_frame_times', wraps=frame.derive_frame_times) as handoff:
            result = self.checked(self.parsed, ref)
            self.assertEqual(handoff.call_args.args[2][0]['time'].raw, '2.0e1')
        self.full_match(result, ref, 2)


class RealMetadataTests(Assertions, unittest.TestCase):
    def test_production_entry_d_one_two_all_5312(self):
        reports = ROOT.parent/'reports/corrected-4dgs'
        authority = json.loads((reports/'issue-49/issue-49-preflight.json').read_text())
        historic = json.loads((reports/'issue-32/issue-32-validation-evidence.json').read_text())['transforms']
        self.assertEqual(historic['status'], 'pass')
        self.assertTrue(historic['frame_path_split_time_order_match'])
        prior = {x['path']: x for x in historic['files']}
        pre = {x['path']: x for x in authority['metadata_reference_files']}
        mapping, file_ids, evidence = {}, {}, []
        source = ROOT.parent/'data/4dgs_sph_scene'
        for split, count in (('train', 5146), ('test', 166)):
            path = source/('transforms_'+split+'.json')
            expected = pre[str(path)]
            self.assertEqual((expected['bytes'], expected['sha256']),
                             (prior[str(path)]['bytes_read'], prior[str(path)]['sha256']))
            before = path.stat()
            with path.open('rb') as stream:
                opened = os.fstat(stream.fileno())
                raw = stream.read(expected['bytes']+1)  # one reference acquisition
                after_fd = os.fstat(stream.fileno())
            after = path.stat()
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            self.assertEqual(signature(before), signature(opened))
            self.assertEqual(signature(before), signature(after_fd))
            self.assertEqual(signature(before), signature(after))
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (expected['bytes'], expected['sha256']))
            # Independent fixture decoding, NOT the production decoder.
            decoded = json.loads(raw.decode('utf-8'), parse_int=p2.NumberToken, parse_float=p2.NumberToken)
            self.assertEqual(len(decoded['frames']), count)
            mapping[split] = copy.deepcopy([{'file_path': f['file_path'], 'time': f['time']} for f in decoded['frames']])
            file_ids[split] = part.FileIdentity(expected['bytes'], expected['sha256'])
            evidence.append(dict(path=str(path), size=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                 frames=count, reference_acquisitions=1, production_acquisitions=0,
                                 before=signature(before), issue32_and_preflight_match=True))
        visited = {'train': set(), 'test': set()}
        for item in historic['frames']:
            for split in ('train', 'test'):
                for index in item[split+'_output_indices']:
                    self.assertNotIn(index, visited[split])
                    visited[split].add(index)
                    self.assertEqual(float(mapping[split][index]['time'].raw), item['time_binary64'])
        for split in ('train', 'test'):
            self.assertEqual(visited[split], set(range(len(mapping[split]))))
        reference = part.MetadataReference(file_ids['train'], file_ids['test'], mapping)
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            output = Path(directory)/'not-created'
            for divisor in (1, 2):
                with self.subTest(divisor=divisor), patch.object(part.os, 'read', wraps=os.read) as read:
                    result = self.checked(config(source, output, 0, 33, divisor), reference)
                    self.assertEqual(read.call_count, 2)
                    self.assertEqual([c.args[1] for c in read.call_args_list],
                                     [reference.train.size+1, reference.test.size+1])
                    self.full_match(result, reference, divisor)
                    self.assertEqual((len(result.train), len(result.test)), (5146, 166))
                    records = result.train+result.test
                    self.assertEqual(sum(r.raw_time == 0 for r in records), 32)
                    self.assertEqual(sum(r.raw_time == 33 for r in records), 32)
                    self.assertEqual(len({r.raw_time for r in records}), 166)
                    self.assertEqual(len({r.file_path for r in records}), 5312)
                    self.assertFalse(output.exists())
                    for item in evidence: item['production_acquisitions'] += 1
        for item in evidence:
            self.assertEqual(signature(Path(item['path']).stat()), tuple(item['before']))
        print('METADATA_EVIDENCE '+json.dumps(evidence, sort_keys=True))


class IsolationTests(unittest.TestCase):
    def test_imports_no_heavy_runtime_or_generator(self):
        for path in (ROOT/'formal_frame_metadata.py', Path(__file__)):
            tree = ast.parse(path.read_text())
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import): names.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p6', 'formal_frame_time', 'formal_frame_metadata'})
            if path.name == 'formal_frame_metadata.py':
                self.assertFalse(any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(tree)))

    def test_isolated_pipeline_no_writer_walk_or_heavy_import(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org': raise ModuleNotFoundError('no Jython', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
            'formal_config_json','formal_config_p1','formal_config_p6','formal_config_time',
            'formal_frame_time','formal_frame_metadata'}:
            raise AssertionError('heavy import')
sys.meta_path.insert(0, OnlyStdlib())
def audit(event,args):
    if event == 'open':
        mode,flags=args[1],args[2]
        if (mode and any(c in mode for c in 'wax+')) or flags & (
            os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND): raise AssertionError('writer')
    if event.startswith(('subprocess.','socket.')) or event in (
        'os.mkdir','os.remove','os.rename','os.system','os.rmdir','os.symlink','os.link',
        'os.chmod','os.chown','os.utime','os.truncate','ctypes.dlopen'):
        raise AssertionError('side effect')
sys.addaudithook(audit)
import formal_config_json as p2
import formal_frame_metadata as part
payload=json.loads(sys.argv[3])
reference=part.MetadataReference(part.FileIdentity(**payload['train']),
    part.FileIdentity(**payload['test']),{'train':[{'file_path':'a','time':p2.NumberToken('2')}],
    'test':[{'file_path':'b','time':p2.NumberToken('22')}]})
allowed={sys.argv[4], 'transforms_train.json','transforms_test.json'}
def limited(event,args):
    if event=='open' and args[0] not in allowed: raise AssertionError('unexpected reader')
    if event in ('os.listdir','os.scandir'): raise AssertionError('walk')
sys.addaudithook(limited)
parsed=p2.parse_json_bytes(sys.argv[2].encode())
result=part.read_frame_metadata(parsed,reference)
assert result.train[0].effective_time==1 and result.test[0].effective_time==11
reference.frames['test'][0]['time']=p2.NumberToken('21')
try: part.read_frame_metadata(parsed,reference)
except p2.JSONInputError as error: assert error.code=='frame_correspondence'
else: raise AssertionError('late mismatch accepted')
assert not any(n.split('.')[0] in {'torch','scene','train','gaussian_renderer'} for n in sys.modules)
print('readonly-pipeline-success-and-rejection')
'''
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            source = Path(directory)/'data'
            source.mkdir()
            payload = {}
            for split, path, time in (('train', 'a', 2), ('test', 'b', 22)):
                data = encode({'frames': [{'file_path': path, 'time': time}]})
                (source/('transforms_'+split+'.json')).write_bytes(data)
                payload[split] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            parsed = config(source, Path(directory)/'unused')
            raw = json.dumps(parsed.root, default=lambda n: int(n.raw) if n.is_integer else float(n.raw))
            run = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', script, str(ROOT),
                                  raw, json.dumps(payload), str(source)],
                                 cwd=directory, text=True, capture_output=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(run.stdout.strip(), 'readonly-pipeline-success-and-rejection')
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['data'])


if __name__ == '__main__':
    unittest.main()
