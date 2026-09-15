"""Independent P2 component fixtures; no full formal-config acceptance claim.

Run from the repository with:
    /usr/bin/python3 -B -S -m unittest discover -s tests -p test_formal_config_json.py -v
Only stdlib, this component, and temporary CPU inputs are used.
"""

import ast
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal_config_json as p2


def parsed_value(token):
    return p2.parse_json_bytes(b'{"value":' + token + b'}').root["value"]


class RejectAssertions:
    def rejects(self, raw, code):
        with self.assertRaises(p2.JSONInputError) as caught:
            p2.parse_json_bytes(raw)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)


class SyntaxTests(RejectAssertions, unittest.TestCase):
    def test_normal_nested_object_is_only_parse_data(self):
        result = p2.parse_json_bytes(
            b' \r\n\t{"arbitrary":{"a":[true,false,null,{},[]]},"s":"x"} \n')
        self.assertIs(type(result), p2.ParsedJSON)
        self.assertEqual(result.root,
                         {"arbitrary": {"a": [True, False, None, {}, []]}, "s": "x"})
        self.assertEqual(p2.parse_json_bytes(b'{}').root, {})
        self.assertFalse(hasattr(result, "verified"))

    def test_root_object_only(self):
        for raw in (b'', b' \n', b'[]', b'null', b'true', b'1', b'"x"'):
            with self.subTest(raw=raw):
                self.rejects(raw, "root_object")

    def test_json_syntax_rejection(self):
        for raw in (
                b'{', b'{"a"}', b'{"a":}', b'{"a" 1}', b'{a:1}', b"{'a':1}",
                b'{"a":1,}', b'{"a":[1,]}', b'{"a":[,1]}', b'{"a":1 "b":2}',
                b'{"a":[1 2]}', b'{"a":True}', b'{"a":None}', b'{/*x*/}',
                b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}',
                b'{"a":01}', b'{"a":+1}', b'{"a":.1}', b'{"a":1.}',
                b'{"a":1e}', b'{"a":1e+}', b'{"a":--1}', b'{"a":0x1}',
                b'{"a":1_000}', b'{"a":truefalse}', b'{"a":null0}',
                b'{"a":1+1}', b'{"a":\x0b1}', '{"a":١}'.encode()):
            with self.subTest(raw=raw):
                self.rejects(raw, "json_syntax")

    def test_trailing_values_and_non_json_whitespace(self):
        for raw in (b'{} {}', b'{}[]', b'{}null', b'{}0', b'{} #x',
                    b'{}\x0b', '{}\u00a0'.encode()):
            with self.subTest(raw=raw):
                self.rejects(raw, "trailing_value")

    def test_recursive_decoded_duplicates(self):
        for raw in (b'{"a":0,"a":1}', b'{"a":0,"\\u0061":1}',
                    b'{"o":{"a":{},"\\u0061":[]}}',
                    b'{"o":[{"x":null,"x":false}]}',
                    '{"😀":1,"\\ud83d\\ude00":2}'.encode()):
            with self.subTest(raw=raw):
                self.rejects(raw, "duplicate_key")
        self.assertEqual(p2.parse_json_bytes(
            b'{"x":{},"o":{"x":{}},"a":[{"x":{}},{"x":{}}]}').root,
            {"x": {}, "o": {"x": {}}, "a": [{"x": {}}, {"x": {}}]})

    def test_utf8_and_bom(self):
        self.rejects(b'\xef\xbb\xbf{}', "bom")
        for bad in (b'\xff', b'\xc0\af', b'\xe2\x82', b'\x80',
                    b'\xed\xa0\x80', b'\xf4\x90\x80\x80'):
            with self.subTest(bad=bad):
                self.rejects(b'{"x":"' + bad + b'"}', "utf8")
        self.assertEqual(parsed_value('"日本語😀\ufeff"'.encode()), '日本語😀\ufeff')

    def test_unicode_scalars_and_no_normalization(self):
        self.assertEqual(parsed_value(b'"\\ud83d\\ude00"'), "😀")
        self.assertEqual(parsed_value(b'"\\udbff\\udfff"'), "\U0010ffff")
        for token in (b'"\\ud800"', b'"\\udfff"', b'"\\ud800x"',
                      b'"\\ud800\\ud800"', b'"\\udc00\\ud800"'):
            with self.subTest(token=token):
                self.rejects(b'{"x":' + token + b'}', "unicode_scalar")
        self.rejects(b'{"\\ud800":0}', "unicode_scalar")
        value = p2.parse_json_bytes('{"é":null,"é":false,"A":true,"a":[]}'.encode())
        self.assertEqual(list(value.root), ['é', 'é', 'A', 'a'])

    def test_string_syntax_and_escapes(self):
        for token in (b'"unterminated', b'"\\x"', b'"\\u123"', b'"\\uZZZZ"',
                      b'"raw\ncontrol"', b'"raw\x00control"', b'"\\"'):
            with self.subTest(token=token):
                self.rejects(b'{"x":' + token + b'}', "json_string")
        self.assertEqual(parsed_value(b'"\\\"\\\\\\/\\b\\f\\n\\r\\t"'),
                         '"\\/\b\f\n\r\t')
        # Path-control rejection is semantic/path-owner work, not a global
        # rejection of legal JSON escaped controls in arbitrary strings.
        self.assertEqual(parsed_value(b'"\\u0000"'), '\x00')

    def test_errors_do_not_echo_input(self):
        marker = b'private-input-marker'
        for raw in (b'{"' + marker + b'":0,"' + marker + b'":1}',
                    b'{"x":"' + marker + b'\\q"}'):
            with self.assertRaises(p2.JSONInputError) as caught:
                p2.parse_json_bytes(raw)
            self.assertNotIn(marker.decode(), str(caught.exception))


class LimitTests(RejectAssertions, unittest.TestCase):
    def test_byte_limits_before_decode_or_parse(self):
        for size in (262143, 262144):
            with self.subTest(size=size):
                data = b'{}' + b' ' * (size - 2)
                self.assertEqual(len(data), size)
                self.assertEqual(p2.parse_json_bytes(data).root, {})
        with patch.object(p2, "_Parser", side_effect=AssertionError("parsed")):
            self.rejects(b'\xff' * 262145, "input_bytes")

    def test_bytes_not_unicode_length(self):
        body = b'{"x":' + ('"' + '日' * 4096 + '"').encode() + b'}'
        exact = body + b' ' * (262144 - len(body))
        self.assertEqual(p2.parse_json_bytes(exact).root['x'], '日' * 4096)
        self.rejects(exact + b' ', "input_bytes")

    def test_container_depth_boundaries(self):
        for depth in (15, 16, 17):
            for kind in ('array', 'object', 'mixed'):
                value = b'0'
                for index in range(depth - 1):
                    value = (b'[' + value + b']' if kind == 'array'
                             or (kind == 'mixed' and index % 2 == 0)
                             else b'{"x":' + value + b'}')
                raw = b'{"x":' + value + b'}'
                with self.subTest(depth=depth, kind=kind):
                    if depth <= 16:
                        self.assertIs(type(p2.parse_json_bytes(raw)), p2.ParsedJSON)
                    else:
                        self.rejects(raw, "container_depth")
        # Reject upon opening depth 17, before malformed tail or full parse.
        self.rejects(b'{"x":' + b'[' * 16 + b'not-json', "container_depth")
        self.rejects(b'{"x":' + b'[' * 10000, "container_depth")

    def test_array_boundaries_per_array(self):
        for count in (4095, 4096, 4097):
            raw = b'{"a":[' + b','.join([b'null'] * count) + b']}'
            with self.subTest(count=count):
                if count <= 4096:
                    self.assertEqual(p2.parse_json_bytes(raw).root['a'], [None] * count)
                else:
                    self.rejects(raw, "array_elements")
        array = b'[' + b','.join([b'null'] * 4096) + b']'
        self.assertEqual(len(p2.parse_json_bytes(
            b'{"a":[' + array + b',' + array + b']}').root['a']), 2)
        self.rejects(b'{"a":[' + b'null,' * 4096 + b'bad', "array_elements")

    def test_string_scalar_boundaries_values_and_keys(self):
        for count in (4095, 4096, 4097):
            for encoded, expected in ((b'a', 'a'), ('😀'.encode(), '😀'),
                                      (b'\\ud83d\\ude00', '😀')):
                for key in (False, True):
                    token = b'"' + encoded * count + b'"'
                    raw = b'{' + token + b':null}' if key else b'{"x":' + token + b'}'
                    with self.subTest(count=count, encoded=encoded, key=key):
                        if count > 4096:
                            self.rejects(raw, "string_scalars")
                        else:
                            root = p2.parse_json_bytes(raw).root
                            self.assertEqual(root, {expected * count: None} if key
                                             else {'x': expected * count})

    def test_bounded_stream_and_short_reads(self):
        class ShortStream(io.BytesIO):
            def __init__(self, data):
                super().__init__(data)
                self.requests = []

            def read(self, size):
                self.requests.append(size)
                return super().read(min(size, 8191))

        for size in (262143, 262144, 262145, 300000):
            stream = ShortStream(b'{}' + b' ' * (size - 2))
            with self.subTest(size=size):
                if size <= 262144:
                    self.assertEqual(p2.read_json(stream).root, {})
                else:
                    with patch.object(p2, 'parse_json_bytes',
                                      side_effect=AssertionError('parsed')):
                        with self.assertRaises(p2.JSONInputError) as caught:
                            p2.read_json(stream)
                    self.assertEqual(caught.exception.code, 'input_bytes')
                self.assertLessEqual(stream.tell(), 262145)
                self.assertTrue(all(0 < n <= 262145 for n in stream.requests))
                self.assertFalse(stream.closed)

    def test_temporary_binary_file_and_wrong_input_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            path.write_bytes(b'{"x":true}')
            with path.open('rb') as stream:
                self.assertEqual(p2.read_json(stream).root, {'x': True})
        for value in ('{}', bytearray(b'{}'), None):
            with self.subTest(value=value):
                with self.assertRaises(p2.JSONInputError):
                    p2.parse_json_bytes(value)
        with self.assertRaises(p2.JSONInputError) as caught:
            p2.read_json(io.StringIO('{}'))
        self.assertEqual(caught.exception.code, 'binary_stream_required')


class NumberTests(unittest.TestCase):
    def test_token_origin_and_precision(self):
        values = [parsed_value(t) for t in (b'1', b'1.0', b'1e0', b'true')]
        self.assertEqual([v.raw for v in values[:3]], ['1', '1.0', '1e0'])
        self.assertEqual([v.is_integer for v in values[:3]], [True, False, False])
        self.assertIs(values[3], True)
        self.assertEqual(len(set(values)), 4)
        self.assertEqual(parsed_value(b'9007199254740993').raw, '9007199254740993')
        self.assertEqual(p2.require_num(parsed_value(b'9007199254740993')),
                         9007199254740992.0)

    def test_int_bounds_and_type(self):
        for raw, expected in ((b'0', 0), (b'-0', 0), (b'1', 1),
                              (b'2147483646', 2147483646),
                              (b'2147483647', 2147483647)):
            with self.subTest(raw=raw):
                self.assertEqual(p2.require_int(parsed_value(raw)), expected)
        for raw in (b'2147483648', b'-1', b'9' * 5000):
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(p2.JSONInputError) as caught:
                    p2.require_int(parsed_value(raw))
                self.assertEqual(caught.exception.code, 'int_range')
        for raw in (b'1.0', b'1e0', b'true', b'false', b'"1"', b'null', b'[]', b'{}'):
            with self.subTest(raw=raw):
                with self.assertRaises(p2.JSONInputError) as caught:
                    p2.require_int(parsed_value(raw))
                self.assertEqual(caught.exception.code, 'int_type')

    def test_num_finite_and_no_inferred_int_or_positive_domain(self):
        for raw, expected in ((b'1', 1.0), (b'1.0', 1.0), (b'1e0', 1.0),
                              (b'-0.25', -0.25), (b'2147483648', 2147483648.0),
                              (b'1.7976931348623157e308', sys.float_info.max),
                              (b'5e-324', float.fromhex('0x0.0000000000001p-1022'))):
            with self.subTest(raw=raw):
                actual = p2.require_num(parsed_value(raw))
                self.assertIs(type(actual), float)
                self.assertEqual(actual, expected)
        for raw in (b'true', b'false', b'"1"', b'null', b'[]', b'{}'):
            with self.subTest(raw=raw):
                with self.assertRaises(p2.JSONInputError) as caught:
                    p2.require_num(parsed_value(raw))
                self.assertEqual(caught.exception.code, 'num_type')

    def test_num_overflow_and_nonzero_underflow(self):
        for raw in (b'1e309', b'-1e309', b'9' * 5000, b'1e' + b'9' * 5000):
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(p2.JSONInputError) as caught:
                    p2.require_num(parsed_value(raw))
                self.assertEqual(caught.exception.code, 'num_overflow')
        for raw in (b'1e-324', b'-1e-324', b'0.001e-9999', b'1e-' + b'9' * 5000):
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(p2.JSONInputError) as caught:
                    p2.require_num(parsed_value(raw))
                self.assertEqual(caught.exception.code, 'num_underflow')

    def test_true_zero_including_huge_exponents(self):
        for raw in (b'0', b'0.0', b'0e123', b'0e-123', b'0.000e999999',
                    b'0e' + b'9' * 5000, b'0e-' + b'9' * 5000):
            for sign in (b'', b'-'):
                with self.subTest(raw=raw[:30], sign=sign):
                    value = p2.require_num(parsed_value(sign + raw))
                    self.assertEqual(value, 0.0)
                    self.assertEqual(math.copysign(1, value), -1 if sign else 1)

    def test_bool_and_unproven_python_numbers(self):
        self.assertIs(p2.require_bool(parsed_value(b'true')), True)
        self.assertIs(p2.require_bool(parsed_value(b'false')), False)
        for value in (parsed_value(b'0'), parsed_value(b'1'), 'true', None, 0, 1, 1.0):
            with self.subTest(value=value):
                with self.assertRaises(p2.JSONInputError):
                    p2.require_bool(value)
        for check in (p2.require_int, p2.require_num):
            for value in (True, False, 1, 1.0, '1'):
                with self.subTest(check=check.__name__, value=value):
                    with self.assertRaises(p2.JSONInputError):
                        check(value)

    def test_number_token_cannot_hide_invalid_lexeme(self):
        for raw in ('NaN', 'Infinity', '01', '+1', '١', '1 ', '', True, 1):
            with self.subTest(raw=raw):
                with self.assertRaises(p2.JSONInputError):
                    p2.NumberToken(raw)


class IsolationTests(unittest.TestCase):
    def test_source_and_test_imports_are_stdlib_or_component_only(self):
        for path in (ROOT / 'formal_config_json.py', Path(__file__)):
            names = set()
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    names.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names | {'formal_config_json'})

    def test_isolated_process_no_heavy_import_no_jit_no_writer(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        # Python 3.10 stdlib copy probes Jython inside try/except ImportError.
        # Deny that optional package as absent, never load it or allow its path.
        # All other non-stdlib requests, including optional heavy ones, fail.
        if fullname == 'org':
            raise ModuleNotFoundError('Jython package unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {'formal_config_json'}:
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
assert p2.require_num(p2.parse_json_bytes(b'{"x":1.0}').root['x']) == 1.0
for raw in (b'{} {}', b'{"a":0,"a":1}', b'{"a":' + b'[' * 17,
            b'{"a":"\ud800"}', b' ' * 262145):
    try:
        p2.parse_json_bytes(raw)
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('invalid input accepted')
assert not any(x.split('.')[0] in {'torch', 'scene', 'gaussian_renderer', 'train'}
               for x in sys.modules)
print(json.dumps({'stdlib_only': True, 'no_writer': True, 'no_heavy_import': True}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-I', '-B', '-S', '-c', script, str(ROOT)],
                cwd=directory, text=True, capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout),
                             {'stdlib_only': True, 'no_writer': True, 'no_heavy_import': True})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
