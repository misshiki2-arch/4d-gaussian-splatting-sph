"""CPU partial fixtures only, NOT runnable formal configurations/run values.

Run -B -S -m unittest discover -s tests -p test_formal_config_time.py -v.
Oracle: Decimal at 120 digits using exact binary64 input values, independent
real-arithmetic formulas (including d*d) and Decimal sqrt/ln. For normal
fixtures use 32*binary64 epsilon relative error, plus that absolute bound for
log near zero: ample margin for the short chain of rounded operations and libm,
not a float32/runtime contract. Subnormal fixtures use exact powers of two.
These criteria are fixed before execution, not fitted to implementation output.
"""

import ast
import copy
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal, localcontext
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
import formal_config_p1 as p1
import formal_config_time as time_part


def fixture(a=2, b=22, d=2, c=5):
    # Empty other-owner objects intentionally do not establish full validity.
    return {
        "schema": "corrected_4dgs_training_config_v1", "run_mode": "from_scratch",
        "dataset": {"kind": "nerf_transforms", "eval": True,
                    "time": {"raw_interval": [a, b], "divisor": d}},
        "model": {"gaussian_dim": 4, "spatial_sh_degree": 3,
                  "temporal_sh_degree": 2, "rot_4d": True,
                  "force_sh_3d": False, "sh_evaluation": "conditional_mean"},
        "renderer": {"compute_cov3D_python": False, "convert_SHs_python": False,
                     "scaling_modifier": 1, "env_map_res": 0,
                     "override_color": None, "temporal_prefilter": "disabled"},
        "initialization": {"time_variance_denominator": c},
        "optimization": {}, "reporting": {}, "checkpoint": {}, "output": {},
    }


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def identity_tree(value):
    if type(value) is dict:
        return (id(value), [(k, identity_tree(v)) for k, v in value.items()])
    if type(value) is list:
        return (id(value), [identity_tree(v) for v in value])
    return (id(value), type(value), value)


def put(value, path, replacement):
    node = value
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement


NUMBER_PATHS = (
    ("dataset", "time", "raw_interval", 0),
    ("dataset", "time", "raw_interval", 1),
    ("dataset", "time", "divisor"),
    ("initialization", "time_variance_denominator"),
)


def decimal_oracle(a, b, d, c):
    # No production helper or production-derived value supplies an expectation.
    with localcontext() as context:
        context.prec = 120
        a, b, d, c = (Decimal.from_float(float(x)) for x in (a, b, d, c))
        length = b - a
        variance = length / c
        effective_variance = length / (c * d * d)
        effective_scale = effective_variance.sqrt()
        values = dict(
            raw_duration=length, effective_duration=b/d - a/d,
            divisor=d, time_variance_denominator=c,
            raw_variance=variance, raw_scale=variance.sqrt(),
            effective_variance=effective_variance, effective_scale=effective_scale,
            log_effective_scale=effective_scale.ln(),
        )
        return {k: float(v) for k, v in values.items()}, (float(a), float(b)), (float(a/d), float(b/d))


class TimeTests(unittest.TestCase):
    def check_parsed(self, parsed, error=None):
        before, identities = copy.deepcopy(parsed), identity_tree(parsed.root)
        result = None
        try:
            if error is None:
                result = time_part.derive_time_interval_and_variance(parsed)
            else:
                with self.assertRaises(p2.JSONInputError) as caught:
                    time_part.derive_time_interval_and_variance(parsed)
                self.assertEqual(caught.exception.code, error)
                self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual(parsed, before)
            self.assertEqual(identity_tree(parsed.root), identities)
        return result

    def check_value(self, value, error=None):
        return self.check_parsed(p2.parse_json_bytes(encode(value)), error)

    def test_exact_v_b_distinguishes_v_a_and_no_double_division(self):
        got = self.check_value(fixture())
        self.assertEqual(got.raw_interval, (2, 22))
        self.assertEqual(got.effective_interval, (1, 11))
        for key, expected in dict(raw_duration=20, effective_duration=10,
                                 divisor=2, time_variance_denominator=5,
                                 raw_variance=4, raw_scale=2,
                                 effective_variance=1, effective_scale=1,
                                 log_effective_scale=0).items():
            self.assertEqual(getattr(got, key), expected)
        self.assertNotEqual(got.effective_variance, 2)  # V-A is not V-B.

    def test_d_one_identity_and_signed_origin(self):
        got = self.check_value(fixture(-7, 13, 1, 5))
        self.assertEqual(got.raw_interval, got.effective_interval)
        self.assertEqual(got.raw_duration, got.effective_duration)
        self.assertEqual(got.raw_variance, got.effective_variance)
        self.assertEqual(got.raw_scale, got.effective_scale)
        self.assertEqual(got.raw_scale, 2)

    def test_fractional_divisor_and_coefficient_are_num_not_int(self):
        got = self.check_value(fixture(-2, 6, 2.5, 0.5))
        self.assertEqual(got.divisor, 2.5)
        self.assertEqual(got.time_variance_denominator, 0.5)
        self.assertEqual(got.raw_variance, 16)
        self.assertEqual(got.raw_scale, 4)
        self.assertEqual(got.effective_interval, (-0.8, 2.4))

    def test_independent_high_precision_oracle(self):
        cases = ((2, 22, 2, 5), (-7, 13, 1, 5), (-2, 6, 2.5, 0.5),
                 (0.125, 3.875, 1.1, 0.7), (-31.75, -2.125, 7.25, 3.1),
                 (-1e140, 2e140, 1e100, 0.125),
                 (0, 1e308, 1e200, 1), (0, 1e-200, 1e10, 1e-100))
        tolerance = 32 * sys.float_info.epsilon
        for args in cases:
            with self.subTest(args=args):
                expected, raw, effective = decimal_oracle(*args)
                got = self.check_value(fixture(*args))
                self.assertEqual(got.raw_interval, raw)
                self.assertEqual(got.effective_interval, effective)
                for key, value in expected.items():
                    self.assertTrue(math.isclose(
                        getattr(got, key), value, rel_tol=tolerance,
                        abs_tol=tolerance if key == 'log_effective_scale' else 0), key)

    def test_large_divisor_does_not_require_finite_d_squared(self):
        self.assertTrue(math.isinf(1e200 * 1e200))
        got = self.check_value(fixture(0, 1e308, 1e200, 1))
        self.assertGreater(got.effective_variance, 0)
        self.assertTrue(math.isfinite(got.log_effective_scale))

    def test_positive_subnormal_endpoints_durations_and_variances(self):
        unit = float.fromhex('0x0.0000000000001p-1022')
        got = self.check_value(fixture(0, 4*unit, 2, 1))
        self.assertEqual(got.raw_duration, 4*unit)
        self.assertEqual(got.effective_duration, 2*unit)
        self.assertEqual(got.raw_variance, 4*unit)
        self.assertEqual(got.effective_variance, unit)
        self.assertEqual(got.effective_scale, math.ldexp(1.0, -537))
        got = self.check_value(fixture(0, unit, 1, unit))
        self.assertEqual(got.raw_variance, 1)
        self.assertEqual(got.log_effective_scale, 0)

    def test_true_zero_and_signed_zero_are_not_nonzero_underflow(self):
        got = self.check_value(fixture(-0.0, 20, 2, 5))
        self.assertEqual(math.copysign(1, got.effective_interval[0]), -1)

    def test_all_required_inputs_no_default(self):
        for path in (("dataset", "time"), ("dataset", "time", "raw_interval"),
                     ("dataset", "time", "divisor"),
                     ("initialization", "time_variance_denominator")):
            with self.subTest(path=path):
                value = fixture()
                node = value
                for key in path[:-1]:
                    node = node[key]
                del node[path[-1]]
                self.check_value(value, 'time_missing_input')

    def test_time_requires_object(self):
        for bad in (None, True, False, 1, 1.0, 'time', []):
            with self.subTest(bad=bad):
                value = fixture()
                value['dataset']['time'] = bad
                self.check_value(value, 'time_object_type')

    def test_interval_requires_exact_two_element_array(self):
        for bad in (None, True, 2, '2,22', {}, [], [2], [2, 22, 42]):
            with self.subTest(bad=bad):
                value = fixture()
                value['dataset']['time']['raw_interval'] = bad
                self.check_value(value, 'time_interval_shape')
        parsed = p2.parse_json_bytes(encode(fixture()))
        parsed.root['dataset']['time']['raw_interval'] = tuple(
            parsed.root['dataset']['time']['raw_interval'])
        self.check_parsed(parsed, 'time_interval_shape')

    def test_all_numbers_reject_wrong_types_including_bool(self):
        for path in NUMBER_PATHS:
            for bad in (True, False, None, '5', [], {}):
                with self.subTest(path=path, bad=bad):
                    value = fixture()
                    put(value, path, bad)
                    self.check_value(value, 'num_type')

    def test_all_numbers_reuse_p2_overflow_nonzero_underflow(self):
        for path in NUMBER_PATHS:
            for token, code in ((b'1e309', 'num_overflow'),
                                (b'-1e309', 'num_overflow'),
                                (b'1e-9999', 'num_underflow'),
                                (b'-1e-9999', 'num_underflow')):
                with self.subTest(path=path, token=token):
                    value = fixture()
                    put(value, path, 'TOKEN')
                    raw = encode(value).replace(b'"TOKEN"', token)
                    self.check_parsed(p2.parse_json_bytes(raw), code)

    def test_p2_syntax_rejection_precedes_component(self):
        for token in (b'NaN', b'Infinity', b'-Infinity'):
            value = fixture()
            put(value, NUMBER_PATHS[0], 'TOKEN')
            raw = encode(value).replace(b'"TOKEN"', token)
            with patch.object(time_part, 'derive_time_interval_and_variance') as derive:
                with self.assertRaises(p2.JSONInputError):
                    derive(p2.parse_json_bytes(raw))
                derive.assert_not_called()
        raw = encode(fixture()).replace(b'"divisor":2', b'"divisor":2,"\\u0064ivisor":2')
        with self.assertRaises(p2.JSONInputError) as caught:
            p2.parse_json_bytes(raw)
        self.assertEqual(caught.exception.code, 'duplicate_key')

    def test_domains_are_not_repaired(self):
        for a, b in ((2, 2), (22, 2), (0, -0.0)):
            self.check_value(fixture(a, b), 'time_interval_order')
        for d in (0, -1, 0.5, math.nextafter(1, 0)):
            self.check_value(fixture(d=d), 'time_divisor_domain')
        for c in (0, -0.0, -0.1):
            self.check_value(fixture(c=c), 'time_denominator_domain')

    def test_binary64_endpoint_collapse_at_input(self):
        self.check_value(fixture(9007199254740992, 9007199254740993),
                         'time_interval_order')

    def test_raw_duration_overflow(self):
        self.check_value(fixture(-1e308, 1e308, 2, 5), 'time_raw_duration')

    def test_nonzero_raw_endpoint_to_zero_rejected_before_log(self):
        unit = float.fromhex('0x0.0000000000001p-1022')
        for a, b in ((unit, 1), (-1, -unit)):
            with patch.object(time_part.math, 'log') as log:
                self.check_value(fixture(a, b, 2, 1), 'time_endpoint_conversion')
                log.assert_not_called()

    def test_effective_endpoint_collapse(self):
        unit = float.fromhex('0x0.0000000000001p-1022')
        self.check_value(fixture(3*unit, 4*unit, 2, 1), 'time_endpoint_collapse')

    def test_raw_variance_overflow_and_underflow(self):
        unit = float.fromhex('0x0.0000000000001p-1022')
        for args in ((0, 1, 1, unit), (0, unit, 1, 2)):
            with self.subTest(args=args):
                with patch.object(time_part.math, 'sqrt') as sqrt:
                    self.check_value(fixture(*args), 'time_raw_variance')
                    sqrt.assert_not_called()

    def test_effective_scale_underflow(self):
        with patch.object(time_part.math, 'log') as log:
            self.check_value(fixture(0, 1e-100, 1e200, 1e200), 'time_effective_scale')
            log.assert_not_called()

    def test_effective_variance_and_intermediate_underflow(self):
        for args, code in (((0, 1, 1e200, 1), 'time_effective_variance'),
                           ((0, 1e-100, 1e30, 1e200), 'time_variance_intermediate')):
            with self.subTest(code=code):
                with patch.object(time_part.math, 'log') as log:
                    self.check_value(fixture(*args), code)
                    log.assert_not_called()

    def test_math_result_guards_fault_injection_not_natural_input_claim(self):
        # Positive finite variance implies finite positive sqrt, and finite
        # positive scale implies finite log for real binary64 inputs. Fault
        # injection checks the explicit postcondition guards, not a new domain.
        for bad in (0.0, -1.0, float('inf'), float('nan')):
            with patch.object(time_part.math, 'sqrt', return_value=bad):
                self.check_value(fixture(), 'time_raw_scale')
        for bad in (float('inf'), float('-inf'), float('nan')):
            with patch.object(time_part.math, 'log', return_value=bad):
                self.check_value(fixture(), 'time_log_scale')

    def test_rechecks_p1_every_call_before_arithmetic(self):
        parsed = p2.parse_json_bytes(encode(fixture()))
        with patch.object(p1, 'check_p1_structure_and_fixed_inputs',
                          wraps=p1.check_p1_structure_and_fixed_inputs) as check:
            self.check_parsed(parsed)
            parsed.root['dataset']['eval'] = False
            with patch.object(time_part.math, 'sqrt') as sqrt:
                self.check_parsed(parsed, 'p1_fixed_value')
                sqrt.assert_not_called()
            self.assertEqual(check.call_count, 2)
        for group in ('dataset', 'model', 'renderer'):
            for key in fixture()[group]:
                if key == 'time':
                    continue
                parsed = p2.parse_json_bytes(encode(fixture()))
                del parsed.root[group][key]
                self.check_parsed(parsed, 'p1_missing_fixed')

    def test_current_num_inputs_rechecked_and_no_python_coercion(self):
        parsed = p2.parse_json_bytes(encode(fixture()))
        self.check_parsed(parsed)
        parsed.root['dataset']['time']['divisor'] = p2.NumberToken('0.5')
        self.check_parsed(parsed, 'time_divisor_domain')
        for path in NUMBER_PATHS:
            for bad in (5, 5.0):
                parsed = p2.parse_json_bytes(encode(fixture()))
                put(parsed.root, path, bad)
                self.check_parsed(parsed, 'num_type')

    def test_num_helper_reused_without_integer_domain_or_token_rewrite(self):
        parsed = p2.parse_json_bytes(encode(fixture(2.0, 22.0, 2.5, 0.5)))
        with patch.object(p2, 'require_num', wraps=p2.require_num) as check:
            self.check_parsed(parsed)
            self.assertEqual(check.call_count, 5)  # P1 scaling plus a,b,d,c.
        self.assertEqual(parsed.root['dataset']['time']['raw_interval'][0].raw, '2.0')

    def test_immutable_result_detached_from_mutable_input(self):
        parsed = p2.parse_json_bytes(encode(fixture()))
        got = self.check_parsed(parsed)
        saved = copy.deepcopy(got)
        for field in fields(got):
            with self.assertRaises(FrozenInstanceError):
                setattr(got, field.name, None)
        with self.assertRaises(TypeError):
            got.raw_interval[0] = 9
        with self.assertRaises(TypeError):
            got.effective_interval[0] = 9
        parsed.root['dataset']['time']['raw_interval'][0] = p2.NumberToken('0')
        parsed.root['initialization'].clear()
        parsed.root.clear()
        self.assertEqual(got, saved)
        self.assertTrue(all(type(getattr(got, f.name)) in (float, tuple) for f in fields(got)))

    def test_unchecked_nested_fields_are_not_aliases_defaults_or_acceptance(self):
        value = fixture()
        baseline = self.check_value(value)
        value['dataset']['frame_ratio'] = 99
        value['dataset']['time']['effective_interval'] = [-100, 100]
        value['dataset']['time']['divisor_alias'] = None
        value['initialization']['fps'] = 5
        for group in ('dataset', 'model', 'renderer', 'initialization', 'optimization',
                      'reporting', 'checkpoint', 'output'):
            value[group]['UNCHECKED'] = {'other': [None, True, 'private', 17]}
        self.assertEqual(self.check_value(value), baseline)
        del value['dataset']['time']['divisor']
        self.check_value(value, 'time_missing_input')
        value = fixture()
        del value['initialization']['time_variance_denominator']
        value['initialization']['coefficient_alias'] = 5
        self.check_value(value, 'time_missing_input')
        self.assertFalse(hasattr(baseline, 'verified'))
        self.assertFalse(hasattr(baseline, 'execution_allowed'))

    def test_entry_type_and_bounded_error(self):
        for value in (None, {}, fixture(), b'{}', p2.ParsedJSON([])):
            with self.assertRaises(p1.P1InputError) as caught:
                time_part.derive_time_interval_and_variance(value)
            self.assertEqual(caught.exception.code, 'p1_parsed_input')
        value = fixture()
        value['dataset']['time']['divisor'] = 'PRIVATE_INPUT_MARKER'
        self.check_value(value, 'num_type')


class IsolationTests(unittest.TestCase):
    def test_source_and_test_imports_only_stdlib_and_three_components(self):
        for path in (ROOT/'formal_config_time.py', Path(__file__)):
            names = set()
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    names.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_time'})

    def test_isolated_success_rejection_no_heavy_import_no_writer(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_time'}:
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
import formal_config_time as part
value = json.loads(sys.argv[2])
result = part.derive_time_interval_and_variance(p2.parse_json_bytes(json.dumps(value).encode()))
assert result.effective_variance == 1
for bad in (None, True, 0, 'secret', 1e200):
    value['dataset']['time']['divisor'] = bad
    try:
        part.derive_time_interval_and_variance(p2.parse_json_bytes(json.dumps(value).encode()))
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('bad time input accepted')
assert not any(x.split('.')[0] in {'torch','scene','gaussian_renderer','train'} for x in sys.modules)
print(json.dumps({'stdlib_only': True, 'no_writer': True, 'no_heavy_import': True}))
'''
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([sys.executable, '-I', '-B', '-S', '-c', script,
                                  str(ROOT), encode(fixture()).decode()], cwd=directory,
                                 capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout),
                             {'stdlib_only': True, 'no_writer': True, 'no_heavy_import': True})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
