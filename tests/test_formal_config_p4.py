"""Independent P1/P4 partial fixtures, NOT runnable formal configurations.

Synthetic dimensions only; expectations follow adopted P4, not output tables.
Run with -B -S -m unittest discover -s tests -p test_formal_config_p4.py -v.
"""

import ast
import copy
from dataclasses import FrozenInstanceError, asdict
import json
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
import formal_config_p4 as p4


def partial_fixture(divisor=2):
    return {
        "schema": "corrected_4dgs_training_config_v1",
        "run_mode": "from_scratch",
        "dataset": {"kind": "nerf_transforms", "eval": True,
                    "resolution": {"mode": "integer_divisor", "divisor": divisor}},
        "model": {"gaussian_dim": 4, "spatial_sh_degree": 3,
                  "temporal_sh_degree": 2, "rot_4d": True,
                  "force_sh_3d": False, "sh_evaluation": "conditional_mean"},
        "renderer": {"compute_cov3D_python": False, "convert_SHs_python": False,
                     "scaling_modifier": 1, "env_map_res": 0,
                     "override_color": None, "temporal_prefilter": "disabled"},
        "initialization": {}, "optimization": {}, "reporting": {},
        "checkpoint": {}, "output": {},
    }


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def identity_tree(value):
    if type(value) is dict:
        return (id(value), [(k, identity_tree(v)) for k, v in value.items()])
    if type(value) is list:
        return (id(value), [identity_tree(v) for v in value])
    return (id(value), type(value), value)


def divisor_token(raw):
    return encode(partial_fixture("TOKEN_MARKER")).replace(b'"TOKEN_MARKER"', raw)


class P4Tests(unittest.TestCase):
    def check_parsed(self, parsed, error=None, width=120, height=80):
        before, identities = copy.deepcopy(parsed), identity_tree(parsed.root)
        try:
            if error is None:
                return p4.derive_p4_resolution(parsed, raw_width=width, raw_height=height)
            with self.assertRaises(p2.JSONInputError) as caught:
                p4.derive_p4_resolution(parsed, raw_width=width, raw_height=height)
            self.assertEqual(caught.exception.code, error)
            self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual(parsed, before)
            self.assertEqual(identity_tree(parsed.root), identities)

    def check_value(self, data, error=None, width=120, height=80):
        return self.check_parsed(p2.parse_json_bytes(encode(data)), error, width, height)

    def test_exact_division_and_unit_divisor(self):
        for d, w, h, expected in ((1, 121, 81, (121, 81)),
                                   (2, 120, 80, (60, 40)),
                                   (3, 123, 81, (41, 27)),
                                   (5, 120, 80, (24, 16)),
                                   (7, 7, 14, (1, 2))):
            with self.subTest(divisor=d):
                result = self.check_value(partial_fixture(d), width=w, height=h)
                self.assertEqual(asdict(result), {
                    "divisor": d, "raw_width": w, "raw_height": h,
                    "width": expected[0], "height": expected[1]})
                self.assertTrue(all(type(v) is int for v in asdict(result).values()))

    def test_each_or_both_nondivisible_dimensions_reject(self):
        for w, h in ((121, 80), (120, 81), (121, 81)):
            self.check_value(partial_fixture(), "p4_not_divisible", w, h)

    def test_divisor_larger_than_either_dimension_rejects(self):
        for w, h in ((1, 80), (120, 1), (1, 1)):
            self.check_value(partial_fixture(), "p4_not_divisible", w, h)

    def test_dimensions_reject_nonpositive_values(self):
        for wrong in (0, -1, -100):
            self.check_value(partial_fixture(), "p4_dimension_range", wrong, 80)
            self.check_value(partial_fixture(), "p4_dimension_range", 120, wrong)

    def test_dimensions_require_exact_integers_without_coercion(self):
        class IntSubclass(int):
            pass
        for wrong in (True, False, 120.0, 1.5, "120", None, [], {},
                      p2.NumberToken("120"), IntSubclass(120)):
            self.check_value(partial_fixture(), "p4_dimension_type", wrong, 80)
            self.check_value(partial_fixture(), "p4_dimension_type", 120, wrong)

    def test_raw_dimensions_are_required_local_arguments(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        for kwargs in ({}, {"raw_width": 120}, {"raw_height": 80}):
            with self.assertRaises(TypeError):
                p4.derive_p4_resolution(parsed, **kwargs)
        self.check_parsed(parsed)
        self.assertNotIn("raw_width", parsed.root["dataset"])
        self.assertNotIn("raw_height", parsed.root["dataset"])

    def test_missing_resolution_or_either_key_rejects(self):
        data = partial_fixture()
        del data["dataset"]["resolution"]
        self.check_value(data, "p4_missing_resolution")
        for key in ("mode", "divisor"):
            data = partial_fixture()
            del data["dataset"]["resolution"][key]
            self.check_value(data, "p4_resolution_keys")
        data["dataset"]["resolution"] = {}
        self.check_value(data, "p4_resolution_keys")

    def test_closed_resolution_object_rejects_extra_keys(self):
        for key in ("unknown", "width", "height", "scale", "raw_width", "raw_height"):
            for extra in (None, False, 1, "", {}, []):
                data = partial_fixture()
                data["dataset"]["resolution"][key] = extra
                self.check_value(data, "p4_resolution_keys")

    def test_legacy_scalar_and_other_nonobject_resolution_rejects(self):
        for wrong in (-1, 1, 2, 1600, 2.0, 0.5, True, False, None,
                      "integer_divisor", [], [2]):
            data = partial_fixture()
            data["dataset"]["resolution"] = wrong
            self.check_value(data, "p4_resolution_object")

    def test_mode_type_and_exact_value_without_normalization(self):
        for wrong in (None, True, False, 1, 1.0, [], {}):
            data = partial_fixture()
            data["dataset"]["resolution"]["mode"] = wrong
            self.check_value(data, "p4_mode_type")
        for wrong in ("", "auto", "target_width", "scale", "INTEGER_DIVISOR",
                      " integer_divisor", "integer_divisor\n"):
            data = partial_fixture()
            data["dataset"]["resolution"]["mode"] = wrong
            self.check_value(data, "p4_mode_value")

    def test_divisor_integer_domain_and_no_large_token_conversion(self):
        for raw, code in ((b'0', 'p4_divisor_range'), (b'-0', 'p4_divisor_range'),
                          (b'-1', 'int_range'), (b'2147483648', 'int_range'),
                          (b'9' * 5000, 'int_range')):
            self.check_parsed(p2.parse_json_bytes(divisor_token(raw)), code)

    def test_divisor_requires_integer_token(self):
        for raw in (b'true', b'false', b'2.0', b'2e0', b'"2"', b'null',
                    b'[]', b'{}', b'"1+1"'):
            self.check_parsed(p2.parse_json_bytes(divisor_token(raw)), "int_type")

    def test_exact_large_arithmetic_without_float_or_size_allocations(self):
        result = self.check_value(partial_fixture(2147483647),
                                  width=4611686014132420609, height=6442450941)
        self.assertEqual((result.width, result.height), (2147483647, 3))
        self.check_value(partial_fixture(2147483647), "p4_not_divisible",
                         4611686014132420610, 6442450941)
        result = self.check_value(partial_fixture(1),
                                  width=9007199254740993, height=9007199254740995)
        self.assertEqual((result.width, result.height),
                         (9007199254740993, 9007199254740995))
        result = self.check_value(partial_fixture(2147483646),
                                  width=4294967292, height=2147483646)
        self.assertEqual((result.width, result.height), (2, 1))

    def test_competing_resolution_scales_explicit_paths_even_inactive(self):
        for place, code in (("root", "p1_top_level_keys"),
                             ("dataset", "p4_competing_input"),
                             ("resolution", "p4_resolution_keys")):
            for extra in (None, False, 0, "", [], {}, [1.0], [2]):
                data = partial_fixture()
                target = (data if place == "root" else data["dataset"]
                          if place == "dataset" else data["dataset"]["resolution"])
                target["resolution_scales"] = extra
                self.check_value(data, code)

    def test_competing_authority_cannot_supply_missing_resolution(self):
        data = partial_fixture()
        del data["dataset"]["resolution"]
        data["dataset"]["resolution_scales"] = [1.0]
        self.check_value(data, "p4_competing_input")

    def test_current_p1_structure_is_required(self):
        for key in partial_fixture():
            data = partial_fixture()
            del data[key]
            self.check_value(data, "p1_top_level_keys")
        for key in ("dataset", "model", "renderer", "initialization",
                    "optimization", "reporting", "checkpoint", "output"):
            data = partial_fixture()
            data[key] = []
            self.check_value(data, "p1_object")
        for key in ("schema", "run_mode"):
            data = partial_fixture()
            data[key] = "wrong"
            self.check_value(data, "p1_literal")

    def test_all_fourteen_p1_fixed_inputs_rechecked(self):
        count = 0
        for group in ("dataset", "model", "renderer"):
            for field, good in partial_fixture()[group].items():
                if field == "resolution":
                    continue
                count += 1
                data = partial_fixture()
                del data[group][field]
                self.check_value(data, "p1_missing_fixed")
                data = partial_fixture()
                data[group][field] = (not good if type(good) is bool else
                                      good + 1 if type(good) is int else "wrong")
                self.check_value(data, "p1_null_type" if good is None
                                 else "p1_fixed_value")
        self.assertEqual(count, 14)

    def test_current_input_rechecked_after_prior_p1_and_p4_success(self):
        for change, code in (("fixed", "p1_fixed_value"),
                             ("divisor", "p4_divisor_range"),
                             ("mode", "p4_mode_value"),
                             ("competing", "p4_competing_input"),
                             ("missing", "p4_resolution_keys")):
            parsed = p2.parse_json_bytes(encode(partial_fixture()))
            self.assertIsNone(p1.check_p1_structure_and_fixed_inputs(parsed))
            self.check_parsed(parsed)
            dataset = parsed.root["dataset"]
            if change == "fixed":
                dataset["eval"] = False
            elif change == "divisor":
                dataset["resolution"]["divisor"] = p2.NumberToken("0")
            elif change == "mode":
                dataset["resolution"]["mode"] = "auto"
            elif change == "competing":
                dataset["resolution_scales"] = None
            else:
                del dataset["resolution"]["divisor"]
            self.check_parsed(parsed, code)

    def test_reuses_current_p1_and_p2_integer_checks(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        with patch.object(p1, "check_p1_structure_and_fixed_inputs",
                          wraps=p1.check_p1_structure_and_fixed_inputs) as check, \
                patch.object(p2, "require_int", wraps=p2.require_int) as ints:
            self.check_parsed(parsed)
            check.assert_called_once_with(parsed)
            self.assertEqual(ints.call_count, 5)  # Four P1 + divisor.
            self.assertIs(ints.call_args_list[-1].args[0],
                          parsed.root["dataset"]["resolution"]["divisor"])

    def test_wrong_entry_and_python_numeric_bypass_reject(self):
        for value in (None, {}, partial_fixture(), b'{}', p2.ParsedJSON([])):
            with self.assertRaises(p1.P1InputError):
                p4.derive_p4_resolution(value, raw_width=120, raw_height=80)
        for value in (True, 2, 2.0, "2"):
            parsed = p2.parse_json_bytes(encode(partial_fixture()))
            parsed.root["dataset"]["resolution"]["divisor"] = value
            self.check_parsed(parsed, "int_type")

    def test_detached_immutable_result_does_not_confer_formal_authority(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        result = self.check_parsed(parsed)
        for name in ("verified", "execution_allowed", "root", "image_verified",
                     "owner_verified", "runtime"):
            self.assertFalse(hasattr(result, name))
        for field in ("divisor", "raw_width", "raw_height", "width", "height"):
            with self.assertRaises(FrozenInstanceError):
                setattr(result, field, None)
        parsed.root["dataset"]["resolution"]["divisor"] = p2.NumberToken("4")
        later = self.check_parsed(parsed)
        self.assertIsNot(later, result)
        parsed.root.clear()
        self.assertEqual(asdict(result), {"divisor": 2, "raw_width": 120,
                                         "raw_height": 80, "width": 60, "height": 40})
        self.assertEqual((later.divisor, later.width, later.height), (4, 30, 20))

    def test_unchecked_nested_fields_are_not_adopted_or_rewritten(self):
        data = partial_fixture()
        for group in ("dataset", "model", "renderer", "initialization",
                      "optimization", "reporting", "checkpoint", "output"):
            data[group]["UNREVIEWED"] = {"resolution_scales": [None, 1.5, "日本語"]}
        data["output"]["directory"] = "not-P6-validated"
        data["model"]["eval_shfs_4d"] = False  # P3 is not run by P4.
        data["optimization"]["total_updates"] = "not-P5-validated"
        data["initialization"]["time_variance_denominator"] = None
        self.assertEqual(self.check_value(data).width, 60)

    def test_p2_syntax_and_all_size_bounds_reject_before_p4(self):
        raw = encode(partial_fixture())
        invalid = [(raw.replace(b'"divisor":2', b'"divisor":2,"di\\u0076isor":2'),
                    "duplicate_key"), (divisor_token(b'1+1'), "json_syntax"),
                   (b'\xef\xbb\xbf' + raw, "bom"), (raw + b' {}', "trailing_value"),
                   (raw + b' ' * (262145 - len(raw)), "input_bytes"),
                   (b'{"x":' + b'[' * 16 + b'0' + b']' * 16 + b'}', "container_depth"),
                   (encode({"x": [None] * 4097}), "array_elements"),
                   (encode({"x": "x" * 4097}), "string_scalars")]
        for raw, code in invalid:
            with patch.object(p4, "derive_p4_resolution") as derive:
                with self.assertRaises(p2.JSONInputError) as caught:
                    derive(p2.parse_json_bytes(raw), raw_width=120, raw_height=80)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(str(caught.exception), code)
                derive.assert_not_called()

    def test_errors_never_echo_private_inputs(self):
        data = partial_fixture("/private/P4_DO_NOT_ECHO")
        self.check_value(data, "int_type")
        data = partial_fixture()
        data["dataset"]["resolution"]["mode"] = "/private/P4_DO_NOT_ECHO"
        self.check_value(data, "p4_mode_value")
        self.check_value(partial_fixture(), "p4_dimension_type", "/private/raw", 80)


class IsolationTests(unittest.TestCase):
    def test_imports_are_stdlib_or_p1_p2_p4_only(self):
        for path in (ROOT / 'formal_config_p4.py', Path(__file__)):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names |
                                 {'formal_config_json', 'formal_config_p1',
                                  'formal_config_p4'})
            if path.name == 'formal_config_p4.py':
                self.assertFalse(any(isinstance(n, ast.Div) for n in ast.walk(tree)))

    def test_isolated_no_heavy_import_writer_or_dataset_access(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython package unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_p4'}:
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
import formal_config_p4 as p4
def no_files(event, args):
    if event in ('open', 'os.listdir', 'os.scandir'):
        raise AssertionError('filesystem input or scan attempted')
sys.addaudithook(no_files)
def check(data, w=120, h=80):
    return p4.derive_p4_resolution(p2.parse_json_bytes(json.dumps(data).encode()),
                                   raw_width=w, raw_height=h)
data = json.loads(sys.argv[2])
assert (check(data).width, check(data).height) == (60, 40)
invalid = [{}, dict(data, unknown=None)]
for field, bad in (('resolution', -1), ('resolution_scales', False)):
    value = json.loads(sys.argv[2])
    value['dataset'][field] = bad
    invalid.append(value)
for value, w, h in [(v, 120, 80) for v in invalid] + [
        (data, 121, 80), (data, 120, 81), (data, True, 80), (data, 0, 80)]:
    try:
        check(value, w, h)
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('invalid input accepted')
data['dataset']['resolution']['divisor'] = 2147483647
assert check(data, 4611686014132420609, 6442450941).width == 2147483647
assert not any(x.split('.')[0] in {'torch','PIL','scene','gaussian_renderer','train',
                                  'formal_config_p3','formal_config_p5'}
               for x in sys.modules)
print(json.dumps({'stdlib_only':True,'no_writer':True,'no_heavy_import':True,
                  'no_dataset_access_during_calls':True}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-I', '-B', '-S', '-c', script, str(ROOT),
                 encode(partial_fixture()).decode()], cwd=directory,
                text=True, capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {
                'stdlib_only': True, 'no_writer': True, 'no_heavy_import': True,
                'no_dataset_access_during_calls': True})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
