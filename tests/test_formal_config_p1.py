"""P1-only fixtures from the plan, NOT examples of runnable formal configs.

Run with -B -S -m unittest discover -s tests -p test_formal_config_p1.py -v.
Expected values are independent of the implementation's constants/helpers.
"""

import ast
import copy
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


def partial_fixture():
    # Only the adopted P1 table; empty later-owner objects are not full configs.
    return {
        "schema": "corrected_4dgs_training_config_v1",
        "run_mode": "from_scratch",
        "dataset": {"kind": "nerf_transforms", "eval": True},
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


def with_token(group, field, token):
    value = partial_fixture()
    value[group][field] = "TOKEN_PLACEHOLDER"
    return encode(value).replace(b'"TOKEN_PLACEHOLDER"', token)


def identity_tree(value):
    """Detect even equal-value replacement or key-order changes, not just ==."""
    if type(value) is dict:
        return (id(value), [(k, identity_tree(v)) for k, v in value.items()])
    if type(value) is list:
        return (id(value), [identity_tree(v) for v in value])
    return (id(value), type(value), value)


class P1Tests(unittest.TestCase):
    def check_raw(self, raw, error=None):
        parsed = p2.parse_json_bytes(raw)
        before, identities = copy.deepcopy(parsed), identity_tree(parsed.root)
        try:
            if error is None:
                self.assertIsNone(p1.check_p1_structure_and_fixed_inputs(parsed))
            else:
                with self.assertRaises(p2.JSONInputError) as caught:
                    p1.check_p1_structure_and_fixed_inputs(parsed)
                self.assertEqual(caught.exception.code, error)
                self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual(parsed, before)
            self.assertEqual(identity_tree(parsed.root), identities)
        return parsed

    def check_value(self, value, error=None):
        return self.check_raw(encode(value), error)

    def test_minimal_p1_success_is_not_formal_acceptance(self):
        parsed = self.check_value(partial_fixture())
        self.assertIs(type(parsed), p2.ParsedJSON)
        self.assertFalse(hasattr(parsed, "verified"))
        self.assertFalse(hasattr(parsed, "execution_allowed"))
        self.assertEqual(parsed.root["optimization"], {})
        self.assertNotIn("source_path", parsed.root["dataset"])
        self.assertNotIn("directory", parsed.root["output"])

    def test_each_of_ten_top_level_keys_is_required(self):
        for key in partial_fixture():
            with self.subTest(key=key):
                value = partial_fixture()
                del value[key]
                self.check_value(value, "p1_top_level_keys")

    def test_extra_top_level_authorities_even_inactive_are_rejected(self):
        for key in ("version", "profile", "eval", "gaussian_dim", "iterations",
                    "model_path", "run_id", "prefilter_var", "debug", "unknown"):
            for extra in (None, False, 0, "", {}):
                with self.subTest(key=key, extra=extra):
                    value = partial_fixture()
                    value[key] = extra
                    self.check_value(value, "p1_top_level_keys")

    def test_renamed_alias_and_changed_hierarchy_are_rejected(self):
        for key in partial_fixture():
            with self.subTest(key=key):
                value = partial_fixture()
                value[key.upper()] = value.pop(key)
                self.check_value(value, "p1_top_level_keys")
        self.check_value({"config": partial_fixture()}, "p1_top_level_keys")

    def test_each_of_eight_objects_rejects_every_other_json_type(self):
        for key in ("dataset", "model", "renderer", "initialization",
                    "optimization", "reporting", "checkpoint", "output"):
            for wrong in (None, True, False, 0, 1.0, "{}", [], [1]):
                with self.subTest(key=key, wrong=wrong):
                    value = partial_fixture()
                    value[key] = wrong
                    self.check_value(value, "p1_object")

    def test_two_literals_reject_wrong_types(self):
        for key in ("schema", "run_mode"):
            for wrong in (None, True, False, 1, 1.0, [], {}):
                with self.subTest(key=key, wrong=wrong):
                    value = partial_fixture()
                    value[key] = wrong
                    self.check_value(value, "p1_string_type")

    def test_two_literals_reject_other_values_without_normalization(self):
        for key, wrongs in (
                ("schema", ("", "corrected_4dgs_training_config_v2", "v1",
                            "corrected_4dgs_training_config_v1 ")),
                ("run_mode", ("", "pilot", "formal", "resume", "warm_start",
                              "FROM_SCRATCH", " from_scratch"))):
            for wrong in wrongs:
                with self.subTest(key=key, wrong=wrong):
                    value = partial_fixture()
                    value[key] = wrong
                    self.check_value(value, "p1_literal")

    def test_all_fourteen_fixed_fields_are_required(self):
        count = 0
        for group in ("dataset", "model", "renderer"):
            for field in partial_fixture()[group]:
                count += 1
                with self.subTest(group=group, field=field):
                    value = partial_fixture()
                    del value[group][field]
                    self.check_value(value, "p1_missing_fixed")
        self.assertEqual(count, 14)

    def test_empty_fixed_input_objects_do_not_supply_defaults(self):
        for group in ("dataset", "model", "renderer"):
            with self.subTest(group=group):
                value = partial_fixture()
                value[group] = {}
                self.check_value(value, "p1_missing_fixed")

    def test_each_fixed_field_rejects_wrong_json_types(self):
        count = 0
        for group in ("dataset", "model", "renderer"):
            for field, expected in partial_fixture()[group].items():
                count += 1
                if type(expected) is bool:
                    wrongs, code = (0, 1, 1.0, "true", None, [], {}), "bool_type"
                elif field == "scaling_modifier":
                    wrongs, code = (True, False, "1", None, [], {}), "num_type"
                elif type(expected) is int:
                    wrongs = (True, False, float(expected), str(expected), None, [], {})
                    code = "int_type"
                elif type(expected) is str:
                    wrongs, code = (True, False, 0, 1.0, None, [], {}), "p1_string_type"
                else:
                    wrongs, code = (True, False, 0, 1.0, "null", [], {}), "p1_null_type"
                for wrong in wrongs:
                    with self.subTest(group=group, field=field, wrong=wrong):
                        value = partial_fixture()
                        value[group][field] = wrong
                        self.check_value(value, code)
        self.assertEqual(count, 14)

    def test_each_non_null_fixed_field_rejects_wrong_value(self):
        count = 0
        for group in ("dataset", "model", "renderer"):
            for field, expected in partial_fixture()[group].items():
                if expected is None:  # Null has only one value; other types above.
                    continue
                count += 1
                wrong = (not expected if type(expected) is bool else
                         expected + 1 if type(expected) is int else "wrong")
                with self.subTest(group=group, field=field):
                    value = partial_fixture()
                    value[group][field] = wrong
                    self.check_value(value, "p1_fixed_value")
        self.assertEqual(count, 13)

    def test_fixed_strings_are_exact_not_trimmed_or_case_folded(self):
        for group, field in (("dataset", "kind"), ("model", "sh_evaluation"),
                             ("renderer", "temporal_prefilter")):
            expected = partial_fixture()[group][field]
            for wrong in ("", expected.upper(), " " + expected, expected + "\n"):
                with self.subTest(group=group, field=field, wrong=wrong):
                    value = partial_fixture()
                    value[group][field] = wrong
                    self.check_value(value, "p1_fixed_value")

    def test_int_token_origin_for_every_fixed_integer(self):
        for group, field, number in (("model", "gaussian_dim", 4),
                                    ("model", "spatial_sh_degree", 3),
                                    ("model", "temporal_sh_degree", 2),
                                    ("renderer", "env_map_res", 0)):
            for suffix in ("", ".0", "e0"):
                with self.subTest(field=field, suffix=suffix):
                    self.check_raw(with_token(group, field, (str(number) + suffix).encode()),
                                   None if not suffix else "int_type")
            for token in (b'-1', b'2147483648'):
                with self.subTest(field=field, token=token):
                    self.check_raw(with_token(group, field, token), "int_range")

    def test_scaling_accepts_num_one_without_changing_token_origin(self):
        for token in (b'1', b'1.0', b'1e0', b'10e-1'):
            with self.subTest(token=token):
                parsed = self.check_raw(with_token("renderer", "scaling_modifier", token))
                self.assertEqual(parsed.root["renderer"]["scaling_modifier"].raw,
                                 token.decode())

    def test_scaling_rejects_nonunit_without_tolerance(self):
        for token in (b'0', b'-1', b'2', b'0.9999999999999999', b'1.0000000000000002'):
            with self.subTest(token=token):
                self.check_raw(with_token("renderer", "scaling_modifier", token),
                               "p1_fixed_value")

    def test_scaling_reuses_p2_overflow_and_underflow_rejection(self):
        for token, code in ((b'1e309', 'num_overflow'), (b'1e-9999', 'num_underflow')):
            with self.subTest(token=token):
                self.check_raw(with_token("renderer", "scaling_modifier", token), code)

    def test_explicit_null_is_distinct_from_missing(self):
        parsed = self.check_value(partial_fixture())
        self.assertIn("override_color", parsed.root["renderer"])
        self.assertIsNone(parsed.root["renderer"]["override_color"])
        value = partial_fixture()
        del value["renderer"]["override_color"]
        self.check_value(value, "p1_missing_fixed")

    def test_nested_unchecked_data_preserved_not_adopted(self):
        value = partial_fixture()
        for group in ("dataset", "model", "renderer", "initialization",
                      "optimization", "reporting", "checkpoint", "output"):
            value[group]["UNREVIEWED"] = {"value": [None, False, "日本語", 2.5]}
        parsed = self.check_value(value)
        for group in value.keys() - {"schema", "run_mode"}:
            self.assertEqual(parsed.root[group]["UNREVIEWED"]["value"][3].raw, "2.5")

    def test_other_owner_semantics_and_adapters_are_not_run(self):
        value = partial_fixture()
        value["dataset"]["resolution"] = "not-P4-validated"
        value["dataset"]["time"] = {"raw_interval": "not-D-TIME-validated"}
        value["initialization"]["time_variance_denominator"] = None
        value["optimization"]["total_updates"] = "not-P5-validated"
        value["output"]["directory"] = "not-P6-validated"
        parsed = self.check_value(value)
        self.assertNotIn("prefilter_var", parsed.root["renderer"])
        self.assertNotIn("eval_shfs_4d", parsed.root["model"])
        self.assertNotIn("save_updates", parsed.root["checkpoint"])

    def test_p2_numeric_helpers_are_called_not_reimplemented(self):
        with patch.object(p2, "require_int", wraps=p2.require_int) as ints, \
                patch.object(p2, "require_num", wraps=p2.require_num) as nums, \
                patch.object(p2, "require_bool", wraps=p2.require_bool) as bools:
            self.check_value(partial_fixture())
            self.assertEqual(ints.call_count, 4)
            self.assertEqual(nums.call_count, 1)
            self.assertEqual(bools.call_count, 5)

    def test_wrong_python_entry_is_not_coerced(self):
        for value in (None, {}, partial_fixture(), b'{}', p2.ParsedJSON([])):
            with self.subTest(kind=type(value).__name__):
                before = copy.deepcopy(value)
                with self.assertRaises(p1.P1InputError) as caught:
                    p1.check_p1_structure_and_fixed_inputs(value)
                self.assertEqual(caught.exception.code, "p1_parsed_input")
                self.assertEqual(value, before)

    def test_errors_do_not_echo_private_input(self):
        value = partial_fixture()
        value["PRIVATE_PATH_MARKER"] = "private-value"
        self.check_value(value, "p1_top_level_keys")
        value = partial_fixture()
        value["dataset"]["kind"] = "private-value"
        self.check_value(value, "p1_fixed_value")

    def test_p2_duplicate_keys_rejected_before_p1(self):
        raw = encode(partial_fixture())
        raw = raw.replace(b'"eval":true', b'"eval":true,"\\u0065val":true')
        with patch.object(p1, "check_p1_structure_and_fixed_inputs") as check:
            with self.assertRaises(p2.JSONInputError) as caught:
                check(p2.parse_json_bytes(raw))
            self.assertEqual(caught.exception.code, "duplicate_key")
            check.assert_not_called()


class IsolationTests(unittest.TestCase):
    def test_imports_are_stdlib_or_the_two_components_only(self):
        for path in (ROOT / 'formal_config_p1.py', Path(__file__)):
            names = set()
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    names.update(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names |
                                 {'formal_config_json', 'formal_config_p1'})

    def test_isolated_success_and_failure_no_heavy_import_or_writer(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython package unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1'}:
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
import formal_config_p1 as p1
data = json.loads(sys.argv[2])
def check(value):
    return p1.check_p1_structure_and_fixed_inputs(
        p2.parse_json_bytes(json.dumps(value).encode()))
assert check(data) is None
invalid = [{}, dict(data, unknown=None), dict(data, schema='v2')]
for group, field, bad in (('dataset','eval',1), ('model','gaussian_dim',4.0),
                          ('renderer','scaling_modifier',True),
                          ('renderer','override_color',False)):
    value = json.loads(sys.argv[2])
    value[group][field] = bad
    invalid.append(value)
for value in invalid:
    try:
        check(value)
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('invalid P1 input accepted')
assert not any(x.split('.')[0] in {'torch', 'scene', 'gaussian_renderer', 'train'}
               for x in sys.modules)
print(json.dumps({'stdlib_only':True, 'no_writer':True, 'no_heavy_import':True}))
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-I', '-B', '-S', '-c', script, str(ROOT),
                 encode(partial_fixture()).decode()], cwd=directory,
                text=True, capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout),
                             {'stdlib_only': True, 'no_writer': True, 'no_heavy_import': True})
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
