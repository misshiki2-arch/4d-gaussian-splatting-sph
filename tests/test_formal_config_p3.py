"""Independent P1/P3 partial fixtures, NOT runnable formal configurations.

Run with -B -S -m unittest discover -s tests -p test_formal_config_p3.py -v.
All expectations come from the adopted contract, not implementation tables.
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
import formal_config_p3 as p3


def partial_fixture():
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


def token_fixture(field, raw):
    value = partial_fixture()
    value["model"][field] = "TOKEN_MARKER"
    return encode(value).replace(b'"TOKEN_MARKER"', raw)


def identity_tree(value):
    if type(value) is dict:
        return (id(value), [(k, identity_tree(v)) for k, v in value.items()])
    if type(value) is list:
        return (id(value), [identity_tree(v) for v in value])
    return (id(value), type(value), value)


class P3Tests(unittest.TestCase):
    def check_parsed(self, parsed, error=None):
        before, identities = copy.deepcopy(parsed), identity_tree(parsed.root)
        try:
            if error is None:
                return p3.derive_p3_maximum_sh(parsed)
            with self.assertRaises(p2.JSONInputError) as caught:
                p3.derive_p3_maximum_sh(parsed)
            self.assertEqual(caught.exception.code, error)
            self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual(parsed, before)
            self.assertEqual(identity_tree(parsed.root), identities)

    def check_value(self, value, error=None):
        return self.check_parsed(p2.parse_json_bytes(encode(value)), error)

    def test_derive_maximum_and_inclusive_layout(self):
        result = self.check_value(partial_fixture())
        self.assertEqual(asdict(result), {
            "max_spatial_degree": 3, "max_temporal_degree": 2,
            "slot_count": 48,
            "mode_slot_ranges": ((0, 15), (16, 31), (32, 47)),
            "legacy_eval_shfs_4d": True,
        })
        self.assertIs(type(result.slot_count), int)
        self.assertIs(result.legacy_eval_shfs_4d, True)
        self.assertEqual([i for lo, hi in result.mode_slot_ranges
                          for i in range(lo, hi + 1)], list(range(48)))

    def test_result_is_detached_partial_not_execution_state(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        result = self.check_parsed(parsed)
        for name in ("verified", "execution_allowed", "active_sh_degree",
                     "active_sh_degree_t", "schedule", "root"):
            self.assertFalse(hasattr(result, name))
        with self.assertRaises(FrozenInstanceError):
            result.slot_count = 1
        with self.assertRaises(TypeError):
            result.mode_slot_ranges[0][0] = 1
        parsed.root["model"]["spatial_sh_degree"] = p2.NumberToken("0")
        self.assertEqual(result.max_spatial_degree, 3)

    def test_each_degree_required(self):
        for field in ("spatial_sh_degree", "temporal_sh_degree"):
            with self.subTest(field=field):
                value = partial_fixture()
                del value["model"][field]
                self.check_value(value, "p1_missing_fixed")

    def test_both_degrees_reject_noninteger_json_types(self):
        for field, integer in (("spatial_sh_degree", "3"),
                               ("temporal_sh_degree", "2")):
            for raw in (b'true', b'false', b'null', b'[]', b'{}',
                        ('"' + integer + '"').encode(),
                        (integer + '.0').encode(), (integer + 'e0').encode()):
                with self.subTest(field=field, raw=raw):
                    self.check_parsed(p2.parse_json_bytes(token_fixture(field, raw)),
                                      "int_type")

    def test_both_degrees_reject_unsupported_and_out_of_range(self):
        for field, fixed in (("spatial_sh_degree", 3), ("temporal_sh_degree", 2)):
            for value in (0, 1, 2, 3, 4, 2147483647, -1, 2147483648):
                if value == fixed:
                    continue
                with self.subTest(field=field, value=value):
                    data = partial_fixture()
                    data["model"][field] = value
                    self.check_value(data, "int_range" if value < 0 or value > 2147483647
                                     else "p1_fixed_value")

    def test_wrong_python_entry_has_no_coercion_or_token_bypass(self):
        for value in (None, {}, partial_fixture(), b'{}', p2.ParsedJSON([])):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(p1.P1InputError):
                    p3.derive_p3_maximum_sh(value)
        for field in ("spatial_sh_degree", "temporal_sh_degree"):
            for value in (True, 3, 2, 3.0, "2"):
                parsed = p2.parse_json_bytes(encode(partial_fixture()))
                parsed.root["model"][field] = value
                self.check_parsed(parsed, "int_type")

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

    def test_all_p1_fixed_fields_are_rechecked(self):
        for group in ("dataset", "model", "renderer"):
            for field, good in partial_fixture()[group].items():
                with self.subTest(group=group, field=field):
                    data = partial_fixture()
                    del data[group][field]
                    self.check_value(data, "p1_missing_fixed")
                    data = partial_fixture()
                    bad = (not good if type(good) is bool else
                           good + 1 if type(good) is int else "wrong")
                    data[group][field] = bad
                    self.check_value(data, "p1_null_type" if good is None
                                     else "p1_fixed_value")

    def test_model_competing_paths_reject_all_values_including_inactive(self):
        # Explicit independent path matrix, not copied from implementation.
        for field in ("eval_shfs_4d", "temporal_sh_enabled", "sh_layout",
                      "sh_slot_count", "sh_degree", "sh_degree_t",
                      "max_sh_degree", "max_sh_degree_t", "max_sh_channels"):
            for extra in (True, False, None, 0, 48, "", [], {}):
                with self.subTest(path="model." + field, extra=extra):
                    data = partial_fixture()
                    data["model"][field] = extra
                    self.check_value(data, "p3_competing_input")

    def test_renderer_competing_paths_reject_inactive_inputs(self):
        for field in ("eval_shfs_4d", "temporal_sh_enabled", "sh_layout", "sh_slot_count"):
            for extra in (True, False, None, 0, 48, "", [], {}):
                with self.subTest(path="renderer." + field, extra=extra):
                    data = partial_fixture()
                    data["renderer"][field] = extra
                    self.check_value(data, "p3_competing_input")

    def test_flat_legacy_and_derived_input_rejected_by_p1(self):
        for field in ("eval_shfs_4d", "temporal_sh_enabled", "sh_layout",
                      "sh_slot_count", "sh_degree", "sh_degree_t",
                      "max_sh_degree", "max_sh_degree_t", "max_sh_channels"):
            data = partial_fixture()
            data[field] = None
            self.check_value(data, "p1_top_level_keys")

    def test_legacy_boolean_cannot_supply_missing_temporal_degree(self):
        for group in ("model", "renderer"):
            for flag in (True, False, None):
                data = partial_fixture()
                del data["model"]["temporal_sh_degree"]
                data[group]["eval_shfs_4d"] = flag
                self.check_value(data, "p1_missing_fixed")

    def test_previous_p1_success_is_not_trusted(self):
        for change, code in (("degree", "p1_fixed_value"),
                             ("flag", "p3_competing_input"),
                             ("fixed", "p1_fixed_value")):
            parsed = p2.parse_json_bytes(encode(partial_fixture()))
            self.assertIsNone(p1.check_p1_structure_and_fixed_inputs(parsed))
            if change == "degree":
                parsed.root["model"]["temporal_sh_degree"] = p2.NumberToken("0")
            elif change == "flag":
                parsed.root["model"]["temporal_sh_enabled"] = False
            else:
                parsed.root["dataset"]["eval"] = False
            self.check_parsed(parsed, code)

    def test_reuses_p1_and_p2_int_helpers(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        with patch.object(p1, "check_p1_structure_and_fixed_inputs",
                          wraps=p1.check_p1_structure_and_fixed_inputs) as check, \
                patch.object(p2, "require_int", wraps=p2.require_int) as ints:
            self.check_parsed(parsed)
            check.assert_called_once_with(parsed)
            self.assertEqual(ints.call_count, 6)  # Four in P1, two for derivation.
            self.assertIs(ints.call_args_list[-2].args[0],
                          parsed.root["model"]["spatial_sh_degree"])
            self.assertIs(ints.call_args_list[-1].args[0],
                          parsed.root["model"]["temporal_sh_degree"])

    def test_unchecked_nested_data_is_not_completed_or_adopted(self):
        data = partial_fixture()
        for group in ("dataset", "model", "renderer", "initialization",
                      "optimization", "reporting", "checkpoint", "output"):
            data[group]["UNREVIEWED"] = {"items": [None, False, 1.5, "日本語"]}
        data["dataset"]["resolution"] = "not-P4-validated"
        data["output"]["directory"] = "not-P6-validated"
        parsed = p2.parse_json_bytes(encode(data))
        self.assertEqual(self.check_parsed(parsed).slot_count, 48)
        self.assertNotIn("eval_shfs_4d", parsed.root["model"])
        self.assertNotIn("sh_slot_count", parsed.root["model"])
        self.assertEqual(parsed.root["model"]["temporal_sh_degree"].raw, "2")
        parsed.root["model"]["sh_slot_count"] = None
        self.check_parsed(parsed, "p3_competing_input")

    def test_active_state_neither_consumed_nor_used_as_maximum(self):
        data = partial_fixture()
        data["model"].update(active_sh_degree=0, active_sh_degree_t=0)
        data["optimization"]["sh_increase_interval"] = "unresolved"
        self.assertEqual(self.check_value(data).slot_count, 48)
        del data["model"]["temporal_sh_degree"]
        data["model"]["active_sh_degree_t"] = 2
        self.check_value(data, "p1_missing_fixed")

    def test_errors_do_not_echo_input_or_path(self):
        data = partial_fixture()
        data["model"]["sh_layout"] = "/private/path/DO_NOT_ECHO"
        self.check_value(data, "p3_competing_input")

    def test_duplicate_decoded_degree_rejected_before_derivation(self):
        raw = encode(partial_fixture()).replace(
            b'"temporal_sh_degree":2',
            b'"temporal_sh_degree":2,"temporal_sh_\\u0064egree":2')
        with patch.object(p3, "derive_p3_maximum_sh") as derive:
            with self.assertRaises(p2.JSONInputError) as caught:
                derive(p2.parse_json_bytes(raw))
            self.assertEqual(caught.exception.code, "duplicate_key")
            derive.assert_not_called()


class IsolationTests(unittest.TestCase):
    def test_imports_are_stdlib_or_components(self):
        for path in (ROOT / 'formal_config_p3.py', Path(__file__)):
            names = set()
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    names.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names |
                                 {'formal_config_json', 'formal_config_p1', 'formal_config_p3'})

    def test_isolated_success_failure_no_heavy_import_or_writer(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython package unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_p3'}:
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
import formal_config_p3 as p3
def check(data):
    return p3.derive_p3_maximum_sh(p2.parse_json_bytes(json.dumps(data).encode()))
data = json.loads(sys.argv[2])
assert check(data).slot_count == 48
invalid = [{}, dict(data, unknown=None)]
for field, bad in (('temporal_sh_degree', 0), ('spatial_sh_degree', 3.0),
                   ('eval_shfs_4d', False), ('sh_slot_count', 48)):
    value = json.loads(sys.argv[2])
    value['model'][field] = bad
    invalid.append(value)
for value in invalid:
    try:
        check(value)
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('invalid input accepted')
assert not any(x.split('.')[0] in {'torch','scene','gaussian_renderer','train'}
               for x in sys.modules)
print(json.dumps({'stdlib_only':True,'no_writer':True,'no_heavy_import':True}))
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
