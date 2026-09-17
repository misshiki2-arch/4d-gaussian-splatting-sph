"""Independent P1/P5 partial fixtures, NOT runnable formal configurations.

Run with -B -S -m unittest discover -s tests -p test_formal_config_p5.py -v.
Expected values come from the plan, not implementation tables or actual output.
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
import formal_config_p5 as p5


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
        "initialization": {}, "optimization": {"total_updates": 10},
        "reporting": {"test_updates": [3, 10]},
        "checkpoint": {"intermediate_updates": [2, 7]}, "output": {},
    }


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def identity_tree(value):
    if type(value) is dict:
        return (id(value), [(k, identity_tree(v)) for k, v in value.items()])
    if type(value) is list:
        return (id(value), [identity_tree(v) for v in value])
    return (id(value), type(value), value)


def token_fixture(group, field, raw, element=False):
    data = partial_fixture()
    data[group][field] = ["TOKEN_MARKER"] if element else "TOKEN_MARKER"
    return encode(data).replace(b'"TOKEN_MARKER"', raw)


SCHEDULE_PATHS = (("checkpoint", "intermediate_updates"),
                  ("reporting", "test_updates"))


class P5Tests(unittest.TestCase):
    def check_parsed(self, parsed, error=None):
        before, identities = copy.deepcopy(parsed), identity_tree(parsed.root)
        try:
            if error is None:
                return p5.derive_p5_schedules(parsed)
            with self.assertRaises(p2.JSONInputError) as caught:
                p5.derive_p5_schedules(parsed)
            self.assertEqual(caught.exception.code, error)
            self.assertEqual(str(caught.exception), error)
        finally:
            self.assertEqual(parsed, before)
            self.assertEqual(identity_tree(parsed.root), identities)

    def check_value(self, data, error=None):
        return self.check_parsed(p2.parse_json_bytes(encode(data)), error)

    def test_normal_schedules_from_one_total(self):
        result = self.check_value(partial_fixture())
        self.assertEqual(asdict(result), {
            "total_updates": 10, "save_updates": (2, 7, 10),
            "test_updates": (3, 10),
        })
        self.assertIs(type(result.total_updates), int)
        self.assertIs(type(result.save_updates), tuple)
        self.assertIs(type(result.test_updates), tuple)
        self.assertEqual(result.save_updates.count(10), 1)

    def test_one_update_and_explicit_empty_schedules(self):
        for tests in ([], [1]):
            data = partial_fixture()
            data["optimization"]["total_updates"] = 1
            data["checkpoint"]["intermediate_updates"] = []
            data["reporting"]["test_updates"] = tests
            result = self.check_value(data)
            self.assertEqual(result.save_updates, (1,))
            self.assertEqual(result.test_updates, () if not tests else (1,))

    def test_no_automatic_final_test_or_contiguous_expansion(self):
        for total in (10, 2147483647):
            data = partial_fixture()
            data["optimization"]["total_updates"] = total
            data["checkpoint"]["intermediate_updates"] = []
            data["reporting"]["test_updates"] = []
            result = self.check_value(data)
            self.assertEqual(result.save_updates, (total,))
            self.assertEqual(result.test_updates, ())
        data["checkpoint"]["intermediate_updates"] = [1, 2147483646]
        data["reporting"]["test_updates"] = [1, 2147483647]
        result = self.check_value(data)
        self.assertEqual(result.save_updates, (1, 2147483646, 2147483647))
        self.assertEqual(result.test_updates, (1, 2147483647))
        data["reporting"]["test_updates"] = [1]
        self.assertEqual(self.check_value(data).test_updates, (1,))

    def test_intermediate_final_rejects_but_test_final_accepts(self):
        for total in (1, 10, 2147483647):
            data = partial_fixture()
            data["optimization"]["total_updates"] = total
            data["checkpoint"]["intermediate_updates"] = [total]
            data["reporting"]["test_updates"] = [total]
            self.check_value(data, "p5_update_range")
            data["checkpoint"]["intermediate_updates"] = []
            result = self.check_value(data)
            self.assertEqual(result.save_updates, (total,))
            self.assertEqual(result.test_updates, (total,))

    def test_each_input_is_required_without_default(self):
        for group, field in (("optimization", "total_updates"), *SCHEDULE_PATHS):
            data = partial_fixture()
            del data[group][field]
            self.check_value(data, "p5_missing_input")

    def test_total_integer_domain(self):
        for raw, code in ((b'0', 'p5_total_range'), (b'-0', 'p5_total_range'),
                          (b'-1', 'int_range'), (b'2147483648', 'int_range'),
                          (b'9' * 5000, 'int_range')):
            self.check_parsed(p2.parse_json_bytes(token_fixture(
                "optimization", "total_updates", raw)), code)

    def test_total_requires_integer_token(self):
        for raw in (b'true', b'false', b'null', b'10.0', b'1e1', b'"10"',
                    b'"1+1"', b'[]', b'{}'):
            self.check_parsed(p2.parse_json_bytes(token_fixture(
                "optimization", "total_updates", raw)), "int_type")

    def test_arrays_require_explicit_json_array(self):
        for group, field in SCHEDULE_PATHS:
            for wrong in (None, False, True, 0, 1.0, "[]", {}, "range(10)"):
                with self.subTest(group=group, wrong=wrong):
                    data = partial_fixture()
                    data[group][field] = wrong
                    self.check_value(data, "p5_array_type")

    def test_all_elements_require_integer_tokens(self):
        for group, field in SCHEDULE_PATHS:
            for raw in (b'true', b'false', b'null', b'1.0', b'1e0', b'"1"',
                        b'"1+1"', b'[]', b'{}'):
                with self.subTest(group=group, raw=raw):
                    self.check_parsed(p2.parse_json_bytes(token_fixture(
                        group, field, raw, element=True)), "int_type")
            for values in ([1, 2.0], [1, 2, False], [1, None, 3]):
                data = partial_fixture()
                data[group][field] = values
                self.check_value(data, "int_type")

    def test_element_range_rejection(self):
        for group, field in SCHEDULE_PATHS:
            for value, code in ((0, "p5_update_range"), (-1, "int_range"),
                                (11, "p5_update_range"),
                                (2147483648, "int_range")):
                data = partial_fixture()
                data[group][field] = [value]
                self.check_value(data, code)
            self.check_parsed(p2.parse_json_bytes(token_fixture(
                group, field, b'-0', element=True)), "p5_update_range")

    def test_strict_order_no_sort_or_deduplication(self):
        for group, field in SCHEDULE_PATHS:
            for values in ([1, 1], [2, 1], [1, 3, 2], [1, 2, 2], [3, 2, 1]):
                with self.subTest(group=group, values=values):
                    data = partial_fixture()
                    data[group][field] = values
                    self.check_value(data, "p5_update_order")

    def test_named_competing_inputs_at_each_schedule_object(self):
        # Independent matrix: P5's two explicit names at its three objects.
        for group in ("optimization", "checkpoint", "reporting"):
            for field in ("final_iteration", "save_iterations"):
                for extra in (False, None, [], {}, 0, "", 10, [10]):
                    with self.subTest(group=group, field=field, extra=extra):
                        data = partial_fixture()
                        data[group][field] = extra
                        self.check_value(data, "p5_competing_input")

    def test_flat_legacy_authorities_rejected_by_p1(self):
        for field in ("final_iteration", "save_iterations", "iterations",
                      "test_iterations", "total_updates", "save_updates"):
            for extra in (None, False, [], 10):
                data = partial_fixture()
                data[field] = extra
                self.check_value(data, "p1_top_level_keys")

    def test_competing_inputs_cannot_complete_missing_total(self):
        for group in ("optimization", "checkpoint", "reporting"):
            for field in ("final_iteration", "save_iterations"):
                data = partial_fixture()
                del data["optimization"]["total_updates"]
                data[group][field] = 10
                self.check_value(data, "p5_competing_input")

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

    def test_all_p1_fixed_inputs_rechecked(self):
        for group in ("dataset", "model", "renderer"):
            for field, good in partial_fixture()[group].items():
                with self.subTest(group=group, field=field):
                    data = partial_fixture()
                    del data[group][field]
                    self.check_value(data, "p1_missing_fixed")
                    data = partial_fixture()
                    data[group][field] = (not good if type(good) is bool else
                                          good + 1 if type(good) is int else "wrong")
                    self.check_value(data, "p1_null_type" if good is None
                                     else "p1_fixed_value")

    def test_past_success_does_not_bypass_current_input_checks(self):
        for change, code in (("fixed", "p1_fixed_value"),
                             ("total", "p5_total_range"),
                             ("order", "p5_update_order"),
                             ("competing", "p5_competing_input"),
                             ("missing", "p5_missing_input")):
            parsed = p2.parse_json_bytes(encode(partial_fixture()))
            self.assertIsNone(p1.check_p1_structure_and_fixed_inputs(parsed))
            self.check_parsed(parsed)
            if change == "fixed":
                parsed.root["dataset"]["eval"] = False
            elif change == "total":
                parsed.root["optimization"]["total_updates"] = p2.NumberToken("0")
            elif change == "order":
                parsed.root["reporting"]["test_updates"].reverse()
            elif change == "competing":
                parsed.root["checkpoint"]["save_iterations"] = None
            else:
                del parsed.root["checkpoint"]["intermediate_updates"]
            self.check_parsed(parsed, code)

    def test_reuses_current_p1_and_p2_integer_checks(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        with patch.object(p1, "check_p1_structure_and_fixed_inputs",
                          wraps=p1.check_p1_structure_and_fixed_inputs) as check, \
                patch.object(p2, "require_int", wraps=p2.require_int) as ints:
            self.check_parsed(parsed)
            check.assert_called_once_with(parsed)
            self.assertEqual(ints.call_count, 9)  # Four P1 + N + four updates.
            expected = [parsed.root["optimization"]["total_updates"],
                        *parsed.root["checkpoint"]["intermediate_updates"],
                        *parsed.root["reporting"]["test_updates"]]
            for call, token in zip(ints.call_args_list[-5:], expected):
                self.assertIs(call.args[0], token)

    def test_wrong_python_entry_and_numeric_token_bypass(self):
        for value in (None, {}, partial_fixture(), b'{}', p2.ParsedJSON([])):
            with self.assertRaises(p1.P1InputError):
                p5.derive_p5_schedules(value)
        for group, field in (("optimization", "total_updates"), *SCHEDULE_PATHS):
            for value in (True, 1, 1.0, "1"):
                parsed = p2.parse_json_bytes(encode(partial_fixture()))
                parsed.root[group][field] = value if group == "optimization" else [value]
                self.check_parsed(parsed, "int_type")
        for group, field in SCHEDULE_PATHS:
            parsed = p2.parse_json_bytes(encode(partial_fixture()))
            parsed.root[group][field] = tuple(parsed.root[group][field])
            self.check_parsed(parsed, "p5_array_type")

    def test_detached_immutable_result_is_only_partial(self):
        parsed = p2.parse_json_bytes(encode(partial_fixture()))
        result = self.check_parsed(parsed)
        for name in ("verified", "execution_allowed", "root", "runtime"):
            self.assertFalse(hasattr(result, name))
        for field in ("total_updates", "save_updates", "test_updates"):
            with self.assertRaises(FrozenInstanceError):
                setattr(result, field, None)
        with self.assertRaises(TypeError):
            result.save_updates[0] = 9
        with self.assertRaises(TypeError):
            result.test_updates[0] = 9
        parsed.root["optimization"]["total_updates"] = p2.NumberToken("20")
        parsed.root["checkpoint"]["intermediate_updates"].clear()
        parsed.root["reporting"]["test_updates"].append(p2.NumberToken("20"))
        self.assertEqual(asdict(result), {"total_updates": 10,
                                         "save_updates": (2, 7, 10),
                                         "test_updates": (3, 10)})
        later = self.check_parsed(parsed)
        self.assertEqual(later.save_updates, (20,))
        self.assertEqual(later.test_updates, (3, 10, 20))

    def test_unchecked_fields_are_not_adopted_or_rewritten(self):
        data = partial_fixture()
        for group in ("dataset", "model", "renderer", "initialization",
                      "optimization", "checkpoint", "reporting", "output"):
            data[group]["UNREVIEWED"] = {"save_iterations": [None, 2.5, "日本語"]}
        data["dataset"]["resolution"] = "not-P4-validated"
        data["output"]["directory"] = "not-P6-validated"
        data["model"]["eval_shfs_4d"] = False  # P3 is not run by P5.
        data["initialization"]["time_variance_denominator"] = None
        parsed = p2.parse_json_bytes(encode(data))
        self.assertEqual(self.check_parsed(parsed).save_updates, (2, 7, 10))
        self.assertNotIn("save_updates", parsed.root["checkpoint"])
        self.assertNotIn("source_path", parsed.root["dataset"])

    def test_maximum_input_array_does_not_truncate_derived_final(self):
        data = partial_fixture()
        data["optimization"]["total_updates"] = 4097
        data["checkpoint"]["intermediate_updates"] = list(range(1, 4097))
        data["reporting"]["test_updates"] = list(range(2, 4098))
        result = self.check_value(data)
        self.assertEqual(result.save_updates, tuple(range(1, 4098)))
        self.assertEqual(len(result.save_updates), 4097)
        self.assertEqual(result.save_updates.count(4097), 1)
        self.assertEqual(result.test_updates, tuple(range(2, 4098)))

    def test_mutated_input_array_still_obeys_p2_bound(self):
        for group, field in SCHEDULE_PATHS:
            parsed = p2.parse_json_bytes(encode(partial_fixture()))
            parsed.root[group][field] = [p2.NumberToken("1")] * 4097
            self.check_parsed(parsed, "p5_array_limit")

    def test_p2_syntax_and_size_reject_before_p5(self):
        raw = encode(partial_fixture())
        invalid = [(raw.replace(b'"total_updates":10',
                               b'"total_updates":10,"total_\\u0075pdates":10'),
                    "duplicate_key"),
                   (token_fixture("optimization", "total_updates", b'1+1'),
                    "json_syntax"),
                   (b'\xef\xbb\xbf' + raw, "bom"),
                   (raw + b' {}', "trailing_value"),
                   (raw + b' ' * (262145 - len(raw)), "input_bytes")]
        for group, field in SCHEDULE_PATHS:
            data = partial_fixture()
            data[group][field] = [1] * 4097
            invalid.append((encode(data), "array_elements"))
        for raw, code in invalid:
            with patch.object(p5, "derive_p5_schedules") as derive:
                with self.assertRaises(p2.JSONInputError) as caught:
                    derive(p2.parse_json_bytes(raw))
                self.assertEqual(caught.exception.code, code)
                derive.assert_not_called()

    def test_errors_never_echo_private_input(self):
        data = partial_fixture()
        data["optimization"]["total_updates"] = "/private/P5_DO_NOT_ECHO"
        self.check_value(data, "int_type")
        data = partial_fixture()
        data["checkpoint"]["save_iterations"] = "/private/P5_DO_NOT_ECHO"
        self.check_value(data, "p5_competing_input")


class IsolationTests(unittest.TestCase):
    def test_imports_are_stdlib_or_p1_p2_p5_only(self):
        for path in (ROOT / 'formal_config_p5.py', Path(__file__)):
            names = set()
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.Import):
                    names.update(a.name.split('.')[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split('.')[0])
            self.assertLessEqual(names, sys.stdlib_module_names |
                                 {'formal_config_json', 'formal_config_p1',
                                  'formal_config_p5'})

    def test_isolated_no_heavy_import_writer_or_dataset_access(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython package unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_p5'}:
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
import formal_config_p5 as p5
# After module loading, neither input checking nor derivation needs filesystem IO.
def no_files(event, args):
    if event in ('open', 'os.listdir', 'os.scandir'):
        raise AssertionError('filesystem input or scan attempted')
sys.addaudithook(no_files)
def check(data):
    return p5.derive_p5_schedules(p2.parse_json_bytes(json.dumps(data).encode()))
data = json.loads(sys.argv[2])
assert check(data).save_updates == (2, 7, 10)
invalid = [{}, dict(data, unknown=None)]
for group, field, bad in (('optimization','total_updates',0),
                         ('checkpoint','intermediate_updates',[10]),
                         ('reporting','test_updates',[3,3]),
                         ('checkpoint','save_iterations',False)):
    value = json.loads(sys.argv[2])
    value[group][field] = bad
    invalid.append(value)
for value in invalid:
    try:
        check(value)
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('invalid input accepted')
data['optimization']['total_updates'] = 2147483647
data['checkpoint']['intermediate_updates'] = []
data['reporting']['test_updates'] = []
assert check(data).save_updates == (2147483647,)
assert not any(x.split('.')[0] in {'torch','scene','gaussian_renderer','train',
                                  'formal_config_p3'} for x in sys.modules)
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
