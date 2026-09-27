"""CPU-only integration fixtures, never actual run choices or training proof."""

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal_config as config
import formal_config_json as p2


def fixture(source, output):
    # Independent complete fixture, not generated from implementation schema.
    return {
        "schema": "corrected_4dgs_training_config_v1", "run_mode": "from_scratch",
        "dataset": {"kind": "nerf_transforms", "source_path": str(source),
                    "eval": True, "extension": "", "white_background": False,
                    "resolution": {"mode": "integer_divisor", "divisor": 2},
                    "time": {"raw_interval": [-2, 6], "divisor": 2}},
        "model": {"gaussian_dim": 4, "spatial_sh_degree": 3, "temporal_sh_degree": 2,
                  "rot_4d": True, "force_sh_3d": False, "sh_evaluation": "conditional_mean"},
        "renderer": {"compute_cov3D_python": False, "convert_SHs_python": False,
                     "scaling_modifier": 1, "env_map_res": 0,
                     "override_color": None, "temporal_prefilter": "disabled"},
        "initialization": {"num_pts": 7, "num_extra_pts": 0,
                           "time_variance_denominator": 2, "seed": 23},
        "optimization": {
            "total_updates": 12, "batch_size": 3,
            "learning_rate": {"position_lr_init": 0.2, "position_lr_final": 0.01,
                              "position_t_lr_init": 0.03, "position_lr_max_steps": 20,
                              "feature_lr": 0.4, "opacity_lr": 0.5,
                              "scaling_lr": 0.6, "rotation_lr": 0.7},
            "loss": {"lambda_dssim": 0.25, "lambda_opa_mask": 0.125,
                     "lambda_rigid": 0, "lambda_motion": 0},
            "population": {"densify_stop_points": "unlimited", "percent_dense": 2,
                           "thresh_opa_prune": 0, "densify_grad_threshold": 0,
                           "densify_from_update": 20, "densify_until_update": 2,
                           "densification_interval": 3, "opacity_reset_interval": 2},
            "sh": {"increase_interval": 5}},
        "reporting": {"test_updates": [2, 12]},
        "checkpoint": {"intermediate_updates": [3, 9]},
        "output": {"directory": str(output)},
    }


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def locate(obj, path):
    for key in path:
        obj = obj[key]
    return obj


def objects(obj, path=()):
    yield path, obj
    for key, value in obj.items():
        if type(value) is dict:
            yield from objects(value, (*path, key))


def snapshot(root):
    # Only isolated test fixtures; access times are not part of the contract.
    return {str(p.relative_to(root)): (p.lstat().st_ino, p.lstat().st_mode,
                                       p.lstat().st_size, p.lstat().st_mtime_ns)
            for p in (root, *root.rglob("*"))}


class IntegratedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "data"
        self.source.mkdir()
        self.output = self.base / "missing-parent" / "output"
        self.legacy = self.base / "legacy"
        self.legacy.mkdir()
        guard = patch.object(config.p6, "_LEGACY_OUTPUT", str(self.legacy))
        guard.start()
        self.addCleanup(guard.stop)

    def fixture(self):
        return fixture(self.source, self.output)

    def resolve(self, value=None, **dimensions):
        if value is None:
            value = self.fixture()
        data = value if type(value) is bytes else encode(value)
        return config.resolve_formal_config(
            data, raw_width=dimensions.get("raw_width", 120),
            raw_height=dimensions.get("raw_height", 80))

    def reject(self, value, code=None, **dimensions):
        before = snapshot(self.base)
        with self.assertRaises(p2.JSONInputError) as caught:
            self.resolve(value, **dimensions)
        error = str(caught.exception)
        self.assertEqual(error, caught.exception.code)
        self.assertLess(len(error), 64)
        self.assertNotIn(str(self.base), error)
        if code is not None:
            self.assertEqual(error, code)
        self.assertEqual(snapshot(self.base), before)

    def token(self, path, raw):
        data = self.fixture()
        locate(data, path[:-1])[path[-1]] = "TOKEN_FIXTURE_ONLY"
        return encode(data).replace(b'"TOKEN_FIXTURE_ONLY"', raw.encode())

    def test_complete_inputs_retained_and_independent_derived_expected(self):
        original = self.fixture()
        before = copy.deepcopy(original)
        state = self.resolve(original)
        self.assertEqual(original, before)
        # Every input leaf must survive in the typed semantic state. Arrays
        # intentionally become tuples; numeric tokens become checked scalars.
        def retained(value, result):
            if isinstance(value, dict):
                for key, item in value.items():
                    retained(item, getattr(result, key))
            elif isinstance(value, list):
                self.assertIs(type(result), tuple)
                self.assertEqual(tuple(value), result)
            else:
                self.assertEqual(value, result)
        retained(original, state)
        self.assertEqual(state.dataset.resolution, ("integer_divisor", 2, 120, 80, 60, 40))
        self.assertEqual(state.maximum_sh, (3, 2, 48, ((0, 15), (16, 31), (32, 47)), True))
        self.assertEqual(state.time_derivation, ((-2.0, 6.0), (-1.0, 3.0), 8.0, 4.0,
                                                2.0, 2.0, 4.0, 2.0, 1.0, 1.0, 0.0))
        self.assertEqual(state.save_updates, (3, 9, 12))
        self.assertEqual(state.paths, (str(self.source), str(self.output)))
        self.assertEqual(state.fixed, (True, "existing_sidecar", True, 0, True, 0,
                                      "log_interpolation", "constant", 20, True, True,
                                      ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2))))
        self.assertFalse(self.output.parent.exists())

    def test_all_fifteen_objects_every_missing_unknown_and_wrong_type(self):
        groups = list(objects(self.fixture()))
        self.assertEqual(len(groups), 15)
        for path, obj in groups:
            for key in obj:
                with self.subTest(path=path, missing=key):
                    data = self.fixture()
                    del locate(data, path)[key]
                    self.reject(data)
            data = self.fixture()
            locate(data, path)["unknown_private_field"] = 0
            self.reject(data)
            for wrong in (None, [], 0, True, "object"):
                data = self.fixture()
                if path:
                    locate(data, path[:-1])[path[-1]] = wrong
                else:
                    data = wrong
                # None must be encoded here rather than used as resolve's
                # test-only fixture shorthand.
                self.reject(encode(data))

    def test_entry_bytes_only_no_prior_partial_or_fabricated_tokens(self):
        parsed = p2.parse_json_bytes(encode(self.fixture()))
        partial = config.p3.derive_p3_maximum_sh(parsed)
        for value in (self.fixture(), parsed, partial, None, "{}", bytearray(b"{}")):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaises(p2.JSONInputError):
                    config.resolve_formal_config(value, raw_width=120, raw_height=80)

    def test_schema_mode_and_all_p1_fixed_values_still_reject(self):
        cases = [(('schema',), "v2"), (('run_mode',), "resume")]
        for path, obj in objects(self.fixture()):
            if path in (("model",), ("renderer",)):
                for key, value in obj.items():
                    bad = (not value if type(value) is bool else
                           value + 1 if type(value) in (int, float) else "other")
                    cases.append(((*path, key), bad))
        cases += [(('dataset', 'kind'), 'other'), (('dataset', 'eval'), False)]
        for path, bad in cases:
            data = self.fixture()
            locate(data, path[:-1])[path[-1]] = bad
            self.reject(data)

    def test_dataset_fixed_values_types_not_coerced(self):
        for field, bad_values in (("extension", (None, 0, False, [], ".png")),
                                  ("white_background", (None, 0, "false", True))):
            for bad in bad_values:
                data = self.fixture()
                data["dataset"][field] = bad
                self.reject(data)

    def test_every_integer_field_preserves_token_type_and_common_bound(self):
        paths = [
            ("dataset", "resolution", "divisor"),
            *(("model", k) for k in ("gaussian_dim", "spatial_sh_degree", "temporal_sh_degree")),
            ("renderer", "env_map_res"),
            *(("initialization", k) for k in ("num_pts", "num_extra_pts", "seed")),
            ("optimization", "total_updates"), ("optimization", "batch_size"),
            ("optimization", "learning_rate", "position_lr_max_steps"),
            *(("optimization", "population", k) for k in
              ("densify_stop_points", "densify_from_update", "densify_until_update",
               "densification_interval", "opacity_reset_interval")),
            ("optimization", "sh", "increase_interval"),
            ("reporting", "test_updates", 0), ("checkpoint", "intermediate_updates", 0),
        ]
        for path in paths:
            for raw in ('true', 'false', 'null', '"1"', '1.0', '1e0', '-1', '2147483648'):
                with self.subTest(path=path, raw=raw):
                    self.reject(self.token(path, raw))

    def test_integer_domain_endpoints_and_negative_zero(self):
        for path in (("initialization", "num_pts"), ("optimization", "total_updates"),
                     ("optimization", "batch_size"),
                     ("optimization", "learning_rate", "position_lr_max_steps"),
                     ("optimization", "sh", "increase_interval"),
                     ("optimization", "population", "densification_interval"),
                     ("optimization", "population", "opacity_reset_interval")):
            self.reject(self.token(path, "0"))
        self.reject(self.token(("initialization", "num_extra_pts"), "1"))
        self.reject(self.token(("optimization", "batch_size"), "5147"))
        for raw in ("1", "5146"):
            self.assertEqual(self.resolve(self.token(("optimization", "batch_size"), raw))
                             .optimization.batch_size, int(raw))
        for raw in ("0", "-0", "2147483647"):
            self.assertEqual(self.resolve(self.token(("initialization", "seed"), raw))
                             .initialization.seed, int(raw))
        self.assertEqual(self.resolve(self.token(("initialization", "num_pts"), "2147483647"))
                         .initialization.num_pts, 2147483647)

    def test_every_num_field_rejects_wrong_types_overflow_underflow(self):
        paths = [("dataset", "time", "raw_interval", 0),
                 ("dataset", "time", "raw_interval", 1), ("dataset", "time", "divisor"),
                 ("initialization", "time_variance_denominator"), ("renderer", "scaling_modifier"),
                 *(("optimization", "learning_rate", k) for k in
                   ("position_lr_init", "position_lr_final", "position_t_lr_init",
                    "feature_lr", "opacity_lr", "scaling_lr", "rotation_lr")),
                 *(("optimization", "loss", k) for k in
                   ("lambda_dssim", "lambda_opa_mask", "lambda_rigid", "lambda_motion")),
                 *(("optimization", "population", k) for k in
                   ("percent_dense", "thresh_opa_prune", "densify_grad_threshold"))]
        for path in paths:
            for raw in ('true', 'false', 'null', '"0.1"', '[]', '{}', '1e309', '1e-9999',
                        'NaN', 'Infinity', '-Infinity'):
                with self.subTest(path=path, raw=raw):
                    self.reject(self.token(path, raw))

    def test_four_lr_domains_no_int_cap_or_added_upper_bound(self):
        for field in ("feature_lr", "opacity_lr", "scaling_lr", "rotation_lr"):
            path = ("optimization", "learning_rate", field)
            for raw in ("0", "-0", "0.0", "0e9999", "1", "1.25", "2e-3", "2147483648",
                        "1e308", "5e-324"):
                result = self.resolve(self.token(path, raw))
                self.assertEqual(getattr(result.optimization.learning_rate, field), float(raw))
            self.reject(self.token(path, "-0.01"), "config_number_domain")

    def test_xyz_pair_temporal_lr_and_no_schedule_alias_or_default(self):
        for initial, final, accepted in ((0, 0, True), (0.1, 0.3, True),
                                        (0, 0.3, False), (0.1, 0, False), (-1, 1, False)):
            data = self.fixture()
            data["optimization"]["learning_rate"].update(
                position_lr_init=initial, position_lr_final=final)
            if accepted:
                self.assertEqual(self.resolve(data).optimization.learning_rate.position_lr_init, initial)
            else:
                self.reject(data)
        self.reject(self.token(("optimization", "learning_rate", "position_t_lr_init"), "-1"))
        state = self.resolve(self.token(("optimization", "learning_rate", "position_t_lr_init"), "0"))
        self.assertEqual(state.optimization.learning_rate.position_t_lr_init, 0)
        self.assertNotEqual(state.optimization.learning_rate.position_lr_max_steps,
                            state.optimization.total_updates)

    def test_loss_domains_separate_from_initial_unsupported_branches(self):
        for value in (0, 1):
            self.resolve(self.token(("optimization", "loss", "lambda_dssim"), str(value)))
        for raw in ("-0.1", "1.01"):
            self.reject(self.token(("optimization", "loss", "lambda_dssim"), raw))
        self.resolve(self.token(("optimization", "loss", "lambda_opa_mask"), "1e308"))
        self.reject(self.token(("optimization", "loss", "lambda_opa_mask"), "-1"))
        for key in ("lambda_rigid", "lambda_motion"):
            for raw in ("0", "-0.0", "0e0"):
                self.resolve(self.token(("optimization", "loss", key), raw))
            self.reject(self.token(("optimization", "loss", key), "-1"), "config_number_domain")
            self.reject(self.token(("optimization", "loss", key), "0.001"),
                        "config_unsupported_loss_branch")

    def test_population_union_domains_no_new_event_constraints_or_sentinel(self):
        path = ("optimization", "population", "densify_stop_points")
        for raw in ('"unlimited"', "1", "2147483647"):
            self.assertEqual(self.resolve(self.token(path, raw)).optimization.population.densify_stop_points,
                             json.loads(raw))
        for raw in ('"Unlimited"', '" unlimited"', '"1"', '"-1"', '""', '0', '-1'):
            self.reject(self.token(path, raw))
        for field, good, bad in (
            ("percent_dense", ("5e-324", "2", "1e308"), ("0", "-1")),
            ("thresh_opa_prune", ("0", "1"), ("-0.1", "1.1")),
            ("densify_grad_threshold", ("0", "1e308"), ("-1",))):
            for raw in good:
                self.resolve(self.token(("optimization", "population", field), raw))
            for raw in bad:
                self.reject(self.token(("optimization", "population", field), raw))
        for start, end in ((20, 2), (0, 0), (2147483647, 2147483647), (0, 2147483647)):
            data = self.fixture()
            data["optimization"]["population"].update(densify_from_update=start,
                                                       densify_until_update=end)
            self.assertEqual(self.resolve(data).optimization.population.densify_until_update, end)

    def test_resolution_owner_inputs_mandatory_and_no_rounding(self):
        with self.assertRaises(TypeError):
            config.resolve_formal_config(encode(self.fixture()))
        for width, height in ((121, 80), (120, 81), (0, 80), (True, 80), (120, 80.0)):
            self.reject(self.fixture(), raw_width=width, raw_height=height)
        data = self.fixture()
        data["dataset"]["resolution"]["divisor"] = 1
        self.assertEqual(self.resolve(data).dataset.resolution.width, 120)

    def test_schedule_order_bounds_final_once_and_huge_n_not_enumerated(self):
        for group, field, bads in (
            ("checkpoint", "intermediate_updates", ([12], [0], [3, 3], [9, 3], [13], None)),
            ("reporting", "test_updates", ([0], [2, 2], [12, 2], [13], None))):
            for bad in bads:
                data = self.fixture()
                data[group][field] = bad
                self.reject(data)
        for total in (1, 2147483647):
            data = self.fixture()
            data["optimization"]["total_updates"] = total
            data["checkpoint"]["intermediate_updates"] = []
            data["reporting"]["test_updates"] = []
            state = self.resolve(data)
            self.assertEqual(state.save_updates, (total,))
            self.assertEqual(state.reporting.test_updates, ())
            self.assertEqual(state.optimization.sh.increase_interval, 5)

    def test_time_health_one_transform_and_nonfixed_positive_denominator(self):
        for path, raw in ((("dataset", "time", "divisor"), "0.5"),
                          (("initialization", "time_variance_denominator"), "0"),
                          (("dataset", "time", "raw_interval"), "[6,6]"),
                          (("dataset", "time", "raw_interval"), "[6,-2]"),
                          (("dataset", "time", "raw_interval"), "[-1e308,1e308]"),
                          (("dataset", "time", "raw_interval"), "[0,5e-324]"),
                          (("dataset", "time", "divisor"), "1e308")):
            self.reject(self.token(path, raw))
        result = self.resolve(self.token(("initialization", "time_variance_denominator"), "5"))
        self.assertAlmostEqual(result.time_derivation.effective_variance, 0.4)
        self.assertAlmostEqual(result.time_derivation.effective_scale, math.sqrt(1.6) / 2)
        self.assertEqual(result.dataset.time.raw_interval, (-2, 6))
        self.assertEqual(result.time_derivation.effective_interval, (-1, 3))

    def test_legacy_inactive_and_competing_fields_rejected_not_merged(self):
        aliases = {
            (): ("config", "quiet", "start_checkpoint", "iterations"),
            ("dataset",): ("frame_ratio", "resolution_scales", "dataloader"),
            ("model",): ("eval_shfs_4d", "sh_slot_count"),
            ("optimization", "learning_rate"): ("position_lr_delay_mult", "scaling_t_lr", "rotation_r_lr"),
            ("optimization", "population"): ("point_cap", "densify_until_num_points",
                                                 "densify_grad_t_threshold", "final_prune_from_iter"),
            ("output",): ("model_path", "run_id", "output_root"),
        }
        for path, names in aliases.items():
            for name in names:
                data = self.fixture()
                locate(data, path)[name] = 0
                self.reject(data)

    def test_p2_duplicate_unicode_syntax_limits_preserved_at_full_entry(self):
        data = encode(self.fixture())
        for bad in (b'\xef\xbb\xbf' + data, data + b'{}', b'\xff',
                    data.replace(b'"seed":23', b'"seed":23,"se\\u0065d":23'),
                    b' ' * (p2.MAX_INPUT_BYTES + 1),
                    self.token(("dataset", "extension"), '"' + 'x' * 4097 + '"'),
                    self.token(("reporting", "test_updates"), '[' + ','.join(['1'] * 4097) + ']'),
                    self.token(("reporting", "test_updates"), '[' * 17 + '0' + ']' * 17)):
            self.reject(bad)

    def test_all_reused_components_called_no_partial_success_handoff(self):
        calls = []
        components = ((config.p1, "check_p1_structure_and_fixed_inputs"),
                      (config.p3, "derive_p3_maximum_sh"), (config.p4, "derive_p4_resolution"),
                      (config.p5, "derive_p5_schedules"),
                      (config.time_config, "derive_time_interval_and_variance"),
                      (config.p6, "preflight_p6_paths"))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for module, name in components:
                original = getattr(module, name)
                def record(*args, _name=name, _original=original, **kwargs):
                    calls.append(_name)
                    return _original(*args, **kwargs)
                stack.enter_context(patch.object(module, name, side_effect=record))
            self.resolve()
        for _, name in components:
            self.assertIn(name, calls)
        self.assertGreater(calls.index("preflight_p6_paths"),
                           calls.index("derive_time_interval_and_variance"))
        for bad in (self.token(("optimization", "batch_size"), "0"),
                    self.token(("optimization", "learning_rate", "rotation_lr"), "-1"),
                    self.token(("optimization", "loss", "lambda_motion"), "1")):
            with patch.object(config.p6, "preflight_p6_paths") as preflight:
                self.reject(bad)
                preflight.assert_not_called()
        with patch.object(config.p6, "preflight_p6_paths",
                          side_effect=config.p6.P6InputError("p6_filesystem")):
            self.reject(self.fixture(), "p6_filesystem")

    def test_p6_current_filesystem_no_claim_and_canonical_paths(self):
        alias = self.base / "data-alias"
        alias.symlink_to(self.source, target_is_directory=True)
        data = self.fixture()
        data["dataset"]["source_path"] = str(alias)
        before = snapshot(self.base)
        state = self.resolve(data)
        self.assertEqual(state.dataset.source_path, str(alias))
        self.assertEqual(state.paths.source_path, str(self.source))
        self.assertEqual(snapshot(self.base), before)
        for destination in (self.source, self.source / "child", self.legacy / "child"):
            data["output"]["directory"] = str(destination)
            self.reject(data)
        dangling = self.base / "dangling"
        dangling.symlink_to(self.base / "missing")
        data["output"]["directory"] = str(dangling)
        self.reject(data)
        self.output.parent.mkdir()
        self.output.mkdir()  # Fixture mutation, not the resolver.
        self.reject(self.fixture(), "p6_output_occupied")
        data = self.fixture()
        data["dataset"]["source_path"] = str(self.base / "absent")
        self.reject(data)

    def test_deep_immutability_and_detachment_including_partial_dataclasses(self):
        data = self.fixture()
        captured = []
        original = config.time_config.derive_time_interval_and_variance
        def capture(parsed):
            result = original(parsed)
            captured.extend((parsed, result))
            return result
        with patch.object(config.time_config, "derive_time_interval_and_variance", side_effect=capture):
            state = self.resolve(data)
        before = copy.deepcopy(state)
        data["dataset"]["time"]["raw_interval"][0] = 100
        captured[0].root.clear()
        captured[1].__dict__["raw_interval"] = ["mutated"]
        self.assertEqual(state, before)
        def immutable(value):
            self.assertFalse(hasattr(value, "__dict__"))
            self.assertNotIsInstance(value, (dict, list, set, p2.NumberToken))
            if isinstance(value, tuple):
                if value:
                    with self.assertRaises(TypeError):
                        value[0] = None
                if hasattr(value, "_fields"):
                    for field in value._fields:
                        with self.assertRaises(AttributeError):
                            setattr(value, field, None)
                for item in value:
                    immutable(item)
            else:
                self.assertIn(type(value), (str, int, float, bool, type(None)))
        immutable(state)

    def test_errors_bounded_and_do_not_echo_private_input(self):
        data = self.fixture()
        data["optimization"]["learning_rate"]["feature_lr"] = "SECRET_MARKER_NOT_A_CREDENTIAL"
        with self.assertRaises(p2.JSONInputError) as caught:
            self.resolve(data)
        self.assertNotIn("SECRET_MARKER", str(caught.exception))
        self.assertLess(len(str(caught.exception)), 64)

    def test_isolated_stdlib_import_no_content_read_rng_or_writer(self):
        script = r'''
import sys, os, random
class OnlyAllowed:
    def find_spec(self, fullname, path=None, target=None):
        # Python 3.10 copy probes Jython optionally. Deny the probe with its
        # normal absence signal; do not allow loading an arbitrary org module.
        if fullname == 'org':
            raise ModuleNotFoundError('optional Jython unavailable')
        root = fullname.split('.')[0]
        if root not in sys.stdlib_module_names and root not in {
            'formal_config', 'formal_config_json', 'formal_config_p1',
            'formal_config_p3', 'formal_config_p4', 'formal_config_p5',
            'formal_config_p6', 'formal_config_time'}:
            raise AssertionError('non-stdlib import')
sys.meta_path.insert(0, OnlyAllowed())
def forbidden(*a, **k):
    raise AssertionError('RNG/content access')
random.seed = random.random = random.Random = random.SystemRandom = forbidden
os.urandom = forbidden
def audit_before(event, args):
    if event in {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.symlink',
                 'subprocess.Popen', 'os.system'}:
        raise AssertionError('mutation/process')
    if event == 'open':
        flags = args[2]
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('writer')
sys.addaudithook(audit_before)
import formal_config
formal_config.p6._LEGACY_OUTPUT = sys.argv[1]  # Isolated test fixture only.
data = sys.stdin.buffer.read()
def audit_after(event, args):
    if event in {'open', 'os.listdir', 'os.scandir'}:
        raise AssertionError('content read/scan')
sys.addaudithook(audit_after)
formal_config.resolve_formal_config(data, raw_width=120, raw_height=80)
print('isolated-validation-ok')
'''
        before = snapshot(self.base)
        result = subprocess.run([sys.executable, "-B", "-c", script, str(self.legacy)], cwd=ROOT,
                                input=encode(self.fixture()), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, b"isolated-validation-ok\n")
        self.assertEqual(snapshot(self.base), before)


if __name__ == "__main__":
    unittest.main()
