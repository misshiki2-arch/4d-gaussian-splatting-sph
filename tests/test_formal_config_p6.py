"""Independent P1/P6 partial fixtures, NOT complete/runnable v1 configs.

Fixture creation and cleanup occur outside the read-only component calls.
No actual dataset or historical output is created, enumerated, or read.
"""

import ast
import copy
from dataclasses import FrozenInstanceError, asdict
import errno
import json
import os
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
import formal_config_p6 as p6


def partial_fixture(source, output):
    return {
        "schema": "corrected_4dgs_training_config_v1",
        "run_mode": "from_scratch",
        "dataset": {"kind": "nerf_transforms", "eval": True,
                    "source_path": str(source)},
        "model": {"gaussian_dim": 4, "spatial_sh_degree": 3,
                  "temporal_sh_degree": 2, "rot_4d": True,
                  "force_sh_3d": False, "sh_evaluation": "conditional_mean"},
        "renderer": {"compute_cov3D_python": False, "convert_SHs_python": False,
                     "scaling_modifier": 1, "env_map_res": 0,
                     "override_color": None, "temporal_prefilter": "disabled"},
        "initialization": {}, "optimization": {}, "reporting": {},
        "checkpoint": {}, "output": {"directory": str(output)},
    }


def encode(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()


def identity_tree(value):
    if type(value) is dict:
        return id(value), [(k, identity_tree(v)) for k, v in value.items()]
    if type(value) is list:
        return id(value), [identity_tree(v) for v in value]
    return id(value), type(value), value


def snapshot(root):
    # Test-only enumeration, outside calls. Ignore access times affected by
    # read-only metadata/fixture observation; include inode/mode/mtime/content.
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        s = path.lstat()
        content = (os.readlink(path) if path.is_symlink() else
                   path.read_bytes() if path.is_file() else None)
        result[str(path.relative_to(root))] = (
            s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, content)
    return result


class P6Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "data"
        self.source.mkdir()
        self.legacy = self.base / "legacy"
        self.legacy.mkdir()
        self.output = self.base / "new-output"
        mock = patch.object(p6, "_LEGACY_OUTPUT", str(self.legacy))
        mock.start()
        self.addCleanup(mock.stop)

    def fixture(self):
        return partial_fixture(self.source, self.output)

    def check_parsed(self, parsed, error=None):
        before, identities = copy.deepcopy(parsed), identity_tree(parsed.root)
        files_before = snapshot(self.base)
        try:
            if error is None:
                return p6.preflight_p6_paths(parsed)
            with self.assertRaises(p2.JSONInputError) as caught:
                p6.preflight_p6_paths(parsed)
            self.assertEqual(caught.exception.code, error)
            self.assertEqual(str(caught.exception), error)
            self.assertLess(len(str(caught.exception)), 64)
        finally:
            self.assertEqual(parsed, before)
            self.assertEqual(identity_tree(parsed.root), identities)
            self.assertEqual(snapshot(self.base), files_before)

    def check_value(self, value, error=None):
        return self.check_parsed(p2.parse_json_bytes(encode(value)), error)

    def test_independent_directories_canonical_partial_result(self):
        result = self.check_value(self.fixture())
        self.assertEqual(asdict(result), {"source_path": str(self.source),
                                         "output_directory": str(self.output)})
        self.assertFalse(self.output.exists())
        data = self.fixture()
        data["output"]["directory"] += "/future/child"
        self.assertEqual(self.check_value(data).output_directory,
                         str(self.output / "future/child"))

    def test_existing_dataset_and_output_parent_symlinks(self):
        link = self.base / "data-link"
        link.symlink_to("data", target_is_directory=True)
        parent = self.base / "parent"
        parent.mkdir()
        alias = self.base / "parent-link"
        alias.symlink_to(parent, target_is_directory=True)
        result = self.check_value(partial_fixture(link, alias / "new"))
        self.assertEqual((result.source_path, result.output_directory),
                         (str(self.source), str(parent / "new")))

    def test_missing_paths_are_not_defaulted(self):
        for group, field in (("dataset", "source_path"), ("output", "directory")):
            data = self.fixture()
            del data[group][field]
            with patch.dict(os.environ, {"OAR_JOB_ID": "not-authority"}):
                self.check_value(data, "p6_missing_path")

    def test_path_types_no_coercion(self):
        for group, field in (("dataset", "source_path"), ("output", "directory")):
            for bad in (None, False, True, 1, 1.0, [], {}):
                data = self.fixture()
                data[group][field] = bad
                self.check_value(data, "p6_path_type")
            parsed = p2.parse_json_bytes(encode(self.fixture()))
            parsed.root[group][field] = self.source
            self.check_parsed(parsed, "p6_path_type")

    def test_empty_relative_expansion_windows_paths_rejected(self):
        bad_paths = ("", ".", "..", "data", "~/data", "/tmp/~user/data",
                     "/tmp/$DATA", "/tmp/${DATA}", "/tmp/*", "/tmp/?",
                     "/tmp/[a-z]", "C:/data", "C:\\data", "C:data",
                     "\\\\wsl.localhost\\Ubuntu-24.04\\home\\demo",
                     "//server/share", "/tmp\\data")
        for group, field in (("dataset", "source_path"), ("output", "directory")):
            for bad in bad_paths:
                with self.subTest(group=group, value=bad):
                    data = self.fixture()
                    data[group][field] = bad
                    self.check_value(data, "p6_path_syntax")

    def test_path_controls_rejected_in_both_inputs(self):
        for group, field in (("dataset", "source_path"), ("output", "directory")):
            for code in (*range(32), *range(127, 160)):
                data = self.fixture()
                data[group][field] = str(self.output) + chr(code)
                self.check_value(data, "p6_path_control")

    def test_current_strings_preserve_p2_unicode_and_length_limits(self):
        for group, field in (("dataset", "source_path"), ("output", "directory")):
            for bad, code in (("/" + "x" * 4096, "p6_path_length"),
                              ("/bad\ud800", "p6_path_unicode")):
                parsed = p2.parse_json_bytes(encode(self.fixture()))
                parsed.root[group][field] = bad
                self.check_parsed(parsed, code)
        # At the P2 boundary strings pass syntax checking, not OS/path limits.
        self.assertEqual(p6._path_string("/" + "日" * 4095), "/" + "日" * 4095)

    def test_unicode_case_and_spaces_not_normalized(self):
        for name in ("日本語😀", "é", "é", "A", "a", " spaced "):
            source = self.base / name
            source.mkdir()
            output = self.base / (name + "-new")
            result = self.check_value(partial_fixture(source, output))
            self.assertEqual((result.source_path, result.output_directory),
                             (str(source), str(output)))

    def test_nonexistent_and_nondirectory_dataset(self):
        data = self.fixture()
        data["dataset"]["source_path"] = str(self.base / "absent")
        self.check_value(data, "p6_path_unresolved")
        file = self.base / "file"
        file.write_text("fixture")
        for suffix in ("", "/.", "/child", "/../data"):
            data["dataset"]["source_path"] = str(file) + suffix
            self.check_value(data, "p6_path_not_directory")

    def test_existing_output_file_directory_symlink_and_dangling(self):
        file = self.base / "file"
        file.write_text("fixture")
        for name, target in (("link-dir", self.source), ("link-file", file),
                             ("dangling", self.base / "absent")):
            (self.base / name).symlink_to(target)
        for path in (file, self.source, self.legacy, self.base / "link-dir",
                     self.base / "link-file", self.base / "dangling"):
            for suffix in ("", "/", "/."):
                data = self.fixture()
                data["output"]["directory"] = str(path) + suffix
                self.check_value(data, "p6_output_occupied")

    def test_same_and_both_directions_of_dataset_containment(self):
        for output, code in ((self.source, "p6_output_occupied"),
                             (self.source / "child", "p6_path_overlap"),
                             (self.base, "p6_output_occupied"),
                             (Path("/"), "p6_output_occupied")):
            self.check_value(partial_fixture(self.source, output), code)
        # Existing dataset implies any canonical ancestor already exists.
        self.assertTrue(p6._overlap(str(self.source), str(self.base)))
        self.assertTrue(p6._overlap(str(self.base), str(self.source)))

    def test_legacy_and_canonical_alias_protected(self):
        link = self.base / "legacy-link"
        link.symlink_to(self.legacy, target_is_directory=True)
        for output in (self.legacy / "child", link / "child"):
            self.check_value(partial_fixture(self.source, output), "p6_path_overlap")
        with patch.object(p6, "_LEGACY_OUTPUT", str(link)):
            self.check_value(partial_fixture(self.source, self.legacy / "new"),
                             "p6_path_overlap")

    def test_absent_legacy_root_still_protects_it_and_ancestors(self):
        protected = self.base / "future-legacy" / "deep"
        with patch.object(p6, "_LEGACY_OUTPUT", str(protected)):
            for output in (protected, protected / "child", protected.parent):
                self.check_value(partial_fixture(self.source, output), "p6_path_overlap")

    def test_similar_prefix_is_not_containment(self):
        for output in (self.base / "data-corrected", self.base / "legacy-new"):
            self.assertEqual(self.check_value(partial_fixture(self.source, output))
                             .output_directory, str(output))

    def test_dataset_symlink_alias_containment(self):
        link = self.base / "alias"
        link.symlink_to(self.source, target_is_directory=True)
        self.check_value(partial_fixture(link, self.source / "child"), "p6_path_overlap")
        self.check_value(partial_fixture(self.source, link / "child"), "p6_path_overlap")
        self.check_value(partial_fixture(link, self.source), "p6_output_occupied")

    def test_symlink_then_dotdot_uses_filesystem_order(self):
        (self.source / "inner").mkdir()
        link = self.base / "link"
        link.symlink_to(self.source / "inner", target_is_directory=True)
        # Lexical collapse would incorrectly make this base/new, not data/new.
        raw = str(link) + "/../new"
        self.check_value(partial_fixture(self.source, raw), "p6_path_overlap")
        result = self.check_value(partial_fixture(str(link) + "/..", self.output))
        self.assertEqual(result.source_path, str(self.source))
        (self.legacy / "inner").mkdir()
        legacy_link = self.base / "legacy-inner"
        legacy_link.symlink_to(self.legacy / "inner", target_is_directory=True)
        self.check_value(partial_fixture(self.source, str(legacy_link) + "/../new"),
                         "p6_path_overlap")

    def test_valid_dotdot_and_redundant_separators(self):
        inner = self.base / "inner"
        inner.mkdir()
        result = self.check_value(partial_fixture(
            str(inner) + "/../data/.", str(inner) + "/..//new/./child/"))
        self.assertEqual((result.source_path, result.output_directory),
                         (str(self.source), str(self.base / "new/child")))

    def test_missing_before_dotdot_is_not_lexically_repaired(self):
        for raw in (str(self.base / "absent") + "/../new",
                    str(self.base / "absent/child") + "/../../new"):
            self.check_value(partial_fixture(self.source, raw), "p6_path_unresolved")

    def test_cycles_and_bad_ancestor_fail_closed(self):
        a, b = self.base / "a", self.base / "b"
        a.symlink_to("b")
        b.symlink_to("a")
        self.check_value(partial_fixture(a, self.output), "p6_path_unresolved")
        self.check_value(partial_fixture(self.source, a / "new"), "p6_path_unresolved")
        file = self.base / "file"
        file.write_text("fixture")
        self.check_value(partial_fixture(self.source, file / "new"),
                         "p6_path_not_directory")

    def test_unverifiable_traversal_before_dotdot_is_rejected(self):
        inner = self.base / "inner"
        inner.mkdir()
        parsed = p2.parse_json_bytes(encode(partial_fixture(
            str(inner) + "/../data", self.output)))
        real_stat = os.stat
        def fail(path, *args, **kwargs):
            if path == str(inner) + "/.":
                raise PermissionError(errno.EACCES, "PRIVATE-PATH")
            return real_stat(path, *args, **kwargs)
        with patch.object(p6.os, "stat", side_effect=fail):
            self.check_parsed(parsed, "p6_filesystem")

    def test_competing_inputs_at_root_dataset_output_even_inactive(self):
        for group in (None, "dataset", "output"):
            for field in ("run_id", "model_path", "output_root"):
                for value in (None, False, 0, "", [], {}, "other"):
                    data = self.fixture()
                    (data if group is None else data[group])[field] = value
                    self.check_value(data, "p1_top_level_keys" if group is None
                                     else "p6_competing_input")

    def test_competing_input_cannot_replace_missing_output(self):
        data = self.fixture()
        del data["output"]["directory"]
        data["output"]["model_path"] = str(self.output)
        self.check_value(data, "p6_competing_input")

    def test_p1_structure_and_all_fixed_inputs_rechecked_before_filesystem(self):
        cases = []
        for key in self.fixture():
            data = self.fixture()
            del data[key]
            cases.append((data, "p1_top_level_keys"))
        for group in ("dataset", "model", "renderer"):
            for field, good in self.fixture()[group].items():
                if field == "source_path":
                    continue
                data = self.fixture()
                del data[group][field]
                cases.append((data, "p1_missing_fixed"))
                data = self.fixture()
                data[group][field] = (not good if type(good) is bool else
                                     good + 1 if type(good) is int else "wrong")
                cases.append((data, "p1_null_type" if good is None else "p1_fixed_value"))
        for data, error in cases:
            parsed = p2.parse_json_bytes(encode(data))
            with patch.object(p6, "_canonical_directory",
                              side_effect=AssertionError("filesystem reached")):
                self.check_parsed(parsed, error)

    def test_past_p1_or_p6_success_not_a_token(self):
        for kind, code in (("p1", "p1_fixed_value"), ("missing", "p6_missing_path"),
                            ("relative", "p6_path_syntax"),
                            ("competing", "p6_competing_input")):
            parsed = p2.parse_json_bytes(encode(self.fixture()))
            p1.check_p1_structure_and_fixed_inputs(parsed)
            self.check_parsed(parsed)
            if kind == "p1":
                parsed.root["dataset"]["eval"] = False
            elif kind == "missing":
                del parsed.root["output"]["directory"]
            elif kind == "relative":
                parsed.root["output"]["directory"] = "relative"
            else:
                parsed.root["output"]["run_id"] = None
            self.check_parsed(parsed, code)

    def test_reuses_p1_and_wrong_entry_cannot_bypass(self):
        parsed = p2.parse_json_bytes(encode(self.fixture()))
        with patch.object(p1, "check_p1_structure_and_fixed_inputs",
                          wraps=p1.check_p1_structure_and_fixed_inputs) as check:
            self.check_parsed(parsed)
            check.assert_called_once_with(parsed)
        for wrong in (None, {}, self.fixture(), b"{}", p2.ParsedJSON([])):
            with self.assertRaises(p1.P1InputError):
                p6.preflight_p6_paths(wrong)

    def test_result_is_immutable_detached_and_not_permission(self):
        parsed = p2.parse_json_bytes(encode(self.fixture()))
        result = self.check_parsed(parsed)
        for field in ("source_path", "output_directory"):
            with self.assertRaises(FrozenInstanceError):
                setattr(result, field, "changed")
        self.assertFalse(hasattr(result, "__dict__"))
        for field in ("verified", "execution_allowed", "root", "reserved", "runtime"):
            self.assertFalse(hasattr(result, field))
        later = self.check_parsed(parsed)
        self.assertIsNot(later, result)
        parsed.root.clear()
        self.assertEqual(asdict(result), {"source_path": str(self.source),
                                         "output_directory": str(self.output)})

    def test_unchecked_other_owner_and_nested_fields_not_adopted(self):
        data = self.fixture()
        for group in ("dataset", "model", "renderer", "initialization",
                      "optimization", "reporting", "checkpoint", "output"):
            data[group]["UNREVIEWED"] = {"model_path": [None, False, 1.5, "日本語"]}
        data["model"]["run_id"] = None
        data["dataset"]["resolution"] = "not-P4-validated"
        data["optimization"]["total_updates"] = "not-P5-validated"
        data["initialization"]["time_variance_denominator"] = None
        self.check_value(data)

    def test_os_errors_are_bounded_and_not_absence(self):
        parsed = p2.parse_json_bytes(encode(self.fixture()))
        real_lstat = os.lstat
        for target in (str(self.source), str(self.legacy), str(self.output)):
            for code in (errno.EACCES, errno.EIO, errno.ENOTDIR, errno.ELOOP):
                def fail(path, *, _target=target, _code=code):
                    if path == _target:
                        raise OSError(_code, "PRIVATE-BODY", "/private/SECRET")
                    return real_lstat(path)
                with patch.object(p6.os, "lstat", side_effect=fail):
                    with self.assertRaises(p6.P6InputError) as caught:
                        p6.preflight_p6_paths(parsed)
                    self.assertEqual(str(caught.exception), "p6_filesystem")
                    self.assertIsNone(caught.exception.__cause__)
                    self.assertTrue(caught.exception.__suppress_context__)
        link = self.base / "link"
        link.symlink_to(self.source)
        parsed.root["dataset"]["source_path"] = str(link)
        with patch.object(p6.os, "readlink", side_effect=OSError("PRIVATE-BODY")):
            with self.assertRaises(p6.P6InputError) as caught:
                p6.preflight_p6_paths(parsed)
            self.assertEqual(str(caught.exception), "p6_filesystem")

    def test_p2_rejects_duplicate_path_before_component(self):
        raw = encode(self.fixture()).replace(b'"source_path":', b'"source_path":null,"source_\\u0070ath":')
        with patch.object(p6, "preflight_p6_paths") as check:
            with self.assertRaises(p2.JSONInputError) as caught:
                check(p2.parse_json_bytes(raw))
            self.assertEqual(caught.exception.code, "duplicate_key")
            check.assert_not_called()

    def test_real_legacy_constant_is_mandatory_not_public_override(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(p6.preflight_p6_paths).parameters),
                         ("parsed",))
        tree = ast.parse((ROOT / "formal_config_p6.py").read_text())
        values = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_LEGACY_OUTPUT"
                          for t in node.targets)]
        self.assertEqual(values, ["/home/demo/work/outputs/sph_scene_4dgs"])


class IsolationTests(unittest.TestCase):
    def test_imports_only_stdlib_p1_p2_p6(self):
        for path in (ROOT / "formal_config_p6.py", Path(__file__)):
            names = set()
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names.update(a.name.split(".")[0] for a in node.names)
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    names.add(node.module.split(".")[0])
            self.assertLessEqual(names, sys.stdlib_module_names |
                                 {"formal_config_json", "formal_config_p1", "formal_config_p6"})

    def test_isolated_no_writer_content_walk_heavy_import_or_jit(self):
        script = r'''
import sys, os, json
sys.path.insert(0, sys.argv[1])
class OnlyStdlib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'org':
            raise ModuleNotFoundError('Jython unavailable', name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names | {
                'formal_config_json', 'formal_config_p1', 'formal_config_p6'}:
            raise AssertionError('non-stdlib import attempted')
sys.meta_path.insert(0, OnlyStdlib())
def audit(event, args):
    if event == 'open':
        mode, flags = args[1], args[2]
        if (mode and any(c in mode for c in 'wax+')) or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('writer attempted')
    if event.startswith(('subprocess.', 'socket.')) or event in (
            'os.mkdir', 'os.remove', 'os.rename', 'os.system', 'os.rmdir',
            'os.symlink', 'os.link', 'os.chmod', 'os.chown', 'os.utime',
            'os.truncate', 'ctypes.dlopen'):
        raise AssertionError('side effect attempted')
sys.addaudithook(audit)
import formal_config_json as p2
import formal_config_p6 as p6
# Explicit test-only fixture substitution, never the real historical directory.
p6._LEGACY_OUTPUT = sys.argv[3]
def no_content(event, args):
    if event in ('open', 'os.listdir', 'os.scandir'):
        raise AssertionError('content read or directory enumeration attempted')
sys.addaudithook(no_content)
def check(data):
    return p6.preflight_p6_paths(p2.parse_json_bytes(json.dumps(data).encode()))
data = json.loads(sys.argv[2])
assert check(data).output_directory == data['output']['directory']
for group, field, bad in (('dataset', 'eval', False),
                          ('output', 'directory', data['dataset']['source_path']),
                          ('output', 'directory', data['dataset']['source_path'] + '/new'),
                          ('output', 'run_id', None)):
    value = json.loads(sys.argv[2])
    value[group][field] = bad
    try:
        check(value)
    except p2.JSONInputError:
        pass
    else:
        raise AssertionError('invalid input accepted')
assert not any(x.split('.')[0] in {'torch','scene','gaussian_renderer','train'}
               for x in sys.modules)
print(json.dumps({'stdlib_only':True,'no_writer':True,'no_heavy_import':True,
                  'no_content_read_or_walk':True}))
'''
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            source, legacy, output = base / "data", base / "legacy", base / "output"
            source.mkdir()
            legacy.mkdir()
            (source / "do-not-read").write_bytes(b"fixture")
            before = snapshot(base)
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", script, str(ROOT),
                 encode(partial_fixture(source, output)).decode(), str(legacy)],
                cwd=base, text=True, capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), {
                "stdlib_only": True, "no_writer": True, "no_heavy_import": True,
                "no_content_read_or_walk": True})
            self.assertEqual(snapshot(base), before)


if __name__ == "__main__":
    unittest.main()
