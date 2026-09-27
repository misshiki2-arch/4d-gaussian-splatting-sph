"""P6 read-only path preflight, NOT full validation or an output reservation.

Only current P1 premises, two explicit paths, and the named competing inputs
immediately under dataset/output are checked. Unchecked nested fields are not
approved extensions. No CLI, dataset-content reader, runtime or writer is here.
Filesystem races remain possible: the later exclusive pre-writer claim is
mandatory even after this partial result succeeds.
"""

from collections import deque
from dataclasses import dataclass
import os
import stat

import formal_config_json as p2
import formal_config_p1 as p1


# Plan: Immutable legacy dataset and checkpoint identity. Not a JSON/CLI input
# or an optional protection list. Tests replace this constant only with mocks
# scoped to an isolated fixture; there is no production bypass parameter.
_LEGACY_OUTPUT = "/home/demo/work/outputs/sph_scene_4dgs"
_COMPETING_INPUTS = frozenset(("run_id", "model_path", "output_root"))


class P6InputError(p2.JSONInputError):
    """Fixed bounded codes only; no input, path, or OS exception text."""


@dataclass(frozen=True, slots=True)
class P6PathsPartial:
    """Detached canonical strings, not content proof or execution permission."""

    source_path: str
    output_directory: str


def _path_string(value):
    if type(value) is not str:
        raise P6InputError("p6_path_type")
    if len(value) > p2.MAX_STRING_SCALARS:
        raise P6InputError("p6_path_length")
    if any(0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise P6InputError("p6_path_unicode")
    if any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value):
        raise P6InputError("p6_path_control")
    if (not value.startswith("/") or value.startswith("//")
            or any(c in value for c in "~$*?[]\\")):
        raise P6InputError("p6_path_syntax")
    return value


def _required_path(obj, key):
    if key not in obj:
        raise P6InputError("p6_missing_path")
    return _path_string(obj[key])


def _canonical_directory(path, *, allow_missing, output=False):
    """Walk metadata only, in filesystem order, without lexical '..' collapse.

    Unlike non-strict realpath/exists, only ENOENT can mean absence. Missing
    tails may name future directories, but cannot be traversed by '..': that
    would invent a filesystem resolution not currently provable. A terminal
    output symlink is occupied even when dangling. Symlink expansion has the
    Linux 40-link traversal bound; loops/errors reject without raw diagnostics.
    No directory enumeration, file-content reads, or filesystem mutations.
    """
    parts = []
    pending = deque(path.split("/"))
    links = 0
    try:
        if not stat.S_ISDIR(os.lstat("/").st_mode):
            raise P6InputError("p6_path_not_directory")
        while pending:
            part = pending.popleft()
            if part in ("", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            candidate = "/" + "/".join((*parts, part))
            try:
                metadata = os.lstat(candidate)
            except FileNotFoundError:
                if not allow_missing or ".." in pending:
                    raise P6InputError("p6_path_unresolved") from None
                tail = [p for p in pending if p not in ("", ".")]
                return "/" + "/".join((*parts, part, *tail)), False
            terminal = not any(p not in ("", ".") for p in pending)
            if output and terminal:
                raise P6InputError("p6_output_occupied")
            if stat.S_ISLNK(metadata.st_mode):
                links += 1
                if links > 40:
                    raise P6InputError("p6_path_unresolved")
                target = os.readlink(candidate)
                if target.startswith("/"):
                    parts = []
                pending.extendleft(reversed(target.split("/")))
                continue
            if not stat.S_ISDIR(metadata.st_mode):
                raise P6InputError("p6_path_not_directory")
            # lstat of the name alone does not prove directory traversal.
            # In particular, do not silently step across an inaccessible
            # directory followed by '.' or '..' via string operations.
            if not stat.S_ISDIR(os.stat(candidate + "/.").st_mode):
                raise P6InputError("p6_path_not_directory")
            parts.append(part)
        return "/" + "/".join(parts), True
    except P6InputError:
        raise
    except (OSError, ValueError):
        # OS errors never become absence or leak exception filenames/messages.
        raise P6InputError("p6_filesystem") from None


def _overlap(left, right):
    """Canonical POSIX component containment, never textual-prefix matching."""
    a, b = left.split("/"), right.split("/")
    if left == "/" or right == "/":
        return True
    return a[:len(b)] == b or b[:len(a)] == a


def preflight_p6_paths(parsed: p2.ParsedJSON) -> P6PathsPartial:
    """Recheck current P1/P6 input and return only a read-only partial result.

    No prior success token, default, path expansion, or alternate output ID.
    Caller-fabricated parse data and concurrent mutation receive no provenance
    guarantee. The fixed historical root is always checked, including when
    its directory is currently absent. This is not the formal runtime handoff.
    """
    p1.check_p1_structure_and_fixed_inputs(parsed)
    dataset, output = parsed.root["dataset"], parsed.root["output"]
    if (_COMPETING_INPUTS.intersection(dataset)
            or _COMPETING_INPUTS.intersection(output)):
        raise P6InputError("p6_competing_input")
    source = _required_path(dataset, "source_path")
    destination = _required_path(output, "directory")
    if os.name != "posix":
        raise P6InputError("p6_platform")
    source, _ = _canonical_directory(source, allow_missing=False)
    legacy, _ = _canonical_directory(_LEGACY_OUTPUT, allow_missing=True)
    destination, occupied = _canonical_directory(
        destination, allow_missing=True, output=True)
    if occupied:
        raise P6InputError("p6_output_occupied")
    if _overlap(source, destination) or _overlap(legacy, destination):
        raise P6InputError("p6_path_overlap")
    return P6PathsPartial(source, destination)
