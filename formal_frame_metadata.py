"""Read the two approved transforms and reuse the partial frame-time check.

The dataset owner must bind byte identities AND the separately preserved frame
reference to approved evidence. These local containers do not certify arbitrary
caller references. This is not a manifest/CLI format, full camera validation,
loader equivalence, verified runtime state, or an output claim. No reports,
dataset-specific hashes/counts, image/PLY readers, writers or heavy imports.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import re
import stat

import formal_config_json as p2
import formal_config_p6 as p6
import formal_frame_time as frame_time


class MetadataInputError(p2.JSONInputError):
    """Bounded code only; never expose file content or OS exception paths."""


@dataclass(frozen=True)
class FileIdentity:
    size: int
    sha256: str


@dataclass(frozen=True)
class MetadataReference:
    """Internal owner input, NOT a verified token; nested frames stay untrusted."""

    train: FileIdentity
    test: FileIdentity
    frames: dict


def _check_reference(reference):
    if type(reference) is not MetadataReference:
        raise MetadataInputError("metadata_reference_type")
    for identity in (reference.train, reference.test):
        if type(identity) is not FileIdentity:
            raise MetadataInputError("metadata_identity_type")
        if type(identity.size) is not int or identity.size <= 0:
            raise MetadataInputError("metadata_reference_size")
        if (type(identity.sha256) is not str or
                re.fullmatch(r"[0-9a-f]{64}", identity.sha256) is None):
            raise MetadataInputError("metadata_reference_hash")
    if (type(reference.frames) is not dict or
            reference.frames.keys() != {"train", "test"} or
            any(type(reference.frames[s]) is not list for s in ("train", "test"))):
        raise MetadataInputError("metadata_reference_frames")


def _file_stamp(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size,
            s.st_mtime_ns, s.st_ctime_ns)


def _directory_stamp(s):
    return s.st_dev, s.st_ino, s.st_mode


def _read_file(directory_fd, name, identity):
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise MetadataInputError("metadata_not_regular")
    if before.st_size != identity.size:
        raise MetadataInputError("metadata_size")
    # NOFOLLOW rejects a substituted symlink; NONBLOCK prevents a raced FIFO
    # from blocking before fstat rejects it. Only the two literal names enter.
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                 dir_fd=directory_fd)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or _file_stamp(opened) != _file_stamp(before):
            raise MetadataInputError("metadata_identity_changed")
        # One acquisition, bounded by approved size plus one detection byte.
        # An incomplete read fails closed; never retry or reopen for hashing.
        data = os.read(fd, identity.size + 1)
        after_fd = os.fstat(fd)
        after_path = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (_file_stamp(after_fd) != _file_stamp(before) or
                _file_stamp(after_path) != _file_stamp(before)):
            raise MetadataInputError("metadata_identity_changed")
        if len(data) != identity.size:
            raise MetadataInputError("metadata_size")
        if hashlib.sha256(data).hexdigest() != identity.sha256:
            raise MetadataInputError("metadata_hash")
        return data
    finally:
        os.close(fd)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise MetadataInputError("metadata_duplicate_key")
        result[key] = value
    return result


def _nonfinite(unused):
    raise MetadataInputError("metadata_nonfinite")


def _decode(data):
    if data.startswith(b"\xef\xbb\xbf"):
        raise MetadataInputError("metadata_bom")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise MetadataInputError("metadata_utf8") from None
    try:
        root = json.loads(text, object_pairs_hook=_pairs,
                          parse_int=p2.NumberToken, parse_float=p2.NumberToken,
                          parse_constant=_nonfinite)
    except (json.JSONDecodeError, RecursionError):
        raise MetadataInputError("metadata_json") from None
    if type(root) is not dict:
        raise MetadataInputError("metadata_root")
    if "frames" not in root or type(root["frames"]) is not list:
        raise MetadataInputError("metadata_frames")
    return root["frames"]


def read_frame_metadata(parsed: p2.ParsedJSON,
                        reference: MetadataReference) -> frame_time.FrameTimes:
    return read_frame_snapshot(parsed, reference)[2]


def read_frame_snapshot(parsed: p2.ParsedJSON, reference: MetadataReference):
    """P6 -> both bounded byte checks -> decode -> current #48/#47 validation.

    Source-directory symlink resolution belongs to P6. Hold the resolved
    directory descriptor across both file reads; check its path identity again
    before use. File identities are checked before/open/after on descriptors
    and names. This detects observed replacement/change, not perpetual file
    stability, concurrent-caller safety, or a general atomic filesystem snapshot.
    No camera metadata is returned as validated state. Output is not created.
    """
    paths = p6.preflight_p6_paths(parsed)  # Recheck current config EVERY call.
    _check_reference(reference)
    try:
        before = os.lstat(paths.source_path)
        if not stat.S_ISDIR(before.st_mode):
            raise MetadataInputError("metadata_directory")
        directory_fd = os.open(paths.source_path, os.O_RDONLY | os.O_DIRECTORY |
                               os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(directory_fd)
            if _directory_stamp(opened) != _directory_stamp(before):
                raise MetadataInputError("metadata_directory_changed")
            train = _read_file(directory_fd, "transforms_train.json", reference.train)
            test = _read_file(directory_fd, "transforms_test.json", reference.test)
            if (_directory_stamp(os.fstat(directory_fd)) != _directory_stamp(before) or
                    _directory_stamp(os.lstat(paths.source_path)) != _directory_stamp(before)):
                raise MetadataInputError("metadata_directory_changed")
        finally:
            os.close(directory_fd)
    except (OSError, ValueError, OverflowError) as error:
        if isinstance(error, p2.JSONInputError):
            raise
        raise MetadataInputError("metadata_io") from None
    # Neither file is decoded until BOTH byte identities have matched.
    times = frame_time.derive_frame_times(parsed, _decode(train), _decode(test),
                                         reference.frames)
    return train, test, times
