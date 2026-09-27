"""Pure D-TIME frame correspondence check and one raw-to-effective transform.

This is a PARTIAL result, not dataset authorization, full metadata validation,
loader equivalence, verified runtime state, or permission to run. The caller's
dataset owner must bind the separately supplied reference to approved evidence;
matching an arbitrary caller-created reference cannot establish that provenance.
Unknown metadata fields are not consumed or certified. No reader, path resolver,
writer, camera construction, Gaussian domain, or runtime handoff is provided.
"""

from dataclasses import dataclass
import math

import formal_config_json as p2
import formal_config_time as time_part


class FrameTimeInputError(p2.JSONInputError):
    """Fixed bounded code, without echoing metadata or paths."""


@dataclass(frozen=True)
class FrameTime:
    """Detached correspondence; index is the original per-split array index."""

    split: str
    index: int
    file_path: str
    raw_time: float
    effective_time: float


@dataclass(frozen=True)
class FrameTimes:
    """Immutable partial result, never a token accepted as raw input."""

    time: time_part.TimeDerivation
    train: tuple[FrameTime, ...]
    test: tuple[FrameTime, ...]


def _record(record, interval, seen):
    if type(record) is not dict:
        raise FrameTimeInputError("frame_record_type")
    if "file_path" not in record or "time" not in record:
        raise FrameTimeInputError("frame_missing_input")
    path = record["file_path"]
    if type(path) is not str:
        raise FrameTimeInputError("frame_path_type")
    raw = p2.require_num(record["time"])
    if not interval[0] <= raw <= interval[1]:
        raise FrameTimeInputError("frame_time_range")
    # Identity is an exact path, not a timestamp. Keep one set across splits:
    # a duplicate in either split or their intersection must fail, not filter.
    if path in seen:
        raise FrameTimeInputError("frame_duplicate_path")
    seen.add(path)
    return path, raw


def derive_frame_times(parsed: p2.ParsedJSON, train: list, test: list,
                       reference: dict) -> FrameTimes:
    """Check all read/decoded records against a separately preserved reference.

    ``reference`` is an internal mapping of train/test to ordered frame lists,
    each record containing exact file_path and P2 NumberToken time. It is not a
    new external JSON schema. Do not build it from a candidate under inspection.
    Recheck current config through the time component on EVERY call; previously
    returned TimeDerivation/FrameTimes objects are not accepted as proof.

    Correspondence uses P2's binary64 Num value (not numeric spelling). Neither
    input nor reference is mutated, sorted, filtered, normalized or repaired.
    All records must succeed before a result is returned. Other-owner config
    fields and concurrent caller mutation remain outside this partial contract.
    """
    derived = time_part.derive_time_interval_and_variance(parsed)
    if type(reference) is not dict or reference.keys() != {"train", "test"}:
        raise FrameTimeInputError("frame_reference_shape")
    candidate_seen, reference_seen = set(), set()
    splits = []
    for split, frames in (("train", train), ("test", test)):
        expected = reference[split]
        if type(frames) is not list or type(expected) is not list:
            raise FrameTimeInputError("frame_array_type")
        if len(frames) != len(expected):
            raise FrameTimeInputError("frame_count")
        records = []
        for index, (frame, original) in enumerate(zip(frames, expected)):
            path, raw = _record(frame, derived.raw_interval, candidate_seen)
            expected_path, expected_raw = _record(
                original, derived.raw_interval, reference_seen)
            if path != expected_path or raw != expected_raw:
                raise FrameTimeInputError("frame_correspondence")
            effective = raw / derived.divisor
            if not math.isfinite(effective) or (raw != 0.0 and effective == 0.0):
                raise FrameTimeInputError("frame_time_conversion")
            records.append(FrameTime(split, index, path, raw, effective))
        splits.append(tuple(records))
    return FrameTimes(derived, splits[0], splits[1])
