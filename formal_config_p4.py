"""P4-only resolution checking and integer dimension derivation.

The raw dimensions are explicit caller inputs, not new JSON fields. Their
type/arithmetic checks do NOT prove agreement with images, dataset/camera
owner verification, or a formal handoff. No images, metadata, or paths are read.
The detached partial result is not full-config verification or execution
permission. Other nested fields/owners are neither validated nor adopted here.
"""

from dataclasses import dataclass

import formal_config_json as p2
import formal_config_p1 as p1


class P4InputError(p2.JSONInputError):
    """Fixed bounded codes only; never include input values or paths."""


@dataclass(frozen=True)
class P4ResolutionPartial:
    divisor: int
    raw_width: int
    raw_height: int
    width: int
    height: int


def derive_p4_resolution(parsed: p2.ParsedJSON, *, raw_width: int,
                         raw_height: int) -> P4ResolutionPartial:
    """Recheck current P1/P4 inputs; derive exact positive integer dimensions.

    No defaults, coercion, rounding, resize, or input mutation. P1 closes the
    root; resolution closes its two keys. At dataset level explicitly reject
    resolution_scales, the competing legacy authority. Do not infer aliases or
    recursively validate unrelated objects. Prior successful checks confer no
    authority on the current mutable ParsedJSON.
    """
    p1.check_p1_structure_and_fixed_inputs(parsed)
    dataset = parsed.root["dataset"]
    if "resolution_scales" in dataset:
        raise P4InputError("p4_competing_input")
    if "resolution" not in dataset:
        raise P4InputError("p4_missing_resolution")
    resolution = dataset["resolution"]
    if type(resolution) is not dict:
        raise P4InputError("p4_resolution_object")
    if resolution.keys() != {"mode", "divisor"}:
        raise P4InputError("p4_resolution_keys")
    if type(resolution["mode"]) is not str:
        raise P4InputError("p4_mode_type")
    if resolution["mode"] != "integer_divisor":
        raise P4InputError("p4_mode_value")
    divisor = p2.require_int(resolution["divisor"])
    if divisor < 1:
        raise P4InputError("p4_divisor_range")
    if type(raw_width) is not int or type(raw_height) is not int:
        raise P4InputError("p4_dimension_type")
    if raw_width < 1 or raw_height < 1:
        raise P4InputError("p4_dimension_range")
    width, width_remainder = divmod(raw_width, divisor)
    height, height_remainder = divmod(raw_height, divisor)
    if width_remainder or height_remainder:
        raise P4InputError("p4_not_divisible")
    # Positive operands and exact divisibility imply positive quotients.
    return P4ResolutionPartial(divisor, raw_width, raw_height, width, height)
