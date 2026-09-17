"""P3 maximum-SH derivation only, NOT whole-config acceptance or an adapter.

Consume current P2 parse data through P1 on every call. Do not mutate input,
choose active degrees/schedules, import runtime code, or write output. Nested
fields outside P1 and the explicit P3 prohibitions below remain UNCHECKED, not
approved extensions. This is not a complete nested schema or alias detector.
"""

from dataclasses import dataclass

import formal_config_json as p2
import formal_config_p1 as p1


# Exact, immediate JSON paths only. P1 already rejects flat top-level inputs.
# P3 forbids the old boolean and second enabled/layout/count authorities.
# Model constructor/state spellings are not alternate maximum-degree inputs.
# No recursive name guessing or closing model to P1's fixed fields.
_MODEL_FORBIDDEN = frozenset((
    "eval_shfs_4d", "temporal_sh_enabled", "sh_layout", "sh_slot_count",
    "sh_degree", "sh_degree_t", "max_sh_degree", "max_sh_degree_t",
    "max_sh_channels",
))
_RENDERER_FORBIDDEN = frozenset((
    "eval_shfs_4d", "temporal_sh_enabled", "sh_layout", "sh_slot_count",
))


class P3InputError(p2.JSONInputError):
    """Fixed bounded code only; no input text, field names, or paths."""


@dataclass(frozen=True)
class P3MaximumSH:
    """Detached PARTIAL result, not verified config or execution permission.

    Inclusive slot ranges are spatial base, temporal mode 1, temporal mode 2.
    No active-degree state, schedule, CUDA buffer, or runtime adapter is here.
    Like parse data, constructing this class does not certify provenance.
    """

    max_spatial_degree: int
    max_temporal_degree: int
    slot_count: int
    mode_slot_ranges: tuple[tuple[int, int], ...]
    legacy_eval_shfs_4d: bool


def derive_p3_maximum_sh(parsed: p2.ParsedJSON) -> P3MaximumSH:
    """Recheck current P1 premises, then derive only the adopted P3 maximum.

    A past P1 success (None) is not a trusted token. Caller-fabricated parse
    data and concurrent mutation are not given a new provenance guarantee.
    Other-owner nested fields, including active degrees, are not consumed.
    """
    p1.check_p1_structure_and_fixed_inputs(parsed)
    model, renderer = parsed.root["model"], parsed.root["renderer"]
    if (_MODEL_FORBIDDEN.intersection(model)
            or _RENDERER_FORBIDDEN.intersection(renderer)):
        raise P3InputError("p3_competing_input")

    # Preserve NumberToken origin; P1 has enforced the explicit fixed 3/2.
    spatial = p2.require_int(model["spatial_sh_degree"])
    temporal = p2.require_int(model["temporal_sh_degree"])
    spatial_slots = (spatial + 1) ** 2
    slots = spatial_slots * (temporal + 1)
    ranges = tuple((mode * spatial_slots, (mode + 1) * spatial_slots - 1)
                   for mode in range(temporal + 1))
    if slots != 48 or ranges != ((0, 15), (16, 31), (32, 47)):
        raise P3InputError("p3_maximum_layout")
    return P3MaximumSH(spatial, temporal, slots, ranges, temporal > 0)
