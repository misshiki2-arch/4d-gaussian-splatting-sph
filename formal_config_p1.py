"""P1 structure/fixed-input checks only, NOT full formal-config acceptance.

Consume P2 ParsedJSON without mutating it or discarding NumberToken origin.
Nested fields outside the P1 fixed-input table remain UNCHECKED, not approved
extensions. Complete nested membership, other field semantics, immutable
verified state, runtime adapters and execution permission belong to later work.
No parser, defaults, locator, CLI, runtime import or output writer lives here.
"""

import formal_config_json as p2


_OBJECT_KEYS = (
    "dataset", "model", "renderer", "initialization", "optimization",
    "reporting", "checkpoint", "output",
)
_TOP_LEVEL_KEYS = frozenset(("schema", "run_mode", *_OBJECT_KEYS))


class P1InputError(p2.JSONInputError):
    """Fixed bounded code only; never include input keys, values or paths."""


def _string(value):
    if type(value) is not str:
        raise P1InputError("p1_string_type")
    return value


def _null(value):
    if value is not None:
        raise P1InputError("p1_null_type")
    return None


def _fixed(obj, key, expected, check):
    if key not in obj:
        raise P1InputError("p1_missing_fixed")
    # P2 helpers validate token type/domain before equality (bool != number).
    # Converted values are local comparisons, never replacements in parse data.
    if check(obj[key]) != expected:
        raise P1InputError("p1_fixed_value")


def check_p1_structure_and_fixed_inputs(parsed: p2.ParsedJSON) -> None:
    """Raise on a P1 violation; return None on this PARTIAL check's success.

    Call with P2 parse data, not a Python/legacy config. This does not establish
    parsing provenance for caller-fabricated or subsequently mutated objects.
    It does not return a verified state or authorize execution. Even minimal
    P1-only input can pass while lacking required later-owner semantics.
    """
    if type(parsed) is not p2.ParsedJSON or type(parsed.root) is not dict:
        raise P1InputError("p1_parsed_input")
    root = parsed.root
    if root.keys() != _TOP_LEVEL_KEYS:
        raise P1InputError("p1_top_level_keys")
    if (_string(root["schema"]) != "corrected_4dgs_training_config_v1"
            or _string(root["run_mode"]) != "from_scratch"):
        raise P1InputError("p1_literal")
    for key in _OBJECT_KEYS:
        if type(root[key]) is not dict:
            raise P1InputError("p1_object")

    dataset, model, renderer = root["dataset"], root["model"], root["renderer"]
    _fixed(dataset, "kind", "nerf_transforms", _string)
    _fixed(dataset, "eval", True, p2.require_bool)
    _fixed(model, "gaussian_dim", 4, p2.require_int)
    _fixed(model, "spatial_sh_degree", 3, p2.require_int)
    _fixed(model, "temporal_sh_degree", 2, p2.require_int)
    _fixed(model, "rot_4d", True, p2.require_bool)
    _fixed(model, "force_sh_3d", False, p2.require_bool)
    _fixed(model, "sh_evaluation", "conditional_mean", _string)
    _fixed(renderer, "compute_cov3D_python", False, p2.require_bool)
    _fixed(renderer, "convert_SHs_python", False, p2.require_bool)
    _fixed(renderer, "scaling_modifier", 1.0, p2.require_num)
    _fixed(renderer, "env_map_res", 0, p2.require_int)
    _fixed(renderer, "override_color", None, _null)
    _fixed(renderer, "temporal_prefilter", "disabled", _string)
