"""P5 schedule checks/derivation only, NOT full formal-config acceptance.

Recheck current P2 parse data through P1 on every call; never mutate input.
Only the three P5 inputs and explicit immediate competing paths below are
checked here. Other nested fields are UNCHECKED, not approved extensions.
No P3 acceptance, run/cadence approval, runtime adapter, checkpoint writer or
test executor is provided. Checkpoint-before-test observation of the same
completed state remains a later runtime obligation, not proven by these lists.
"""

from dataclasses import dataclass

import formal_config_json as p2
import formal_config_p1 as p1


# P5's named competing authorities at the three update/schedule-owning objects.
# P1 rejects flat top-level names. No recursive alias guessing or full schema.
_SCHEDULE_OBJECTS = ("optimization", "checkpoint", "reporting")
_COMPETING_INPUTS = frozenset(("final_iteration", "save_iterations"))


class P5InputError(p2.JSONInputError):
    """Fixed bounded code only; never echo input text, keys, values or paths."""


@dataclass(frozen=True)
class P5Schedules:
    """Detached immutable PARTIAL result, not verified state or permission.

    Direct construction does not certify provenance. No active runtime state,
    event executor, reporting cadence decision or publication identity is here.
    """

    total_updates: int
    save_updates: tuple[int, ...]
    test_updates: tuple[int, ...]


def _required(obj, key):
    if key not in obj:
        raise P5InputError("p5_missing_input")
    return obj[key]


def _updates(value, upper):
    if type(value) is not list:
        raise P5InputError("p5_array_type")
    # Preserve P2's input bound even after mutation of ordinary parse data.
    # The derived save tuple may have one more element; it is not JSON input.
    if len(value) > p2.MAX_ARRAY_ELEMENTS:
        raise P5InputError("p5_array_limit")
    result = []
    previous = 0
    for token in value:
        update = p2.require_int(token)
        if not 1 <= update <= upper:
            raise P5InputError("p5_update_range")
        if update <= previous:
            raise P5InputError("p5_update_order")
        result.append(update)
        previous = update
    return tuple(result)


def derive_p5_schedules(parsed: p2.ParsedJSON) -> P5Schedules:
    """Validate current P1/P5 premises, then append N only to save updates.

    Past P1 success (None) is not a verified token. Caller-fabricated parse data
    and concurrent mutation are not given a new provenance guarantee. Work and
    storage depend on input schedule lengths, never the magnitude of N.
    """
    p1.check_p1_structure_and_fixed_inputs(parsed)
    root = parsed.root
    for group in _SCHEDULE_OBJECTS:
        if _COMPETING_INPUTS.intersection(root[group]):
            raise P5InputError("p5_competing_input")
    total = p2.require_int(_required(root["optimization"], "total_updates"))
    if total < 1:
        raise P5InputError("p5_total_range")
    intermediate = _updates(
        _required(root["checkpoint"], "intermediate_updates"), total - 1)
    tests = _updates(_required(root["reporting"], "test_updates"), total)
    return P5Schedules(total, intermediate + (total,), tests)
