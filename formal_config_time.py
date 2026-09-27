"""D-TIME interval and V-B initialization arithmetic, NOT whole-config acceptance.

Only current P1 premises and the explicitly consumed time fields are checked.
Other nested fields remain UNCHECKED, not approved extensions. No alias/default
authority, frame/PLY reader, individual timestamp conversion, dataset identity
check, runtime handoff, writer, or float32 acceptance is provided here.
"""

from dataclasses import dataclass
import math

import formal_config_json as p2
import formal_config_p1 as p1


class TimeInputError(p2.JSONInputError):
    """Fixed stage-specific code only; never echo input values or paths."""


@dataclass(frozen=True)
class TimeDerivation:
    """Detached immutable PARTIAL result, not verified state or permission.

    Direct construction does not certify provenance. Variance, activated scale
    and stored log-scale are distinct. V-B assumes initial R=I, identity qL/qR,
    scaling_modifier=1; it is not a rotated-checkpoint transformation.
    """

    raw_interval: tuple[float, float]
    effective_interval: tuple[float, float]
    raw_duration: float
    effective_duration: float
    divisor: float
    time_variance_denominator: float
    raw_variance: float
    raw_scale: float
    effective_variance: float
    effective_scale: float
    log_effective_scale: float


def _required(obj, key):
    if key not in obj:
        raise TimeInputError("time_missing_input")
    return obj[key]


def _positive(value, code):
    # Positive subnormals are not rejected merely for being subnormal.
    if not math.isfinite(value) or value <= 0.0:
        raise TimeInputError(code)
    return value


def _endpoint(raw, divisor):
    effective = raw / divisor
    if not math.isfinite(effective) or (raw != 0.0 and effective == 0.0):
        raise TimeInputError("time_endpoint_conversion")
    return effective


def derive_time_interval_and_variance(parsed: p2.ParsedJSON) -> TimeDerivation:
    """Recheck current P1 and P2 Num inputs, then derive the adopted V-B subset.

    No mutation, cached validation, legacy fallback or semantic default. Past
    P1 success and caller-fabricated/concurrently mutated parse data acquire no
    new provenance guarantee. The first baseline must explicitly supply c=5;
    the field domain remains positive Num, not an implicit 5 or a fixed-only 5.
    Test examples do not select an actual run interval, divisor or coefficient.
    """
    p1.check_p1_structure_and_fixed_inputs(parsed)
    time = _required(parsed.root["dataset"], "time")
    if type(time) is not dict:
        raise TimeInputError("time_object_type")
    interval = _required(time, "raw_interval")
    if type(interval) is not list or len(interval) != 2:
        raise TimeInputError("time_interval_shape")
    a, b = (p2.require_num(token) for token in interval)
    d = p2.require_num(_required(time, "divisor"))
    c = p2.require_num(_required(parsed.root["initialization"],
                                 "time_variance_denominator"))
    if not a < b:
        raise TimeInputError("time_interval_order")
    if d < 1.0:
        raise TimeInputError("time_divisor_domain")
    if c <= 0.0:
        raise TimeInputError("time_denominator_domain")

    raw_duration = _positive(b - a, "time_raw_duration")
    a_eff, b_eff = _endpoint(a, d), _endpoint(b, d)
    if not a_eff < b_eff:
        raise TimeInputError("time_endpoint_collapse")
    effective_duration = _positive(b_eff - a_eff, "time_effective_duration")
    raw_variance = _positive(raw_duration / c, "time_raw_variance")
    raw_scale = _positive(math.sqrt(raw_variance), "time_raw_scale")
    effective_scale = _positive(raw_scale / d, "time_effective_scale")

    # V-B: L_raw/(c*d^2), NOT V-A: L_eff/c. Avoid constructing d*d or c*d*d.
    # Since d>=1, these divisions cannot create an unnecessary large temporary;
    # if the temporary becomes zero, the final variance cannot be positive.
    variance_over_d = _positive(raw_variance / d, "time_variance_intermediate")
    effective_variance = _positive(variance_over_d / d, "time_effective_variance")
    log_effective_scale = math.log(effective_scale)
    if not math.isfinite(log_effective_scale):
        raise TimeInputError("time_log_scale")

    return TimeDerivation(
        raw_interval=(a, b), effective_interval=(a_eff, b_eff),
        raw_duration=raw_duration, effective_duration=effective_duration,
        divisor=d, time_variance_denominator=c, raw_variance=raw_variance,
        raw_scale=raw_scale, effective_variance=effective_variance,
        effective_scale=effective_scale, log_effective_scale=log_effective_scale,
    )
