"""Stdlib-only binary32/time handoff; no second time transformation.

Runtime bindings protect internal consumer correspondence, not arbitrary
hostile Python mutation. Learned time/scale/rotation remain trainable.
"""
from dataclasses import dataclass
import math
import struct


def binary32(value):
    try:
        result = struct.unpack('=f', struct.pack('=f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError('formal_time_binary32') from None
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError('formal_time_binary32')
    return result


def validate_classes(raw, effective):
    """Duplicates are legal; distinct raw classes must remain ordered.

    Sorting is validation only: never reorders, deduplicates or filters data.
    Callers provide the already once-converted effective values.
    """
    if len(raw) != len(effective):
        raise ValueError('formal_time_rows')
    classes = {}
    for a, b in zip(raw, effective):
        if not math.isfinite(a):
            raise ValueError('formal_time_raw')
        b = binary32(b)
        if a in classes and classes[a] != b:
            raise ValueError('formal_time_class')
        classes[a] = b
    ordered = [classes[a] for a in sorted(classes)]
    if any(a >= b for a, b in zip(ordered, ordered[1:])):
        raise ValueError('formal_time_class_collapse')


def validate_time(time, frames=()):
    a, b = map(binary32, time.effective_interval)
    if not a < b or binary32(time.effective_duration) <= 0:
        raise ValueError('formal_time_interval_binary32')
    binary32(time.log_effective_scale)
    validate_classes([f.raw_time for f in frames], [f.effective_time for f in frames])


@dataclass(frozen=True, eq=False)
class TimeBinding:
    config: object
    model: object
    camera: object
    time: object

    def validate(self, settings=None, prefilter_var=-1.0):
        self.camera.validate()
        if (self.config.time_derivation is not self.time
                or getattr(self.model, '_formal_time', None) is not self.time
                or tuple(self.model.time_duration) != self.time.effective_interval
                or type(prefilter_var) is not float or prefilter_var != -1.0
                or type(self.model.prefilter_var) is not float or self.model.prefilter_var != -1.0
                or self.config.renderer.temporal_prefilter != 'disabled'
                or self.model.gaussian_dim != 4 or not self.model.rot_4d):
            raise ValueError('formal_time_binding')
        validate_time(self.time, (self.camera.frame,))
        if settings is not None and (
                not settings.formal_time or not settings.formal_camera
                or settings.camera_binding is not self.camera
                or settings.timestamp != self.camera.frame.effective_time
                or settings.time_duration != self.time.effective_duration
                or settings.gaussian_dim != 4 or not settings.rot_4d):
            raise ValueError('formal_time_settings')


def bind_time(config, model, camera):
    binding = TimeBinding(config, model, camera, config.time_derivation)
    binding.validate()
    return binding


def validate_settings(settings, prefilter_var):
    if not settings.formal_time:
        if settings.time_binding is not None or settings.formal_camera:
            raise ValueError('formal_time_missing_binding')
        return
    if not isinstance(settings.time_binding, TimeBinding):
        raise ValueError('formal_time_missing_binding')
    settings.time_binding.validate(settings, prefilter_var)
