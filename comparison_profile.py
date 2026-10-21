"""Closed, adopted four-stage comparison. Not a formal config or resume API.

Only stage and output locations vary. Leaf value types/projections are shared
with the formal consumers; no formal JSON is manufactured or resolved state
patched. This module performs no heavy imports and grants no execution rights.
"""
from dataclasses import dataclass
import math

import formal_config as f
from formal_time_handoff import InheritedTime

DOMAIN = 'corrected_original_comparison_v1'
STAGES = ('stabilization', 'growth', 'e2_like', 'e3_like')
# N, xyz initial/final, scale, rotation, opacity, mask, from/until/interval,
# gradient threshold, opacity threshold, point threshold. Adopted A values.
_VALUES = (
    (12000, .00008, .0000008, .0001, .0005, .01, .02, 1000000000, 0, 100, 1., 0., -1),
    (30000, .00008, .0000008, .0002, .0005, .01, .02, 14000, 26000, 200, .0006, .002, -1),
    (12000, .00005, .0000005, .0003, .0003, .015, .01, 200, 3000, 400, .0006, .05, 2500000),
    (12000, .00002, .0000002, .00015, .00015, .01, .01, 2500, 9000, 100, .00022, .02, 3200000),
)


def state_byte_budget(rows):
    if type(rows) is not int or rows <= 0:
        raise ValueError('comparison_rows')
    return math.ceil((rows*796+9*8)*1.01)+1024**2


@dataclass(frozen=True)
class Initialization:
    # Scene still consumes the same PLY sampling RNG, but never builds its GPU model.
    num_pts: int = 600000
    num_extra_pts: int = 0
    seed: int = 0


@dataclass(frozen=True)
class ComparisonConfig:
    stage: str
    dataset: f.Dataset
    model: f.Model
    renderer: f.Renderer
    initialization: Initialization
    optimization: f.Optimization
    reporting: f.Reporting
    checkpoint: f.Checkpoint
    output: f.Output
    maximum_sh: f.MaximumSH
    time_derivation: InheritedTime
    save_updates: tuple
    paths: f.CanonicalPaths
    fixed: f.FixedConditions


def stage_config(stage, output_directory):
    from formal_inputs import _reference
    ref = _reference()
    return _stage_config(stage, output_directory, ref)


def _stage_config(stage, output_directory, ref):
    if type(stage) is not str or stage not in STAGES:
        raise ValueError('comparison_stage')
    n, first, last, scale, rotation, opacity, mask, start, until, interval, grad, prune, stop = _VALUES[STAGES.index(stage)]
    resolution = f.Resolution('fixed', 1, ref['width'], ref['height'], ref['width'], ref['height'])
    return ComparisonConfig(stage,
        f.Dataset('blender', ref['source_path'], True, '', False, resolution, f.TimeInput((0., 33.), 1.)),
        f.Model(4, 3, 2, True, False, 'native'),
        f.Renderer(False, False, 1., 0, None, 'disabled'), Initialization(),
        f.Optimization(n, 1, f.LearningRate(first, last, first, 12000, .0025, opacity, scale, rotation),
            f.Loss(.2, mask, 0., 0.),
            f.Population('unlimited' if stop == -1 else stop, .01, prune, grad, start, until, interval, 1000000000),
            f.SHSchedule(1000000000)),
        f.Reporting((n//2, n)), f.Checkpoint((n//2,)), f.Output(output_directory),
        f.MaximumSH(3, 2, 16, ((0, 16),), True), InheritedTime(), (n//2, n),
        f.CanonicalPaths(ref['source_path'], output_directory),
        f.FixedConditions(True, 'mask', True, 0, True, 0, 'exponential', 'constant', 20, True, True,
                          ((0, 0),)*6))


def validate_config(config):
    if type(config) is not ComparisonConfig:
        raise ValueError('comparison_profile')
    # Immutable dataset identity is checked on acquisition. Runtime predicates
    # must not reread the whole frame reference on every learning update.
    ref = dict(source_path=config.paths.source_path, width=config.dataset.resolution.raw_width,
               height=config.dataset.resolution.raw_height)
    if config != _stage_config(config.stage, config.paths.output_directory, ref):
        raise ValueError('comparison_profile')
    return config


def population_active(config, iteration, points):
    validate_config(config)
    p = config.optimization.population
    return iteration < p.densify_until_update and (
        p.densify_stop_points == 'unlimited' or points < p.densify_stop_points)
