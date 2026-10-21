"""Comparison-only physical16 state; shared strict Torch payload primitives.

No generic resume: a stage initializer installs only nine Parameters, then the
existing training_setup starts Adam/statistics from zero. Reader restore uses
restore(None) and never training_setup. No changes to formal reader acceptance.
"""
from dataclasses import dataclass

import torch

import formal_checkpoint_state as state
from formal_checkpoint_identity import GROUPS, require
from comparison_profile import DOMAIN, validate_config, state_byte_budget

SHAPES = {**state.SHAPES, 3: (15, 3)}
INITIAL_KEYS = ('xyz', 'features_dc', 'features_rest', 'scaling', 'rotation',
                'opacity', 't', 'scaling_t', 'rotation_r')
SLOTS = (1, 2, 3, 4, 5, 6, 13, 14, 15)


def byte_budget(rows):
    return state_byte_budget(rows)


def validate_state(payload, meta, config):
    validate_config(config)
    return state._validate_state(payload, meta, config, DOMAIN, SHAPES)


def snapshot(model, config, k):
    validate_config(config)
    return state._snapshot(model, config, k, DOMAIN, SHAPES)


def load_state(stream, meta, config):
    validate_config(config)
    return state._load_state(stream, meta, config, DOMAIN, SHAPES, byte_budget(meta['rows']))


def _parameters(values):
    require(type(values) is tuple and len(values) == 9, 'comparison_parameters')
    rows = len(values[0])
    require(rows > 0, 'comparison_rows')
    for slot, tensor in zip(SLOTS, values):
        require(type(tensor) is torch.Tensor and tensor.device.type == 'cpu' and
                tensor.dtype == torch.float32 and tuple(tensor.shape) == (rows,)+SHAPES[slot] and
                not tensor.requires_grad and bool(torch.isfinite(tensor).all()), 'comparison_initial_shape')
        if slot in (4, 14):
            scale = tensor.exp()
            require(bool(torch.isfinite(scale).all() and (scale > 0).all()), 'comparison_initial_scale')
        if slot in (5, 15):
            norm = torch.linalg.vector_norm(tensor, dim=1)
            require(bool(torch.isfinite(norm).all() and (norm > 0).all()), 'comparison_initial_rotation')
    return values


@dataclass
class Initializer:
    parameters: tuple
    active: tuple
    parent_stage: str | None
    consumed: bool = False

    @classmethod
    def from_initial(cls, initial):
        require(type(initial) is dict and set(initial) == set(INITIAL_KEYS), 'comparison_initial_fields')
        values = tuple(initial[name].transpose(1, 2) if name in ('features_dc', 'features_rest')
                       else initial[name] for name in INITIAL_KEYS)
        return cls(_parameters(values), (0, 0), None)

    @classmethod
    def from_checkpoint(cls, parent, next_config):
        from comparison_profile import STAGES
        from staged_comparison_results import read_confirmed
        validate_config(next_config)
        require(type(parent) is dict and set(parent)=={'kind','path','identity','stage','completed','verification'},
                'comparison_parent_unverified')
        result=read_confirmed(parent,next_config)
        c, meta, payload = result['config'], result['metadata'], result['state']
        validate_state(payload, meta, c)
        index = STAGES.index(next_config.stage)
        require(index > 0 and c.stage == STAGES[index-1] and meta['final'] is True and
                payload['k'] == c.optimization.total_updates,
                'comparison_parent_final')
        a = payload['capture']
        return cls(_parameters(tuple(a[s] for s in SLOTS)), (a[0], a[18]), c.stage)

    def initialize(self, model, extent, config):
        from comparison_profile import STAGES
        validate_config(config)
        index = STAGES.index(config.stage)
        require(self.parent_stage == (None if index == 0 else STAGES[index-1]) and
                self.active == (0, 0) and model.optimizer is None and not self.consumed, 'comparison_initial_stage')
        values = _parameters(self.parameters)
        # Same mapping/transpose semantics as create_from_pth, without reopening
        # pickle, loading a legacy checkpoint or constructing a disposable model.
        for slot, tensor in zip(SLOTS, values):
            attr = next(attr for s, attr in GROUPS.values() if s == slot)
            setattr(model, attr, torch.nn.Parameter(tensor.to('cuda').contiguous().clone()))
        model.active_sh_degree, model.active_sh_degree_t = self.active
        model.spatial_lr_scale = extent
        model.max_radii2D = model._xyz.new_zeros(len(model._xyz))
        model._formal_time = config.time_derivation
        # Do not retain the former stage's CPU Parameter storage for all N updates.
        self.parameters = ()
        self.consumed = True


def restore_diagnostic(result, device):
    c, payload, meta = result['config'], result['state'], result['metadata']
    validate_state(payload, meta, c)
    # Native evidence must have been checked by the comparison reader before
    # entering the heavy runtime. This adapter never loads a native library.
    from scene.gaussian_model import GaussianModel
    model = GaussianModel(3, gaussian_dim=4, time_duration=c.time_derivation.effective_interval,
                          rot_4d=True, force_sh_3d=False, sh_degree_t=2, prefilter_var=-1.0)
    a = list(payload['capture'])
    for slot in SHAPES:
        a[slot] = a[slot].to(device).clone()
    model.restore(tuple(a), training_args=None)
    model.xyz_gradient_accum, model.t_gradient_accum, model.denom = a[8], a[9], a[10]
    model.percent_dense = c.optimization.population.percent_dense
    model._formal_time = c.time_derivation
    state.model_contract(model, c)
    require(model.optimizer is None, 'comparison_reader_optimizer')
    return model
