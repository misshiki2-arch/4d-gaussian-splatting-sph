"""Read-only live checks, imported only by the heavy formal runtime.

No snapshots, repairs or history monitor. Predicates synchronize with the
device; their cost is intentionally not claimed to be negligible.
"""
import math
import numbers

import torch

from formal_checkpoint_identity import GROUPS
from formal_checkpoint_state import SHAPES, GROUP_KEYS, _rates


class HealthError(ValueError):
    pass


def require(ok, field):
    if not ok:
        raise HealthError('formal_health:' + field)


def tensor(value, shape, device, field, dtype=torch.float32):
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
            and value.device == device and value.dtype == dtype, field + '/layout')
    require(bool(torch.isfinite(value.detach()).all()), field + '/finite')


def scalar(value, field):
    require(isinstance(value, torch.Tensor) and value.numel() == 1 and
            bool(torch.isfinite(value.detach()).all()), field)


def metric(value, field):
    """Evaluation consumers return either a real Python number or a tensor.

    Read only; never convert/replace a loss, detach its graph, or extend the
    channel-specific PSNR exception to other metrics.
    """
    if isinstance(value, torch.Tensor):
        require(value.ndim == 0 and value.dtype != torch.bool and not value.is_complex()
                and bool(torch.isfinite(value.detach())), field)
    else:
        require(isinstance(value, numbers.Real) and not isinstance(value, bool)
                and math.isfinite(value), field)


@torch.no_grad()
def statistics(model):
    m, device = len(model._xyz), model._xyz.device
    for name, shape in (('xyz_gradient_accum', (m, 1)), ('t_gradient_accum', (m, 1)),
                        ('denom', (m, 1)), ('max_radii2D', (m,))):
        tensor(getattr(model, name), shape, device, name)
    require(bool((model.xyz_gradient_accum >= 0).all() and (model.max_radii2D >= 0).all()
                 and (model.denom >= 0).all()), 'statistics/nonnegative')
    require(torch.equal(model.denom, model.denom.trunc()), 'statistics/denom_integer')
    unseen = model.denom == 0
    require(bool((model.xyz_gradient_accum[unseen] == 0).all() and
                 (model.t_gradient_accum[unseen] == 0).all()), 'statistics/unobserved')


def _binding(model):
    groups = model.optimizer.param_groups
    require(len(groups) == len(GROUPS) and {g.get('name') for g in groups} == set(GROUPS), 'groups')
    for g in groups:
        require(set(g) == set(GROUP_KEYS.split()) and len(g['params']) == 1 and
                g['params'][0] is getattr(model, GROUPS[g['name']][1]), 'parameter_binding')
    require(all(any(p is g['params'][0] for g in groups) for p in model.optimizer.state), 'extra_state')
    return groups


def optimizer_settings(model, config, k):
    rates = _rates(config, model.spatial_lr_scale, k) if k else None
    for g in _binding(model):
        lr = g['lr']
        require(isinstance(lr, numbers.Real) and not isinstance(lr, bool) and math.isfinite(lr) and lr >= 0,
                'lr/' + g['name'])
        if rates is not None:
            require(float(lr) == rates[g['name']], 'lr_contract/' + g['name'])
        require(g['betas'] == (0.9, 0.999) and type(g['betas']) is tuple and g['eps'] == 1e-15 and
                g['weight_decay'] == 0 and all(g[n] is False for n in
                ('amsgrad', 'maximize', 'capturable', 'differentiable')) and
                g['foreach'] is None and g['fused'] is None, 'optimizer_settings/' + g['name'])


def _parameter(model, name, shapes=SHAPES):
    slot, attr = GROUPS[name]
    p = getattr(model, attr)
    require(isinstance(p, torch.nn.Parameter), attr + '/parameter')
    tensor(p, (len(model._xyz),) + shapes[slot], model._xyz.device, attr)
    if name in ('scaling', 'scaling_t'):
        scale = p.detach().exp()
        require(bool(torch.isfinite(scale).all() and (scale > 0).all()), attr + '/activation')
    if name in ('rotation', 'rotation_r'):
        norm = torch.linalg.vector_norm(p.detach(), dim=1)
        require(bool(torch.isfinite(norm).all() and (norm > 0).all()), attr + '/norm')
    return p


def _adam(model, p, k):
    state = model.optimizer.state.get(p)
    if not state:  # Adam's legitimate lazy state; do not create it by lookup.
        return
    require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'adam/fields')
    step = state['step']
    require(isinstance(step, torch.Tensor) and step.ndim == 0 and step.dtype in
            (torch.float32, torch.float64, torch.int64) and
            (step.device.type == 'cpu' or step.device == p.device), 'adam/step_layout')
    require(bool(torch.isfinite(step) and (step >= 0) and (step <= k) and (step == step.trunc())), 'adam/step')
    for name in ('exp_avg', 'exp_avg_sq'):
        tensor(state[name], p.shape, p.device, 'adam/' + name)
    require(bool((state['exp_avg_sq'] >= 0).all()), 'adam/second_moment')


@torch.no_grad()
def model_state(model, config, k):
    from comparison_profile import ComparisonConfig, validate_config
    shapes = SHAPES
    if type(config) is ComparisonConfig:
        validate_config(config)
        shapes = dict(SHAPES, **{})
        shapes[3] = (15, 3)
    require(len(model._xyz) > 0, 'empty_population')
    require(model.env_map is None or (k == 0 and isinstance(model.env_map, torch.Tensor)
                                    and model.env_map.numel() == 0), 'env_map')
    optimizer_settings(model, config, k)
    for name in GROUPS:
        p = _parameter(model, name, shapes)
        _adam(model, p, k)
    statistics(model)


@torch.no_grad()
def opacity_reset(model, k):
    _binding(model)
    _adam(model, _parameter(model, 'opacity'), k)


@torch.no_grad()
def raw_render(value, gt, model):
    require(isinstance(gt, torch.Tensor) and gt.ndim == 3 and gt.shape[0] == 3 and
            gt.shape[1] > 0 and gt.shape[2] > 0, 'GT/shape')
    device, m = model._xyz.device, len(model._xyz)
    tensor(gt, gt.shape, device, 'GT')
    for name, shape in (('render', gt.shape), ('alpha', (1, *gt.shape[1:])),
                        ('depth', (1, *gt.shape[1:])), ('viewspace_points', (m, 3))):
        require(name in value, 'render/missing_' + name)
        tensor(value[name], shape, device, 'render/' + name)
    # Native radii are int32; CPU controls may use floating radii.
    radii = value['radii']
    require(isinstance(radii, torch.Tensor) and radii.dtype in (torch.int32, torch.float32), 'radii/type')
    tensor(radii, (m,), device, 'radii', radii.dtype)
    require(bool((radii >= 0).all()), 'radii/nonnegative')
    tensor(value['visibility_filter'], (m,), device, 'visibility', torch.bool)


@torch.no_grad()
def gradients(model, screen):
    for name, (_, attr) in GROUPS.items():
        p = getattr(model, attr)
        if p.grad is not None:
            tensor(p.grad, p.shape, p.device, 'gradient/' + name)
    require(screen.grad is not None and model._t.grad is not None, 'required_gradient')
    tensor(screen.grad, screen.shape, model._xyz.device, 'gradient/screen')


@torch.no_grad()
def statistic_input(model, screen_norm, time_grad):
    m, device = len(model._xyz), model._xyz.device
    tensor(screen_norm, (m, 1), device, 'statistic_input/screen')
    require(bool((screen_norm >= 0).all()), 'statistic_input/nonnegative')
    tensor(time_grad, (m, 1), device, 'statistic_input/time')


@torch.no_grad()
def psnr_channels(result, image, gt):
    require(image.shape == gt.shape and image.ndim == 3 and bool(torch.isfinite(image).all()
            and torch.isfinite(gt).all()), 'psnr/input')
    require(tuple(result.shape) == (image.shape[0], 1), 'psnr/shape')
    equal = (image == gt).reshape(image.shape[0], -1).all(dim=1, keepdim=True)
    require(bool((torch.isfinite(result) | (torch.isposinf(result) & equal)).all()), 'psnr/value')
    return result
