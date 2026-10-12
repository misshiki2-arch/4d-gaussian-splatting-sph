"""Stage-T Torch CPU payload validation and diagnostic-only model adapter."""
import math
import zipfile

import numpy as np
import torch

from formal_checkpoint_identity import (SCHEMA, GROUPS, NativeEvidence, canonical,
    keys, require, validate_native, validate_optimizer_descriptor)
SHAPES = {1:(3,), 2:(1,3), 3:(47,3), 4:(3,), 5:(4,), 6:(1,), 7:(),
          8:(1,), 9:(1,), 10:(1,), 13:(1,), 14:(1,), 15:(4,)}
GROUP_KEYS = 'params name lr betas eps weight_decay amsgrad maximize foreach capturable differentiable fused'


def _clone(value):
    if isinstance(value, torch.Tensor): return value.detach().to('cpu').contiguous().clone()
    if isinstance(value, np.generic): return value.item()
    if type(value) is dict: return {k: _clone(v) for k, v in value.items()}
    if type(value) in (tuple, list): return type(value)(_clone(v) for v in value)
    require(type(value) in (int, float, bool, str, type(None)), 'snapshot_type')
    return value


def equal_state(left, right):
    require(type(left) is type(right), 'roundtrip_type')
    if isinstance(left, torch.Tensor):
        require(left.dtype == right.dtype and left.shape == right.shape and
                torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)), 'roundtrip_tensor')
    elif type(left) is dict:
        require(left.keys() == right.keys(), 'roundtrip_keys')
        for k in left: equal_state(left[k], right[k])
    elif type(left) in (tuple, list):
        require(len(left) == len(right), 'roundtrip_length')
        for a, b in zip(left, right): equal_state(a, b)
    else: require(left == right, 'roundtrip_value')


def model_contract(model, c):
    require(model.gaussian_dim == 4 and model.rot_4d is True and model.force_sh_3d is False and
            model.max_sh_degree == 3 and model.max_sh_degree_t == 2 and model.env_map is None and
            type(model.prefilter_var) is float and model.prefilter_var == -1.0 and
            tuple(model.time_duration) == c.time_derivation.effective_interval and
            model._formal_time is c.time_derivation and
            model.percent_dense == c.optimization.population.percent_dense, 'model_semantics')


def _rates(c, extent, k):
    lr = c.optimization.learning_rate
    # Existing Scene normalization produces binary32 extent; preserve its
    # conversion/multiplication semantics used in the existing scheduler.
    extent = np.float32(extent)
    first, last = lr.position_lr_init*extent, lr.position_lr_final*extent
    if first == last == 0: xyz = 0.0
    else:
        t = np.clip(k/lr.position_lr_max_steps, 0, 1)
        xyz = np.exp(np.log(first)*(1-t)+np.log(last)*t)
    return dict(xyz=float(xyz), f_dc=lr.feature_lr, f_rest=lr.feature_lr/20,
        opacity=lr.opacity_lr, scaling=lr.scaling_lr, rotation=lr.rotation_lr,
        t=float(lr.position_t_lr_init*extent), scaling_t=lr.scaling_lr, rotation_r=lr.rotation_lr)


def validate_state(payload, meta, c):
    keys(payload, 'schema layout k capture')
    require(payload['schema'] == SCHEMA and type(payload['layout']) is int and payload['layout'] == 1 and
            type(payload['k']) is int and payload['k'] == meta['k'], 'payload_version_label')
    a = payload['capture']; k, m = meta['k'], meta['rows']
    require(type(a) is tuple and len(a) == 19, 'capture_layout')
    stage = c.fixed.sh_stages[min(k//c.optimization.sh.increase_interval, 5)]
    require(type(a[0]) is int and type(a[18]) is int and (a[0], a[18]) == stage and
            a[16] is True and a[17] is None, 'capture_semantics')
    for i, tail in SHAPES.items():
        t = a[i]
        require(type(t) is torch.Tensor and t.device.type == 'cpu' and t.dtype == torch.float32 and
                tuple(t.shape) == (m,)+tail and t.is_contiguous() and not t.requires_grad and
                bool(torch.isfinite(t).all()), 'tensor_shape_type_finite')
    for i in (4,14):
        activated = torch.exp(a[i])
        require(bool(torch.isfinite(activated).all() and (activated > 0).all()), 'scale_activation')
    for i in (5,15):
        norm = torch.linalg.vector_norm(a[i], dim=1)
        require(bool(torch.isfinite(norm).all() and (norm > 0).all()), 'quaternion')
    for i in (7,8,10): require(bool((a[i] >= 0).all()), 'nonnegative_statistic')
    require(torch.equal(a[10], torch.trunc(a[10])), 'denom')
    require(type(a[12]) is float and math.isfinite(a[12]) and a[12] > 0, 'extent')
    opt = a[11]; keys(opt, 'state param_groups')
    require(type(opt['state']) is dict and type(opt['param_groups']) is list and
            len(opt['param_groups']) == len(GROUPS), 'optimizer')
    desc = meta['optimizer']
    validate_optimizer_descriptor(desc)  # Also covers direct snapshot/adapter calls.
    names, ids = set(), set()
    expected_rates = _rates(c, a[12], k)
    for group in opt['param_groups']:
        keys(group, GROUP_KEYS)
        name = group['name']
        require(type(name) is str and name in GROUPS and name not in names, 'optimizer_name')
        names.add(name)
        require(type(group['params']) is list and len(group['params']) == 1, 'optimizer_params')
        identity = group['params'][0]
        require(type(identity) is int and identity >= 0 and identity not in ids, 'optimizer_id'); ids.add(identity)
        require(type(group['lr']) is float and math.isfinite(group['lr']) and
                group['lr'] == expected_rates[name] and group['betas'] == (0.9, 0.999) and
                type(group['betas']) is tuple and all(type(v) is float for v in group['betas']) and
                type(group['eps']) is float and group['eps'] == 1e-15 and
                type(group['weight_decay']) in (int, float) and group['weight_decay'] == 0 and
                all(group[n] is False for n in ('amsgrad','maximize','capturable','differentiable')) and
                group['foreach'] is None and group['fused'] is None, 'optimizer_settings')
        d = desc[name]
        require(d['id'] == identity and
                d['initialized'] == (identity in opt['state']), 'optimizer_binding')
        if not d['initialized']:
            continue
        state = opt['state'][identity]; keys(state, 'step exp_avg exp_avg_sq')
        step = state['step']
        require(type(step) is torch.Tensor and step.device.type == 'cpu' and step.ndim == 0 and
                step.dtype in (torch.float32, torch.float64, torch.int64) and
                str(step.dtype) == d['step_dtype'] and torch.isfinite(step).item() and
                0 <= step.item() <= k and step.item() == math.trunc(step.item()), 'adam_step')
        for field in ('exp_avg', 'exp_avg_sq'):
            moment = state[field]
            require(type(moment) is torch.Tensor and moment.device.type == 'cpu' and
                    moment.dtype == torch.float32 and moment.shape == a[GROUPS[name][0]].shape and
                    moment.is_contiguous() and not moment.requires_grad and bool(torch.isfinite(moment).all()), 'moment')
        require(bool((state['exp_avg_sq'] >= 0).all()), 'second_moment')
    require(all(type(i) is int and i in ids for i in opt['state']), 'extra_optimizer_state')
    return payload


def snapshot(model, c, k):
    require(type(k) is int and 1 <= k <= c.optimization.total_updates and k in c.save_updates, 'save_label')
    model_contract(model, c)
    groups = model.optimizer.param_groups
    require(len(groups) == 9 and {g['name'] for g in groups} == set(GROUPS), 'live_groups')
    for g in groups:
        require(len(g['params']) == 1 and g['params'][0] is getattr(model, GROUPS[g['name']][1]) and
                g['params'][0].grad is None, 'live_parameter')
    captured = _clone(model.capture())
    payload = dict(schema=SCHEMA, layout=1, k=k, capture=captured)
    opt = captured[11]; desc = {}
    for group in opt['param_groups']:
        i, name = group['params'][0], group['name']
        desc[name] = dict(slot=GROUPS[name][0], id=i, initialized=i in opt['state'],
            step_dtype=str(opt['state'][i]['step'].dtype) if i in opt['state'] else None)
    meta = dict(k=k, rows=len(captured[1]), optimizer=desc)
    validate_state(payload, meta, c)
    return payload, desc


def load_state(stream, meta, c):
    # weights_only is not sufficient: TorchScript ZIPs dispatch to jit.load.
    stream.seek(0)
    try:
        with zipfile.ZipFile(stream) as archive:
            entries = archive.infolist(); names = [i.filename for i in entries]
            require(len(names) == len(set(names)) and 0 < len(names) <= 2000, 'zip_entries')
            prefixes = {n.split('/')[0] for n in names}
            require(len(prefixes) == 1, 'zip_root')
            for row in entries:
                suffix = row.filename.split('/',1)[-1]
                allowed = suffix in ('data.pkl','byteorder','version','.data/serialization_id') or (
                    suffix.startswith('data/') and suffix[5:].isdigit())
                require(allowed and row.compress_type == zipfile.ZIP_STORED and not row.flag_bits & 1, 'zip_format')
            require(sum(i.file_size for i in entries) <= 8*1024**3 and
                    any(n.endswith('/data.pkl') for n in names), 'zip_budget')
        stream.seek(0)
        payload = torch.load(stream, map_location='cpu', weights_only=True)
    except (zipfile.BadZipFile, EOFError):
        raise ValueError('checkpoint_payload_file') from None
    return validate_state(payload, meta, c)


def restore_diagnostic(result, device):
    """Existing constructor/restore(None), never optimizer or training setup."""
    c, payload, meta = result['config'], result['state'], result['metadata']
    validate_state(payload, meta, c)
    validate_native(NativeEvidence(canonical(meta['base']['native'])), live=True)
    from scene.gaussian_model import GaussianModel
    model = GaussianModel(3, gaussian_dim=4, time_duration=c.time_derivation.effective_interval,
        rot_4d=True, force_sh_3d=False, sh_degree_t=2, prefilter_var=-1.0)
    a = list(payload['capture'])
    for slot in SHAPES: a[slot] = a[slot].to(device).clone()
    model.restore(tuple(a), training_args=None)
    model.xyz_gradient_accum, model.t_gradient_accum, model.denom = a[8], a[9], a[10]
    model.percent_dense = c.optimization.population.percent_dense
    model._formal_time = c.time_derivation
    model_contract(model, c)
    require(model.optimizer is None, 'diagnostic_optimizer')
    for slot, attr in [(v[0], v[1]) for v in GROUPS.values()] + [
            (7,'max_radii2D'), (8,'xyz_gradient_accum'), (9,'t_gradient_accum'), (10,'denom')]:
        equal_state(getattr(model, attr).detach().cpu(), payload['capture'][slot])
        require(getattr(model, attr).data_ptr() != payload['capture'][slot].data_ptr(), 'diagnostic_storage')
    return model
