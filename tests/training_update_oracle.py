"""Step 12 independent CPU/binary64 equations, not a second trainer.

Observed VJPs and normal samples are inputs. No production optimizer,
population, rotation or rasterizer function supplies an expected result.
"""
import math
import struct

import numpy as np

GROUPS = dict(xyz='_xyz', f_dc='_features_dc', f_rest='_features_rest',
              opacity='_opacity', scaling='_scaling', rotation='_rotation',
              t='_t', scaling_t='_scaling_t', rotation_r='_rotation_r')
STATS = ('max_radii2D', 'xyz_gradient_accum', 't_gradient_accum', 'denom')
TOLERANCES = dict(parameter=(1e-6, 1e-5), moment=(1e-10, 1e-5),
                  variance=(1e-14, 1e-5), statistic=(1e-9, 1e-5),
                  opacity=(1e-8, 1e-5))


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def compare(actual, expected, label, records=None, tolerance='parameter', exact=False):
    a, b = array(actual), array(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label+': shape/nonfinite')
    error = np.abs(a.astype(np.float64)-b.astype(np.float64))
    if exact:
        ratio = float(not np.array_equal(a, b))
    else:
        atol, rtol = TOLERANCES[tolerance]
        ratio = float(np.max(error/(atol+rtol*np.abs(b)), initial=0))
    if records is not None:
        records.append(dict(check=label, max_abs=float(np.max(error, initial=0)),
                            max_bound_ratio=ratio, exact=exact))
    require(ratio == 0 if exact else ratio <= 1, label+': mismatch')


def adam(p, m, v, step, gradient, lr):
    """None does not even create state; a zero tensor advances/decays state."""
    if gradient is None:
        return p.copy(), m.copy(), v.copy(), step
    g = array(gradient).astype(np.float64)
    require(g.shape == p.shape and np.isfinite(g).all(), 'adam_gradient')
    m = .9*m+.1*g
    v = .999*v+.001*g*g
    step += 1
    p = p-lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-15)
    return p, m, v, step


def binary32(value):
    """Independent round-to-binary32, returned exactly widened to Python float."""
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def lr_endpoints(config, extent, *, ideal=False):
    """Config/geometry inputs only; never read a production closure or LR."""
    require(type(extent) in (float, np.float32, np.float64), 'LR_extent_type')
    require(math.isfinite(extent) and extent > 0, 'LR_extent_domain')
    c = config.optimization.learning_rate
    require(type(c.position_lr_max_steps) is int and c.position_lr_max_steps > 0, 'LR_max_steps')
    for name, value in c._asdict().items():
        if name != 'position_lr_max_steps':
            require(type(value) is float and math.isfinite(value) and value >= 0, 'LR_config/'+name)
    require((c.position_lr_init == 0) == (c.position_lr_final == 0), 'LR_endpoint_pair')
    def product(value):
        if type(extent) is np.float32 and not ideal:
            # NumPy 2 weak Python scalar promotion, then binary32 multiply.
            return binary32(binary32(value)*binary32(extent))
        return value*float(extent)
    values = tuple(product(v) for v in (c.position_lr_init, c.position_lr_final, c.position_t_lr_init))
    require(all(math.isfinite(v) for v in values), 'LR_product_finite')
    require(c.position_lr_init == 0 or min(values[:2]) > 0, 'LR_product_underflow')
    return values


def learning_rates(config, extent, k, *, ideal=False):
    """Explicit existing rounding model; ideal=True is diagnostic binary64."""
    require(type(k) is int and k >= 0, 'LR_k')
    c = config.optimization.learning_rate
    a, b, temporal = lr_endpoints(config, extent, ideal=ideal)
    u = min(1., max(0., k/c.position_lr_max_steps))
    if a == b == 0:
        xyz = 0.
    else:
        left, right = math.log(a), math.log(b)
        if type(extent) is np.float32 and not ideal:
            left, right = binary32(left), binary32(right)
        xyz = math.exp((1-u)*left+u*right)
    return dict(xyz=xyz, f_dc=c.feature_lr, f_rest=c.feature_lr/20,
                opacity=c.opacity_lr, scaling=c.scaling_lr, rotation=c.rotation_lr,
                t=temporal, scaling_t=c.scaling_lr, rotation_r=c.rotation_lr)


def optimization_loss(config, components):
    """Independent scalar assembly, matching separate float32 eager operations.

    Real L1/DSSIM/mask components are inputs, not the observed total loss.
    This is the bounded fixture model, not a new production dtype policy.
    """
    c = config.optimization.loss
    require(c.lambda_rigid == c.lambda_motion == 0, 'loss_fixture_branches')
    require(set(components) == {'l1', 'dssim', 'mask'}
            and all(math.isfinite(v) for v in components.values()), 'loss_components')
    r = binary32
    color = r(r(r(1-c.lambda_dssim)*components['l1'])
              + r(r(c.lambda_dssim)*components['dssim']))
    total = r(color+r(r(c.lambda_opa_mask)*components['mask']))
    return total, r(total/config.optimization.batch_size)


def batch_statistics(views, time_gradient):
    b = len(views)
    visible = np.stack([v['visible'] for v in views])
    h = visible.sum(axis=0)
    union = h > 0
    radius = np.max(np.stack([v['radii'] for v in views]), axis=0)
    screen = sum(np.linalg.norm(v['screen'][:, :2].astype(np.float64), axis=1) for v in views)
    t = array(time_gradient).astype(np.float64).reshape(-1)
    screen[union] *= b/h[union]
    t[union] *= b/h[union]
    return h, union, radius, screen[:, None], t[:, None]


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def rotate(sample, left, right):
    left, right = left/np.linalg.norm(left), right/np.linalg.norm(right)
    return multiply(multiply(left, sample[::-1]), right*np.array([1, -1, -1, -1]))[::-1]


class Trajectory:
    def __init__(self, snapshot):
        self.groups = {}
        for name in GROUPS:
            p = array(snapshot['groups'][name]['p']).astype(np.float64)
            require(snapshot['groups'][name]['step'] == 0, 'initial_state_not_empty')
            self.groups[name] = dict(p=p, m=np.zeros_like(p), v=np.zeros_like(p), step=0)
        self.stats = {n: array(snapshot['stats'][n]).astype(np.float64) for n in STATS}
        self.lineage = [('initial', i) for i in range(len(self.groups['xyz']['p']))]
        self.records = []

    def check(self, snapshot, label):
        for name, expected in self.groups.items():
            actual = snapshot['groups'][name]
            for field, tol in (('p', 'parameter'), ('m', 'moment'), ('v', 'variance')):
                compare(actual[field], expected[field], f'{label}/{name}/{field}', self.records, tol)
                if field != 'p':
                    zero = expected[field] == 0
                    require(np.all(actual[field][zero] == 0), f'{label}/{name}/{field}: zero')
            require(actual['step'] == expected['step'], f'{label}/{name}/step')
        for name in STATS:
            compare(snapshot['stats'][name], self.stats[name], label+'/'+name, self.records,
                    'statistic', exact=name in ('max_radii2D', 'denom'))

    def update(self, gradients, rates):
        for name, s in self.groups.items():
            s['p'], s['m'], s['v'], s['step'] = adam(s['p'], s['m'], s['v'], s['step'], gradients[name], rates[name])

    def add_statistics(self, views, time_gradient, update_radius, can_grow):
        h, visible, radius, screen, t = batch_statistics(views, time_gradient)
        if update_radius:
            self.stats['max_radii2D'][visible] = np.maximum(self.stats['max_radii2D'][visible], radius[visible])
            if can_grow:
                self.stats['xyz_gradient_accum'][visible] += screen[visible]
                self.stats['t_gradient_accum'][visible] += t[visible]
                self.stats['denom'][visible] += 1
        return h

    def gradient(self):
        a, d = self.stats['xyz_gradient_accum'], self.stats['denom']
        return np.divide(a, d, out=np.zeros_like(a), where=d != 0)

    def selection(self, gradients, threshold, boundary, split=False):
        count = len(self.lineage)
        g = np.zeros(count)
        g[:len(gradients)] = np.asarray(gradients).reshape(-1)
        scale = np.exp(self.groups['scaling']['p']).max(axis=1)
        return np.flatnonzero((g >= threshold) & (scale > boundary if split else scale <= boundary))

    def append(self, rows, values, kind, radii):
        for name, s in self.groups.items():
            child = values[name]
            s['p'] = np.concatenate([s['p'], child])
            s['m'] = np.concatenate([s['m'], np.zeros_like(child)])
            s['v'] = np.concatenate([s['v'], np.zeros_like(child)])
        lineage = [self.lineage[int(i)] for i in rows]
        self.lineage += [(kind, parent, j) for j, parent in enumerate(lineage)]
        for n in STATS:
            old = self.stats[n]
            new = radii if n == 'max_radii2D' else np.zeros((len(rows), 1))
            self.stats[n] = np.concatenate([old, new])

    def clone(self, rows):
        values = {n: s['p'][rows].copy() for n, s in self.groups.items()}
        self.append(rows, values, 'clone', self.stats['max_radii2D'][rows].copy())

    def split(self, rows, samples):
        rows = np.tile(rows, 2)
        values = {n: s['p'][rows].copy() for n, s in self.groups.items()}
        samples = array(samples).astype(np.float64)
        require(samples.shape == (len(rows), 4), 'normal_sample_shape')
        for j in range(len(rows)):
            delta = rotate(samples[j], values['rotation'][j], values['rotation_r'][j])
            values['xyz'][j] += delta[:3]
            values['t'][j] += delta[3:]
        for n in ('scaling', 'scaling_t'):
            values[n] -= math.log(1.6)
        self.append(rows, values, 'split', np.zeros(len(rows)))

    def prune_mask(self, opacity, extent, screen):
        mask = (1/(1+np.exp(-self.groups['opacity']['p']))).ravel() < opacity
        if screen is not None:
            mask |= self.stats['max_radii2D'] > screen
            mask |= np.exp(self.groups['scaling']['p']).max(axis=1) > .1*extent
        return mask

    def prune(self, mask):
        keep = ~mask
        for s in self.groups.values():
            for n in ('p', 'm', 'v'):
                s[n] = s[n][keep].copy()
        self.stats = {n: v[keep].copy() for n, v in self.stats.items()}
        self.lineage = [v for v, retained in zip(self.lineage, keep) if retained]

    def reset(self):
        s = self.groups['opacity']
        opacity = np.minimum(1/(1+np.exp(-s['p'])), .01)
        s['p'] = np.log(opacity/(1-opacity))
        s['m'].fill(0)
        s['v'].fill(0)

    def end_window(self):
        for value in self.stats.values():
            value.fill(0)


def delta_check(before, after, gradients, rates, records):
    """Local independent formula supplements (never resets) the trajectory."""
    representative = {}
    for name, s in before['groups'].items():
        p, _, _, _ = adam(s['p'].astype(np.float64), s['m'].astype(np.float64),
                          s['v'].astype(np.float64), s['step'], gradients[name], rates[name])
        delta = after['groups'][name]['p'].astype(np.float64)-s['p']
        expected = p-s['p']
        ulp = np.abs(np.spacing(s['p'].astype(np.float32))).astype(np.float64)
        bound = 2*ulp+1e-5*np.abs(expected)
        ratio = np.divide(np.abs(delta-expected), bound, out=np.zeros_like(bound), where=bound != 0)
        worst = float(np.max(ratio, initial=0))
        records.append(dict(check=name+'/local_delta', max_bound_ratio=worst,
                            max_abs=float(np.max(np.abs(delta-expected), initial=0))))
        require(worst <= 1, name+': delta_mismatch')
        representative[name] = (np.abs(delta) > 10*ulp).reshape(len(delta), -1).any(axis=1)
    return representative


def exact_carry(before, after, rows, child_start=None, computed=()):
    """Copy/slice is exact; separately calculated geometry uses numeric checks."""
    for name, a in after['groups'].items():
        b = before['groups'][name]
        if name not in computed:
            compare(a['p'], b['p'][rows], name+'/copy', exact=True)
        for key in ('m', 'v'):
            expected = b[key][rows].copy()
            if child_start is not None:
                expected[child_start:] = 0
            compare(a[key], expected, name+'/'+key+'/carry', exact=True)
        require(a['step'] == b['step'], name+'/step_carry')
