"""Step 9 independent CPU prefix/suffix reference; not a renderer fallback.

Only fixture inputs and independent SH/Hamilton mathematics are shared. No
production renderer, Scene, extension, camera math or diagnostic output is used.
Equality's chosen derivative is explicit, never torch.minimum's convention.
"""
import math
import struct

import torch
import sh_oracle as sh

CAP = struct.unpack('f', struct.pack('f', .99))[0]
C0 = 1 / math.sqrt(4 * math.pi)
SPEC = dict(width=9, height=9, fx=12., fy=12., timestamp=.5, duration=1.,
            pixel=(4, 4), background=(.125, .25, .0625))
LOSSES = dict(rgb=(1., 0., 0., 0., 0., 0., 0.),
              flow=(0., 0., 0., 1., 0., 0., 0.),
              depth=(0., 0., 0., 0., 0., 1., 0.),
              mask=(0., 0., 0., 0., 0., 0., 1.),
              mixed=(.25, -.125, .375, .25, -.5, .125, .75),
              background=(1., -1., 0., 0., 0., 0., 0.))
TOLERANCES = dict(anchor=(1e-12, 1e-10), cpu_fd=(1e-8, 2e-6),
                  forward=(2e-6, 8e-6), direct=(5e-6, 5e-5),
                  formal=(2e-5, 2e-4), cuda_fd=(2e-3, 5e-3))
CPU_STEPS = (1e-6, 5e-7)
CUDA_STEPS = (2**-10, 2**-11)


def neighbors():
    c = torch.tensor(CAP, dtype=torch.float32)
    return tuple(float(v) for v in (torch.nextafter(c, c.new_tensor(-math.inf)),
                                   c, torch.nextafter(c, c.new_tensor(math.inf))))


def fixture(opacity=.997, *, centered=False, formal=False, depth=3.):
    values = dict(mean=[0., 0., depth] if centered else
                  [.03125 * depth/3, .015625 * depth/3, depth],
                  time=[.46875 if formal else .5],
                  log_scale=[math.log(v) for v in
                             (.5*depth/3, .625*depth/3, .75*depth/3, 1.)],
                  left=[math.cos(.1), 0., 0., math.sin(.1)] if formal else [1., 0., 0., 0.],
                  right=[1., 0., 0., 0.],
                  opacity=[math.log(opacity/(1-opacity)) if formal else opacity],
                  flow=[0., 0.] if formal else [.25, -.125],
                  features=[[0., 0., 0.] for _ in range(48)])
    if formal:
        for i, v in ((1, [.08, -.035, .025]), (16, [.03, .02, .01]), (32, [.017, .009, .012])):
            values['features'][i] = v
    # CUDA inputs are float32; independent math widens exactly those inputs.
    return {k: torch.tensor(v, dtype=torch.float32).double().requires_grad_()
            for k, v in values.items()}


def clone(points, *, rounded=False):
    return [{k: (v.detach().float().double() if rounded else v.detach()).clone().requires_grad_()
             for k, v in p.items()} for p in points]


def cases():
    result = []
    def add(label, points, loss='mixed', dim=4, formal=False, pixels=((4, 4),), boundary=False):
        result.append(dict(label=label, points=points, loss=LOSSES[loss], dim=dim,
                           formal=formal, pixels=pixels, boundary=boundary))
    for dim in (3, 4):
        for name, value in zip(('below', 'equal', 'above'), neighbors()):
            add(f'B/{dim}/{name}', [fixture(value, centered=True)], 'mask', dim, boundary=True)
        for opacity in (.6, .997):
            for loss in ('rgb', 'flow', 'depth', 'mask', 'mixed'):
                add(f'S/{dim}/{opacity}/{loss}', [fixture(opacity)], loss, dim)
    add('BG', [fixture()], 'background')
    for name, opacities in (('front', (.997, .4)), ('back', (.4, .997))):
        add('M/'+name, [fixture(opacities[0]), fixture(opacities[1], depth=4.)])
    for name, pixels in (('p0', ((4, 4),)), ('p1', ((5, 4),)), ('sum', ((4, 4), (5, 4)))):
        add('P/'+name, [fixture()], 'mask', pixels=pixels)
    for value in (.6, .997):
        add(f'F/{value}', [fixture(value, formal=True)], formal=True)
    return result


def geometry(p, pixel, *, dim=4, formal=False):
    if any(v.device.type != 'cpu' or v.dtype != torch.float64 for v in p.values()):
        raise ValueError('oracle_requires_cpu_binary64')
    if dim == 4:
        mean, cov, marginal, sigma = sh.conditional(p, SPEC)
    elif dim == 3:
        q = p['left']/p['left'].norm()
        qc = q*q.new_tensor([1, -1, -1, -1])
        basis = torch.eye(3, dtype=q.dtype)
        rot = torch.stack([sh.hamilton(sh.hamilton(q, torch.cat((q[:1]*0, e))), qc)[1:]
                           for e in basis], dim=1)
        cov = (rot*p['log_scale'][:3].exp().square()) @ rot.T
        mean, marginal, sigma = p['mean'], q.new_tensor(1.), cov
    else:
        raise ValueError('fixture_dimension')
    x, y, z = mean.unbind()
    zero = z*0
    jac = torch.stack((torch.stack((12/z, zero, -12*x/z**2)),
                       torch.stack((zero, 12/z, -12*y/z**2))))
    screen_cov = jac @ cov @ jac.T + torch.eye(2, dtype=z.dtype)*.3
    xy = torch.stack((12*x/(z+1e-7)+4, 12*y/(z+1e-7)+4))
    delta = xy - xy.new_tensor(pixel)
    inverse = torch.linalg.inv(screen_cov)
    power = -.5 * delta @ inverse @ delta
    opacity = p['opacity'][0].sigmoid() if formal else p['opacity'][0]
    effective_opacity = opacity * marginal
    raw = effective_opacity * power.exp()
    # DC-only axis cases must not evaluate singular atan2/Legendre derivatives.
    color = sh.color_at(p, mean, (3, 2), SPEC) if formal else .5+C0*p['features'][0]
    return dict(mean=mean, covariance=cov, sigma=sigma, marginal=marginal,
                screen_cov=screen_cov, inverse=inverse, xy=xy, delta=delta,
                power=power, effective_opacity=effective_opacity, raw=raw,
                color=color, flow=p['flow']*0 if formal else p['flow'], depth=z)


def composite(raw, colors, flows, depths, background, loss):
    """Independent forward and closed derivative using prefix/suffix, no reverse recurrence."""
    a = torch.where(raw < CAP, raw, raw.new_full(raw.shape, CAP))
    trans = torch.cat((raw.new_ones(1), (1-a).cumprod(0)))
    weights = trans[:-1]*a
    rgb = weights @ colors + trans[-1]*background
    flow = weights @ flows
    depth, mask = weights @ depths, 1-trans[-1]
    outputs = torch.cat((rgb, flow, depth.view(1), mask.view(1)))
    u = raw.new_tensor(loss)
    v = colors @ u[:3] + flows @ u[3:5] + depths*u[5] + u[6]
    bg = background @ u[:3]
    suffix = torch.stack([(weights[i+1:]*v[i+1:]).sum()+trans[-1]*bg
                          for i in range(raw.numel())])
    da = trans[:-1]*v-suffix/(1-a)
    dr = torch.where(raw < CAP, da, torch.zeros_like(da))
    return dict(outputs=outputs, loss=outputs @ u, alpha=a, trans=trans, weights=weights,
                da=da, dr=dr, color_grad=weights[:, None]*u[:3],
                flow_grad=weights[:, None]*u[3:5], depth_grad=weights*u[5],
                background_da=-trans[-1]*bg/(1-a))


def evaluate(points, pixel=(4, 4), *, dim=4, formal=False, loss=LOSSES['mixed']):
    geo = [geometry(p, pixel, dim=dim, formal=formal) for p in points]
    stack = lambda name: torch.stack([g[name] for g in geo])
    core = composite(stack('raw'), stack('color'), stack('flow'), stack('depth'),
                     points[0]['mean'].new_tensor(SPEC['background']), loss)
    core['geometry'] = geo
    core['screen'] = torch.stack([torch.cat((
        g['raw']*core['dr'][i]*(-g['inverse'] @ g['delta'])*4.5,
        core['depth_grad'][i].view(1))) for i, g in enumerate(geo)])
    core['opacity_grad'] = torch.stack([g['marginal']*g['power'].exp()*core['dr'][i]
                                       for i, g in enumerate(geo)])
    return core


def evaluate_case(case, points=None):
    return [evaluate(case['points'] if points is None else points, pixel,
                     dim=case['dim'], formal=case['formal'], loss=case['loss'])
            for pixel in case['pixels']]


def parameter_gradients(case):
    """Analytic compositor VJP into independent geometry/SH CPU Jacobians."""
    points = case['points']
    flat = [v for p in points for v in p.values()]
    terms = []
    for result in evaluate_case(case):
        for i, g in enumerate(result['geometry']):
            terms.append(g['raw']*result['dr'][i].detach()
                         + g['color'] @ result['color_grad'][i].detach()
                         + g['flow'] @ result['flow_grad'][i].detach()
                         + g['depth']*result['depth_grad'][i].detach())
    grads = iter(torch.autograd.grad(sum(terms), flat, allow_unused=True))
    result = []
    for p in points:
        output = {}
        for k, v in p.items():
            g = next(grads)
            output[k] = torch.zeros_like(v) if g is None else g
        result.append(output)
    return result


def fd_coordinates(case):
    if case['label'].startswith('S/'):
        return [(0, 'opacity', 0)]+[(0, 'mean', i) for i in range(3)]
    if case['label'].startswith('M/'):
        return [(i, 'opacity', 0) for i in range(2)]
    if case['formal']:
        return ([(0, 'opacity', 0)]+[(0, 'mean', i) for i in range(3)]
                +[(0, 'time', 0), (0, 'log_scale', 0), (0, 'log_scale', 3),
                  (0, 'left', 3), (0, 'right', 3)]
                +[(0, 'features', 3*i) for i in (1, 16, 32)])
    return []


def perturb(points, coordinate, relative_step, *, rounded=False):
    i, name, offset = coordinate
    v = points[i][name].detach().flatten()[offset].item()
    h = relative_step*max(1., abs(v))
    plus, minus = clone(points), clone(points)
    with torch.no_grad():
        plus[i][name].view(-1)[offset] += h
        minus[i][name].view(-1)[offset] -= h
    if rounded:
        plus, minus = clone(plus, rounded=True), clone(minus, rounded=True)
    denominator = (plus[i][name].flatten()[offset]-minus[i][name].flatten()[offset]).item()
    if not denominator > 0: raise AssertionError('FD endpoints collapsed')
    return plus, minus, denominator


def domain(case, points=None):
    """Design 5.4 qualification only: rejects, never edits fixtures/tolerances."""
    records = []
    for pixel, result in zip(case['pixels'], evaluate_case(case, points)):
        if not float(result['trans'][-1]) > .003: raise AssertionError('final_T domain')
        nodes = []
        for g in result['geometry']:
            if not all(torch.isfinite(v).all() for v in g.values()): raise AssertionError('nonfinite fixture')
            if not torch.linalg.eigvalsh(g['sigma']).min() > 0: raise AssertionError('positive covariance')
            det = float(torch.linalg.det(g['screen_cov']))
            if not det > 1: raise AssertionError('screen determinant')
            x, y, z = g['mean'].detach().tolist()
            if not (z > 2 and abs(x/z) < 9/24 and abs(y/z) < 9/24): raise AssertionError('viewport domain')
            if not (g['color'] > .2).all(): raise AssertionError('SH clamp domain')
            raw = float(g['raw'])
            if not (min(raw, CAP) > .01 and float(g['power']) <= 0 and g['marginal'] > .05):
                raise AssertionError('contribution domain')
            if not case['boundary'] and not (raw <= .985 or raw >= .992):
                raise AssertionError(f'cap_margin: {case["label"]} pixel={pixel} raw={raw}')
            cov = g['screen_cov'].detach()
            mid = float(cov.trace())/2
            radius = math.ceil(3*math.sqrt(mid+math.sqrt(max(.1, mid*mid-det))))
            xy = g['xy'].detach().tolist()
            tiles = tuple(max(0, min(1, int((xy[i]+s*radius+(15 if s > 0 else 0))/16)))
                          for i in (0, 1) for s in (-1, 1))
            if tiles != (0, 1, 0, 1): raise AssertionError('single tile required')
            if max(abs(xy[i]-pixel[i]) for i in (0, 1)) >= radius: raise AssertionError('pixel outside support')
            nodes.append(dict(raw=raw, capped=raw >= CAP, radius=radius, tiles=tiles,
                              z=z, determinant=det, power=float(g['power'])))
        order = tuple(sorted(range(len(nodes)), key=lambda i: nodes[i]['z']))
        if order != tuple(range(len(nodes))) or len({n['z'] for n in nodes}) != len(nodes):
            raise AssertionError('strict contributor order')
        records.append(dict(pixel=pixel, final_T=float(result['trans'][-1]), nodes=nodes,
                            topology=[(n['radius'], n['tiles'], n['capped']) for n in nodes]))
    return records
