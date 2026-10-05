"""Independent Step 10 CPU mathematics and frozen synthetic inputs.

Column-vector pinhole/matrix calculus, not a production renderer. No camera,
Scene, renderer, extension or CUDA import. Values start as actual float32
inputs widened to float64; all oracle computation stays on the CPU.
"""
import itertools
import math
import struct

import torch
import sh_oracle as sh
from alpha_cap_oracle import composite, LOSSES

TOLERANCES = dict(analytic=(2e-11, 2e-10), cpu_fd=(3e-7, 2e-5),
                  screen=(4e-4, 2e-6), geometry=(3e-6, 5e-5),
                  pixel=(8e-6, 3e-5), gradient=(6e-5, 6e-4),
                  cuda_fd=(3e-3, 1.2e-2), atomic=(4e-6, 4e-5))
CPU_STEPS = (2**-17, 2**-18)
CUDA_STEPS = (2**-10, 2**-11)
EPS = struct.unpack('f', struct.pack('f', 1e-7))[0]


def f32(x):
    return struct.unpack('f', struct.pack('f', float(x)))[0]


def bits(x):
    return struct.unpack('I', struct.pack('f', float(x)))[0]


def tensor(x):
    return torch.tensor(x, dtype=torch.float64)


def pose(identity=False):
    if identity:
        return torch.eye(3, dtype=torch.float64), tensor([0., 0., 0.])
    a, b, c = (math.radians(v) for v in (10, -15, 20))
    rx = tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = tensor([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = tensor([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx, tensor([.4, -.3, .2])


def camera(label='smooth', *, width=80, height=40, fx=64., fy=64.,
           mode='intrinsics', divisor=1, identity=False):
    r, center = pose(identity)
    return dict(label=label, raw_width=width, raw_height=height,
                raw_fx=fx, raw_fy=fy, width=width//divisor, height=height//divisor,
                fx=fx/divisor, fy=fy/divisor, tanx=width/(2*fx), tany=height/(2*fy),
                divisor=divisor, mode=mode, rotation=r, center=center,
                timestamp=.5, duration=1., background=(.07, .11, .03))


def matrices(s):
    """Independent mathematical matrices, not copies of runtime tensors."""
    view = torch.eye(4, dtype=torch.float64)
    view[:3, :3] = s['rotation']
    view[:3, 3] = -s['rotation'] @ s['center']
    p = torch.zeros(4, 4, dtype=torch.float64)
    p[0, 0], p[1, 1] = 1/s['tanx'], 1/s['tany']
    p[2, 2], p[2, 3], p[3, 2] = 100/99.99, -1/99.99, 1
    return view, p, p @ view


def camera_cases():
    return [camera('K-'+mode+str(d), width=1280, height=720,
                   fx=1777.7777777777778, fy=1777.7777777777778,
                   mode='intrinsics' if mode == 'I' else 'fov', divisor=d)
            for mode in ('I', 'F') for d in (1, 2)] + [
        camera('A-I', width=80, height=48, fx=60., fy=72.),
        camera('A-P', width=80, height=48, fx=60., fy=72., mode='pair')]


def limits(s):
    # CUDA multiplies two float scalars; equality fixtures use these exact bits.
    return tensor([f32(f32(1.3)*f32(s['tanx'])), f32(f32(1.3)*f32(s['tany']))])


def footprint(t, sigma, s, *, selected=None):
    z = t[2]
    ratio = t[:2]/z
    lim = limits(s)
    inside = (ratio >= -lim) & (ratio <= lim)
    if selected is not None:
        inside = torch.tensor(selected, dtype=torch.bool)
    # where chooses the approved interior derivative AT equality, unlike
    # minimum/maximum's split derivative. Outside the selected side is constant.
    q = torch.where(inside, ratio, ratio.sign()*lim)
    f = tensor([s['fx'], s['fy']])
    a = torch.cat((torch.diag(f/z), (-f*q/z)[:, None]), dim=1)
    cov = a @ sigma @ a.T + .3*torch.eye(2, dtype=torch.float64)
    return dict(ratio=ratio, clamped=torch.cat((q*z, z.view(1))),
                inside=inside, jacobian=a, screen_cov=cov, inverse=torch.linalg.inv(cov))


def footprint_vjp(t, sigma, s, h, *, old=False, selected=None):
    """Matrix adjoints and chain rule independently derived from C=A S A^T."""
    g = footprint(t, sigma, s, selected=selected)
    a, z = g['jacobian'], t[2]
    da = (h+h.T) @ a @ sigma
    ds = a.T @ h @ a
    m = g['inside'].to(torch.float64)
    f = tensor([s['fx'], s['fy']])
    xy = -m*f/z.square()*da[:, 2]
    coefficients = torch.full_like(m, 2.) if old else 1+m
    gz = -(f*da.diagonal()[:2]).sum()/z.square()
    gz = gz+(coefficients*f*g['clamped'][:2]*da[:, 2]).sum()/z**3
    return torch.cat((xy, gz.view(1))), ds


def conditional(p, s, dim=4):
    if dim == 4:
        return sh.conditional(p, s)
    q = p['left']/p['left'].norm()
    qc = q*tensor([1., -1., -1., -1.])
    rot = torch.stack([sh.hamilton(sh.hamilton(q, torch.cat((q[:1]*0, e))), qc)[1:]
                       for e in torch.eye(3, dtype=torch.float64)], dim=1)
    cov = (rot*p['log_scale'][:3].exp().square()) @ rot.T
    return p['mean'], cov, tensor(1.), cov


def color(p, mean, s, dim=4):
    degree, temporal = (3, 2) if dim == 4 else (3, 0)
    y = sh.basis(mean-s['center'], degree)
    value = .5+y @ p['features'][:16]
    theta = 2*math.pi*(p['time'][0]-s['timestamp'])/s['duration']
    for k in range(1, temporal+1):
        value = value+torch.cos(k*theta)*(y @ p['features'][16*k:16*(k+1)])
    return value.clamp_min(0)


def geometry(p, s, pixel, dim=4, selected=None):
    if any(v.device.type != 'cpu' or v.dtype != torch.float64 for v in p.values()):
        raise ValueError('independent_oracle_requires_cpu_float64')
    mean, cov, marginal, sigma4 = conditional(p, s, dim)
    r = s['rotation']
    t = r @ (mean-s['center'])
    sigma = r @ cov @ r.T
    g = footprint(t, sigma, s, selected=selected)
    xy = tensor([s['fx'], s['fy']])*t[:2]/(t[2]+EPS)+tensor([(s['width']-1)/2, (s['height']-1)/2])
    delta = xy-tensor(pixel)
    power = -.5*delta @ g['inverse'] @ delta
    opacity = p['opacity'].sigmoid()[0]*marginal
    g.update(mean=mean, covariance=cov, sigma4=sigma4, marginal=marginal,
             view=t, view_cov=sigma, xy=xy, delta=delta, power=power,
             effective_opacity=opacity, raw=opacity*power.exp(),
             color=color(p, mean, s, dim), flow=tensor([0., 0.]), depth=t[2])
    return g


def evaluate(case, points=None, selected=None):
    points = case['points'] if points is None else points
    out = []
    for pixel in case['pixels']:
        geo = [geometry(p, case['camera'], pixel, case['dim'], selected) for p in points]
        stack = lambda k: torch.stack([g[k] for g in geo])
        c = composite(stack('raw'), stack('color'), stack('flow'), stack('depth'),
                      tensor(case['camera']['background']), case['loss'])
        c['geometry'] = geo
        s = case['camera']
        c['screen'] = torch.stack([torch.cat((
            -g['raw']*c['dr'][i]*(g['inverse'] @ g['delta'])*tensor([s['width']/2, s['height']/2]),
            c['depth_grad'][i].view(1))) for i, g in enumerate(geo)])
        out.append(c)
    return out


def gradients(case, *, method='analytic', selected=None):
    """Analytic geometry/compositor VJP into independent 4D + Legendre graph.

    'exact' is full float64 autograd. 'compat' changes ONLY inverse adjoints
    by rho; 'old' changes ONLY the two clamp z coefficients. Neither is used
    to construct the strict mathematical expected result.
    """
    points, s = case['points'], case['camera']
    results = evaluate(case, selected=selected)
    terms = []
    if method == 'exact':
        terms = [r['loss'] for r in results]
    else:
        for result in results:
            for i, g in enumerate(result['geometry']):
                power_upstream = (g['raw']*result['dr'][i]).detach()
                dq = -.5*power_upstream*torch.outer(g['delta'], g['delta'])
                q = g['inverse']
                h = -q.T @ dq @ q.T
                if method == 'compat':
                    d = torch.linalg.det(g['screen_cov'])
                    h = h*(d.square()/(d.square()+EPS))
                dt, ds = footprint_vjp(g['view'], g['view_cov'], s, h,
                                       old=method == 'old', selected=selected)
                ds_world = s['rotation'].T @ ds @ s['rotation']
                dt_world = s['rotation'].T @ dt
                terms += [g['mean'] @ dt_world.detach(), (g['covariance']*ds_world.detach()).sum(),
                          g['xy'] @ (-power_upstream*q @ g['delta']).detach(),
                          g['effective_opacity']*(g['power'].exp()*result['dr'][i]).detach(),
                          g['color'] @ result['color_grad'][i].detach(),
                          g['depth']*result['depth_grad'][i].detach()]
    flat = [v for p in points for v in p.values()]
    values = iter(torch.autograd.grad(sum(terms), flat, allow_unused=True))
    return [{k: torch.zeros_like(v) if (g := next(values)) is None else g
             for k, v in p.items()} for p in points]


def covariance_view_gradient(case):
    total = tensor([0., 0., 0.])
    for result in evaluate(case):
        g = result['geometry'][0]
        dq = -.5*(g['raw']*result['dr'][0])*torch.outer(g['delta'], g['delta'])
        h = -g['inverse'].T @ dq @ g['inverse'].T
        total = total+footprint_vjp(g['view'], g['view_cov'], case['camera'], h)[0]
    return total.detach()


def point(s, view, *, dim=4, boundary=False, scale=1.):
    p = sh.fixture()
    with torch.no_grad():
        p['log_scale'].copy_(tensor([math.log(scale*v) for v in (.7, .9, 1.1, 1.2)]))
        p['time'].fill_(s['timestamp'] if boundary else .375)
        p['opacity'].fill_(math.log(.6/.4))
        p['features'].mul_(.18)
        if boundary:
            p['left'].copy_(tensor([1., 0., 0., 0.])); p['right'].copy_(p['left'])
        # Set conditional mean using independent input mathematics, not runtime output.
        shift = conditional(p, s, dim)[0]-p['mean']
        p['mean'].copy_(s['rotation'].T @ tensor(view)+s['center']-shift)
    return {k: v.detach().float().double().requires_grad_() for k, v in p.items()}


def pixels_for(p, s, dim=4):
    xy = geometry(p, s, (0, 0), dim)['xy'].detach()
    # Nine/seven-pixel displacement gives nonzero footprint derivatives inside;
    # outside points select the closest supported edge plus inward neighbors.
    x = min(s['width']-2, max(1, round(float(xy[0]))+9))
    y = min(s['height']-2, max(1, round(float(xy[1]))+7))
    return ((x, y), (x-1, y), (x, y-1))


def make_case(label, s, p, *, loss='mask', dim=4, kind='smooth', pixels=None):
    return dict(label=label, camera=s, points=[p], loss=LOSSES[loss], dim=dim, kind=kind,
                pixels=pixels or pixels_for(p, s, dim))


def cases():
    result = []
    s = camera()
    for i, (x, y) in enumerate(itertools.product((-1.10, .25, 1.10), repeat=2)):
        view = [3*x*float(limits(s)[0]), 3*y*float(limits(s)[1]), 3.]
        # Fixed CPU-selected scales put the affected radii near half-integers.
        # All original FD widths/coordinates, including S7's raw 4D parameters,
        # are checked below by the CPU suite; no search or CUDA fitting occurs.
        scale = {0: .993443, 1: .993615, 6: .994680, 7: 1.005995}.get(i, 1.)
        p = point(s, view, scale=scale)
        # S3's old x/y covariance terms almost cancelled at the former pixels.
        # Sample near projected x, retaining the off-screen y and nonzero signal.
        pixels = ((53, 1), (52, 1), (53, 0)) if i == 3 else None
        c = make_case('S'+str(i), s, p, pixels=pixels)
        result.append(c)
        if i in (4, 7, 8):
            for loss in ('depth', 'rgb', 'mixed'):
                result.append(dict(c, label=c['label']+'/'+loss, loss=LOSSES[loss], kind='mixed'))
    representative = next(c for c in result if c['label'] == 'S7/mixed')
    result.append(dict(representative, label='D/single', kind='diagnostic',
                       pixels=representative['pixels'][:1]))
    for s in camera_cases():
        result.append(make_case(s['label'], s, point(s, [.24, .17, 3.]), loss='mixed', kind='camera'))
    s = camera('equality', identity=True)
    for axis, sign in itertools.product(range(2), (-1, 1)):
        equal = torch.tensor(float(limits(s)[axis])*sign, dtype=torch.float32)
        for side in ('inside', 'equal', 'outside'):
            value = equal if side == 'equal' else torch.nextafter(
                equal, equal.new_tensor(0. if side == 'inside' else sign*math.inf))
            view = [.125, .09375, 1.]; view[axis] = float(value)
            c = make_case(f'E/{axis}/{sign}/{side}', s, point(s, view, boundary=True), kind='equality')
            c.update(axis=axis, sign=sign, side=side, boundary_bits=bits(value))
            result.append(c)
    near = torch.tensor(.2, dtype=torch.float32)
    depths = [.01, .1, float(torch.nextafter(near, near.new_tensor(0))), float(near),
              float(torch.nextafter(near, near.new_tensor(1))), 1., 99., 100., 101.]
    for i, z in enumerate(depths):
        s = camera('depth', identity=True)
        p = point(s, [.0625*z, .046875*z, z], boundary=True, scale=z/3)
        result.append(make_case('Z'+str(i), s, p, kind='cull', loss='mixed'))
    s = camera()
    result.append(make_case('3D', s, point(s, [3*1.1*float(limits(s)[0]), .25, 3.], dim=3), dim=3))
    s = camera('addition')
    p = point(s, [.5, .3, 3.]); q = point(s, [.65, .4, 4.], scale=1.15)
    pixels = pixels_for(p, s)[:2]
    for label, px in (('p0', pixels[:1]), ('p1', pixels[1:]), ('sum', pixels)):
        c = make_case('M/'+label, s, p, kind='addition', pixels=px, loss='mixed')
        c['points'] = [p, q]; result.append(c)
    # Dedicated exact forward selection pairs. Do not apply smooth FD to them.
    s = camera('discrete', identity=True)
    z = 3.
    for side, delta in (('below', -.002), ('above', .002)):
        # Start on the negative OUTER clamp branch. Its covariance/radius no
        # longer depends on x, so solve the tile crossing using that footprint.
        p = point(s, [-4., .15, z], boundary=True)
        g = geometry(p, s, (40, 20)); radius = topology(g, s)['radius']
        xy = 3*16.-15.-radius+delta  # upper tile cast crosses 3 after moving x.
        with torch.no_grad(): p['mean'][0] = f32((xy-(s['width']-1)/2)*(z+EPS)/s['fx'])
        moved = geometry(p, s, (40, 20))
        assert not bool(moved['inside'][0]) and topology(moved, s)['radius'] == radius, 'tile_construction_branch'
        result.append(make_case('tile/'+side, s, p, kind='discrete'))
        # C00 at X=0 has scale-dependent variance. Solve the larger eigenvalue
        # threshold using a view-axis diagonal, equal x/y footprint; ceil at 30.
        p = point(s, [0., 0., z], boundary=True)
        target = (30.+delta)**2/9
        variance = target-math.sqrt(.1)-.3
        with torch.no_grad():
            p['log_scale'][:2] = math.log(math.sqrt(variance)*z/s['fx'])
            p['mean'][0] = f32(.003)
            p['mean'][1] = f32(.002)
        result.append(make_case('radius/'+side, s, p, kind='discrete'))
    return result


def topology(g, s):
    c, xy = g['screen_cov'].detach(), g['xy'].detach()
    mid = float(c.trace()/2)
    radius_float = 3*math.sqrt(mid+math.sqrt(max(.1, mid*mid-float(torch.linalg.det(c)))))
    radius = math.ceil(radius_float)
    grid = ((s['width']+15)//16, (s['height']+15)//16)
    lo = [max(0, min(grid[i], int((float(xy[i])-radius)/16))) for i in range(2)]
    hi = [max(0, min(grid[i], int((float(xy[i])+radius+15)/16))) for i in range(2)]
    return dict(radius=radius, radius_float=radius_float, tiles=lo+hi,
                clamp=g['inside'].tolist(), visible=float(g['view'][2]) > f32(.2))


def domain(case, points=None, *, signal=False):
    if points is not None:
        case = dict(case, points=points)  # Signal/P2 checks must use FD endpoints too.
    observations = []
    for result in evaluate(case):
        nodes = []
        for g in result['geometry']:
            t = topology(g, case['camera'])
            c = g['screen_cov'].detach()
            assert all(torch.isfinite(v).all() for v in g.values()), 'nonfinite_fixture'
            if case['kind'] != 'cull' or t['visible']:
                assert t['radius'] > 0 and t['tiles'][0] < t['tiles'][2] and t['tiles'][1] < t['tiles'][3], 'empty_tiles'
                assert .02 <= float(g['raw']) <= .8 and float(g['power']) < 0, 'raw_alpha_domain'
                assert float(g['marginal']) > .5 and bool((g['color'] > .1).all()), 'marginal_color_domain'
                assert float(torch.linalg.det(c)) >= 1 and float(torch.linalg.cond(c)) <= 100, 'SPD_domain'
            if case['kind'] in ('smooth', 'mixed', 'camera', 'addition', 'diagnostic'):
                assert float(g['view'][2]) > .5, 'view_depth_domain'
                assert abs(t['radius_float']-round(t['radius_float'])) > .01, 'radius_margin'
            nodes.append(dict(t, raw_alpha=float(g['raw']), determinant=float(torch.linalg.det(c)),
                              condition=float(torch.linalg.cond(c)), marginal=float(g['marginal'])))
        assert float(result['trans'][-1]) > .01, 'transmittance_domain'
        observations.append(nodes)
    if signal:
        exact = gradients(case)
        covz = float(abs(covariance_view_gradient(case)[2]))
        atol, rtol = TOLERANCES['gradient']
        covariance_ratio = covz/(atol+rtol*covz)
        assert covariance_ratio > 10, 'covariance_gradient_signal'
        norm = float(exact[0]['mean'].norm())
        total_ratio = norm/(atol+rtol*norm)
        assert total_ratio > 10, 'total_gradient_signal'
        old_ratio = None
        if not all(observations[0][0]['clamp']):
            old = gradients(case, method='old')[0]['mean']
            gap = (old-exact[0]['mean']).abs()
            old_ratio = float((gap/(atol+rtol*exact[0]['mean'].abs())).max())
            assert old_ratio > 20, 'old_formula_gap'
        compat = gradients(case, method='compat')
        p2_ratio = 0.
        for p, q in zip(exact, compat):
            for k in p:
                ratio = float(((p[k]-q[k]).abs()/(atol+rtol*p[k].abs())).max())
                p2_ratio = max(p2_ratio, ratio)
                assert ratio <= .1, 'P2_budget'
        observations[0][0]['signal'] = dict(covariance_bound_ratio=covariance_ratio,
            total_bound_ratio=total_ratio, old_gap_bound_ratio=old_ratio, p2_bound_ratio=p2_ratio)
    return observations


def topology_key(domain_result):
    return [[(n['radius'], n['tiles'], n['clamp'], n['visible']) for n in nodes]
            for nodes in domain_result]


def fd_coordinates(case):
    if case['kind'] != 'smooth':
        return []
    result = [(0, 'mean', i) for i in range(3)]
    if case['label'] == 'S7':
        result += [(0, 'time', 0)] + [(0, 'log_scale', i) for i in range(4)]
        result += [(0, k, i) for k in ('left', 'right') for i in range(4)]
    return result


def perturb(points, coordinate, step, rounded=False):
    i, name, offset = coordinate
    value = float(points[i][name].detach().flatten()[offset])
    h = step*max(1., abs(value))
    ends = []
    for sign in (1, -1):
        p = [sh.clone(x) for x in points]
        with torch.no_grad(): p[i][name].flatten()[offset] = f32(value+sign*h) if rounded else value+sign*h
        ends.append(p)
    den = float(ends[0][i][name].flatten()[offset]-ends[1][i][name].flatten()[offset])
    assert den > 0
    return *ends, den


def loss(case, points=None):
    return sum(r['loss'] for r in evaluate(case, points))
