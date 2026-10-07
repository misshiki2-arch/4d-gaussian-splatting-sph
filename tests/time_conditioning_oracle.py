"""Independent Step 11 CPU mathematics; no production or GPU imports.

Hamilton covariance and pinhole/compositor reuse independent Step 8/10 code.
The conditional VJP below is analytic, not a copy of CUDA/GLM packed adjoints.
"""
import math
import struct
import torch
import sh_oracle as sh
import camera_projection_oracle as camera

TOLERANCES = dict(camera.TOLERANCES, scale=(1e-7, 2e-6), variance=(2e-7, 5e-6), marginal=(1e-7, 3e-6))
PARAMETERS = ('mean', 'time', 'log_scale', 'left', 'right', 'opacity')
CPU_STEPS, CUDA_STEPS = camera.CPU_STEPS, camera.CUDA_STEPS


def conditioning_loss(p, spec):
    mean, cov, marginal, _ = sh.conditional(p, spec)
    g = mean.new_tensor([.3, -.2, .4])
    h = cov.new_tensor([[.2, .07, -.04], [.07, -.1, .03], [-.04, .03, .15]])
    return mean @ g + (cov*h).sum() + .8*p['opacity'].sigmoid()[0]*marginal


def analytic_conditioning_vjp(p, spec):
    sigma = sh.covariance(p)
    b, c = sigma[:3, 3].detach(), sigma[3, 3].detach()
    dt = spec['timestamp']-p['time'][0].detach()
    m = torch.exp(-dt.square()/(2*c))
    opacity = p['opacity'].detach().sigmoid()[0]
    g = sigma.new_tensor([.3, -.2, .4])
    h = sigma.new_tensor([[.2, .07, -.04], [.07, -.1, .03], [-.04, .03, .15]])
    gm = .8*opacity
    gb = dt*g/c-(h+h.T)@b/c
    gc = -dt*(g@b)/c.square() + b@h@b/c.square() + gm*m*dt.square()/(2*c.square())
    gt = -(g@b)/c + gm*m*dt/c
    # Both symmetric b entries share one mathematical gradient.
    gs = torch.zeros_like(sigma)
    gs[:3, :3], gs[:3, 3], gs[3, :3], gs[3, 3] = h, gb/2, gb/2, gc
    proxy = (sigma*gs).sum() + p['mean']@g + p['time'][0]*gt
    proxy = proxy + p['opacity'][0]*(.8*m*opacity*(1-opacity))
    return sh.gradients(proxy, p)


def cases():
    values = []
    variants = [('T0', True, 0., .4, (0, 0), 4.)]
    variants += [(f'T1/{c}/{sign}', True, sign*.3, c, (0, 0), 4.)
                 for c in (.25, .64) for sign in (-1, 1)]
    variants += [(f'T2/{sign}', False, sign*.2, .64, (0, 0), 4.) for sign in (-1, 1)]
    variants += [(f'T3/{duration}/{sign}', False, sign*.2, .64, (3, 2), duration)
                 for duration in (8., 4., 3.2) for sign in (-1, 1)]
    for label, identity, delta, variance, stage, duration in variants:
        p = sh.fixture()
        with torch.no_grad():
            p['log_scale'][3] = math.log(math.sqrt(variance))
            if identity:
                p['left'].copy_(p['left'].new_tensor([1., 0, 0, 0]))
                p['right'].copy_(p['left'])
            if stage == (0, 0): p['features'][1:].zero_()
        p = sh.clone(sh.clone(p, dtype=torch.float32))
        s = camera.camera(label, width=64, height=48, fx=48., fy=40., identity=True)
        s.update(timestamp=camera.f32(float(p['time'][0])+delta), duration=duration,
                 raw_interval=[-2, 6], time_divisor=8/duration, pixel=(37, 26))
        values.append(dict(label=label, points=[p], camera=s, stage=stage, dim=4,
                           pixels=[(37, 26)], loss=(.6, -.3, .2, 0., 0., 0., 0.), kind='smooth'))
    for culled in (False, True):
        p, q = sh.clone(values[6]['points'][0]), sh.clone(values[6]['points'][0])
        with torch.no_grad():
            q['mean'][2] += .4
            if culled: q['time'].fill_(30.)
        values.append(dict(values[6], label=f'T4/{culled}', points=[p, q], pixels=[(37, 26), (36, 26)],
                           kind='time_cull' if culled else 'multiple'))
    return values


def require(record, condition, reason):
    """Keep the failed premise in the caller's result before raising."""
    if not condition:
        record.update(status='rejected', reason=reason)
        raise AssertionError(reason)


def contributor_order(case, points=None, evidence=None):
    """CPU conditional view-depth keys; never reorder or replace input leaves.

    Sorting is discrete. Only its scalar keys leave autograd; compositing below
    still uses the original point references. Distinct depths collapsed to one
    binary32 key are outside this independent oracle's verified order domain.
    """
    points = case['points'] if points is None else points
    record = {} if evidence is None else evidence
    record.update(status='checking', nodes=[], order=[])
    s = case['camera']
    for i, p in enumerate(points):
        mean, _, m, _ = sh.conditional(p, s)
        depth = float((s['rotation'] @ (mean-s['center']))[2])
        node = dict(gaussian_id=i, marginal=float(m), time_active=float(m) > .05,
                    depth=depth)
        record['nodes'].append(node)
        require(record, math.isfinite(float(m)), 'order_nonfinite_marginal')
        if not node['time_active']: continue
        require(record, math.isfinite(depth) and depth > 0, 'order_nonpositive_or_nonfinite_depth')
        try: rounded = camera.f32(depth)
        except OverflowError: rounded = float('inf')
        require(record, math.isfinite(rounded) and rounded > 0, 'order_binary32_depth')
        node.update(depth_f32=rounded, depth_key=camera.bits(rounded))
        record['order'].append(i)
    record['order'].sort(key=lambda i: (record['nodes'][i]['depth_key'], i))
    for a, b in zip(record['order'], record['order'][1:]):
        x, y = record['nodes'][a], record['nodes'][b]
        require(record, x['depth_key'] != y['depth_key'] or x['depth'] == y['depth'],
                'order_ambiguous_rounding_tie')
    record['status'] = 'ordered'
    return record['order']


def initialization_case(data, indices, time, timestamp, label):
    """Existing initialization expectation, independent of model/KNN outputs."""
    import numpy as np
    xyz = torch.tensor(np.stack([data[k][indices] for k in ('x','y','z')], 1), dtype=torch.float64)
    times = (data['time'][indices].astype('float64')/time.divisor).astype('float32')
    knn = knn_squared(xyz)
    rgb = np.stack([data[k][indices] for k in ('red','green','blue')], 1)/255
    points = []
    for i in range(len(indices)):
        p = sh.fixture()
        with torch.no_grad():
            p['mean'].copy_(xyz[i]); p['time'][0] = float(times[i])
            p['log_scale'][:3] = knn[i].sqrt().log()
            p['log_scale'][3] = camera.f32(time.log_effective_scale)
            p['left'].copy_(camera.tensor([1.,0,0,0])); p['right'].copy_(p['left'])
            p['opacity'][0] = math.log(.1/.9)
            p['features'].zero_()
            p['features'][0] = camera.tensor((rgb[i]-.5)/math.sqrt(1/(4*math.pi)))
        points.append(sh.clone(sh.clone(p, dtype=torch.float32)))
    s = camera.camera(label, width=64, height=48, fx=48., fy=40., identity=True)
    s.update(timestamp=timestamp, duration=time.effective_duration)
    return dict(label=label, points=points, camera=s, stage=(0,0), dim=4,
                pixels=[(32,24)], loss=(.6,-.3,.2,0.,0.,0.,0.))


def initialization_domain(case, record):
    """Premises for the SIX initialization comparisons, NOT training policy.

    No general pixel branch simulator: the existing compositor is applicable
    only if every time-active point contributes, with no cap/skip/early-out.
    The identity pose/rotations make view depth exactly the stored binary32 z;
    hence even duplicate-row ties have independently certified native keys.
    """
    order = contributor_order(case, evidence=record)
    points, s = case['points'], case['camera']
    require(record, case['dim'] == 4 and case['stage'] == (0,0) and len(case['pixels']) == 1,
            'initial_case_domain')
    require(record, torch.equal(s['rotation'], torch.eye(3, dtype=torch.float64)) and
            not bool(s['center'].count_nonzero()), 'initial_exact_depth_pose')
    x, y = case['pixels'][0]; tile = (x//16, y//16)
    record.update(pixel=[x,y], image=[s['width'],s['height']], tile=list(tile), final_transmittance=1.)
    require(record, 0 <= x < s['width'] and 0 <= y < s['height'], 'initial_pixel_outside')
    for i, (p, node) in enumerate(zip(points, record['nodes'])):
        if 'row_indices' in record: node['source_row'] = record['row_indices'][i]
        require(record, abs(node['marginal']-.05) > 1e-6, 'initial_time_boundary')
        if not node['time_active']:
            node.update(reason='time_cull', radius=0)
            continue
        identity = camera.tensor([1.,0,0,0])
        require(record, torch.equal(p['left'], identity) and torch.equal(p['right'], identity) and
                node['depth'] == float(p['mean'][2]) == node['depth_f32'], 'initial_exact_depth_key')
        g = camera.geometry(p, s, (x,y)); top = camera.topology(g, s)
        node.update(radius=top['radius'], tiles=list(top['tiles']), visible=top['visible'],
                    power=float(g['power']), raw_alpha=float(g['raw']), xy=g['xy'].tolist())
        require(record, all(bool(torch.isfinite(v).all()) for v in g.values()), 'initial_geometry_nonfinite')
        require(record, top['visible'] and node['depth'] > camera.f32(.2)+1e-6, 'initial_frustum')
        lo_x, lo_y, hi_x, hi_y = top['tiles']
        require(record, top['radius'] > 0 and lo_x <= tile[0] < hi_x and lo_y <= tile[1] < hi_y,
                'initial_target_tile')
        require(record, node['power'] <= 0, 'initial_power_skip')
        require(record, 1/255+1e-6 < node['raw_alpha'] < .99-1e-6, 'initial_alpha_domain')
        node['reason'] = 'contribute'
    for i in order:
        node = record['nodes'][i]; before = record['final_transmittance']
        after = before*(1-node['raw_alpha'])
        node.update(trans_before=before, trans_after=after)
        record['final_transmittance'] = after
        require(record, after > .0001+1e-6, 'initial_early_out_domain')
    record['status'] = 'applicable'
    return record


def evaluate(case, points=None):
    """Exact time predicate precedes the independent camera/alpha compositor."""
    points = case['points'] if points is None else points
    active = contributor_order(case, points)
    if not active:
        # Preserve a zero derivative path for every input without fake points.
        z = sum(v.sum()*0 for p in points for v in p.values())
        sample = torch.cat((camera.tensor(case['camera']['background']), camera.tensor([0., 0., 0., 0.])))+z
        return [dict(outputs=sample, loss=sample@camera.tensor(case['loss']), geometry=[]) for _ in case['pixels']]
    degree, temporal = case['stage']
    count = (degree+1)**2 if temporal == 0 else 16*(temporal+1)
    selected = []
    for i in active:
        p = points[i]; mask = torch.zeros_like(p['features']); mask[:count] = 1
        selected.append(dict(p, features=p['features']*mask))
    result = camera.evaluate(dict(case, points=selected))
    for row in result: row['active'] = active
    return result


def loss(case, points=None):
    return sum(x['loss'] for x in evaluate(case, points))


def gradients(case):
    flat = [v for p in case['points'] for v in p.values()]
    gradients = iter(torch.autograd.grad(loss(case), flat, allow_unused=True))
    return [{k: torch.zeros_like(v) if (g := next(gradients)) is None else g for k, v in p.items()}
            for p in case['points']]


def domain(case, points=None):
    """Time-specific smooth domain; never weakens the old SH/camera tests."""
    points = case['points'] if points is None else points
    result = []
    for pixel in case['pixels']:
        row = []
        for p in points:
            g = camera.geometry(p, case['camera'], pixel)
            assert all(bool(torch.isfinite(v).all()) for v in g.values()), 'time_fixture_nonfinite'
            c, m = float(g['sigma4'][3, 3]), float(g['marginal'])
            topo = camera.topology(g, case['camera'])
            assert .1 <= c <= 2 and topo['visible'], 'time_fixture_geometry'
            if m > .05:
                assert m > .2 and .008 < float(g['raw']) < .9, 'time_fixture_alpha_margin'
                assert bool((g['color'] > .1).all()), 'time_fixture_color_margin'
                assert float(torch.linalg.det(g['screen_cov'])) >= 1, 'time_fixture_covariance'
            row.append((topo['radius'], topo['tiles'], topo['clamp'], topo['visible'], m > .05,
                        float(g['raw']) >= 1/255, float(g['raw']) < .99))
        result.append(row)
    return result


def coordinates(case):
    return [(i, k, j) for i, p in enumerate(case['points']) for k in PARAMETERS for j in range(p[k].numel())]


def threshold():
    marker = camera.f32(.05); bits = camera.bits(marker)
    lower = struct.unpack('f', struct.pack('I', bits-1))[0]
    assert lower < .05 < marker
    return dict(literal=.05, marker=marker, lower=lower, exact_equality_reachable=False)


def boundary_grid():
    """Fixed 64x64 real-input search, not marginal injection or oracle fitting."""
    def neighbors(x):
        n = camera.bits(x)
        return [struct.unpack('f', struct.pack('I', n+i))[0] for i in range(-32, 32)]
    for logscale in neighbors(math.log(1/math.sqrt(-2*math.log(.05)))):
        for timestamp in neighbors(1.):
            yield logscale, timestamp


def knn_squared(points):
    delta = points[:, None, :]-points[None, :, :]
    distances = delta.square().sum(-1)
    distances.fill_diagonal_(float('inf'))
    return distances.sort(dim=1).values[:, :3].mean(1).clamp_min(1e-7)
