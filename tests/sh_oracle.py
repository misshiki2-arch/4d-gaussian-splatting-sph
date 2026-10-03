"""Independent single-Gaussian CPU reference, NOT a production renderer.

Real SH: normalized associated Legendre recurrence (Condon--Shortley).
SO(4): Hamilton products in the audited reversed-xyzt convention.
No production mathematics, Scene, renderer, CUDA extension or RNG imports.
"""
import math
import torch

STAGES = ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2))
WEIGHTS = (.6, -.3, .2)
SPEC = dict(width=64, height=48, fx=48., fy=40., timestamp=.4, duration=1.,
            pixel=(37, 26), background=(.07, .11, .03))


def fixture(dtype=torch.float64, device='cpu'):
    """Explicit synthetic input, never a dataset/run configuration."""
    values = dict(mean=[.35, .25, 3.], time=[.2],
                  log_scale=[math.log(v) for v in (.2, .35, .5, .8)],
                  left=[1., .2, -.1, .15], right=[1., -.15, .12, .08],
                  opacity=[math.log(.35 / .65)])
    values['features'] = [[.012*math.sin((j+1)*(c+1)) for c in range(3)] for j in range(48)]
    values['features'][1] = [.5, -.35, .25]
    values['features'][16] = [.3, .2, .1]
    values['features'][32] = [.17, .09, .12]
    return {k: torch.tensor(v, dtype=dtype, device=device, requires_grad=True)
            for k, v in values.items()}


def clone(p, *, dtype=torch.float64, device='cpu'):
    return {k: v.detach().to(device=device, dtype=dtype).clone().requires_grad_(True)
            for k, v in p.items()}


def acceptance_cases():
    """Shared *inputs* only; no measured output feeds an expected value."""
    cases = [(str(stage), fixture(), stage, dict(SPEC), False) for stage in STAGES]
    for mode in (1, 2):
        for stamp in (0., .4):
            p = fixture()
            with torch.no_grad():
                p['features'][16*(3-mode):16*(4-mode)].zero_()
            cases.append((f'mode{mode}-{stamp}', p, (3,2), dict(SPEC, timestamp=stamp), False))
    for sign in (-1, 1):
        p = fixture()
        with torch.no_grad():
            p['features'].zero_()
            p['features'][1] = p['features'].new_tensor([.5,-.35,.25])
            p['mean'][1] = sign*.25
        cases.append((f'coefficient1-{sign}', p, (1,0), dict(SPEC), False))
    for control in ('delta', 'cross', '3d'):
        p = fixture()
        with torch.no_grad():
            if control == 'delta': p['time'].fill_(SPEC['timestamp'])
            if control == 'cross':
                p['left'].copy_(p['left'].new_tensor([1.,0,0,0]))
                p['right'].copy_(p['left'])
        cases.append((control, p, (3,0) if control == '3d' else (3,2), dict(SPEC), control == '3d'))
    return cases


def hamilton(a, b):
    w, x, y, z = a.unbind()
    v, i, j, k = b.unbind()
    return torch.stack((w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                        w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v))


def covariance(p):
    l = p['left'] / p['left'].norm()
    r = p['right'] / p['right'].norm()
    conjugate = r * r.new_tensor([1, -1, -1, -1])
    basis = torch.eye(4, dtype=l.dtype, device=l.device)
    a = torch.stack([hamilton(hamilton(l, e.flip(0)), conjugate).flip(0)
                     for e in basis], dim=1)
    return (a * p['log_scale'].exp().square()) @ a.T


def conditional(p, spec=SPEC):
    sigma = covariance(p)
    v, s = sigma[:3, 3], sigma[3, 3]
    delta = spec['timestamp'] - p['time'][0]
    mean = p['mean'] + v * delta / s
    cov = sigma[:3, :3] - v[:, None] * v[None, :] / s
    marginal = torch.exp(-delta.square() / (2*s))
    return mean, cov, marginal, sigma


def basis(direction, degree=3):
    """m ordering -l..l, j=l(l+1)+m; nonsingular directions for gradients."""
    x, y, z = (direction / direction.norm()).unbind()
    phi = torch.atan2(y, x)
    p = {}
    p[0, 0] = z*0 + 1
    radial = torch.sqrt(1-z*z)
    for m in range(1, degree+1):
        p[m, m] = -(2*m-1) * radial * p[m-1, m-1]
    for m in range(degree):
        p[m+1, m] = (2*m+1) * z * p[m, m]
    for m in range(degree+1):
        for l in range(m+2, degree+1):
            p[l, m] = ((2*l-1)*z*p[l-1, m] - (l+m-1)*p[l-2, m])/(l-m)
    result = []
    for l in range(degree+1):
        for m in range(-l, l+1):
            a = abs(m)
            n = math.sqrt((2*l+1)/(4*math.pi)*math.factorial(l-a)/math.factorial(l+a))
            value = n*p[l, a]
            if m: value = value*math.sqrt(2)*(torch.sin(a*phi) if m < 0 else torch.cos(a*phi))
            result.append(value)
    return torch.stack(result)


def color_at(p, mean, stage, spec=SPEC):
    if stage not in STAGES: raise ValueError('unsupported_active_pair')
    degree, temporal = stage
    y = basis(mean, degree)  # camera is at the origin in these fixtures
    result = .5 + y @ p['features'][:len(y)]
    theta = 2*math.pi*(p['time'][0]-spec['timestamp'])/spec['duration']
    for k in range(1, temporal+1):
        result = result + torch.cos(k*theta)*(y @ p['features'][16*k:16*(k+1)])
    return result.clamp_min(0)


def color(p, stage, spec=SPEC):
    return color_at(p, conditional(p, spec)[0], stage, spec)


def pixel(p, stage, spec=SPEC):
    """Smooth one-Gaussian pinhole oracle; no visibility/alpha boundary claim."""
    mean, cov, marginal, sigma = conditional(p, spec)
    x, y, z = mean.unbind()
    fx, fy = spec['fx'], spec['fy']
    zero = z*0
    jac = torch.stack((torch.stack((fx/z, zero, -fx*x/z**2)),
                       torch.stack((zero, fy/z, -fy*y/z**2))))
    screen_cov = jac @ cov @ jac.T + torch.eye(2, dtype=z.dtype, device=z.device)*.3
    # Full projection with centered principal point, then ndc2Pix. The CUDA
    # forward uses homogeneous division with +1e-7, not a changed camera.
    xy = torch.stack((fx*x/(z+1e-7)+(spec['width']-1)/2,
                      fy*y/(z+1e-7)+(spec['height']-1)/2))
    displacement = xy - xy.new_tensor(spec['pixel'])
    power = -.5 * displacement @ torch.linalg.solve(screen_cov, displacement)
    alpha = p['opacity'].sigmoid()[0]*marginal*power.exp()
    c = color_at(p, mean, stage, spec)
    bg = c.new_tensor(spec['background'])
    rgb = alpha*c + (1-alpha)*bg
    return dict(rgb=rgb, alpha=alpha, color=c, mean=mean, covariance=cov,
                sigma=sigma, marginal=marginal, screen_cov=screen_cov, xy=xy)


def isolate(rgb, alpha, background):
    # Deliberately NOT detach: cancellation is part of the test loss graph.
    return (rgb-(1-alpha)*rgb.new_tensor(background))/alpha


def loss(value):
    return value @ value.new_tensor(WEIGHTS)


def gradients(value, p):
    gs = torch.autograd.grad(value, tuple(p.values()), allow_unused=True)
    return {k: torch.zeros_like(p[k]) if g is None else g for k, g in zip(p, gs)}


def finite_difference(fn, p, relative_step):
    result = {}
    for name, v in p.items():
        g = torch.zeros_like(v)
        for i in range(v.numel()):
            h = relative_step*max(1., abs(v.detach().flatten()[i].item()))
            plus, minus = clone(p), clone(p)
            with torch.no_grad():
                plus[name].view(-1)[i] += h
                minus[name].view(-1)[i] -= h
            g.view(-1)[i] = (fn(plus).detach()-fn(minus).detach())/(2*h)
        result[name] = g
    return result


def assert_safe(p, spec=SPEC, stage=(3, 2)):
    """Reject an unsuitable test fixture, never repair a production input."""
    a = pixel(p, stage, spec)
    assert all(torch.isfinite(v).all() for v in a.values())
    assert torch.linalg.eigvalsh(a['sigma']).min() > 0
    assert a['sigma'][3, 3] > 0 and a['mean'].norm() > .5
    assert a['marginal'] > .5 and .05 < a['alpha'] < .5
    assert torch.all((a['color'] > .2) & (a['color'] < .8))
    assert a['mean'][2] > 1
    assert abs(a['mean'][0]/a['mean'][2]) < spec['width']/(2*spec['fx'])
    assert abs(a['mean'][1]/a['mean'][2]) < spec['height']/(2*spec['fy'])
    assert torch.linalg.det(a['screen_cov']) > 1
    # Verify radius and tile topology used by the production forward, purely
    # for domain selection; these discontinuous quantities are not an oracle
    # derivative. Pixel membership must stay well inside the support.
    c = a['screen_cov'].detach()
    midpoint = (c[0, 0]+c[1, 1])/2
    radicand = max(.1, (midpoint**2-torch.linalg.det(c)).item())
    radius_float = 3*math.sqrt(midpoint.item()+math.sqrt(radicand))
    assert abs(radius_float-round(radius_float)) > .01
    radius = math.ceil(radius_float)
    xy = a['xy'].detach().tolist()
    assert max(abs(xy[i]-spec['pixel'][i]) for i in range(2)) < radius-2
    bounds = tuple(int((xy[i]+sign*radius+(15 if sign > 0 else 0))/16)
                   for i in range(2) for sign in (-1, 1))
    return dict(alpha=a['alpha'].item(), marginal=a['marginal'].item(),
                determinant=torch.linalg.det(c).item(), radius=radius, tiles=bounds,
                color=a['color'].detach().tolist())
