"""Pass-through boundary observations shared by CPU and future CUDA cases.

All faults are injected here, never in production. No oracle scans corrupted
state before the production health guard. Checkpoints use real save/read-back.
"""
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from pathlib import Path
from unittest.mock import patch

MODES = ('normal', 'invisible', 'loss', 'gradient', 'statistics', 'post_adam',
         'empty_population', 'reset', 'evaluation', 'metric', 'save', 'png', 'writer')


def files(directory):
    return {str(p.relative_to(directory)): (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in directory.rglob('*') if p.is_file()}


def exercise(p, renderer, mode, writer=None, continuation=None, budget=None):
    import torch
    import train
    import formal_checkpoint as cp
    from formal_training_health import HealthError
    assert mode in MODES
    record, m = p.health_run, p.gaussians
    counts = dict(fetch=0, backward=0, adam=0, population=0, reset=0, render=0)
    reports, saves, preserved, checkpoints = [], [], {}, []
    original_loader, original_l1, original_msssim = p.dataloader, train.l1_loss, train.msssim
    real_save, real_report = p.scene.save, train.training_report
    real_step, real_population, real_reset = m.optimizer.step, m.densify_and_prune, m.reset_opacity
    real_stats = m.add_densification_stats_grad
    report_k = [None]
    injected = []
    invisible_outputs = []
    class Batches:
        def __iter__(self):
            for batch in original_loader:
                counts['fetch'] += 1
                yield batch
    def fire(name):
        injected.append(name)
    def render(*a, **kw):
        counts['render'] += 1
        if budget is not None:
            budget['forwards'] += 1
            assert budget['forwards'] <= 500 and len(m._xyz) <= 64
        value = renderer(*a, **kw)
        if mode == 'invisible':
            assert len(m._xyz) > 0 and not bool(value['visibility_filter'].any())
            assert bool((value['radii'] == 0).all() and (value['depth'] == 0).all())
            invisible_outputs.append(True)
        if mode == 'evaluation' and report_k[0] == 2:
            fire('evaluation');value['render'] = value['render'].clone()
            value['render'][0, 0, 0] = float('inf')
        if report_k[0] is None:
            value['render'].register_hook(lambda g: (counts.__setitem__('backward', counts['backward']+1), g)[1])
        return value
    def l1(*a, **kw):
        result = original_l1(*a, **kw)
        if mode == 'loss' and report_k[0] is None and record.attempted == 2:
            fire('loss');return result * float('nan')
        return result
    def gradient(g):
        if mode == 'gradient' and record.attempted == 2:
            fire('gradient');return torch.full_like(g, float('inf'))
        return g
    def msssim(*a, **kw):
        result = original_msssim(*a, **kw)
        if mode == 'metric' and report_k[0] == 2:
            fire('metric');return result * float('nan')
        return result
    def step(*a, **kw):
        counts['adam'] += 1
        result = real_step(*a, **kw)
        if mode == 'post_adam' and record.attempted == 2:
            fire('post_adam');m.optimizer.state[m._xyz]['exp_avg'][0, 0] = float('nan')
        return result
    def stats(*a, **kw):
        result = real_stats(*a, **kw)
        if mode == 'statistics' and record.attempted == 2:
            fire('statistics');m.t_gradient_accum[0, 0] = float('inf')
        return result
    def population(*a, **kw):
        counts['population'] += 1
        result = real_population(*a, **kw)
        if mode == 'empty_population' and record.attempted == 2:
            fire('empty_population');m.prune_points(torch.ones(len(m._xyz), device=m._xyz.device, dtype=torch.bool))
            assert len(m._xyz) == 0
        return result
    def reset(*a, **kw):
        counts['reset'] += 1
        result = real_reset(*a, **kw)
        if mode == 'reset' and record.attempted == 2:
            fire('reset');m._opacity[0, 0] = float('nan')
        return result
    def save(k):
        result = real_save(k)
        directory = Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k))
        # Independent public reader, while state is still valid.
        read = cp.read_checkpoint(directory)
        assert read['metadata']['k'] == k
        preserved[k] = files(directory)
        checkpoints.append(dict(k=k,path=str(directory),marker=read['marker_identity'],
                                reference=__import__('json').loads(p.inputs.reference_bytes)))
        if mode == 'save' and k == 2:
            fire('save');raise OSError('injected post-publication failure')
        saves.append(k)
        return result
    def report(*a, **kw):
        report_k[0] = a[1]
        try:
            result = real_report(*a, **kw)
            reports.append(a[1]);return result
        finally:
            report_k[0] = None
    with ExitStack() as stack:
        for owner, name, value in ((train, 'render', render), (train, 'l1_loss', l1),
                (train, 'msssim', msssim),
                (m.optimizer, 'step', step), (m, 'densify_and_prune', population),
                (m, 'reset_opacity', reset), (m, 'add_densification_stats_grad', stats),
                (p.scene, 'save', save), (train, 'training_report', report)):
            stack.enter_context(patch.object(owner, name, value))
        hook = m._xyz.register_hook(gradient);stack.callback(hook.remove)
        if mode == 'png':
            real_png = train.Image.Image.save
            def png(image, *a, **kw):
                if report_k[0] == 2:
                    fire('png');raise OSError('injected PNG failure')
                return real_png(image,*a,**kw)
            stack.enter_context(patch.object(train.Image.Image, 'save', png))
        if mode == 'writer':
            real_scalar = writer.add_scalar
            def scalar(*a, **kw):
                if record.completed == 2:
                    fire('writer');raise OSError('injected writer failure')
                return real_scalar(*a,**kw)
            stack.enter_context(patch.object(writer, 'add_scalar', scalar))
        try:
            returned = (continuation or train.continue_formal_training)(p._replace(dataloader=Batches()), writer)
        except BaseException as error:
            expected = {'loss':'loss', 'gradient':'gradients', 'statistics':'statistics',
                'post_adam':'post_adam', 'empty_population':'post_population', 'reset':'post_reset',
                'evaluation':'report', 'metric':'report', 'save':'save', 'png':'report', 'writer':'report'}
            assert mode in expected and injected, (mode, type(error).__name__, str(error))
            assert isinstance(error, HealthError if mode not in ('save','png','writer') else OSError)
            assert record.stage == expected[mode], (record.stage, expected[mode])
            assert record.completed == (2 if mode in ('evaluation','metric','save','png','writer') else 1)
            assert counts['fetch'] == 2 and reports == [0, 1]
            batch = p.inputs.config.optimization.batch_size
            assert counts['backward'] == (batch if mode == 'loss' else batch+1 if mode == 'gradient' else batch*2)
            assert counts['adam'] == (1 if mode in ('loss','gradient','statistics') else 2)
            if mode in ('loss','gradient','statistics','post_adam'):
                assert counts['population'] == counts['reset'] == 0
            assert saves == ([1, 2] if mode in ('evaluation','metric','png','writer') else [1])
            if mode == 'save':
                assert record.save_outcome == 'failed_or_uncertain' and record.last_confirmed_save == 1
            outcome = dict(run='failed_expected', error_type=type(error).__name__, diagnostic=asdict(record))
        else:
            assert mode in ('normal','invisible') and returned == p.inputs.config.optimization.total_updates
            outcome = dict(run='returned', returned=returned, diagnostic=asdict(record))
    for k, before in preserved.items():
        directory = Path(p.scene.model_path)/'checkpoints'/('completed-'+str(k))
        assert files(directory) == before
        assert cp.read_checkpoint(directory)['metadata']['k'] == k
    if mode == 'invisible':
        assert invisible_outputs and bool((m.denom == 0).all())
    return dict(status='passed', mode=mode, outcome=outcome, counts=counts, reports=reports,
                saves=saves, preserved=list(preserved), checkpoints=checkpoints, injected=injected)
