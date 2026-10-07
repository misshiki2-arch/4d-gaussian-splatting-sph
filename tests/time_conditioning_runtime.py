"""Step 11 synthetic real-consumer fixture. GPU isolation is caller-owned.

Only the fixed-reference acquisition is replaced locally. No consumer,
sampling or KNN substitution here; no formal output, optimizer or training.
"""
import hashlib
import io
import json
from unittest.mock import patch


def initialize(base, divisor, sampled=False):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData, PlyElement
    import formal_inputs
    from formal_views import ConsumerView
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from test_formal_config import fixture, encode
    from train import setup_seed

    source = base/'data'; source.mkdir()
    (source/'images').mkdir(); (source/'masks').mkdir()
    value = fixture(source, base/'unused-output')
    value['dataset']['resolution']['divisor'] = 1
    value['dataset']['time'].update(raw_interval=[-2, 6], divisor=divisor)
    value['initialization'].update(time_variance_denominator=5, num_pts=12 if sampled else 16)
    reference = dict(source_path=str(source), width=64, height=48, files={}, frames={})
    def store(name, data):
        (source/name).write_bytes(data)
        reference['files'][name] = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    for split, stamps in (('train', (0., 0., .2)), ('test', (-2., 6.))):
        frames = []; reference['frames'][split] = []
        for i, stamp in enumerate(stamps):
            filename = f'images/{split}{i}.png'; mask = filename.replace('images/', 'masks/')
            Image.new('RGBA', (64, 48), (90, 80, 70, 255)).save(source/filename)
            Image.new('L', (64, 48), 255).save(source/mask)
            pose = np.diag([1., -1., -1., 1.]); pose[0, 3] = .05*i
            frames.append(dict(file_path=filename, time=stamp, transform_matrix=pose.tolist()))
            reference['frames'][split].append([filename, stamp, mask])
        store(f'transforms_{split}.json', json.dumps(dict(w=64, h=48, fl_x=48., fl_y=40.,
                    cx=32., cy=24., frames=frames)).encode())
    rows = np.zeros(16, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('time', '<f4')])
    i = np.arange(16)
    rows['x'], rows['y'], rows['z'] = .08*(i%4-1.5), .06*(i//4-1.5), 3+.015*i
    rows['red'], rows['green'], rows['blue'] = 20+3*i, 50+2*i, 100+i
    rows['time'] = [-2., 0., .2, 2., 6., 0., .2, 2., 0., .2, 2., 6., -2., 0., .2, 2.]
    stream = io.BytesIO(); PlyData([PlyElement.describe(rows, 'vertex')], byte_order='<').write(stream)
    store('points3d.ply', stream.getvalue())
    with patch.object(formal_inputs, '_reference', return_value=reference):
        inputs = formal_inputs.verify_inputs(encode(value))
    config = inputs.config
    setup_seed(config.initialization.seed)
    expected = np.random.RandomState(config.initialization.seed).randint(0, 16, 12) if sampled else i
    draws = []; original = np.random.randint
    def observe(*args, **kwargs):
        result = original(*args, **kwargs); draws.append(result.copy()); return result
    model = GaussianModel(3, gaussian_dim=4, time_duration=config.time_derivation.effective_interval,
                          rot_4d=True, force_sh_3d=False, sh_degree_t=2, prefilter_var=-1.0)
    with patch.object(np.random, 'randint', side_effect=observe):
        scene = Scene(ConsumerView(config, 'dataset'), model, shuffle=False,
                      num_pts=config.initialization.num_pts,
                      time_duration=config.time_derivation.effective_interval, formal_inputs=inputs)
    if sampled:
        if len(draws) != 1 or not np.array_equal(draws[0], expected): raise AssertionError('sampling_draw')
    elif draws: raise AssertionError('unexpected_sampling')
    if (base/'unused-output').exists() or scene.cameras_extent <= 0: raise AssertionError('initialization_scope')
    return inputs, model, scene, rows, expected
