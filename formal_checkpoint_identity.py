"""Checkpoint identity and stage-M utilities. Standard library only."""
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCHEMA = 'corrected-formal-checkpoint-v1'
GROUPS = dict(xyz=(1, '_xyz'), f_dc=(2, '_features_dc'), f_rest=(3, '_features_rest'),
    opacity=(6, '_opacity'), scaling=(4, '_scaling'), rotation=(5, '_rotation'),
    t=(13, '_t'), scaling_t=(14, '_scaling_t'), rotation_r=(15, '_rotation_r'))
KNN_UNITS = ('simple-knn/spatial.cu', 'simple-knn/simple_knn.cu', 'simple-knn/ext.cpp')
RASTER_UNITS = tuple('diff-gaussian-rasterization/' + p for p in (
    'cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
    'cuda_rasterizer/backward.cu', 'rasterize_points.cu', 'ext.cpp'))


def require(condition, code):
    if not condition:
        raise ValueError('checkpoint_' + code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def strict_json(data, limit=8 * 1024 * 1024):
    require(type(data) is bytes and len(data) <= limit, 'json_size')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate_key')
            result[key] = value
        return result
    def bad(_):
        raise ValueError('checkpoint_nonfinite_json')
    try:
        result = json.loads(data.decode('utf-8'), object_pairs_hook=unique, parse_constant=bad)
        def visit(value, depth=0):
            require(depth <= 16, 'json_depth')
            if type(value) is dict:
                for k, v in value.items():
                    require(type(k) is str, 'json_key')
                    visit(v, depth+1)
            elif type(value) is list:
                for v in value: visit(v, depth+1)
            elif type(value) is float:
                require(math.isfinite(value), 'json_number')
            else:
                require(type(value) in (str, int, bool, type(None)), 'json_type')
        visit(result)
        return result
    except (UnicodeError, RecursionError, json.JSONDecodeError):
        raise ValueError('checkpoint_json') from None


def keys(value, expected):
    require(type(value) is dict and set(value) == set(expected.split()), 'keys')


def validate_optimizer_descriptor(desc):
    """Metadata-only meaning, shared by stage M and in-memory snapshots.

    Payload ID/state/dtype correspondence is separately checked in stage T.
    Dtype names here are strings: accepting metadata must not import Torch.
    """
    require(type(desc) is dict and set(desc) == set(GROUPS), 'optimizer_descriptor')
    ids = set()
    for name, (slot, _) in GROUPS.items():
        d = desc[name]
        keys(d, 'slot id initialized step_dtype')
        require(type(d['slot']) is int and d['slot'] == slot and
                type(d['initialized']) is bool, 'optimizer_binding')
        require(type(d['id']) is int and d['id'] >= 0 and d['id'] not in ids, 'optimizer_id')
        ids.add(d['id'])
        if d['initialized']:
            require(type(d['step_dtype']) is str and
                    d['step_dtype'] in ('torch.float32', 'torch.float64', 'torch.int64'), 'adam_step')
        else:
            require(d['step_dtype'] is None, 'uninitialized_step')


def projection(value):
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return {n: projection(getattr(value, n)) for n in value._fields}
    if isinstance(value, (list, tuple)):
        return [projection(v) for v in value]
    if type(value) is dict:
        return {k: projection(v) for k, v in value.items()}
    require(type(value) in (str, int, float, bool, type(None)), 'projection')
    return value


def byte_identity(data):
    return dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())


def validate_identity(value):
    keys(value, 'size sha256')
    require(type(value['size']) is int and value['size'] >= 0 and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'identity')


def open_directory(path):
    """Walk absolute components by fd, never follow a parent symlink."""
    p = Path(path)
    require(p.is_absolute() and str(p) == str(path) and '..' not in p.parts, 'directory_path')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in p.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def checked_file(dir_fd, name, limit):
    require(type(name) is str and Path(name).name == name, 'file_name')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, 'file_size_type')
        stream = os.fdopen(fd, 'rb'); fd = None
        return stream, before
    finally:
        if fd is not None: os.close(fd)


def stream_identity(stream, before):
    stream.seek(0); h = hashlib.sha256(); size = 0
    while True:
        data = stream.read(1024*1024)
        if not data: break
        size += len(data); h.update(data)
    after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            and size == before.st_size, 'file_changed')
    stream.seek(0)
    return dict(size=size, sha256=h.hexdigest())


def file_identity(path):
    p = Path(path)
    require(p.resolve() == p, 'aliased_file')
    fd = open_directory(str(p.parent))
    try:
        stream, before = checked_file(fd, p.name, 8 * 1024**3)
        with stream: return stream_identity(stream, before)
    finally:
        os.close(fd)


def source_snapshot():
    paths = {ROOT/'train.py', ROOT/'simple-knn/setup.py', ROOT/'formal_dataset_reference.json'}
    paths.update(ROOT.glob('formal*.py'))
    for package in ('arguments', 'scene', 'utils', 'gaussian_renderer'):
        paths.update((ROOT/package).rglob('*.py'))
    for package in ('simple-knn', 'diff-gaussian-rasterization'):
        paths.update(p for p in (ROOT/package).rglob('*')
                     if p.suffix in ('.cu', '.cpp', '.h', '.cuh', '.hpp', '.inl'))
    require(0 < len(paths) <= 3000, 'source_count')
    return {str(p.relative_to(ROOT)): file_identity(p) for p in sorted(paths)}


def verify_source(files):
    require(files == source_snapshot(), 'source_changed')
    for name in files:
        if not name.endswith('.py') or name.startswith(('simple-knn/', 'diff-gaussian-')): continue
        modname = name[:-3].replace('/', '.')
        if modname.endswith('.__init__'): modname = modname[:-9]
        module = sys.modules.get(modname)
        if module is not None:
            require(Path(getattr(module, '__file__', '')).resolve() == ROOT/name, 'module_source')


@dataclass(frozen=True)
class NativeEvidence:
    encoded: bytes

    def value(self):
        return strict_json(self.encoded)


def validate_native(evidence, *, live=False):
    require(type(evidence) is NativeEvidence, 'native_type')
    e = evidence.value()
    keys(e, 'schema repository source build_root knn raster environment base_head')
    require(e['schema'] == 'formal-native-v1' and e['repository'] == str(ROOT), 'native_schema')
    verify_source(e['source'])
    root = Path(e['build_root'])
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'build_root')
    require(not any(p == root or p in root.parents for p in (ROOT,)), 'build_overlap')
    for name, units in (('knn', KNN_UNITS), ('raster', RASTER_UNITS)):
        row = e[name]
        keys(row, 'binary recipe objects command')
        require(type(row['command']) is list and row['command'] and
                all(type(v) is str for v in row['command']), 'build_command')
        for field in ('binary', 'recipe'):
            keys(row[field], 'path identity')
        require(type(row['objects']) is dict and len(row['objects']) == len(units), 'object_count')
        entries = [(row[f]['path'], row[f]['identity']) for f in ('binary', 'recipe')]
        entries += list(row['objects'].items())
        for path, identity in entries:
            validate_identity(identity)
            require(root in Path(path).parents and file_identity(Path(path)) == identity, 'build_file')
        recipe = Path(row['recipe']['path']).read_text()
        require(all(str(ROOT/u) in recipe for u in units), 'recipe_sources')
    keys(e['environment'], 'python executable torch torch_cuda')
    require(all(type(v) is str and v for v in e['environment'].values()), 'environment')
    require(type(e['base_head']) is str and re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', e['base_head']), 'base_head')
    if live:
        knn = sys.modules.get('simple_knn._C')
        binding = sys.modules.get('gaussian_renderer.diff_gaussian_rasterization')
        require(knn is not None and binding is not None, 'native_missing')
        for module, row in ((knn, e['knn']), (binding._C, e['raster'])):
            require(Path(getattr(module, '__file__', '')).resolve() == Path(row['binary']['path']), 'native_loaded_path')
        model = sys.modules.get('scene.gaussian_model')
        require(model is not None and model.distCUDA2 is knn.distCUDA2, 'knn_function')
    return e
