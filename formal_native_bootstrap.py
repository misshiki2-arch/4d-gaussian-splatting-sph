"""Fixed formal KNN/rasterizer bootstrap; no general build/cache manager.

Only invoked after stdlib input checks. The native compiler/loader boundaries
are isolated in CPU tests; those tests are not build provenance acceptance.
"""
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

from formal_checkpoint_identity import (ROOT, KNN_UNITS, RASTER_UNITS, NativeEvidence,
    canonical, file_identity, require, validate_native, verify_source)


def _load_knn(binary):
    package = types.ModuleType('simple_knn'); package.__path__ = [str(binary.parent)]
    sys.modules['simple_knn'] = package
    spec = importlib.util.spec_from_file_location('simple_knn._C', binary)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sys.modules['simple_knn._C'] = module; package._C = module
    return module


def _record(directory, binary, units, command):
    recipes = list(directory.rglob('build.ninja')); objects = list(directory.rglob('*.o'))
    require(len(recipes) == 1 and len(objects) == len(units), 'fresh_build_files')
    require(all(str(ROOT/u) in recipes[0].read_text() for u in units), 'fresh_recipe')
    def item(p): return dict(path=str(p.resolve()), identity=file_identity(p.resolve()))
    return dict(binary=item(binary), recipe=item(recipes[0]),
                objects={str(p.resolve()): file_identity(p.resolve()) for p in sorted(objects)}, command=command)


def bootstrap(source):
    """Return the real imported runtime and internally generated load evidence."""
    require(not any(n in sys.modules for n in ('train', 'simple_knn._C',
        'gaussian_renderer', 'gaussian_renderer.diff_gaussian_rasterization',
        'diff_gaussian_rasterization')), 'native_preloaded')
    verify_source(source)
    import torch
    from torch.utils.cpp_extension import get_default_build_root
    cache = Path(get_default_build_root()).resolve()
    require(ROOT != cache and ROOT not in cache.parents, 'cache_repository_overlap')
    cache.mkdir(parents=True, exist_ok=True)
    area = Path(tempfile.mkdtemp(prefix='formal-checkpoint-', dir=cache))
    require(not list(area.iterdir()), 'fresh_area')
    lib, temp = area/'knn-lib', area/'knn-build'
    command = [sys.executable, '-B', 'setup.py', 'build_ext', '--build-lib', str(lib), '--build-temp', str(temp)]
    with (area/'knn-build.log').open('x') as log:
        subprocess.run(command, cwd=ROOT/'simple-knn', stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=1200)
    binaries = list((lib/'simple_knn').glob('_C*.so'))
    require(len(binaries) == 1, 'knn_binary_count')
    knn = _record(temp, binaries[0], KNN_UNITS, command)
    _load_knn(binaries[0].resolve())
    original = os.environ.get('TORCH_EXTENSIONS_DIR')
    os.environ['TORCH_EXTENSIONS_DIR'] = str(area/'raster')
    try:
        runtime = importlib.import_module('train')
    finally:
        if original is None: os.environ.pop('TORCH_EXTENSIONS_DIR', None)
        else: os.environ['TORCH_EXTENSIONS_DIR'] = original
    binding = sys.modules['gaussian_renderer.diff_gaussian_rasterization']
    raster = _record(area/'raster', Path(binding._C.__file__).resolve(), RASTER_UNITS,
                     ['torch.utils.cpp_extension.load', 'diff_gaussian_rasterization',
                      '-I '+str(ROOT/'diff-gaussian-rasterization/third_party/glm/')+'/', '-g'])
    verify_source(source)
    head = subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    evidence = NativeEvidence(canonical(dict(schema='formal-native-v1', repository=str(ROOT),
        source=source, build_root=str(area), knn=knn, raster=raster, base_head=head,
        environment=dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                         torch_cuda=str(torch.version.cuda)))))
    validate_native(evidence, live=True)
    return runtime, evidence
