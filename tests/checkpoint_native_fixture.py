"""Synthetic compiler/native boundaries only; never evidence of a CUDA build."""
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

from formal_checkpoint_identity import (ROOT, KNN_UNITS, RASTER_UNITS, NativeEvidence,
    canonical, source_snapshot, validate_native)
import formal_native_bootstrap as native


@contextmanager
def cpu_device_queries():
    """Install before dependent imports, also in fresh-process CPU probes."""
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU query/NVML/CUDA forbidden in CPU fixture')
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=0))
        for name in ('_lazy_init', '_raw_device_count_nvml', '_raw_device_uuid_nvml',
                     'mem_get_info', 'memory_stats'):
            stack.enter_context(patch.object(torch.cuda, name, side_effect=forbidden))
        yield


def products(directory, units, binary):
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'build.ninja').write_text('\n'.join('build '+str(i)+'.o: cuda_compile '+str(ROOT/u)
                                                for i,u in enumerate(units)))
    for i,u in enumerate(units): (directory/(str(i)+'.o')).write_bytes(('CPU boundary '+u).encode())
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b'CPU native boundary, NOT ELF/CUDA')


def evidence_for_loaded_cpu_modules(base):
    """Old handoff fixtures only; the new entry test uses compiler_boundaries."""
    import torch
    area = base/'native-fixture'; area.mkdir()
    knn_binary = area/'knn-lib/simple_knn/_C.so'; raster_binary = area/'raster/_C.so'
    products(area/'knn-build', KNN_UNITS, knn_binary)
    products(area/'raster', RASTER_UNITS, raster_binary)
    sys.modules['simple_knn._C'].__file__ = str(knn_binary)
    binding = sys.modules['gaussian_renderer.diff_gaussian_rasterization']
    binding._C.__file__ = str(raster_binary)
    # Reuse the actual collector/schema/file/module checks, not a success stub.
    evidence = NativeEvidence(canonical(dict(schema='formal-native-v1', repository=str(ROOT),
        source=source_snapshot(), build_root=str(area),
        knn=native._record(area/'knn-build', knn_binary, KNN_UNITS, ['CPU-compiler-boundary']),
        raster=native._record(area/'raster', raster_binary, RASTER_UNITS, ['CPU-loader-boundary']),
        environment=dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                         torch_cuda=str(torch.version.cuda)),
        base_head=subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())))
    validate_native(evidence, live=True)
    return evidence


@contextmanager
def compiler_boundaries(base):
    """Fresh-process ordinary-entry test: real bootstrap, no injected receipt."""
    with cpu_device_queries(), _compiler_boundaries(base):
        yield


@contextmanager
def _compiler_boundaries(base):
    import torch
    import torch.utils.cpp_extension as ext
    def compile_knn(command, **kwargs):
        if command[2:4] != ['setup.py','build_ext']: raise AssertionError('unexpected compiler command')
        lib = Path(command[command.index('--build-lib')+1]); temp = Path(command[command.index('--build-temp')+1])
        products(temp, KNN_UNITS, lib/'simple_knn/_C.so')
        return subprocess.CompletedProcess(command, 0)
    def load_knn(binary):
        package = types.ModuleType('simple_knn'); package.__path__ = [str(binary.parent)]
        mod = types.ModuleType('simple_knn._C'); mod.__file__ = str(binary)
        mod.distCUDA2 = lambda xyz: torch.ones(len(xyz), dtype=torch.float32)
        sys.modules['simple_knn'] = package; sys.modules['simple_knn._C'] = mod; package._C = mod
        return mod
    def load_raster(*args, **kwargs):
        directory = Path(os.environ['TORCH_EXTENSIONS_DIR'])/'diff_gaussian_rasterization'
        binary = directory/'diff_gaussian_rasterization.so'
        products(directory, RASTER_UNITS, binary)
        mod = types.ModuleType('diff_gaussian_rasterization'); mod.__file__ = str(binary)
        def forbidden(*args, **kwargs): raise AssertionError('CUDA kernel forbidden')
        mod.__getattr__ = lambda n: forbidden
        return mod
    def forbidden(*args, **kwargs): raise AssertionError('real CUDA forbidden')
    pointops = types.ModuleType('pointops2.functions.pointops')
    pointops.knnquery = pointops.furthestsampling = forbidden
    with ExitStack() as stack:
        # Explicit scoped GPU module substitutions; real CPU deps remain real.
        modules = {'pointops2':types.ModuleType('pointops2'),
                   'pointops2.functions':types.ModuleType('pointops2.functions'),
                   'pointops2.functions.pointops':pointops}
        for name, mod in modules.items():
            relative = name.replace('.', '/')
            mod.__file__ = str(ROOT/(relative+'.py' if name.endswith('.pointops')
                                     else relative+'/__init__.py'))
            previous = sys.modules.get(name)
            sys.modules[name] = mod
            stack.callback(lambda n=name, p=previous: sys.modules.pop(n,None) if p is None else sys.modules.__setitem__(n,p))
        stack.enter_context(patch.object(native, 'subprocess', types.SimpleNamespace(
            run=compile_knn, check_output=subprocess.check_output, STDOUT=subprocess.STDOUT)))
        stack.enter_context(patch.object(native, '_load_knn', load_knn))
        stack.enter_context(patch.object(ext, 'load', load_raster))
        stack.enter_context(patch.object(ext, 'get_default_build_root', return_value=str(base/'cache')))
        stack.enter_context(patch.object(torch.cuda, '_lazy_init', side_effect=forbidden))
        stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
        stack.enter_context(patch.object(torch.cuda, 'set_device'))
        stack.enter_context(patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False))
        stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self,*a,**k:self))
        old_to = torch.Tensor.to
        def cpu_to(self, *args, **kw):
            if args and str(args[0]).startswith('cuda'): args=('cpu',)+args[1:]
            if str(kw.get('device','')).startswith('cuda'): kw['device']='cpu'
            return old_to(self,*args,**kw)
        stack.enter_context(patch.object(torch.Tensor, 'to', cpu_to))
        for name in ('tensor','zeros','ones','empty','zeros_like'):
            original = getattr(torch,name)
            def factory(*a,_original=original,**kw):
                if str(kw.get('device','')).startswith('cuda'): kw['device']='cpu'
                return _original(*a,**kw)
            stack.enter_context(patch.object(torch,name,factory))
        yield
