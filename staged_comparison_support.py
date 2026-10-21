"""Fixed Step18 adapters to reviewed recording/monitoring helpers (no trainer).

Only function/class definitions are reused from the old monitor/resources;
their scoped globals point at B demand accounting, never the A runtime.
"""
import ast
from datetime import datetime, timezone
import errno
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace, FunctionType

from formal_checkpoint_identity import ROOT, file_identity, require

REPORTS = ROOT.parent/'reports/corrected-4dgs/phase3'
OLD = REPORTS/'step17/step17-fix3'
IMPL = REPORTS/'step16/step16-implement4'
HERE = REPORTS/'step18/step18-fix3'
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
NINE = (0,15,30,2573,2588,2603,5115,5130,5145)
HISTORY = (4020,1740,1007,3074,2957)
VIEWS = NINE + HISTORY
RESOURCES = dict(policy='incremental_demand_v1', disk_reserve_bytes=8*1024**3,
    memory_margin_bytes=64*1024**2, memory_margin_fraction=[1,4],
    serialization_extra_fraction=[1,100], serialization_extra_bytes=1024**2,
    record_allowance_bytes=16*1024**2, gpu_query_timeout_seconds=20,
    poll_seconds=2, term_seconds=5, kill_seconds=5, elapsed_time_recorded=True,
    elapsed_time_stop=False, peak_usage_caps=False, live_capacity_verified=False)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def definitions(path, scope):
    tree = ast.parse(Path(path).read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def write_json(path, value):
    # The established exclusive write + independent finite JSON readback.
    return preparation().write_json(path, value)


def preparation():
    name = '_step18_preparation'
    if name not in sys.modules:
        module(name, IMPL/'comparison_prepare.py')
    return sys.modules[name]


def memory_helper():
    name = '_step18_memory'
    if name not in sys.modules:
        preparation()  # Supplies comparison_prepare to the unchanged helper.
        module(name, REPORTS/'step17/step17-fix2/step17_fix2_memory.py')
    return sys.modules[name]


def payload(n):
    require(type(n) is int and n > 0, 'resource_points')
    # Nine float32 Parameters:65 floats/row. Adam two moments, gradients,
    # radius and three accumulators; no A capture layout is used for restore.
    return dict(parameter=260*n, capture=796*n+72, training=1056*n+72,
                largest=180*n, raw=14*(20*1280*720+5*n))


def demand(kind, n, *, initial=False, gradients=None):
    p=payload(n); frame=128*1280*720; chunks=16*1024**2; cameras=5312*8192
    if kind=='worker_load':
        host=(p['parameter'] if initial else p['capture'])+cameras+frame+chunks
        gpu=p['training']+p['parameter']+frame
    elif kind=='reader_load':
        host=2*p['capture']+p['raw']+cameras+frame+chunks
        gpu=2*(p['parameter']+16*n)+frame
    elif kind=='observe':
        g=p['parameter'] if gradients is None else gradients
        require(type(g) is int and g>=0, 'gradient_bytes')
        host=p['capture']+g+p['raw']+frame+chunks; gpu=frame+5*n
    elif kind=='save':
        host=2*p['capture']+p['parameter']+chunks; gpu=None
    elif kind=='topology':
        # Existing split can add two children per selected row, plus clone;
        # conservative all-row concatenation/replacement transient, not a cap.
        host=chunks; gpu=8*p['training']+frame
    elif kind=='continue': host=frame+chunks; gpu=p['parameter']+frame
    elif kind=='reader_view': host=frame+chunks; gpu=frame+5*n
    elif kind=='comparison': host=frame+chunks; gpu=None
    elif kind=='parent': host=chunks; gpu=None
    else: raise ValueError('memory_boundary')
    margin=lambda size:64*1024**2+(size+3)//4
    return dict(boundary=kind,points=n,host_payload=host,gpu_payload=gpu,
        host_margin=margin(host),gpu_margin=None if gpu is None else margin(gpu),
        host_needed=host+margin(host),gpu_needed=None if gpu is None else gpu+margin(gpu),
        excludes='unknown native workspace/concurrent allocation; no completion guarantee')


def topology_demand(n, added):
    require(type(added) is int and added>=0,'topology_added_rows')
    d=demand('topology',n)
    # Replacement Parameter/moments/statistics plus new selected children and
    # masks/rotation/sampling temporaries. Already resident state is additional.
    gpu=2*payload(n+added)['training']+payload(max(added,1))['parameter']+128*1280*720
    margin=64*1024**2+(gpu+3)//4
    return dict(d,added_rows=added,gpu_payload=gpu,gpu_margin=margin,gpu_needed=gpu+margin)


def topology(directory,n,added):
    original=resources().memory
    fn=FunctionType(original.__code__,dict(original.__globals__,demand=lambda *a,**kw:topology_demand(n,added)),
                    original.__name__,original.__defaults__)
    fn.__kwdefaults__=original.__kwdefaults__
    return fn(directory,'topology',n)


def resources():
    if hasattr(resources, 'cached'): return resources.cached
    facade=SimpleNamespace(RESOURCES=RESOURCES,require=require)
    scope=dict(__name__='_step18_resources',json=json,os=os,Path=Path,re=re,
        resource=resource,shutil=shutil,subprocess=subprocess,time=time,c=facade,
        MIB=1024**2,PIXELS=1280*720,RECORD=RESOURCES['record_allowance_bytes'],
        RESERVE=RESOURCES['disk_reserve_bytes'],CHUNKS=16*1024**2)
    definitions(OLD/'step17-fix3-resources.py',scope)
    # Retain judge/start/memory/disk/parent predicate, replacing only B demands
    # and the same bounded probes with their original definitions.
    def collectors():
        s=SimpleNamespace(require=require,read_json=lambda p:json.loads(Path(p).read_text()))
        probe=dict(s=s,Path=Path,re=re,shutil=shutil,subprocess=subprocess,
                   GPU_REFERENCE=REPORTS/'step16/step16-validation5/step16-validation5-final-resources.json')
        tree=ast.parse((REPORTS/'step17/step17-fix1/stage_parent.py').read_text())
        tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('host_available','gpu_free')]
        exec(compile(tree,'stage_parent_probes','exec'),probe)
        return probe
    from staged_comparison_storage import disk_need
    scope.update(payload=payload,demand=demand,collectors=collectors,disk_need=disk_need)
    resources.cached=SimpleNamespace(**scope)
    return resources.cached


def monitor():
    if hasattr(monitor,'cached'):return monitor.cached
    facade=SimpleNamespace(RESOURCES=RESOURCES,local=lambda name:resources())
    scope=dict(__name__='_step18_monitor',datetime=datetime,timezone=timezone,
        json=json,os=os,Path=Path,shutil=shutil,signal=signal,subprocess=subprocess,
        sys=sys,time=time,c=facade,LIMITS=RESOURCES,
        errno=errno,traceback=traceback,
        KEYS=('boot_id','pid','start_ticks','pgid','sid','uid'),
        identity=file_identity,write_json=write_json)
    definitions(OLD/'step17-fix3-monitor.py',scope)
    scope['resource_sample']=scope['sample']
    definitions(ROOT/'staged_comparison_monitor.py',scope)
    monitor.cached=SimpleNamespace(**scope)
    return monitor.cached


def process_record(root, mode):
    # Retain the established full process/log/terminal validation. No eligible
    # historical failure is accepted on this new path.
    scope=dict(__name__='_step18_process_record',Path=Path,json=json,
        read_json=lambda p:json.loads(Path(p).read_text()),identity=file_identity,
        require=require,local=lambda name:monitor())
    tree=ast.parse((REPORTS/'step16/step16-fix7/step16-fix7-support.py').read_text())
    tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_record']
    exec(compile(tree,'step18_process_record','exec'),scope)
    return scope['process_record'](Path(root),mode)


def execution_files():
    """Independently enumerated required set, not candidate-selected membership."""
    paths=[OLD/'step17-fix3-monitor.py',OLD/'step17-fix3-resources.py',
        REPORTS/'step16/step16-fix7/step16-fix7-support.py',
        REPORTS/'step17/step17-fix1/stage_parent.py',
        REPORTS/'step17/step17-fix2/step17_fix2_memory.py',
        REPORTS/'step16/step16-validation5/step16-validation5-final-resources.json',
        HERE/'step18-fix3-entry.py',ROOT/'staged_comparison_monitor.py',
        ROOT/'staged_comparison_followup.py',
        REPORTS/'step18/step18-implement3/step18-implement3-instruction-evidence.json']
    paths.extend(IMPL/n for n in ('comparison_prepare.py','comparison_state.py',
                                 'comparison_observation.py'))
    # Observation/reader reuse test-side independent oracles and camera helpers.
    paths.extend(sorted((ROOT/'tests').glob('*.py')))
    paths.extend(sorted((ROOT/'tests/cuda').glob('*.py')))
    return {str(p):file_identity(p) for p in paths}


def check_execution_files(value):
    require(value==execution_files(),'comparison_execution_dependencies')
