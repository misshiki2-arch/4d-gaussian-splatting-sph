"""Step 12 separately authorized one-run CUDA acceptance (NOT a CPU test).

Record/preflight are stdlib only. A source candidate is not advisor approval.
Uses fresh isolated KNN/rasterizer builds, never another entry's main().
"""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT.parent/'reports/corrected-4dgs/phase1/step12'
PYTHON = '/home/demo/miniconda3/envs/4dgs310/bin/python'
SCHEMA = 'step12-training-inputs-v1'
HELPER = 'tests/cuda/time_conditioning_cuda_acceptance.py'
EXTRA = ('tests/training_update_oracle.py', 'tests/training_update_runtime.py',
         'tests/test_training_update_cpu.py', 'tests/test_training_update_source_identity.py',
         'tests/cuda/training_update_cuda_acceptance.py', 'tests/test_training_update_result.py',
         'tests/formal_training_fixture.py',
         'tests/test_formal_connection.py', 'tests/test_formal_training.py', 'tests/test_formal_population.py')
HEAVY = ('torch', 'numpy', 'PIL', 'scene', 'train', 'gaussian_renderer', 'simple_knn', 'pointops2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def helper():
    spec = importlib.util.spec_from_file_location('step12_time_helper', ROOT/HELPER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    if value.ROOT != ROOT or Path(value.__file__).resolve() != ROOT/HELPER:
        raise RuntimeError('helper_repository_mismatch')
    return value


def source_identity():
    value = helper().source_identity()
    value['schema'] = SCHEMA
    for name in EXTRA:
        path = ROOT/name
        if path.resolve() != path or not path.is_file():
            raise RuntimeError('input_missing_or_aliased: '+name)
        value['files'][name] = digest(path)
    value['files'] = dict(sorted(value['files'].items()))
    return value


def validate_evidence(value):
    if (type(value) is not dict or set(value) != {'schema', 'repository', 'base_head', 'files'}
            or value['schema'] != SCHEMA or value['repository'] != str(ROOT)
            or type(value['base_head']) is not str or not re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}', value['base_head'])
            or type(value['files']) is not dict or not set(EXTRA+(HELPER,)) <= value['files'].keys()):
        raise RuntimeError('evidence_schema')
    for name, sha in value['files'].items():
        if (type(name) is not str or not name or Path(name).is_absolute() or '..' in Path(name).parts
                or Path(name).as_posix() != name or type(sha) is not str or not re.fullmatch('[0-9a-f]{64}', sha)):
            raise RuntimeError('evidence_file')
    return value


def load_source_evidence(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError('duplicate_evidence_key')
            result[key] = value
        return result
    return validate_evidence(json.loads(path.read_text(), object_pairs_hook=unique))


def verify_source(expected):
    validate_evidence(expected)
    # Every candidate byte, including helper hierarchy, before executing helpers.
    for name, sha in expected['files'].items():
        path = ROOT/name
        if path.resolve() != path or not path.is_file() or digest(path) != sha:
            raise RuntimeError('reviewed_input_mismatch: '+name)
    actual = source_identity()
    if any(actual[k] != expected[k] for k in ('schema', 'repository', 'files')):
        raise RuntimeError('source_inputs_mismatch')
    return actual


def preflight(expected, out):
    before = verify_source(expected)
    if any(k.startswith('STEP90_CUDA_DEBUG') and v for k, v in os.environ.items()):
        raise RuntimeError('external_diagnostic_override')
    if not out.is_absolute() or out.exists() or out.is_symlink() or out.resolve() != out:
        raise RuntimeError('new_unaliased_absolute_output_required')
    if REPORTS not in out.parents or not out.parent.is_dir():
        raise RuntimeError('new_step12_run_directory_required')
    return before


def git_state():
    def run(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-C', str(ROOT), *args])
    index = Path(run('rev-parse', '--path-format=absolute', '--git-path', 'index').decode().strip())
    return dict(head=run('rev-parse', 'HEAD').decode().strip(), branch=run('branch', '--show-current').decode().strip(),
                status=run('status', '--porcelain=v1', '--untracked-files=all').decode(),
                entries=hashlib.sha256(run('ls-files', '--stage', '-z')).hexdigest(),
                flags=hashlib.sha256(run('ls-files', '-v', '-z')).hexdigest(), raw_index=digest(index))


def same_git(a, b):
    return all(a[k] == b[k] for k in ('head', 'branch', 'status', 'entries', 'flags'))


def loaded_test_inputs(expected):
    result = {}
    for file in expected['files']:
        if not file.startswith('tests/') or not file.endswith('.py') or '/cuda/' in file:
            continue
        name = Path(file).stem
        module = sys.modules.get(name)
        if module is not None:
            path = Path(getattr(module, '__file__', '')).resolve()
            if path != ROOT/file or digest(path) != expected['files'][file]:
                raise RuntimeError('unexpected_import_source: '+name)
            result[name] = str(path)
    return result


def result_payload(value, active=None):
    """Copy Step 12 results to JSON values, without lossy default=str coercion."""
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):  # torch.__version__ is a str subclass.
        return str.__str__(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite_result_value')
        return value
    if type(value) not in (dict, list, tuple):
        raise TypeError('unsupported_result_value: '+type(value).__name__)
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError('cyclic_result_value')
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError('nonstring_result_key')
            return {key: result_payload(item, active) for key, item in value.items()}
        return [result_payload(item, active) for item in value]
    finally:
        active.remove(id(value))


def read_result(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate_result_key')
            value[key] = item
        return value
    def nonfinite(unused):
        raise ValueError('nonfinite_result_value')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=nonfinite)


def result_failure(out, result, error):
    """One bounded failure record; never retry or replace result.json."""
    def bounded(value):
        return str(value)[:4000]
    failure = dict(status='failed', stage='result_finalization',
                   execution_status=bounded(result.get('status')),
                   execution_stage=bounded(result.get('stage')),
                   primary_error=bounded(result['error']) if 'error' in result else None,
                   preservation_errors={name: bounded(value) for name, value in
                       result.get('preservation', {}).items() if value is not True},
                   finalization_error=None if error is None else
                       bounded(type(error).__name__+': '+str(error)))
    def diagnostic(value):
        try:
            print('STEP12_RESULT_FAILED '+json.dumps(value, allow_nan=False), file=sys.stderr, flush=True)
        except OSError:
            pass  # The exception still propagates; the file may remain writable.
    diagnostic(failure)
    try:
        with (out/'failure.json').open('x', encoding='utf-8') as stream:
            json.dump(failure, stream, indent=2, allow_nan=False)
    except Exception as record_error:
        diagnostic(dict(failure, failure_record_error=bounded(type(record_error).__name__+': '+str(record_error))))


def finalize_result(out, result):
    """A saved payload is nonterminal; only a verified process receipt completes it.

    Acceptance requires exit 0 AND STEP12_RESULT_COMPLETE bound to result.json.
    This avoids a passed file left behind by write/close/read-back failure. No
    atomic publication, replacement, retry or checkpoint protocol is involved.
    """
    try:
        payload = result_payload(result)
        if (payload.get('status') not in ('checks_passed', 'failed')
                or 'finalization' in payload):
            raise ValueError('nonterminal_result_required')
        payload['finalization'] = 'requires_verified_process_receipt'
        path = out/'result.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        actual = result_payload(read_result(path))
        # Canonical JSON comparison also distinguishes true/1 and 1/1.0.
        encode = lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
        if encode(actual) != encode(payload):
            raise RuntimeError('result_readback_mismatch')
        preservation = payload.get('preservation', {})
        if (payload['status'] != 'checks_passed' or payload.get('stage') != 'checks_complete'
                or 'error' in payload
                or set(preservation) != {'source', 'git_content', 'rasterizer', 'KNN'}
                or any(value is not True for value in preservation.values())):
            result_failure(out, result, None)
            return False
        receipt = dict(status='passed', stage='complete', result_file=path.name, result_sha256=digest(path))
        print('STEP12_RESULT_COMPLETE '+json.dumps(receipt), flush=True)
        return True
    except BaseException as error:
        result_failure(out, result, error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--record-source-evidence', action='store_true')
    parser.add_argument('--source-evidence', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if sys.executable != PYTHON or not sys.dont_write_bytecode:
        raise RuntimeError('4dgs310_B_required')
    if any(n in sys.modules for n in HEAVY):
        raise RuntimeError('fresh_process_required')
    if args.record_source_evidence:
        if args.source_evidence is not None or args.output_dir is not None:
            parser.error('record_cannot_execute')
        print(json.dumps(source_identity(), indent=2))
        return
    if args.source_evidence is None or args.output_dir is None:
        parser.error('reviewed_source_and_new_output_required')
    expected = load_source_evidence(args.source_evidence)
    before = preflight(expected, args.output_dir)
    time_entry = helper()
    camera, alpha = time_entry.helper(), time_entry.helper().helper()
    sh = alpha.helper()
    git_before = git_state()
    out = args.output_dir
    out.mkdir()
    build = out/'rasterizer-build'
    build.mkdir()
    os.environ['TORCH_EXTENSIONS_DIR'] = str(build)
    sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
    result = dict(status='failed', stage='heavy_import', python=sys.executable, source_before=before,
                  git_before=git_before, checks=[], cases=[], CPU_success_is_not_CUDA_acceptance=True)
    started = time.monotonic()
    deadline = started+1800
    budget = dict(forwards=0)
    original_run = subprocess.run
    def bounded_run(*a, **kw):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('step12_wall_clock_bound')
        kw['timeout'] = min(remaining, kw.get('timeout') or remaining)
        return original_run(*a, **kw)
    def timeout(*unused):
        raise TimeoutError('step12_wall_clock_bound')
    old_alarm = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        with ExitStack() as guards:
            guards.enter_context(patch.object(subprocess, 'run', bounded_run))
            import torch
            result['environment'] = dict(python=sys.version, torch=torch.__version__, torch_cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), nvcc=sh.bounded_command(['nvcc', '--version']),
                compiler=sh.bounded_command(['c++', '--version']),
                driver=sh.bounded_command(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']))
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA_required_no_fallback')
            result['environment']['gpu'] = torch.cuda.get_device_name(0)
            result['stage'] = 'fresh_KNN'
            verify_source(expected)
            result['knn'] = time_entry.build_knn(out)
            result['stage'] = 'fresh_rasterizer'
            verify_source(expected)
            with (out/'rasterizer-build.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
                import gaussian_renderer as renderer
                from gaussian_renderer import diff_gaussian_rasterization as binding
            result['extension'] = alpha.binary_identity(binding, build, sh)
            import pointops2_cuda
            result['pointops_import_only'] = dict(path=pointops2_cuda.__file__, sha256=digest(Path(pointops2_cuda.__file__)))
            blocked = []
            def forbidden(*a, **kw):
                raise AssertionError('pointops_kernel_out_of_scope')
            for name, value in vars(pointops2_cuda).copy().items():
                if not name.startswith('_') and callable(value):
                    guards.enter_context(patch.object(pointops2_cuda, name, side_effect=forbidden))
                    blocked.append(name)
            if not blocked:
                raise RuntimeError('pointops_guard_missing')
            result['pointops_import_only']['blocked_native_calls'] = blocked
            from scene import gaussian_model
            import simple_knn._C as knn
            if (gaussian_model.distCUDA2 is not knn.distCUDA2 or knn.distCUDA2.__module__ != 'simple_knn._C'
                    or Path(knn.__file__).resolve() != Path(result['knn']['path'])):
                raise RuntimeError('actual_model_KNN_binding_mismatch')
            import training_update_runtime as runtime
            result['stage'] = 'training_updates'
            for case in runtime.CASES:
                result['stage'] = 'training/'+case['name']
                observer = []
                p = runtime.prepare(out/case['name'], case)
                try:
                    with (out/(case['name']+'.log')).open('x') as log, redirect_stdout(log), redirect_stderr(log):
                        runtime.execute_case(p, case, budget, observer_callback=observer.append)
                    torch.cuda.synchronize()
                finally:
                    if observer:
                        trace = runtime.serializable(observer[0].trace)
                        with (out/(case['name']+'.json')).open('x') as stream:
                            json.dump(trace, stream, indent=2, allow_nan=False)
                        result['cases'].append(dict(case=case['name'], status=trace['status'],
                                                   steps=len(trace['steps']), trace=case['name']+'.json'))
            if len(result['cases']) != 12 or sum(c['steps'] for c in result['cases']) != 35:
                raise AssertionError('normal_case_bound')
            result['stage'] = 'same_binary_initialization'
            time_entry.run_initialization(torch, binding, renderer, out, result)
            if len(result['initialization']) != 6:
                raise AssertionError('initialization_case_count')
            result['stage'] = 'same_binary_SH'
            sr = dict(source_before=dict(schema=sh.INPUT_SCHEMA, repository=str(ROOT), base_head=expected['base_head'],
                      files={n: expected['files'][n] for n in sh.source_identity()['files']}), checks=[])
            result['SH_regression'] = sr
            sh.run_checks(torch, binding, renderer, sr)
            if sr['case_count'] != 15:
                raise AssertionError('SH_case_count')
            result['loaded_inputs'] = camera.verify_loaded_inputs(time_entry.subset(expected, camera), alpha, sh)
            result['loaded_tests'] = loaded_test_inputs(expected)
            result['status'], result['stage'] = 'checks_passed', 'checks_complete'
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'[:4000]
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)
        result['elapsed_seconds'] = time.monotonic()-started
        result['normal_forward_count'] = budget['forwards']
        result['preservation'] = {}
        for name, check in (
            ('source', lambda: verify_source(expected)['files'] == expected['files']),
            ('git_content', lambda: same_git(git_state(), git_before)),
            ('rasterizer', lambda: 'extension' not in result or alpha.binary_identity(binding, build, sh) == result['extension']),
            ('KNN', lambda: 'knn' not in result or (digest(Path(result['knn']['path'])) == result['knn']['sha256']
                and digest(Path(result['knn']['recipe']['path'])) == result['knn']['recipe']['sha256']
                and all(digest(Path(p)) == sha for p, sha in result['knn']['objects'].items())))):
            try:
                result['preservation'][name] = check()
            except Exception as error:
                result['preservation'][name] = str(error)[:1000]
        try:
            result['git_after'] = git_state()
        except Exception as error:
            result['preservation']['git_content'] = f'{type(error).__name__}: {error}'[:1000]
        if any(v is not True for v in result['preservation'].values()):
            result['status'] = 'failed'
        if not finalize_result(out, result) and 'error' not in result:
            raise RuntimeError('preservation_failed')


if __name__ == '__main__':
    main()
