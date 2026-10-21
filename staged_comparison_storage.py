"""B producers' additional disk demand; no relocation, deletion or save reduction.

Published checkpoint markers link the private marker inode, and comparison
publication renames its directory on the same filesystem: neither duplicates
the payload. Already saved checkpoints/images/expected are not new demand.
"""
from pathlib import Path
from types import FunctionType

from formal_checkpoint_identity import require
import staged_comparison_support as support

RECORD = support.RESOURCES['record_allowance_bytes']


def serialized(size):
    require(type(size) is int and size >= 0, 'storage_payload')
    return size + (size + 99) // 100 + 1024**2


def png(count, width, height, panels=1):
    require(all(type(v) is int and v > 0 for v in (width, height, panels))
            and type(count) is int and count >= 0, 'storage_image_shape')
    scanlines = 3 * width * height * panels + height
    return count * (scanlines + (scanlines + 99) // 100 + 64 * 1024)


def collection(kind, n, *, width=1280, height=720, fixed=14, tests=166,
               tensorboard=True, initial_bytes=0):
    p = support.payload(n)
    require(type(fixed) is int and fixed == 14 and type(tests) is int and tests >= 0,
            'storage_population')
    require(type(initial_bytes) is int and initial_bytes >= 0, 'storage_initial')
    # training_report always selects five train positions (even if repeated),
    # and saves the first five test positions. Each produces four RGB PNGs.
    ordinary = 5 + min(5, tests)
    raw = fixed * (20 * width * height + 5 * n)
    amounts = dict(
        initial=initial_bytes + RECORD,
        ordinary=png(4 * ordinary, width, height) +
                 (png(ordinary, 2 * width + 6, 2 * height + 6) if tensorboard else 0) + RECORD,
        fixed=png(4 * fixed, width, height) + RECORD,
        expected=serialized(p['capture'] + raw) + RECORD,
        checkpoint=serialized(p['capture']) + RECORD,
        comparison=png(3 * fixed * 2, width, height, 3) + RECORD,
        record=RECORD)
    # Three evaluation/fixed boundaries; two checkpoints; final expected once.
    # A planning estimate at current n, never permission for future writes.
    if kind in ('stage_initial', 'stage_capture'):
        parts = dict(initial=amounts['initial'], ordinary=3 * amounts['ordinary'],
                     fixed=3 * amounts['fixed'], expected=amounts['expected'],
                     checkpoint=2 * amounts['checkpoint'], comparison=amounts['comparison'])
    else:
        require(kind in amounts, 'disk_boundary')
        parts = {kind: amounts[kind]}
    return dict(components=parts, bytes=sum(parts.values()),
                pngs_per_fixed=4 * fixed, pngs_per_ordinary=4 * ordinary,
                comparison_pngs=3 * fixed * 2, tensorboard=tensorboard,
                scope='additional writes only; private/public payload is not duplicated')


def disk_need(kind, n):
    return collection(kind, n)['bytes']


def check(directory, kind, n, **shape):
    """Reuse the existing predicate/journal against the actual target filesystem."""
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'storage_directory')
    demand = collection(kind, n, **shape)
    original = support.resources().disk
    emit = original.__globals__['emit']
    filesystem = dict(path=str(directory.resolve()), device=directory.stat().st_dev)
    fn = FunctionType(original.__code__, dict(original.__globals__,
        disk_need=lambda *_: demand['bytes'],
        emit=lambda where,event:emit(where,dict(event,filesystem=filesystem,collection=demand))),
        original.__name__, original.__defaults__)
    return fn(directory, kind, n)
