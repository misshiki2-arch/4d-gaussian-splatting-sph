"""Fixed dataset-owner binding; stdlib only, no writers or runtime imports.

formal_dataset_reference.json is the limited projection of approved #32/#50
evidence. It is not a registry, a run setting, or an auto-updatable manifest.
Images remain lazy: header checks establish dimensions, not pixel provenance.
"""
import json
import os
from pathlib import Path
import stat
import struct
from typing import NamedTuple
import zlib

import formal_config as config
import formal_config_json as p2
import formal_frame_metadata as metadata
import formal_camera as camera
from formal_time_handoff import validate_time


class Frame(NamedTuple):
    split: str
    index: int
    file_path: str
    raw_time: float
    effective_time: float


class Inputs(NamedTuple):
    config: config.FormalConfig
    config_bytes: bytes
    train_bytes: bytes
    test_bytes: bytes
    train_frames: tuple
    test_frames: tuple
    ply_bytes: bytes
    train_cameras: tuple
    test_cameras: tuple


def _reference():
    # No caller-supplied locator or fallback to current dataset fingerprints.
    return json.loads(Path(__file__).with_name('formal_dataset_reference.json').read_bytes())


def png_header(path):
    """Read one PNG IHDR, allowing existing regular-file symlink resolution."""
    try:
        before = os.stat(path)
        if not stat.S_ISREG(before.st_mode):
            raise p2.JSONInputError('image_not_regular')
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            opened = os.fstat(fd)
            data = os.read(fd, 33)
            if any(metadata._file_stamp(s) != metadata._file_stamp(before)
                   for s in (opened, os.fstat(fd), os.stat(path))):
                raise p2.JSONInputError('image_changed')
        finally:
            os.close(fd)
    except OSError:
        raise p2.JSONInputError('image_io') from None
    if (len(data) != 33 or data[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
            or zlib.crc32(data[12:29]) != struct.unpack('>I', data[29:33])[0]):
        raise p2.JSONInputError('image_header')
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', data[16:29])
    if not width or not height or depth != 8 or compression or filtering or interlace not in (0, 1):
        raise p2.JSONInputError('image_header')
    return width, height, color


def verify_inputs(data):
    ref = _reference()
    state = config.resolve_formal_config(data, raw_width=ref['width'], raw_height=ref['height'])
    if state.paths.source_path != ref['source_path']:
        raise p2.JSONInputError('dataset_owner')
    # First-baseline constraints, not new generic resolver numeric domains.
    if state.initialization.time_variance_denominator != 5.0:
        raise p2.JSONInputError('baseline_time_variance')
    if state.optimization.loss.lambda_rigid != 0 or state.optimization.loss.lambda_motion != 0:
        raise p2.JSONInputError('baseline_loss_branch')
    if any(os.environ.get(name) for name in (
            'STEP90_CUDA_DEBUG_PIXEL', 'STEP90_CUDA_DEBUG_MAX_ENTRIES',
            'STEP90_CUDA_DEBUG_PREPROCESS_INDEX')):
        raise p2.JSONInputError('formal_debug_override')
    reference = metadata.MetadataReference(
        metadata.FileIdentity(**ref['files']['transforms_train.json']),
        metadata.FileIdentity(**ref['files']['transforms_test.json']),
        {s: [dict(file_path=f[0], time=p2.NumberToken(str(f[1])))
             for f in ref['frames'][s]] for s in ('train', 'test')})
    train, test, times = metadata.read_frame_snapshot(p2.parse_json_bytes(data), reference)
    validate_time(state.time_derivation, times.train + times.test)
    resolution = state.dataset.resolution
    cameras = {}
    for split, contents in (('train', train), ('test', test)):
        root = json.loads(contents)
        cameras[split] = []
        for index, (frame, approved) in enumerate(zip(root['frames'], ref['frames'][split])):
            verified_camera = camera.from_metadata(root, frame, resolution, (split, index, approved[0]))
            cameras[split].append(verified_camera)
            width, height = verified_camera.raw.width, verified_camera.raw.height
            path, _, mask = approved
            if (Path(path).is_absolute() or '..' in Path(path).parts or
                    not path.startswith('images/') or mask != path.replace('images/', 'masks/', 1)):
                raise p2.JSONInputError('dataset_reference_path')
            if png_header(os.path.join(state.paths.source_path, path)) != (width, height, 6):
                raise p2.JSONInputError('image_dimensions')
            if png_header(os.path.join(state.paths.source_path, mask)) != (resolution.width, resolution.height, 0):
                raise p2.JSONInputError('mask_resolution')
    fd = os.open(state.paths.source_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        ply = metadata._read_file(fd, 'points3d.ply', metadata.FileIdentity(**ref['files']['points3d.ply']))
    finally:
        os.close(fd)
    # Header identity is already hash-bound; check required time-bearing format
    # before heavy import, then let the existing PlyData parser check the body.
    header = ply[:4096].split(b'end_header\n', 1)
    if (len(header) != 2 or not header[0].startswith(b'ply\nformat binary_little_endian 1.0\n')
            or b'property float time\n' not in header[0]):
        raise p2.JSONInputError('ply_header')
    freeze = lambda frames: tuple(Frame(f.split, f.index, f.file_path, f.raw_time, f.effective_time) for f in frames)
    return Inputs(state, data, train, test, freeze(times.train), freeze(times.test), ply,
                  tuple(cameras['train']), tuple(cameras['test']))
