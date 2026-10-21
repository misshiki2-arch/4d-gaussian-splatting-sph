"""Comparison dataset binding, separate from the formal JSON authority.

Reuse the approved dataset reference, byte acquisition, PNG/camera/time checks.
Initializer construction is a heavy, separately checked boundary; no Torch is
imported while acquiring this metadata. A bundle is not an execution permit.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path

import formal_camera
import formal_inputs
import formal_frame_metadata as metadata
import formal_config_p6 as paths
from formal_checkpoint_identity import require
from formal_time_handoff import validate_time
from comparison_profile import stage_config, validate_config

OUTPUT_OWNER = Path('/home/demo/work/reports/corrected-4dgs/phase3/step18')


@dataclass(frozen=True)
class ComparisonInputs:
    config: object
    train_bytes: bytes
    test_bytes: bytes
    train_frames: tuple
    test_frames: tuple
    ply_bytes: bytes
    train_cameras: tuple
    test_cameras: tuple
    reference_bytes: bytes
    initializer: object


def output_path(output):
    dest, occupied = paths._canonical_directory(paths._path_string(output), allow_missing=True, output=True)
    require(not occupied and dest == output, 'comparison_output')
    # New outputs may live under reports but not under any old task/artifact,
    # either research source, the data root, or the immutable legacy output.
    require(OUTPUT_OWNER in Path(dest).parents, 'comparison_output_owner')
    source = formal_inputs._reference()['source_path']
    require(not paths._overlap(dest, source) and not paths._overlap(dest, paths._LEGACY_OUTPUT),
            'comparison_output_overlap')
    return dest


def acquire(stage, output, initializer):
    c = stage_config(stage, output_path(output))
    return _acquire_dataset(c, initializer)


def _acquire_dataset(c, initializer):
    ref = formal_inputs._reference()
    fd = os.open(c.paths.source_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        data = {name: metadata._read_file(fd, name, metadata.FileIdentity(**identity))
                for name, identity in ref['files'].items()}
    finally:
        os.close(fd)
    frames, cameras = {}, {}
    for split in ('train', 'test'):
        raw = json.loads(data['transforms_'+split+'.json'])
        approved = ref['frames'][split]
        require(len(raw['frames']) == len(approved), 'comparison_frames')
        frames[split], cameras[split] = [], []
        for i, (frame, expected) in enumerate(zip(raw['frames'], approved)):
            path, time, mask = expected
            require(frame['file_path'] == path and frame['time'] == time and
                    Path(path).is_absolute() is False and '..' not in Path(path).parts and
                    path.startswith('images/') and mask == path.replace('images/', 'masks/', 1),
                    'comparison_frame_identity')
            require(type(time) in (int, float) and 0 <= time <= 33, 'comparison_frame_time')
            frames[split].append(formal_inputs.Frame(split, i, path, float(time), float(time)))
            camera = formal_camera.from_metadata(raw, frame, c.dataset.resolution, (split, i, path))
            cameras[split].append(camera)
            for name, color in ((path, 6), (mask, 0)):
                require(formal_inputs.png_header(Path(c.paths.source_path)/name) ==
                        (camera.raw.width, camera.raw.height, color), 'comparison_image_dimensions')
    validate_time(c.time_derivation, frames['train']+frames['test'])
    return ComparisonInputs(c, data['transforms_train.json'], data['transforms_test.json'],
        tuple(frames['train']), tuple(frames['test']), data['points3d.ply'],
        tuple(cameras['train']), tuple(cameras['test']),
        json.dumps(ref, sort_keys=True, separators=(',', ':')).encode(), initializer)


def validate_inputs(inputs):
    require(type(inputs) is ComparisonInputs, 'comparison_inputs')
    validate_config(inputs.config)
    from staged_comparison_state import Initializer
    require(type(inputs.initializer) is Initializer, 'comparison_initializer')
    validate_time(inputs.config.time_derivation, inputs.train_frames+inputs.test_frames)
