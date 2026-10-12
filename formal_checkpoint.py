"""Formal checkpoint stage M, context and exclusive file publication.

No Torch import until metadata is accepted. No training/P6 bypass, resume,
legacy conversion, recovery, overwrite or whole-run transaction.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path

import formal_config as config
import formal_config_json as p2
import formal_config_p6 as p6
from formal_checkpoint_identity import (SCHEMA, NativeEvidence, byte_identity,
    canonical, checked_file, file_identity, keys, open_directory, projection,
    require, stream_identity, strict_json, validate_identity, validate_native,
    validate_optimizer_descriptor)

MAX_METADATA = 8 * 1024**2
MAX_STATE = 8 * 1024**3  # Local I/O budget, not a population policy.


def _dataset(inputs):
    return dict(reference=strict_json(inputs.reference_bytes),
        files={n: byte_identity(b) for n, b in (
            ('transforms_train.json', inputs.train_bytes), ('transforms_test.json', inputs.test_bytes),
            ('points3d.ply', inputs.ply_bytes))},
        frames=projection((inputs.train_frames, inputs.test_frames)),
        cameras=projection((inputs.train_cameras, inputs.test_cameras)))


@dataclass(frozen=True)
class CheckpointContext:
    inputs: object
    native: NativeEvidence
    base: bytes


def prepare_context(inputs, native):
    e = validate_native(native, live=True)
    require(not any(p6._overlap(e['build_root'], p) for p in inputs.config.paths), 'build_dataset_overlap')
    base = dict(raw_config=inputs.config_bytes.decode('utf-8'),
        config_identity=byte_identity(inputs.config_bytes), effective=projection(inputs.config),
        dataset=_dataset(inputs), native=e)
    require(len(canonical(base)) <= MAX_METADATA, 'metadata_budget')
    return CheckpointContext(inputs, native, canonical(base))


def _diagnostic_config(base, output_root):
    """Pure re-derivation then separate read-only existing-root validation."""
    keys(base, 'raw_config config_identity effective dataset native')
    require(type(base['raw_config']) is str, 'raw_config')
    raw = base['raw_config'].encode('utf-8')
    validate_identity(base['config_identity'])
    require(byte_identity(raw) == base['config_identity'], 'config_identity')
    dataset = base['dataset']; keys(dataset, 'reference files frames cameras')
    import formal_inputs
    reference = formal_inputs._reference()
    require(canonical(dataset['reference']) == canonical(reference), 'dataset_reference')
    c = config._derive_formal_config(raw, raw_width=reference['width'], raw_height=reference['height'])
    source, _ = p6._canonical_directory(p6._path_string(c.dataset.source_path), allow_missing=False)
    dest, _ = p6._canonical_directory(p6._path_string(c.output.directory), allow_missing=False)
    legacy, _ = p6._canonical_directory(p6._LEGACY_OUTPUT, allow_missing=True)
    require(dest == str(output_root) and not p6._overlap(source, dest) and
            not p6._overlap(legacy, dest), 'read_root')
    c = c._replace(paths=config.CanonicalPaths(source, dest))
    require(canonical(projection(c)) == canonical(base['effective']), 'effective_config')
    require(source == reference['source_path'], 'dataset_owner')
    require(dataset['files'] == reference['files'], 'dataset_identity')
    retained = {}
    for name, identity in dataset['files'].items():
        validate_identity(identity)
        require(name in ('transforms_train.json', 'transforms_test.json', 'points3d.ply') and
                file_identity(Path(source)/name) == identity, 'dataset_changed')
        if name != 'points3d.ply':
            fd = open_directory(source)
            try:
                stream, before = checked_file(fd, name, 64*1024**2)
                with stream:
                    require(stream_identity(stream, before) == identity, 'metadata_changed')
                    retained[name] = stream.read()
                    require(byte_identity(retained[name]) == identity, 'metadata_read')
            finally: os.close(fd)
    # Reconstruct approved frame/camera projection from hash-bound metadata,
    # without the new-training resolver, pixel reads, CUDA or a writer.
    import formal_camera
    frames, cameras = [], []
    for split in ('train', 'test'):
        data = json.loads(retained['transforms_'+split+'.json'])
        expected = reference['frames'][split]
        require(len(data['frames']) == len(expected), 'frame_count')
        fs, cs = [], []
        for i, (frame, approved) in enumerate(zip(data['frames'], expected)):
            require(frame['file_path'] == approved[0] and frame['time'] == approved[1], 'frame_identity')
            fs.append(formal_inputs.Frame(split, i, approved[0], float(approved[1]), float(approved[1])/c.time_derivation.divisor))
            cs.append(formal_camera.from_metadata(data, frame, c.dataset.resolution, (split, i, approved[0])))
        frames.append(fs); cameras.append(cs)
    require(canonical(projection(frames)) == canonical(dataset['frames']) and
            canonical(projection(cameras)) == canonical(dataset['cameras']), 'frame_camera_projection')
    validate_native(NativeEvidence(canonical(base['native'])))
    return c


def validate_metadata(metadata, output_root):
    keys(metadata, 'schema layout boundary k n final rows base optimizer')
    require(metadata['schema'] == SCHEMA and type(metadata['layout']) is int and metadata['layout'] == 1
            and metadata['boundary'] == 'completed-after-topology-reset', 'version_boundary')
    c = _diagnostic_config(metadata['base'], output_root)
    k, n, m = metadata['k'], metadata['n'], metadata['rows']
    require(type(k) is int and type(n) is int and type(m) is int and m > 0 and
            1 <= k <= n == c.optimization.total_updates and k in c.save_updates and
            type(metadata['final']) is bool and metadata['final'] == (k == n), 'completed_label')
    # 161 parameter floats + 4 statistics + two full parameter moments per row.
    # Finite I/O budget only, never reinterpret the population stop threshold.
    require(m * (161 * 3 + 4) * 4 + 9 * 8 <= MAX_STATE, 'tensor_budget')
    validate_optimizer_descriptor(metadata['optimizer'])
    return c


def _read_pair(fd, root, marker=None):
    with ExitStack() as stack:
        streams = {}
        for name, limit in (('metadata.json', MAX_METADATA), ('state.pt', MAX_STATE)):
            stream, before = checked_file(fd, name, limit); stack.enter_context(stream)
            identity = stream_identity(stream, before)
            if marker is not None:
                require(identity == marker['files'][name], 'component_hash')
            streams[name] = stream, before, identity
        metadata = strict_json(streams['metadata.json'][0].read(), MAX_METADATA)
        c = validate_metadata(metadata, root)  # M ends here; still no Torch.
        if marker is not None:
            require(type(marker['k']) is int and marker['k'] == metadata['k'], 'marker_label')
        from formal_checkpoint_state import load_state
        payload = load_state(streams['state.pt'][0], metadata, c)
        for stream, before, identity in streams.values():
            require(stream_identity(stream, before) == identity, 'readback_changed')
        return metadata, payload, c, {n: s[2] for n, s in streams.items()}


def read_checkpoint(directory):
    directory = Path(directory)
    require(directory.parent.name == 'checkpoints', 'checkpoint_directory')
    root = directory.parent.parent
    fd = open_directory(str(directory))
    try:
        stream, before = checked_file(fd, 'complete.json', 4096)
        with stream:
            identity = stream_identity(stream, before)
            marker = strict_json(stream.read(), 4096)
        keys(marker, 'schema k files')
        require(marker['schema'] == SCHEMA, 'marker_schema')
        keys(marker['files'], 'metadata.json state.pt')
        for v in marker['files'].values(): validate_identity(v)
        require(directory.name == 'completed-'+str(marker['k']), 'directory_label')
        metadata, state, c, _ = _read_pair(fd, root, marker)
        return dict(metadata=metadata, state=state, config=c, marker_identity=identity)
    finally:
        os.close(fd)


def _write(fd, name, writer):
    raw = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(raw, 'wb') as stream:
        writer(stream)
        stream.flush(); os.fsync(stream.fileno())


class CheckpointWriter:
    def __init__(self, context, claim, scene):
        require(type(context) is CheckpointContext and scene._formal_inputs is context.inputs and
                context.inputs.config.paths.output_directory == claim.path, 'context_claim')
        claim.check()
        validate_native(context.native, live=True)
        self.context, self.claim, self.scene = context, claim, scene
        self.failed = False

    def save(self, k):
        require(not self.failed, 'writer_failed')
        self.failed = True  # Any uncertain outcome is terminal for this writer.
        self.claim.check()
        validate_native(self.context.native, live=True)
        from formal_checkpoint_state import snapshot, equal_state
        state, descriptor = snapshot(self.scene.gaussians, self.context.inputs.config, k)
        c = self.context.inputs.config
        meta = dict(schema=SCHEMA, layout=1, boundary='completed-after-topology-reset',
            k=k, n=c.optimization.total_updates, final=k == c.optimization.total_updates,
            rows=len(state['capture'][1]), base=strict_json(self.context.base), optimizer=descriptor)
        encoded = canonical(meta); require(len(encoded) <= MAX_METADATA, 'metadata_budget')
        parent = child = None
        try:
            try: os.mkdir('checkpoints', mode=0o700, dir_fd=self.claim.fd)
            except FileExistsError: pass
            parent = os.open('checkpoints', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.claim.fd)
            name = 'completed-'+str(k)
            os.mkdir(name, mode=0o700, dir_fd=parent)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            _write(child, 'metadata.json', lambda f: f.write(encoded))
            import torch
            _write(child, 'state.pt', lambda f: torch.save(state, f))
            read_meta, read_state, _, files = _read_pair(child, Path(self.claim.path))
            require(canonical(read_meta) == encoded, 'metadata_readback')
            equal_state(state, read_state)
            marker = canonical(dict(schema=SCHEMA, k=k, files=files))
            _write(child, '.complete.pending', lambda f: f.write(marker))
            stream, before = checked_file(child, '.complete.pending', 4096)
            with stream:
                require(stream_identity(stream, before) == byte_identity(marker) and stream.read() == marker, 'marker_readback')
            self.claim.check()
            os.link('.complete.pending', 'complete.json', src_dir_fd=child, dst_dir_fd=child, follow_symlinks=False)
            os.fsync(child); os.fsync(parent); os.fsync(self.claim.fd)
            result = read_checkpoint(Path(self.claim.path)/'checkpoints'/name)
            equal_state(state, result['state'])
            require(result['marker_identity'] == byte_identity(marker), 'published_identity')
            os.unlink('.complete.pending', dir_fd=child)
            self.failed = False
            return result['marker_identity']
        finally:
            if child is not None: os.close(child)
            if parent is not None: os.close(parent)
