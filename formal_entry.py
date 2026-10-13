"""Locator-only formal bootstrap. Running it is not Step 5/Gate A acceptance.

Validation and owner binding finish before importing train in this process.
Import/JIT and preparation may have general runtime/cache effects but cannot
create the formal output. Output claim is exclusive, not atomic publication.
"""
import argparse
import json
import os
import stat
import sys

import formal_config_json as p2
import formal_config_p6 as p6
from formal_inputs import verify_inputs
from formal_run_diagnostic import RunDiagnostic


class OutputClaim:
    def __init__(self, inputs):
        paths = p6.preflight_p6_paths(p2.parse_json_bytes(inputs.config_bytes))
        if paths.source_path != inputs.config.paths.source_path or paths.output_directory != inputs.config.paths.output_directory:
            raise p2.JSONInputError('claim_path_changed')
        self.path = paths.output_directory
        self.fd = None
        # Existing P6 permits missing parents. Create them only at claim stage,
        # walking canonical components and refusing symlink substitution.
        parts = self.path.strip('/').split('/')
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in parts[:-1]:
                try:
                    os.mkdir(component, dir_fd=parent)
                except FileExistsError:
                    pass
                next_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent)
                parent = next_fd
            os.mkdir(parts[-1], dir_fd=parent)  # No reuse, retry, or cleanup.
            self.fd = os.open(parts[-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            s = os.fstat(self.fd)
            self.identity = (s.st_dev, s.st_ino)
            self.check()
        except BaseException:
            self.close()
            raise
        finally:
            os.close(parent)

    def check(self):
        s = os.lstat(self.path)
        if self.fd is None or not stat.S_ISDIR(s.st_mode) or (s.st_dev, s.st_ino) != self.identity:
            raise p2.JSONInputError('claim_identity')

    def open(self, name, mode):
        self.check()
        if name not in ('cfg_args', 'input.ply', 'cameras.json') or mode not in ('w', 'wb'):
            raise p2.JSONInputError('claim_writer')
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o666, dir_fd=self.fd)
        return os.fdopen(fd, mode)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def run(locator):
    record = RunDiagnostic()
    claim = writer = None
    close_attempted = False
    claim_close_attempted = False
    try:
        with open(locator, 'rb') as stream:
            data = stream.read(p2.MAX_INPUT_BYTES + 1)
        inputs = verify_inputs(data)
        from formal_checkpoint_identity import source_snapshot
        source = source_snapshot()  # Still stdlib, before cache or output writes.
        record.stage = 'bootstrap'
        from formal_native_bootstrap import bootstrap
        train, native = bootstrap(source)
        record.stage = 'preparation'
        prepared = train.prepare_formal_runtime(inputs, native_evidence=native)
        record = prepared.health_run
        record.stage = 'claim'
        claim = OutputClaim(inputs)
        record.stage = 'begin'
        writer = train.begin_formal_output(prepared, claim)
        result = train.continue_formal_training(prepared, writer)
        record.finish_updates(inputs.config, result)
        if writer is not None:
            record.stage = 'writer_flush'
            writer.flush()
            record.stage = 'writer_close'
            close_attempted = True
            writer.close()
        record.stage = 'claim_close'
        claim_close_attempted = True
        claim.close()
        claim = None
        record.stage = 'normal_return'
        return result
    except BaseException as error:
        record.failure(error)
        # Each cleanup is attempted at most once. A secondary failure cannot
        # replace the original traceback or imply a successful run.
        if writer is not None and not close_attempted:
            try:
                writer.close()
            except BaseException:
                pass
        if claim is not None and not claim_close_attempted:
            try:
                claim.close()
            except BaseException:
                pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('config', help='formal JSON locator (no semantic CLI overrides)')
    args = parser.parse_args(argv)
    try:
        run(args.config)
    except p2.JSONInputError as error:
        print(json.dumps({'error': error.code}), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
