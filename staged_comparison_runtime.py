"""Heavy comparison connection. No alternate trainer, resume, or build fallback."""
import os
from pathlib import Path

from formal_entry import OutputClaim
from formal_checkpoint_identity import open_directory
from comparison_inputs import output_path


class ComparisonClaim(OutputClaim):
    def __init__(self, inputs):
        self.path=output_path(inputs.config.paths.output_directory)
        self.fd=None
        path=Path(self.path)
        parent=open_directory(str(path.parent))
        try:
            os.mkdir(path.name,mode=0o700,dir_fd=parent)
            self.fd=os.open(path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            s=os.fstat(self.fd);self.identity=(s.st_dev,s.st_ino)
            self.check()
        except BaseException:
            self.close();raise
        finally:os.close(parent)


def begin_output(prepared,claim,provenance,parent):
    from staged_comparison_checkpoint import prepare_context,CheckpointWriter
    import train
    context=prepare_context(prepared.inputs,provenance,parent)
    prepared.scene._checkpoint_writer=CheckpointWriter(context,claim,prepared.scene)
    return train.prepare_output_and_logger(prepared.dataset,claim=claim,scene=prepared.scene)
