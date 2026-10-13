"""Small run-local record. Stdlib only; not a public manifest or retry log."""
from dataclasses import asdict, dataclass
import json
import sys


@dataclass
class RunDiagnostic:
    stage: str = 'preflight'
    attempted: int = 0
    completed: int = 0
    save_attempt: int = 0
    save_outcome: str = 'not_attempted'
    last_confirmed_save: int = 0
    saves: int = 0
    tests: int = 0

    def finish_updates(self, config, returned):
        if not (returned == self.completed == config.optimization.total_updates and
                self.last_confirmed_save == returned and self.saves == len(config.save_updates) and
                self.tests == len(config.reporting.test_updates)):
            raise RuntimeError('formal_run_incomplete')

    def failure(self, error):
        # Do not stringify the exception or tensors/locals. Broken stderr must
        # not replace the primary exception, and recording is not guaranteed.
        try:
            print(json.dumps(dict(formal_failure=asdict(self), error_type=type(error).__name__)), file=sys.stderr)
        except BaseException:
            pass
