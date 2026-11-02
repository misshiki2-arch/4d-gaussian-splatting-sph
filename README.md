# Corrected 4DGS for SPH

This repository contains the corrected SPH-oriented 4D Gaussian Splatting
training and CUDA Reference code. The `main` branch carries the accepted
Corrected implementation; it supersedes the earlier SPH code as the standard
code version for this project.

This is research code with a bounded, accepted baseline, not a general-purpose
training package validated for arbitrary inputs.

## Status and accepted scope

Corrected 4DGS is complete within the adopted scope recorded in the
[canonical plan](docs/corrected_camera_intrinsics_retraining_plan.md#adopted-scope-completion).
That scope includes the necessary camera/renderer/training corrections,
focused CPU/CUDA validation, the original-A versus corrected-B training
comparison, and adoption of the saved B E3 final.

- The adopted B baseline follows the successful four-stage process:
  stabilization (12,000 updates), growth (30,000), E2 (12,000), and E3 (12,000).
  After stabilization, each stage inherits its own preceding confirmed final;
  optimizer/statistics/schedule state restarts at each stage. It is not a single 66,000-update run.
- The accepted E3 final contains 3,248,901 Gaussians. Its active SH degrees
  are spatial/time 0/0; maximum-degree labels 3/2 do not mean active
  high-order training.
- The accepted output is the all-point SPL4-v2 bundle and 14-view CUDA
  Reference set. Gates A/B/C are accepted within their recorded scopes.
- The accepted Viewer handoff is this bundle's Gate D **input acceptance**.
  Further Viewer rendering, performance, or feature work belongs to the
  [separate Viewer project](https://github.com/misshiki2-arch/4dgs-viewer)
  and does not reopen Corrected completion.

See [baseline adoption and profile scope](docs/corrected_camera_intrinsics_retraining_plan.md#current-baseline-adoption-and-profile-scope)
for the conditions, evidence, and distinction from the older formal profile.

## Code and evidence navigation

| Purpose | Entry or reference |
| --- | --- |
| Adopted conditions, completed work, and limitations | [Canonical plan](docs/corrected_camera_intrinsics_retraining_plan.md) |
| Fixed four-stage comparison orchestration | [staged_comparison_entry.py](staged_comparison_entry.py) and [comparison_profile.py](comparison_profile.py) |
| Selected saved-final continuation used by the accepted run | [staged_comparison_followup.py](staged_comparison_followup.py) |
| Comparison-domain checkpoint reader and completion checks | [staged_comparison_checkpoint.py](staged_comparison_checkpoint.py) and [staged_comparison_results.py](staged_comparison_results.py) |
| Adopted B to SPL4/CUDA Reference export | [tools/export_adopted_reference.py](tools/export_adopted_reference.py) |
| Shared training implementation and legacy YAML CLI | [train.py](train.py) |
| Older formal-profile entry (a distinct technical contract) | [formal_entry.py](formal_entry.py) |
| Regression tests | [tests/](tests/) |

The accepted runs used the local `4dgs310` environment and recorded native
extension binaries. The inherited [environment.yml](environment.yml) is not
a specification of that validated environment. Simply running the legacy
`train.py --config ...` CLI or the old formal-profile entry is not a
reproduction of the adopted four-stage B process.

The orchestration and export entries bind specific source, environment,
input, and saved-evidence identities. They refer to the original local
workspace and are not portable, ready-to-run commands from a fresh clone.
A new run needs its own reviewed inputs and output location; old approved
run files must not be reused to overwrite published results.

## Data availability and limitations

Datasets, the initial state, learned checkpoints, SPL4/Reference payloads,
native build caches, and the full execution reports are **not bundled in
this Git repository**. Some links in the canonical plan point outside the
repository to the local research workspace; those links will not resolve
from a standalone GitHub clone. The tracked code and plan alone are not a
complete downloadable reproduction package.

B improved on A in the accepted comparison, but local blur and differences
from GT remain. The result is not a claim of GT equivalence, full-paper or
high-order-SH reproduction, or isolation of every individual fix's effect.
Exclusion of test information from the original 3DGS-derived initialization
has not been established. Unused features and arbitrary conditions are not
covered by the adopted acceptance.

## Preserved original version

The pre-integration SPH/A version is preserved at
[`original-sph-before-corrected`](https://github.com/misshiki2-arch/4d-gaussian-splatting-sph/tree/original-sph-before-corrected).
The earlier
[`legacy-camera-raster-baseline-20260901`](https://github.com/misshiki2-arch/4d-gaussian-splatting-sph/tree/legacy-camera-raster-baseline-20260901)
tag is also retained; it identifies an earlier historical snapshot.

The `research/corrected-training-baseline-v1` branch retains the accepted
Corrected development history. Promoting that history to `main` and updating
this README do not regenerate or alter any saved research artifact.

## Upstream and licenses

The configured upstream is
[Fudan Zhang Vision Group's 4D Gaussian Splatting implementation](https://github.com/fudan-zvg/4d-gaussian-splatting).
The repository retains [LICENSE](LICENSE) and
[LICENSE_gaussian_splatting.md](LICENSE_gaussian_splatting.md), along with
component notices. Refer to those files and the upstream project for the
applicable licenses and original research attribution.
