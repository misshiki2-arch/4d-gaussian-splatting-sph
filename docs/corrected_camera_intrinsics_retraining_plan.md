# Corrected Camera Intrinsics Retraining Plan

Status: **plan-only / Investigation1-4 static audit complete / audit integration documented / four-formal-policy-groups-approved / remaining-formal-policy-open / source-fixes-not-started / pilot-not-started / formal-retraining-not-started / Viewer-frozen**

This document records the approved transition from the historical split
camera/raster baseline toward a corrected Fudan Native 4DGS baseline that will
be numerically validated and retrained from scratch. It does not implement a
camera, renderer, or CUDA fix; run a diagnostic render; start training; export
assets; generate a CUDA Reference; or authorize Git operations.

## Purpose and decision

The project will not reproduce the historical negative-FoV-sentinel footprint
semantics in the WebGPU Viewer. The shared 4DGS renderer camera-to-rasterizer
handoff will instead be corrected, validated, and used for from-scratch
retraining. The resulting checkpoint, SPL4, CUDA Reference, and Viewer
comparison baseline will have new identities and non-overwriting output paths.

The formal training-provenance classification is
`D: training-provenance-insufficient`: saved artifacts cannot prove every
renderer byte and runtime value used at every training iteration. The source
lineage and artifact timing nevertheless strongly indicate that the legacy
checkpoint was optimized through the historical split camera/raster semantics.
Because renderer RGB/alpha affects loss and gradients, and visibility/radii
affect densification plus clone/split/prune, a checkpoint is not independent of
the renderer semantics that produced it.

## Audit status and current boundary

The pre-fix source audit is complete. Its purpose was to prevent a local camera
fix from concealing independent renderer, training-state, or artifact-identity
defects. Completion here means static read-only source and bounded historical-
artifact audit, not implementation or post-fix runtime validation.

- Investigation1 is complete for dataset, camera, projection, camera identity,
  camera metadata, and legacy training provenance.
- Investigation2 is complete for renderer and CUDA forward/backward, 4D
  covariance, SH, compositor, and branch consistency.
- Investigation3 is complete for the training loop, optimizer, densification,
  checkpoint, resume, evaluation, best-checkpoint selection, determinism, and
  failure handling.
- Investigation4 is complete for checkpoint consumers, PLY/SPL4 export, SPL4
  parsing, checkpoint-SPL4 provenance, CUDA Reference reconstruction,
  manifest, cross-binding, Viewer handoff, and output ownership/atomicity.
- The Investigation1-4 findings, dependencies, ownership boundaries, and
  pre-implementation gates are integrated in this document.

The Fudan Native model configuration, existing train/test-only dataset and
evaluation policy, checkpoint-foundation/exact-resume staging policy, and
formal camera/effective-`eval` policy were approved by the user on 2026-09-03
JST and are integrated below. Remaining formal policy selection, source fixes,
focused validation, pilot training, formal retraining, corrected artifact
generation, and Viewer restart have not started. P0 findings block only the
gate whose accepted output would reach the defect; Viewer-only defects do not
unnecessarily block corrected training, and training-state defects cannot be
deferred to artifact generation. Camera policy approval and document
synchronization do not close P0-0 or constitute source-fix, focused-validation,
runtime, or scientific acceptance.

## Repository identity

- fork: `misshiki2-arch/4d-gaussian-splatting-sph`
- origin: `git@github.com:misshiki2-arch/4d-gaussian-splatting-sph.git`
- legacy fork HEAD: `7663366f823b0beea9cedf76013840f20f7cd563`
- official upstream: `https://github.com/fudan-zvg/4d-gaussian-splatting`
- official upstream base: `63725f21d4adc29669e565ae10e6b3ad6e0d1250`
- Investigation1-4 audit source identity:
  `921534f0d66cf02868168cb4bed6db09435fbc09`

Git branch creation, commit, and push are user responsibilities. None is part
of this documentation task.

## Immutable legacy dataset and checkpoint identity

Dataset:

- `/home/demo/work/data/4dgs_sph_scene/transforms_train.json`
  - SHA-256: `6c29326bc590774b534f0d334e5fd03f841101cfb23dced5dd18bcc2a13a6068`
  - 5,146 frames
- `/home/demo/work/data/4dgs_sph_scene/transforms_test.json`
  - SHA-256: `c8e3ab9e45fa2e80d885a4610a78c77b8c813cda45c6a1b63c6ca5509e9c81fa`
  - 166 frames
- combined training camera count under `eval=False`: 5,312
- intrinsics: `fl_x = fl_y = 1777.7777777777778`, `cx = 640`,
  `cy = 360`, 1280 x 720

Legacy training output: `/home/demo/work/outputs/sph_scene_4dgs`

- `cfg_args`
  - SHA-256: `7cbfc1a967be13bca1709bd74e475b73283d0babf5f27fd9c515f9fff9be5499`
- `cameras.json`
  - SHA-256: `98193b7ae6b8c0da4b36b757b243c1492edae98cbe937173f5c265a41ee282a3`
- `chkpnt_best.pth`
  - SHA-256: `bd44500c3474ef67cd4fe44f26a2c83b6953e30315d07220769002b61b74dcb4`
  - size: 2,572,356,878 bytes
  - iteration: 12,000
  - record count: 3,231,588
  - mtime: 2026-03-27 13:30:15 JST

The legacy output, checkpoint, SPL4, CUDA References, JSON, PNG, and Viewer
artifacts are immutable historical evidence. They must not be deleted,
renamed, moved, overwritten, or reused as a corrected output directory.

## Approved formal dataset and evaluation policy

The first corrected baseline preserves the existing dataset population rather
than introducing a validation split:

- train is all 5,146 frames in `transforms_train.json`, views `v01-v31`;
- test is all 166 frames in `transforms_test.json`, view `v00`;
- train/test separation is enabled, test frames never enter training, and
  `v31` is not removed from training as a validation view;
- the formal invariant that implements this separation is effective
  `eval=True` after all CLI/config merging is complete;
- effective `eval=False` is forbidden and must fail closed at the formal-run
  entry because it merges test into training, producing 5,312 train cameras
  and zero test cameras;
- no validation population is created;
- test is not used for checkpoint selection, parameter tuning, or early
  stopping; and
- the canonical checkpoint is the completed state at a final iteration fixed
  before the run, not a validation- or test-selected best checkpoint.

The numeric final iteration, training schedule, and test-reporting cadence
remain open. Any future validation-based experiment requires a separate
experiment identity and output owner and is not this formal baseline.

## Confirmed camera handoff defect

### P0-0: camera identity split

For an intrinsics camera, the dataset loader supplies valid
`fl_x/fl_y/cx/cy` and raw `FoVx = FoVy = -1` sentinels. The camera full
projection uses the positive focal intrinsics. The shared renderer historically
computed `tan(FoV / 2)` without first distinguishing the camera mode, so the
footprint rasterizer received:

- `tanFovX = tanFovY = -0.5463024974`
- `focalX = -1171.512085`
- `focalY = -658.975586`

The coherent intrinsics contract for the current dataset is:

- `tanFovX = 0.36`
- `tanFovY = 0.2025`
- `focalX = focalY = 1777.7777777777778`

The split affects camera-space clamping, projection Jacobian, screen
covariance, determinant/conic, radius, tile bounds, and `tilesTouched`.

This camera identity split is **P0-0**. Formal training, render acceptance, and
CUDA Reference generation cannot use a camera contract in which full
projection and footprint rasterization consume different identities.

## Corrected effective-intrinsics contract

The formal baseline supports exactly two explicit input modes:

1. **Intrinsics mode** requires complete `fl_x/fl_y/cx/cy`, finite positive
   `fl_x/fl_y`, finite `cx/cy`, and positive effective width and height. The
   first formal baseline supports only a centered principal point, defined
   mathematically by `cx=width/2` and `cy=height/2`; the implementation
   tolerance remains open. A negative raw FoV sentinel may be retained only as
   provenance and must never become effective FoV or effective tanFov.
2. **FoV-only mode** requires complete geometrically valid input, positive
   effective width and height, finite `FoVx/FoVy`, and
   `0 < FoVx,FoVy < π`. The common camera owner derives every focal, tanFov,
   and projection value required by downstream consumers.

Both modes must yield one canonical effective-camera state consumed by full
projection and by rasterizer forward and backward. The canonical state keeps
raw metadata, raw sentinel, input mode, effective dimensions, effective
`fx/fy/cx/cy`, effective `FoVx/FoVy`, effective `tanFovX/tanFovY`, projection
matrix, projection near/far, and raster visibility-cull semantics distinct.
Raw sentinels or input defaults must not be published as executed effective
state.

The first formal baseline rejects the following before GPU execution:

- off-center principal points;
- simultaneous intrinsics and valid FoV, even when apparently redundant;
- partial intrinsics and every partial-frame fallback to root intrinsics or
  FoV;
- inconsistent frame/root intrinsics and any inconsistent or ambiguous camera
  metadata;
- zero or negative focal values, nonfinite values, invalid dimensions, and
  FoV values that are zero, negative, nonfinite, or at least `π`;
- mismatch between declared JSON dimensions and effective image dimensions;
  and
- invalid projection near/far.

Future support for numerically consistent redundant intrinsics/FoV or for
off-center cameras requires a separate policy, validation boundary, and, when
needed, independent root Fix. It is not part of the first formal baseline.

One common camera contract must own validation and canonicalization after
resolution scaling. Camera projection and rasterizer setup must consume that
same result; manifest publication and tests must not independently reconstruct
it. The exact builder API, class/function/field names, placement, validation
error schema, and centered-camera tolerance remain implementation decisions.

### Approved projection clipping and raster visibility semantics

The first formal baseline preserves the current Fudan-derived semantics as
separate contracts:

- projection near is `0.01`;
- projection far is `100.0`;
- CUDA raster visibility requires `p_view.z > 0.2`; and
- CUDA raster far-cull is disabled/nonexistent.

These values must not be collapsed into one near/far pair. P0-0 must not change
the CUDA threshold from `0.2` to `0.01`, change projection near to `0.2`, or
add a far-cull, because doing so could change the visible Gaussian population.
This policy preserves the first-baseline visibility semantics; it does not
establish that `0.2` is scientifically optimal. A future change is an
independent camera P1 policy and comparison responsibility.

### P0-0 implementation boundary

P0-0 remains open. Its candidate root responsibility is to validate raw
`CameraInfo` or equivalent metadata once, create the canonical effective-camera
state once, and make projection plus rasterizer forward/backward consume it.
This includes explicit mode selection, input validation, raw/effective
separation, post-resolution canonicalization, identity-consistent handoff,
pre-GPU rejection, and focused validation. It excludes manifest schema and
publication, P0-A6 implementation, Viewer/artifact provenance, scientific
retuning of projection/cull values, off-center expansion, and unrelated SH,
training, or checkpoint findings. No part of that source Fix or validation has
been performed by this policy synchronization.

## Confirmed renderer and CUDA P0 blockers

Investigation2 confirmed three additional P0 blockers. The exact correction
policy and implementation remain separate future responsibilities.

### P0-1: 4D spatial SH coefficient-one backward basis

For 4D SH coefficient `sh[1]`, forward uses the basis `-C1 * y`, while the
coefficient gradient in backward uses `C0`. Consequently the update reaching
`_features_rest[:, 0]` is not the derivative of the forward graph.

### P0-2: SH evaluation position mismatch

The geometry path uses the conditional current-time mean. CUDA SH forward uses
the original mean, while SH backward uses the conditional mean. SH-derived
gradients returned to xyz, temporal center, scale, and quaternion parameters
therefore do not differentiate the executed forward graph. The formal policy
selects the **conditional mean** for SH evaluation. Forward and backward must
use the same conditional-mean dependency and gradient path. The original-mean
branch is unreachable for the formal baseline, but the finding remains open
until the source is corrected and focused forward/gradient validation is
accepted.

### P0-3: temporal SH degree, layout, and time-gradient contract

Temporal SH evaluation is nested under spatial degree greater than two. At
lower spatial degrees, model allocation and evaluator use do not agree. In
addition, the derivative of `cos(k * omega * delta_t)` has the wrong sign, and
temporal degree two overwrites rather than adds the degree-one time-gradient
contribution.

The formal Fudan Native branch fixes spatial SH degree at 3, enables temporal
SH at degree 2, and uses the 48-slot fixed layout: slots 0-15 are the spatial
base, 16-31 temporal mode 1, and 32-47 temporal mode 2. It also fixes
`rot_4d=true` and `force_sh_3d=false`. The temporal derivative and accumulation
defects therefore require source correction and focused gradient validation.
Original-mean SH, spatial degree 2, disabled temporal SH, or any other reduced
configuration is unreachable for the formal baseline and must fail closed.
This policy repairs the implementation while preserving the paper-intended
spatiotemporal Fudan Native model; it is not a request to reproduce inconsistent
official-code behavior. Python precompute, alpha-cap derivative, scaling
modifier, and the other unapproved renderer branches remain open.

## Confirmed training-lifecycle P0 blockers

Investigation3 confirmed the following independent training-state blockers.
They retain their original IDs.

### P0-T1: iteration and optimizer off-by-one

The loop increments `iteration` before processing a batch and executes
`optimizer.step()` only while `iteration < opt.iterations`. The first update
number and the final requested iteration therefore do not form an exact
one-update-per-iteration transaction. This is unconditional for the current
training loop and must be fixed before pilot training.

### P0-T2: checkpoint iteration is not a completed-state boundary

Evaluation and checkpoint save occur before densification, opacity reset, and
the optimizer step for the same iteration. A checkpoint labelled with that
iteration does not represent the completed state transition named by the
label. This is unconditional for saved checkpoints on the current path and
must be fixed before pilot training.

### P0-T3: Gaussian gradients are lost on densification iterations

Densification replaces optimizer-owned Gaussian parameter tensors before the
same iteration's optimizer step. The freshly installed parameters do not own
the gradients produced by the just-completed backward pass, so the Gaussian
update is silently lost on those iterations. Densification topology helpers
may still preserve record alignment; the defect is the transaction ordering,
not an SPL4 record-order defect.

### P0-T4: held-out evaluation and best selection under `eval=False`

`eval=False` is a historically source-reachable all-images-training policy,
but it does not provide a non-empty independent held-out population. It becomes
a P0 when that mode is combined with claims of held-out evaluation or with
`chkpnt_best.pth`
selection based on the resulting empty/invalid test metric. This historical
finding remains. The formal baseline instead preserves the declared train and
test files as disjoint populations: train contains all 5,146 `v01-v31` frames,
test contains all 166 `v00` frames, and no validation population is introduced.
This requires effective `eval=True` after CLI/config merge. Effective
`eval=False`, which merges the populations into 5,312 train cameras and zero
test cameras, is unsupported and must fail closed at the formal-run entry.
The test set cannot enter training, best selection, parameter tuning, or early
stopping. The best-checkpoint branch is unsupported for this baseline; the
canonical selection is the completed state at the pre-fixed final iteration.

### P0-T5: resume destroys global-best identity

`best_psnr` and the metric provenance are not restored. This P0 is reached
when resume is permitted: the resumed run starts best tracking from zero and
may overwrite the prior global best with a merely local post-resume result.
Pilot resume is prohibited, and formal resume remains fail-closed unless an
independent exact-resume Fix and its equivalence gate are accepted, so this
branch is currently unreachable. That prohibition does not repair resume
fidelity: P0-T5 remains a conditional finding that must be addressed before
resume support can be enabled. For this no-best formal baseline the best path
must remain fail closed; any future best branch must also restore and verify
its global lineage.

### P0-T6: semantic checkpoint incompatibility is not fail-closed

Checkpoint restoration does not bind or validate all model, renderer,
dataset, and effective-config semantics. This P0 is reached when warm-start,
resume, or any cross-semantics load is permitted. Formal from-scratch startup
avoids warm-start reachability, but versioned semantic validation remains
required for the diagnostic checkpoint foundation, any formal resume, and
later artifact consumers.

### P0-T7: environment-map startup/resume lifecycle mismatch

This P0 is reached only when `env_map_res > 0`. The model/checkpoint may carry
an environment map, while training startup creates and assigns a new parameter
and optimizer on a separate lifecycle; restore does not establish a coherent
parameter/optimizer/schedule continuation. The formal branch must either
define and validate this lifecycle or reject environment maps.

### Approved checkpoint foundation and exact-resume staging

Checkpoint normalization is required independently of whether continuation is
ever enabled. The initial training-lifecycle responsibility must provide a
versioned semantic schema, exact completed-update labels, completed-transaction
save boundaries, diagnostic state, and dataset/config/source/renderer
provenance. Incomplete, unknown-version, or semantically incompatible state
must fail closed. The schema's field-level design remains open.

Pilot runs start from iteration 0, prohibit resume, and reject every resume
entry. Legacy checkpoint warm-start is always prohibited. Exact resume is not
a prerequisite for the known camera/renderer P0 fixes or for a no-resume
formal run. Only after the known P0 fixes and training transaction are stable
may exact resume be implemented as an independent root Fix and ticket.

Before resume can participate in formal training or bug diagnosis, a bounded
resume-equivalence gate must compare uninterrupted continuation past a
completed checkpoint with restoration in a separate process, including loss,
parameters, optimizer state, population, topology events, and RNG/sampler
state. Its numerical or bitwise threshold remains open. Until that gate is
accepted, resume stays fail closed and only an uninterrupted completed run can
be canonical. Once validated, exact resume may diagnose late nonfinite,
densification, clone/split/prune, opacity-reset, point-cap, and loss-divergence
failures. Verified interruption recovery within the same from-scratch run is
not legacy warm-start.

## Confirmed artifact-lifecycle P0 blockers

Investigation4 confirmed the following checkpoint-consumer, publication, and
Viewer-handoff blockers.

### P0-A1: checkpoint internal iteration and CUDA Reference label mismatch

The CUDA Reference loader can select a file using a CLI iteration and then
discard the checkpoint's internal iteration while using the CLI value in
output naming and publication. This is the consumer-side propagation of
P0-T1, P0-T2, P0-T5, and P0-T6, not a second training-loop fix.

### P0-A2: CUDA reconstruction from an incomplete config snapshot

Training writes only the model-parameter subset to `cfg_args`; the CUDA
Reference path fills missing Gaussian dimension, force/rotation/time,
prefilter, and pipeline semantics from CLI values or defaults. Checkpoint
restore does not replace every missing semantic. This propagates P0-T6,
P0-T7, and the incomplete effective-config snapshot into a consumer.

### P0-A3: SPL4 parser accepts malformed or incompatible layouts

The Viewer v2 parser validates magic/version but not the complete payload
length, semantic dimensions, reserved/raw flags, overflow, finite values,
quaternion validity, or scale validity. Truncated data can be zero-filled and
extra bytes ignored. This is an independent Viewer/parser P0 and belongs to the
Viewer-restart gate, not the pilot-training gate.

### P0-A4: population provenance is not cross-bound into Viewer comparison

Checkpoint-SPL4 provenance can exactly validate files and selected bytes, but
the Viewer parser/comparison path does not consume that provenance or a raw
source-index mapping. Historical fixed-range constants can therefore be paired
with a different otherwise-valid asset. This is an independent
artifact/Viewer-handoff P0.

### P0-A5: diagnostic rerender is labelled as the production invocation

Direct-evidence rows are obtained by separate debug rerenders after the PNG
render but are published with `sameProductionRasterizerInvocation=true`.
This is a P0 only if direct evidence is used for formal acceptance. The branch
may instead be explicitly classified as diagnostic-only.

### P0-A6: manifest vocabulary and effective runtime state disagree

Manifest data may come from CLI/default arguments rather than the restored
runtime model, and the shared vocabulary declares quaternion components as
`(x,y,z,w)` although checkpoint, exporter, Viewer evaluator, and CUDA use
`(w,x,y,z)`. The camera portion propagates P0-0; the quaternion vocabulary and
runtime-state publication defects are independent artifact issues.

### P0-A7: source, dataset, and loaded-binary lineage is incomplete

The manifest records repository revision/dirty state and the loaded binary's
file hash, but does not bind transforms/images/masks or a source/build recipe,
toolchain, and source digest to that binary. This is an independent build and
artifact-provenance P0.

### P0-A8: best metric and checkpoint identity are not bound

`chkpnt_best.pth` does not carry the selecting metric, population, renderer
semantics, or global-best lineage. This P0 is reached when that filename is
used as the formal source checkpoint. This historical artifact-selection
propagation of P0-T4 and P0-T5 remains, but the formal baseline makes the
legacy best lineage unreachable. Formal selection must bind the pre-fixed
final iteration, exact completed update count, and dataset, effective config,
source, camera, and renderer semantics. A future validation-selected experiment
would need a separate identity, output owner, and complete metric/population/
best lineage.

## Root and dependent finding relationships

Root fixes and dependent consumer checks must be owned separately so the same
cause is not patched twice:

| Root finding | Dependent finding | Required boundary |
|---|---|---|
| P0-T1/T2/T6 | P0-A1 | Fix the training/checkpoint transaction first; make the loader verify and publish it second. |
| P0-T6/T7 plus incomplete config snapshot | P0-A2 | Define effective config/checkpoint semantics first; reconstruct and assert them in the CUDA loader second. |
| P0-T4 | P0-A8 | Enforce the approved train/test identity, test non-selection, and pre-fixed final completed checkpoint policy first; require that identity at artifact selection second. |
| P0-T5 | P0-A8 when resume and best tracking are enabled | Keep both branches unreachable for the formal baseline; any future validation-selected resume branch requires a separate policy/identity and verified global lineage. |
| P0-0 | camera portion of P0-A6 | Correct one common camera contract first; publish the exact executed values second. |
| none | P0-A3/A4/A5, quaternion vocabulary, source-build binding | Independent parser, evidence, handoff, and provenance responsibilities. |
| P0-T3 | population identity changes | Repair the optimizer/densification transaction; do not alter SPL4 source order as a supposed fix. |

## Renderer and CUDA P1 disposition

The camera P1 policy for the first formal baseline is decided: only centered
cameras are supported; off-center input fails closed; projection near/far stay
`0.01`/`100.0`; CUDA raster visibility keeps the separate
`p_view.z > 0.2` near-cull and has no far-cull. The numerical centered-camera
tolerance and common-builder implementation details remain open. The fixed
values preserve current visibility semantics and are not a claim of scientific
optimality.

The following other confirmed issues still require an explicit disposition
during formal policy selection and before their branch is implemented:

- missing temporal backward propagation when `rot_4d=false`;
- non-equivalent Python-precompute and CUDA-direct branches;
- disagreement between the forward alpha `0.99` clamp and its backward
  derivative;
- missing scale-gradient factors and Python temporal-marginal inconsistency
  when `scaling_modifier != 1`;
- a factor-of-two z-gradient error in projected 2D covariance outside the
  viewport clamp;
- disagreement between SH allocation and evaluator layouts;
- insufficient fail-closed validation for invalid or nonfinite inputs.

An issue needed by the selected formal branch must be corrected and validated.
An unused branch must be rejected explicitly rather than left silently
available. Items such as alpha-cap derivative, Python-precompute, scaling
modifier, and environment-map policy remain open; the camera policy above is
decided but its P0-0 source Fix and focused validation have not started.

## Bounded P2 findings

The following are recorded for later disposition but are not automatically
promoted to formal-retraining blockers:

- the determinant epsilon approximation in inverse-covariance backward;
- conservative radius inflation from the eigenvalue lower bound;
- possible `uint32` prefix-sum and signed `num_rendered` overflow;
- runtime behavior for an empty Gaussian population, which remains unverified;
- unclear combined meaning of `force_sh_3d` and temporal degree outside the
  approved `force_sh_3d=false`, temporal-degree-2 formal configuration;
- a misleading radius-threshold expression.

## Integrated P1/P2 disposition by owner

The following register preserves Investigation1-4 P1/P2 facts while avoiding
repetition of a P0 root cause. A selected formal branch must correct or
fail-closed reject its P1 items. P2 items remain bounded robustness,
observability, determinism, or performance work unless later evidence promotes
them.

| Owner group | P1 disposition | P2 or bounded follow-up |
|---|---|---|
| camera/projection policy | Support only complete centered intrinsics and complete geometrically valid FoV-only input; reject mixed/partial/ambiguous/nonfinite/off-center input; keep raw sentinel separate from canonical effective state; preserve projection `0.01`/`100.0` and CUDA near-cull `0.2` with no far-cull. One common builder owns execution; P0-A6 separately owns later publication. | Define centered tolerance, field-level validation/error schema, and bounded supported-mode fixtures; any future off-center or visibility-semantics change needs separate policy and validation. |
| renderer branch policy | Implement conditional-mean SH, spatial degree 3, temporal degree 2 with the fixed 48-slot layout, `rot_4d=true`, and `force_sh_3d=false`; resolve alpha-cap derivative, scaling modifier, Python covariance/SH precompute, and the remaining unsupported-branch rejection separately. | Determinant epsilon, radius inflation, empty population, overflow, and radius-threshold diagnostics. |
| optimizer/learning-rate policy | Define temporal-position LR scheduling and supported parameter-group behavior; do not infer it from the spatial-only scheduler. | Log effective per-group LR without adding a second schedule owner. |
| densification/population policy | Define temporal selection, strict point-cap semantics, clone/split/prune ordering, prune-only behavior, opacity reset, and screen/world pruning. | Measure cap overshoot, nonfinite accumulator frequency, and memory pressure in bounded pilot runs. |
| evaluation/metric policy | Preserve the approved train/test identity, create no validation population, keep test selection-free, and use the pre-fixed final completed checkpoint. Test-report metric/cadence and clamp/channel/background remain open; any future best branch needs a separate experiment policy. | Keep diagnostic train samples separate from test reporting and visualization. |
| config/run mode/provenance | Require effective `eval=True` after CLI/config merge, reject effective `eval=False`, define the remaining formal run-mode fields and precedence, reject legacy output/path reuse, capture the complete effective config, and replace executable config parsing where it reaches formal tooling. | Report path-remap and duplicate-basename ambiguity as bounded diagnostics. |
| checkpoint foundation | P0-T1/T2/T6 own completed transaction, exact update label, versioned semantic schema, provenance, and diagnostic state independently of resume; atomic writes and field-level state remain to be defined. | Record interruption/OOM behavior and checkpoint-size/hash cost. |
| exact resume (conditional) | After known P0 and transaction stability, a separate root Fix owns complete restore and resume equivalence; until acceptance, resume fails closed. | Compare scheduler/RNG/sampler/dataloader, optimizer, population, and topology continuation only if resume will be enabled. |
| dataset/mask | Bind the approved `v01-v31` train and `v00` test transforms/images/masks without adding validation; define whether alpha mask or luminance-derived sky participates in the formal loss. | Measure missing-mask/fallback incidence; do not infer correctness from a few debug images. |
| PLY/legacy v1/diagnostic export | Exclude 3D-sequence PLY from formal parity; exclude lossy SPL4-v1; either specify PLY SH property order and metadata or keep PLY diagnostic-only. | External SuperSplat PLY convention remains unverified. |
| manifest/publication | Require immutable output, temporary write and atomic rename, final external digest, completion index, required-field validation, correct debug vocabulary, PNG encoding, and `num_rendered`/tile-reference policy. | Full-file rehash cost and diagnostic metadata volume are performance/observability issues. |
| Viewer parser/handoff | P0-A3/A4 own strict parsing and provenance binding; additionally decide canonical scale representation and remove or version-isolate historical hard-coded selection. | Structured fetch failure and allocation/overflow guards are bounded robustness. |
| determinism/failure handling | Define seed ownership, fatal nonfinite behavior, partial-output cleanup, and the deterministic claims allowed for CUDA and equal-depth sorting. | Runtime nonfinite/OOM frequency and equal-depth ordering remain evidence gaps. |
| logging/visualization | Keep training-report PNG, alpha/depth visualization, helper quantization, and diagnostic rerenders outside acceptance unless their exact contract is selected. | `round` versus truncation and depth colormap differences are bounded when one formal writer is fixed. |

## Confirmed-correct renderer core

The audit does not show that the complete Fudan renderer is unusable. Static
source inspection and bounded CPU-double checks found the following core
contracts consistent:

- qL/qR component order `(w, x, y, z)`;
- SO(4) left/right multiplication and transpose conventions;
- Python/CUDA 4D covariance construction;
- qL/qR gradients when the scaling modifier is one;
- conditional covariance and conditional mean derivatives;
- temporal marginal derivatives with respect to `Sigma_tt`, temporal center,
  and opacity;
- temporal-prefilter application scope and ineligible-record gradient
  isolation;
- spatial SH coefficients 0-15 other than the P0-1 coefficient gradient;
- temporal coefficient gradients 16-47 in the spatial-degree-three layout;
- the compositor away from cap, skip, and early-out boundaries, including
  RGB, alpha, depth, flow, and background gradients;
- tile half-open `[min, max)` ranges, positive finite-depth sort keys, and
  backward reuse of the forward population.
- supported Gaussian optimizer groups have one parameter tensor per named
  group, without duplicate ownership;
- Adam row state is extended with zero moments for added rows and sliced with
  the same mask for removed rows;
- Gaussian record tensors remain aligned through clone, split, and prune on
  the audited supported 4D branch;
- basic multi-view batch gradient normalization accounts for visibility count;
- 4D checkpoint capture/restore field order is symmetric for the 19-field
  format, subject to P0-T6's missing semantic schema;
- SPL4-v2 preserves the nine rendering-field order, little-endian header and
  payload, source record order, and valid-v2 raw/runtime representations;
- checkpoint-SPL4 provenance can rehash both current files and verify an exact
  selected contiguous range under the asset's scale representation;
- provenance verification requires the full asset record count to equal the
  checkpoint count and does not silently reinterpret a full asset as a subset;
- selected CUDA reconstruction slices all per-Gaussian rendering tensors with
  the same `[start,end)` range and verifies the resulting rasterizer count;
- Viewer infrastructure hashes the loaded asset and preserves bounded workset
  order, range bounds, capacity checks, and `sourceIndex=start+localIndex`.

These results support continuing with a corrected Fudan Native 4DGS baseline
rather than changing methods before the audit and correction program is
complete. Evaluation of other 4DGS methods remains a future comparison
responsibility and does not change the current baseline track.

Every confirmed-correct item is limited to its stated branch and preconditions.
It is reusable infrastructure, not evidence that the complete current
training-to-Viewer pipeline is accepted.

## Evidence still insufficient

The static audit does not establish:

- identity between the inspected source and any currently compiled CUDA
  extension;
- CUDA float32, atomic-add, and scheduling behavior;
- deterministic ordering for equal-depth sort keys;
- the historical full training/render invocation and effective values;
- the historical checkpoint's internal tensor state, shapes, finite values,
  optimizer state, and internal iteration, because the large checkpoint was
  not loaded during the audit;
- the exact source/build relationship of the currently loaded CUDA binary;
- the corrected checkpoint, population count, representative selection, and
  fixed range, none of which exist yet;
- post-fix camera and renderer runtime state;
- full-scene `num_rendered` and tile-reference population;
- post-fix forward and gradient correctness;
- the runtime frequency and formal-dataset impact of nonfinite values, OOM,
  and other numerical edge cases;
- the external SuperSplat PLY coefficient/property convention;
- remaining formal policy choices listed below;
- runtime acceptance of a corrected checkpoint/SPL4/CUDA Reference/Viewer
  bundle.

These items must not be reclassified as confirmed-correct without focused
post-fix validation, controlled metadata inspection, a corrected run, or
runtime artifact acceptance as applicable.

## Formal policy decisions and intentionally open items

Audit integration classifies when each remaining decision is required. The
four user-approved policy groups are decided here; temporary candidates for
the remaining items must not be presented as the formal contract.

| Decision stage | Required decisions |
|---|---|
| decided and immutable | Do not reproduce historical negative-FoV-sentinel semantics; use the corrected shared renderer; train the formal baseline from scratch; never overwrite historical artifacts; keep the Viewer frozen; create a new population identity; use a fresh formal CUDA Reference; exclude legacy PNG reuse, SPL4-v1, and 3D-sequence PLY from formal evidence/parity. |
| decided: Fudan Native model | Evaluate SH at the conditional mean; spatial SH degree 3; temporal SH enabled at degree 2; fixed slots 0-15 spatial, 16-31 temporal mode 1, 32-47 temporal mode 2; `rot_4d=true`; `force_sh_3d=false`; forward/backward share the conditional-mean dependency and gradient path; reduced alternatives are unsupported and must fail closed. |
| decided: camera/effective state | Support exactly two modes: complete centered intrinsics with finite positive `fl_x/fl_y` and finite `cx/cy`, and complete geometrically valid FoV-only input with finite `0 < FoVx,FoVy < π`; canonicalize raw metadata once after resolution scaling; keep negative FoV sentinel as raw provenance only; make projection and rasterizer forward/backward consume one effective state; fail closed on off-center, mixed, partial, ambiguous, inconsistent, nonfinite, invalid-dimension, and invalid-clipping input; preserve projection near/far `0.01`/`100.0`, CUDA near-cull `p_view.z > 0.2`, and no CUDA far-cull as distinct semantics. |
| decided: dataset/evaluation | Preserve train as all 5,146 `v01-v31` frames and test as all 166 `v00` frames; require effective `eval=True` after CLI/config merge; reject effective `eval=False`, which yields 5,312 train and zero test cameras; keep the populations disjoint; add no validation population; keep `v31` in training; never use test for training, selection, tuning, or early stopping; use the pre-fixed final completed iteration as canonical; any validation-based experiment gets a separate identity/output owner. |
| decided: checkpoint/resume staging | Normalize completed-state checkpoints, exact completed-update labels, versioned semantics, diagnostics, and provenance independently of resume; prohibit pilot resume and every legacy warm-start; treat exact resume as a later independent root Fix with an equivalence gate; until accepted, resume fails closed and only uninterrupted completed formal runs can be canonical. |
| required before implementation | Python covariance/SH precompute support; alpha-cap derivative; scaling modifier; environment-map support; camera centered-tolerance, field-level validation/error schema, and common-builder API/location; remaining field-level formal run mode and checkpoint schema; evaluation/save/densification/reset/step ordering consistent with completed updates; temporal densification policy; strict point-cap/prune/opacity-reset policy. |
| required before formal retraining | Numeric final iteration and pilot/formal schedules; pilot/formal point caps; densification/prune/reset numeric schedules; complete effective-config snapshot; deterministic seed ownership details; test-report metric/cadence; nonfinite/OOM/partial-failure policy; immutable output directory and atomic publication; and, only if resume will be enabled, field-level restore state plus numerical/bitwise equivalence acceptance thresholds. |
| required before formal artifact generation | SPL4-v2 log/linear scale representation; PNG clamp/round/color/codec; full/range CUDA Reference purposes; manifest schema and validator; source-to-binary build provenance; bundle/index/external-digest ownership; direct evidence as formal same-invocation evidence or diagnostic-only. |
| required before Viewer restart | Corrected population and fixed range; Viewer provenance binding; strict parser acceptance; removal or versioned isolation of historical hard-coded ranges; Viewer capture/comparison bundle identity. |

## Pre-retraining validation requirements

After each separately authorized correction, validation proceeds from pure
contracts to bounded integration. No validation in this table was executed by
this documentation sync.

| # | Validation layer | Primary findings closed |
|---:|---|---|
| 1 | Pure CPU canonical-camera tests for SPH intrinsics, both supported modes, raw-sentinel isolation, dimensions/resolution scaling, clipping/cull separation, and the mixed/partial/nonfinite/invalid/off-center rejection matrix | P0-0 and camera/projection P1; P0-A6 consumes the accepted result later |
| 2 | Config precedence and post-merge effective-`eval` tests, formal-run rejection for `eval=False`, remaining run-mode/legacy-path checks, and unsupported-branch negative matrix | P0-T4, P0-A2, P0-T6/T7, config/run-mode P1 |
| 3 | Versioned checkpoint serialization/validation, completed-state boundary, and exact completed-update label test | P0-T1/T2/T6, P0-A1 |
| 4 | Training state-machine mock for iteration, evaluate, save, densify, reset, and step ordering | P0-T1/T2 |
| 5 | Optimizer-step and current-gradient preservation test | P0-T3, optimizer P1 |
| 6 | Densification clone/split/prune topology and Adam-row-state test | P0-T3 boundary, population P1 |
| 7 | Approved train/test identity with effective `eval=True` yielding 5,146/166, explicit rejection of `eval=False` yielding 5,312/0, disjointness, no-validation policy, test non-selection, and pre-fixed final completed-checkpoint selection test | P0-T4/A8 |
| 8 | Conditional resume-equivalence test across uninterrupted and separate-process restored continuation; required only before enabling resume | P0-T5/T6 and resume fidelity |
| 9 | Camera handoff numeric test that projection and rasterizer forward/backward consume identical canonical focal/tan values for centered intrinsics and valid FoV-only cameras | P0-0, camera P1 |
| 10 | Independent one-Gaussian forward oracle for projection, covariance, SH, alpha, and compositor | P0-1/P0-2/P0-3 and renderer P1 |
| 11 | CPU autograd and finite differences for xyz/time/scale/qL/qR/opacity/SH/screen mean | P0-1/P0-2/P0-3 and gradient P1 |
| 12 | CUDA forward and gradient smoke over every supported branch and boundary, with off-center and other unsupported camera input rejected before CUDA | P0-0 through P0-3, branch P1 |
| 13 | SPL4-v2 golden header/payload byte test | SPL4 representation and exporter P1 |
| 14 | Malformed/truncated/extra/overflow/nonfinite SPL4 parser matrix | P0-A3 |
| 15 | Checkpoint-SPL4 wrong-pair, count, range, order, and stale-file test | P0-A4 plus provenance P1 |
| 16 | CUDA manifest missing/wrong iteration/config/camera/population/artifact identity test | P0-A1/A2/A6/A8 |
| 17 | Clean-source/build-recipe/toolchain-to-loaded-binary identity test | P0-A7 |
| 18 | Direct-evidence production-invocation identity test | P0-A5 |
| 19 | Viewer bundle wrong-pair, stale provenance, wrong range, and historical-range rejection | P0-A3/A4 |
| 20 | Bounded no-resume pilot acceptance for updates, metrics, nonfinite state, checkpoints, and completion | all reachable Gate A and no-resume Gate B training findings |
| 21 | Formal output completeness, atomicity, index, external digest, and parent-binding check | publication P1 and P0-A6/A7/A8 |

Independent forward oracles precede CUDA comparison; Python-precompute paths
are not assumed to be independent oracles. One-sided checks cover temporal
eligibility, near/far, alpha skip/cap, early termination, determinant validity,
and tile bounds. The relevant validation report must be reviewed before its
next gate is opened.

For camera/evaluation specifically, the future focused validation must derive
the positive SPH canonical focal/tan values on CPU; prove that a raw negative
sentinel never enters effective state; cover both supported camera modes;
reject mixed, partial, inconsistent, nonfinite, invalid-dimension, invalid-FoV,
invalid-clipping, and off-center inputs before GPU execution; preserve camera
identity after resolution scaling; compare projection with rasterizer forward
and backward focal/tan; verify `eval=True` as 5,146/166 and reject
`eval=False` as the 5,312/0 branch; and distinguish projection `0.01`/`100.0`
from raster near-cull `0.2` and absent far-cull. Because off-center is
unsupported, it requires a pre-CUDA rejection test, not a CUDA correctness
claim. None of these tests has been created or run by this documentation sync.

## Manifest provenance requirements

The manifest must distinguish rather than conflate:

- the explicit input camera mode;
- raw FoV values, including any sentinel;
- raw `fl_x/fl_y/cx/cy` values when present;
- declared dataset dimensions, effective image dimensions, and resolution
  scale;
- effective `fx/fy/cx/cy`, FoV, and tanFov;
- projection matrix and the separate projection near/far values
  `0.01`/`100.0`;
- CUDA raster visibility near-cull `0.2` and the fact that raster far-cull is
  absent;
- the canonical effective-camera identity consumed by projection and by
  rasterizer forward/backward;
- post-merge effective `eval=True` and the resulting 5,146-train/166-test
  population identity;
- exact checkpoint identity and internal iteration;
- the canonical pre-fixed final checkpoint identity, its exact completed update
  count, and evidence that test was not a selection input;
- only for a separately identified future validation-based experiment, its
  selecting metric, validation population, and best-checkpoint lineage;
- complete effective Gaussian, renderer, environment, and pipeline state;
- source revision and dirty state;
- transforms, images, masks, and dataset identity;
- loaded CUDA binary identity plus its source/build/toolchain binding;
- full or selected population, source/local index mapping, and actual
  rasterizer Gaussian count;
- fresh render versus legacy copy versus diagnostic rerender;
- PNG clamp, quantization, color, and codec contract;
- render, GT, alpha, depth, meta, and direct-evidence identities;
- renderer/source identity, immutable output owner, run ID, completion index,
  and external bundle digest.

It must not publish normalized positive values while execution consumes a
different raw value, publish a raw negative FoV sentinel as effective FoV or
tanFov, publish CLI/default state as restored runtime state, or self-hash a file
by writing its claimed hash back into the same bytes. Required fields and
cross-bindings need a fail-closed acceptance validator. Unknown evidence must
remain explicitly unknown rather than inferred. These publication and
validation duties belong to P0-A6 after the P0-0 common camera contract is
implemented and accepted; they must not become a second camera-value owner
inside P0-0.

## Diagnostic render boundary

After all audits are integrated, the remaining required formal policy is
fixed, required corrections are implemented, and focused validation is
accepted, the legacy checkpoint may be rendered under the corrected renderer
as a controlled diagnostic. Compare that output with ground truth and the
legacy-renderer output to measure the effect of renderer correction on a fixed
checkpoint. This diagnostic must not be promoted to the formal corrected
checkpoint and must not overwrite any legacy artifact.

## From-scratch retraining requirement

Formal corrected-baseline training starts from the normal initial state, not
from the legacy checkpoint. It uses the corrected renderer for the complete
optimization and densification lifecycle and writes to a new output identity.
The current candidate path is:

`/home/demo/work/outputs/sph_scene_4dgs_corrected_intrinsics_v1`

This path is a plan-only candidate. It must be reviewed in the execution
instruction and must not be created by this documentation task.

## Comparison conditions

Under matched dataset, camera, frame/time, viewport, background, renderer
configuration, and output encoding, compare:

1. ground truth;
2. legacy checkpoint with legacy renderer;
3. legacy checkpoint with corrected renderer;
4. new checkpoint with corrected renderer.

The comparison must record exact source, checkpoint, renderer-semantics,
camera/time, and generated-artifact identities. It must keep diagnostic results
separate from formal corrected-baseline acceptance.

## New output identities

The corrected track must publish unique, non-overwriting identities for:

- training output directory and configuration;
- checkpoint and iteration;
- population provenance and record count;
- exported SPL4;
- corrected CUDA Reference and manifest;
- camera, time, viewport, and population selection;
- Viewer capture, JSON, PNG, and comparison artifacts.

The legacy count 3,231,588 and fixed range `[524288, 1048576)` are not assumed
for the new checkpoint. Representative records, tile-reference count, and the
fixed range must be selected again after the new population identity is known.

## Gate separation

### Gate A: before pilot training

Gate A requires the approved two-mode canonical camera contract with centered-
only intrinsics, off-center and invalid/ambiguous input rejection, distinct
projection `0.01`/`100.0` and raster near-cull `0.2`/no-far-cull semantics;
the approved conditional-mean, spatial-degree-3, temporal-degree-2 48-slot
branch with `rot_4d=true` and `force_sh_3d=false`; the pilot-reachable parts of
P0-0 through P0-3; P0-T1, P0-T2, and P0-T3; one formal run mode and effective
config whose post-merge `eval` is true; a new non-overwriting output owner; a
minimum versioned completed-state checkpoint foundation; fail-closed finite/
failure behavior; and the approved train/test-only pilot evaluation policy.

Pilot training may start only when:

- the formal pilot branch policy is unique and unsupported branches reject;
- P0-0 uses one canonical effective-camera state for projection and rasterizer
  forward/backward, every unsupported camera input rejects before GPU, and the
  focused camera forward/gradient validation is accepted;
- the remaining camera/renderer P0s reached by the branch are corrected and
  their focused forward/gradient validation is accepted;
- the requested update count is exact, including the final update;
- checkpoint/save state boundaries are unambiguous;
- densification iterations preserve the intended gradient/update transaction;
- train is all 5,146 `v01-v31` frames, test is all 166 `v00` frames, the
  populations are disjoint, no validation population is created, and the
  effective value after CLI/config merge is `eval=True`;
- effective `eval=False` and its 5,312-train/zero-test population reject at the
  formal-run entry;
- test never enters training, checkpoint selection, parameter tuning, or early
  stopping, and the best-checkpoint branch is rejected;
- the pilot starts at iteration 0 with resume and legacy warm-start entry
  points fail closed;
- output is new, the legacy path is rejected, and the complete effective
  config is snapshotted;
- checkpoint schema is versioned, labels the exact completed update, records a
  completed-state boundary and diagnostic provenance, and nonfinite state fails
  closed.

The pilot is a bounded intermediate run, not a formal checkpoint.

### Gate B: before formal retraining

Gate B requires accepted pilot evidence; a frozen canonical camera/eval
contract and publication identity, including the supported camera mode,
effective dimensions/intrinsics/FoV/tan, centered-only disposition, distinct
projection/cull semantics, and post-merge `eval=True`; frozen remaining formal
config and schedule; the exact approved train/test identity; test
non-selection; a pre-fixed numeric final iteration whose completed state is
canonical; clean source revision; reproducible build/runtime identity;
approved renderer branches; approved training, densification, reporting, seed,
and failure policies; the full versioned checkpoint/iteration transaction;
immutable atomic output publication; no legacy checkpoint warm-start; and no
legacy output reuse. A validation-based best checkpoint is not part of this
baseline. Camera policy approval alone does not satisfy Gate A or Gate B;
source correction, focused validation, pilot evidence, and every other listed
requirement remain necessary.

Resume support is not an unconditional Gate-B prerequisite. If it has not been
implemented and accepted as an independent root Fix with the resume-equivalence
gate, every resume entry must fail closed and only an uninterrupted completed
run may be canonical. If formal training will use resume, P0-T5/T6 and complete
continuation state must be closed and the equivalence gate accepted first.
P0-T4's required formal disposition is the approved train/test and
test-non-selection policy, which still requires implementation and validation;
P0-T7 is reached only when `env_map_res > 0`. Every forbidden branch must fail
closed rather than remain a silent option.

### Gate C: before corrected SPL4 and CUDA Reference generation

Gate C requires the exact formal checkpoint internal iteration and completed
update count; pre-fixed-final selection, effective-config, dataset, source, and
checkpoint binding; evidence that test did not select the checkpoint; P0-A1
and P0-A2 closure; SPL4-v2-only policy; exact checkpoint-to-SPL4 full mapping;
a clean source-to-loaded-binary build binding; strict exported header/payload
validation; fresh CUDA rendering; runtime camera/time/effective-renderer
publication that reproduces the accepted P0-0 canonical camera/eval identity;
manifest acceptance validation; P0-A6, P0-A7, and P0-A8 closure;
and a unique output directory with atomic artifact publication, completion
index, and external digest. Gate-C acceptance also fixes the corrected
population count and records a newly selected range; Gate D separately decides
whether that range is suitable and correctly bound for Viewer comparison.

P0-A5 must either produce evidence from the same production rasterizer
invocation or be explicitly diagnostic-only. Direct evidence is a P0 at this
gate only when it is part of formal acceptance.

### Gate D: before Viewer restart

Gate D requires P0-A3 and P0-A4 closure; strict SPL4 header/payload/semantic
validation; one corrected checkpoint/SPL4/provenance/CUDA Reference bundle; a
new corrected population count, representative selection, and fixed range;
removal or versioned isolation of historical hard-coded ranges; wrong-pair and
stale-artifact rejection; and Viewer capture/comparison bundle identity.

P0-A3/A4 are Viewer-specific and do not block Gate A or corrected training.
They do block promotion of corrected artifacts into Viewer acceptance.

## Root-finding owner table

Each Fix should close one root responsibility and its necessary tests. It must
not mix unrelated changes from several owners merely because they share a
later gate.

| Owner | Root findings/responsibility | Dependent consumer |
|---|---|---|
| common camera contract | P0-0; explicit two-mode validation, raw/effective separation, post-resolution canonical camera identity, projection/rasterizer forward/backward handoff, and pre-GPU rejection; excludes off-center expansion and visibility-semantics retuning | manifest/runtime camera publication in P0-A6 after P0-0 acceptance |
| renderer forward/backward | P0-1 plus conditional-mean P0-2 and spatial-3/temporal-2 fixed-48-slot P0-3 under `rot_4d=true`, `force_sh_3d=false`; remaining renderer P1 is separately bounded | training validation and CUDA Reference semantics |
| training state machine | P0-T1, P0-T2 | checkpoint labels and P0-A1 |
| optimizer/densification transaction | P0-T3 and population policy | final checkpoint population identity |
| dataset/evaluation policy | P0-T4; post-merge effective `eval=True`, exact `v01-v31` train and `v00` test identity, no validation population, test non-selection, and formal rejection of `eval=False` | fixed-final checkpoint selection and P0-A8 |
| evaluation/best policy | Formal reporting is selection-free and the best branch is unsupported; any future validation/best experiment needs a separate policy, identity, and output owner | conditional best lineage in P0-T5/A8 only outside this baseline |
| fixed-final checkpoint selection | Canonical identity is the pre-fixed final iteration at an exact completed-update boundary, not `chkpnt_best.pth` | P0-A8 manifest and artifact consumers |
| config/run launcher | complete post-merge effective config including required `eval=True`, formal mode, and legacy-path rejection | P0-A2 and manifest config identity |
| diagnostic checkpoint foundation | P0-T1/T2/T6; versioned semantics, completed-state boundary, exact update label, diagnostics, and provenance independent of resume | P0-A1/A2/A8 consumers |
| exact resume (conditional independent root) | P0-T5/T6 plus complete continuation state and equivalence, only after known P0/transaction stability | formal continuation only after gate acceptance; otherwise fail closed |
| environment-map lifecycle | P0-T7 when enabled | CUDA reconstruction and runtime publication |
| SPL4 exporter/provenance | v2 representation, atomic export, exact population binding | Viewer bundle input |
| SPL4 parser | P0-A3 | production evaluation inputs |
| CUDA Reference loader | P0-A1/A2, selected tensor mapping, fresh render | manifest and bundle output |
| manifest builder/validator | P0-A5/A6/A8 publication claims, including exact accepted P0-0 camera mode/effective state and projection/raster cull distinction without sentinel substitution | Viewer/comparison acceptance |
| build provenance | P0-A7 source-to-binary relation | formal CUDA Reference identity |
| Viewer handoff/bundle contract | P0-A4, corrected range and wrong-pair rejection | capture/comparison artifacts |
| output publication | unique owner, atomic write, completion/index/digest | every formal output stage |

## Viewer unfreeze and acceptance gates

Viewer development remains frozen until all of the following hold:

- Gates A-C are accepted and from-scratch corrected training is complete;
- corrected checkpoint, SPL4, provenance, fresh CUDA Reference, and bundle
  identities are fixed;
- the strict parser and provenance cross-binding reject malformed, stale, and
  wrong-pair inputs;
- the new population count, representative selection, and fixed-range design
  are reviewed without inheriting historical values;
- Viewer capture and comparison outputs bind the corrected bundle identity.

After unfreeze, reuse the completed Step118-122 production, diagnostic,
comparison, capture, and PNG infrastructure, but rerun asset-dependent
acceptance:

1. canonical representative comparison;
2. new fixed-range semantic comparison;
3. tile-reference, sort, compositor, and PNG comparison;
4. corrected full-scene gate over the new canonical population count without
   silent omission;
5. interactive camera/time, performance, scalability, and LOD validation.

Asset-dependent Step118-122 acceptance must be rerun. Infrastructure reuse
does not authorize reuse of legacy acceptance values, `3,231,588` records, or
the historical `[524288,1048576)` range.

## Dependency order

### Phase 0: audit integration

1. Investigation1-4 static audits complete. **Complete.**
2. Integrate P0/P1/P2, confirmed-correct, and insufficient-evidence findings.
   **Complete in this document.**
3. Integrate the approved Fudan Native model, train/test-only dataset,
   checkpoint/resume-staging, and camera/effective-`eval` policies.
   **Complete in this document.** Decide the remaining formal policy fields
   before their applicable gates.
4. Confirm one root owner and one bounded Fix responsibility at a time.

### Phase 1: training-critical foundation

5. Define the complete effective-config, run-mode, and non-overwriting output
   identity contract.
6. Implement the minimum versioned diagnostic checkpoint foundation and exact
   completed-update transaction independently of resume; leave field-level
   schema choices pending their later implementation policy approval.
7. Implement the common P0-0 camera-handoff Fix with the approved two-mode,
   centered-only, fail-closed, and distinct projection/cull contract; do not
   mix P0-A6 publication, off-center expansion, or visibility retuning into it.
8. Implement selected-branch renderer forward/backward P0 Fixes.
9. Implement the training iteration/step/evaluate/save transaction Fix.
10. Implement densification gradient and optimizer-mutation transaction Fix.
11. Implement post-merge effective `eval=True`, the approved train/test
    separation, test non-selection, and fixed-final completed-checkpoint
    selection; reject `eval=False`, validation/best, and resume branches at
    their entry points.
12. Correct every conditional P0 reached by the formal branch and fail-closed
    reject every unsupported branch.
13. Implement nonfinite/failure handling and atomic checkpoint publication.

### Phase 2: focused validation

14. Run the config/run/checkpoint negative matrix, including post-merge
    effective-`eval` acceptance and rejection.
15. Run independent canonical-camera CPU contracts and camera/renderer forward
    oracles, including supported-mode, sentinel-isolation, scaling, clipping/
    cull, and pre-GPU rejection cases.
16. Run CPU autograd and finite differences.
17. Build the selected CUDA source and run CUDA forward/gradient validation.
18. Run training state-machine, optimizer, and densification mocks.
19. Run train/test identity, test-non-selection, fixed-final checkpoint, and
    output-identity tests.
20. Run an integrated bounded small-scene smoke.

### Phase 3: pilot and formal training

21. Run small corrected pilot training from iteration 0 with resume prohibited
    after Gate A.
22. Review pilot updates, metrics, topology, checkpoints, and failure evidence.
23. Only if formal resume is wanted after known P0 and transaction stability,
    implement it under an independent root ticket and accept the resume-
    equivalence gate; otherwise keep resume fail closed.
24. Freeze the canonical camera/eval publication identity, formal config,
    dataset, source/build, schedule, and remaining policies.
25. Run the formal corrected baseline from scratch after Gate B, using resume
    only if step 23 was accepted.
26. Select and immutably publish the pre-fixed final completed checkpoint and
    its test-non-selection binding.

### Phase 4: corrected artifact path

27. Harden SPL4-v2 export, atomic publication, and population provenance.
28. Harden CUDA Reference reconstruction and runtime-state publication.
29. Establish the source/binary/dataset/checkpoint/render bundle contract.
30. Generate fresh corrected SPL4 and CUDA Reference after Gate C.
31. Fix the corrected population identity and record count.

### Phase 5: Viewer restart

32. Implement and validate the strict SPL4 parser.
33. Bind corrected provenance/bundle identity into Viewer handoff.
34. Design new representative records and a new fixed range.
35. Unfreeze the Viewer only after Gate D.
36. Rerun record-local, tile, sort, compositor, and PNG validation.
37. Run the matched corrected full-scene gate.
38. Continue only then to performance, interactive behavior, scalability, and
    LOD.

This ordering deliberately keeps Viewer-only P0-A3/A4 out of the pilot path,
while keeping training-state and checkpoint-identity defects before any
checkpoint or artifact that would inherit them.

## Completion and outstanding boundary

Complete at this milestone:

- Investigation1-4 static read-only audits;
- finding classification and root/dependent consolidation;
- integrated responsibility, dependency, validation, and gate design;
- historical-artifact preservation and non-overwrite policy;
- the decision to keep the Viewer frozen;
- repository synchronization of the user-approved Fudan Native model,
  train/test-only evaluation, checkpoint/exact-resume staging, and formal
  camera/effective-`eval` policies.

Not complete and not authorized by this document sync:

- remaining formal policy selection listed in Open items;
- source, config, test, or tool fixes;
- focused validation or CUDA build;
- pilot training or formal retraining, including any exact-resume Fix or
  equivalence acceptance;
- a corrected checkpoint, SPL4, CUDA Reference, or population identity;
- Viewer restart, fixed-range parity, full-scene correctness, or Acceptance
  Level 4;
- performance, interactive behavior, scalability, or LOD work.

## Open items

- remaining formal renderer branch and fail-closed matrix, including Python
  covariance/SH precompute, alpha-cap derivative, scaling modifier, and
  environment-map policy;
- camera implementation details: centered-principal-point numerical tolerance,
  field-level validation/error schema, and exact common-builder API/location;
- remaining field-level formal run-mode specification; effective `eval=True`
  is already decided and is not an open item;
- numeric final iteration and pilot/formal training schedules;
- numeric pilot/formal point caps and densification, prune, and opacity-reset
  schedules and detailed policy;
- final corrected output directory name and run identity;
- independent forward-oracle and gradient-validation fixtures, including the
  approved camera supported-mode and pre-GPU rejection matrix;
- manifest schema/version impact;
- selection-free test-report metric, cadence, and acceptance criteria;
- checkpoint field-level semantic/state schema;
- only if exact resume is pursued, its restored field ownership and numerical
  or bitwise equivalence threshold;
- deterministic seed ownership details and nonfinite/OOM/partial-failure
  policy;
- complete effective-config, dataset, source, environment, and build identity;
- formal output owner, atomic publication, completion index, and digest;
- new checkpoint iteration and population count;
- SPL4-v2 representation and export identity;
- corrected CUDA Reference output identity;
- manifest schema, runtime-state validator, and direct-evidence boundary;
- strict SPL4 parser and Viewer provenance/bundle binding;
- representative selection and fixed-range design;
- corrected semantic and image thresholds;
- full-scene resource and performance plan.

No camera, renderer, SH, backward, training-state, checkpoint, exporter,
parser, CUDA Reference, manifest, or Viewer fix; build; test; CUDA execution;
render; training; export; artifact generation; branch operation; commit; or
push has been performed by this documentation sync. The formal camera/eval
policy is synchronized, but P0-0 and P0-A6 remain open until their separate
source responsibilities and required validation are completed and accepted.
