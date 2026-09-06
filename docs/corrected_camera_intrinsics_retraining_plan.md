# Corrected Camera Intrinsics Retraining Plan

Status: **plan-only / Investigation1-4 and Issue-#10 static audit complete / audit integration documented / eight-formal-policy-groups-approved / Issue-#11-policy-sync / remaining-formal-policy-open / source-fixes-not-started / focused-validation-not-started / CUDA-not-run / pilot-not-started / formal-retraining-not-started / Viewer-frozen**

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
- Issue #8's static investigation of completed-update count, checkpoint and
  evaluation boundaries, and optimizer/densification parameter identity is
  complete and was independently reviewed before the policy below was
  approved by the user.
- Issue #10's read-only static investigation of formal configuration
  resolution, import/JIT order, output identity, and downstream consumer
  authority is complete. Its independently reviewed policy is approved by the
  user and synchronized here under Issue #11.
- The Investigation1-4 findings, dependencies, ownership boundaries, and
  pre-implementation gates are integrated in this document.

The Fudan Native model configuration, existing train/test-only dataset and
evaluation policy, checkpoint-foundation/exact-resume staging policy, and
formal camera/effective-`eval` policy were approved by the user on 2026-09-03
JST. The formal renderer invocation policy was approved on 2026-09-04 JST.
The alpha-cap derivative policy was approved by the user and synchronized here
on 2026-09-06 JST. The completed-update training transaction policy was then
approved by the user after independent review and synchronized here on
2026-09-06 JST. The independent formal-entry/effective-configuration policy was
subsequently approved after the Issue #10 investigation and is synchronized
under Issue #11. All eight policy groups are integrated below. Remaining formal
policy selection, source fixes, focused validation, CUDA execution, pilot
training, formal retraining, corrected artifact generation, and Viewer restart
have not started. P0 findings block only the gate whose accepted output would
reach the defect; Viewer-only defects do not unnecessarily block corrected
training, and training-state defects cannot be deferred to artifact generation.
Policy approval and document synchronization do not close any P0/P1 finding or
Gate A and do not constitute source-fix, focused-validation, runtime, or
scientific acceptance.

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
- the canonical checkpoint is mandatory checkpoint `N`, where numeric final
  `N` is fixed before the run, the save condition is built from effective
  post-merge `N`, and the saved state includes update `N` plus its scheduled
  topology/reset; it is not a validation- or test-selected best checkpoint.

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
official-code behavior. The alpha-cap derivative is decided independently
below; other genuinely undecided renderer behavior remains open. The
invocation branches selected below are no longer policy-open.

### Approved formal renderer invocation contract

The first formal baseline has exactly one effective renderer invocation:

- `compute_cov3D_python=False`;
- `convert_SHs_python=False`;
- `scaling_modifier=1.0` exactly;
- `env_map_res=0`; and
- `override_color=None`.

This selects CUDA-direct covariance and CUDA-direct SH, uses the unmodified
scale, disables the environment map, and forbids color override or any other
SH bypass. The same five-field effective contract applies to training,
evaluation/test rendering, CUDA Reference generation, and every checkpoint
consumer that can reach a formal render. A common post-merge validator must
reject every mismatch before importing, initializing, or invoking the formal
renderer or CUDA/JIT path. Matching source or CLI defaults is not proof of
effective identity, and no consumer may silently reconstruct these values from
its own defaults.

The branch restriction is intentional. Python covariance precomputation is not
equivalent to CUDA-direct covariance in population, temporal marginal, and
prefilter behavior. Python SH precomputation detaches direction and time and
therefore conflicts with the approved conditional-mean gradient contract.
Non-unit scaling has forward/backward scale-gradient and Python temporal-
marginal inconsistencies. No SPH source or dataset evidence requires an
environment map, while its startup, checkpoint, optimizer, and resume lifecycle
is the P0-T7 boundary. A non-`None` override bypasses CUDA SH. One supported
path therefore isolates the effect of the P0-1, P0-2, and P0-3 corrections.
Upstream defaults alone are not a correctness argument.

This invocation decision does not accept the current renderer. The selected
CUDA-direct path remains unusable for formal work until P0-1, P0-2, and P0-3
are corrected in source and their focused validation is accepted. Formal-
configuration enforcement is also unimplemented. The exact shared-validator
API, location, error schema, and mechanism that guarantees failure before
renderer import/CUDA JIT remain implementation-open. Future support for any
rejected branch requires separate policy, any necessary Fix and validation,
and a distinct run or artifact identity when semantics differ.

### Approved independent formal entry and effective-configuration contract

The first formal baseline has one authoritative, pure resolver/validator for
its effective execution state. It must be independent of `torch`, the renderer,
`Scene`, and CUDA-related modules. In the same training process it resolves one
explicit formal configuration, validates it fail closed, and produces one
verified typed state before any import of the renderer, `Scene`, CUDA extension,
or other module that can initialize or JIT-compile the heavy path. Investigation
#10 Candidate B, the common authority, and Candidate A, the in-process pre-JIT
bootstrap, are complementary parts of this contract rather than alternatives.
Candidate C, a parent process that launches the existing training program as a
subprocess, and Candidate D, validation inside training after renderer import or
JIT, are rejected because neither establishes the required same-process,
pre-heavy-import authority boundary.

Semantic CLI overrides are forbidden. The resolver must accept exactly one
explicit formal input authority, resolve and validate it once, and reject a
missing or unknown field, an implicit semantic default, duplicate or competing
authorities, and any later semantic overwrite. The exact allowlist of purely
operational CLI controls remains undecided; this policy does not invent option
names. Helper name, file placement, API, error schema, and the concrete typed-
state representation also remain implementation-open.

The verified state must bind the already approved dataset/evaluation, Fudan
Native model, camera/effective state, and five-field renderer values; effective
post-merge completed-update count `N`; and a unique non-overwriting output
identity. It must prove no resume, warm-start, best-checkpoint, environment-map
checkpoint, legacy output, or pre-existing output-directory state is admitted.
`scaling_modifier=1.0` and `override_color=None` are explicit verified fields,
not call-site defaults. The final-checkpoint condition and test/save lists must
be constructed only from verified post-merge `N`. This does not move ownership
of the P0-T1/T2 state machine or P0-T3 optimizer/densification transaction into
the resolver; those remain separate source responsibilities that consume the
verified state.

The same verified identity is consumed by training and, later, by formal
evaluation and CUDA Reference generation. Those consumers may not reconstruct
values from local defaults. `cfg_args` is neither a complete execution identity
nor a formal input authority, and a formal consumer must not retain an
executable `eval` route for it. Checkpoint/manifest publication may record and
verify the identity, but cannot become another policy owner. The resolver does
not own camera or renderer mathematics, the checkpoint schema in P0-T6, atomic
publication, deterministic seed details, reporting policy, exact resume, or the
P0-T1/T2/P0-T3 transaction.

The present source does not implement this contract:

- `train.py` imports `gaussian_renderer` and `Scene` before parsing formal
  input, while `gaussian_renderer/diff_gaussian_rasterization.py` performs
  module-level `torch.utils.cpp_extension.load()`;
- `save_iterations` receives the CLI/default `iterations` value before YAML is
  loaded, then the recursive YAML merge overwrites parsed CLI/default fields;
- repository YAMLs are legacy/general configurations rather than a complete
  formal authority, and some carry `loaded_pth` or best-checkpoint paths that
  the formal entry must reject;
- output setup reuses an existing directory, overwrites `cfg_args`, and then
  `Scene` can write camera/input artifacts or load `loaded_pth` state;
- the initial training report occurs before any optimizer update;
- renderer calls rely on the default `scaling_modifier=1.0` and
  `override_color=None` rather than an explicit verified identity; and
- `get_combined_args` can execute `eval()` on `cfg_args`, while its merge route
  is not unified with the training YAML route or any formal consumer contract.

These are current-source findings, not completed Fixes. No resolver,
pre-heavy-import enforcement, typed state, consumer integration, or focused
validation has been implemented by this synchronization.

### Approved alpha-cap derivative contract

The user approved this corrected mathematical contract as Candidate A on
2026-09-06 JST. For each contributing Gaussian/pixel pair, define

`G = exp(power)`

`raw_alpha = effective_opacity * G`

`alpha = min(0.99f, raw_alpha)`

The forward `0.99f` cap is preserved. Backward applies the piecewise derivative

`s(raw_alpha) = 1` when `raw_alpha < 0.99f`, and
`s(raw_alpha) = 0` when `raw_alpha >= 0.99f`,

so that `dL/draw_alpha = s(raw_alpha) * dL/dalpha`. Bitwise equality with the
float32 value `0.99f` is on the saturated branch and its selected subgradient
is zero. CUDA therefore uses `raw_alpha < 0.99f` as the only uncapped test; it
must not use `<=` or infer the branch from a rounded or already capped value.

RGB, flow, depth, mask, and background contributions continue to aggregate
into the existing `dL/dalpha`. The raw-alpha gate is applied exactly once after
that aggregation. It gates only the alpha-mediated chain: opacity, `G`, screen
mean x/y, conic/covariance, and the downstream 3D/4D parameters reached through
those quantities. It does not gate the direct gradients to color values, flow
values, or depth values, nor the direct depth-to-screen-mean-z path. A parameter
may also receive gradients through a direct path, an uncapped contribution, or
another pixel, so its total gradient is not promised to be zero merely because
one contribution is saturated.

This contract changes no forward output or visibility decision. The `0.99f`
cap, alpha-below-`1/255` skip, early termination, contributor ordering, tile
membership, and visibility semantics remain unchanged. The formal baseline
does not adopt a straight-through or other surrogate derivative above the cap,
remove the cap, replace it with a smooth cap, or reject a formal input merely
because it reaches the cap.

The official-code ungated backward is source-lineage evidence, not the formal
derivative. This corrected baseline instead differentiates the piecewise graph
actually executed by forward while retaining its numerical-stability
semantics. No external paper is claimed to prescribe this exact threshold,
parameter-chain boundary, or equality subgradient; the contract rests on
executed-graph consistency, preservation of the existing forward behavior,
isolatable validation, and explicit user approval.

The CPU oracle must implement this branch independently and must not inherit
whatever equality subgradient a framework `min` or `clamp` happens to choose.
Boundary fixtures distinguish float32 `c = 0.99f`,
`nextafterf(c, -infinity)`, bitwise equality with `c`, and
`nextafterf(c, +infinity)`. Central finite difference at equality is not an
acceptance criterion. At equality, validation separately checks the left
one-sided slope of one, the right one-sided slope of zero, and the selected
analytic subgradient of zero.

The first candidate for a future, separately authorized minimal Fix is local
to `backward.cu`: derive the gate from the already recomputed uncapped
`raw_alpha`, then apply it once to the aggregated `dL/dalpha` before the
alpha-mediated chain. This is a candidate implementation boundary, not an
implementation instruction or evidence that the Fix exists. Only if focused
runtime validation cannot establish float32 forward/backward branch parity may
a separately reviewed design consider storing a cap-active flag from forward.
Expansion into buffers, headers, bindings, or the Python API is not authorized
by this policy or documentation sync.

## Confirmed training-lifecycle P0 blockers

Investigation3 confirmed the following independent training-state blockers.
They retain their original IDs.

### P0-T1: iteration and optimizer off-by-one

The loop increments `iteration` before processing a batch and executes
`optimizer.step()` only while `iteration < opt.iterations`. From scratch, the
current source processes labels `2..N`, calls the optimizer at most `N-2`
times, omits the final update at label `N`, and fetches an extra batch before
breaking at `N+1`. The approved contract instead starts with completed update
count zero and executes transactions `k=1..N` exactly once each, with one
batch fetch and one optimizer step per transaction, including `k=N`, and no
batch fetch for `N+1`. Requested count, transaction label, and completed update
count must have this single consistent meaning. The source Fix and focused
validation have not started, so P0-T1 remains open and blocks pilot training.

### P0-T2: checkpoint iteration is not a completed-state boundary

Evaluation and checkpoint save occur before densification, opacity reset, and
the optimizer step for the same iteration. A checkpoint labelled with that
iteration does not represent the completed state transition named by the
label. Under the approved contract, completed state `k` exists only after the
optimizer update and every scheduled topology and opacity-reset event for
transaction `k` have succeeded. A checkpoint labelled `k` saves that completed
state, never the state before those events. Checkpoint save precedes test
evaluation when both observe the same completed state. Final checkpoint `N` is
mandatory, and its save condition must be constructed and verified from the
effective `N` after CLI/config merge. Atomic write, incomplete-state recovery,
and field-level checkpoint schema remain separate open responsibilities. The
source Fix and focused validation have not started, so P0-T2 remains open.

### P0-T3: Gaussian gradients are lost on densification iterations

Densification replaces optimizer-owned Gaussian parameter tensors before the
same iteration's optimizer step. The freshly installed parameters do not own
the gradients produced by the just-completed backward pass, so the Gaussian
update is silently lost on those iterations. Densification topology helpers
may still preserve record alignment; the defect is the transaction ordering,
not an SPL4 record-order defect. The approved transaction collects current
visibility, radii, screen-space gradient, and time gradient and applies the
required densification statistics before any Parameter replacement. It then
steps the same Parameter identities that received backward gradients, performs
optimizer zero-grad, and only then executes scheduled densification/clone/
split/prune followed by scheduled opacity reset. The source Fix and focused
validation have not started, so P0-T3 remains open.

### Approved completed-update training transaction

For every requested transaction `k=1..N`, execution starts at completed state
`k-1` and has this conceptual order:

1. fetch exactly the one batch used by transaction `k`;
2. apply the learning-rate schedule and SH schedule event associated with `k`
   before forward;
3. run forward, compute optimization input loss `k`, and run backward;
4. capture the current render's visibility, radii, screen-space gradient, and
   time gradient before Parameter replacement and update the required
   densification statistics;
5. apply one optimizer step to the same Parameter identities used by backward,
   including at `k=N`, then perform optimizer zero-grad;
6. execute scheduled densification/clone/split/prune, then scheduled opacity
   reset; and
7. after every required event succeeds, declare completed state `k`, save any
   scheduled checkpoint with completed label `k`, then evaluate the same
   completed state for scheduled test reporting.

When densification/prune and opacity reset coincide, their relative order is
densification/prune first and opacity reset second. Pruning reads post-step,
pre-reset opacity; clone/split children derive from post-step parent state; and
opacity reset applies to survivors and new children. The formal opacity group
is stepped before reset, and reset is an explicit algorithm event rather than
an implicit skipped opacity update. Numeric thresholds, intervals, point cap,
prune schedule, opacity-reset schedule, and temporal densification policy remain
open.

Optimization input loss `k` is computed by the forward from completed state
`k-1` and is the input that produces optimizer update `k`. A completed-state
test metric `k` is obtained by newly rendering completed state `k` after its
optimizer, topology, and reset events. Moving training-loss output processing
after transaction completion does not turn that loss into a completed-state
metric. Test remains selection-free. Checkpoint-first and test-evaluation-
second does not decide run acceptance after evaluation failure, atomic
publication, or a general partial-failure policy.

Initial state `0` is not an optimizer update, formal iteration, completed
checkpoint, or test-selection state. Whether the current unconditional
`training_report(0)` is retained is deferred to the reporting policy. If
retained, it is a selection-free initial-state diagnostic; its PNG,
TensorBoard, CUDA-cache, and failure contracts remain open.

Candidate B is the approved contract because it applies the current gradient
to the same Parameter before existing topology helpers replace that Parameter,
and therefore needs no new gradient-transport mechanism. Candidate A, the
current order, does not close P0-T1/T2/T3. Candidate C, pre-step mutation plus
gradient transfer, would require new clone/split/prune/reset gradient mappings
without need. Candidate D, step then checkpoint before mutation, would omit
transaction `k`'s scheduled topology/reset from checkpoint `k`. This selection
rests on completed-transaction consistency and bug isolation, not identity
with official code. Helper boundaries, loop syntax, and local names remain
implementation decisions; no source or validation work is completed by this
policy synchronization.

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
parameter/optimizer/schedule continuation. The approved `env_map_res=0`
contract makes this branch unreachable in the first formal baseline and every
environment-map parameter or checkpoint state must be rejected before formal
rendering. This does not fix P0-T7. Future environment-map support requires a
separate policy plus lifecycle, checkpoint, optimizer, resume, and renderer
validation before it can become reachable.

### Approved checkpoint foundation and exact-resume staging

Checkpoint normalization is required independently of whether continuation is
ever enabled. The diagnostic-checkpoint foundation consumes the approved
completed-state label and save boundary from P0-T1/T2, but separately owns
P0-T6's versioned semantic schema, diagnostic state, semantic validation, and
dataset/config/source/renderer provenance. It does not own the loop count,
same-Parameter optimizer transaction, or exact-resume continuation state.
Incomplete, unknown-version, or semantically incompatible state must fail
closed. The schema's field-level design remains open.

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
| P0-T1/T2 | P0-A1 | Fix the exact-N training state machine and completed checkpoint boundary first; make the loader verify and publish its label second. |
| P0-T6 | P0-A1/A2 | Define the diagnostic checkpoint's versioned field semantics, validation, and provenance independently of the transaction loop and exact resume; make consumers reject and publish them second. |
| P0-T6/T7 plus incomplete config snapshot | P0-A2 | Define effective config/checkpoint semantics first; for the first baseline reject every environment-map state under `env_map_res=0`; reconstruct and assert the accepted five-field renderer invocation in the CUDA loader second. |
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

The following other confirmed issues remain recorded. The approved invocation
contract makes the indicated alternative branches unsupported rather than
silently available:

- missing temporal backward propagation when `rot_4d=false` (unsupported by
  the approved model contract);
- non-equivalent Python-precompute and CUDA-direct branches (Python covariance
  and SH precompute are unsupported);
- disagreement between the forward alpha `0.99` clamp and its ungated backward
  derivative (policy decided; source Fix and focused validation not started);
- missing scale-gradient factors and Python temporal-marginal inconsistency
  when `scaling_modifier != 1` (non-unit and nonfinite values are unsupported);
- a factor-of-two z-gradient error in projected 2D covariance outside the
  viewport clamp;
- disagreement between SH allocation and evaluator layouts;
- insufficient fail-closed validation for invalid or nonfinite inputs; and
- `override_color` bypass of CUDA SH (every non-`None` value is unsupported).

An issue needed by the selected formal branch must be corrected and validated.
An unused branch must be rejected explicitly rather than left silently
available. The alpha-cap derivative policy is decided, but its source Fix,
focused validation, and float32 runtime branch-parity confirmation have not
started. Python covariance/SH precompute, non-unit scaling, environment maps,
and color override are decided as unsupported for the first formal baseline;
their historical findings remain recorded, but their adoption is not an open
policy question. The camera policy above is decided but its P0-0 source Fix and
focused validation have not started. P0-1, P0-2, and P0-3 source fixes and
validation have likewise not started.

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
| renderer branch policy | Implement conditional-mean SH, spatial degree 3, temporal degree 2 with the fixed 48-slot layout, `rot_4d=true`, and `force_sh_3d=false`; enforce `compute_cov3D_python=False`, `convert_SHs_python=False`, exact `scaling_modifier=1.0`, `env_map_res=0`, and `override_color=None` before any formal renderer; apply the separately owned alpha-cap piecewise derivative to the alpha-mediated chain while preserving direct value/depth-z paths; resolve other supported-path findings separately. | Determinant epsilon, radius inflation, empty population, overflow, radius-threshold diagnostics, and separately identified future support for rejected invocation branches. |
| optimizer/learning-rate policy | Define temporal-position LR scheduling and supported parameter-group behavior; do not infer it from the spatial-only scheduler. | Log effective per-group LR without adding a second schedule owner. |
| densification/population policy | Preserve the approved post-step topology boundary and densify/prune-before-reset relative order; define temporal selection, strict point-cap semantics, clone/split/prune internal behavior, prune-only behavior, numeric reset schedule, and screen/world pruning. | Measure cap overshoot, nonfinite accumulator frequency, and memory pressure in bounded pilot runs. |
| evaluation/metric policy | Preserve the approved train/test identity, create no validation population, keep test selection-free, and use the pre-fixed final completed checkpoint. Test-report metric/cadence and clamp/channel/background remain open; any future best branch needs a separate experiment policy. | Keep diagnostic train samples separate from test reporting and visualization. |
| config/run mode/provenance | Require effective `eval=True` after CLI/config merge, reject effective `eval=False`, define the remaining formal run-mode fields and precedence, reject legacy output/path reuse, capture the complete effective config, and replace executable config parsing where it reaches formal tooling. | Report path-remap and duplicate-basename ambiguity as bounded diagnostics. |
| training state machine | P0-T1/T2 own exact `k=1..N`, the final step, no `N+1` batch fetch, completed labels, and the completed-state checkpoint/test boundary. | Keep loop/helper API local; consumers verify rather than redefine completed count. |
| optimizer/densification transaction | P0-T3 owns current statistics capture, same-Parameter step, zero-grad, then topology and reset mutation, with densify/prune before reset when simultaneous. | Numeric temporal selection, cap, threshold, interval, and prune/reset schedules remain separate population policy. |
| diagnostic checkpoint foundation | P0-T6 owns versioned field semantics, semantic validation, provenance, and diagnostic state; it consumes the P0-T1/T2 completed label but does not own loop order or exact resume. Atomic writes and field-level state remain to be defined. | Record interruption/OOM behavior and checkpoint-size/hash cost. |
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
eight user-approved policy groups are decided here; temporary candidates for
the remaining items must not be presented as the formal contract.

| Decision stage | Required decisions |
|---|---|
| decided and immutable | Do not reproduce historical negative-FoV-sentinel semantics; use the corrected shared renderer; train the formal baseline from scratch; never overwrite historical artifacts; keep the Viewer frozen; create a new population identity; use a fresh formal CUDA Reference; exclude legacy PNG reuse, SPL4-v1, and 3D-sequence PLY from formal evidence/parity. |
| decided: Fudan Native model | Evaluate SH at the conditional mean; spatial SH degree 3; temporal SH enabled at degree 2; fixed slots 0-15 spatial, 16-31 temporal mode 1, 32-47 temporal mode 2; `rot_4d=true`; `force_sh_3d=false`; forward/backward share the conditional-mean dependency and gradient path; reduced alternatives are unsupported and must fail closed. |
| decided: camera/effective state | Support exactly two modes: complete centered intrinsics with finite positive `fl_x/fl_y` and finite `cx/cy`, and complete geometrically valid FoV-only input with finite `0 < FoVx,FoVy < π`; canonicalize raw metadata once after resolution scaling; keep negative FoV sentinel as raw provenance only; make projection and rasterizer forward/backward consume one effective state; fail closed on off-center, mixed, partial, ambiguous, inconsistent, nonfinite, invalid-dimension, and invalid-clipping input; preserve projection near/far `0.01`/`100.0`, CUDA near-cull `p_view.z > 0.2`, and no CUDA far-cull as distinct semantics. |
| decided: dataset/evaluation | Preserve train as all 5,146 `v01-v31` frames and test as all 166 `v00` frames; require effective `eval=True` after CLI/config merge; reject effective `eval=False`, which yields 5,312 train and zero test cameras; keep the populations disjoint; add no validation population; keep `v31` in training; never use test for training, selection, tuning, or early stopping; use the pre-fixed final completed iteration as canonical; any validation-based experiment gets a separate identity/output owner. |
| decided: checkpoint/resume staging | Normalize completed-state checkpoints, exact completed-update labels, versioned semantics, diagnostics, and provenance independently of resume; prohibit pilot resume and every legacy warm-start; treat exact resume as a later independent root Fix with an equivalence gate; until accepted, resume fails closed and only uninterrupted completed formal runs can be canonical. |
| decided: renderer invocation | Require one effective contract everywhere: `compute_cov3D_python=False`, `convert_SHs_python=False`, `scaling_modifier=1.0` exactly, `env_map_res=0`, and `override_color=None`; reject every other or nonfinite value before renderer import/CUDA JIT or formal render; training, evaluation/test render, CUDA Reference, and checkpoint consumers share the same validated identity and may not substitute defaults. This decision does not accept the current CUDA-direct renderer, which remains blocked on P0-1/P0-2/P0-3 source fixes and focused validation. |
| decided: alpha-cap derivative | Keep `alpha=min(0.99f, raw_alpha)` in forward; after aggregating all `dL/dalpha`, use the independent piecewise gate `raw_alpha < 0.99f` for the alpha-mediated opacity/G/screen-xy/conic/covariance chain and zero that chain for `raw_alpha >= 0.99f`, including a zero selected subgradient at bitwise-equal float32 `0.99f`; preserve direct color/flow/depth and depth-to-screen-z gradients. Do not use an STE/surrogate, cap removal, smooth cap, or cap-triggered formal rejection. Source Fix and focused validation remain required. |
| decided: completed-update transaction | Start from completed count zero and execute `k=1..N` exactly once with no `N+1` fetch; apply schedules before forward; forward/loss/backward; collect and apply current densification statistics before Parameter replacement; same-Parameter optimizer step including `k=N`; zero-grad; scheduled densify/clone/split/prune; scheduled opacity reset; then declare completed state `k`, save checkpoint `k`, and evaluate that same state. Densify/prune precedes reset when simultaneous; prune reads post-step/pre-reset opacity; children derive from post-step parents; reset reaches survivors and children. Final checkpoint `N` is mandatory from effective post-merge `N`. Optimization input loss `k` remains distinct from completed-state test metric `k`; initial state zero is not an update or selection state. Candidate B is adopted and A/C/D are rejected for the bounded reasons above. Source Fix and focused validation remain required. |
| decided: independent formal entry/effective configuration | Use one pure resolver/validator, independent of `torch`, renderer, `Scene`, and CUDA modules, as the sole authority for one explicit formal configuration; combine the common owner of Investigation #10 Candidate B with the same-process pre-JIT bootstrap of Candidate A; reject parent-subprocess Candidate C and post-JIT Candidate D; forbid semantic CLI override and fail closed on missing/unknown fields, implicit semantic defaults, multiple authorities, later semantic overwrite, resume/warm-start/best/environment-map checkpoint state, legacy output, and pre-existing output directories before renderer import/JIT/CUDA/render/output write. Bind approved dataset/model/camera/renderer values, explicit `scaling_modifier=1.0` and `override_color=None`, verified post-merge `N`, final-checkpoint/test/save construction, and output identity once for all consumers. `cfg_args` is not a formal authority or complete identity and no formal consumer may execute it. Exact operational CLI allowlist, helper/API/location/error schema, and typed representation remain implementation-open. |
| required before implementation | Camera centered-tolerance, field-level validation/error schema, and common-builder API/location; independent resolver helper/API/location/error schema/typed state, exact operational CLI allowlist, and same-process pre-heavy-import enforcement; remaining field-level formal run mode and checkpoint schema; concrete training loop/helper API; temporal densification policy; strict numeric point-cap/prune/opacity-reset policy. The completed-update event order and formal-entry authority policy themselves are not open. |
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
| 2 | Pure resolver/validator tests, with no `torch`, renderer, `Scene`, or CUDA dependency: accept one explicit authority and the approved dataset/model/camera/renderer state; prove effective post-merge `eval=True`, explicit `scaling_modifier=1.0`/`override_color=None`, verified `N`, final-checkpoint/test/save construction from that `N`, and unique output identity; reject missing/unknown fields, implicit defaults, duplicate authority or overwrite, legacy/general YAML, `eval=False`, resume/warm-start/best/environment checkpoint state, legacy or pre-existing output, and every unsupported branch before renderer import/CUDA JIT/output write | independent formal-entry policy, P0-T4, P0-A2, P0-T6/T7, renderer/config/run-mode P1 |
| 3 | Versioned checkpoint field serialization, semantic validation, diagnostic provenance, and rejection matrix; consume rather than redefine the completed-state label from P0-T1/T2 | P0-T6, P0-A1/A2 |
| 4 | Training state-machine mock proving completed count starts at zero, `k=1..N` performs exactly `N` batch fetches and optimizer steps including final `N`, no `N+1` fetch occurs, and checkpoint-first/test-second both observe the post-topology/post-reset completed state | P0-T1/T2 |
| 5 | Optimizer transaction test proving current visibility/radii/screen/time statistics are captured before replacement, the backward-owned Parameter is stepped, zero-grad precedes mutation, and ordinary/final/reset/topology transactions follow the approved event order | P0-T3, optimizer P1 |
| 6 | Densification clone/split/prune and opacity-reset test proving post-step parent/survivor values, Adam-row preservation or zero initialization, post-step/pre-reset prune opacity, densify/prune-before-reset when simultaneous, and reset coverage of survivors and children | P0-T3 boundary, population P1 |
| 7 | Approved train/test identity with effective `eval=True` yielding 5,146/166, explicit rejection of `eval=False` yielding 5,312/0, disjointness, no-validation policy, test non-selection, and pre-fixed final completed-checkpoint selection test | P0-T4/A8 |
| 8 | Conditional resume-equivalence test across uninterrupted and separate-process restored continuation; required only before enabling resume | P0-T5/T6 and resume fidelity |
| 9 | Camera handoff numeric test that projection and rasterizer forward/backward consume identical canonical focal/tan values for centered intrinsics and valid FoV-only cameras | P0-0, camera P1 |
| 10 | Independent one-Gaussian forward oracle for projection, covariance, SH, alpha, and compositor | P0-1/P0-2/P0-3 and renderer P1 |
| 10a | Independent single-Gaussian/single-pixel CPU alpha-cap oracle and focused gradient suite: well below/above the cap; float32 nextafter below/equal/above; individual and mixed RGB/flow/depth/mask loss paths; nonzero background; multiple contributors; opacity plus screen-mean or conic gradients; non-crossing finite differences below/above; separate equality one-sided slopes and selected subgradient; pre-Fix negative regression; unchanged direct gradients and forward output; and both 3D and 4D CUDA-direct paths under the exact five-field invocation | alpha-cap renderer-math responsibility |
| 11 | CPU autograd and finite differences for xyz/time/scale/qL/qR/opacity/SH/screen mean | P0-1/P0-2/P0-3 and gradient P1 |
| 12 | CUDA forward and gradient smoke over the one supported renderer invocation and every supported camera boundary; verify identical effective invocation across training, evaluation/test render, CUDA Reference, and checkpoint consumers, with every unsupported input rejected before CUDA/formal rendering | P0-0 through P0-3, P0-T7 reachability, branch P1 |
| 13 | SPL4-v2 golden header/payload byte test | SPL4 representation and exporter P1 |
| 14 | Malformed/truncated/extra/overflow/nonfinite SPL4 parser matrix | P0-A3 |
| 15 | Checkpoint-SPL4 wrong-pair, count, range, order, and stale-file test | P0-A4 plus provenance P1 |
| 16 | CUDA manifest missing/wrong iteration/config/camera/population/artifact identity test | P0-A1/A2/A6/A8 |
| 17 | Clean-source/build-recipe/toolchain-to-loaded-binary identity test | P0-A7 |
| 18 | Direct-evidence production-invocation identity test | P0-A5 |
| 19 | Viewer bundle wrong-pair, stale provenance, wrong range, and historical-range rejection | P0-A3/A4 |
| 20 | Bounded no-resume pilot acceptance for updates, metrics, nonfinite state, checkpoints, and completion | all reachable Gate A and no-resume Gate B training findings |
| 21 | Formal output completeness, atomicity, index, external digest, and parent-binding check | publication P1 and P0-A6/A7/A8 |

The focused transaction suite must preserve the current-source negative
regression for requested `N=1,2,3`, where optimizer-step calls are currently
`0,0,1`, and require the corrected result `1,2,3`. It must trace ordinary,
final, densify, prune-only, reset-only, and simultaneous topology/reset cases;
inspect Parameter identity, gradient, value, and Adam state before and after
each boundary; prove the final update affects parameters; and prove final
checkpoint `N` is scheduled from the effective post-merge value. It must also
show that a checkpoint includes its transaction's topology/reset, checkpoint
and test evaluation observe the same completed state in that order, training
loss retains optimization-input meaning, and any retained initial diagnostic
cannot enter formal iteration, checkpoint, test selection, or best selection.
The state-machine, optimizer/densification, checkpoint-schema, and exact-resume
fixtures remain separately owned even where they share `train.py` boundaries.

Renderer-invocation validation must positively accept the exact tuple
`(False, False, 1.0, 0, None)` in the field order above. It must reject
`compute_cov3D_python=True`, `convert_SHs_python=True`, every non-unit or
nonfinite `scaling_modifier`, every nonzero `env_map_res`, any non-empty
environment-map checkpoint state even when structurally valid, and every
non-`None` `override_color`. Negative cases must prove that no renderer import,
CUDA JIT, CUDA call, or formal render is reached. Cross-entrypoint tests must
prove that training, evaluation/test rendering, CUDA Reference generation, and
checkpoint consumers use one effective identity rather than independently
matching defaults. These tests and the enforcement they require do not yet
exist.

Formal-entry validation must additionally prove import order without importing
the heavy path in the test fixture: resolution and validation finish first,
then and only then may the same process import renderer/`Scene`/CUDA-related
modules. It must prove the verified state is the sole semantic authority, that
operational CLI handling cannot change it, and that formal evaluation and CUDA
Reference consumers later receive the same identity without defaults or
`cfg_args` reconstruction. The exact operational allowlist and implementation
API/location remain open, so no speculative flag names are acceptance criteria.

Independent forward oracles precede CUDA comparison; Python-precompute paths
are not assumed to be independent oracles. One-sided checks cover temporal
eligibility, near/far, alpha skip/cap, early termination, determinant validity,
and tile bounds. The relevant validation report must be reviewed before its
next gate is opened.

For the alpha-cap contract specifically, the focused fixture keeps alpha well
above the `1/255` skip for every perturbation, never crosses the early-
termination boundary, and fixes contributor order, tile membership, and
visibility. It covers raw alpha well below and above `0.99f`, the immediate
float32 values below/equal/above it, RGB/flow/depth/mask separately and mixed,
a nonzero background, and multiple contributors. It observes opacity and at
least one screen-mean or conic gradient. Below and above the cap, finite-
difference perturbations must remain on their own side; equality uses separate
left/right one-sided checks, never a central-difference pass criterion. Before
the Fix, the focused negative regression must isolate the current nonzero
analytic alpha-mediated gradient above the cap against a zero forward finite
difference; after the Fix, that regression must pass. Below-cap gradients must
match the independent oracle and finite difference within the later approved
tolerance; above-cap alpha-mediated gradients must be zero. Direct color,
flow-value, depth-value, and depth-to-screen-z gradients, and all forward
outputs, must remain unchanged. The suite covers 3D and 4D CUDA-direct paths
with `compute_cov3D_python=False`, `convert_SHs_python=False`, exact
`scaling_modifier=1.0`, `env_map_res=0`, and `override_color=None`.

Numeric tolerance is intentionally not guessed here. A future focused-
validation instruction must state dtype, independent-oracle arithmetic,
finite-difference step, and acceptance tolerance. The source Fix, fixture,
CUDA build, and CUDA execution have not been implemented or run.

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
- the post-merge formal renderer invocation identity:
  `compute_cov3D_python=False`, `convert_SHs_python=False`, exact
  `scaling_modifier=1.0`, `env_map_res=0`, and `override_color=None`;
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
branch with `rot_4d=true` and `force_sh_3d=false`; the exact approved renderer
invocation `compute_cov3D_python=False`, `convert_SHs_python=False`,
`scaling_modifier=1.0`, `env_map_res=0`, and `override_color=None`; the pilot-
reachable parts of P0-0 through P0-3; the approved alpha-cap derivative;
the approved completed-update transaction for P0-T1, P0-T2, and P0-T3; one
formal run mode and effective config whose post-merge `eval` is true; a new
non-overwriting output owner; a minimum versioned completed-state checkpoint
foundation; fail-closed finite/failure behavior; and the approved train/test-
only pilot evaluation policy.

Pilot training may start only when:

- the pure formal resolver/validator is the sole effective-state authority,
  finishes in the same process before importing renderer, `Scene`, or CUDA-
  related modules, and every invalid/unsupported input rejects before JIT,
  CUDA, rendering, output-directory creation, or any output write;
- one explicit formal authority binds the approved dataset/model/camera/
  renderer values, verified post-merge `N`, final checkpoint and test/save
  lists derived only from that `N`, and a new output identity; semantic CLI
  overwrite, implicit defaults, multiple authorities, legacy/general config,
  `cfg_args`, resume/warm-start/best/environment checkpoint state, and existing
  output directories all fail closed;
- P0-0 uses one canonical effective-camera state for projection and rasterizer
  forward/backward, every unsupported camera input rejects before GPU, and the
  focused camera forward/gradient validation is accepted;
- P0-1, P0-2, and P0-3 on the selected CUDA-direct branch are corrected in
  source and their focused forward/gradient validation is accepted;
- the alpha-cap piecewise derivative is implemented in source as its separate
  renderer-math responsibility, and its independent CPU-oracle, float32-
  boundary, finite-difference, negative-regression, direct-gradient, unchanged-
  forward, and 3D/4D CUDA-direct focused validation is accepted;
- completed count starts at zero, transactions `k=1..N` fetch exactly `N`
  batches and perform exactly `N` optimizer steps including final `N`, and no
  batch is fetched for `N+1`;
- each transaction applies its LR/SH schedule before forward, captures and
  applies current densification statistics before Parameter replacement, steps
  the backward-owned Parameter, zeroes gradients, and only then performs
  scheduled densification/prune followed by opacity reset;
- clone/split children derive from post-step parents, pruning reads post-step/
  pre-reset opacity, and a simultaneous reset reaches every survivor and new
  child;
- completed state and checkpoint label `k` include the optimizer update and
  every scheduled topology/reset event for `k`; checkpoint save occurs before
  test evaluation of that same state, and final checkpoint `N` is guaranteed
  from the effective post-merge `N`;
- optimization input loss `k` is kept distinct from completed-state test
  metric `k`, and any retained initial-state-zero report is diagnostic-only and
  cannot participate in formal iteration, checkpoint, or selection;
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
The approved policy and this document synchronization alone do not satisfy
Gate A. The pure resolver, pre-heavy-import enforcement path, P0 source fixes,
consumer integration, and validation above are all still unimplemented.

### Gate B: before formal retraining

Gate B requires accepted pilot evidence; the accepted pure resolver and its
same-process pre-heavy-import boundary; one frozen verified state connected to,
but not owned by, the accepted P0-T1/T2/P0-T3 completed transaction; a frozen
canonical camera/eval contract and publication identity, including the
supported camera mode,
effective dimensions/intrinsics/FoV/tan, centered-only disposition, distinct
projection/cull semantics, and post-merge `eval=True`; the frozen and published
five-field renderer invocation identity shared by every formal entrypoint,
with no alternate path able to substitute defaults or reach rendering
silently; frozen remaining formal config and schedule; the exact approved
train/test identity; test non-selection; a pre-fixed numeric final iteration
whose completed state is canonical; accepted exact-N, final-step, same-
Parameter-step, zero-grad-before-mutation, densify/prune-before-reset,
checkpoint-first/test-second transaction evidence; clean source revision;
reproducible build/runtime identity; corrected and validated P0-1/P0-2/P0-3
renderer behavior; approved remaining numeric densification, reporting, seed,
and failure policies; the full versioned checkpoint schema and iteration
transaction; immutable atomic output publication; no legacy checkpoint warm-
start; and no legacy output reuse. A validation-based best checkpoint is not
part of this baseline. Policy approval and document synchronization alone do
not satisfy Gate A or Gate B; source correction, shared enforcement, focused
validation, pilot evidence, and every other listed requirement remain
necessary.

Resume support is not an unconditional Gate-B prerequisite. If it has not been
implemented and accepted as an independent root Fix with the resume-equivalence
gate, every resume entry must fail closed and only an uninterrupted completed
run may be canonical. If formal training will use resume, P0-T5/T6 and complete
continuation state must be closed and the equivalence gate accepted first.
P0-T4's required formal disposition is the approved train/test and
test-non-selection policy, which still requires implementation and validation.
Approved `env_map_res=0` makes P0-T7 unreachable for this baseline but does not
fix it; every environment-map checkpoint or invocation state must fail closed.
Every forbidden branch must fail closed rather than remain a silent option.
Neither Gate A nor Gate B is passed.

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

The independent formal-entry resolver/validator is the approved sole owner of
the complete verified effective state. Investigation #10 Candidate B supplies
the common authority and Candidate A supplies its same-process pre-JIT bootstrap;
they are complementary. Training, evaluation/test render, CUDA Reference, and
checkpoint consumers must consume that same verified identity;
they must not become independent default or policy owners. The exact API,
location, error schema, and import/JIT-before-validation prevention mechanism
remain implementation-open. Configuration enforcement must be a bounded
responsibility separate from the P0-1/P0-2/P0-3 renderer-math Fixes. Checkpoint
and manifest code verify and publish the accepted identity later; neither owns
a second copy of the policy.

| Owner | Root findings/responsibility | Dependent consumer |
|---|---|---|
| common camera contract | P0-0; explicit two-mode validation, raw/effective separation, post-resolution canonical camera identity, projection/rasterizer forward/backward handoff, and pre-GPU rejection; excludes off-center expansion and visibility-semantics retuning | manifest/runtime camera publication in P0-A6 after P0-0 acceptance |
| renderer forward/backward | P0-1 plus conditional-mean P0-2 and spatial-3/temporal-2 fixed-48-slot P0-3 under `rot_4d=true`, `force_sh_3d=false`; this math Fix does not own formal-config enforcement; remaining supported-path renderer P1 is separately bounded | training validation and CUDA Reference semantics |
| alpha-cap derivative (separate renderer math) | Preserve the forward `0.99f` cap and direct value/depth-z paths; gate the aggregated `dL/dalpha` once for the alpha-mediated chain with uncapped `raw_alpha < 0.99f` and equality saturated. The first future candidate is a local `backward.cu` Fix; it is not part of P0-1/P0-2/P0-3 and does not authorize buffer/binding/Python expansion. | focused independent oracle, float32 boundary/branch-parity, finite-difference, negative-regression, unchanged-forward, and 3D/4D CUDA-direct acceptance before Gate A |
| training state machine | P0-T1/P0-T2: exact `k=1..N`, final step, no `N+1` batch fetch, completed labels, checkpoint after every scheduled mutation, and checkpoint-first/test-second completed-state observation | diagnostic checkpoint foundation consumes the label/boundary; P0-A1 verifies it later |
| optimizer/densification transaction | P0-T3: collect current statistics before replacement, same-Parameter step, zero-grad, scheduled topology, then scheduled reset; densify/prune precedes reset when simultaneous | final checkpoint population identity; numeric population policy remains separate |
| dataset/evaluation policy | P0-T4; post-merge effective `eval=True`, exact `v01-v31` train and `v00` test identity, no validation population, test non-selection, and formal rejection of `eval=False` | fixed-final checkpoint selection and P0-A8 |
| evaluation/best policy | Formal reporting is selection-free and the best branch is unsupported; any future validation/best experiment needs a separate policy, identity, and output owner | conditional best lineage in P0-T5/A8 only outside this baseline |
| fixed-final checkpoint selection | Canonical identity is mandatory final checkpoint `N`, constructed from effective post-merge `N` and saved after update/topology/reset at its exact completed boundary, not `chkpnt_best.pth` | P0-A8 manifest and artifact consumers |
| independent formal entry / pure effective-state resolver | Sole owner of one explicit formal authority and verified typed state, independent of `torch`, renderer, `Scene`, and CUDA modules; resolve and reject before heavy import/JIT/output creation; bind approved dataset/model/camera/renderer values, effective `eval=True`, explicit scale/color fields, post-merge `N`, derived final-checkpoint/test/save conditions, and unique output identity; forbid semantic CLI override, competing authorities, defaults, legacy/loaded/best/resume/environment state, existing output, and executable `cfg_args`. Excludes P0-T1/T2/T3, P0-T6, camera/renderer math, seed, reporting, exact resume, and atomic publication. | same-process bootstrap imports heavy modules only after acceptance; training and later evaluation/CUDA Reference consume the state; P0-A2 and manifest publish/verify rather than redefine it |
| diagnostic checkpoint foundation | P0-T6: version, field-level schema, semantic validation, diagnostics, and provenance; consume the P0-T1/T2 completed label without owning loop or optimizer order | P0-A1/A2/A8 consumers |
| exact resume (conditional independent root) | Complete continuation state and equivalence, including any resume-reachable P0-T5/T6 handling, only after known P0/transaction stability; do not merge it into the diagnostic foundation or transaction Fix | formal continuation only after gate acceptance; otherwise fail closed |
| environment-map lifecycle (future conditional root) | P0-T7 remains unfixed but unreachable under `env_map_res=0`; any future enablement requires separate policy and complete lifecycle/checkpoint/optimizer/resume validation | first-baseline CUDA reconstruction must reject environment state; future runtime publication follows only after separate acceptance |
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
   **Complete in this document.**
4. Integrate the approved one-path renderer invocation contract.
   **Complete in this document.** Integrate the approved alpha-cap derivative
   contract. **Complete in this document on 2026-09-06 JST.** Integrate the
   approved completed-update training transaction as a named Phase-0 policy
   work package without creating a new roadmap number. **Complete in this
   document on 2026-09-06 JST.** Integrate the Issue #10 approved independent
   formal-entry/effective-configuration policy under Issue #11, combining the
   common owner with same-process pre-JIT bootstrap. **Complete in this
   document.** Its source Fix and focused validation have not started. After
   document review and the user-owned Git checkpoint, the
   desktop advisor determines the next formal candidate; this document sync
   and CODEX do not select or start it. Camera implementation details, complete
   run mode, numeric densification/population fields, checkpoint schema, and
   other remaining policy fields stay undecided. Confirm one root owner and one
   bounded Fix responsibility at a time only after the applicable policy is
   decided. The formal-entry authority is decided, while its helper/API/location,
   typed state, exact operational CLI allowlist, and integration are not.

### Phase 1: training-critical foundation

5. Implement the approved pure resolution/validation boundary, then its same-
   process pre-heavy-import enforcement, and only afterward integrate the
   verified state into downstream training fixes and formal consumers. Complete
   the still-open run-mode fields, operational CLI allowlist, and non-overwriting
   output identity without creating a second semantic authority.
6. After its field-level schema policy is approved, implement the minimum
   versioned P0-T6 diagnostic checkpoint foundation independently of the
   training-state-machine and exact-resume roots; consume the approved
   completed label/boundary rather than redefining it.
7. Implement the common P0-0 camera-handoff Fix with the approved two-mode,
   centered-only, fail-closed, and distinct projection/cull contract; do not
   mix P0-A6 publication, off-center expansion, or visibility retuning into it.
8. Implement selected-branch renderer forward/backward P0 Fixes.
9. Implement the approved P0-T1/T2 exact-N training-state-machine Fix, including
   final step, no extra batch fetch, completed labels, final-save construction
   from effective `N`, and checkpoint-first/test-second completed observation.
10. Implement the approved P0-T3 optimizer/densification transaction Fix:
    current statistics, same-Parameter step, zero-grad, topology, then reset.
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
18. Run separately owned training-state-machine and optimizer/densification
    mocks for the shared approved event boundary; do not fold P0-T6 schema or
    exact-resume equivalence into these fixtures.
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
  camera/effective-`eval` policies; and
- repository synchronization of the user-approved formal renderer invocation
  contract: `compute_cov3D_python=False`, `convert_SHs_python=False`, exact
  `scaling_modifier=1.0`, `env_map_res=0`, and `override_color=None`; and
- repository synchronization of the user-approved 2026-09-06 JST alpha-cap
  derivative contract, including saturated equality with selected subgradient
  zero and the future focused-validation boundary; and
- repository synchronization of the user-approved completed-update training
  transaction: exact `k=1..N`, final step, no extra batch fetch, current
  statistics before same-Parameter step, zero-grad before topology, densify/
  prune before opacity reset, completed checkpoint `k` before test evaluation,
  mandatory effective-final checkpoint `N`, loss/metric separation, and the
  initial-state-zero boundary; and
- Investigation #10 and repository synchronization under Issue #11 of the
  user-approved independent formal-entry/effective-configuration policy: one
  pure authority, same-process pre-JIT enforcement, Candidate B plus Candidate
  A, Candidate C/D rejection, explicit verified identity, and fail-closed
  exclusion of competing/default/legacy/resume/best/environment/output reuse.

Not complete and not authorized by this document sync:

- remaining formal policy selection listed in Open items;
- source, config, test, or tool fixes, including the formal resolver/bootstrap,
  consumer integration, and P0-T1/T2/T3 transaction fixes;
- focused validation or CUDA build;
- pilot training or formal retraining, including any exact-resume Fix or
  equivalence acceptance;
- a corrected checkpoint, SPL4, CUDA Reference, or population identity;
- Viewer restart, fixed-range parity, full-scene correctness, or Acceptance
  Level 4;
- performance, interactive behavior, scalability, or LOD work.

## Open items

- alpha-cap source Fix, focused validation, runtime float32 forward/backward
  branch-parity confirmation, and validation tolerance; the derivative policy
  itself is decided;
- any other still-undecided supported-path renderer behavior;
- formal-entry implementation details: exact pure resolver helper/API/location,
  typed verified-state representation, error schema, exact operational CLI
  allowlist, consumer integration, and same-process mechanism that completes
  before renderer, `Scene`, or CUDA-related import/JIT;
- camera implementation details: centered-principal-point numerical tolerance,
  field-level validation/error schema, and exact common-builder API/location;
- remaining field-level formal run-mode specification; effective `eval=True`
  is already decided and is not an open item;
- completed-update transaction source Fix, focused validation, and concrete
  loop/helper API; the exact-N, final-step, same-Parameter-step, zero-grad,
  densify/prune-then-reset, completed-checkpoint, and checkpoint-first/test-
  second event order is approved and is not open;
- whether to retain the current unconditional initial-state-zero diagnostic,
  and, if retained, its PNG/TensorBoard/CUDA-cache/failure contract; it remains
  outside formal iteration, checkpoint, test selection, and best selection;
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
- remaining fields of the complete effective-config, dataset, source,
  environment, and build identity; the single-authority/pre-JIT policy is
  decided;
- formal output owner, atomic publication, completion index, and digest;
- new checkpoint iteration and population count;
- SPL4-v2 representation and export identity;
- corrected CUDA Reference output identity;
- manifest schema, runtime-state validator, and direct-evidence boundary;
- strict SPL4 parser and Viewer provenance/bundle binding;
- representative selection and fixed-range design;
- corrected semantic and image thresholds;
- full-scene resource and performance plan;
- governance for any separate future proposal covering Python covariance/SH
  precompute, non-unit scaling, environment maps, or color override: a new
  policy, necessary Fix and validation, and distinct run/artifact identity when
  semantics differ. None is an open adoption choice for this baseline.

No camera, renderer, alpha-cap, SH, backward, training-state, checkpoint,
exporter, parser, CUDA Reference, manifest, or Viewer fix; build; test; CUDA
execution; render; training; export; artifact generation; branch operation;
commit; or push has been performed by this documentation sync. The formal
camera/eval, renderer-invocation, alpha-cap derivative, completed-update
transaction, and independent formal-entry/effective-configuration policies are
synchronized, but their enforcement, source fixes, consumer integration, and
focused validation are not implemented. P0-0, P0-1, P0-2, P0-3,
P0-T1, P0-T2, P0-T3, the separate alpha-cap renderer-math responsibility, and
P0-A6 remain open until their source responsibilities and required validation
are completed and accepted; P0-T7 remains unfixed but unreachable for the
first baseline. Checkpoint field schema/semantic validation remains the P0-T6
diagnostic-foundation root, while exact resume remains a later independent
root.
