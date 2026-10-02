# Corrected Camera Intrinsics Retraining Plan

Status: **Step-5-and-Step-6-complete-and-user-research-Git-integrated / Step-7-update-consistency-A-functionally-accepted / Step-7-document-review-and-user-research-Git-pending / plan-with-accepted-P1-P2-P3-P4-P5-P6-components / Investigation1-4 and Issue-#10/#12/#18 static audit complete / audit integration documented / eight-formal-policy-groups-approved / Issue-#11-policy-sync / Issue-#17-formal-entry-and-4dgs310-sync / Issue-#22-adopted-JSON-and-run-mode-sync / Issue-#24-adopted-partial-field-contract-sync / Issue-#26-adopted-prefilter-policy-sync / Issue-#30-adopted-three-time-contract-sync / Issue-#33-adopted-current-PLY-raw-time-sync / Issue-#34-adopted-initial-time-variance-sync / remaining-formal-policy-open / Issue-#35-P2-component-accepted / Issue-#36-P1-partial-component-accepted / Issue-#37-P3-partial-component-accepted / Issue-#38-P5-partial-component-accepted / Issue-#39-P4-partial-component-accepted / Issue-#41-P6-partial-component-accepted / other-owner-source-fixes-and-validation-outstanding / CUDA-not-run / pilot-not-started / formal-retraining-not-started / Viewer-frozen**

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

### Program-wide role boundary

The program-wide research purpose and project roles are owned by
[`50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md`](../../4dgs-development-governance/50_4DGS_RESEARCH_PROGRAM_GOALS_JA.md).
Within that hierarchy, Corrected 4DGS is the normal image-reconstruction-based
comparison baseline and the reference system supporting Viewer semantic
correctness. It is not a required input, algorithm, or design owner for the
direct converter. The corrected-baseline work in this plan remains necessary,
but it is not itself the central research result of the program.

### Management boundaries (2026-09-20)

Functional Step definitions are owned by
[GOV-STEP](../../4dgs-development-governance/00_GOVERNANCE_AND_ROLE_BOUNDARIES_JA.md),
Step design and documentation timing by
[ADV-STEP / ADV-DOC](../../4dgs-development-governance/10_DESKTOP_ADVISOR_COMMAND_DESIGN_RULES_JA.md),
Git timing by [USR-GIT-09/10](../../4dgs-development-governance/40_USER_GIT_AND_ACCEPTANCE_RUNBOOK_JA.md),
and overall progress by [RM-PROGRESS](../../4dgs-development-governance/30_REDMINE_WORKFLOW_JA.md).
Technical owners, dependencies and Gate conditions remain unchanged. The
current functional Step after completed Steps 5 and 6 is
[Step 7: learning-update consistency and CPU validation](#step-7-learning-update-consistency-and-cpu-validation).
Former roadmap numbers, including 6=checkpoint, 7=camera and 9/10=training
state/optimizer-population responsibilities, remain historical
references rather than the current execution sequence. Future functional Step
numbers, order and scope are not assigned by this synchronization.
The component work in Issues #35–#39 and #41 belongs within the existing Step 5;
those issue numbers and component counts are not overall progress measures or new Steps.
Historical component-level documentation/Git checkpoints below are records, not mandatory
triggers for future checkpoints. Unexecuted instructions must be reviewed against the
current governance and working tree before reuse (ADV-DOC-04).
This management note does not accept a component, complete Step 5 or Gate A, change
formal contracts, start runtime work, or lift the Viewer freeze.

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
- Issue #12's read-only investigation of the formal-entry implementation
  boundary and execution environment is complete. Its independently reviewed
  implementation-boundary and `4dgs310` environment clarifications are
  approved by the user and synchronized here under Issue #17.
- Issue #18's investigation and supplemented independent review are accepted.
  Its strict nested semantic JSON v1 / immutable verified-state choice and
  shared from-scratch semantic mode are user-adopted and synchronized under
  Issue #22. Issue #23's user-adopted P1–P6 partial field contracts are
  synchronized under Issue #24; remaining field/owner/run decisions stay open as
  classified in [the adopted JSON contract](#approved-json-and-single-run-mode-contract).
- Issue #25's D-PREFILTER investigation, PF-A/PF-B/PF-C, and supplemental
  safety conditions are user-accepted and synchronized under Issue #26 in
  [the renderer-owner contract](#approved-temporal-prefilter-contract).
  Policy adoption and the Step 5 Python handoff are accepted; CUDA safety
  validation and later consumer bindings remain open.
- Issue #27's bounded D-TIME investigation is accepted and complete. Its
  user-adopted three input/retention/handoff contracts are synchronized under
  Issue #30 in [the time contract](#adopted-d-time-input-retention-and-handoff-contracts).
  D-TIME as a whole, other D-* owners, runnable v1, Issue #2, Phase 0, and all
  Gates remain incomplete. The accepted Step 5 connection below does not
  complete the time owner's numerical/runtime validation.
- Issues #31 and #32 are accepted and complete as a bounded D-INIT
  investigation and input Validation. Their user-adopted
  [current initial-PLY policy](#adopted-current-initial-ply-reuse-and-raw-time-connection)
  is synchronized under Issue #33. This input Validation is not post-fix
  focused validation, CUDA acceptance, or completion of D-INIT/D-TIME.
- Issue #33 is accepted and complete. Issue #34 synchronizes the subsequent
  user-adopted [initial temporal-variance contract](#adopted-initial-temporal-variance-and-d-time-connection)
  from the Issue #31 revision-2 proposal. This closes that bounded adoption
  question, not parser/model implementation, numerical validation, or the
  remaining D-INIT/time/run decisions.
- The Investigation1-4 findings, dependencies, ownership boundaries, and
  pre-implementation gates are integrated in this document.

Issue #2 synchronizes the user-adopted
[D-DATA minimal connection](#adopted-d-data-minimal-connection-contract) and
[minimal initialization/LR/batch/SH policies](#adopted-minimal-initialization-optimizer-and-sh-contracts),
with the subsequent [configurability/population adoption](#adopted-configurability-and-population-conditions)
and [Step 5 minimal connection contract](#adopted-step-5-minimal-connection-contract).
Issue #50's accepted investigation remains input evidence, not source implementation.
The completed milestone is [Step 5 functional acceptance](#step-5-functional-acceptance):
configuration validation and the minimal same-process connection to existing
runtime are accepted, including reuse of the time/V-B, frame and metadata
components. Earlier component reports retain their historical acceptance stages;
they are not additional current approval or Git prerequisites. Documentation
review and consolidated user-owned research Git are complete, as recorded in
the [Step 5 completion record](../../reports/corrected-4dgs/phase1/step5/step5-completion-report.md).
The [Step 6 camera A function](#step-6-canonical-camera-handoff) is also complete,
including documentation review and user research Git, as recorded in the
[Step 6 completion record](../../reports/corrected-4dgs/phase1/step6/step6-completion-report.txt).
The current [Step 7 A function](#step-7-learning-update-consistency-and-cpu-validation)
is implemented and CPU-validated, advisor-reviewed and user-functionally-accepted.
Review of this document sync, consolidated user-owned research Git, push
confirmation and Step 7 completion processing remain pending. This bounded
acceptance is not training readiness or passage of any Gate.

The Fudan Native model configuration, existing train/test-only dataset and
evaluation policy, checkpoint-foundation/exact-resume staging policy, and
formal camera/effective-`eval` policy were approved by the user on 2026-09-03
JST. The formal renderer invocation policy was approved on 2026-09-04 JST.
The alpha-cap derivative policy was approved by the user and synchronized here
on 2026-09-06 JST. The completed-update training transaction policy was then
approved by the user after independent review and synchronized here on
2026-09-06 JST. The independent formal-entry/effective-configuration policy was
subsequently approved after the Issue #10 investigation and is synchronized
under Issue #11. Issue #12's approved implementation and environment
clarifications refine that formal-entry policy without adding a ninth policy
group and are synchronized under Issue #17. Issue #22 further concretizes that
same formal-entry group with Issue #18's adopted JSON/run-mode choices. Issue
#24 integrates Issue #23's adopted P1–P6 within that group, not a ninth group
or a wholesale adoption of the 62-field register. The subsequent Step 5 minimal
connection adoption closes the JSON required/allowed sets, now enforced in the
accepted Step 5 connection; remaining owner contracts and run values stay open. All eight policy
groups are integrated below. Issue #26 supplements the existing renderer policy
group with the adopted D-PREFILTER contract; it neither reselects the five
renderer values nor adds a ninth group. Issue #30 integrates only the three
adopted D-TIME contracts within the existing formal-entry field/owner boundary,
not a new policy group or Step. Issue #33 applies the accepted current-input
adoption within that same boundary; Issue #34 adds only the adopted initial
variance and its explicit coefficient under existing D-INIT. Remaining formal
policy closure, source fixes, post-fix focused validation, CUDA execution, pilot training, formal
retraining, corrected artifact generation, and Viewer restart remain incomplete.
The implemented and CPU-accepted scope includes the integrated configuration
and minimal existing-runtime connection in
[Step 5 functional acceptance](#step-5-functional-acceptance), built from P1–P6
and time/frame components, and the bounded
[Step 6 camera A handoff](#step-6-canonical-camera-handoff) and
[Step 7 update/population/save-test connection](#step-7-learning-update-consistency-and-cpu-validation).
Remaining camera and transaction runtime/CUDA validation, renderer and checkpoint
corrections and actual GPU/training validation remain outstanding. P0 findings block only the gate whose accepted output
would reach the defect; Viewer-only defects do not unnecessarily block corrected
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
   mathematically by `cx=width/2` and `cy=height/2`; the numerical implementation
   choice is recorded in the reviewed [Step 6 design](#step-6-canonical-camera-handoff).
   A negative raw FoV sentinel may be retained only as
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

P0-0 remains open. Its root responsibility is to validate raw
`CameraInfo` or equivalent metadata once, create the canonical effective-camera
state once, and make projection plus rasterizer forward/backward consume it.
This includes explicit mode selection, input validation, raw/effective
separation, post-resolution canonicalization, identity-consistent handoff,
pre-GPU rejection, and focused validation. It excludes manifest schema and
publication, P0-A6 implementation, Viewer/artifact provenance, scientific
retuning of projection/cull values, off-center expansion, and unrelated SH,
training, or checkpoint findings. The historical policy synchronization itself
did not perform source Fix or validation.

The [Step 6 A connection](#step-6-canonical-camera-handoff) through the real
Python forward/context/backward handoff is now implemented and CPU-validated
with GPU/JIT isolated, advisor-reviewed and user-accepted. This acceptance is
bounded to that function; full P0-0 runtime/CUDA validation and Gate
obligations remain open.

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
are corrected in source and their focused validation is accepted. The Step 5
formal entry now enforces configuration before heavy import and explicitly
connects existing training/render consumers; later CUDA Reference and checkpoint
consumer enforcement and CUDA validation remain outstanding. Future support for any
rejected branch requires separate policy, any necessary Fix and validation,
and a distinct run or artifact identity when semantics differ.

#### Approved temporal-prefilter contract

Issue #25's PF-A/PF-B/PF-C and supplemental safety conditions are approved,
not pending alternatives. They supplement the renderer invocation owner above;
the existing five-field contract and selected Fudan Native branch are unchanged.

- **PF-A — semantics:** additional temporal variance is disabled for the first
  corrected baseline. Preserve the original `Σ_tt` temporal marginal,
  effective-opacity weighting, conditional mean/covariance, and existing
  temporal cull. This does not disable the Gaussian's learned time variance,
  repair nonpositive/nonfinite `Σ_tt`, or resolve D-TIME.
- **PF-B — input:** `renderer.temporal_prefilter` is a required string with
  exactly the lowercase value `"disabled"`. Reject missing, null, bool, number,
  empty string, case/whitespace differences, any other value, and duplicate
  decoded keys. Reject the old `prefilter_var`, separate enabled/variance
  fields, and aliases at any scope, even with `-1`, `0`, `false`, or `null`.
  No trimming, coercion, implicit default, or additional semantic CLI is
  permitted. The P1 fixed-input table records this approved addition; schema
  literal, ten top-level keys, and the existing P1–P6 meanings are unchanged.
  The later [Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
  closes nested key sets; runnable training remains unaccepted.
- **PF-C — handoff:** retain `disabled` in the one immutable verified state.
  From that state explicitly derive, supply, and verify the internal float
  `prefilter_var=-1.0` at model construction and actual renderer/rasterizer
  arguments. Model state, forward's consumed value, and backward's saved `ctx`
  value must agree. Reject missing, wrong-type, wrong-value, nonfinite, or
  mismatched state before GPU execution; do not hide invalid input behind a
  positive-value comparison, `getattr`, `setdefault`, or conversion to disabled.
  Other nonpositive values are not accepted adapter values. Consumers must not
  independently default, mutate, or remerge JSON/CLI/YAML authority.

Invalid formal input rejects before heavy import/JIT. Runtime handoff mismatch
rejects before GPU execution; it is not retroactively a preflight rejection.
The read-only preflight → heavy/JIT/side-effect-free preparation → exclusive
claim → first-writer order and the existing failure-stage matrix remain intact.
Configuration verification alone is not CUDA correctness verification.

Training, test/evaluation, formal CUDA Reference, and checkpoint consumers must
bind the same accepted policy/config identity. Checkpoint/manifest owners must
reject missing or wrong identity rather than accept a legacy checkpoint alone
or a manifest's args/default display as executed-state evidence. Their schema,
publication, and implementation remain separate responsibilities. Renderer
semantics stay with this owner; the resolver does not become a second camera,
time, or mathematics owner.

The [Issue #25 revised report](../../reports/corrected-4dgs/issue-25/issue-25-prefilter-policy-proposal.md)
distinguishes the papers' original time variance from the later optional
prefilter feature in the official implementation. Its reviewed paper evidence
does not establish the added-variance feature or a measured benefit; this is
not a claim about every paper. The implementer's specific motivation and
measured improvement remain unconfirmed. The source-level effect of positive
added variance is broader temporal opacity weighting and potentially changed
cull membership, not guaranteed training stability, image quality, exposure
integration, or normalized convolution.

Unlike the camera defect that evaluated the FoV sentinel `-1` as an angle,
the inspected prefilter consumers add variance only when the argument is
positive: internal `-1.0` does not subtract one from `Σ_tt`. That static branch
is not a safety proof for all consumers or the loaded binary. A lost branch,
direct addition in another consumer, swapped argument, forward/backward
mismatch, stale binary, or default completion can reintroduce the same kind of
semantic error. Explicitly passing `-1.0` is necessary but not sufficient;
the independent value/gradient and source/binary checks
[below](#temporal-prefilter-validation-requirements) are required.

Successful training does not schedule automatic prefilter enablement or an
in-run switch. Only demonstrated need may motivate a separately approved
comparison experiment with its own identity; without that need it stays
disabled. The [acceptance record](../../reports/corrected-4dgs/issue-25/issue-25-acceptance-notes.txt)
and its [revised-report evidence](../../reports/corrected-4dgs/issue-25/issue-25-acceptance-evidence.json)
close D-PREFILTER's adoption question, not implementation, numerical safety,
checkpoint/manifest binding, normal training, Phase 0, or any Gate. Initial
proposal/evidence wording remains historical and is not the revised report's
identity authority.

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

The approved implementation boundary separates a lightweight formal-only
bootstrap, a stdlib-only pure resolver, and the heavy training runtime. The
bootstrap's top-level dependency graph must remain lightweight; only after the
resolver and read-only preflight accept the state may the same process load the
runtime that imports `torch`, the renderer, `Scene`, or CUDA-related modules.
The existing general CLI and legacy configuration path remain historical/non-
formal paths and cannot invoke or bypass this formal authority.

Semantic CLI overrides are forbidden. The resolver must accept exactly one
explicit formal input authority, resolve and validate it once, and reject a
missing or unknown field, an implicit semantic default, duplicate or competing
authorities, and any later semantic overwrite. Strict nested semantic JSON v1
with an immutable verified state is adopted under Issue #18; its remaining
external field contract is classified below. The initial formal CLI
accepts only its configuration locator; `quiet` is excluded because its current
call path is coupled to RNG and CUDA-device side effects. Helper and file names,
local API and error-code details, and internal control structure remain bounded
source-implementation choices for CODEX. A four-file layout is a first
candidate, not an immutable change cap or architecture requirement.

The verified state also carries the
[adopted temporal-prefilter identity](#approved-temporal-prefilter-contract)
without becoming its semantic owner or relying on a consumer default.
The verified state must bind the already approved dataset/evaluation, Fudan
Native model, camera/effective state, and five-field renderer values; effective
post-merge completed-update count `N`; and a unique non-overwriting output
identity. It must prove no resume, warm-start, best-checkpoint, environment-map
checkpoint, legacy output, or pre-existing output-directory state is admitted.
`scaling_modifier=1.0` and `override_color=None` are explicit verified fields,
not call-site defaults. The final-checkpoint condition and test/save lists must
be constructed only from verified post-merge `N` and the approved schedule
inputs, with mandatory final checkpoint `N` included exactly once and test
kept selection-free. This does not move ownership of the P0-T1/T2 state machine
or P0-T3 optimizer/densification transaction into
the resolver; those remain separate source responsibilities that consume the
verified state.

The typed state is limited to the formal semantic authority, approved fixed
values, conditions derived from that authority such as final/test/save
schedules, proof that unsupported branches are absent, and the output identity.
The [adopted seed input and connection](#adopted-step-5-minimal-connection-contract)
bind to this state without moving RNG initialization into the pure resolver.
Repository and dataset digests, runtime/toolchain identity, checkpoint and
manifest state, camera mathematics, RNG execution, reporting, and publication lifecycle
remain with their existing owners and may bind to or verify the state without
becoming fields merely to centralize provenance.

Existing-output rejection is a read-only pre-heavy-import preflight. After that
preflight, heavy loading/JIT and side-effect-free runtime preparation must
succeed before an exclusive output claim is made immediately before the first
writer. Thus a heavy-load or JIT failure does not create the formal output
identity, while a race appearing after preflight still fails closed at the
exclusive claim. Atomic publication, cleanup, and completion remain separately
owned and are not decided by this claim boundary.

The same verified identity is consumed by training and, later, by formal
evaluation and CUDA Reference generation. Those consumers may not reconstruct
values from local defaults. `cfg_args` is neither a complete execution identity
nor a formal input authority, and a formal consumer must not retain an
executable `eval` route for it. Checkpoint/manifest publication may record and
verify the identity, but cannot become another policy owner. The resolver does
not own camera or renderer mathematics, the checkpoint schema in P0-T6, atomic
publication, deterministic seed details, reporting policy, exact resume, or the
P0-T1/T2/P0-T3 transaction.

The pre-connection audit found these legacy/general-path defects (historical
findings, not the current formal-entry implementation status):

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

The accepted Step 5 path now supplies the resolver, immutable state, lightweight
entry, delayed same-process heavy import, consumer handoff and output claim;
it does not reuse that legacy/general authority. See
[the current acceptance record](#step-5-functional-acceptance) for evidence and
limits. [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) now
accepts the initial-diagnostic/loop CPU connection; remaining reporting,
later checkpoint/reference and environment-provenance work retain their owners.

#### Approved JSON and single run-mode contract

Issue #18 Candidate B, **strict nested semantic JSON v1 + immutable verified
state**, is adopted, not merely a first candidate. Its Candidate labels refer
to JSON alternatives, not the separate #10/#12 architecture comparison.
Issue #23's P1–P6 below are user-adopted partial external contracts, synchronized
under Issue #24. These labels identify the proposal clauses, not new P0/P1
finding IDs or policy groups. They refine this formal-entry owner without
transferring the existing dataset, camera, renderer, or lifecycle owners.

Pilot and formal retraining share one from-scratch semantic mode, with separate
schedule, output identity, gate, and acceptance roles. A pilot is not promoted
to a formal checkpoint. Resume, warm-start, and best-selection remain forbidden
under the existing first-baseline contract.

Formal JSON remains the configurable input: read JSON → validate → immutable
verified state → runtime handoff. Immutability forbids reconstruction or
overwrite after verification within one invocation; it does not forbid
changing JSON for a later run within its approved contracts. Keep LR, batch,
initial point count, population thresholds/coefficients, start/end boundaries,
and densification/reset intervals as explicit settings, not replacement code
constants. Validate each run against the adopted or separately approved
types/domains; no run values are selected here. Existing model/renderer and
from-scratch restrictions, strict JSON/P2, and no-default-completion rules
remain binding. This is not permission for unsupported branches or a return
to YAML/CLI merging; the population inputs follow the
[adopted contract below](#adopted-configurability-and-population-conditions).

##### Adopted P1: structure and explicit fixed inputs

The required literals are `schema="corrected_4dgs_training_config_v1"` and
`run_mode="from_scratch"`. The closed top level has exactly ten required keys:
`schema`, `run_mode`, `dataset`, `model`, `renderer`, `initialization`,
`optimization`, `reporting`, `checkpoint`, and `output`; all except the first
two are objects. Separate version/profile inputs, changed hierarchy, flat
legacy names, aliases, and unknown keys are rejected as competing authority.
Required inputs are explicit, never completed from defaults. The following
fixed inputs express the already approved semantics or P1's adopted literals:

| JSON path | Required type and value |
|---|---|
| `dataset.kind` | string `"nerf_transforms"` |
| `dataset.eval` | Bool `true` |
| `model.gaussian_dim` | Int `4` |
| `model.spatial_sh_degree` | Int `3` |
| `model.temporal_sh_degree` | Int `2`, with the P3 derivation below |
| `model.rot_4d` | Bool `true` |
| `model.force_sh_3d` | Bool `false` |
| `model.sh_evaluation` | string `"conditional_mean"` |
| `renderer.compute_cov3D_python` | Bool `false` |
| `renderer.convert_SHs_python` | Bool `false` |
| `renderer.scaling_modifier` | Num exactly `1`; JSON `1` and `1.0` are accepted, not `true` or a string; explicit runtime value `1.0` |
| `renderer.env_map_res` | Int `0` |
| `renderer.override_color` | explicit `null`, corresponding to the approved runtime `None` |
| `renderer.temporal_prefilter` | required string `"disabled"`; Issue #25 PF-B, with [renderer-owned meaning and handoff](#approved-temporal-prefilter-contract) |

The transforms reader cannot switch to Colmap or lego/validation branches by
filesystem discovery. Dataset population retention is still separately
verified. Raw/effective camera values, near/far, projection, digests, manifest,
and checkpoint state are not duplicated in this JSON; accepted owner results
must still be bound or checked. The subsequent
[Step 5 connection table](#adopted-step-5-minimal-connection-contract) closes
the nested required/allowed sets, now enforced in Step 5; this does not complete
all downstream owner validation or runnable training. Any necessary additional field requires prior review
and canonical synchronization, not a generic extension area or tolerated key.
Issue #34 adds the required/allowed nested field
`initialization.time_variance_denominator` under existing D-INIT; its positive
P2 Num domain, explicit first-baseline value, and non-defaulted meaning are
owned by [the adopted initialization contract](#adopted-initial-temporal-variance-and-d-time-connection).
That addition did not change the ten roots; the later table closes the other sets.

##### Adopted P2: strict parsing, types, and limits

- Input is UTF-8 without BOM, containing one root object; leading/trailing
  JSON whitespace is permitted. Reject comments, trailing commas, a second
  trailing JSON value, YAML, and Python expressions.
- Limits are 262144 input bytes, container depth 16 with root at depth 1,
  4096 elements per array, and 4096 Unicode scalar values per string. These
  are parser limits, not training values or GPU-capacity guarantees. Byte/depth
  bounds are not checked only after parsing finishes; excess input must not
  reach heavy import or a writer.
- Reject invalid UTF-8, isolated surrogates, and control characters in paths.
  Do not implicitly normalize Unicode or change case.
- `Int` requires a JSON integer token, not bool, an exponent/fraction token,
  or a coerced string. The common nonnegative integer upper bound is
  2147483647; field-specific lower bounds still apply.
- `Num` requires a JSON number representable as finite binary64, not bool.
  Reject NaN/Infinity, overflow, and underflow from a nonzero token to zero;
  enforce each separately approved domain. `Bool` accepts only JSON
  `true`/`false`, not `0`/`1` or strings.
- Reject missing, unknown, or duplicate decoded keys recursively, including
  differently escaped spellings of the same key; reject fixed-value, type,
  and range mismatches. There is no expression evaluation, implicit coercion
  or default, environment expansion, YAML merge, `cfg_args` eval, legacy
  override, or last-value-wins repair.
- Unsupported fields must be absent: `debug=false` is not an exception.
  Verified inactive runtime-adapter values and the approved explicit
  `renderer.override_color=null` are not permission for other null inputs.
  Immutability covers nested objects and arrays; consumers cannot mutate or
  reconstruct the semantic authority from another input.

Parser/helper/class/error design remains CODEX's later bounded implementation
responsibility. This clause does not select a digest serialization scheme;
GOV-ID and the existing provenance owner retain that responsibility.

##### Adopted P3: one maximum temporal-SH representation

`model.temporal_sh_degree=2` is the sole temporal-SH input. Reject the legacy
`eval_shfs_4d` boolean, another enabled flag, or an independent layout/slot-count
input. Derive and verify the approved 48 slots from maximum spatial degree 3
and temporal degree 2; derive any adapter-required legacy boolean `true` from
that same state. Maximum degrees are distinct from training-time active degrees.
The subsequent [D-SH adoption](#adopted-minimal-initialization-optimizer-and-sh-contracts)
selects the existing staged warm-up, not always-active 3/2. This is separate
from P3's maximum-degree representation and does not change its accepted
component's function. Intermediate renderer correctness remains unvalidated;
an event-count inequality alone cannot prove it.

##### Adopted P4: explicit integer-divisor resolution

`dataset.resolution` is exactly `{"mode":"integer_divisor","divisor":d}`,
with fixed mode and Int `d >= 1` under P2's common bound. `d=1` preserves raw
resolution. Both owner-verified raw width and height must be divisible by `d`,
yielding positive integer effective dimensions. Reject nondivisible dimensions
instead of rounding/truncating them, auto `-1`, an integer interpreted as a
target width, float/arbitrary scale inputs, a second `resolution_scales`
authority, and upscaling. Dataset/camera read-only preflight must verify
declared dimensions against actual images. Canonical camera generation and
intrinsics/projection calculation have one camera owner, not a second resolver
implementation. Actual divisor, interpolation, mask/alpha/depth treatment, and
FoV-only implementation/validation remain with D-DATA/P0-0.

##### Adopted P5: completed-update save and test inputs

`optimization.total_updates=N` is Int in `1..2147483647`.
`checkpoint.intermediate_updates` and `reporting.test_updates` are required
arrays; each may be empty, but omission is not normalized to empty. Both are
strictly increasing Int sequences: intermediate values satisfy `1 <= k < N`,
test values `1 <= k <= N`. Reject duplicates, reverse order, zero, negatives,
out-of-range values, bool, and expressions; do not silently sort or deduplicate.
Derive immutable `save_updates` by appending `N` exactly once to the intermediate
sequence. Independent `final_iteration` or `save_iterations` inputs are
forbidden. Do not automatically add `N` to the test sequence: final-test needs
are decided by reporting/Gate owners for the approved run schedule.

When both are scheduled, checkpoint then test observe the same completed
state, without best selection, early stopping, or parameter tuning. Initial
state-zero diagnostics are outside these schedules and remain a reporting
decision. Empty-array validity does not waive actual formal-run reporting or
cadence approval. This closes checkpoint/test list boundaries only, not
densify/prune/reset/SH/LR event boundaries or numeric run schedules.

##### Adopted P6: locator and output identity

The initial CLI has only one config locator. JSON `dataset.source_path` and
`output.directory` are explicit absolute POSIX paths; reject empty values,
`~`, environment/glob expansion, Windows drive/UNC forms, and path controls.
Windows references used to open a WSL file are not formal Linux path inputs.
Canonicalize paths once, resolving existing symlinks and checking the resulting
identity: dataset must be an existing directory and output a new, non-legacy
directory. Reject existing output, data/output aliases, mutual containment,
or a destination that writes into dataset/legacy territory. A dangling symlink
occupies the output name and is rejected, not followed or overwritten by mkdir.
The explicit `output.directory` is the run output identity; empty-input UUID
or `OAR_JOB_ID` fallback and competing `run_id`, `model_path`, or `output_root`
inputs are forbidden. Any later opaque publication identity belongs to its
separately accepted machine-generated contract, not a new input invented here.

Keep read-only existing-output preflight, then heavy/JIT and side-effect-free
preparation, then exclusive claim immediately before the first writer.
Preflight alone cannot exclude a later race. The failure-stage matrix below
still distinguishes invalid input, heavy failure, and a claim loser; it does
not require every JIT cache or the competing winner's directory to be absent.

##### Adopted D-TIME input, retention, and handoff contracts

Issue #30 synchronizes the three user-adopted contracts based on Issue #27's
T-A. They refine the existing formal-entry field contract and time-owner
handoff, without changing P1–P6 or making the resolver a second mathematics,
initialization, or runtime owner.

- **A — explicit input and one origin-preserving transform:** require
  `dataset.time.raw_interval=[a,b]` with exactly two elements and
  `dataset.time.divisor=d`. All three values are P2 `Num`, with `a<b` and
  `d>=1`; reject bool, string, null, implicit defaults, and type coercion.
  The temporal divisor is not integer-only and is distinct from P4's image
  resolution divisor. Raw time is the coordinate in the approved transforms;
  do not infer seconds or FPS. Derive `t_eff=t_raw/d`, `a_eff=a/d`, `b_eff=b/d`,
  and `L=b_eff-a_eff` once, preserving the origin. Do not auto-normalize to
  `[0,1]`, shift the origin, or accept separate effective-time inputs or legacy
  `frame_ratio`/defaults as another authority. Preserve P2 input rejection;
  reject nonfinite derived values, nonzero-to-zero conversion, endpoint
  collapse, and nonpositive `L`. These binary64 input/derivation conditions
  alone do not prove the later adopted float32 handoff's runtime health or tolerances.
- **B — closed interval and complete frame retention:** validate every frame
  against the declared interval including both endpoints. Any missing,
  invalid, or out-of-range time rejects the whole read-only preflight before
  heavy import/JIT. Do not silently filter, fill with zero, clamp, or round to
  pass. Preserve all approved train 5,146/test 166 mappings by split, source
  frame index, exact path, and raw time; reject missing or duplicate frames and
  changed split membership. Counts or time alone are not frame identity.
  Image identity and loader equivalence remain D-DATA; sampler/batch use
  remains D-OPT.
- **C — common verified coordinate handoff:** Camera timestamp, Gaussian time,
  renderer timestamp, and temporal-SH duration must consume the same verified
  time coordinate. Consumers must not divide again or reconstruct defaults.
  Initial PLY raw/effective provenance and units must be proven before use;
  perform only the necessary transform and reject unknown provenance rather
  than infer units from a similar range. Do not introduce a unit override or
  alias. The frame interval does not determine the Gaussian-center generation
  domain or the initial temporal-scale/`Σ_tt` formula.

The following boundaries remain unresolved, not implicitly adopted:

- D-INIT retains actual GPU initialization/health validation beyond the
  accepted seed/cast CPU connection. The
  [current-PLY adoption below](#adopted-current-initial-ply-reuse-and-raw-time-connection)
  resolves that input's reuse/raw-coordinate evidence, and the
  [adopted initial variance below](#adopted-initial-temporal-variance-and-d-time-connection)
  resolves the V-B choice and coefficient contract. The subsequent
  [minimal D-INIT adoption](#adopted-minimal-initialization-optimizer-and-sh-contracts)
  separately decides initial-center closed-interval validation, sampling,
  extra points, and rejection of missing/time-absent/unknown input; it is not
  inferred from the earlier frame rule. The later
  [Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
  adopts seed and time/cast handoffs, now connected and CPU-validated in Step 5;
  actual GPU cast/activation acceptance remains separate.
- Time/runtime/renderer owners retain binary64-to-float32 representability,
  time collapse and tolerances at that boundary, compiled-binary identity,
  and actual CUDA validation; binary64 preflight is not post-cast proof.
- D-OPT and related owners retain the positive-loss branches' time increment
  `0.1` and knn conditions for future separately adopted use, run values, and
  normalization-change effects. The initial rigid/motion-zero branch is now
  adopted in the Step 5 connection contract. Constant temporal
  LR and staged active SH are now adopted below and their settings connected;
  intermediate-degree forward/backward correctness remains outstanding.
- Every run's interval/divisor and other numeric values remain unselected.
  Observed `[0,33]` and legacy YAML are evidence, not approved execution values.

Mapping the interval and each frame once is not a same-timestamp double-
division bug. Changing the divisor can affect current initial time variance
and loss increments; it is not guaranteed to be a training-invariant unit
change. This is separate from re-adopting additional prefilter variance:
[PF-A/B/C and their safety conditions](#approved-temporal-prefilter-contract)
remain unchanged. T-B's `d<1` expansion, T-C's automatic normalization, and
Issue #27's entire proposed validation matrix are not adopted. The three
contracts do not complete D-TIME, other D-* owners, runnable v1, Issue #2,
Phase 0, or any Gate, nor establish implementation or measured acceptance.

Evidence: [Issue #27 investigation](../../reports/corrected-4dgs/issue-27/issue-27-investigation-report.md),
[advisor review](../../reports/corrected-4dgs/issue-27/issue-27-advisor-review.md),
and [user acceptance and three-contract approval evidence](../../reports/corrected-4dgs/issue-27/issue-27-completion-evidence.json).
Adoption-pending wording in those historical reports describes the pre-approval
state; it neither reverses this adoption nor adopts their remaining proposals.

##### Adopted current initial PLY reuse and raw-time connection

The user accepted Issue #31's investigation and Issue #32's input Validation
after independent review and approved the following three points, synchronized
here under Issue #33. This is a bounded application of the existing D-TIME
A/C contract within Phase 1 Step 5, not a new policy group, Step, or owner.

- **A — reuse the verified current input:** continue using
  `/home/demo/work/data/4dgs_sph_scene/points3d.ply` as the Corrected initial
  input. This adoption is bound to the current input identified by the
  [Issue #32 machine evidence](../../reports/corrected-4dgs/issue-32/issue-32-validation-evidence.json);
  it does not authorize an unknown replacement. The observed 6,310,009
  vertices do not select a run's initial point-count input. Sampling is now
  separately owned by the [minimal D-INIT adoption](#adopted-minimal-initialization-optimizer-and-sh-contracts).
- **B — recognize its dataset raw coordinate:** this PLY's time is the
  float32 representation of the dataset raw coordinate shared with the
  approved transforms. The generating rule is `(frame-35)/5` for frames
  35..200. Issue #32 verified all rows from the 166 source PLYs, preserving
  xyz/RGB bytes and row/concatenation order with the expected float32 time;
  the Windows and WSL combined PLYs match in all bytes. All train 5,146/test
  166 metadata mappings match by frame, exact path, split, time, and source
  camera metadata. JSON binary64 versus PLY float32 is a representation
  distinction, not adoption of a runtime tolerance or rounding repair.
- **C — apply the existing one-transform handoff:** the generating FPS
  division has already occurred; do not repeat it on stored time. Apply the
  separately selected formal divisor through the existing origin-preserving
  `t_eff=t_raw/d` derivation exactly once as needed. Camera, Gaussian,
  renderer, and temporal SH must consume the common verified coordinate,
  without consumer redivision, default reconstruction, or a new unit
  override/alias. This adds no new transform. Generating FPS=5 does not
  select formal `d=5`, and observed `[0,33]` does not select a run interval.

Issue #31's initial limited-scope provenance gap is supplemented, for this
adoption only, by its
[history review](../../reports/corrected-4dgs/issue-31/issue-31-chat-history-review.md)
and the [Issue #32 validation report](../../reports/corrected-4dgs/issue-32/issue-32-validation-report.md).
Content reproduction is not proof of the exact historical process, date,
package environment, or complete copy history. Neither that complete history
nor the original simulation's physical seconds is an additional adoption
condition. SPH particle identity, physical-quantity preservation, image-content
identity, and loader equivalence are not proven by these results.

Acceptance trace: [Issue #31 investigation](../../reports/corrected-4dgs/issue-31/issue-31-investigation-report.md),
[Issue #32 independent review](../../reports/corrected-4dgs/issue-32/issue-32-advisor-review.md)
and [review evidence](../../reports/corrected-4dgs/issue-32/issue-32-advisor-review-evidence.json),
plus the user-acceptance/adoption records for
[Issue #31](../../reports/corrected-4dgs/issue-31/issue-31-completion-evidence.json)
and [Issue #32](../../reports/corrected-4dgs/issue-32/issue-32-completion-evidence.json).
Earlier unknown-provenance or approval-pending wording records its historical
stage; it neither reverses this adoption nor adopts the remaining proposals.

The remaining D-INIT/time/runtime/run choices listed above stay open.
Issue #33 did not select an initial variance; the subsequent
[Issue #34 adoption below](#adopted-initial-temporal-variance-and-d-time-connection)
now owns that bounded choice. The subsequent
[minimal D-INIT adoption](#adopted-minimal-initialization-optimizer-and-sh-contracts)
decides the initial-center domain and sampling/extra/time-absent boundary.
The later [Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
adopts the time/cast handoff. Formal divisor/interval, remaining runtime
tolerances and validation, and local helper/API/file/calculation implementation
choices remain separate from that adoption.
The frame closed-interval rule alone did not define the initial-center domain;
ordinary initial temporal variance is distinct from additional prefilter
variance. Legacy reader/model behavior is not thereby formal-compliant.
The bounded verified consumer integration is now accepted in Step 5. Remaining
D-INIT/D-TIME numerical/GPU validation and owner/run decisions are outstanding;
this does not complete runnable training, Phase 0, or any Gate.

##### Adopted initial temporal variance and D-TIME connection

Issue #34 synchronizes the user-adopted revision-2 contract under existing
D-INIT / `initialization`, within Phase 1 Step 5. It adds no policy group,
Step, or D-* owner and does not reselect the current-PLY/raw-time adoption.

- **A — raw initialization and explicit coefficient:** for the explicit
  `dataset.time.raw_interval=[a,b]`, define `L_raw=b-a` and
  `v_raw=L_raw/c_init`. The formal field
  `initialization.time_variance_denominator` is required and allowed for the
  shared pilot/formal entry. It is a positive P2 `Num`: a finite binary64
  JSON number, including positive fractions, not bool, string, null, zero,
  negative, NaN/Infinity, overflow, or nonzero-token-to-zero underflow.
  The first baseline must explicitly supply `5`; omission must not recover
  the old default. Retain the value in the one verified effective configuration.
  A future separate run may explicitly change it; no search values or
  experimental runs are selected here. There is no alias, generic extension
  area, new recording mechanism, or initialization-method selector.
  This is a heuristic tied to the fixed dataset raw coordinate, not a
  universal dimensionless constant, physical seconds, or an optimum.
  A unit-bearing interpretation is `v_raw=L_raw*tau_init`, where
  `tau_init` has numeric value `1/c_init` in raw-time units (initially
  `0.2`). Do not transfer the same numeric coefficient unconditionally to
  another raw unit or infer `L_raw` from observed extrema/frame-number gaps.
- **B — adopted V-B, one coordinate mapping:** connect to the existing
  origin-preserving `t_eff=t_raw/d`, `d>=1`, through
  `s_raw=sqrt(v_raw)`, `s_eff=s_raw/d`,
  `v_eff=L_raw/(c_init*d^2)`, and stored `ell_t=log(s_eff)`.
  With `c_init=5,d=1` this equals the current initialization formula.
  For `d>1` it differs from the current effective heuristic
  `L_raw/(5*d)` (V-A). V-B preserves the initial marginal ratio
  `(delta_t_eff)^2/v_eff=(delta_t_raw)^2/v_raw` as a coordinate mapping
  in real arithmetic; it does not promise training-wide or image-quality
  invariance, including LR/loss, sampling, SH, and numerical effects.
  This adoption does not retrospectively label V-A a bug prohibited by the
  earlier D-TIME contract.
  `c_init`, `dataset.time.divisor`, and the generating PLY FPS=5 are
  separate authorities. Do not repeat the FPS division or add consumer
  redivision/default reconstruction. The equations specify meaning, not
  evaluation order, helpers, or APIs; direct computation of `d^2` is not
  required if a bounded implementation preserves the contract.
- **C — initial state and fail-closed health:** preserve from-scratch,
  `qL=qR=(1,0,0,0)`, initial `R=I`, and `scaling_modifier=1`.
  Under these conditions initial `Sigma_tt=s_eff^2` and space-time cross
  covariance is zero. Stored log parameter, activated scale, and variance
  are different quantities. Dividing only the time scale is not a general
  transformation of a rotated Gaussian. The general relation
  `A=diag(1,1,1,1/d)`, `Sigma_eff=A*Sigma_raw*A^T` explains that boundary;
  it does not authorize checkpoint conversion or reuse.
  At the necessary input, derivation, and handoff stages, require finite
  positive `c_init`, raw/effective durations, variance, and scale, finite log
  values, finite positive scale/`Sigma_tt` after runtime conversion and
  activation, and matching row counts with `N x 1` temporal shape.
  Reject intermediate overflow/underflow and invalid
  results rather than hide them with defaults, epsilon, clamps, or rounding
  repair. Binary64 acceptance does not prove float32 health. Distinguish
  pre-JIT-observable checks from runtime checks; concrete float32 tolerances,
  time collapse, compiled-binary identity, and CUDA validation remain with
  their existing owners. The existing preflight/heavy/claim/writer order stays
  intact.
- **D — paper, code, adoption, and learning are distinct:** the reviewed
  [paper v3, section 4.2](https://arxiv.org/html/2310.10642v3#S4.SS2)
  and [Appendix E](https://arxiv.org/html/2310.10642v3#A5) describe initial
  scale `L/2`, hence variance `L^2/4` under the initial conditions above.
  The reviewed official code and current local source instead use
  `sqrt(L/5)` and `L/5`. These generally differ; the derivation/optimality
  of 5 and the reason for the discrepancy remain unproven. Replacing 5
  with 2 does not produce the paper formula.
  Duration-relative V-C with `kappa=1/2` remains an unadopted, unmeasured
  comparison candidate; V-B and the explicit initial `c_init=5` are the
  adopted contract, not proof of paper reproduction or local performance.
  Per-Gaussian `_scaling_t` is already learned using `scaling_lr` in the
  current source. The initialization hyperparameter is neither that learned
  parameter nor its learning rate; this adoption changes no LR/optimizer
  policy. Ordinary initial temporal variance is not additional prefilter
  variance; PF-A/B/C and their safety conditions remain unchanged.

Evidence and adoption trace:
[Issue #31 revision-2 proposal](../../reports/corrected-4dgs/issue-31/issue-31-initial-time-variance-policy-proposal.md)
and [proposal evidence](../../reports/corrected-4dgs/issue-31/issue-31-initial-time-variance-proposal-evidence.json)
record the paper/public-code comparison and inspected local source identity;
[Issue #31 investigation](../../reports/corrected-4dgs/issue-31/issue-31-investigation-report.md)
records the bounded source facts and earlier alternatives.
The proposal's adoption-pending wording is its presentation-time record.
The subsequent user adoption is recorded in
[Issue #34's description](../../reports/corrected-4dgs/issue-34/issue-34-description.md)
and [creation approval](../../reports/corrected-4dgs/issue-34/issue-34-creation-evidence.json),
after [Issue #33 acceptance/Git completion](../../reports/corrected-4dgs/issue-33/issue-33-completion-evidence.json).
Those reports are not rewritten or treated as this plan's permanent owner.

Only this field's required/allowed membership, domain, initial value, and the
A–D contract are adopted here. The subsequent
[minimal-policy adoption](#adopted-minimal-initialization-optimizer-and-sh-contracts)
separately decides initial-center/sampling/extra/input restrictions, LR/batch
semantics, and active SH. The later
[Step 5 connection contract](#adopted-step-5-minimal-connection-contract) adopts
the closed key sets, time/cast handoff and initial loss branch. Remaining
runtime collapse/tolerances, other D-* contracts, and each run's interval/
divisor, initial point count, and other values remain open. Observed `[0,33]` selects
neither a run interval nor a Gaussian-center domain. Step 5 now accepts parser
enforcement and initial V-B/model handoff on CPU, not actual GPU cast/activation
or numerical CUDA acceptance. D-INIT/D-TIME as a whole, runnable training,
Phase 0 and all Gates remain incomplete.

##### Adopted D-DATA minimal connection contract

Within Issue #2's existing Phase-0 policy responsibility, the user adopted the
following two fields and existing-owner connection in the
[acceptance record](../../reports/corrected-4dgs/issue-2/issue-2-d-data-acceptance.md).
This is the bounded adoption of the
[minimal-change correction](../../reports/corrected-4dgs/issue-50/issue-50-minimal-change-correction.md),
not the old A/B/C proposals or Issue #23's whole field register. Under
ADV-DOC-02, this permanent prerequisite is synchronized before the affected
Step 5 implementation; it does not create a Step or a Git checkpoint trigger.

| Required/allowed JSON path | Accepted input and rejection | Existing-owner handoff |
|---|---|---|
| `dataset.extension` | Required P2 String, exactly `""`. Reject missing, null, non-string, and every non-empty value. The empty string is explicit input, not default completion. | Pass the verified value to the existing loader; use the approved extension-bearing `file_path` without suffix addition, replacement, or alternative-name search. Issue #23's unadopted non-empty-suffix proposal is not adopted. |
| `dataset.white_background` | Required P2 Bool, exactly `false`. Reject missing, null, numeric `0`, strings, and `true`. | Teacher-image loading and renderer background consume the same verified value; this does not decide D-POP reset conditions or other owners' policies. |

The first formal path is fixed to the existing delayed loader (internal
`dataloader=true`) and existing corresponding sidecar masks. Do not add an
external loader selector, mask-source enum, PNG-alpha substitute, or mask-
generation branch to JSON. Bind these accepted owner conditions to the single
verified state; consumers must not reconstruct them from legacy defaults.
Local helpers, APIs, and internal implementation structure remain CODEX's
later bounded implementation choices.

Preserve RGBA background compositing → uint8 RGB → existing resize/tensor
processing, and sidecar L conversion → `>=128` → 0/1 →
`gt_alpha_mask` handoff with the existing background-opacity-suppression loss.
Do not newly multiply teacher RGB by the sidecar, substitute the fully opaque
PNG alpha for that mask, or silently replace the existing processing order.

The accepted independent
[Issue #50 investigation](../../reports/corrected-4dgs/issue-50/issue-50-investigation-report.md)
and [advisor checks](../../reports/corrected-4dgs/issue-50/issue-50-advisor-review-checks.json)
confirm all 5,312 referenced images and masks exist and decode at 1280×720,
matching metadata, with image alpha entirely 255 and mask values only 0/255.
Complete generation provenance and all-frame semantic proof remain unproven;
neither another image scan nor generation-history search is an added adoption
prerequisite. Issue #50's investigation acceptance is distinct from Issue #2's
policy responsibility; earlier adoption-pending wording is superseded only
within the acceptance record's scope.

Fallback removal, generic mask resizing, and correction of the shape branch
unreached at the current original resolution are not additional mandatory
implementation tasks of this adoption. This does not permit formal connection
to silently switch from the specified masks to another input. Preserve the
existing input-check and same-state handoff duties; address a run-dependent
inconsistency separately only when its actual necessity is established.

P4 is unchanged. Actual run divisor, interpolation/downscaled alignment, loss
coefficients and other unresolved D-* contracts stay undecided; the later
[Step 5 connection table](#adopted-step-5-minimal-connection-contract) closes key sets.
Observed legacy `resolution=1` and `lambda_opa_mask=0.01` are not approved run
values. Only these two fields' required/allowed membership and the owner
connection are newly adopted: the existing 14 P1 fixed entries, P1–P6/PF/time/
V-B, ten roots, schema, Fudan Native, from-scratch, test non-selection, and
Viewer freeze remain unchanged. These fields and the minimal formal consumer
connection are now enforced and accepted in Step 5; D-DATA as a whole,
run-dependent alignment, runnable training and Gate A remain incomplete.

##### Adopted minimal initialization, optimizer, and SH contracts

Issue #2's [minimal-policy acceptance](../../reports/corrected-4dgs/issue-2/issue-2-minimal-policy-acceptance.md)
adopts only the following bounded contracts and retained owner boundaries.
The [remaining-contracts proposal](../../reports/corrected-4dgs/issue-2/issue-2-step5-remaining-contracts-proposal.md)
and [source evidence](../../reports/corrected-4dgs/issue-2/issue-2-step5-remaining-contracts-evidence.json)
are the adoption trace, not a wholesale implementation specification. Their
historical pending wording is superseded only for this accepted subset.
This is Phase-0 policy closure within the existing formal-entry group, not a
new policy group, Step, full nested schema, or source/runtime acceptance.

**D-INIT — current input and sampling.** Require the approved current
time-bearing PLY under the [existing input identity](#adopted-current-initial-ply-reuse-and-raw-time-connection).
Reject missing input, unknown replacement, or time-absent input; do not fall
back to random point generation or dataset writes. Require
`initialization.num_extra_pts` as P2 Int exactly `0` and
`initialization.num_pts` as P2 Int strictly positive, retaining P2's common
upper bound. For input row count `M > num_pts`, preserve existing sampling
with replacement; for `M <= num_pts`, use all rows. Do not substitute sampling
without replacement, deduplication, or a new sampler. Actual initial row count
and distinct source-row count are different; no numeric `num_pts`, including
the legacy 600000, is selected here.

Validate initial input time in the declared closed interval in its approved
raw coordinate, then apply the existing one-transform handoff. The old
sampling-only endpoint-exclusion filter is not part of the formal path.
Preserve xyz/RGB/time row correspondence; do not filter or clamp invalid input
into validity. This initial-center closed interval is this D-INIT adoption,
not a consequence retroactively attributed to the frame contract. It adds no
training-time center clamp, GPU Gaussian-generation method, or knn formula.
Seed and time-cast integration are CPU-accepted in Step 5; actual GPU
cast/activation validation remains outstanding.

**D-OPT / D-DELAY — existing LR groups and batch semantics.** Preserve xyz's
existing log interpolation, a constant temporal-position LR, and the existing
group sharing: feature rest uses `feature_lr/20`, spatial and temporal scale
share `scaling_lr`, and qL/qR share `rotation_lr`. This retains existing
behavior; 20 is not claimed as a paper-required or optimal coefficient. Do
not add a temporal schedule or independently tunable shared-group rates.
The corresponding paths are under `optimization.learning_rate` as mapped in
the [Issue #23 register](../../reports/corrected-4dgs/issue-23/issue-23-formal-json-field-contract-proposal.md):
`position_lr_init`, `position_lr_final`, `position_t_lr_init`,
`position_lr_max_steps`, `feature_lr`, `opacity_lr`, `scaling_lr`, and
`rotation_lr`. This path mapping adopts only the stated meanings, not the
whole register; the later Step 5 connection table closes required/allowed sets.

Issue #51's [four-LR acceptance](../../reports/corrected-4dgs/issue-51/issue-51-lr-domain-acceptance.md)
requires `feature_lr`, `opacity_lr`, `scaling_lr`, and `rotation_lr` under
`optimization.learning_rate` as P2 Num values `>=0`. Apply the existing
[P2 number-token and finite-binary64 checks](#adopted-p2-strict-parsing-types-and-limits),
including nonzero-token-to-zero underflow rejection. Integer, fractional, and
exponent number forms follow P2 Num; reject strings, bool, null, negative and
nonfinite values, and overflow. Do not impose the P2 Int upper bound or an
additional LR-specific upper bound. Zero is permitted in the input domain,
not selected as a run value here: each run must supply explicit JSON values,
without YAML/default completion. Preserve the existing group assignments and
sharing above, including `feature_lr/20`; add no new schedule or independent
shared-group rates. These four input domains are adopted and enforced in the
accepted Step 5 resolver/consumer connection. Passing them does not guarantee training
numerical stability or CUDA correctness; zero LR does not guarantee whole-model
invariance through pruning or opacity reset. This adopts neither the whole
Issue #23 register nor other fields' undecided contracts.

`position_t_lr_init` must be an explicit nonnegative P2 Num; reject negative
fallback such as `-1` rather than substituting xyz's initial LR. The xyz
endpoints must both be positive or both zero; reject one-sided zero.
`position_lr_max_steps` is a positive P2 Int; do not automatically equate it
with `N`. Actual LR values and any run's use of zero LR remain run decisions.
Delay is explicitly disabled. Exclude external `position_lr_delay_mult` from
the allowed set as inactive, including its proposed nested path; do not
restore the legacy default as authority. Supply internal `delay_steps=0`
explicitly from that same verified disabled contract wherever the API needs
it. No new delay mechanism is adopted.

Require `optimization.batch_size` as P2 Int in `1..5146`. Preserve existing
shuffle, workers `0`, and `drop_last=True`; reject a batch larger than the
train population rather than permit an empty-loader loop. Keeping the full
5,146-frame train dataset is distinct from discarding a partial final batch
on each pass. This does not select a new sampler or `drop_last=False`.
Preserve L1/SSIM and the D-DATA sidecar background-opacity-suppression loss.
The basic P2 Num domains, `lambda_dssim` in `0..1` and other loss weights
nonnegative, are not adoption of every loss branch or of actual run weights.
The subsequent [Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
requires explicit `lambda_rigid=lambda_motion=0` for the initial formal path,
not a permanently zero-only numeric type. Future positive-branch use and its
time increment `0.1`/knn conditions remain separately unadopted.

**D-SH — active schedule, not a different model.** Adopt the existing stages
`0/0 → 1/0 → 2/0 → 3/0 → 3/1 → 3/2` (spatial/temporal active degrees),
starting at `0/0` and reaching the maximum in five increment events. Maximum
degrees `3/2` and the 48-slot layout are unchanged; intermediate active
degrees do not select a reduced model/layout. Require explicit
`optimization.sh.increase_interval` as positive P2 Int. Increase before the
render of the associated update. The checkpoint owner records actual active
state; this adoption invents no checkpoint field names, schema, or API.
The run owner must choose a schedule reaching maximum degrees before formal
result observations. Do not require every short diagnostic smoke to execute
five events, or adopt the legacy huge interval or a numeric `N` here.
Intermediate-active forward/backward validation remains with the renderer
owner and Gate A; policy adoption or event count is not correctness proof.

**Retained boundaries and undecided choices.** The subsequent
[D-POP adoption below](#adopted-configurability-and-population-conditions)
now owns stopping-threshold/unlimited semantics, existing spatial selection
and 4D split, external inactive-temporal-threshold exclusion, and event/prune/
reset conditions, now with the adopted eight-field JSON input contract.
Its input enforcement and bounded consumer handoff are accepted in Step 5;
screen-statistics correction and transaction CPU validation are accepted within
[Step 7 A](#step-7-learning-update-consistency-and-cpu-validation), not actual GPU/training acceptance.
It does not adopt a strict cap or a new temporal selection method.
Maintain the direction of one seed owner before sampling, shuffle, and
Gaussian creation, no later reseeding, and no implicit device fallback.
The [Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
now adopts seed input/state binding, the single-device baseline, and time/cast
handoff, now connected on CPU. Actual run seed, device evidence, remaining
tolerances and runtime/CUDA validation remain outstanding.
Do not add a root object or semantic CLI override to fill those gaps.

P5/P6 retain reporting/output schedule and claim ordering. A Scene writer
must not precede exclusive claim; the necessary preparation/writer separation
does not authorize a full Scene redesign or new logging infrastructure.
The later closed JSON sets and optional-TensorBoard/initial-diagnostic boundary
do not complete remaining reporting/failure details, checkpoint schema/publication,
or actual run values; these remain with their existing owners. A full metric/PNG/
visualization specification is not an added Step 5 prerequisite. Step 5 owns
configuration, immutable verified state, lightweight bootstrap/heavy-runtime
handoff, read-only preflight, and claim, not camera/renderer/loop/transaction
Fixes. Missing downstream implementations remain integration dependencies;
mocks alone cannot prove actual connection or training readiness. The accepted
Step 5 CPU evidence uses existing consumers with GPU boundaries isolated;
runnable training and Gate A remain incomplete.

##### Adopted configurability and population conditions

Issue #2's [configurability/population acceptance](../../reports/corrected-4dgs/issue-2/issue-2-population-config-acceptance.md)
adopts the following meanings with explicit representation and implementation
limits. The [old population proposal](../../reports/corrected-4dgs/issue-2/issue-2-population-policy-proposal.md)
and [static evidence](../../reports/corrected-4dgs/issue-2/issue-2-population-policy-evidence.json)
are historical evidence, not an unconditional specification. In particular,
the proposed mandatory-positive-only threshold and blanket no-cap rejection
are withdrawn. Under ADV-DOC-02, this sync prevents configuration hardcoding
and that withdrawn restriction from becoming prerequisites for Step 5; it
creates neither a Step nor a Git/push trigger.

**Point-count condition.** With a finite configured threshold `T`, the current
count `M < T` permits existing clone → split → prune at the applicable event;
`M >= T` takes prune-only. One growth event may exceed `T`. If subsequent
pruning reduces the count below `T`, growth may resume at a later applicable
event: reaching `T` is not a permanent stop. Initial `M >= T` does not cause
truncation. This is a growth-stopping threshold, not a strict point-count or
VRAM guarantee; add no candidate ranking/allocation or forced excess deletion.
Keep the existing unlimited choice available: it disables this count-based
growth stop, not event schedules or pruning conditions. Neither a concrete
threshold nor initial point count is selected here.

**Adopted population JSON inputs.** The subsequent
[population JSON acceptance](../../reports/corrected-4dgs/issue-2/issue-2-population-json-acceptance.md)
adopts the recommended contract in the
[population JSON proposal](../../reports/corrected-4dgs/issue-2/issue-2-population-json-contract-proposal.md).
That proposal's pending label records its presentation-time state; its compared
alternatives are not adopted. Under ADV-DOC-02, synchronize this permanent input
contract before implementation, not as a claim of implementation completion.
This does not adopt the whole Issue #23 register or old population proposal.

`optimization.population` is required and has exactly the following eight
required/allowed fields. Every value is explicit; reject omission, unknown or
duplicate decoded keys, implicit defaults, and legacy YAML/CLI merging.
P2's common Int/Num/String meanings and numeric health checks remain unchanged:
Int uses an integer token, and Num requires finite binary64 with nonzero-token-
to-zero underflow rejected. No actual run values or defaults are selected here.

| Field under `optimization.population` | Accepted type and range | Existing-consumer meaning |
|---|---|---|
| `densify_stop_points` | P2 Int `1..2147483647` or exact String `"unlimited"` | Count-based growth condition; the bounded handoff below targets `densify_until_num_points`. |
| `percent_dense` | P2 Num `>0`, with no added upper bound of 1 | Spatial clone/split scale boundary multiplied by extent, not a percentage. |
| `thresh_opa_prune` | P2 Num in `[0,1]`, both endpoints included | Strict `<` comparison with post-step/pre-reset activated opacity. |
| `densify_grad_threshold` | P2 Num `>=0` | Existing spatial-gradient `>=` selection. |
| `densify_from_update` | P2 Int `0..2147483647` | Exclusive start boundary, corresponding to legacy `densify_from_iter`. |
| `densify_until_update` | P2 Int `0..2147483647` | Exclusive end boundary, corresponding to legacy `densify_until_iter`. |
| `densification_interval` | P2 Int `1..2147483647` | Growth/prune event interval. |
| `opacity_reset_interval` | P2 Int `1..2147483647` | Reset interval and the existing size-condition start boundary. |

For `densify_stop_points`, reject zero, negative numbers, null, Bool,
fraction/exponent number tokens, numeric strings, and strings other than the
exact lowercase `"unlimited"`, including case or whitespace variants. Do not
coerce values, interpret omission as unlimited, or accept `point_cap` or
`densify_until_num_points` as external aliases. Do not add a negative-number
exception to P2's common `require_int`. JSON `-1`, null, zero-as-unlimited,
and a separate mode/threshold selector are not adopted alternatives.
The eight-field closure is local to population; the subsequent Step 5 connection
table closes the other sets without changing this contract.
Passing these input domains does not guarantee stable training,
a nonempty population, VRAM fit, or correction of existing bugs. A zero opacity
prune threshold disables only that opacity comparison, not size pruning.

**Adopted bounded handoff (Step 5 connection accepted).** Retain the validated finite
integer or explicit unlimited state in the one immutable verified state.
Only when connecting that state to the existing `densify_until_num_points`
consumer, explicitly convert unlimited to internal `-1`; pass a finite value
unchanged. Do not rely on the old default, pass the String into numeric
comparison or CUDA, or propagate a negative sentinel to any other setting.
This specifies the one-way handoff, not a local helper/type/API design or a
generic union framework. Step 5 implements and CPU-validates this setting
handoff, not population events or their transaction correctness. Neither P2
alone nor the accepted connection establishes a training-ready path.

**Selection and 4D split.** Preserve spatial-gradient/spatial-scale selection
and clone-before-split order. Gradient at least the configured threshold and
maximum spatial scale `<= percent_dense * extent` selects clone; above that
scale boundary selects split. Gradient threshold and `percent_dense` remain
configurable; the latter multiplies extent, not a reinterpreted 0..100 percent.
Preserve `rot_4d=true` four-dimensional rotation and xyzt splitting, including
time components. No new temporal-selection method is added. Exclude inactive
`densify_grad_t_threshold` from formal external settings; do not make it look
effective merely by accepting a value. This does not authorize removal of
internal time statistics or APIs. Keep the already forbidden unused
`final_prune_from_iter` excluded.

**Events, opacity pruning, and reset.** For approved update labels `k=1..N`,
`from`, `until`, `interval`, and `reset_interval` below denote the verified
`densify_from_update`, `densify_until_update`, `densification_interval`, and
`opacity_reset_interval` respectively, not additional inputs:

- growth/prune occurs only when `from < k < until` and `k % interval == 0`;
  both endpoints are exclusive;
- opacity reset occurs when `k < until` and `k % reset_interval == 0`,
  independently of `from`, so it can occur before the growth-start boundary;
- retain `white_background=false`; do not add the white-background-only reset;
- do not introduce `from < until`, `until <= N`, or an event in every short
  smoke as an additional acceptance condition. With `from >= until`, there
  are no growth/prune events, but independent reset events may still occur;
  with `until=0`, neither event occurs. Do not add a disabling selector or
  require materializing every event in an array.

Keep the approved transaction: current statistics → same-Parameter optimizer
step → zero-grad → topology/prune → reset → completed-state declaration.
Preserving current selection does not adopt the old pre-optimizer topology
ordering. Opacity pruning removes points whose post-step/pre-reset activated
opacity is strictly below the configured threshold; prune precedes a
simultaneous reset. Preserve reset's current `min(opacity, 0.01)` meaning,
not assignment of every opacity to 0.01 or raising smaller values.

The current size conditions enable screen radius `>20` and maximum spatial
scale `>0.1 * extent` when `k > reset_interval`. These are code-derived
conditions; the corrected screen-statistics path has only the bounded Step 7
CPU acceptance below. The reset value 0.01, screen threshold 20, size coefficient 0.1,
split child count 2, and scale-shrink coefficient 0.8 are code-derived, not
paper-required/optimal or permanently unchangeable constants. Keeping the
existing adjustable settings is not a mandate to parameterize every internal
constant in this work.

**Population dependency: bounded implementation/CPU acceptance.** The pre-fix
postfix cleared all `max_radii2D` before growth's final screen-size prune.
The existing population owner (former responsibility 10) retains ownership;
[Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) now accepts
the correction, child-radius treatment, row correspondence and statistics-window
lifetime in implementation and CPU tests. These are no longer unselected
details; actual GPU measurement, training quality and optimality remain unproven.

Step 5 retains only configuration verification, immutable state, lightweight
bootstrap/heavy-runtime handoff, read-only preflight, and claim. It does not
absorb topology, optimizer, or renderer Fixes. The population input and bounded
handoff contracts are adopted and their bounded integration is accepted in Step 5.
The subsequent Step 5 connection adoption below closes bounded seed/loss/time
and key-set choices, not remaining owner details or run values. Gate A remains
incomplete; Step 7's population/transaction CPU acceptance is not GPU/training
acceptance or a transfer of those responsibilities into Step 5.

##### Adopted Step 5 minimal connection contract

Issue #2's [connection acceptance](../../reports/corrected-4dgs/issue-2/issue-2-step5-connection-acceptance.md)
adopts the bounded choices in the
[minimal connection decisions](../../reports/corrected-4dgs/issue-2/issue-2-step5-minimal-connection-decisions.md).
Their proposal-time pending labels are superseded only for this accepted scope.
The [Step 5 boundary approval](../../reports/corrected-4dgs/issue-2/issue-2-step5-boundary-acceptance.md)
and [reuse/boundary correction](../../reports/corrected-4dgs/issue-2/issue-2-next-step-minimal-correction.md)
remain binding. The withdrawn S/L/C/R/K proposal and whole Issue #23 register
are not adopted. This is ADV-DOC-02 permanent-contract synchronization before
implementation, not a new Step, ticket, progress increment, or push trigger.

**Seed input and one initialization.** Require `initialization.seed` as P2 Int
in `0..2147483647`, retained in the one immutable verified state. It is one
run-wide seed despite its initialization path. Pass it once to the existing
Python/NumPy/Torch initialization before the first RNG use, including PLY
sampling, Scene shuffle, and DataLoader use. Do not fill omission with 6666 or
0, reseed per stage, or let later `safe_state()` reset it to 0. Reuse the
necessary `setup_seed`/`safe_state` connection; do not build a new RNG framework
or comprehensively rewrite stdout utilities. Use the existing single logical
GPU `cuda:0` baseline, separating device-suitability checks from RNG
initialization. Add no device selector or silent CPU/alternate-device fallback.
Runtime/RNG execution and actual GPU recording stay with their existing owner,
not the stdlib resolver. Numeric seed, actual device evidence, and CUDA
reproducibility/validation remain outstanding; equal seeds do not prove
bitwise CUDA determinism or exact resume.

**Four loss inputs and the initial formal branch.** Require the four fields
under `optimization.loss`: P2 Num `lambda_dssim` in `[0,1]`, and P2 Num
`lambda_opa_mask`, `lambda_rigid`, and `lambda_motion` each `>=0`. Separately
from those common numeric domains, the initial formal path requires explicit
zero for rigid and motion; reject positive values as unsupported, never coerce
them to zero. Keep existing fields and regularization code: this is not a
permanent zero-only type. Future positive-branch use, including time increment
`0.1` and knn conditions, requires separate adoption and validation by the loss
owner. Preserve existing `(1-w)*L1+w*(1-SSIM)` and sidecar background-opacity
suppression, not a replacement full BCE. This disables neither ordinary learned
time variance/centers, 4D rotation, nor temporal SH. Actual dssim/opacity-mask
weights remain unselected; legacy 0.2/0.01 are not defaults or adopted run values.

**One time transform and V-B initialization handoff.** Pass the existing
`formal_config_time` binary64-derived effective interval into training and
the existing `formal_frame_time` identity-corresponding effective timestamps
into the loader. Do not redivide them or reconstruct raw JSON defaults.
For initial PLY, reuse the existing reader, validate every raw input row in
the declared closed interval, preserve xyz/RGB/time correspondence and the
adopted replacement-sampling contract, and widen each stored float32 time to
binary64 without changing its value. Compute `raw/divisor` exactly once and
pass that effective time as float32 to the existing Gaussian consumer; no
second conversion at Gaussian generation. Do not snap, round-repair, or add
arbitrary epsilon to erase frame/PLY original-precision differences, and do
not require their two representations to be bitwise identical.

Pass the already derived V-B stored log-scale as float32 to the existing
GaussianModel `_scaling_t` initial parameter; do not reconstruct it from the
old effective duration/5. The explicit denominator field, first-baseline 5,
and future run configurability remain as adopted above. Preserve required
finite/positive duration, scale and variance, finite log, row count and temporal
shape checks at CPU derivation and actual cast/activation. Reject numerical
states losing the required interval/correspondence; CPU component success is
not runtime/CUDA proof. Remaining necessary tolerance/health validation stays
with time/runtime owners; no all-LR/all-scalar float32 conversion or generic
tolerance configuration is introduced. Local helpers/APIs/calculation details
remain implementation discretion. Do not reimplement the loader or Gaussian
generation; learned parameter updates retain their existing owner.

**Closed required and allowed JSON keys.** Every listed key is both required
and allowed in its object, with no additional key. Preserve all previously
adopted types, fixed values, and cross-field conditions. Reject missing,
unknown, duplicate decoded keys, old aliases, implicit defaults, and legacy
override authority under P2. This table closes membership, not all owner
contracts, runtime integration, or runnable-v1 acceptance.

| Object | Required and allowed keys |
|---|---|
| root | schema, run_mode, dataset, model, renderer, initialization, optimization, reporting, checkpoint, output |
| dataset | kind, source_path, eval, extension, white_background, resolution, time |
| dataset.resolution | mode, divisor |
| dataset.time | raw_interval, divisor |
| model | gaussian_dim, spatial_sh_degree, temporal_sh_degree, rot_4d, force_sh_3d, sh_evaluation |
| renderer | compute_cov3D_python, convert_SHs_python, scaling_modifier, env_map_res, override_color, temporal_prefilter |
| initialization | num_pts, num_extra_pts, time_variance_denominator, seed |
| optimization | total_updates, batch_size, learning_rate, loss, population, sh |
| optimization.learning_rate | position_lr_init, position_lr_final, position_t_lr_init, position_lr_max_steps, feature_lr, opacity_lr, scaling_lr, rotation_lr |
| optimization.loss | lambda_dssim, lambda_opa_mask, lambda_rigid, lambda_motion |
| optimization.population | densify_stop_points, percent_dense, thresh_opa_prune, densify_grad_threshold, densify_from_update, densify_until_update, densification_interval, opacity_reset_interval |
| optimization.sh | increase_interval |
| reporting | test_updates |
| checkpoint | intermediate_updates |
| output | directory |

The population eight-field contract and explicit unlimited-to-internal-`-1`
handoff to its sole designated consumer remain unchanged. No TensorBoard,
PNG/metric, runtime device information, provenance, or new root input is added.

**Existing reporting and implementation boundaries.** Keep TensorBoard's
existing use-if-available behavior and create its writer only after exclusive
claim. Absence alone is not a new failure condition; this does not authorize
swallowing writer I/O failures. Do not adopt blanket removal of the existing
initial-state-zero diagnostic. Keep it separate from completed `test_updates`,
which never includes zero; remaining necessary diagnostic/reporting/failure
conditions stay with that owner. Do not make a complete metric, PNG, or optional
visualization redesign a Step 5 prerequisite.

Step 5 combines adopted checks/derivations, immutable authority, the lightweight
entry's same-process handoff to existing heavy runtime, and preflight/claim.
Reuse the existing loader, GaussianModel, renderer, loss, and logger; limit
changes to necessary local separation and argument connections. It is not a
new training/renderer implementation or an absorption of later camera,
renderer, loop, transaction, checkpoint, or artifact responsibilities.
The Step 5 CPU validation distinguishes invalid-input no-heavy/no-writer rejection,
same-state handoff without overwrite, one time transform/V-B without old-formula
reconstruction, existing-output/race rejection, and optional TensorBoard
absence from actual I/O failure. Its accepted evidence and existing-consumer
coverage are consolidated in [Step 5 functional acceptance](#step-5-functional-acceptance),
not tests run by this synchronization. GPU boundaries were isolated; CUDA
correctness and training readiness are not proven. Remaining run values and
owner details stay open. Step 5 documentation review and user research Git
are complete; the next selected function is the
[Step 6 camera handoff](#step-6-canonical-camera-handoff), not another Step 5 Fix.

##### Remaining field and owner boundary

Separate ownership does not itself complete a runnable contract. D-PREFILTER
is [adopted](#approved-temporal-prefilter-contract), with the Step 5 Python
handoff accepted; numerical/CUDA validation and later consumer binding remain
outstanding. The
[three D-TIME contracts](#adopted-d-time-input-retention-and-handoff-contracts)
and their [adopted current-PLY connection](#adopted-current-initial-ply-reuse-and-raw-time-connection)
are decided, as is the [initial-variance contract](#adopted-initial-temporal-variance-and-d-time-connection).
Their remaining time/owner/run boundaries are still open. The
[D-DATA minimal connection](#adopted-d-data-minimal-connection-contract) is also
adopted; remaining dataset/mask and runtime conditions still require
accepted owner contracts rather than default substitution. Effective `eval=true`
alone does not prove that all
approved train/test frames were retained; time filtering or another implicit
subset change is not allowed. The
[minimal initialization/LR/batch/SH contracts](#adopted-minimal-initialization-optimizer-and-sh-contracts)
are also adopted, as are the bounded
[population conditions](#adopted-configurability-and-population-conditions).
The [Step 5 minimal connection contract](#adopted-step-5-minimal-connection-contract)
also adopts seed binding, initial loss branches, time/cast handoff, and the
complete required/allowed key sets. Their bounded enforcement and CPU connection
are accepted in Step 5; population/statistics CPU connection is accepted in
[Step 7 A](#step-7-learning-update-consistency-and-cpu-validation). Actual
GPU/runtime health, remaining reporting/failure details and run values stay open. P3 fixes the
maximum-SH input; the separate D-SH adoption selects staged active degrees,
whose renderer correctness still requires validation.
None of these clarifications reselects the approved Fudan Native branch.

| Required point | Still undecided | Existing owner boundary |
|---|---|---|
| Before the corresponding implementation | Only remaining owner details beyond the adopted contracts, including necessary runtime health/tolerance and reporting/failure boundaries; closed key membership, seed binding, initial loss branches and time/cast handoff are not open choices or a requirement to redesign all reporting | Formal-entry field policy with the relevant existing owners; advisor review and user approval precede any new policy. Local helper/API/file/error design remains CODEX discretion. |
| Before downstream execution acceptance | Remaining D-DATA, D-TIME, D-INIT, D-OPT, D-POP, D-SH, D-SEED-DEVICE and D-REPORT responsibilities beyond the accepted Step 5/6/7 CPU scopes; GPU numerical health and actual-runtime transaction validation are not established by CPU connection | Dataset/camera/mask, time, initialization, renderer, optimizer/population, SH schedule, determinism/runtime and reporting retain their responsibilities. Do not turn their outstanding work into additional Step 5 Fix conditions or replace owner results with defaults; applicable Gate A/B requirements remain. |
| Before each run | Numeric seed, `N`, batch, LR and non-fixed loss weights, time interval/divisor, pilot/formal schedules and population stopping-threshold/unlimited choice within the adopted input contract, test/save cadence, actual output path/run identity | Each existing owner and the user fix explicit run values; actual GPU evidence and CUDA validation remain runtime duties. Legacy YAML values are not automatically adopted. |

The D-* definitions and investigation-to-field mapping remain in the
[Issue #23 proposal](../../reports/corrected-4dgs/issue-23/issue-23-formal-json-field-contract-proposal.md).
Its historical D-PREFILTER adoption-pending entry is superseded only for that
decision by [Issue #25's adopted contract](#approved-temporal-prefilter-contract).
Its D-TIME adoption-pending mapping is superseded only for the
[three contracts above](#adopted-d-time-input-retention-and-handoff-contracts);
the initial-variance subset of D-INIT is superseded by
[Issue #34's adoption](#adopted-initial-temporal-variance-and-d-time-connection).
The D-DATA subset is superseded only by the
[minimal connection above](#adopted-d-data-minimal-connection-contract).
The initialization/LR/batch/SH subset is now superseded only by the
[minimal-policy adoption above](#adopted-minimal-initialization-optimizer-and-sh-contracts).
The population subset is now superseded only by the
[configurability/population adoption](#adopted-configurability-and-population-conditions),
including withdrawal of mandatory-positive-only/no-cap rejection and the
subsequently adopted eight-field JSON contract, not wholesale adoption of the
historical proposals. The Step 5 minimal connection adoption supersedes the
seed/initial-loss/time-handoff/key-set pending entries only for its stated scope.
All other remaining owner decisions stay open.
The inspected caller omits `lr_delay_steps`, whose default is zero, so
`position_lr_delay_mult` does not act on that path. D-DELAY's adoption is now
explicitly disabled with the external inactive field excluded as specified
above; the Step 5 formal connection enforces it. The static finding does not
prove every historical run's behavior.

Unspecified run values and undecided validation specifications are different
gaps. Do not fill either with defaults, TBD/null, empty placeholder objects,
tolerated unknown keys, or generic extensions. P1–P6 do not finish runnable v1,
authorize source Fix or focused validation, or pass Gate A/B. The remaining checkpoint,
publication, and environment responsibilities retain their separate gates.

Acceptance trace: [Issue #18 advisor review](../../reports/corrected-4dgs/issue-18/issue-18-advisor-review.md),
[detailed result journal](../../reports/corrected-4dgs/issue-18/issue-18-investigation-journal-105.txt),
and [user acceptance record](../../reports/corrected-4dgs/issue-18/issue-18-acceptance-notes.txt).
The 62-field register is investigation evidence, not a copied or wholly adopted
schema. Issue #23's [partial acceptance record](../../reports/corrected-4dgs/issue-23/issue-23-acceptance-notes.txt)
adopts P1–P6 only and is synchronized here under Issue #24. Earlier proposal or
evidence wording such as adoption-pending records its historical point in time;
it does not undo that acceptance or promote the remaining D-* candidates.

#### Approved execution-environment clarification

[`environment.yml`](../environment.yml) is the unchanged upstream-release
specification and records Python 3.7.13, PyTorch 1.12.1, torchvision 0.13.1,
torchaudio 0.12.1, and CUDA Toolkit 11.6 as historical provenance. Python 3.7
compatibility is not a formal
entry requirement, and `environment.yml` is not the execution authority. The
stdlib-only resolver boundary exists to prevent pre-acceptance PyTorch/CUDA and
other heavy imports, not to support Python 3.7.

The first candidate environment for pilot training, formal retraining,
corrected CUDA Reference generation, and CUDA focused validation is
`4dgs310`. Reproducible commands use
`/home/demo/miniconda3/envs/4dgs310/bin/python` and verify that the running
`sys.executable` is that interpreter before relying on it. This candidate
selection does not authorize an environment or package change.

Current read-only evidence records Python 3.10.14, PyTorch 2.3.1, torchvision
0.18.1, torchaudio 2.3.1, NumPy 2.2.6, OmegaConf 2.3.0, PyTorch CUDA 12.1,
cuDNN 8.9.2, system nvcc 12.0.140, and an NVIDIA GeForce RTX 3080 Ti Laptop
GPU. These are observed current values, not implicit permanent pins for driver
or host state and not permission to update packages alongside a source Fix.

Conda history, installed-extension timing, TensorBoard events, `cfg_args`, and
checkpoint timing strongly support that the historical checkpoint was produced
under `4dgs310`, but no same-invocation command or manifest binds the checkpoint
to its executable, package state, source revision, or loaded extension binary.
That relationship remains a strong inference, not formal provenance.

Final toolchain acceptance requires a fresh build from the corrected tracked-
clean source and focused CUDA validation. It must record the actual Python,
package, PyTorch CUDA, compiler, driver, GPU, source revision, and loaded
extension-binary provenance. In particular, the observed PyTorch CUDA 12.1 and
system nvcc 12.0 separation is evidence to validate, not a reason to silently
modify the environment or reuse an existing extension cache as authority.

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

The pre-fix loop incremented `iteration` before processing a batch and executed
`optimizer.step()` only while `iteration < opt.iterations`. From scratch, the
pre-fix source processed labels `2..N`, called the optimizer at most `N-2`
times, omitted the final update at label `N`, and fetched an extra batch before
breaking at `N+1`. The approved contract instead starts with completed update
count zero and executes transactions `k=1..N` exactly once each, with one
batch fetch and one optimizer step per transaction, including `k=N`, and no
batch fetch for `N+1`. Requested count, transaction label, and completed update
count must have this single consistent meaning. Implementation and existing-loop
CPU validation are accepted in [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation).
Actual device/CUDA and real-data training validation remain open; this bounded
acceptance does not close the whole P0-T1 root or authorize pilot training.

### P0-T2: checkpoint iteration is not a completed-state boundary

Pre-fix evaluation and checkpoint save occurred before densification, opacity
reset and the optimizer step. Such a checkpoint labelled with that
iteration did not represent the completed state transition named by the
label. Under the approved contract, completed state `k` exists only after the
optimizer update and every scheduled topology and opacity-reset event for
transaction `k` have succeeded. A checkpoint labelled `k` saves that completed
state, never the state before those events. Checkpoint save precedes test
evaluation when both observe the same completed state. Final checkpoint `N` is
mandatory, and its save condition must be constructed and verified from the
effective `N` after CLI/config merge. Atomic write, incomplete-state recovery,
and field-level checkpoint schema remain separate open responsibilities.
[Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) accepts the
implemented save-then-test connection and CPU observation of completed state.
Actual runtime validation remains open; the Scene.save/capture observer does
not prove serialization, read-back, publication or resume, or close P0-T2 unconditionally.

### P0-T3: Gaussian gradients are lost on densification iterations

Pre-fix densification replaced optimizer-owned Gaussian parameter tensors before
the same iteration's optimizer step. The freshly installed parameters did not
own the just-produced backward gradients, silently losing the Gaussian update
on those iterations. Densification topology helpers
may still preserve record alignment; the defect is the transaction ordering,
not an SPL4 record-order defect. The approved transaction collects current
visibility, radii, screen-space gradient, and time gradient and applies the
required densification statistics before any Parameter replacement. It then
steps the same Parameter identities that received backward gradients, performs
optimizer zero-grad, and only then executes scheduled densification/clone/
split/prune followed by scheduled opacity reset. Implementation, real Adam and
population CPU connection are accepted in
[Step 7 A](#step-7-learning-update-consistency-and-cpu-validation). Actual CUDA
gradients, device execution and real-data training remain unverified; P0-T3 is
not unconditionally closed by that CPU acceptance.

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
an implicit skipped opacity update. The
[adopted population conditions](#adopted-configurability-and-population-conditions)
specify stopping-threshold/unlimited semantics, selection, and event/prune/reset
conditions without adopting the old pre-step topology order. Actual thresholds
and intervals remain run decisions. Step 5 accepts only inputs and setting
handoff; [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation)
accepts transaction/statistics correction and CPU connection, not actual GPU validation.

Optimization input loss `k` is computed by the forward from completed state
`k-1` and is the input that produces optimizer update `k`. A completed-state
test metric `k` is obtained by newly rendering completed state `k` after its
optimizer, topology, and reset events. Moving training-loss output processing
after transaction completion does not turn that loss into a completed-state
metric. Test remains selection-free. Checkpoint-first and test-evaluation-
second does not decide run acceptance after evaluation failure, atomic
publication, or a general partial-failure policy.

Initial state `0` is not an optimizer update, formal iteration, completed
checkpoint, or test-selection state. The
[Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
does not adopt blanket removal of the current `training_report(0)` diagnostic.
Keep it selection-free and separate from completed `test_updates`; remaining
necessary PNG/TensorBoard/CUDA-cache/failure conditions stay with reporting,
not a prerequisite to redesign that entire subsystem in Step 5.

Candidate B is the approved contract because it applies the current gradient
to the same Parameter before existing topology helpers replace that Parameter,
and therefore needs no new gradient-transport mechanism. Candidate A, the
pre-fix order, does not close P0-T1/T2/T3. Candidate C, pre-step mutation plus
gradient transfer, would require new clone/split/prune/reset gradient mappings
without need. Candidate D, step then checkpoint before mutation, would omit
transaction `k`'s scheduled topology/reset from checkpoint `k`. This selection
rests on completed-transaction consistency and bug isolation, not identity
with official code. Helper boundaries, loop syntax, and local names remain
implementation decisions. That historical policy synchronization did not
implement the contract; the later Step 7 A implementation/CPU acceptance is
recorded separately below. Step 7's A/B scope comparison does not rename the
earlier transaction-order Candidate B.

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
`p_view.z > 0.2` near-cull and has no far-cull. Numerical centered-camera and
common-builder implementation choices are in the reviewed
[Step 6 design](#step-6-canonical-camera-handoff), not additional policy decisions.
Their Python implementation and CPU validation are accepted within that
bounded A function; actual device/CUDA validation remains outstanding. The fixed
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
policy question. The camera policy above is decided and its
[Step 6 A implementation/CPU connection](#step-6-canonical-camera-handoff) is
accepted; full P0-0 runtime/CUDA validation remains open. P0-1, P0-2, and P0-3
source fixes and validation have not started.

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
| camera/projection policy | Support only complete centered intrinsics and complete geometrically valid FoV-only input; reject mixed/partial/ambiguous/nonfinite/off-center input; keep raw sentinel separate from canonical effective state; preserve projection `0.01`/`100.0` and CUDA near-cull `0.2` with no far-cull. One common builder owns execution; P0-A6 separately owns later publication. | Preserve the accepted [Step 6 A implementation/CPU scope](#step-6-canonical-camera-handoff), with actual device/CUDA validation still open; any future off-center or visibility-semantics change needs separate policy and validation. |
| renderer branch policy | Implement conditional-mean SH, spatial degree 3, temporal degree 2 with the fixed 48-slot layout, `rot_4d=true`, and `force_sh_3d=false`; enforce `compute_cov3D_python=False`, `convert_SHs_python=False`, exact `scaling_modifier=1.0`, `env_map_res=0`, and `override_color=None` before any formal renderer; apply the separately owned alpha-cap piecewise derivative to the alpha-mediated chain while preserving direct value/depth-z paths; resolve other supported-path findings separately. | Determinant epsilon, radius inflation, empty population, overflow, radius-threshold diagnostics, and separately identified future support for rejected invocation branches. |
| optimizer/learning-rate policy | Enforce the [adopted xyz/constant-t LR, group sharing, disabled delay, and batch contract](#adopted-minimal-initialization-optimizer-and-sh-contracts); apply the [adopted initial four-loss branch](#adopted-step-5-minimal-connection-contract) separately from common numeric types. Non-fixed run weights and future positive rigid/motion use remain unselected. | Log effective per-group LR without adding a second schedule owner. |
| densification/population policy | Enforce the [adopted stopping-threshold/unlimited, eight-field input/handoff, spatial-selection/4D-split, inactive-field exclusion, and event/prune/reset contracts](#adopted-configurability-and-population-conditions), preserving post-step topology and prune-before-reset. Step 5 accepts setting handoff; [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) accepts population/statistics implementation and CPU validation, not a new strict-cap algorithm or GPU/training acceptance. | Threshold overshoot, nonfinite accumulator frequency, and memory pressure remain bounded pilot observations, not VRAM guarantees. |
| evaluation/metric policy | Preserve the approved train/test identity, create no validation population, keep test selection-free, and use the pre-fixed final completed checkpoint. Test-report metric/cadence and clamp/channel/background remain open; any future best branch needs a separate experiment policy. | Keep diagnostic train samples separate from test reporting and visualization. |
| config/run mode/provenance | Require effective `eval=True`, reject effective `eval=False`, apply the adopted strict nested JSON / immutable state and single from-scratch mode, and enforce the [adopted closed key sets](#adopted-step-5-minimal-connection-contract) without legacy override authority; remaining owner details and integration are separate. Reject legacy output/path reuse, capture the complete effective config, and replace executable config parsing where it reaches formal tooling. | Report path-remap and duplicate-basename ambiguity as bounded diagnostics. |
| training state machine | P0-T1/T2 own exact `k=1..N`, the final step, no `N+1` batch fetch, completed labels, and the completed-state checkpoint/test boundary. [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) accepts implementation/CPU connection, not full runtime closure. | Keep loop/helper API local; consumers verify rather than redefine completed count. |
| optimizer/densification transaction | P0-T3 owns current statistics capture, same-Parameter step, zero-grad, then topology and reset mutation, with densify/prune before reset when simultaneous. | Population owner retains the adopted event/input/handoff conditions, statistics lifetime/point correspondence, and actual threshold/interval/schedule values; [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) accepts bounded transaction implementation/CPU validation, with actual GPU/training validation still open. |
| diagnostic checkpoint foundation | P0-T6 owns versioned field semantics, semantic validation, provenance, and diagnostic state; it consumes the P0-T1/T2 completed label but does not own loop order or exact resume. Atomic writes and field-level state remain to be defined. | Record interruption/OOM behavior and checkpoint-size/hash cost. |
| exact resume (conditional) | After known P0 and transaction stability, a separate root Fix owns complete restore and resume equivalence; until acceptance, resume fails closed. | Compare scheduler/RNG/sampler/dataloader, optimizer, population, and topology continuation only if resume will be enabled. |
| dataset/mask | Bind the approved `v01-v31` train and `v00` test transforms/images/masks without adding validation; preserve the [adopted delayed-loader/sidecar connection](#adopted-d-data-minimal-connection-contract) and its existing RGB/mask/loss meaning through the single verified state. | Issue #50 supplies bounded current-input evidence, not complete generation/semantic proof; remaining run-dependent alignment and loss values stay with this owner, without automatic fallback-removal or generic-resize work. |
| PLY/legacy v1/diagnostic export | Exclude 3D-sequence PLY from formal parity; exclude lossy SPL4-v1; either specify PLY SH property order and metadata or keep PLY diagnostic-only. | External SuperSplat PLY convention remains unverified. |
| manifest/publication | Require immutable output, temporary write and atomic rename, final external digest, completion index, required-field validation, correct debug vocabulary, PNG encoding, and `num_rendered`/tile-reference policy. | Full-file rehash cost and diagnostic metadata volume are performance/observability issues. |
| Viewer parser/handoff | P0-A3/A4 own strict parsing and provenance binding; additionally decide canonical scale representation and remove or version-isolate historical hard-coded selection. | Structured fetch failure and allocation/overflow guards are bounded robustness. |
| determinism/failure handling | Enforce adopted one-seed initialization while retaining runtime ownership, define fatal nonfinite behavior, partial-output cleanup, and the deterministic claims allowed for CUDA and equal-depth sorting. | Runtime nonfinite/OOM frequency and equal-depth ordering remain evidence gaps. |
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
- a same-invocation binding between the historical checkpoint and `4dgs310`,
  including its executable, package state, source revision, and loaded
  extension binary;
- the historical checkpoint's internal tensor state, shapes, finite values,
  optimizer state, and internal iteration, because the large checkpoint was
  not loaded during the audit;
- the exact source/build relationship of the currently loaded CUDA binary;
- the corrected checkpoint, population count, representative selection, and
  fixed range, none of which exist yet;
- post-fix camera and renderer runtime state;
- full-scene `num_rendered` and tile-reference population;
- post-fix forward and gradient correctness;
- acceptance of the current `4dgs310` candidate for corrected work through a
  fresh corrected-source build and focused CUDA validation;
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
| decided: renderer invocation | Require one effective contract everywhere: `compute_cov3D_python=False`, `convert_SHs_python=False`, `scaling_modifier=1.0` exactly, `env_map_res=0`, and `override_color=None`; reject every other or nonfinite value before renderer import/CUDA JIT or formal render; training, evaluation/test render, CUDA Reference, and checkpoint consumers share the same validated identity and may not substitute defaults. The [adopted temporal-prefilter supplement](#approved-temporal-prefilter-contract) adds PF-A/B/C and safety conditions within this owner without redefining these five values. This decision does not accept the current CUDA-direct renderer, which remains blocked on P0-1/P0-2/P0-3 source fixes and focused validation. |
| decided: alpha-cap derivative | Keep `alpha=min(0.99f, raw_alpha)` in forward; after aggregating all `dL/dalpha`, use the independent piecewise gate `raw_alpha < 0.99f` for the alpha-mediated opacity/G/screen-xy/conic/covariance chain and zero that chain for `raw_alpha >= 0.99f`, including a zero selected subgradient at bitwise-equal float32 `0.99f`; preserve direct color/flow/depth and depth-to-screen-z gradients. Do not use an STE/surrogate, cap removal, smooth cap, or cap-triggered formal rejection. Source Fix and focused validation remain required. |
| decided: completed-update transaction | Start from completed count zero and execute `k=1..N` exactly once with no `N+1` fetch; apply schedules before forward; forward/loss/backward; collect and apply current densification statistics before Parameter replacement; same-Parameter optimizer step including `k=N`; zero-grad; scheduled densify/clone/split/prune; scheduled opacity reset; then declare completed state `k`, save checkpoint `k`, and evaluate that same state. Densify/prune precedes reset when simultaneous; prune reads post-step/pre-reset opacity; children derive from post-step parents; reset reaches survivors and children. Final checkpoint `N` is mandatory from effective post-merge `N`. Optimization input loss `k` remains distinct from completed-state test metric `k`; initial state zero is not an update or selection state. Candidate B is adopted and A/C/D are rejected for the bounded reasons above. [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) accepts implementation and CPU validation; actual GPU/training and separate checkpoint-owner validation remain required. |
| decided: independent formal entry/effective configuration | Use a lightweight formal-only bootstrap, one stdlib-only pure resolver/validator, and a separately loaded heavy runtime. The resolver is the sole authority for one explicit formal configuration and completes read-only validation in the same process before heavy import/JIT. Issue #18's strict nested semantic JSON v1 + immutable verified state and single from-scratch mode, refined by Issue #23's adopted P1–P6 under Issue #24, are fixed under [the JSON contract](#approved-json-and-single-run-mode-contract); the initial formal CLI accepts only its locator, excludes `quiet`, and permits no semantic override or legacy/general bypass. The typed state is limited to semantic authority, approved fixed values, derived final/test/save schedules, unsupported-branch absence, and output identity; the [Step 5 connection contract](#adopted-step-5-minimal-connection-contract) binds seed input, initial loss branch and time/V-B handoff under closed key sets without moving runtime, digest, checkpoint, manifest, camera-math, RNG, reporting or publication owners into the resolver. Existing output rejects before heavy import; heavy load/JIT and side-effect-free preparation precede an exclusive claim immediately before the first writer. Issue #10 Candidate C/D remain rejected. Helper/file names, local API/error details, control structure, and whether the first-candidate four-file layout is suitable remain CODEX implementation discretion. |
| required before implementation | Remaining owner details beyond the adopted closed-key/input/handoff subsets [above](#remaining-field-and-owner-boundary) and accepted Step 5 connection; checkpoint schema. Camera choices in [Step 6 A](#step-6-canonical-camera-handoff), loop connection and population statistics-lifetime/point-correspondence correction in [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) are implemented and CPU-accepted, not pending local-design approval. The completed-update event order, formal-entry authority, JSON/mode choice, adopted P1–P6, PF/time/V-B, D-DATA, minimal initialization/LR/batch/SH, configurability/population conditions and eight-field input/handoff contract, Step 5 seed/initial-loss/time-cast/key-set adoption, formal-only locator CLI, and output-claim ordering are not open. Run values are distinct from these specification decisions. |
| required before downstream execution acceptance | Remaining D-* owner decisions and implementation/validation beyond the accepted Step 5 CPU connection, as classified [above](#remaining-field-and-owner-boundary), including PF numerical/CUDA and later-consumer binding. Owner separation permits neither implicit defaults nor a claim of training readiness. |
| required before formal retraining | Numeric final iteration and pilot/formal schedules; explicit pilot/formal population stopping-threshold/unlimited settings within the adopted input contract; densification/prune/reset numeric schedules; complete effective-config snapshot; explicit run seed and runtime determinism/device evidence; test-report metric/cadence; nonfinite/OOM/partial-failure policy; immutable output directory and atomic publication; and, only if resume will be enabled, field-level restore state plus numerical/bitwise equivalence acceptance thresholds. |
| required before formal artifact generation | SPL4-v2 log/linear scale representation; PNG clamp/round/color/codec; full/range CUDA Reference purposes; manifest schema and validator; source-to-binary build provenance; bundle/index/external-digest ownership; direct evidence as formal same-invocation evidence or diagnostic-only. |
| required before Viewer restart | Corrected population and fixed range; Viewer provenance binding; strict parser acceptance; removal or versioned isolation of historical hard-coded ranges; Viewer capture/comparison bundle identity. |

## Pre-retraining validation requirements

After each separately authorized correction, validation proceeds from pure
contracts to bounded integration. No validation in this table was executed by
this documentation sync. The accepted component, integrated resolver and
existing-consumer CPU evidence is consolidated in
[Step 5 functional acceptance](#step-5-functional-acceptance). It establishes
the bounded configuration/bootstrap/claim connection, not every obligation
in the layers below or any actual GPU/training Gate.

The accepted [Step 6 A scope](#step-6-canonical-camera-handoff) combines the
camera input, projection and actual Python binding checks below as one
implemented, CPU-validated function, using only the existing `4dgs310`
interpreter and independent numeric expectations. Existing results, not tests
rerun by this sync, support that acceptance. Boundary-observer outputs are not
CUDA value/gradient evidence and do not replace later actual-device/CUDA
requirements.

The accepted [Step 7 A scope](#step-7-learning-update-consistency-and-cpu-validation)
adds existing-loop, real Adam/population and save/test CPU connection evidence
for layers 4–6. It does not close every root named in the table: actual GPU
execution/gradients, real-data training and the separately owned checkpoint
schema/read-back/publication/resume obligations remain open. These are saved
implementation results, not tests rerun by this documentation sync.

| # | Validation layer | Primary findings closed |
|---:|---|---|
| 1 | Pure CPU canonical-camera tests for SPH intrinsics, both supported modes, raw-sentinel isolation, dimensions/resolution scaling, clipping/cull separation, and the mixed/partial/nonfinite/invalid/off-center rejection matrix | P0-0 and camera/projection P1; P0-A6 consumes the accepted result later |
| 2 | Pure resolver and lightweight-bootstrap tests for the [adopted strict JSON/single-mode and P1–P6 contract](#approved-json-and-single-run-mode-contract), with no `torch`, renderer, `Scene`, or CUDA dependency: accept one explicit authority and the approved dataset/model/camera/renderer state; allow only the locator CLI and reject `quiet`, semantic options, legacy/general bypass, recursive missing/unknown/duplicate fields, type/nonfinite/bool-as-int errors, expressions/coercion/aliases, implicit defaults, authority overwrite, `eval=False`, resume/warm-start/best/environment checkpoint state, unsupported JSON keys even with inactive values, and legacy/pre-existing output at preflight. Prove immutable typed-state scope, explicit fixed values including `scaling_modifier=1.0`/`override_color=None`, verified `N` with mandatory final `N` exactly once, selection-free derived test/save conditions, one output identity, and the separate failure-stage expectations below. | independent formal-entry policy, P0-T4, P0-A2, P0-T6/T7, renderer/config/run-mode P1 |
| 2a | Execution-environment identity test using the `4dgs310` first candidate: invoke its explicit Python path, verify `sys.executable`, record package/PyTorch-CUDA/compiler/driver/GPU identities without changing them, build the extension freshly from tracked-clean corrected source, and bind the loaded binary to that source/build before focused CUDA acceptance. Existing cache and upstream `environment.yml` are rejected as execution authority. | formal environment boundary and P0-A7 build provenance |
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

The focused transaction suite retains the pre-fix negative observation for
requested `N=1,2,3` (optimizer-step calls `0,0,1`) and the corrected `1,2,3`
result recorded in Step 7 A. It must trace ordinary, final, densify, prune-only,
reset-only and simultaneous topology/reset cases, and inspect Parameter
identity, gradient, value and Adam state before and after
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
matching defaults. Step 5 accepts the initial formal-entry and training/render
Python connection; the full later-checkpoint/reference cross-entrypoint suite
and CUDA acceptance remain outstanding.

Formal-entry validation must additionally prove import order without importing
the heavy path in the test fixture: resolution and validation finish first,
then and only then may the same process import renderer/`Scene`/CUDA-related
modules. It must prove the verified state is the sole semantic authority, that
the initial CLI accepts only the formal config locator and cannot change its
semantics, that `quiet` and every legacy/general entry are excluded, and that
formal evaluation and CUDA Reference consumers later receive the same identity
without defaults or `cfg_args` reconstruction. It must separately prove the
read-only existing-output preflight, side-effect-free runtime preparation, and
exclusive pre-writer claim order, including no formal output after heavy/JIT
failure. Helper names, local API/error details, and file boundaries are not
acceptance criteria when an equally bounded implementation preserves these
observable contracts.

For adopted P1–P6, the full focused suite must check each fixed literal/value
and responsibility-object boundary, escaped decoded-key duplicates, invalid
UTF-8/BOM/surrogates, numeric token types, binary64 overflow/nonzero-to-zero
underflow, and the byte/depth/array/string limits just below, at, and above
their boundaries. Derivation tests distinguish P3 maximum SH from the
separately adopted D-SH schedule; they do not establish intermediate renderer
correctness. Camera-owner tests cover divisible/nondivisible dimensions,
actual-image agreement, and one shared canonical camera for both modes.
Schedule fixtures include `N=1` with empty intermediates, strictly increasing
lists, duplicate/reverse-order rejection, intermediate `N` rejection, and final
save exactly once without automatically adding a final test. They need not
allocate a list of length `N`. Path fixtures cover POSIX versus Windows input,
canonical aliases/containment, existing/legacy output, and dangling symlinks.
The accepted component and integrated strict-v1 CPU fixtures are recorded in
Step 5; the camera Python/CPU evidence is accepted in Step 6 A. Remaining
actual-camera-device/CUDA, renderer-math and downstream-consumer obligations
are future work; no tests were run by this sync. Those fixtures do not decide run
values or downstream owner contracts, and contain no placeholder authority.

The JSON input's unsupported-field absence must not be confused with a runtime
adapter's inactive value or the approved explicit null. Validate population
retention separately from `eval=true`; no implicit time-filtered subset is
accepted. Keep the current pre-merge final-save construction and YAML-over-CLI
authority defects as negative regressions without inheriting legacy defaults. Failure
expectations are stage-specific:

| Failure stage | Required observation |
|---|---|
| Invalid JSON/CLI or read-only preflight rejection | Heavy import/JIT, output claim, and writers are not reached; the rejected invocation creates no formal output. |
| Heavy load/JIT or runtime-preparation failure after valid resolution/preflight | Heavy import may have been attempted; formal output claim and first writer are not reached. This does not guarantee absence of every JIT-managed cache. |
| Exclusive-claim race after successful preflight/preparation | Only the successful owner may reach a writer. The losing invocation does not write; the winner's output directory is not required to be absent. |

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

For camera/evaluation specifically, validation must derive
the positive SPH canonical focal/tan values on CPU; prove that a raw negative
sentinel never enters effective state; cover both supported camera modes;
reject mixed, partial, inconsistent, nonfinite, invalid-dimension, invalid-FoV,
invalid-clipping, and off-center inputs before GPU execution; preserve camera
identity after resolution scaling; compare projection with rasterizer forward
and backward focal/tan; verify `eval=True` as 5,146/166 and reject
`eval=False` as the 5,312/0 branch; and distinguish projection `0.01`/`100.0`
from raster near-cull `0.2` and absent far-cull. Because off-center is
unsupported, it requires a pre-CUDA rejection test, not a CUDA correctness
claim. The accepted [Step 6 A scope](#step-6-canonical-camera-handoff) supplies
the bounded Python/CPU camera evidence; actual device/CUDA evidence remains
open. None of these tests is created or rerun by this documentation sync.

### Temporal-prefilter validation requirements

The following adopted requirements span the accepted Step 5 Python handoff
and still-unaccepted numerical/CUDA/later-consumer checks. No tests were
performed by this synchronization:

- From a valid fixture after its full contracts are resolved, independently
  invert each PF-B missing/type/value/alias/decoded-duplicate condition and
  prove no heavy import/JIT, claim, or writer is reached. Do not fabricate a
  complete valid v1 using unknown fields or placeholder objects.
- Pure resolver/adapter mocks must prove one explicit disabled-to-`-1.0`
  derivation and agreement across model, rasterizer, forward, and backward
  context. Missing/wrong-type values, zero, negative values other than `-1.0`, or positive
  values, NaN/Infinity, and path/consumer mismatches must reject rather than
  normalize to disabled; runtime mismatches reject before GPU execution.
- For valid finite `Σ_tt > 0` and finite timestamps, an independent oracle
  computes the original-variance marginal, effective opacity, conditional
  mean, and conditional covariance. Agreement of current disabled formulas
  for zero and `-1.0` does not enlarge the allowed adapter domain.
- A separately authorized fresh corrected-source build with verified
  source/binary binding must compare values and opacity/time/scale/rotation
  gradients on the selected CUDA forward/backward path against the oracle
  and finite differences in smooth regions that do not cross the mask.
  Known P0 dependencies must first be corrected and validated where reached.
  Merely observing the argument `-1.0` cannot pass this requirement.
- Separately inspect float32 inputs near the existing `m > 0.05` mask boundary,
  forward/backward decisions, and saved state. An analytically chosen time is
  not proof of bitwise equality without actual floating-point evidence. Do
  not pass by crossing the mask with a central difference or relaxing tolerance.
- Training/test/reference/checkpoint consumers must share the policy/config
  binding and reject missing policy, wrong config, or legacy-checkpoint-only
  evidence. Manifest args are not proof of executed identity.

Keep `prefiltered=False`, screen-space low-pass `0.3`, camera projection/cull,
the original temporal cull, five renderer values, SH, alpha-cap, and training
transaction semantics unchanged. Concrete fixtures, local APIs, and numeric
tolerances belong to a later bounded implementation/Validation responsibility;
none is invented or executed here. These conditions refine existing renderer
and consumer validation obligations, not a new roadmap Step.

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

The existing Gate A/B renderer/config obligations include the adopted
[PF-A/B/C handoff and numerical checks](#temporal-prefilter-validation-requirements);
Gate C's consumer/publication obligations include their accepted policy/config
binding. This reference does not change the gate order or pass any gate.

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
- the `4dgs310` first candidate is invoked through its explicit Python path and
  verified `sys.executable`; its actual package/CUDA/compiler/driver/GPU state
  is recorded without implicit updates; and a fresh corrected-source extension
  build plus focused CUDA validation is accepted instead of trusting
  `environment.yml` or an existing extension cache;
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
Gate A. The accepted Step 5 resolver, pre-heavy-import boundary and minimal
existing-runtime connection, Step 6 A camera handoff and
[Step 7 A transaction connection](#step-7-learning-update-consistency-and-cpu-validation)
satisfy only their bounded CPU/function scopes. Actual camera/transaction
device/CUDA validation, remaining renderer math, checkpoint foundation, fresh
build and real-data training validation above remain outstanding; these are
later owner duties, not additional Step 5/6/7 CPU acceptance requirements.

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
reproducible accepted execution-environment, build, and runtime identity,
including the verified interpreter and loaded extension provenance; corrected
and validated P0-1/P0-2/P0-3
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
location and error schema are local implementation details, not new policy
owners. The accepted Step 5 entry now enforces validation before heavy import. Configuration enforcement must be a bounded
responsibility separate from the P0-1/P0-2/P0-3 renderer-math Fixes. Checkpoint
and manifest code verify and publish the accepted identity later; neither owns
a second copy of the policy.

Issue #18's adopted JSON/mode contract and Issue #23's adopted P1–P6 refine this
same formal-entry owner;
its [remaining dependencies](#approved-json-and-single-run-mode-contract) do
not transfer dataset, renderer, optimizer/population, or seed/device ownership.
Issue #25's [D-PREFILTER supplement](#approved-temporal-prefilter-contract)
retains that separation: renderer owns meaning, the resolver validates one
state, and checkpoint/manifest consumers bind rather than redefine it.

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
| independent formal entry / pure effective-state resolver | Sole owner of one explicit formal authority and bounded verified typed state, implemented through a lightweight formal-only bootstrap, stdlib-only resolver, and separately loaded heavy runtime. Initial CLI is config-locator-only and excludes `quiet`; legacy/general entry cannot bypass it. Resolve and reject before heavy import/JIT, run the read-only existing-output preflight, load and prepare the runtime without writers, then claim output exclusively immediately before the first writer. Bind approved semantic values and derived conditions, including [adopted seed input and time/loss connections](#adopted-step-5-minimal-connection-contract), without transferring runtime/RNG/digest/checkpoint/manifest/camera-math/reporting/publication ownership. | training and later evaluation/CUDA Reference consume the state; P0-A2 and manifest publish/verify rather than redefine it; local names/API/error structure remain CODEX discretion |
| diagnostic checkpoint foundation | P0-T6: version, field-level schema, semantic validation, diagnostics, and provenance; consume the P0-T1/T2 completed label without owning loop or optimizer order | P0-A1/A2/A8 consumers |
| exact resume (conditional independent root) | Complete continuation state and equivalence, including any resume-reachable P0-T5/T6 handling, only after known P0/transaction stability; do not merge it into the diagnostic foundation or transaction Fix | formal continuation only after gate acceptance; otherwise fail closed |
| environment-map lifecycle (future conditional root) | P0-T7 remains unfixed but unreachable under `env_map_res=0`; any future enablement requires separate policy and complete lifecycle/checkpoint/optimizer/resume validation | first-baseline CUDA reconstruction must reject environment state; future runtime publication follows only after separate acceptance |
| SPL4 exporter/provenance | v2 representation, atomic export, exact population binding | Viewer bundle input |
| SPL4 parser | P0-A3 | production evaluation inputs |
| CUDA Reference loader | P0-A1/A2, selected tensor mapping, fresh render | manifest and bundle output |
| manifest builder/validator | P0-A5/A6/A8 publication claims, including exact accepted P0-0 camera mode/effective state and projection/raster cull distinction without sentinel substitution | Viewer/comparison acceptance |
| execution environment and build provenance | `4dgs310` is the first candidate for pilot, formal retraining, corrected CUDA Reference, and CUDA focused validation; verify its explicit interpreter and record actual package/CUDA/compiler/driver/GPU state without implicit updates. P0-A7 owns the tracked-clean corrected-source-to-fresh-binary relation; neither upstream `environment.yml` nor an existing cache is execution authority. | Gate A/B runtime acceptance and formal CUDA Reference identity |
| Viewer handoff/bundle contract | P0-A4, corrected range and wrong-pair rejection | capture/comparison artifacts |
| output publication | unique owner, atomic write, completion/index/digest | every formal output stage |

The training-state and optimizer/population owners retain the bounded
[Step 7 A implementation/CPU result](#step-7-learning-update-consistency-and-cpu-validation).
That result neither transfers checkpoint/publication ownership nor completes
their remaining actual device/CUDA and training validation.

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

This register retains the original responsibility numbers for historical
cross-references. Steps 5 and 6 are complete; the current functional Step 7 is
[learning-update consistency and CPU validation](#step-7-learning-update-consistency-and-cpu-validation),
covering former responsibilities 9/10, not former item 6's checkpoint task.
Other unstarted work remains identified
by its finding/owner and Gate dependency. The old numbered lists do not fix
future execution-Step numbering, order or scope; technical prerequisites and
Phase goals remain binding.

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
   document.** Integrate the Issue #12 approved lightweight-bootstrap/pure-
   resolver/heavy-runtime boundary, formal-only CLI and output-claim ordering,
   plus the `4dgs310` first-candidate environment contract under Issue #17.
   **Complete in this document.** Integrate Issue #18's user-adopted strict
   nested semantic JSON / immutable state and single from-scratch mode under
   Issue #22, retaining the classified undecided field/owner/run boundaries.
   **Complete in this document.** Integrate Issue #23's adopted P1–P6 partial
   field contracts under Issue #24, without adopting the whole 62-field
   register or resolving the remaining D-* and run values.
   **Complete in this document.** Integrate Issue #25's adopted PF-A/PF-B/PF-C
   and supplemental safety conditions under Issue #26, closing only the
   D-PREFILTER adoption question. **Complete in this document.**
   Integrate only the [three adopted D-TIME contracts](#adopted-d-time-input-retention-and-handoff-contracts)
   under Issue #30. **Complete in this document; remaining time/owner/run
   boundaries stay open.** Integrate the
   [accepted current-PLY connection](#adopted-current-initial-ply-reuse-and-raw-time-connection)
   under Issue #33. **Complete in this document; input Validation is distinct
   from post-fix focused validation.** Integrate the
   [adopted initial-variance contract](#adopted-initial-temporal-variance-and-d-time-connection)
   under Issue #34. **Complete in this document; implementation and numerical
   validation remain outstanding.** Integrate Issue #2's
   [minimal initialization/LR/batch/SH contracts](#adopted-minimal-initialization-optimizer-and-sh-contracts).
   **Policy synchronization complete; remaining choices and runtime enforcement
   are not completed.** Integrate the subsequent
   [configurability/population conditions](#adopted-configurability-and-population-conditions),
   including the withdrawn no-cap ban and the subsequent adopted eight-field
   population JSON input/handoff contract.
   **Policy synchronization complete, not population enforcement or validation.**
   Integrate the [Step 5 minimal connection contract](#adopted-step-5-minimal-connection-contract).
   **Policy synchronization complete for seed, initial loss branches, time/V-B
   handoff and closed key sets, not implementation or runtime validation.**
   Step 5 now accepts the integrated configuration and minimal same-process
   existing-runtime connection, including the time/V-B and frame components,
   as recorded below. The preceding policy milestones describe what their
   synchronization alone established, not today's implementation status.
   Camera/transaction runtime/CUDA validation, remaining renderer/checkpoint
   and other later owner corrections, fresh corrected-source build and
   toolchain/CUDA acceptance remain open.
   Remaining owner contracts beyond the adopted subsets, run values and
   checkpoint schema stay undecided; closed JSON membership and its Step 5
   enforcement are no longer pending. Step 5 documentation review and user
   research Git are complete, as are the Step 6 camera A completion operations.
   [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation) now has
   advisor-reviewed, user-accepted implementation and CPU validation.
   This sync records that bounded result; Step 7 document review and consolidated
   user research Git remain pending, not a new policy selection or source-work start.
   Confirm one root owner and one
   bounded Fix responsibility at a time only after the applicable policy is
   decided. The formal-entry authority, component separation, initial formal
   CLI boundary, JSON/mode choice and adopted P1–P6, typed-state ownership
   boundary, and output-claim ordering are decided. Helper/file names, local API/error
   structure, and internal control
   flow remain implementation discretion rather than separate policy choices.

### Phase 1: training-critical foundation

5. Implement the approved lightweight formal-only bootstrap, stdlib-only pure
   resolver, and separately loaded heavy runtime. Use the config-locator-only
   initial CLI, keep `quiet` and the legacy/general path outside formal entry,
   reject existing output in read-only preflight, and claim it exclusively only
   after side-effect-free runtime preparation and immediately before the first
   writer. Apply the adopted JSON/single-mode and P1–P6 partial contracts and
   the [D-PREFILTER supplement](#approved-temporal-prefilter-contract) and
   [three D-TIME contracts](#adopted-d-time-input-retention-and-handoff-contracts),
   including the [adopted current-PLY connection](#adopted-current-initial-ply-reuse-and-raw-time-connection)
   and [initial-variance contract](#adopted-initial-temporal-variance-and-d-time-connection),
   plus the [D-DATA minimal connection](#adopted-d-data-minimal-connection-contract)
   and [minimal initialization/LR/batch/SH contracts](#adopted-minimal-initialization-optimizer-and-sh-contracts)
   with the [configurability/population conditions and eight-field input/handoff contract](#adopted-configurability-and-population-conditions)
   and [Step 5 minimal connection contract](#adopted-step-5-minimal-connection-contract).
   These input/handoff choices are implemented within the accepted minimal
   connection; downstream execution still needs its owners' validation.
   Bind non-overwriting output identity
   without creating a second semantic authority; choose bounded local
   helper/API/file/error details during the source Fix.

   Reuse existing checks, loader, GaussianModel, renderer, loss and logger,
   with only necessary local separation and argument connections, not a new
   training/renderer or full reporting design. Step 5's configuration/handoff
   responsibility does not absorb camera, renderer, loop, completed-update
   transaction, or checkpoint Fixes. Accepted Step 6/7 CPU results are recorded
   below; other owner implementations and actual runtime validation remain
   explicit dependencies for training readiness, not
   additional Step 5 Fix conditions. The current accepted scope below uses
   existing CPU consumers, not only success-stub handoffs.

   **Current status:** [the minimal connection is functionally accepted](#step-5-functional-acceptance).
   Documentation review and consolidated user-owned research Git are complete;
   no GPU/training or Gate acceptance is implied.

   **Historical component records:** the following results describe each
   component's own acceptance scope, not limits on the integrated resolver now
   implemented. Their Python 3.12/3.10 runs are past evidence, not a renewed
   dual-environment requirement. Step 5 connection/Fix/Validation used only
   the existing `4dgs310` interpreter.

   **Partial implementation accepted under Issue #35:** the independent P2
   JSON reader/common-type component
   ([source](../formal_config_json.py), [CPU tests](../tests/test_formal_config_json.py))
   is implemented. Python 3.12 and `4dgs310` Python 3.10 each passed 25 focused
   CPU tests; the advisor independently reproduced both results, and the user
   accepted this bounded component. See the
   [implementation report](../../reports/corrected-4dgs/issue-35/issue-35-implementation-report.txt),
   [independent review](../../reports/corrected-4dgs/issue-35/issue-35-advisor-review.txt),
   and [acceptance/sync authorization](../../reports/corrected-4dgs/issue-35/issue-35-documentation-authorization-evidence.json).
   Its research Git integration/push and Issue #35 closure are confirmed by the
   [push verification](../../reports/corrected-4dgs/issue-35/issue-35-push-verification.json)
   and [completion evidence](../../reports/corrected-4dgs/issue-35/issue-35-completion-evidence.json).

   **Partial implementation accepted under Issue #36:** the P1 component
   ([source](../formal_config_p1.py), [CPU tests](../tests/test_formal_config_p1.py))
   consumes P2 parse data and checks the closed set of 10 top-level keys,
   eight objects, two string literals, and required presence/type/value of the
   14 entries in the adopted P1 fixed-input table. Python 3.12 and `4dgs310`
   Python 3.10 each passed P1's 25 and P2's 25 CPU tests; the advisor
   independently reproduced all four executions, and the user accepted the
   bounded P1 result. See the
   [implementation report](../../reports/corrected-4dgs/issue-36/issue-36-implementation-report.txt),
   [independent review](../../reports/corrected-4dgs/issue-36/issue-36-advisor-review.txt),
   and [acceptance/sync authorization](../../reports/corrected-4dgs/issue-36/issue-36-documentation-authorization-evidence.json).
   Its document review, user-owned research-branch commit/push, and Issue #36
   closure are confirmed by the
   [push verification](../../reports/corrected-4dgs/issue-36/issue-36-push-verification.json)
   and [completion evidence](../../reports/corrected-4dgs/issue-36/issue-36-completion-evidence.json);
   this is not a claim of integration into main.

   **Partial implementation accepted under Issue #37:** the P3 component
   ([source](../formal_config_p3.py), [CPU tests](../tests/test_formal_config_p3.py))
   rechecks current input through P1 on every call and reuses P2 Int checks.
   It derives and verifies 48 slots and the three approved ranges from maximum
   spatial/temporal degrees 3/2, and derives boolean true for a later adapter.
   Python 3.12 and `4dgs310` Python 3.10 each passed P3's 20, P1's 25, and P2's
   25 CPU tests (six executions); the advisor independently reproduced all six,
   and the user accepted this bounded component. See the
   [implementation report](../../reports/corrected-4dgs/issue-37/issue-37-implementation-report.txt)
   and [validation evidence](../../reports/corrected-4dgs/issue-37/issue-37-validation-evidence.json),
   [independent review](../../reports/corrected-4dgs/issue-37/issue-37-advisor-review.txt)
   and [review evidence](../../reports/corrected-4dgs/issue-37/issue-37-advisor-review-evidence.json),
   and [acceptance/sync authorization](../../reports/corrected-4dgs/issue-37/issue-37-documentation-authorization-evidence.json).
   Its document review, user-owned research-branch commit/push, and Issue #37
   closure are confirmed by the
   [push verification](../../reports/corrected-4dgs/issue-37/issue-37-push-verification.json)
   and [completion evidence](../../reports/corrected-4dgs/issue-37/issue-37-completion-evidence.json);
   this is not a claim of integration into main.

   P3's competing-input rejection covers only the explicit immediate paths
   (model: nine; renderer: four) listed in the implementation report, not every
   unknown spelling or the remaining nested required/allowed sets. Unchecked
   fields are not allowed/adopted extensions or formal-validated values.
   Maximum degrees are not active degrees: the subsequent
   [D-SH policy](#adopted-minimal-initialization-optimizer-and-sh-contracts)
   adopts staged warm-up, but this accepted P3 component does not implement
   that runtime schedule or validate intermediate-degree correctness.

   **Partial implementation accepted under Issue #38:** the P5 component
   ([source](../formal_config_p5.py), [CPU tests](../tests/test_formal_config_p5.py))
   rechecks current input through P1 on every call and reuses P2 Int checks for
   the required N and two required arrays. It checks type, range, and strict
   increase under adopted P5, distinguishes missing arrays from explicit empty
   arrays, and derives the save schedule by appending N exactly once without
   adding it to the test schedule. Input values, number tokens, nested identity,
   and order remain unchanged on success and failure. Its immutable partial
   result is detached from input containers. The input-array limit is not
   misapplied to the derived save tuple: the final N is not truncated.
   Python 3.12.3 and `4dgs310` Python 3.10.14 each passed P5's 27, P1's 25,
   P2's 25, and P3's 20 CPU tests (eight executions, 194 test executions in
   total); the advisor independently reproduced all eight, and the user
   accepted this bounded component. See the
   [implementation report](../../reports/corrected-4dgs/issue-38/issue-38-implementation-report.txt)
   and [validation evidence](../../reports/corrected-4dgs/issue-38/issue-38-validation-evidence.json),
   [independent review](../../reports/corrected-4dgs/issue-38/issue-38-advisor-review.txt)
   and [review evidence](../../reports/corrected-4dgs/issue-38/issue-38-advisor-review-evidence.json),
   and [acceptance/sync authorization](../../reports/corrected-4dgs/issue-38/issue-38-documentation-authorization-evidence.json).
   Its document review, user-owned research-branch commit/push, and Issue #38
   closure are confirmed by the
   [push verification](../../reports/corrected-4dgs/issue-38/issue-38-push-verification.json)
   and [completion evidence](../../reports/corrected-4dgs/issue-38/issue-38-completion-evidence.json);
   this is not a claim of integration into main.

   P5's competing-input checks are limited to the six immediate
   `final_iteration`/`save_iterations` paths under optimization/checkpoint/reporting,
   plus P1's closed top-level checks; the implementation report owns the detailed
   path/fixture mapping. Other objects, deeper nested paths, unknown spellings,
   and the full nested required/allowed sets remain unchecked, not approved
   extensions. Empty-array acceptance does not approve a run cadence; final-test
   needs, run values, and state-zero diagnostics remain with the existing owners.

   **Partial implementation accepted under Issue #39:** the P4 component
   ([source](../formal_config_p4.py), [CPU tests](../tests/test_formal_config_p4.py))
   rechecks current input through P1 on every call and reuses P2 Int checks.
   It requires `dataset.resolution`, closes that object to `mode`/`divisor`,
   requires the fixed string `integer_divisor`, and checks divisor
   `1..2147483647`. In `derive_p4_resolution`, raw width/height are explicit
   positive exact-integer local API inputs, not new JSON fields. Integer
   `divmod` derives positive effective dimensions only when both are divisible;
   `d=1` preserves the dimensions. There is no coercion, rounding/truncation,
   float-scale or upscaling repair, nor a newly adopted raw-dimension upper
   bound. Input values, number tokens, nested identity, and order are unchanged
   on success and failure; the immutable P4 partial result shares no input
   containers. This proves only the supplied values' type and arithmetic,
   not actual-image agreement, owner verification, or formal handoff.

   Python 3.12.3 and `4dgs310` Python 3.10.14 each passed P4's 26, P1's 25,
   P2's 25, P3's 20, and P5's 27 CPU tests (ten executions, 246 test executions
   in total). The advisor independently reproduced all ten with no blocking
   findings, and the user accepted this bounded component. See the
   [implementation report](../../reports/corrected-4dgs/issue-39/issue-39-implementation-report.txt)
   and [validation evidence](../../reports/corrected-4dgs/issue-39/issue-39-validation-evidence.json),
   [independent review](../../reports/corrected-4dgs/issue-39/issue-39-advisor-review.txt)
   and [review evidence](../../reports/corrected-4dgs/issue-39/issue-39-advisor-review-evidence.json),
   and [acceptance/sync authorization](../../reports/corrected-4dgs/issue-39/issue-39-documentation-authorization-evidence.json).
   Its document review, user-owned research-branch commit/push, and Issue #39
   closure are confirmed by the
   [push verification](../../reports/corrected-4dgs/issue-39/issue-39-push-verification.json)
   and [completion evidence](../../reports/corrected-4dgs/issue-39/issue-39-completion-evidence.json);
   this is not a claim of integration into main. These component-level Git
   checkpoints are historical, not future mandatory triggers.

   P4 rejects `resolution_scales` at the root through P1, immediately under
   dataset through P4, and inside the resolution object through its two-key
   closure, even with inactive values. The implementation report owns the
   detailed path/fixture mapping; this adds no schema or forbidden-name table.
   At this component milestone, other objects/deeper paths and full nested
   membership, actual-image/metadata agreement and the formal runtime handoff
   were not implemented. The integrated Step 5 resolver/connection now supplies
   its bounded checks and handoffs. Actual run divisor, remaining run-dependent
   alignment and canonical-camera/FoV-only validation stay with D-DATA/P0-0.

   **Partial implementation accepted under Issue #41:** the P6 read-only path
   component ([source](../formal_config_p6.py), [CPU tests](../tests/test_formal_config_p6.py))
   rechecks current input through P1 on every call and checks the required
   string paths `dataset.source_path` and `output.directory` from P2 parse data.
   It accepts explicit absolute POSIX paths without expansion, resolves existing
   symlinks and `..` in filesystem order, verifies directory traversal, and
   rejects OS errors. Dataset must exist as a directory; existing output names,
   including dangling symlinks, are rejected. Component-wise identity and
   mutual-containment checks reject output overlapping dataset or the fixed
   historical output protected by this plan, without confusing similar prefixes.
   The checks are metadata-only, with no writer. Input is unchanged, and the
   immutable P6 partial result is independent of input containers. No new
   JSON/CLI/registry or optional protection parameter was introduced. This is
   not dataset-content proof, full verified state, execution permission, or an
   output reservation.

   Competing `run_id`/`model_path`/`output_root` checks cover only the closed
   root through P1 and the immediate dataset/output objects. Other objects,
   deeper nested paths, unknown spellings, and the full nested schema remain
   unchecked, not allowed/adopted extensions. The implementation report owns
   the detailed path/fixture mapping; no new forbidden-name table is added here.

   The worker's two-environment record includes 14 executions and 372 test
   executions including retests. The advisor independently tested the final
   version in Python 3.12 and `4dgs310` Python 3.10: each passed P6=32, P1=25,
   P2=25, P3=20, P4=26, and P5=27 (12 executions, 310 test executions total).
   No blocking findings remained, and the user accepted the bounded component.
   See the [implementation report](../../reports/corrected-4dgs/issue-41/issue-41-implementation-report.txt)
   and [validation evidence](../../reports/corrected-4dgs/issue-41/issue-41-validation-evidence.json),
   [independent review](../../reports/corrected-4dgs/issue-41/issue-41-advisor-review.txt)
   and [review evidence](../../reports/corrected-4dgs/issue-41/issue-41-advisor-review-evidence.json),
   and [user acceptance record](../../reports/corrected-4dgs/issue-41/issue-41-documentation-authorization-evidence.json).
   That last record is acceptance history, not the current clean/hash precondition;
   the [v2 synchronization scope](../../reports/corrected-4dgs/issue-41/issue-41-documentation-description-v2.md)
   and [v2 preflight](../../reports/corrected-4dgs/issue-41/issue-41-documentation-v2-preflight.json)
   identify this correction's protected starting state. This sync records existing
   test evidence without rerunning tests. It corrects the misleading P6-unimplemented
   status under ADV-DOC-02; it does not introduce a Step or progress increment.
   Overall progress management is Issue #43, distinct from parent policy Issue #2.

   At the P6 component milestone, there was no full immutable verified state,
   CLI/bootstrap/runtime connection or exclusive claim, and `train.py` was
   unchanged. Those historical limits are superseded by the integrated
   Step 5 acceptance below, not by P6 alone. The component's document/Git
   review notes are historical checkpoints, not current per-component work.
   P2 parse data and detached P1/P3/P4/P5/P6 partial results still are not by
   themselves the full semantic authority. At that milestone, atomic publication
   and checkpoint-then-test observation were separately open. Step 7 now accepts
   the latter's CPU connection; formal publication and read-back remain open.

   **Step 5 functional acceptance and completion** are recorded below.
   Step 6 is also complete; the current function is Step 7 below. The legacy
   register preserves responsibility references, not an unchanged Step sequence.

### Step 6: canonical camera handoff

**Current status:** Step 6 is complete within A's camera connection and CPU
validation scope. Functional acceptance, documentation review, consolidated
user-owned research Git and push confirmation are recorded in the
[Step 6 completion record](../../reports/corrected-4dgs/phase1/step6/step6-completion-report.txt).
The accepted function repairs the negative-FoV-sentinel split
between intrinsics projection and rasterizer camera values, from raw input
through the existing consumers.
This continues the minimal Fudan Native corrections toward a normal
from-scratch Karman-vortex comparison baseline, not a new training system.

The accepted implementation and CPU evidence cover the following together:

- validate both approved camera modes before root/frame fallback can hide
  mixed, partial or inconsistent metadata; preserve raw/effective separation;
- reject preflight-observable camera errors before heavy import/GPU/writers,
  and runtime handoff mismatches before the relevant GPU call;
- apply resolution correction once and derive one effective camera state
  for existing Camera/projection, shared render, Python forward arguments,
  saved context and backward arguments, without sentinel abs/clamp repair,
  competing defaults or a second camera authority;
- exercise the actual consumers on CPU with GPU/JIT boundaries isolated,
  comparing both modes, original/downscaled resolution, the old sentinel
  defect and rejection cases against independent numeric expectations; and
- preserve Step 5's same configuration, time/V-B, frame correspondence,
  train/test separation, mask/loss and claim ordering with focused regression.

Helper-only success, tan-sign correction alone, success stubs or documentation
alone do not meet this function's acceptance. A proves Python-side connection
through the values handed to the C++/CUDA boundary; it does not prove actual
device transfer, CUDA arithmetic/gradients or training. The observer's
substitute images and gradients are only Python-path scaffolding, not CUDA
correctness evidence. B's added CUDA execution is not adopted for this Step
scope. Full P0-0 runtime validation, Phase 1 and Gates A–D remain unpassed.
Projection/cull, PF (strict float `-1.0`), time/V-B, mask/loss, claim, other
renderer math, checkpoint, manifest and Viewer freeze remain unchanged.

Existing `4dgs310` results are **275 passing formal-suite tests and 2 passing
Step 5 connection tests**, recorded in the
[implementation report](../../reports/corrected-4dgs/phase1/step6/step6-implement1/step6-implement1-report.md)
and [implementation advisor review](../../reports/corrected-4dgs/phase1/step6/step6-implement1/step6-implement1-advisor-review.txt).
The review's user-acceptance-pending wording records its original stage;
the [Step 6 sync instruction](../../reports/corrected-4dgs/phase1/step6/step6-docsync2/step6-docsync2-instruction.txt)
records subsequent user acceptance. This document sync reruns no tests.

Design and acceptance detail: [redesign v2](../../reports/corrected-4dgs/phase1/step6/step6-design1/step6-design1-advisor-redesign-v2.md),
[CODEX detailed design](../../reports/corrected-4dgs/phase1/step6/step6-design1/step6-design1-design-report.md),
and [advisor review](../../reports/corrected-4dgs/phase1/step6/step6-design1/step6-design1-advisor-review.md).
The proposal/review-pending labels in earlier records describe their original
stage; the pre-implementation [v2 sync instruction](../../reports/corrected-4dgs/phase1/step6/step6-docsync1/step6-docsync1-instruction-v2.txt)
adopts this reviewed boundary. Local types/APIs, helper layout, file count and
the full fixture/tolerance specifications remain in that design rather than
being duplicated here as new permanent policies.

The document-review/Git-pending notes in the Step 6 records describe their
original stage; the later completion record supersedes those as current work.
This sync carries that completion forward without repeating Git or acceptance
operations. The current function is Step 7 below; full camera runtime/CUDA and
Gate obligations remain unchanged.

### Step 7: learning-update consistency and CPU validation

**Current status:** reviewed A (learning-update consistency plus existing-consumer
CPU validation) is implemented, advisor-reviewed and user-functionally-accepted.
Document review, consolidated user research Git, push confirmation and Step 7
completion processing remain pending. This function covers former Phase 1
responsibilities 9/10 without renumbering their owners or planning future Steps.

The bounded implementation/CPU acceptance establishes:

- the real loop performs exactly `N` updates, including final `N`, without an
  `N+1` batch fetch; the pre-fix `N=1,2,3` observation of `0,0,1` Adam steps is
  corrected to `1,2,3`, with batch/label identity and independent Adam values checked;
- after forward/backward, current statistics are applied before replacement,
  then the same backward-owned Parameter is stepped: statistics → step →
  zero-grad → growth/prune → reset → completed → scheduled save → scheduled test;
  schedules still precede forward, and the approved event conditions are unchanged;
- failure before transaction completion cannot advance completed state or reach
  downstream observers; save/report failure leaves the completed update count
  intact and stops downstream work. This is not rollback, retry or atomic publication;
- initial diagnostic `0` remains separate from updates, completed save/test
  schedules and selection. Optimization loss is from the pre-update forward;
  test metrics observe the post-update/topology/reset state;
- old rows retain screen-radius history through final prune, clone children
  inherit parent radius, and split children have radius `0` meaning unobserved.
  Parameter, Adam and statistics rows remain aligned. Child moments start at
  zero while the group's scalar Adam step is inherited, not a new per-row step;
- growth resets the statistics window only after final prune. Prune-only
  retains sliced history and reset-only adds no statistics-window reset.
  These are accepted implementation/CPU choices, not proof of GPU measurement,
  learning quality or optimal population policy.

Existing `4dgs310` evidence records **291 passing formal-suite tests (275 existing
and 16 new) and 2 passing Step 5 connection probes**, with zero failure/error/skip.
These are saved results, not tests rerun by this sync; earlier focused runs are
not added again. Evidence: [implementation report](../../reports/corrected-4dgs/phase1/step7/step7-implement1/step7-implement1-report.md),
[validation evidence](../../reports/corrected-4dgs/phase1/step7/step7-implement1/step7-implement1-validation-evidence.json),
and [advisor implementation review](../../reports/corrected-4dgs/phase1/step7/step7-implement1/step7-implement1-advisor-review.txt).
The review's pending-user-acceptance wording is historical; the
[current synchronization instruction](../../reports/corrected-4dgs/phase1/step7/step7-docsync1/step7-docsync1-instruction.txt)
records subsequent user acceptance. Detailed methods remain in the
[reviewed design](../../reports/corrected-4dgs/phase1/step7/step7-design1/step7-design1-design-report.md)
and its [review](../../reports/corrected-4dgs/phase1/step7/step7-design1/step7-design1-advisor-review.txt),
not duplicated here as permanent helper/API or fixture prescriptions.

The tests use the real training loop, Adam, existing clone/split/prune/reset,
Scene.save/capture observers and real report/metric/PNG consumers on CPU.
Differentiable render inputs and isolated GPU/C-boundary images/gradients are
scaffolding, not CUDA numerical evidence. Scene.save/capture observation is
not formal checkpoint serialization, read-back, atomic publication or resume.
Actual device/CUDA/gradient validation, fresh corrected-source build, real-data
training, other renderer/SH/alpha-cap and checkpoint/manifest owners remain
open. P0-T1/T2/T3 are not unconditionally closed; Phase/Gates and Viewer freeze
are unchanged. Run values, population comparisons and event conditions are not reselected.

**Unresolved constant-depth visualization:** existing `easy_cmap` divides by
`max-min`; constant depth can produce invalid diagnostic PNG/TensorBoard output.
The nonconstant fixture passes the connection tests but does not repair the
production helper. Reporting interruption under warning-as-error remains
unverified. This does not block Step 7 A acceptance; review the handling and
any necessary bounded correction before accepting real-GPU/pilot diagnostic
output. No Fix, new Step or ticket is authorized by this documentation sync.

Next: advisor document review → user research Git consolidating the accepted
source/test and this plan → push confirmation and Step completion processing.
This sync performs none of those Git/completion operations, creates no separate
document-only Git checkpoint and does not start the next function. The advisor
considers that function from the completion result and remaining duties.

### Historical Phase 1 responsibility register (former items 6–13)

The following are preserved legacy references. Former item 7 maps to the
completed Step 6 camera function; former items 9/10 map to the current
[Step 7 A scope](#step-7-learning-update-consistency-and-cpu-validation).
Their accepted implementation/CPU results are not unstarted work, while their
actual runtime validation remains open. Former item 6's P0-T6 checkpoint foundation
remains a required separate responsibility, without transfer to another owner.
Withdrawn checkpoint-oriented Step 6 instructions are not execution authority.

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
    Keep the [population statistics-lifetime/point-correspondence dependency](#adopted-configurability-and-population-conditions)
    with this existing population owner: Step 7 A accepts the bounded correction
    of `max_radii2D` lifetime, child-radius handling and row correspondence,
    not actual GPU/training validation.
11. Implement post-merge effective `eval=True`, the approved train/test
    separation, test non-selection, and fixed-final completed-checkpoint
    selection; reject `eval=False`, validation/best, and resume branches at
    their entry points.
12. Correct every conditional P0 reached by the formal branch and fail-closed
    reject every unsupported branch.
13. Implement nonfinite/failure handling and atomic checkpoint publication.

### Step 5 functional acceptance

**Accepted scope:** configuration validation through the minimal same-process
connection to existing runtime. This milestone integrates the previously
accepted P1–P6 and time/frame parts, reviewed implementation, fix1 and the
supplemental validation1 evidence. Its documentation review and consolidated
user-owned research Git are also complete, as recorded in the
[Step 5 completion record](../../reports/corrected-4dgs/phase1/step5/step5-completion-report.md).
Earlier acceptance/review/Git-pending statements are historical, not reasons
to reopen Step 5 or request another push. This completion remains distinct
from actual GPU/training acceptance and Gates A–D.

- The integrated resolver closes the adopted key sets, reuses P1–P6/time/frame
  checks and produces one immutable verified configuration. The lightweight
  formal entry completes read-only configuration/input preflight before
  loading the existing heavy runtime in the same process.
- Existing consumers receive explicit read-only views of that same state,
  not reconstructed CLI/YAML/default authority. The approved dataset
  reference and retained metadata/PLY bytes connect input checks to the
  existing loader. Seed initialization precedes sampling/shuffle once;
  frame and initial-PLY time use the one adopted transform, and Gaussian
  initialization consumes the derived V-B log-scale, not duration/5.
- Existing Scene/Gaussian/optimizer/loader preparation is output-free, then
  the exclusive claim precedes the first writer. Fix1 restores the legacy
  loaded-iteration diagnostic guard while retaining formal preparation and
  normal legacy initial writes. It does not implement legacy load support.
- Validation1 A passes explicit strict float `-1.0` through the actual
  Python render body to the Rasterizer call boundary; the observer requires
  that keyword without a default and stops before binding/kernel execution.
  B inserts a competing leaf after successful final preflight: actual
  exclusive `mkdir` rejects with `EEXIST`, no loser writer or continuation
  is reached, and the winner's directory/content remain unchanged.

Evidence: [reviewed design](../../reports/corrected-4dgs/phase1/step5/step5-implement1/step5-implement1-design-report.md),
[implementation report](../../reports/corrected-4dgs/phase1/step5/step5-implement1/step5-implement1-implementation-resumed-report.md)
and [implementation review](../../reports/corrected-4dgs/phase1/step5/step5-implement1/step5-implement1-implementation-advisor-review.txt),
[fix1 report](../../reports/corrected-4dgs/phase1/step5/step5-fix1/step5-fix1-report.md)
and [fix1 review](../../reports/corrected-4dgs/phase1/step5/step5-fix1/step5-fix1-advisor-review.txt),
[validation1 report](../../reports/corrected-4dgs/phase1/step5/step5-validation1/step5-validation1-report.md),
[CPU evidence](../../reports/corrected-4dgs/phase1/step5/step5-validation1/step5-validation1-evidence.json)
and [combined acceptance review](../../reports/corrected-4dgs/phase1/step5/step5-validation1/step5-validation1-advisor-review.txt).
The [Step 5 docsync authorization](../../reports/corrected-4dgs/phase1/step5/step5-docsync1/step5-docsync1-instruction.txt)
advanced the review's then-pending user-acceptance boundary; its later
completion is recorded above.
Validation1's two tests supplement, rather than replace, the earlier setting
and existing-consumer CPU evidence. NumPy/Pillow/Torch CPU and existing
consumers were exercised with GPU/JIT boundaries isolated, using only
`/home/demo/miniconda3/envs/4dgs310/bin/python`. No tests were rerun for this sync.

**Remaining limits:** the PF observation is not CUDA forward/backward or saved
backward-context proof. The race fixture fixes ordering in one process, not
parallel stress or general filesystem-attack defense. Packed legacy PLY
all-row retention still has the known stride limitation; the legacy
`load_ply` API is absent. Load-boundary fixtures prove diagnostic/writer
non-reachability, not real loading. Neither repairing nor permanently
disabling those legacy paths is newly adopted; they are not additional
Step 5 acceptance conditions for the formal from-scratch connection.

Actual GPU cast/activation health, canonical camera and renderer mathematics,
CUDA forward/backward, actual-runtime completed-update/optimizer/population transactions,
checkpoint semantics, actual training and Gate A/B remain with their existing
owners. No run values, new schema, policy or downstream execution authorization
are introduced by Step 5 completion. The subsequent
[Step 6 camera handoff](#step-6-canonical-camera-handoff) is also complete.
Current [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation)
accepts update/population/save-test implementation and CPU connection, with
its document review and user research Git still pending.
Viewer freeze and all Gate requirements remain unchanged.

### Phase 2: focused validation

14. Run the config/run/checkpoint negative matrix, including post-merge
    effective-`eval` acceptance and rejection.
15. Run independent canonical-camera CPU contracts and camera/renderer forward
    oracles, including supported-mode, sentinel-isolation, scaling, clipping/
    cull, and pre-GPU rejection cases.
16. Run CPU autograd and finite differences.
17. From the `4dgs310` first candidate's explicit verified interpreter, build
    the selected corrected CUDA source freshly, record the actual environment/
    toolchain/source/binary identity, and run CUDA forward/gradient validation
    without treating the existing cache or upstream `environment.yml` as
    execution authority.
18. Run separately owned training-state-machine and optimizer/densification
    mocks for the shared approved event boundary; do not fold P0-T6 schema or
    exact-resume equivalence into these fixtures.
19. Run train/test identity, test-non-selection, fixed-final checkpoint, and
    output-identity tests.
20. Run an integrated bounded small-scene smoke.

### Phase 3: pilot and formal training

21. Run small corrected pilot training from iteration 0 with resume prohibited
    after Gate A, using `4dgs310` only after its fresh-build and focused-CUDA
    environment acceptance succeeds.
22. Review pilot updates, metrics, topology, checkpoints, and failure evidence.
23. Only if formal resume is wanted after known P0 and transaction stability,
    implement it under an independent root ticket and accept the resume-
    equivalence gate; otherwise keep resume fail closed.
24. Freeze the canonical camera/eval publication identity, formal config,
    dataset, source/build, schedule, and remaining policies.
25. Run the formal corrected baseline from scratch after Gate B in the accepted
    recorded `4dgs310` execution identity, using resume only if step 23 was
    accepted.
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

These technical dependencies keep Viewer-only P0-A3/A4 out of the pilot path,
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
  exclusion of competing/default/legacy/resume/best/environment/output reuse;
  and
- Investigation #12 and repository synchronization under Issue #17 of the
  approved lightweight-bootstrap/stdlib-only-resolver/heavy-runtime boundary,
  formal-only locator CLI, typed-state owner boundary, pre-writer exclusive
  claim order, and `4dgs310` first-candidate environment contract; and
- Investigation #18's accepted result and supplemented review, and repository
  synchronization under Issue #22 of its adopted strict nested semantic JSON /
  immutable verified state, single from-scratch mode, strict parsing and
  failure-stage distinctions, with unresolved field/owner/run decisions kept
  separate; and
- repository synchronization under Issue #24 of Issue #23's user-adopted
  P1–P6 partial field contracts, with remaining D-* and run values explicitly
  unresolved. This is not whole-register, runnable-v1, source-implementation,
  focused-validation, or Gate acceptance; and
- repository synchronization under Issue #26 of Issue #25's adopted
  D-PREFILTER policy and safety conditions. Step 5's Python handoff is accepted;
  numerical/CUDA validation and checkpoint/manifest binding remain incomplete.
- Issue #27's accepted bounded investigation and repository synchronization
  under Issue #30 of the [three adopted D-TIME contracts](#adopted-d-time-input-retention-and-handoff-contracts).
  This closes only those adoption questions, not D-TIME as a whole, remaining
  D-* owners, runnable v1, Issue #2, Phase 0, or any Gate.
- Issues #31/#32's accepted investigation/input Validation and synchronization
  under Issue #33 of the
  [adopted current initial-PLY connection](#adopted-current-initial-ply-reuse-and-raw-time-connection).
  This is not post-fix focused/CUDA validation or completion of the remaining
  D-INIT/time/runtime/run contracts.
- repository synchronization under Issue #34 of the
  [adopted initial-variance contract](#adopted-initial-temporal-variance-and-d-time-connection).
  Its field/initial-value/V-B adoption and Step 5 CPU initialization handoff
  are complete, not actual GPU numerical validation or remaining time/run decisions.
- Issue #35's bounded P2, Issue #36's bounded P1, Issue #37's bounded P3,
  Issue #38's bounded P5, Issue #39's bounded P4, and Issue #41's bounded P6 component implementation
  and historical CPU acceptance, now integrated into
  [Step 5 functional acceptance](#step-5-functional-acceptance), not Gate closure;
- Step 5's integrated immutable configuration, lightweight entry and minimal
  same-process existing-runtime connection, including the reviewed Fix and
  supplemental CPU validation, followed by completed documentation review and
  consolidated user-owned research Git;
- the [Step 6 camera A implementation and CPU validation](#step-6-canonical-camera-handoff),
  followed by completed documentation review, user research Git, push confirmation
  and Step 6 completion, not full P0-0 runtime, CUDA/training or Gate closure;
- the [Step 7 A implementation and CPU validation](#step-7-learning-update-consistency-and-cpu-validation),
  advisor-reviewed and user-functionally-accepted. Its document review,
  consolidated user research Git, push confirmation and Step completion remain pending.

The adopted D-DATA, initialization/LR/batch/SH, population inputs and bounded
handoff, seed/initial-loss/time/V-B and closed-key contracts are enforced within
that accepted Step 5 connection. Step 7 separately accepts population execution
and screen statistics only within its bounded CPU scope, not actual GPU health,
formal reporting/publication or other later owners, nor run values.
The withdrawn positive-only/no-cap restriction
remains withdrawn; no approved policy is reselected.

Not complete and not authorized by this document sync:

- remaining formal policy selection listed in Open items;
- Step 7 document review, consolidated user research Git, push confirmation
  and completion processing;
- remaining source, config, test or tool fixes beyond the accepted Step 5
  connection, Step 6 A camera scope and Step 7 A update/population/save-test scope,
  including remaining camera runtime/CUDA and renderer mathematics, later
  checkpoint/reference consumers and environment provenance;
- actual-runtime/device/CUDA and real-data training validation of P0-T1/T2/T3
  beyond their accepted Step 7 implementation/CPU connection;
- remaining post-fix focused validation or CUDA build;
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
- D-PREFILTER validation beyond the accepted Python handoff: independent
  value/gradient and fresh-binary CUDA validation, and checkpoint/manifest identity binding under the
  [adopted contract](#approved-temporal-prefilter-contract); adoption and safety
  conditions are decided, while original `Σ_tt` health and the remainder of
  D-TIME remain unresolved. Other D-* and run values are not selected by this
  decision;
- D-TIME work beyond the accepted CPU connection and the unresolved time, D-INIT,
  runtime/renderer, D-OPT, D-SH/P0, and run-value boundaries listed with
  [its three adopted contracts](#adopted-d-time-input-retention-and-handoff-contracts);
  those three contracts and the
  [current initial-PLY adoption](#adopted-current-initial-ply-reuse-and-raw-time-connection)
  and [initial-variance adoption](#adopted-initial-temporal-variance-and-d-time-connection)
  are no longer open adoption questions; actual GPU cast/activation and
  numerical health validation remain outstanding;
- advisor review of this post-acceptance sync, then consolidated user research
  Git, push confirmation and completion processing for
  [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation). Steps 5 and 6
  are complete, not awaiting another Fix, local-design approval or Git checkpoint.
  Their actual runtime/CUDA and separate downstream owner dependencies remain
  open. The advisor considers the next functional Step
  only after completion, without a new number or scope assigned here;
- the remaining pre-integration owner details and per-run values classified
  [above](#remaining-field-and-owner-boundary), not reopening P1–P6, PF/time/V-B,
  D-DATA, minimal initialization/LR/batch/SH, population inputs/conditions,
  JSON/mode, or effective `eval=True`. The
  [Step 5 connection contract](#adopted-step-5-minimal-connection-contract)
  also closes seed binding, initial loss branches, time/cast handoff and key
  membership; bounded enforcement and CPU connection are accepted, while
  actual GPU cast/activation health and CUDA validation remain unfinished. Positive rigid/motion use is unsupported, not enabled by
  the common nonnegative type; future use needs separate adoption/validation;
- actual device/CUDA and real-data training validation of the completed-update
  transaction beyond [Step 7 A](#step-7-learning-update-consistency-and-cpu-validation);
  its implementation/CPU connection and approved event order are accepted,
  not pending source Fix or local loop/helper design;
- remaining necessary initial-state-zero diagnostic/reporting/failure details;
  blanket deletion is not adopted, and optional TensorBoard use is retained.
  No complete metric/PNG/visualization redesign is an added Step 5 prerequisite;
  the diagnostic remains outside formal iteration, completed test schedules,
  checkpoint, test selection, and best selection;
- the existing constant-depth `easy_cmap` diagnostic-output problem recorded
  in [Step 7](#step-7-learning-update-consistency-and-cpu-validation): the production
  helper is unmodified. Review its handling and any necessary bounded correction
  before real-GPU/pilot diagnostic acceptance, without an automatic Fix or new Step;
- numeric final iteration and pilot/formal training schedules;
- actual GPU/training population validation beyond the accepted
  [eight-field input and setting handoff](#adopted-configurability-and-population-conditions)
  and [Step 7 CPU connection](#step-7-learning-update-consistency-and-cpu-validation);
  field names, finite/unlimited representation, domains, stopping-threshold
  meaning, inactive temporal-field exclusion, and event/prune/reset conditions
  are adopted, not open choices; remaining owner details stay open;
- actual pilot/formal population settings and densification/prune/reset values;
- actual GPU validation of the population owner's statistics/point correspondence
  (former responsibility 10); child-radius treatment and window lifetime are
  implemented and CPU-accepted in Step 7, not unresolved design choices or
  measured GPU results;
- final corrected output directory name and run identity;
- remaining independent forward-oracle and gradient-validation fixtures beyond
  the accepted Step 6 A CPU supported-mode/pre-GPU rejection coverage,
  including actual camera device/CUDA forward/gradient validation;
- manifest schema/version impact;
- selection-free test-report metric, cadence, and acceptance criteria;
- checkpoint field-level semantic/state schema;
- only if exact resume is pursued, its restored field ownership and numerical
  or bitwise equivalence threshold;
- actual run seed, runtime GPU evidence, deterministic claims/CUDA validation,
  and nonfinite/OOM/partial-failure policy; the one-seed input/initialization
  connection is adopted, not a guarantee of bitwise reproduction or resume;
- fresh corrected-source build and focused CUDA acceptance of the `4dgs310`
  first candidate, including the observed PyTorch-CUDA/system-nvcc separation
  and the required execution/build provenance record; no package update is
  implied;
- remaining fields of the complete effective-config, dataset, source,
  environment, and build provenance schema; the single-authority/pre-JIT
  policy and `4dgs310` first-candidate role are decided;
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
synchronized together with the Issue #17 implementation/environment
clarification, Issue #22 adopted JSON/single-mode boundary, and Issue #24's
adopted P1–P6 partial field contracts and Issue #26's adopted D-PREFILTER
supplement, Issue #30's three adopted D-TIME contracts, Issue #33's
[adopted current-PLY connection](#adopted-current-initial-ply-reuse-and-raw-time-connection),
and Issue #34's [adopted initial-variance contract](#adopted-initial-temporal-variance-and-d-time-connection).
The [Step 5 minimal connection](#step-5-functional-acceptance) and
[Step 6 camera A](#step-6-canonical-camera-handoff) are complete, including
documentation review and user research Git. Current
[Step 7 A implementation and CPU validation](#step-7-learning-update-consistency-and-cpu-validation)
are advisor-reviewed and user-functionally-accepted. Next are advisor document review,
consolidated user research Git, push confirmation and completion processing;
this document task does not perform them or complete Step 7. The advisor
considers the next function only after completion. Later-owner source fixes,
downstream consumer integration, fresh build, actual device transfer and
GPU/training validation remain outstanding, not new Step 5/6/7 CPU
acceptance conditions. No separate document-only Git trigger is created.
P0-0, P0-1, P0-2, P0-3,
P0-T1, P0-T2, P0-T3, the separate alpha-cap renderer-math responsibility, and
P0-A6 retain their remaining runtime/owner validation requirements. The Step 6
and Step 7 CPU results do not unconditionally close those roots or any Gate;
P0-T7 remains unfixed but unreachable for the
first baseline. Checkpoint field schema/semantic validation remains the P0-T6
diagnostic-foundation root, while exact resume remains a later independent
root.
