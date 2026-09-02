# Corrected Camera Intrinsics Retraining Plan

Status: **plan-only / Investigation1-4 static audit complete / audit integration documented / formal-policy-not-finalized / source-fixes-not-started / pilot-not-started / formal-retraining-not-started / Viewer-frozen**

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

Formal policy selection, source fixes, focused validation, pilot training,
formal retraining, corrected artifact generation, and Viewer restart have not
started. P0 findings block only the gate whose accepted output would reach the
defect; Viewer-only defects do not unnecessarily block corrected training, and
training-state defects cannot be deferred to artifact generation.

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

The implementation must distinguish two input modes:

1. intrinsics cameras use valid `fl_x/fl_y/cx/cy`, width, and height to derive
   the effective tanFov/focal vocabulary; a negative FoV sentinel remains raw
   provenance only;
2. legacy FoV-only cameras retain their valid FoV-derived path.

Both paths must yield one canonical effective-intrinsics result consumed by
full projection and footprint rasterization. If a contract is added or
expanded, one common builder must own it; renderer setup, manifest publication,
and tests must not independently reconstruct the same values.

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
therefore do not differentiate the executed forward graph. Whether the formal
SH evaluation position should be the original or conditional mean remains an
open design decision.

### P0-3: temporal SH degree, layout, and time-gradient contract

Temporal SH evaluation is nested under spatial degree greater than two. At
lower spatial degrees, model allocation and evaluator use do not agree. In
addition, the derivative of `cos(k * omega * delta_t)` has the wrong sign, and
temporal degree two overwrites rather than adds the degree-one time-gradient
contribution.

If formal temporal SH is enabled, these defects require correction. If the
formal policy instead disables temporal SH, allocation must still agree with
the evaluator and unsupported degree/layout combinations must fail closed
before formal training is allowed.

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

`eval=False` is a possible all-images-training policy, but it does not provide
a non-empty independent held-out population. It becomes a P0 when that mode is
combined with claims of held-out evaluation or with `chkpnt_best.pth`
selection based on the resulting empty/invalid test metric. Formal policy must
either define a real train/validation/test split or prohibit held-out/best
claims in all-images-training mode.

### P0-T5: resume destroys global-best identity

`best_psnr` and the metric provenance are not restored. This P0 is reached
when resume is permitted: the resumed run starts best tracking from zero and
may overwrite the prior global best with a merely local post-resume result.
Disabling resume for a run makes this branch unreachable but does not repair
resume fidelity.

### P0-T6: semantic checkpoint incompatibility is not fail-closed

Checkpoint restoration does not bind or validate all model, renderer,
dataset, and effective-config semantics. This P0 is reached when warm-start,
resume, or any cross-semantics load is permitted. Formal from-scratch startup
avoids warm-start reachability, but versioned semantic validation remains
required for any formal resume and for later artifact consumers.

### P0-T7: environment-map startup/resume lifecycle mismatch

This P0 is reached only when `env_map_res > 0`. The model/checkpoint may carry
an environment map, while training startup creates and assigns a new parameter
and optimizer on a separate lifecycle; restore does not establish a coherent
parameter/optimizer/schedule continuation. The formal branch must either
define and validate this lifecycle or reject environment maps.

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
used as the formal source checkpoint. It is the artifact-selection
propagation of P0-T4 and P0-T5.

## Root and dependent finding relationships

Root fixes and dependent consumer checks must be owned separately so the same
cause is not patched twice:

| Root finding | Dependent finding | Required boundary |
|---|---|---|
| P0-T1/T2/T5/T6 | P0-A1 | Fix the training/checkpoint transaction first; make the loader verify and publish it second. |
| P0-T6/T7 plus incomplete config snapshot | P0-A2 | Define effective config/checkpoint semantics first; reconstruct and assert them in the CUDA loader second. |
| P0-T4/T5 | P0-A8 | Define evaluation/best identity first; require that identity at artifact selection second. |
| P0-0 | camera portion of P0-A6 | Correct one common camera contract first; publish the exact executed values second. |
| none | P0-A3/A4/A5, quaternion vocabulary, source-build binding | Independent parser, evidence, handoff, and provenance responsibilities. |
| P0-T3 | population identity changes | Repair the optimizer/densification transaction; do not alter SPL4 source order as a supposed fix. |

## Renderer and CUDA P1 disposition

The following confirmed issues require an explicit disposition during formal
policy selection and before their branch is implemented:

- near, far, and off-center camera policy differences;
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
available. Items such as alpha-cap derivative policy and near/far behavior
require a design decision; this plan does not choose that policy yet.

## Bounded P2 findings

The following are recorded for later disposition but are not automatically
promoted to formal-retraining blockers:

- the determinant epsilon approximation in inverse-covariance backward;
- conservative radius inflation from the eigenvalue lower bound;
- possible `uint32` prefix-sum and signed `num_rendered` overflow;
- runtime behavior for an empty Gaussian population, which remains unverified;
- unclear combined meaning of `force_sh_3d` and temporal degree;
- a misleading radius-threshold expression.

## Integrated P1/P2 disposition by owner

The following register preserves Investigation1-4 P1/P2 facts while avoiding
repetition of a P0 root cause. A selected formal branch must correct or
fail-closed reject its P1 items. P2 items remain bounded robustness,
observability, determinism, or performance work unless later evidence promotes
them.

| Owner group | P1 disposition | P2 or bounded follow-up |
|---|---|---|
| camera/projection policy | Decide near/far, off-center principal point, raw sentinel, and effective-intrinsics validation. One common builder owns execution and publication. | Preserve finite/input diagnostics and bounded legacy-FoV coverage. |
| renderer branch policy | Resolve alpha-cap derivative, `rot_4d=false`, scaling modifier, Python covariance/SH precompute, allocation/evaluator layout, and unsupported-branch rejection. | Determinant epsilon, radius inflation, empty population, overflow, and radius-threshold diagnostics. |
| optimizer/learning-rate policy | Define temporal-position LR scheduling and supported parameter-group behavior; do not infer it from the spatial-only scheduler. | Log effective per-group LR without adding a second schedule owner. |
| densification/population policy | Define temporal selection, strict point-cap semantics, clone/split/prune ordering, prune-only behavior, opacity reset, and screen/world pruning. | Measure cap overshoot, nonfinite accumulator frequency, and memory pressure in bounded pilot runs. |
| evaluation/metric policy | P0-T4/T5 own split and global-best correctness; additionally decide metric, tie behavior, frequency, selection bias, clamp/channel/background, and whether test is selection-free. | Keep diagnostic train samples separate from acceptance metrics and visualization. |
| config/run mode/provenance | Define CLI/config precedence, reject legacy output/path reuse, capture the complete effective config, and replace executable config parsing where it reaches formal tooling. | Report path-remap and duplicate-basename ambiguity as bounded diagnostics. |
| checkpoint/resume | P0-T1/T2/T6 own transaction/schema correctness; additionally require atomic writes, scheduler/RNG/sampler/dataloader state policy, and explicit resume fidelity. | Record interruption/OOM behavior and checkpoint-size/hash cost. |
| dataset/mask | Bind transforms/images/masks and define whether alpha mask or luminance-derived sky participates in the formal loss. | Measure missing-mask/fallback incidence; do not infer correctness from a few debug images. |
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
- formal policy choices listed below;
- runtime acceptance of a corrected checkpoint/SPL4/CUDA Reference/Viewer
  bundle.

These items must not be reclassified as confirmed-correct without focused
post-fix validation, controlled metadata inspection, a corrected run, or
runtime artifact acceptance as applicable.

## Formal policy decisions intentionally left open

Audit integration classifies when each decision is required; it does not make
the decisions.

| Decision stage | Required decisions |
|---|---|
| decided and immutable | Do not reproduce historical negative-FoV-sentinel semantics; use the corrected shared renderer; train the formal baseline from scratch; never overwrite historical artifacts; keep the Viewer frozen; create a new population identity; use a fresh formal CUDA Reference; exclude legacy PNG reuse, SPL4-v1, and 3D-sequence PLY from formal evidence/parity. |
| required before implementation | Original or conditional SH evaluation position; spatial SH degree; temporal SH enablement/degree/layout; `rot_4d`; `force_sh_3d`; Python precompute support; near/far/off-center camera policy; alpha-cap derivative; environment-map support; formal run mode; exact iteration transaction; evaluation/save/densification/step ordering; temporal densification policy; strict point-cap/prune/opacity-reset policy. |
| required before formal retraining | Train/validation/test split; best metric and tie/update rule; resume fidelity or prohibition; versioned checkpoint schema; complete effective-config snapshot; deterministic seed ownership; pilot/formal schedules; nonfinite/failure policy; immutable output directory and atomic publication. |
| required before formal artifact generation | SPL4-v2 log/linear scale representation; PNG clamp/round/color/codec; full/range CUDA Reference purposes; manifest schema and validator; source-to-binary build provenance; bundle/index/external-digest ownership; direct evidence as formal same-invocation evidence or diagnostic-only. |
| required before Viewer restart | Corrected population and fixed range; Viewer provenance binding; strict parser acceptance; removal or versioned isolation of historical hard-coded ranges; Viewer capture/comparison bundle identity. |

Temporary branch candidates must not be presented as the formal contract.

## Pre-retraining validation requirements

After each separately authorized correction, validation proceeds from pure
contracts to bounded integration. No validation in this table was executed by
this documentation sync.

| # | Validation layer | Primary findings closed |
|---:|---|---|
| 1 | Pure CPU camera/field/index contract tests | P0-0, P0-A6, camera/projection P1 |
| 2 | Config precedence, run-mode, legacy-path, and unsupported-branch negative matrix | P0-A2, P0-T6/T7, config/run-mode P1 |
| 3 | Versioned checkpoint schema and exact iteration round-trip | P0-T1/T2/T6, P0-A1 |
| 4 | Training state-machine mock for iteration, evaluate, save, densify, reset, and step ordering | P0-T1/T2 |
| 5 | Optimizer-step and current-gradient preservation test | P0-T3, optimizer P1 |
| 6 | Densification clone/split/prune topology and Adam-row-state test | P0-T3 boundary, population P1 |
| 7 | Evaluation split, non-empty validation, best selection, and resume-global-best test | P0-T4/T5/A8 |
| 8 | Camera handoff numeric test for intrinsics and valid-FoV cameras | P0-0, camera P1 |
| 9 | Independent one-Gaussian forward oracle for projection, covariance, SH, alpha, and compositor | P0-1/P0-2/P0-3 and renderer P1 |
| 10 | CPU autograd and finite differences for xyz/time/scale/qL/qR/opacity/SH/screen mean | P0-1/P0-2/P0-3 and gradient P1 |
| 11 | CUDA forward and gradient smoke over every supported branch and boundary | P0-0 through P0-3, branch P1 |
| 12 | SPL4-v2 golden header/payload byte test | SPL4 representation and exporter P1 |
| 13 | Malformed/truncated/extra/overflow/nonfinite SPL4 parser matrix | P0-A3 |
| 14 | Checkpoint-SPL4 wrong-pair, count, range, order, and stale-file test | P0-A4 plus provenance P1 |
| 15 | CUDA manifest missing/wrong iteration/config/camera/population/artifact identity test | P0-A1/A2/A6/A8 |
| 16 | Clean-source/build-recipe/toolchain-to-loaded-binary identity test | P0-A7 |
| 17 | Direct-evidence production-invocation identity test | P0-A5 |
| 18 | Viewer bundle wrong-pair, stale provenance, wrong range, and historical-range rejection | P0-A3/A4 |
| 19 | Bounded pilot acceptance for updates, metrics, nonfinite state, checkpoints, and completion | all Gate A/B training findings |
| 20 | Formal output completeness, atomicity, index, external digest, and parent-binding check | publication P1 and P0-A6/A7/A8 |

Independent forward oracles precede CUDA comparison; Python-precompute paths
are not assumed to be independent oracles. One-sided checks cover temporal
eligibility, near/far, alpha skip/cap, early termination, determinant validity,
and tile bounds. The relevant validation report must be reviewed before its
next gate is opened.

## Manifest provenance requirements

The manifest must distinguish rather than conflate:

- camera mode;
- raw FoV values, including any sentinel;
- raw `fl_x/fl_y/cx/cy` values when present;
- image width and height;
- effective focal, tanFov, and FoV;
- the effective-intrinsics identity consumed by both projection and footprint
  rasterization;
- exact checkpoint identity and internal iteration;
- the selecting metric, validation population, and best-checkpoint lineage;
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
different raw value, publish CLI/default state as restored runtime state, or
self-hash a file by writing its claimed hash back into the same bytes. Required
fields and cross-bindings need a fail-closed acceptance validator. Unknown
evidence must remain explicitly unknown rather than inferred.

## Diagnostic render boundary

After all audits are integrated, the formal policy is fixed, required
corrections are implemented, and focused validation is accepted, the legacy
checkpoint may be rendered under the corrected renderer as a controlled
diagnostic. Compare that output with ground truth and the legacy-renderer
output to measure the effect of renderer correction on a fixed checkpoint.
This diagnostic must not be promoted to the formal corrected checkpoint and
must not overwrite any legacy artifact.

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

Gate A requires one formal camera contract; one selected SH/temporal branch;
the pilot-reachable parts of P0-0 through P0-3; P0-T1, P0-T2, and P0-T3; one
formal run mode and effective config; a new non-overwriting output owner; a
minimum versioned checkpoint schema; fail-closed finite/failure behavior; and
an explicit pilot evaluation split policy.

Pilot training may start only when:

- the formal pilot branch policy is unique and unsupported branches reject;
- camera/renderer P0s reached by that branch are corrected and focused
  forward/gradient validation is accepted;
- the requested update count is exact, including the final update;
- checkpoint/save state boundaries are unambiguous;
- densification iterations preserve the intended gradient/update transaction;
- train, validation, and test populations are explicit;
- any best metric uses a non-empty validation population;
- output is new, the legacy path is rejected, and the complete effective
  config is snapshotted;
- checkpoint schema is versioned and nonfinite state fails closed.

The pilot is a bounded intermediate run, not a formal checkpoint.

### Gate B: before formal retraining

Gate B requires accepted pilot evidence; frozen formal config and schedule;
exact dataset identity; clean source revision; reproducible build/runtime
identity; approved renderer branches; approved training, densification,
evaluation, metric, resume, seed, and failure policies; the full versioned
checkpoint/iteration transaction; immutable atomic output publication; no
legacy checkpoint warm-start; and no legacy output reuse.

Conditional P0s are handled here by reachability: P0-T4 must be resolved only
when held-out/best claims are made; P0-T5 only when resume is allowed; P0-T6
when resume/warm-start/cross-semantics loading is allowed; and P0-T7 only when
`env_map_res > 0`. A forbidden branch must fail closed rather than remain a
silent option.

### Gate C: before corrected SPL4 and CUDA Reference generation

Gate C requires the exact formal checkpoint internal iteration; metric,
effective-config, dataset, source, and checkpoint binding; P0-A1 and P0-A2
closure; SPL4-v2-only policy; exact checkpoint-to-SPL4 full mapping; a clean
source-to-loaded-binary build binding; strict exported header/payload
validation; fresh CUDA rendering; runtime camera/time/effective-renderer
publication; manifest acceptance validation; P0-A6, P0-A7, and P0-A8 closure;
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
| common camera contract | P0-0 and effective intrinsics | manifest/runtime camera publication in P0-A6 |
| renderer forward/backward | P0-1, P0-2, P0-3 and selected renderer P1 | training validation and CUDA Reference semantics |
| training state machine | P0-T1, P0-T2 | checkpoint labels and P0-A1 |
| optimizer/densification transaction | P0-T3 and population policy | final checkpoint population identity |
| evaluation/checkpoint selection | P0-T4, P0-T5 | P0-A8 |
| config/run launcher | complete effective config, formal mode, legacy-path rejection | P0-A2 and manifest config identity |
| versioned checkpoint schema | P0-T6, iteration/config/semantic fields | P0-A1/A2/A8 consumers |
| environment-map lifecycle | P0-T7 when enabled | CUDA reconstruction and runtime publication |
| dataset/evaluation split | train/validation/test and mask identity | best metric and artifact lineage |
| SPL4 exporter/provenance | v2 representation, atomic export, exact population binding | Viewer bundle input |
| SPL4 parser | P0-A3 | production evaluation inputs |
| CUDA Reference loader | P0-A1/A2, selected tensor mapping, fresh render | manifest and bundle output |
| manifest builder/validator | P0-A5/A6/A8 publication claims | Viewer/comparison acceptance |
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
3. Decide the formal camera, renderer, training, and branch policies.
4. Confirm one root owner and one bounded Fix responsibility at a time.

### Phase 1: training-critical foundation

5. Define the complete effective-config, run-mode, and non-overwriting output
   identity contract.
6. Define the versioned checkpoint schema and exact iteration transaction.
7. Implement the common camera-handoff Fix.
8. Implement selected-branch renderer forward/backward P0 Fixes.
9. Implement the training iteration/step/evaluate/save transaction Fix.
10. Implement densification gradient and optimizer-mutation transaction Fix.
11. Implement evaluation, best-checkpoint, and resume Fixes or formal branch
    rejection.
12. Correct every conditional P0 reached by the formal branch and fail-closed
    reject every unsupported branch.
13. Implement nonfinite/failure handling and atomic checkpoint publication.

### Phase 2: focused validation

14. Run the config/run/checkpoint negative matrix.
15. Run independent camera/renderer forward oracles.
16. Run CPU autograd and finite differences.
17. Build the selected CUDA source and run CUDA forward/gradient validation.
18. Run training state-machine, optimizer, and densification mocks.
19. Run evaluation, resume, and output-identity tests.
20. Run an integrated bounded small-scene smoke.

### Phase 3: pilot and formal training

21. Run small corrected pilot training after Gate A.
22. Review pilot updates, metrics, topology, checkpoints, and failure evidence.
23. Freeze the formal config, dataset, source/build, schedule, and policies.
24. Run the formal corrected baseline from scratch after Gate B.
25. Select and immutably publish the formal checkpoint and metric binding.

### Phase 4: corrected artifact path

26. Harden SPL4-v2 export, atomic publication, and population provenance.
27. Harden CUDA Reference reconstruction and runtime-state publication.
28. Establish the source/binary/dataset/checkpoint/render bundle contract.
29. Generate fresh corrected SPL4 and CUDA Reference after Gate C.
30. Fix the corrected population identity and record count.

### Phase 5: Viewer restart

31. Implement and validate the strict SPL4 parser.
32. Bind corrected provenance/bundle identity into Viewer handoff.
33. Design new representative records and a new fixed range.
34. Unfreeze the Viewer only after Gate D.
35. Rerun record-local, tile, sort, compositor, and PNG validation.
36. Run the matched corrected full-scene gate.
37. Continue only then to performance, interactive behavior, scalability, and
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
- the decision to keep the Viewer frozen.

Not complete and not authorized by this document sync:

- formal policy selection;
- source, config, test, or tool fixes;
- focused validation or CUDA build;
- pilot training or formal retraining;
- a corrected checkpoint, SPL4, CUDA Reference, or population identity;
- Viewer restart, fixed-range parity, full-scene correctness, or Acceptance
  Level 4;
- performance, interactive behavior, scalability, or LOD work.

## Open items

- formal supported renderer branch and fail-closed branch matrix;
- original or conditional SH evaluation position;
- spatial and temporal SH degree/layout policy;
- alpha-cap derivative policy;
- `rot_4d`, `force_sh_3d`, Python-precompute, and scaling-modifier policy;
- near, far, and off-center camera policy;
- final corrected output directory name and run identity;
- implementation location and exact common-builder API;
- independent forward-oracle and gradient-validation fixtures;
- manifest schema/version impact;
- corrected training schedule and acceptance criteria;
- corrected densification and pruning policy;
- train/validation/test split, best metric, and resume policy;
- versioned checkpoint and exact iteration transaction schema;
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
push has been performed by this documentation sync.
