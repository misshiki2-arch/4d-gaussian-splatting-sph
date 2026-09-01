# Corrected Camera Intrinsics Retraining Plan

Status: **plan-only / implementation-not-started**

This document records the approved transition from the historical split
camera/raster baseline to a corrected camera-intrinsics training baseline. It
does not implement the renderer fix, run a diagnostic render, start training,
export assets, generate a CUDA Reference, or authorize Git operations.

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

## Repository identity

- fork: `misshiki2-arch/4d-gaussian-splatting-sph`
- origin: `git@github.com:misshiki2-arch/4d-gaussian-splatting-sph.git`
- legacy fork HEAD: `7663366f823b0beea9cedf76013840f20f7cd563`
- official upstream: `https://github.com/fudan-zvg/4d-gaussian-splatting`
- official upstream base: `63725f21d4adc29669e565ae10e6b3ad6e0d1250`

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

## Focused validation requirements

Before retraining, focused tests must cover at least:

- intrinsics-camera effective tanFov and focal values;
- agreement between full projection and footprint rasterizer identity;
- preservation of the legacy valid-FoV-only camera path;
- width/height and principal-point handling;
- fail-closed treatment of invalid or incomplete camera inputs;
- manifest values matching the exact effective settings used by execution.

The implementation and test report must be reviewed before any formal
corrected-baseline output is generated.

## Manifest provenance requirements

The manifest must distinguish rather than conflate:

- camera mode;
- raw FoV values, including any sentinel;
- raw `fl_x/fl_y/cx/cy` values when present;
- image width and height;
- effective focal, tanFov, and FoV;
- the effective-intrinsics identity consumed by both projection and footprint
  rasterization;
- renderer/source identity and output identity.

It must not publish normalized positive values while execution consumes a
different raw value.

## Diagnostic render boundary

After the handoff and focused tests are accepted, the legacy checkpoint should
be rendered under the corrected renderer as a controlled diagnostic. Compare
that output with ground truth and the legacy-renderer output to measure the
effect of the renderer correction on a fixed checkpoint. This diagnostic must
not be promoted to the formal corrected checkpoint and must not overwrite any
legacy artifact.

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
for the new checkpoint. Representative records and the fixed range must be
selected again after the new population identity is known.

## Viewer unfreeze and acceptance gates

Viewer development remains frozen until all of the following hold:

- corrected camera handoff implementation is reviewed;
- focused intrinsics and legacy-FoV validation is accepted;
- from-scratch corrected training is complete;
- new checkpoint, SPL4, and corrected CUDA Reference identities are fixed;
- new population count and fixed-range comparison design are reviewed.

After unfreeze, reuse the completed Step118-122 production, diagnostic,
comparison, capture, and PNG infrastructure, but rerun asset-dependent
acceptance:

1. canonical representative comparison;
2. new fixed-range semantic comparison;
3. tile-reference, sort, compositor, and PNG comparison;
4. corrected full-scene gate over the new canonical population count without
   silent omission;
5. interactive camera/time, performance, scalability, and LOD validation.

Infrastructure reuse does not authorize reuse of legacy acceptance values.

## Dependency order

1. Complete the Investigation11 documentation synchronization.
2. User creates Viewer and CUDA/training Git checkpoints.
3. Implement the shared renderer camera intrinsics handoff fix.
4. Run focused intrinsics-camera and legacy-FoV-camera tests.
5. Publish raw input and effective rasterizer intrinsics in the manifest.
6. Render the legacy checkpoint with the corrected renderer for diagnosis.
7. Keep the diagnostic result out of formal checkpoint acceptance.
8. Retrain from scratch in a new output directory.
9. Run the four-way ground-truth/legacy/corrected comparison.
10. Export a new SPL4.
11. Generate a corrected CUDA Reference in a new directory.
12. Fix new population provenance, record count, camera, and time.
13. Unfreeze the Viewer.
14. Run representative comparison.
15. Establish a new fixed-range semantic gate.
16. Validate tile-reference, sort, compositor, and PNG parity.
17. Run the corrected full-scene gate.
18. Continue to interactive, performance, scalability, and LOD work.

## Open items

- final corrected output directory name and run identity;
- implementation location and exact common-builder API;
- focused test fixtures for both camera modes;
- manifest schema/version impact;
- corrected training schedule and acceptance criteria;
- new checkpoint iteration and population count;
- SPL4 export identity;
- corrected CUDA Reference output identity;
- representative selection and fixed-range design;
- corrected semantic and image thresholds;
- full-scene resource and performance plan.

No camera fix, render, training, export, CUDA Reference generation, branch,
commit, or push has been performed by creating this plan.
