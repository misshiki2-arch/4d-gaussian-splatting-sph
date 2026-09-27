"""Integrated formal JSON validation, not a CLI or permission to train.

The sole validation entry accepts bytes, never caller-built parse data or
partial success tokens. Semantic checks precede the read-only P6 filesystem
preflight. Only their combined success returns a detached, deeply immutable
configuration. Raw dimensions are explicit owner inputs, not image evidence.
Dataset contents, first-run values (including c=5), float32 health, RNG,
runtime adapters, exclusive output claim and writers remain later owners.
Direct construction of a result type is not a verification credential.
"""

from typing import NamedTuple

import formal_config_json as p2
import formal_config_p1 as p1
import formal_config_p3 as p3
import formal_config_p4 as p4
import formal_config_p5 as p5
import formal_config_p6 as p6
import formal_config_time as time_config


class ConfigInputError(p2.JSONInputError):
    """Bounded codes only; no input names, values, paths or OS diagnostics."""


# The adopted fifteen object memberships, not a generic schema framework.
_OBJECT_KEYS = (
    ((), "schema run_mode dataset model renderer initialization optimization reporting checkpoint output"),
    (("dataset",), "kind source_path eval extension white_background resolution time"),
    (("dataset", "resolution"), "mode divisor"),
    (("dataset", "time"), "raw_interval divisor"),
    (("model",), "gaussian_dim spatial_sh_degree temporal_sh_degree rot_4d force_sh_3d sh_evaluation"),
    (("renderer",), "compute_cov3D_python convert_SHs_python scaling_modifier env_map_res override_color temporal_prefilter"),
    (("initialization",), "num_pts num_extra_pts time_variance_denominator seed"),
    (("optimization",), "total_updates batch_size learning_rate loss population sh"),
    (("optimization", "learning_rate"), "position_lr_init position_lr_final position_t_lr_init position_lr_max_steps feature_lr opacity_lr scaling_lr rotation_lr"),
    (("optimization", "loss"), "lambda_dssim lambda_opa_mask lambda_rigid lambda_motion"),
    (("optimization", "population"), "densify_stop_points percent_dense thresh_opa_prune densify_grad_threshold densify_from_update densify_until_update densification_interval opacity_reset_interval"),
    (("optimization", "sh"), "increase_interval"),
    (("reporting",), "test_updates"),
    (("checkpoint",), "intermediate_updates"),
    (("output",), "directory"),
)


class Resolution(NamedTuple):
    mode: str
    divisor: int
    raw_width: int
    raw_height: int
    width: int
    height: int


class TimeInput(NamedTuple):
    raw_interval: tuple[float, float]
    divisor: float


class Dataset(NamedTuple):
    kind: str
    source_path: str
    eval: bool
    extension: str
    white_background: bool
    resolution: Resolution
    time: TimeInput


class Model(NamedTuple):
    gaussian_dim: int
    spatial_sh_degree: int
    temporal_sh_degree: int
    rot_4d: bool
    force_sh_3d: bool
    sh_evaluation: str


class Renderer(NamedTuple):
    compute_cov3D_python: bool
    convert_SHs_python: bool
    scaling_modifier: float
    env_map_res: int
    override_color: None
    temporal_prefilter: str


class Initialization(NamedTuple):
    num_pts: int
    num_extra_pts: int
    time_variance_denominator: float
    seed: int


class LearningRate(NamedTuple):
    position_lr_init: float
    position_lr_final: float
    position_t_lr_init: float
    position_lr_max_steps: int
    feature_lr: float
    opacity_lr: float
    scaling_lr: float
    rotation_lr: float


class Loss(NamedTuple):
    lambda_dssim: float
    lambda_opa_mask: float
    lambda_rigid: float
    lambda_motion: float


class Population(NamedTuple):
    densify_stop_points: int | str
    percent_dense: float
    thresh_opa_prune: float
    densify_grad_threshold: float
    densify_from_update: int
    densify_until_update: int
    densification_interval: int
    opacity_reset_interval: int


class SHSchedule(NamedTuple):
    increase_interval: int


class Optimization(NamedTuple):
    total_updates: int
    batch_size: int
    learning_rate: LearningRate
    loss: Loss
    population: Population
    sh: SHSchedule


class Reporting(NamedTuple):
    test_updates: tuple[int, ...]


class Checkpoint(NamedTuple):
    intermediate_updates: tuple[int, ...]


class Output(NamedTuple):
    directory: str


class MaximumSH(NamedTuple):
    max_spatial_degree: int
    max_temporal_degree: int
    slot_count: int
    mode_slot_ranges: tuple[tuple[int, int], ...]
    legacy_eval_shfs_4d: bool


class TimeDerivation(NamedTuple):
    raw_interval: tuple[float, float]
    effective_interval: tuple[float, float]
    raw_duration: float
    effective_duration: float
    divisor: float
    time_variance_denominator: float
    raw_variance: float
    raw_scale: float
    effective_variance: float
    effective_scale: float
    log_effective_scale: float


class CanonicalPaths(NamedTuple):
    source_path: str
    output_directory: str


class FixedConditions(NamedTuple):
    """Approved meanings, not implemented runtime adapters or new inputs."""

    dataloader: bool
    mask_source: str
    shuffle: bool
    num_workers: int
    drop_last: bool
    delay_steps: int
    xyz_lr_schedule: str
    temporal_lr_schedule: str
    feature_rest_divisor: int
    spatial_temporal_scale_shared: bool
    left_right_rotation_shared: bool
    sh_stages: tuple[tuple[int, int], ...]


class FormalConfig(NamedTuple):
    """Validated values only when returned by resolve_formal_config.

    Declared paths retain explicit input; paths holds P6's canonical identity.
    Neither is a reservation or protection against later filesystem races.
    All containers below are immutable tuples; partial component dataclasses
    and mutable ParsedJSON are deliberately not retained in this state.
    """

    schema: str
    run_mode: str
    dataset: Dataset
    model: Model
    renderer: Renderer
    initialization: Initialization
    optimization: Optimization
    reporting: Reporting
    checkpoint: Checkpoint
    output: Output
    maximum_sh: MaximumSH
    time_derivation: TimeDerivation
    save_updates: tuple[int, ...]
    paths: CanonicalPaths
    fixed: FixedConditions


def _closed_objects(root):
    for path, names in _OBJECT_KEYS:
        obj = root
        for name in path:
            obj = obj[name]  # Parents have already passed their closed sets.
        if type(obj) is not dict:
            raise ConfigInputError("config_object_type")
        if obj.keys() != set(names.split()):
            raise ConfigInputError("config_object_keys")


def _integer(value, minimum=0, maximum=p2.MAX_INT):
    result = p2.require_int(value)
    if not minimum <= result <= maximum:
        raise ConfigInputError("config_integer_domain")
    return result


def _number(value, *, positive=False, upper=None):
    result = p2.require_num(value)
    if result < 0 or (positive and result == 0):
        raise ConfigInputError("config_number_domain")
    if upper is not None and result > upper:
        raise ConfigInputError("config_number_domain")
    return result


def resolve_formal_config(data: bytes, *, raw_width: int,
                          raw_height: int) -> FormalConfig:
    """P2 bytes -> pure checks/derivations -> P6 read-only -> immutable state.

    No YAML, defaults, semantic CLI overrides, injected partial results or
    runtime imports. Failure raises JSONInputError and returns no state.
    Cost is bounded by input size, not total_updates or population magnitude.
    """
    parsed = p2.parse_json_bytes(data)
    root = parsed.root
    _closed_objects(root)
    p1.check_p1_structure_and_fixed_inputs(parsed)
    sh = p3.derive_p3_maximum_sh(parsed)
    resolution = p4.derive_p4_resolution(
        parsed, raw_width=raw_width, raw_height=raw_height)
    schedules = p5.derive_p5_schedules(parsed)
    time = time_config.derive_time_interval_and_variance(parsed)

    # Pure field/domain checks. Preserve number-token distinctions until the
    # owning P2 check; do not round/cast all inputs or impose float32 policy.
    dataset = root["dataset"]
    if type(dataset["extension"]) is not str or dataset["extension"] != "":
        raise ConfigInputError("config_extension")
    if p2.require_bool(dataset["white_background"]):
        raise ConfigInputError("config_background")

    init = root["initialization"]
    initialization = Initialization(
        _integer(init["num_pts"], 1), _integer(init["num_extra_pts"], 0, 0),
        time.time_variance_denominator, _integer(init["seed"]))
    opt = root["optimization"]
    batch_size = _integer(opt["batch_size"], 1, 5146)
    lr = opt["learning_rate"]
    initial, final = _number(lr["position_lr_init"]), _number(lr["position_lr_final"])
    if (initial == 0) != (final == 0):
        raise ConfigInputError("config_xyz_lr_pair")
    learning_rate = LearningRate(
        initial, final, _number(lr["position_t_lr_init"]),
        _integer(lr["position_lr_max_steps"], 1), _number(lr["feature_lr"]),
        _number(lr["opacity_lr"]), _number(lr["scaling_lr"]),
        _number(lr["rotation_lr"]))
    loss_input = opt["loss"]
    loss = Loss(_number(loss_input["lambda_dssim"], upper=1),
                _number(loss_input["lambda_opa_mask"]),
                _number(loss_input["lambda_rigid"]),
                _number(loss_input["lambda_motion"]))
    if loss.lambda_rigid != 0 or loss.lambda_motion != 0:
        raise ConfigInputError("config_unsupported_loss_branch")

    pop = opt["population"]
    limit = pop["densify_stop_points"]
    if type(limit) is not str or limit != "unlimited":
        limit = _integer(limit, 1)
    population = Population(
        limit, _number(pop["percent_dense"], positive=True),
        _number(pop["thresh_opa_prune"], upper=1),
        _number(pop["densify_grad_threshold"]),
        _integer(pop["densify_from_update"]), _integer(pop["densify_until_update"]),
        _integer(pop["densification_interval"], 1),
        _integer(pop["opacity_reset_interval"], 1))
    optimization = Optimization(
        schedules.total_updates, batch_size, learning_rate, loss, population,
        SHSchedule(_integer(opt["sh"]["increase_interval"], 1)))

    # The only filesystem boundary. No state escapes before P6 succeeds;
    # no directory/content reader, output creation or exclusive claim here.
    paths = p6.preflight_p6_paths(parsed)

    model, renderer = root["model"], root["renderer"]
    return FormalConfig(
        schema=root["schema"], run_mode=root["run_mode"],
        dataset=Dataset(
            dataset["kind"], dataset["source_path"], dataset["eval"],
            dataset["extension"], dataset["white_background"],
            Resolution(dataset["resolution"]["mode"], resolution.divisor,
                       resolution.raw_width, resolution.raw_height,
                       resolution.width, resolution.height),
            TimeInput(time.raw_interval, time.divisor)),
        model=Model(p2.require_int(model["gaussian_dim"]),
                    sh.max_spatial_degree, sh.max_temporal_degree,
                    model["rot_4d"], model["force_sh_3d"], model["sh_evaluation"]),
        renderer=Renderer(renderer["compute_cov3D_python"], renderer["convert_SHs_python"],
                          p2.require_num(renderer["scaling_modifier"]),
                          p2.require_int(renderer["env_map_res"]),
                          renderer["override_color"], renderer["temporal_prefilter"]),
        initialization=initialization, optimization=optimization,
        reporting=Reporting(schedules.test_updates),
        checkpoint=Checkpoint(schedules.save_updates[:-1]),
        output=Output(root["output"]["directory"]),
        maximum_sh=MaximumSH(sh.max_spatial_degree, sh.max_temporal_degree,
                             sh.slot_count, sh.mode_slot_ranges, sh.legacy_eval_shfs_4d),
        time_derivation=TimeDerivation(*(getattr(time, f) for f in TimeDerivation._fields)),
        save_updates=schedules.save_updates,
        paths=CanonicalPaths(paths.source_path, paths.output_directory),
        fixed=FixedConditions(True, "existing_sidecar", True, 0, True, 0,
                              "log_interpolation", "constant", 20, True, True,
                              ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2))),
    )
