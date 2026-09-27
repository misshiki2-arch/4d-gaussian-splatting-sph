"""Read-only legacy-name projections of the SAME immutable FormalConfig.

Not an alternate resolver, CLI, set of defaults, or authentication token.
Constructed only by the validated formal entry's runtime preparation.
"""
from typing import NamedTuple


class ConsumerView(NamedTuple):
    config: object
    group: str

    def __getattr__(self, name):
        c = self.config
        if self.group == 'dataset':
            values = dict(source_path=c.paths.source_path, model_path=c.paths.output_directory,
                          sh_degree=c.model.spatial_sh_degree, white_background=c.dataset.white_background,
                          eval=c.dataset.eval, extension=c.dataset.extension, resolution=c.dataset.resolution.divisor,
                          frame_ratio=c.dataset.time.divisor, num_extra_pts=c.initialization.num_extra_pts,
                          dataloader=c.fixed.dataloader, data_device='cuda', loaded_pth='', prefilter_var=-1.0)
        elif self.group == 'optimization':
            p = c.optimization.population
            values = dict(c.optimization.learning_rate._asdict(), **c.optimization.loss._asdict())
            values.update(iterations=c.optimization.total_updates, percent_dense=p.percent_dense,
                          thresh_opa_prune=p.thresh_opa_prune, densify_grad_threshold=p.densify_grad_threshold,
                          densify_from_iter=p.densify_from_update, densify_until_iter=p.densify_until_update,
                          densification_interval=p.densification_interval, opacity_reset_interval=p.opacity_reset_interval,
                          densify_until_num_points=-1 if p.densify_stop_points == 'unlimited' else p.densify_stop_points,
                          densify_grad_t_threshold=None,
                          sh_increase_interval=c.optimization.sh.increase_interval)
        elif self.group == 'pipeline':
            values = dict(compute_cov3D_python=c.renderer.compute_cov3D_python,
                          convert_SHs_python=c.renderer.convert_SHs_python,
                          env_map_res=c.renderer.env_map_res, eval_shfs_4d=c.maximum_sh.legacy_eval_shfs_4d,
                          debug=False, debug_pixel_x=-1, debug_pixel_y=-1,
                          debug_pixel_max_entries=0, debug_preprocess_target_index=-1)
        else:
            raise AttributeError(name)
        try:
            return values[name]
        except KeyError:
            raise AttributeError(name) from None
