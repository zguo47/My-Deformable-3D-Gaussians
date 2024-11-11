#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.load2gpu_on_the_fly = False
        self.is_blender = False
        self.is_6dof = False

        # Dynamic model
        self.dynamic = False      
        self.dynamic_model = "deform" # or "per_frame"
        self.canonical_frame_id = 0
        self.shuffle_frames = False # Set True if "per_frame"

        self.view_start = 1 # dynamic = 1, static = 0.
        self.num_views = 1
        self.total_num_views = 1

        self.D = 6
        self.W = 64
        self.xyz_multires = 6
        self.t_multires = 4

        # ToRF dataset
        self.dataset_type = "quad" # or "mitsuba" or "real"
        self.total_num_views = 60 # add 1 for real world sequences
        self.train_views = ""
        self.total_num_spiral_views = 60

        self.tof_image_width = 320
        self.tof_image_height = 240
        self.tof_scale_factor = 1.0

        self.color_image_width = 320
        self.color_image_height = 240
        self.color_scale_factor = 1.0
        
        self.min_depth_fac = 0.05
        self.max_depth_fac = 0.55
        self.depth_range = 16.0 # c/f, twice the unambiguous range of the ToF sensor
        self.phase_offset = 0.0

        self.use_view_dependent_phase = False

        self.init_method = "random" # or "phase"
        self.num_points = 10_000 
        self.phase_resolution_stride = 20
        self.initial_opacity = 0.1

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.warm_up = 3_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.deform_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0007

        self.lambda_depth = 1.0
        super().__init__(parser, "Optimization Parameters")
    
    def extract(self, args):
        g = super().extract(args)
        if args.dynamic:
            # if args.dynamic_model == "per_frame":
            #     g.iterations = g.dynamic_first_iterations + (dataset_args.num_views - 1) * g.dynamic_rest_iterations
            #     g.position_lr_max_steps = [g.dynamic_first_iterations] + [g.dynamic_rest_iterations] * (dataset_args.num_views - 1)

            if args.dynamic_model == "deform":
                # g.iterations = g.warm_up + g.adaptive_step_iterations * dataset_args.num_views // (g.adaptive_step * 2)
                
                g.position_lr_max_steps = [g.warm_up] + [g.densify_until_iter - g.warm_up] + [g.iterations - g.densify_until_iter]
                # g.position_lr_max_steps = [g.warm_up] + [g.adaptive_step_iterations] * (dataset_args.num_views // (g.adaptive_step * 2))
        else:
            g.position_lr_max_steps = g.iterations
            
            g.densify_until_iter = g.iterations // 2
        return g


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
