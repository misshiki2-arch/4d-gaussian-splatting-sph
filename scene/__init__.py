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

import os
import torch
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.data_utils import CameraDataset

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], num_pts=100_000, num_pts_ratio=1.0, time_duration=None, *, formal_inputs=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.white_background = args.white_background
        self._formal_inputs = formal_inputs
        self._formal_resolution = None if formal_inputs is None else formal_inputs.config.dataset.resolution
        if formal_inputs is not None and (load_iteration is not None or args.loaded_pth or resolution_scales != [1.0]):
            raise ValueError('formal_scene_override')

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if formal_inputs is not None:
            scene_info = sceneLoadTypeCallbacks['Blender'](args.source_path, args.white_background, args.eval,
                num_pts=num_pts, time_duration=time_duration, extension=args.extension,
                num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader,
                formal_inputs=formal_inputs)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, num_pts_ratio=num_pts_ratio)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, num_pts=num_pts, time_duration=time_duration, extension=args.extension, num_extra_pts=args.num_extra_pts, frame_ratio=args.frame_ratio, dataloader=args.dataloader)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            self._initial_ply_path = scene_info.ply_path
            # Only fresh initialization prepares diagnostics. Formal still
            # prepares before claim and writes afterwards; legacy writes here.
            self._initial_cameras = [camera_to_JSON(i, cam) for i, cam in enumerate(
                list(scene_info.test_cameras) + list(scene_info.train_cameras))]
            if formal_inputs is None:
                self.write_initial_files()

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, formal_resolution=self._formal_resolution)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, formal_resolution=self._formal_resolution)
            
        if args.loaded_pth:
            self.gaussians.create_from_pth(args.loaded_pth, self.cameras_extent)
        else:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                if formal_inputs is None:
                    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                else:
                    self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent,
                        formal_time=formal_inputs.config.time_derivation)

    def write_initial_files(self, claim=None):
        if self._formal_inputs is not None:
            if claim is None or claim.path != self.model_path:
                raise ValueError('formal_scene_claim')
            with claim.open('input.ply', 'wb') as dest:
                dest.write(self._formal_inputs.ply_bytes)
            with claim.open('cameras.json', 'w') as dest:
                json.dump(self._initial_cameras, dest)
        else:
            with open(self._initial_ply_path, 'rb') as src, open(os.path.join(self.model_path, 'input.ply'), 'wb') as dest:
                dest.write(src.read())
            with open(os.path.join(self.model_path, 'cameras.json'), 'w') as dest:
                json.dump(self._initial_cameras, dest)

    def save(self, iteration):
        torch.save((self.gaussians.capture(), iteration), self.model_path + "/chkpnt" + str(iteration) + ".pth")

    def getTrainCameras(self, scale=1.0):
        return CameraDataset(self.train_cameras[scale].copy(), self.white_background, formal_resolution=self._formal_resolution)
        
    def getTestCameras(self, scale=1.0):
        return CameraDataset(self.test_cameras[scale].copy(), self.white_background, formal_resolution=self._formal_resolution)
