#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:09:39 2021

@author: gustavo
"""

#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from data.base_dataset import BaseDataset
import os
from copy import deepcopy
from sklearn.model_selection import KFold
from collections import OrderedDict


import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d


from nnUNet.nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnUNet.nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform

from nnUNet.nnunet.training.dataloading.dataset_loading import DataLoader3D, load_dataset
from util.dataset_Testloading import DataLoaderTest3D

import pickle


try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None



class nnUNetDataset(BaseDataset):
    
    def initialize(self, opt):
        
        self.opt=opt
        self.default_3D_augmentation_params = {
            "selected_data_channels": None,
            "selected_seg_channels": None,

            "do_elastic": True,
            "elastic_deform_alpha": (0., 900.),
            "elastic_deform_sigma": (9., 13.),
            "p_eldef": 0.2,

            "do_scaling": True,
            "scale_range": (0.85, 1.25),
            "independent_scale_factor_for_each_axis": False,
            "p_independent_scale_per_axis": 1,
            "p_scale": 0.2,

            "do_rotation": True,
            "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_p_per_axis": 1,
            "p_rot": 0.2,

            "random_crop": False,
            "random_crop_dist_to_border": None,

        "do_gamma": True,
        "gamma_retain_stats": True,
        "gamma_range": (0.7, 1.5),
       "p_gamma": 0.3,

        "do_mirror": True,
        "mirror_axes": (0, 1, 2),

        "dummy_2D": False,
        "mask_was_used_for_normalization": None,
        "border_mode_data": "constant",

        "all_segmentation_labels": None,  # used for cascade
        "move_last_seg_chanel_to_data": False,  # used for cascade
        "cascade_do_cascade_augmentations": False,  # used for cascade
        "cascade_random_binary_transform_p": 0.4,
        "cascade_random_binary_transform_p_per_label": 1,
        "cascade_random_binary_transform_size": (1, 8),
        "cascade_remove_conn_comp_p": 0.2,
        "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
        "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

        "do_additive_brightness": False,
        "additive_brightness_p_per_sample": 0.15,
        "additive_brightness_p_per_channel": 0.5,
        "additive_brightness_mu": 0.0,
        "additive_brightness_sigma": 0.1,

        "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
        "num_cached_per_thread": 1,
        }

        default_2D_augmentation_params = deepcopy(self.default_3D_augmentation_params)

        default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
        default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
        default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        default_2D_augmentation_params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
        default_2D_augmentation_params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

    # sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
    # not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
    # transform them back after augmentation
        default_2D_augmentation_params["dummy_2D"] = False
        default_2D_augmentation_params["mirror_axes"] = (0, 1)  # this can be (0, 1, 2) if dummy_2D=True
        
        self.MoreAug=opt.MoreAug
        if self.opt.Deterministic:##fix seed to always generate the same augmentation
            self.seeds_train, self.seeds_val= [i for i in range(self.default_3D_augmentation_params.get('num_threads'))], [i for i in range(self.default_3D_augmentation_params.get('num_threads'))] ##fix seed to always generate the same augmentation
        else:
            self.seeds_train, self.seeds_val= None, None 
           
        self.n_splits=self.opt.n_splits# seteado parra solo 5 splits por el momento
        self.random_state=12345
        self.fold=opt.fold

        #default preprocessing folder - default plan
        self.preprocessing_output_dir='./nnUNet/data/nnUnet_preprocessed' 

        
        
        #check if I have to fuse region, this is particular useful for brats dataset
        if opt.region == ['None']:
            self.regions=None
        else:
            self.regions={ str(i): list(opt.region)[i] for i in range(len(opt.region))}
        
        
        
    def get_patch_size(self,final_patch_size, rot_x, rot_y, rot_z, scale_range):
        if isinstance(rot_x, (tuple, list)):
            rot_x = max(np.abs(rot_x))
        if isinstance(rot_y, (tuple, list)):
            rot_y = max(np.abs(rot_y))
        if isinstance(rot_z, (tuple, list)):
            rot_z = max(np.abs(rot_z))
        rot_x = min(90 / 360 * 2. * np.pi, rot_x)
        rot_y = min(90 / 360 * 2. * np.pi, rot_y)
        rot_z = min(90 / 360 * 2. * np.pi, rot_z)
        coords = np.array(final_patch_size)
        final_shape = np.copy(coords)
        if len(coords) == 3:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        elif len(coords) == 2:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
        final_shape /= min(scale_range)
        return final_shape.astype(int)


    def get_default_augmentation(self,dataloader_train, patch_size,
                                 border_val_seg=-1, pin_memory=True):
        
        regions=self.regions
        params=self.default_3D_augmentation_params
        assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
        tr_transforms = []

        if params.get("selected_data_channels") is not None:
            tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if params.get("dummy_2D") is not None and params.get("dummy_2D"):
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
            alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
            border_cval_seg=border_val_seg,
            order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
            ))
        if params.get("dummy_2D") is not None and params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

        if params.get("do_mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

        if params.get("mask_was_used_for_normalization") is not None:
            mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
            tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
            if params.get("cascade_do_cascade_augmentations") and not None and params.get(
                    "cascade_do_cascade_augmentations"):
                tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    p_per_sample=params.get("cascade_random_binary_transform_p"),
                    key="data",
                    strel_size=params.get("cascade_random_binary_transform_size")))
                tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                    key="data",
                    p_per_sample=params.get("cascade_remove_conn_comp_p"),
                    fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                    dont_do_if_covers_more_than_X_percent=params.get("cascade_remove_conn_comp_fill_with_other_class_p")))

        tr_transforms.append(RenameTransform('seg', 'label', True))
        tr_transforms.append(RenameTransform('data', 'image', True))

        if regions is not None:
            tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'label', 'label'))
            #tr_transforms.append(ConvertSegToRegionsTransform(regions,keys="label"))

        tr_transforms.append(NumpyToTensor(['image', 'label'], 'float'))

        tr_transforms = Compose(tr_transforms)


        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                     params.get("num_cached_per_thread"), seeds=self.seeds_train,
                                                    pin_memory=pin_memory)

        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        if params.get("selected_data_channels") is not None:
            val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'label', True))
        val_transforms.append(RenameTransform('data', 'image', True))

        if regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'label', 'label'))
            #val_transforms.append(ConvertSegToRegionsTransform(regions,keys="label"))

        val_transforms.append(NumpyToTensor(['image', 'label'], 'float'))
        self.val_transforms = Compose(val_transforms)

        batchgenerator_val = DataLoaderTest3D(self.dataset_val,self.val_transforms,self.opt.Val_batchSize)
        # batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
        #                                          params.get("num_cached_per_thread"), seeds=self.seeds_val,
        #                                          pin_memory=pin_memory)
        return batchgenerator_train, batchgenerator_val


    def do_split(self,dataset,fold):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
            """
        dataset_directory=self.opt.expr_dir
        splits_file = os.path.join(dataset_directory, "splits_final.pkl")
        if not os.path.isfile(splits_file):
            splits = []
            all_keys_sorted = np.sort(list(dataset.keys()))
            
            if self.opt.loadsplit is not None:
                splits_file=os.path.join('./splits_plk', self.opt.loadsplit) #splits_final.pkl    
                print("Loading split...:%s" % splits_file)
                ##### temporary solution for create a new custom pkl split
                # from util.split_fnct import Brats_splitprogressive2
                # splits=Brats_splitprogressive2(all_keys_sorted,5)
                # with open(splits_file, "wb") as fout:
                #     pickle.dump(splits, fout, protocol=-1)
            else:
                print("Creating new split...")
                kfold = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                with open(splits_file, "wb") as fout:
                    pickle.dump(splits, fout, protocol=-1)
                    #save_pickle(splits, splits_file)
                    
        with open(splits_file, "rb") as f:
            splits = pickle.load(f)

        if self.fold == "all":
            tr_keys = val_keys = list(dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        dataset_tr = OrderedDict()
        for i in tr_keys:
            dataset_tr[i] = dataset[i]

        dataset_val = OrderedDict()
        for i in val_keys:
            dataset_val[i] = dataset[i]


        return dataset_tr,dataset_val


    def LoadData(self):  
        
        task=self.opt.dataroot
        p = os.path.join(self.preprocessing_output_dir,task,self.opt.planning_stage)# take always last stage(FULLRES)
        dataset = load_dataset(p, 0)
        
        # plan=self.opt.plan
        # with open(os.path.join(os.path.join(self.preprocessing_output_dir,task,plan)), 'rb') as f:
        #     plans = pickle.load(f)

        basic_patch_size = self.get_patch_size(np.array(self.opt.imageSize),
                                          self.default_3D_augmentation_params['rotation_x'],
                                          self.default_3D_augmentation_params['rotation_y'],
                                          self.default_3D_augmentation_params['rotation_z'],
                                          self.default_3D_augmentation_params['scale_range'])

        self.dataset_tr,self.dataset_val=self.do_split(dataset,self.fold)
        dtran = DataLoader3D(self.dataset_tr, basic_patch_size, np.array(self.opt.imageSize).astype(int), self.opt.batchSize)
        #dval = DataLoader3D(self.dataset_val, np.array(plans['plans_per_stage'][0]['median_patient_size_in_voxels']), np.array(plans['plans_per_stage'][0]['median_patient_size_in_voxels']), self.opt.Val_batchSize)
        tr, val = self.get_default_augmentation(dtran, np.array(self.opt.imageSize).astype(int))
        self.train_loader, self.val_loader=tr, val
        
        return self.train_loader, self.val_loader
    
    def __len__(self):
        return len(self.dataset_tr)+len(self.dataset_val)
    
    def lengthData(self):
        return (len(self.dataset_tr),len(self.dataset_val))

    def name(self):
        return self.opt.dataroot#self.opt.dataroot.split('/')[-2]    
    
    

class ConvertSegToRegionsTransform(AbstractTransform):
    def __init__(self, regions: dict, seg_key: str = "label", output_key: str = "label", seg_channel: int = 0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict
    
    
    
    