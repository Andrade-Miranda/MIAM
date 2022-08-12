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


import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d


from nnUNet.nnunet.training.data_augmentation.custom_transforms import  ConvertSegmentationToRegionsTransform
from nnUNet.nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData

from nnUNet.nnunet.training.dataloading.dataset_loading import load_dataset
from util.dataset_Testloading import DataLoaderTest3D
#from nnUNet.nnunet.training.dataloading.dataset_loading import DataLoader3D


import pickle


try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


class nnUNetDatasetTest(BaseDataset):
    
    def initialize(self, opt):
        
        self.opt=opt
        self.default_3D_augmentation_params = {
            "selected_data_channels": None,
            "selected_seg_channels": None,
            "scale_range": (0.85, 1.25),
            "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
            "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),

         "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
         "num_cached_per_thread": 1,
        } 
        self.seeds_val= None
        self.random_state=12345
        self.fold=opt.fold

        #default preprocessing folder - default plan
        self.preprocessing_output_dir='./nnUNet/data/nnUnet_preprocessed' 


        #check if I have to fuse region, this is particular useful for brats dataset
        if opt.region == 'None':
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

    def get_default_augmentation(self,
                                 border_val_seg=-1, pin_memory=True):
        
        regions=self.regions
        params=self.default_3D_augmentation_params

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
        val_transforms = Compose(val_transforms)

        return val_transforms


    def LoadData(self):  
        
        task=self.opt.dataroot+'Test'
        p = os.path.join(self.preprocessing_output_dir,task,'nnUNetData_plans_v2.1_stage0')
        dataset = load_dataset(p, 0)
        self.dataset_val=dataset
        plan=self.opt.plan
        with open(os.path.join(os.path.join(self.preprocessing_output_dir,task,plan)), 'rb') as f:
            plans = pickle.load(f)

        #plans['plans_per_stage'][0]['patch_size']=self.opt.imageSize
        # basic_patch_size = self.get_patch_size(np.array(plans['plans_per_stage'][0]['patch_size']),
        #                                       self.default_3D_augmentation_params['rotation_x'],
        #                                       self.default_3D_augmentation_params['rotation_y'],
        #                                      self.default_3D_augmentation_params['rotation_z'],
        #                                       self.default_3D_augmentation_params['scale_range'])
    
        transform = self.get_default_augmentation()
        #dval = DataLoader3D(self.dataset_val, basic_patch_size, np.array(plans['plans_per_stage'][0]['patch_size']).astype(int), self.opt.Val_batchSize)
        dval = DataLoaderTest3D(self.dataset_val,transform,self.opt.Val_batchSize)
        self.val_loader=dval
        
        return self.val_loader
    
    def __len__(self):
        return len(self.dataset_val)
    

    def name(self):
        return self.opt.dataroot+'Test'  
    
    

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
    
    
    
    
