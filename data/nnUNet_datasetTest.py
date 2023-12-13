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
import torch
import SimpleITK as sitk

from multiprocessing import Process, Queue,Pool


import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
from batchgenerators.utilities.file_and_folder_operations import *


from nnUNet.nnunet.training.data_augmentation.custom_transforms import  ConvertSegmentationToRegionsTransform
from nnUNet.nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData

from nnUNet.nnunet.training.dataloading.dataset_loading import load_dataset
from util.dataset_Testloading import DataLoaderTest3D
#from nnUNet.nnunet.training.dataloading.dataset_loading import DataLoader3D
from nnUNet.nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer

import nnUNet.nnunet
import pickle


try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


class nnUNetDatasetTest(BaseDataset):
    
    def initialize(self, opt):
        
        self.opt=opt
        # self.default_3D_augmentation_params = {
        #     "selected_data_channels": None,
        #     "selected_seg_channels": None,
        #     "scale_range": (0.85, 1.25),
        #     "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        #     "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
        #     "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),

        #  "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
        #  "num_cached_per_thread": 1,
        # } 
        self.seeds_val= None
        self.random_state=int(opt.seed)
        self.fold=opt.fold
        self.segmentation_export_kwargs=None
        self.segs_from_prev_stage=None
        #default preprocessing folder - default plan
        self.preprocessing_output_dir='./nnUNet/data/nnUnet_preprocessed' 
        self.overwrite_existing=False
        self.save_npz=False
        self.threeD=True
        
        
        task=self.opt.dataroot
        p = os.path.join(self.preprocessing_output_dir,task)        
        self.plans=load_pickle(join(p, opt.plan))

        #set the len of the data
        self.dataset_IDs = list(check_input_folder_and_return_caseIDs(self.opt.input_fold_prediction, load_pickle(join(p, self.opt.plan))['num_modalities']))
        

        #check if I have to fuse region, this is particular useful for brats dataset
        if opt.region == 'None':
            self.regions=None
        else:
            self.regions={ str(i): list(opt.region)[i] for i in range(len(opt.region))}
        
        

    def LoadData(self):  
        
        task=self.opt.dataroot
        p = os.path.join(self.preprocessing_output_dir,task)
        input_folder=self.opt.input_fold_prediction
        assert isfile(join(p, self.opt.plan)), "Folder must contain a plans.pkl file"
        expected_num_modalities = load_pickle(join(p, self.opt.plan))['num_modalities']
        segmentation_export_kwargs=self.segmentation_export_kwargs
        segs_from_prev_stage=self.segs_from_prev_stage
        
        plans=self.plans
        # check input folder integrity
        case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)

        output_files = [join(self.opt.output_dir, i + ".nii.gz") for i in case_ids]
        all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
        list_of_lists = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                      len(i) == (len(j) + 12)] for j in case_ids]
        
        assert len(list_of_lists) == len(output_files)
        if segs_from_prev_stage is not None: assert len(segs_from_prev_stage) == len(output_files)
        
        pool = Pool(self.opt.num_threads_nifti_save)
        results = []
        cleaned_output_files = []
        for o in output_files:
            dr, f = os.path.split(o)
            if len(dr) > 0:
                maybe_mkdir_p(dr)
            if not f.endswith(".nii.gz"):
                f, _ = os.path.splitext(f)
                f = f + ".nii.gz"
            cleaned_output_files.append(join(dr, f))
            
        if not self.overwrite_existing:
            print("number of cases:", len(list_of_lists))
            # if save_npz=True then we should also check for missing npz files
            not_done_idx = [i for i, j in enumerate(cleaned_output_files) if (not isfile(j)) or (self.save_npz and not isfile(j[:-7] + '.npz'))]

            cleaned_output_files = [cleaned_output_files[i] for i in not_done_idx]
            list_of_lists = [list_of_lists[i] for i in not_done_idx]
            if segs_from_prev_stage is not None:
                segs_from_prev_stage = [segs_from_prev_stage[i] for i in not_done_idx]

        print("number of cases that still need to be predicted:", len(cleaned_output_files))

        print("emptying cuda cache")
        torch.cuda.empty_cache()
            
        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in plans.keys():
                force_separate_z = plans['segmentation_export_params']['force_separate_z']
                interpolation_order = plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        print("starting preprocessing generator")
        preprocessing = self.preprocess_multithreaded(plans, list_of_lists, cleaned_output_files, self.opt.num_threads_preprocessing,
                                                 segs_from_prev_stage)
        
        print("starting prediction...")
           
        return preprocessing
    
    def __len__(self):
        return len(self.dataset_IDs)
    

    def name(self):
        return self.opt.dataroot+'Test'  
    
    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnUNet.nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans['preprocessor_name']
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(nnUNet.nnunet.__path__[0], "preprocessing")],
                                                         preprocessor_name,
                                                         current_module="nnUNet.nnunet.preprocessing")
        assert preprocessor_class is not None, "Could not find preprocessor %s in nnUNet.nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.plans['normalization_schemes'], self.plans['use_mask_for_norm'],
                                          self.plans['transpose_forward'], self.plans['dataset_properties']['intensityproperties'])

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.opt.stage][
                                                                 'current_spacing'])
        return d, s, properties
    
    
    def preprocess_save_to_queue(self, preprocess_fn, q, list_of_lists, output_files, segs_from_prev_stage, classes,
                             transpose_forward):
    # suppress output
    # sys.stdout = open(os.devnull, 'w')

        errors_in = []
        for i, l in enumerate(list_of_lists):
            try:
                output_file = output_files[i]
                print("preprocessing", output_file)
                d, _, dct = preprocess_fn(l)
                # print(output_file, dct)
                if segs_from_prev_stage[i] is not None:
                    assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(
                        ".nii.gz"), "segs_from_prev_stage" \
                                " must point to a " \
                                "segmentation file"
                    seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                    # check to see if shapes match
                    img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                    assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), "image and segmentation from previous " \
                                                                                 "stage don't have the same pixel array " \
                                                                                 "shape! image: %s, seg_prev: %s" % \
                                                                                 (l[0], segs_from_prev_stage[i])
                    seg_prev = seg_prev.transpose(transpose_forward)
                    seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                    seg_reshaped = to_one_hot(seg_reshaped, classes)
                    d = np.vstack((d, seg_reshaped)).astype(np.float32)
                    """There is a problem with python process communication that prevents us from communicating obejcts 
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is 
                                                        communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long 
                                                        enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually 
                                                        patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will 
                                                        then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either 
                                                        filename or np.ndarray and will handle this automatically"""
                print(d.shape)
                if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
                    print(
                    "This output is too large for python process-process communication. "
                    "Saving output temporarily to disk")
                    np.save(output_file[:-7] + ".npy", d)
                    d = output_file[:-7] + ".npy"
                q.put((output_file, (d, dct)))
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print("error in", l)
                print(e)
        q.put("end")
        if len(errors_in) > 0:
            print("There were some errors in the following cases:", errors_in)
            print("These cases were ignored.")
        else:
            print("This worker has ended successfully, no errors to report")
    # restore output
    # sys.stdout = sys.__stdout__


    def preprocess_multithreaded(self,trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None):
        if segs_from_prev_stage is None:
            segs_from_prev_stage = [None] * len(list_of_lists)

        num_processes = min(len(list_of_lists), num_processes)
    
        if trainer['num_classes']==1:
            trainer['num_classes']+=1

        classes = list(range(1, trainer['num_classes']))
        q = Queue(1)
        processes = []
        for i in range(num_processes):
            pr = Process(target=self.preprocess_save_to_queue, args=(self.preprocess_patient, q,
                                                            list_of_lists[i::num_processes],
                                                            output_files[i::num_processes],
                                                            segs_from_prev_stage[i::num_processes],
                                                            classes, trainer['transpose_forward']))
            pr.start()
            processes.append(pr)

        try:
            end_ctr = 0
            while end_ctr != num_processes:
                item = q.get()
                if item == "end":
                    end_ctr += 1
                    continue
                else:
                    yield item

        finally:
            for p in processes:
               if p.is_alive():
                   p.terminate()  # this should not happen but better safe than sorry right
               p.join()

            q.close()




def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids),
          np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining),
              np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids



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
    
    
    
    
