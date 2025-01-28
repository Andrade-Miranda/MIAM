#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 23:48:10 2022

@author: gustavo
"""
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *

class nnUNETPlanning():
    
    def __init__(self,opt):
        self.opt=opt
        self.preprocessing_output_dir='./nnUNet/data/nnUnet_preprocessed'
    
    def load_my_plans(self):
        plan = load_pickle(join(self.preprocessing_output_dir,self.opt.dataroot,self.opt.plan),mode="rb")
        return plan    
    
    def load_properties_case_identifier(self, case_identifier):
        with open(join(self.preprocessing_output_dir, "%s.pkl" % case_identifier), 'rb') as f:
            properties = pickle.load(f)
        return properties





def create_lists_from_splitted_dataset_folder(folder):
    """
    does not rely on dataset.json
    :param folder:
    :return:
    """
    caseIDs = get_caseIDs_from_splitted_dataset_folder(folder)
    list_of_lists = []
    for f in caseIDs:
        list_of_lists.append(subfiles(folder, prefix=f, suffix=".nii.gz", join=True, sort=True))
    return list_of_lists


def get_caseIDs_from_splitted_dataset_folder(folder):
    files = subfiles(folder, suffix=".nii.gz", join=False)
    # all files must be .nii.gz and have 4 digit modality index
    files = [i[:-12] for i in files]
    # only unique patient ids
    files = np.unique(files)
    return files

def separate_modalities(preprocessing_output_dir,dataroot):
    properties= load_pickle(join(preprocessing_output_dir,dataroot,'dataset_properties.pkl'), 'rb')
    return properties