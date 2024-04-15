#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:33:36 2021

@author: gustavo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import shutil
import sys
# setting path
sys.path.append('./util')
#from util import normalization_imgs

import nibabel
import logging
import os
import re
import monai



class preProcessing():

    def __init__(self, opt):# MR will load the three 
        
        self.datasave=opt.datasave
        self.dataroot=opt.dataroot
        self.modal=opt.modal.split('-')
        self.option=opt.option
        self.extension=opt.extension
        self.selectedIds=opt.selectedIds
        self.rate=[float(opt.rate[i]) for i in range(len(opt.rate))]
        self.retrieve_info()
    
    # Function to extract all the numbers from the given string
    def getNumbers(self,str):
        array = re.findall(r'[0-9]+', str)
        return array
    
    def retrieve_info(self):
        self.data=[]
        for fname in os.listdir(self.dataroot):
            if not fname.startswith('.') and not fname.endswith('csv'):
                #self.data.append((self.getNumbers(fname)[2],fname.split('_')[0]))#depend of the dataset, choose id after two _
                self.data.append(fname)#depend of the dataset, choose id after two _
                #self.folderNew.append(fname.split('_')[0])#use 1 if 'Hektor2021-CHUS007' else 0 Hektor2021
                # if dummy:
                #     assert(folderOld not in self.folderNew)
                #     folderOld=self.folderNew
                # else:
                #     folderOld=self.folderNew
                #     dummy=True

    def exam_upload(self,id_): # modality = T2, T1, Fa, MR or CT
        folder=id_[1]
        path_patient=os.path.join(self.dataroot,folder+'_'+id_[0])
        self.patients={}
        data=[]
        for modal_name in self.modal:
            folder=path_patient.split('/')[-1]
            path=os.path.join(path_patient,folder+'_'+modal_name+self.extension)
            data.append(nibabel.as_closest_canonical(nibabel.load(path)))
            if modal_name==self.modal[-1]:
                self.patients[id_[0]]=data
                data=[]

    def exam_uploadVProstate(self,id_,position): # modality = T2, T1, Fa, MR or CT
        folder=id_
        path_patientFolder=os.path.join(self.dataroot,folder)
        countModal=0
        listaPerPatient=os.listdir(path_patientFolder)
        listaPerPatient.sort()
        for fname in listaPerPatient:
            src_path=os.path.join(path_patientFolder,fname)
            if id_ in fname:
                for modality in self.modal[:-1]:
                    if modality in fname:
                        dest_path=os.path.join(self.datasave,'imagesTr')
                        new_name=id_+'_'+str(position).zfill(5)+'_'+str(countModal).zfill(4)+self.extension
                        countModal+=1
                        self.copy_and_rename(src_path, dest_path, new_name)
            elif "mask_" in fname:
                if self.modal[-1] in fname:
                    dest_path=os.path.join(self.datasave,'labelsTr')
                    new_name=id_+'_'+str(position).zfill(5)+self.extension
                    self.copy_and_rename(src_path, dest_path, new_name)
            else:
                continue
            
    def copy_and_rename(self,src_path, dest_path, new_name):
        # Copy the file
        shutil.copy(src_path, dest_path)
        # Rename the copied file
        new_path = f"{dest_path}/{new_name}"
        shutil.move(f"{dest_path}/{src_path.split('/')[-1]}", new_path)

        
    def normalize(self,crop=True,roi_size=[128,128,128]):
        normalized_Moda={}
        transformation=monai.transforms.Compose([
            monai.transforms.NormalizeIntensity(),
            monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
            ])
        crop=monai.transforms.Compose([
            monai.transforms.AddChannel(),
            monai.transforms.CenterSpatialCrop(roi_size=[128,128,128])
            ])
        for key in self.modalities:
            if key!='seg':
                single_modal=self.modalities[key][0].get_fdata()[:,:,:]
                normalized_Moda[key]=transformation(single_modal)
            else:
                normalized_Moda[key]=self.modalities[key][0].get_fdata()[:,:,:]
    
            if crop:
                normalized_Moda[key]=crop(normalized_Moda[key])[0,:,:,:]
        return normalized_Moda
    
    
    def extractModal(self,id_):
        Moda=nibabel.funcs.concat_images(self.patients[id_[0]][0:-1], check_affines=True, axis=None)
        return Moda,self.patients[id_[0]][-1]

        

    def print_info(self):
        
        logging.basicConfig(level=logging.INFO, format='\n %(levelname)s: %(message)s')
        logging.info(f'''exam {self.id} uploaded:
        serie:         {self.serie}
        center:        {self.center}
        sex:           {self.sex}
        size:          {self.size}
        age:           {self.age}        
        T2:            {self.T2_exist}
        T1:            {self.T1_exist}
        Fa:            {self.Fa_exist}
        CT:            {self.CT_exist}
        row_annt:      {self.row_annt}
        atypical:      {self.atypical}
        artifact:      {self.artifact}
        mayo:          {self.mayo}
        mutation:      {self.mutation}
        proPKD:        {self.proPKD}
        DFG:           {self.DFG}
        ''')