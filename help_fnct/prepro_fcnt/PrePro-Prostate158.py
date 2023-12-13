import SimpleITK as sitk
from preprocessing_Picai import Sample    
from preprocessing_Picai import *
from pathlib import Path
from tqdm import tqdm
import os

import shutil
from typing import Union


PathLike = Union[str, Path]


def preprocess_picai_annotation(lbl: sitk.Image) -> sitk.Image:
    """Binarize the granular ISUP ≥ 2 annotations"""
    lbl_arrtemp = [sitk.GetArrayFromImage(i) for i in lbl]

    # convert granular PI-CAI csPCa annotation to binary csPCa annotation
    lbl_arr = [(arr >= 1).astype('uint8') for arr in lbl_arrtemp]

    # Convert binary arrays back to SimpleITK images
    lbl_new = [sitk.GetImageFromArray(arr) for arr in lbl_arr]

    # Copy information from the original SimpleITK images
    for i in range(len(lbl)):
        lbl_new[i].CopyInformation(lbl[i])


    return lbl_new



def atomic_image_write(
    image: sitk.Image,
    path: PathLike,
    backup_existing_file: bool = False,
    compress: bool = True,
    mkdir: bool = False
):
    """
    Safely write image to disk, by:
    1. Writing the image to a temporary file: path/to/tmp_[filename]
    2. IF writing succeeded:
    2a. (optional) rename existing file to path/to/backup_[filename]
    2b. rename file to target name, which is an atomic operation
    This way, no partially written files can exist at the target path (except for temporary files)
    """
    path = Path(path)

    if mkdir:
        os.makedirs(path.parent, exist_ok=True)

    # save image to temporary file
    path_tmp = path.with_name(f"tmp_{path.name}")
    sitk.WriteImage(image, path_tmp.as_posix(), useCompression=compress)

    # backup existing file?
    if backup_existing_file and path.exists():
        dst_path_bak = path.with_name(f"backup_{path.name}")
        if dst_path_bak.exists():
            raise FileExistsError(f"Existing backup file found at {dst_path_bak}.")
        os.replace(path, dst_path_bak)

    # rename temporary file
    os.replace(path_tmp, path)


def atomic_file_copy(
    src_path: PathLike,
    dst_path: PathLike,
    backup_existing_file: bool = False,
    mkdir: bool = False
):
    """
    Safely copy file, by:
    1. Writing the file to a temporary file: path/to/.tmp.[filename]
    2. IF writing succeeded:
    2a. (optional) backup existing file to path/to/backup.[filename]
    2b. rename file to target name, which is an atomic operation
    This way, no partially written files can exist at the target path (except for temporary files)
    """
    dst_path = Path(dst_path)

    if mkdir:
        os.makedirs(dst_path.parent, exist_ok=True)

    # copy file to temporary location
    dst_path_tmp = dst_path.with_name(f".tmp.{dst_path.name}")
    try:
        shutil.copy(src_path, dst_path_tmp)
    except PermissionError:
        shutil.copyfile(src_path, dst_path_tmp)

    # backup existing file?
    if backup_existing_file and dst_path.exists():
        dst_path_bak = dst_path.with_name(f"backup.{dst_path.name}")
        if dst_path_bak.exists():
            raise FileExistsError(f"Existing backup file found at {dst_path_bak}.")
        os.replace(dst_path, dst_path_bak)

    # rename temporary file
    os.replace(dst_path_tmp, dst_path)


# paths
path_dir = Path("/home/gustavo/Data/dataset/ProstateDATA/prostate158_train/Data/")
output_dirimages = Path('/home/gustavo/Data/dataset/picai/dataset_test/prostate158-new/images')
annotations_out_dir= Path('/home/gustavo/Data/dataset/picai/dataset_test/prostate158-new/labels')
annotations_out_dir_WG= Path('/home/gustavo/Data/dataset/picai/dataset_test/prostate158-new/labels-WG')

if __name__ == "__main__":
    
    healthy=[]
    for patient_id in tqdm(sorted(os.listdir(path_dir))):
        annotation_path=[]
        crop_path=[]
        verified_scan_paths=[Path(os.path.join(path_dir,patient_id,'t2.nii.gz')),
                             Path(os.path.join(path_dir,patient_id,'adc.nii.gz')),
                             Path(os.path.join(path_dir,patient_id,'dwi.nii.gz'))]
        
        if 'empty.nii.gz' in os.listdir(os.path.join(path_dir,patient_id)):
            annotation_path=[Path(os.path.join(path_dir,patient_id,'empty.nii.gz')),#reader1 adc
                             Path(os.path.join(path_dir,patient_id,'empty.nii.gz')),#reader1 t2
                             Path(os.path.join(path_dir,patient_id,'empty.nii.gz')),#reader2 adc
                             ]
            healthy.append(str(20000+int(patient_id))+'_'+str(20000+int(patient_id)))
        else:
            for labels in os.listdir(os.path.join(path_dir,patient_id)):
                if 'tumor' in labels and ('reader1' in labels or 'reader2' in labels):
                    annotation_path.append(Path(os.path.join(path_dir,patient_id,labels)))
                else:
                    continue
        for labels in os.listdir(os.path.join(path_dir,patient_id)):
            if 'anatomy' in labels and ('reader1' in labels or 'reader2' in labels):
                crop_path.append(Path(os.path.join(path_dir,patient_id,labels)))
            else:
                continue
        
        scans = [sitk.ReadImage(path.as_posix()) for path in verified_scan_paths]
        lbl = [sitk.ReadImage(path.as_posix()) for path in annotation_path]
        prostate = [sitk.ReadImage(path.as_posix()) for path in crop_path] #load region used to crop (prostate)

        print(annotation_path)
        print(crop_path)
        # set up Sample
        subject_id=str(20000+int(patient_id))+'_'+str(20000+int(patient_id))
        sample = Sample(
            scans=scans,
            lbl=lbl,
            prostate=prostate,
            name=subject_id,
            lbl_preprocess_func=preprocess_picai_annotation
        )

        # perform preprocessing
        sample.preprocess()

        # write images
        for i, scan in enumerate(sample.scans):
            destination_path = output_dirimages / f"{subject_id}_{i:04d}.nii.gz"
            atomic_image_write(scan, path=destination_path, mkdir=True)

        for i, lbl in enumerate(sample.lbl):
            atomic_image_write(lbl, path=annotations_out_dir / f"{i}" / f"{subject_id}.nii.gz", mkdir=True)

        #if len(sample.lbl_WG)>1:
            #flat_list = [item for sublist in sample.lbl_WG for item in sublist]
            #for i, lbl in enumerate(flat_list[0]):# take only first
        #    print(len(sample.lbl_WG))
        #    atomic_image_write(sample.lbl_WG[0][0], path=annotations_out_dir_WG / f"{0}" / f"{subject_id}.nii.gz", mkdir=True)        
        #else:
        for i, lbl in enumerate(sample.lbl_WG):
            atomic_image_write(lbl, path=annotations_out_dir_WG / f"{i}" / f"{subject_id}.nii.gz", mkdir=True)
