

import json
from tqdm import tqdm
import os
from pathlib import Path

import SimpleITK as sitk



def generate_settings(
    archive_dir: Path,
    output_path: Path,
    annotations_dir: Path
    ):
    """
    Create mha2nnunet_settings.json (for inference) for an MHA archive with the following structure:
    /path/to/archive/
    ├── [patient UID]/
        ├── [patient UID]_[study UID]_[modality].mha
        ...

    Parameters:
    - archive_dir: path to MHA archive
    - output_path: path to store MHA -> nnUNet settings JSON to
        (parent folder should exist)
    """
    archive_list = []

    # traverse MHA archive
    for patient_id in tqdm(sorted(os.listdir(archive_dir))):
        # traverse each patient's studies
        patient_dir = os.path.join(archive_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # collect list of available studies
        files = os.listdir(patient_dir)
        files = [fn.replace(".mha", "") for fn in files if ".mha" in fn and "._" not in fn]
        subject_ids = ["_".join(fn.split("_")[0:2]) for fn in files]
        subject_ids = sorted(list(set(subject_ids)))

        # check which studies are complete
        for subject_id in subject_ids:
            patient_id, study_id = subject_id.split("_")

            # construct scan paths
            scan_paths = [
                f"{patient_id}/{subject_id}_{modality}.mha"
                for modality in ["t2w", "adc", "hbv"]
            ]
            all_scans_found = all([
                os.path.exists(os.path.join(archive_dir, path))
                for path in scan_paths
            ])

            # construct annotation path
            annotation_path = f"{subject_id}.nii.gz"

            if annotations_dir is not None:
                # check if annotation exists
                if not os.path.exists(os.path.join(annotations_dir, annotation_path)):
                    # could not find annotation, skip case
                    continue

            if all_scans_found:
                # store info for complete studies
                archive_list += [{
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "scan_paths": scan_paths,
                    "annotation_path": annotation_path,
                }]

    mha2nnunet_settings = {
        "dataset_json": {
            "description": "bpMRI scans from PI-CAI dataset to train nnUNet baseline",
            "tensorImageSize": "4D",
            "reference": "",
            "licence": "",
            "release": "1.0",
            "modality": {
                "0": "T2W",
                "1": "ADC",
                "2": "HBV"
            },
            "labels": {
                "0": "background",
                "1": "prostate"
            }
        },
        "archive": archive_list
    }

    if not len(archive_list):
        raise ValueError(f"Did not find any MHA scans in {archive_dir}, aborting.")

    with open(output_path, "w") as fp:
        json.dump(mha2nnunet_settings, fp, indent=4)

    print(f""""
    Saved mha2nnunet_settings to {output_path}, with {len(archive_list)} cases.
    """)
    return archive_list


def CroppingVolumes(annotations_dir,archive_dir,output_dir,archive_list):

    for scan_patient in archive_list:
        mask = sitk.ReadImage(os.path.join(annotations_dir,scan_patient['annotation_path']))
        T2W=sitk.ReadImage(os.path.join(archive_dir,scan_patient['scan_paths'][0]))
        ADC=sitk.ReadImage(os.path.join(archive_dir,scan_patient['scan_paths'][1]))
        HBV=sitk.ReadImage(os.path.join(archive_dir,scan_patient['scan_paths'][2]))

        inside_value = 0
        outside_value = 1
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(mask)
        xstart, ystart, zstart, xsize, ysize, zsize = label_shape_filter.GetBoundingBox(outside_value)#[xstart, ystart, zstart, xsize, ysize, zsize]
        print('Bounding box:' + str((xstart, ystart, zstart, xsize, ysize, zsize)))
        print('Before cropping mask:')
        print('origin: ' + str(mask.GetOrigin()))
        print('size: ' + str(mask.GetSize()))
        print('spacing: ' + str(mask.GetSpacing()))
        print('direction: ' + str(mask.GetDirection()))
        print('pixel type: ' + str(mask.GetPixelIDTypeAsString()))
        print('number of pixel components: ' + str(mask.GetNumberOfComponentsPerPixel()))

        mask=mask[xstart-30:(xsize+xstart+30),ystart-30:(ysize+ystart+30),zstart-2:(zsize+zstart+2)]
        # save image 
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(output_dir,'labels',scan_patient['annotation_path']))
        writer.Execute(mask)

        print('\nAfter cropping mask:')
        print('origin: ' + str(mask.GetOrigin()))
        print('spacing: ' + str(mask.GetSpacing()))
        print('size: ' + str(mask.GetSize()))
        print('direction: ' + str(mask.GetDirection()))
        print('pixel type: ' + str(mask.GetPixelIDTypeAsString()))
        print('number of pixel components: ' + str(mask.GetNumberOfComponentsPerPixel()))

        print('\nBefore cropping scan:')
        print('origin: ' + str(T2W.GetOrigin()))
        print('size: ' + str(T2W.GetSize()))
        print('spacing: ' + str(T2W.GetSpacing()))
        print('direction: ' + str(T2W.GetDirection()))
        print('pixel type: ' + str(T2W.GetPixelIDTypeAsString()))
        print('number of pixel components: ' + str(T2W.GetNumberOfComponentsPerPixel()))

        T2W=T2W[xstart-30:(xsize+xstart+30),ystart-30:(ysize+ystart+30),zstart-2:(zsize+zstart+2)]
        ADC=ADC[xstart-30:(xsize+xstart+30),ystart-30:(ysize+ystart+30),zstart-2:(zsize+zstart+2)]
        HBV=HBV[xstart-30:(xsize+xstart+30),ystart-30:(ysize+ystart+30),zstart-2:(zsize+zstart+2)]

        # save image
        os.makedirs(Path(os.path.join(output_dir,'images',scan_patient['scan_paths'][0])).parent, exist_ok=True) 
        writer0 = sitk.ImageFileWriter()
        writer0.SetFileName(os.path.join(output_dir,'images',scan_patient['scan_paths'][0]))
        writer0.Execute(T2W)

        writer1 = sitk.ImageFileWriter()
        writer1.SetFileName(os.path.join(output_dir,'images',scan_patient['scan_paths'][1]))
        writer1.Execute(ADC)

        writer2 = sitk.ImageFileWriter()
        writer2.SetFileName(os.path.join(output_dir,'images',scan_patient['scan_paths'][2]))
        writer2.Execute(HBV)

        print('\nAfter cropping scan:')
        print('origin: ' + str(T2W.GetOrigin()))
        print('spacing: ' + str(T2W.GetSpacing()))
        print('size: ' + str(T2W.GetSize()))
        print('direction: ' + str(T2W.GetDirection()))
        print('pixel type: ' + str(T2W.GetPixelIDTypeAsString()))
        print('number of pixel components: ' + str(T2W.GetNumberOfComponentsPerPixel()))







if __name__ == "__main__":

    # paths
    annotations_dir = Path("/home/gustavo/Data/dataset/picai/picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b")
    output_json =Path("/home/gustavo/Data/dataset/picai/Cropped/datalist.json")
    archive_dir = Path('/home/gustavo/Data/dataset/picai/images')
    output_dir = Path('/home/gustavo/Data/dataset/picai/Cropped')

    archive_list=generate_settings(archive_dir,output_json,annotations_dir)
    CroppingVolumes(annotations_dir,archive_dir,output_dir,archive_list)
    