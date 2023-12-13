import SimpleITK as sitk
import numpy as np
import os
from pathlib import Path
from monai.transforms import FillHoles


def Picai_Postprocessing(input,prostateMask,postpro_dir):
    file=Path(os.path.join(postpro_dir,prostateMask.split('/')[-1]))
    #outside_value = 1
    #mask=np.zeros_like(input)
        
    # convert prostate + tumor in one label (1)
    original=sitk.ReadImage(file.as_posix())
    lbl_arr = sitk.GetArrayFromImage(original)#voy a tomar solo la primera segmentaticon de la lista
    lbl_arr = (lbl_arr >= 1).astype('uint8')
    #lbl_new: sitk.Image = sitk.GetImageFromArray(lbl_arr)
    #lbl_new.CopyInformation(original)
 
    #label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    #label_shape_filter.Execute(lbl_new)
    #xstart, ystart, zstart, xsize, ysize, zsize = label_shape_filter.GetBoundingBox(outside_value)#[xstart, ystart, zstart, xsize, ysize, zsize]

    #validate limits
    # if xstart<0:
    #     xinit=1
    # else:
    #     xinit=xstart

    # if xsize+xstart > input.shape[-1]:
    #     xfin=input.shape[-1]
    # else:
    #     xfin=xsize+xstart
    # ##y
    # if ystart<0:
    #     yinit=1
    # else:
    #     yinit=ystart

    # if ysize+ystart > input.shape[-2]:
    #     yfin=input.shape[-2]
    # else:
    #     yfin=ysize+ystart

    # ##z
    # if zstart<0:
    #     zinit=1
    # else:
    #     zinit=zstart

    # if zsize+zstart > input.shape[-3]:
    #     zfin=input.shape[-3]
    # else:
    #     zfin=zsize+zstart

    #mask[zinit:zfin,yinit:yfin,xinit:xfin]= 1  
    #fillHoles=FillHoles(applied_labels=1)
    return input*lbl_arr#fillHoles(input*lbl_arr)

  