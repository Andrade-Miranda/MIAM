#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import SimpleITK as sitk
from numpy.testing import assert_allclose
from scipy import ndimage

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass


@dataclass
class PreprocessingSettings():
    """
    Preprocessing settings
    - matrix_size: number of voxels output image (z, y, x)
    - spacing: output voxel spacing in mm/voxel (z, y, x)
    - physical_size: size in mm of the target image (z, y, x)
    - crop_only: only crop to specified size (i.e., do not pad)
    - align_segmentation: whether to align the scans using the centroid of the provided segmentation
    """
    matrix_size: Optional[Iterable[int]] = None
    spacing: Optional[Iterable[float]] = None
    physical_size: Optional[Iterable[float]] = None
    crop_only: bool = False
    align_segmentation: Optional[sitk.Image] = None

    def __post_init__(self):
        if self.physical_size is None and self.spacing is not None and self.matrix_size is not None:
            # calculate physical size
            self.physical_size = [
                voxel_spacing * num_voxels
                for voxel_spacing, num_voxels in zip(
                    self.spacing,
                    self.matrix_size
                )
            ]

        if self.spacing is None and self.physical_size is not None and self.matrix_size is not None:
            # calculate spacing
            self.spacing = [
                size / num_voxels
                for size, num_voxels in zip(
                    self.physical_size,
                    self.matrix_size
                )
            ]

        if self.align_segmentation is not None:
            raise NotImplementedError("Alignment of scans based on segmentation is not implemented yet.")


def resample_img(
    image: sitk.Image,
    out_spacing: Iterable[float] = (2.0, 2.0, 2.0),
    out_size: Optional[Iterable[int]] = None,
    is_label: bool = False,
    pad_value: Optional[Union[float, int]] = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image


def input_verification_crop_or_pad(
    image: "Union[sitk.Image, npt.NDArray[Any]]",
    size: Optional[Iterable[int]] = (20, 256, 256),
    physical_size: Optional[Iterable[float]] = None,
) -> Tuple[Iterable[int], Iterable[int]]:
    """
    Calculate target size for cropping and/or padding input image

    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)
    - physical_size: target size in mm (z, y, x)

    Either size or physical_size must be provided.

    Returns:
    - shape of original image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    - size of target image (in convention of SimpleITK (x, y, z) or numpy (z, y, x))
    """
    # input conversion and verification
    if physical_size is not None:
        # convert physical size to voxel size (only supported for SimpleITK)
        if not isinstance(image, sitk.Image):
            raise ValueError("Crop/padding by physical size is only supported for SimpleITK images.")
        spacing_zyx = list(image.GetSpacing())[::-1]
        size_zyx = [length/spacing for length, spacing in zip(physical_size, spacing_zyx)]
        size_zyx = [int(np.round(x)) for x in size_zyx]

        if size is None:
            # use physical size
            size = size_zyx
        else:
            # verify size
            if list(size) != list(size_zyx):
                raise ValueError(f"Size and physical size do not match. Size: {size}, physical size: "
                                 f"{physical_size}, spacing: {spacing_zyx}, size_zyx: {size_zyx}.")

    if isinstance(image, sitk.Image):
        # determine shape and convert convention of (z, y, x) to (x, y, z) for SimpleITK
        shape = image.GetSize()
        size = list(size)[::-1]
    else:
        # determine shape for numpy array
        assert isinstance(image, (np.ndarray, np.generic))
        shape = image.shape
        size = list(size)
    rank = len(size)
    assert rank <= len(shape) <= rank + 1, \
        f"Example size doesn't fit image size. Got shape={shape}, output size={size}"

    return shape, size


def crop_or_pad(
    image: "Union[sitk.Image, npt.NDArray[Any]]",
    size: Optional[Iterable[int]] = (20, 256, 256),
    physical_size: Optional[Iterable[float]] = None,
    crop_only: bool = False,
) -> "Union[sitk.Image, npt.NDArray[Any]]":
    """
    Resize image by cropping and/or padding

    Parameters:
    - image: image to be resized (sitk.Image or numpy.ndarray)
    - size: target size in voxels (z, y, x)
    - physical_size: target size in mm (z, y, x)

    Either size or physical_size must be provided.

    Returns:
    - resized image (same type as input)
    """
    # input conversion and verification
    shape, size = input_verification_crop_or_pad(image, size, physical_size)

    # set identity operations for cropping and padding
    rank = len(size)
    padding = [[0, 0] for _ in range(rank)]
    slicer = [slice(None) for _ in range(rank)]

    # for each dimension, determine process (cropping or padding)
    for i in range(rank):
        if shape[i] < size[i]:
            if crop_only:
                continue

            # set padding settings
            padding[i][0] = (size[i] - shape[i]) // 2
            padding[i][1] = size[i] - shape[i] - padding[i][0]
        else:
            # create slicer object to crop image
            idx_start = int(np.floor((shape[i] - size[i]) / 2.))
            idx_end = idx_start + size[i]
            slicer[i] = slice(idx_start, idx_end)

    # crop and/or pad image
    if isinstance(image, sitk.Image):
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound([pad[0] for pad in padding])
        pad_filter.SetPadUpperBound([pad[1] for pad in padding])
        return pad_filter.Execute(image[tuple(slicer)])
    else:
        return np.pad(image[tuple(slicer)], padding)


@dataclass
class Sample:
    scans: List[sitk.Image]
    lbl: Optional[sitk.Image] = None
    prostate: Optional[sitk.Image] = None
    name: Optional[str] = None
    settings: PreprocessingSettings = None
    lbl_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    lbl_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_preprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    scan_postprocess_func: Optional[Callable[[sitk.Image], sitk.Image]] = None
    num_gt_lesions: Optional[int] = None

    def __post_init__(self):
        self.num_gt_lesions=[]
        if self.lbl is not None:
            # keep track of connected components
            for label in self.lbl:
                lbl = sitk.GetArrayFromImage(label)
                _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
                self.num_gt_lesions.append(num_gt_lesions)

        if self.settings is None:
            self.settings = PreprocessingSettings()

    def createBoundingBox(self):
        
        inside_value = 0
        outside_value = 1
        
         # convert prostate + tumor in one label (1)
        
        lbl_arr = sitk.GetArrayFromImage(self.prostate[0])#voy a tomar solo la primera segmentaticon de la lista
        lbl_arr = (lbl_arr >= 1).astype('uint8')
        lbl_new: sitk.Image = sitk.GetImageFromArray(lbl_arr)
        lbl_new.CopyInformation(self.prostate[0])
        
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(lbl_new)
        xstart, ystart, zstart, xsize, ysize, zsize = label_shape_filter.GetBoundingBox(outside_value)#[xstart, ystart, zstart, xsize, ysize, zsize]

        #validate limits
        if xstart-30<0:
            xinit=1
        else:
            xinit=xstart-30

        if xsize+xstart+30 > self.lbl[0].GetSize()[0]:
            xfin=self.lbl[0].GetSize()[0]
        else:
            xfin=xsize+xstart+30
        ##y
        if ystart-30<0:
            yinit=1
        else:
            yinit=ystart-30

        if ysize+ystart+30 > self.lbl[0].GetSize()[1]:
            yfin=self.lbl[0].GetSize()[1]
        else:
            yfin=ysize+ystart+30

        ##z
        if zstart-2<0:
            zinit=1
        else:
            zinit=zstart-2

        if zsize+zstart+2 > self.lbl[0].GetSize()[2]:
            zfin=self.lbl[0].GetSize()[2]
        else:
            zfin=zsize+zstart+2

        for i in range(len(self.lbl)):

            print("Case: "+ self.name)
            print('Bounding box:' + str((xinit, yinit, zinit, xfin, yfin, zfin)))
            print('Label Before cropping mask:')
            print('origin: ' + str(self.lbl[i].GetOrigin()))
            print('size: ' + str(self.lbl[i].GetSize()))
            print('spacing: ' + str(self.lbl[i].GetSpacing()))
            print('direction: ' + str(self.lbl[i].GetDirection()))
            print('pixel type: ' + str(self.lbl[i].GetPixelIDTypeAsString()))
            print('number of pixel components: ' + str(self.lbl[i].GetNumberOfComponentsPerPixel()))

            self.lbl[i]=self.lbl[i][xinit:xfin,yinit:yfin,zinit:zfin]

            print('\nLabel After cropping mask:')
            print('origin: ' + str(self.lbl[i].GetOrigin()))
            print('spacing: ' + str(self.lbl[i].GetSpacing()))
            print('size: ' + str(self.lbl[i].GetSize()))
            print('direction: ' + str(self.lbl[i].GetDirection()))
            print('pixel type: ' + str(self.lbl[i].GetPixelIDTypeAsString()))
            print('number of pixel components: ' + str(self.lbl[0].GetNumberOfComponentsPerPixel()))

        print('\n\nScans Before cropping scan:')
        print('origin: ' + str(self.scans[0].GetOrigin()))
        print('size: ' + str(self.scans[0].GetSize()))
        print('spacing: ' + str(self.scans[0].GetSpacing()))
        print('direction: ' + str(self.scans[0].GetDirection()))
        print('pixel type: ' + str(self.scans[0].GetPixelIDTypeAsString()))
        print('number of pixel components: ' + str(self.scans[0].GetNumberOfComponentsPerPixel()))

        self.scans = [scan[xinit:xfin,yinit:yfin,zinit:zfin] for scan in self.scans]

        print('\nScans After cropping scan:')
        print('origin: ' + str(self.scans[0].GetOrigin()))
        print('spacing: ' + str(self.scans[0].GetSpacing()))
        print('size: ' + str(self.scans[0].GetSize()))
        print('direction: ' + str(self.scans[0].GetDirection()))
        print('pixel type: ' + str(self.scans[0].GetPixelIDTypeAsString()))
        print('number of pixel components: ' + str(self.scans[0].GetNumberOfComponentsPerPixel()))

        for i in range(len(self.prostate)):
            self.prostate[i]=self.prostate[i][xinit:xfin,yinit:yfin,zinit:zfin]
        self.lbl_WG=[]
        #for prostate in self.prostate:
        lbl_arr = sitk.GetArrayFromImage(self.prostate[0])#voy a tomar solo la primera segmentaticon de la lista
        for label in self.lbl: 
            lbl_prost = (lbl_arr >= 1).astype('uint8')
            lbl_tumor = sitk.GetArrayFromImage(label)
            lbl_tumor = (lbl_tumor >= 1).astype('uint8')*2
            final_arr= lbl_prost + lbl_tumor
            final_arr[final_arr==3]=2
            lbl_WG: sitk.Image = sitk.GetImageFromArray(final_arr)
            lbl_WG.CopyInformation(label)
            self.lbl_WG.append(lbl_WG)

    def resample_to_first_scan(self):
        """Resample scans and label to the first scan"""
        # set up resampler to resolution, field of view, etc. of first scan
        resampler = sitk.ResampleImageFilter()  # default linear
        resampler.SetReferenceImage(self.scans[0])
        resampler.SetInterpolator(sitk.sitkBSpline)

        # resample other images
        self.scans[1:] = [resampler.Execute(scan) for scan in self.scans[1:]]

        # resample annotation
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        if self.lbl is not None:
            for i in range(len(self.lbl)):
                self.lbl[i] = resampler.Execute(self.lbl[i])

        # resample annotation WG
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        if self.prostate is not None:
            for i in range(len(self.prostate)):
                self.prostate[i] = resampler.Execute(self.prostate[i])

    def resample_spacing(self, spacing: Optional[Iterable[float]] = None):
        """Resample scans and label to the target spacing"""
        if spacing is None:
            assert self.settings.spacing is not None
            spacing = self.settings.spacing

        # resample scans to target resolution
        self.scans = [
            resample_img(scan, out_spacing=spacing, is_label=False)
            for scan in self.scans
        ]

        # resample annotation to target resolution
        if self.lbl is not None:
            self.lbl = resample_img(self.lbl, out_spacing=spacing, is_label=True)

    def centre_crop_or_pad(self):
        """Centre crop and/or pad scans and label"""
        kwargs = {
            "size": self.settings.matrix_size,
            "physical_size": self.settings.physical_size,
            "crop_only": self.settings.crop_only,
        }
        self.scans = [
            crop_or_pad(scan, **kwargs)
            for scan in self.scans
        ]

        if self.lbl is not None:
            self.lbl = crop_or_pad(self.lbl, **kwargs)

    def align_physical_metadata(self, check_almost_equal=True):
        """Align the origin and direction of each scan, and label"""
        case_origin, case_direction, case_spacing = None, None, None
        for img in self.scans:
            # copy metadata of first scan (nnUNet and nnDetection require this to match exactly)
            if case_origin is None:
                case_origin = img.GetOrigin()
                case_direction = img.GetDirection()
                case_spacing = img.GetSpacing()
            else:
                if check_almost_equal:
                    # check if current scan's metadata is almost equal to the first scan
                    assert_allclose(img.GetOrigin(), case_origin)
                    assert_allclose(img.GetDirection(), case_direction)
                    assert_allclose(img.GetSpacing(), case_spacing)

                # copy over first scan's metadata to current scan
                img.SetOrigin(case_origin)
                img.SetDirection(case_direction)
                img.SetSpacing(case_spacing)

        if self.lbl is not None:
            assert case_origin is not None and case_direction is not None and case_spacing is not None
            for i in range(len(self.lbl)):
                self.lbl[i].SetOrigin(case_origin)
                self.lbl[i].SetDirection(case_direction)
                self.lbl[i].SetSpacing(case_spacing)

    def preprocess(self):
        """Perform all preprocessing steps"""
        # user-defined preprocessing steps
        if self.lbl is not None and self.lbl_preprocess_func:
            # apply label transformation
            self.lbl = self.lbl_preprocess_func(self.lbl)
        if self.scan_preprocess_func:
            # apply scan transformation
            self.scans = [self.scan_preprocess_func(scan) for scan in self.scans]

        if self.settings.spacing is not None:
            # resample scans and label to specified spacing
            self.resample_spacing()

        if self.settings.matrix_size is not None or self.settings.physical_size is not None:
            # perform centre crop and/or pad
            self.centre_crop_or_pad()

        # resample scans and label to first scan's spacing, field-of-view, etc.
        self.resample_to_first_scan()

        # copy physical metadata to align subvoxel differences between sequences
        self.align_physical_metadata()

        self.createBoundingBox()

        if self.lbl is not None:
            # check connected components of annotation
            for i in range(len(self.lbl)):
                lbl = sitk.GetArrayFromImage(self.lbl[i])
                _, num_gt_lesions = ndimage.label(lbl, structure=np.ones((3, 3, 3)))
                #assert self.num_gt_lesions == num_gt_lesions, \
                #f"Label has changed due to resampling/other errors for {self.name}! " \
                #+ f"Have {self.num_gt_lesions[i]} -> {num_gt_lesions} isolated ground truth lesions"

        # user-defined postprocessing steps
        if self.lbl is not None and self.lbl_postprocess_func:
            # apply label transformation
            self.lbl = self.lbl_postprocess_func(self.lbl)
        if self.scan_postprocess_func:
            # apply scan transformation
            self.scans = [self.scan_postprocess_func(scan) for scan in self.scans]


def resample_to_reference_scan(
    image: "Union[npt.NDArray[Any], sitk.Image]",
    reference_scan_original: sitk.Image,
    reference_scan_preprocessed: Optional[sitk.Image] = None,
    interpolator: Optional[sitk.ResampleImageFilter] = None,
) -> sitk.Image:
    """
    Translate image/prediction/annotation to physical space of original scan (e.g., T2-weighted scan)

    Parameters:
    - image: scan, detection map or (softmax) prediction
    - reference_scan_original: SimpleITK image to which the prediction should be resampled and resized
    - reference_scan_preprocessed: (Optional) SimpleITK image with physical metadata for `image`
        (e.g., scan in nnUNet Raw Data Archive). If not provided, `image` should be a SimpleITK image.

    Returns:
    - resampled image, in same physical space as reference_scan_original
    """
    # convert image to SimpleITK image and copy physical metadata
    if not isinstance(image, sitk.Image):
        if reference_scan_preprocessed is None:
            raise ValueError("Need SimpleITK Image or reference scan for phsyical metadata!")
        image: sitk.Image = sitk.GetImageFromArray(image)

    if reference_scan_preprocessed is not None:
        image.CopyInformation(reference_scan_preprocessed)

    if interpolator is None:
        # determine interpolation method based on image dtype
        dtype_name = image.GetPixelIDTypeAsString()
        if "integer" in dtype_name:
            interpolator = sitk.sitkNearestNeighbor
        elif "float" in dtype_name:
            interpolator = sitk.sitkLinear
        else:
            raise ValueError(f"Unknown pixel type {dtype_name}")

    # prepare resampling to original scan
    resampler = sitk.ResampleImageFilter()  # default linear
    resampler.SetReferenceImage(reference_scan_original)
    resampler.SetInterpolator(interpolator)

    # resample image to original scan
    image = resampler.Execute(image)

    return image
