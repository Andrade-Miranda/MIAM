from data.base_dataset import BaseDataset
import numpy as np

from monai.data import ThreadDataLoader
#from monai.utils import set_determinism
#import torch
from monai.transforms import (
#    Activations,
#    Activationsd,
#    AsDiscrete,
#    AsDiscreted,
    Compose,
    ToDeviced,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    RandRotate90d,
    EnsureTyped,
    CropForegroundd,
    ToTensord
)


from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
#    decollate_batch,
)



"""--------------------------------------------------------------------"""
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 2 the peritumoral edematous/invaded tissue
    label 4 is the GD-enhancing tumor
    label 1 is the necrotic tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 4 and label 1 to construct TC
            result.append(np.logical_or(d[key] == 4, d[key] == 1))
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 1, d[key] == 4), d[key] == 2
                )
            )
            # label 4 is ET
            result.append(d[key] == 4)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
"""----------------------------------------------------------"""




class Brats2021Dataset(BaseDataset):
    def initialize(self, opt):
        """----------------------------------------------------------------------------"""
        """ data transfomation either for training and for validation"""
        self.opt=opt
        
        self.train_transform = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                    ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                EnsureTyped(keys=["image", "label"]),
                CenterSpatialCropd(keys=["image", "label"], roi_size=[160, 160, 160]),
                RandSpatialCropd(keys=["image", "label"], roi_size=self.opt.imageSize, random_size=False),
                #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                #RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                
                ]
            )
        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys="image"),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.0, 1.0, 1.0),
                    mode=("bilinear", "nearest"),
                    ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                EnsureTyped(keys=["image", "label"]),
                #ToDeviced(keys=["image", "label"], device=opt.device)
            ]
            )
        """----------------------------------------------------------------------------"""
        """ data Loading JSON file """
        datasets = self.opt.dataroot
        self.train_files = load_decathlon_datalist(datasets, True, "training")
        self.val_files = load_decathlon_datalist(datasets, True, "validation")
        
        """------------------------------------------------------------"""

    def LoadData(self):
 
        """ LOAD TRAINING DATA"""
        train_ds = CacheDataset(
        data=self.train_files,
        transform=self.train_transform,
        cache_rate=1.0,
        cache_num=24,
        num_workers=6,
        )
        self.train_loader = DataLoader(
        train_ds, batch_size=self.opt.batchSize, shuffle=True, num_workers=6, pin_memory=True
        )

        """ LOAD VALIDATION DATA"""
        val_ds = CacheDataset(
        data=self.val_files, 
        transform=self.val_transform, 
        cache_rate=1.0,
        cache_num=6, 
        num_workers=4, 
        )
        self.val_loader = DataLoader(
        val_ds, batch_size=self.opt.Val_batchSize, num_workers=4
        )
        """---------------------------------------------------"""
        return self.train_loader, self.val_loader

    def __len__(self):
        return len(self.train_loader)+len(self.val_loader)

    def name(self):
        return 'BratsDataset2021'




# import matplotlib.pyplot as plt
# plt.figure("image", (24, 6))
# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     plt.title(f"image channel {i}")
#     plt.imshow(val_ds[0]["image"][i, :, :, 60], cmap="gray")
# plt.show()
# # also visualize the 3 channels label corresponding to this image
# plt.figure("label", (18, 6))
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.title(f"label channel {i}")
#     plt.imshow(val_ds[0]["label"][i, :, :, 60])
