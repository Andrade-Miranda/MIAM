from data.base_dataset import BaseDataset
import numpy as np

from monai.data import partition_dataset
#from monai.utils import set_determinism
#import torch
from monai.transforms import (
#    Activations,
#    Activationsd,
#    AsDiscrete,
#    AsDiscreted,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
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
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d
"""----------------------------------------------------------"""




class BratsDataset(BaseDataset):
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
                #RandSpatialCropd(keys=["image", "label"], roi_size=self.opt.imageSize, random_size=False),
                CenterSpatialCropd(keys=["image", "label"], roi_size=self.opt.imageSize),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
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
            ]
            )
        """----------------------------------------------------------------------------"""

    def LoadData(self):
        """ data Loading JSON file """
        datasets = self.opt.dataroot
        train_files = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")
        """------------------------------------------------------------"""
        
        # """ data partition training and validation no cross validation for the moment """
        # data_train, data_val = partition_dataset(
        # data=train_files,
        # ratios=[0.8, 0.2],
        # shuffle=True,
        # )
        # """---------------------------------------------------"""
        
        """ LOAD TRAINING DATA"""
        train_ds = CacheDataset(
        data=train_files,
        transform=self.train_transform,
        cache_num=24,
        cache_rate=1.0,
        num_workers=6,
        )
        self.train_loader = DataLoader(
        train_ds, batch_size=self.opt.batchSize, shuffle=True, num_workers=6, pin_memory=True
        )


        """ LOAD VALIDATION DATA"""
        val_ds = CacheDataset(
        data=val_files, transform=self.val_transform, cache_num=6, cache_rate=1.0, num_workers=4
        )
        self.val_loader = DataLoader(
        val_ds, batch_size=self.opt.batchSize, shuffle=False, num_workers=4, pin_memory=True
        )
        """---------------------------------------------------"""
        return self.train_loader, self.val_loader

    def __len__(self):
        return (len(self.train_loader),len(self.val_loader))

    def name(self):
        return 'BratsDataset'


