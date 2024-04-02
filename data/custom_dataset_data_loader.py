from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'Brats2021': #debe trabajar con json and monai, debe adaptarse para que trabaje en conjunto con NNUNET
        from data.Brats2021_dataset import Brats2021Dataset # dataloader para brats
        dataset = Brats2021Dataset()
    elif opt.dataset_mode == 'nnUNet': #dataloader based on nnUNet
        from data.nnUNet_dataset import nnUNetDataset
        dataset = nnUNetDataset()
    elif opt.dataset_mode == 'nnUNetWPriors': #dataloader based on nnUNet for include priors
        from data.nnUNetWPriors_dataset import nnUNetWPriorsDataset
        dataset = nnUNetWPriorsDataset()
    elif opt.dataset_mode == 'nnUNetAmos': #dataloader for nnUNet AMOS
        from data.nnUNetAmos_dataset import nnUNetAmosDataset
        dataset = nnUNetAmosDataset()
    elif opt.dataset_mode == 'nnUNetExtchan': #dataloader extra channel input
        from data.nnUNetExtchan_dataset import nnUNetExtchanDataset
        dataset = nnUNetExtchanDataset()
    elif opt.dataset_mode == 'test' or opt.dataset_mode=='MeanEnsemb' or opt.dataset_mode=='Nfold' or opt.dataset_mode=='TTA' or opt.dataset_mode=='MCdropOut':
        from data.nnUNet_datasetTest import nnUNetDatasetTest
        dataset = nnUNetDatasetTest()   
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    
    dataset.initialize(opt)
    print("dataset [%s] was created using [%s] dataloader" % (dataset.name(),opt.dataset_mode))
    
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        
    def load_data(self):
        self.train_loader,self.val_loader,self.test_loader=self.dataset.LoadData()
        self.datalen=self.dataset.lengthData()
        return self.train_loader,self.val_loader,self.test_loader,self.datalen
    
    def load_test(self):
        self.test_loader=self.dataset.LoadData()
        return self.test_loader

    def __len__(self):
        return self.dataset.__len__()
