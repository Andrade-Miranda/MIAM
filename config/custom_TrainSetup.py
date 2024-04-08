from config.base_train_setup import BaseTrainSetup


def CreateTrainConfig(opt,model):
    Config = None
    if opt.TrainConfig == 'BaseConfig':#for training setup
        from config.Base_Config import BaseConfig
        Config = BaseConfig()
    elif opt.TrainConfig == 'TIMMConfig':#for training using TIMM library
        from config.TIMM_Config import TIMMConfig
        Config = TIMMConfig()
    elif opt.TrainConfig == 'PRIORSConfig':#configuration for Priors mask
        from config.PRIORS_Config import PRIORSConfig
        Config = PRIORSConfig()
    elif opt.TrainConfig == 'AMOSConfig':# COnfig AMOS challenge
        from config.AMOS_Config import AMOSConfig
        Config = AMOSConfig()
    elif opt.TrainConfig == 'ContrastConfig':#for training using contrastive loss
        from config.Contrastive_Config import ContrastiveConfig
        Config = ContrastiveConfig()
    elif opt.TrainConfig == 'TransFuseConfig':#for methods based on parallel transformer and CNN
        from config.TransFuse_Config import TransFuseConfig
        Config = TransFuseConfig()
    elif opt.TrainConfig == 'nnUNetConfig': #nnUNEt setup
        from config.nnUNet_Config import nnUNetConfig
        Config = nnUNetConfig() 
    elif opt.TrainConfig == 'TestConfig':#for test setup
        from config.Test_Config import TestConfig
        Config = TestConfig()   
    elif opt.TrainConfig == 'Test_ConfigBrats':#for test setup
        from config.Test_ConfigBrats import TestConfig
        Config = TestConfig()  
    else:
        raise ValueError("Configuration [%s] not recognized." % opt.TrainConfig)

    print(f"Configuration [{Config.name()}] was created" )
    Config.initialize(opt,model)
    return Config


class CustomTrainConfigLoader(BaseTrainSetup):
    def name(self):
        return 'CustomTrainConfigLoader'

    def initialize(self, opt,model):
        BaseTrainSetup.initialize(self, opt,model)
        self.Config = CreateTrainConfig(opt,model)
        
    def load_data(self):
        self.train_loader,self.val_loader=self.Config.LoadData()
        return self.train_loader,self.val_loader

    def __len__(self):
        return len(self.train_loader)+len(self.val_loader)
