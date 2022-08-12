from config.base_train_setup import BaseTrainSetup


def CreateTrainConfig(opt,model):
    Config = None
    if opt.TrainConfig == 'BaseConfig':#for training setup
        from config.Base_Config import BaseConfig
        Config = BaseConfig()
    elif opt.TrainConfig == 'TIMMConfig':#for training using TIMM library
        from config.TIMM_Config import TIMMConfig
        Config = TIMMConfig()
    elif opt.TrainConfig == 'BaseConfig+ContrasLearn':#for training using contrastive loss
        from config.BaseContrLrn_Config import BaseContrLrnConfig
        Config = BaseContrLrnConfig()
    elif opt.TrainConfig == 'BaseTrainConfig':#for test setup
        from config.BaseTrainConfig import BaseTrainConfig
        Config = BaseTrainConfig()
    elif opt.TrainConfig == 'TestConfig':#for test setup
        from config.Test_Config import TestConfig
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
