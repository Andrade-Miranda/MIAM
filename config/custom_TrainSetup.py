from config.base_train_setup import BaseTrainSetup


def CreateTrainConfig(opt,model):
    Config = None
    if opt.TrainConfig == 'BaseConfig':#for training setup
        from config.Base_Config import BaseConfig
        Config = BaseConfig()
    elif opt.TrainConfig == 'DefaultConfig':#####################default  segmentation training setup
        from config.Default_Config import DefaultConfig
        Config = DefaultConfig()
    elif opt.TrainConfig == 'PICAIConfig':##############################configuration for Priors mask
        from config.PICAI_Config import PICAIConfig
        Config = PICAIConfig()
    elif opt.TrainConfig == 'nnUNetConfig': ###################################nnUNEt setup by default using same parameters and optimizer
        from config.nnUNet_Config import nnUNetConfig
        Config = nnUNetConfig() 
    elif opt.TrainConfig == 'TestConfig':#####################################for testing setup
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
