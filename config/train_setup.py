
def TrainSetup(opt,model):
    from config.custom_TrainSetup import CustomTrainConfigLoader
    data_TrainSetup = CustomTrainConfigLoader()
    data_TrainSetup.initialize(opt,model)
    return data_TrainSetup

