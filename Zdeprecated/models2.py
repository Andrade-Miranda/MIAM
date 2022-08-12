def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == '4CNN_1T':
        from .CNN4_1T import Multipath1TModel    
        model = Multipath1TModel(opt)
    elif opt.model == '1CNN_1T':
        from .CNN1_1T import Early1TModel
        model = Early1TModel(opt)
    elif opt.model == '4CNN_4T':
        from .CNN4_4T import Multipath4TModel
        model = Multipath4TModel(opt)
    elif opt.model == '4CNN_2T':
        from .CNN4_2T import Hierarchical2TModel
        model = Hierarchical2TModel(opt)
    elif opt.model == 'UNETR':
        from .netMisc import UNETRModel
        model = UNETRModel(opt)
    elif opt.model == '4TR-1T_CNN':#no implemented
        from .TR4_1TCNN import Multi4TransformerModel
        model = Multi4TransformerModel(opt)
    elif opt.model == 'SegResNet':
        from .netMisc import SegResNetModel
        model = SegResNetModel(opt)
    elif opt.model == 'Unet':
        from .netMisc import UnetMonai
        model = UnetMonai(opt)
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    print("model [%s] was created" % (model.name()))
    return model
