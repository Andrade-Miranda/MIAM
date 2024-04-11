import models.Priors.networks as arch
import models.Priors as modelArch

def create_model(opt):
    model = None
    encoderOption=['UNETR_MAE','UNETR']
        
    assert any(opt.encoder==name for name in encoderOption)
    if opt.encoder=='UNETR_MAE':#si
        from .Priors.unetr3d import UNETR3D
        opt.enc_arch=getattr(arch, opt.enc_arch)
        opt.dec_arch=getattr(arch, opt.dec_arch)
        model = UNETR3D(opt)
        model.name()
    if opt.encoder == 'UNETR':
        from .netMisc import UNETRModel
        model = UNETRModel(opt)
        print("model [%s] was created" % (model.name()))
    
    return model.init_net(model)
