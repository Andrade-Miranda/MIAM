def create_model(opt):
    model = None
    encoderOption=['CNN_h+VIT_n','CNN_l+VIT_n','MCNN_h+VIT_n','MCNN_h+VIT_s','MCNN_h+VIT_m','Unet','MCNN_l+VIT_s','MCNN_l+VIT_m',
                   'VIT_n','VIT_s','VIT_m','SegResNetVAE','SegResNet','UNETR','SwinTrans3DSimple','SwinTrans3D','nnFormer',
                   'SCNN_h+VIT_n','MCNN_l+VIT_n','SCNN_h+VIT_s','SCNN_h+VIT_m','Transfuse','Swinfuse','MCNN_h+VIT_n-CL','SCNN_h+VIT_n-CL',
                   'VIT_s-T1T2','VIT_m-T1T2','MCNN_h+VIT_n-T1T2','MCNN_h+VIT_s-T1T2','MCNN_h+VIT_m-T1T2', 'MCNN_h','ConVnext-UNet','MCNN_h+VIT-backbone']
    decoderOption=['linear','CNN_PUP+MLA', 'VIT_PUP+MLA', 'VIT']
    
    print(f"Model is hybrid: {opt.hybrid}. Encoder: {opt.encoder}, Decoder: {opt.decoder}")
    
    if opt.hybrid:
        assert any(opt.encoder==name for name in encoderOption)
        assert any(opt.decoder==name for name in decoderOption)
        if opt.encoder=='CNN_h+VIT_n':#si
            from .CNNh_VITn import CNNHeavy_VITNaive   
            model = CNNHeavy_VITNaive(opt)
        elif opt.encoder=='MCNN_h+VIT_n':#si
            from .MCNNh_VITn import MultiCNNHeavy_VITNaive    
            model = MultiCNNHeavy_VITNaive(opt)
        elif opt.encoder=='MCNN_h+VIT_s':#si
            from .MCNNh_VITs import MultiCNNHeavy_VITsingle   
            model = MultiCNNHeavy_VITsingle(opt)
        elif opt.encoder=='MCNN_h+VIT_m':#si
            from .MCNNh_VITm import MultiCNNHeavy_VITmultiple   
            model = MultiCNNHeavy_VITmultiple(opt)
        elif opt.encoder=='CNN_l+VIT_n':#si
            from .CNNl_VITn import CNNlight_VITNaive    
            model = CNNlight_VITNaive(opt)
        elif opt.encoder=='MCNN_l+VIT_n':#si
            from .MCNNl_VITn import MCNNlight_VITnaive  
            model = MCNNlight_VITnaive(opt)
        elif opt.encoder=='MCNN_l+VIT_s':#si to check
            from .MCNNl_VITs import MCNNlight_VITsingle   
            model = MCNNlight_VITsingle(opt)
        elif opt.encoder=='MCNN_l+VIT_m':#si to check
            from .MCNNl_VITm import MCNNlight_VITmultiple   
            model = MCNNlight_VITmultiple(opt)
        elif opt.encoder=='SCNN_h+VIT_n':#si
            from .SCNNh_VITn import SharedCNN_VITNaive   
            model = SharedCNN_VITNaive(opt)
        elif opt.encoder=='SCNN_h+VIT_s':#si
            from .SCNNh_VITs import SharedCNN_VITSimple  
            model = SharedCNN_VITSimple(opt)
        elif opt.encoder=='SCNN_h+VIT_m':#si
            from .SCNNh_VITm import SharedCNN_VITMultiple  
            model = SharedCNN_VITMultiple(opt)
        elif opt.encoder=='nnFormer':#si
            from .nnFormer import nnformer   
            model = nnformer(opt)
            model = SharedCNN_VITMultiple(opt)
        elif opt.encoder=='Transfuse':#si
            from .TransFuse import TransFuse_S   
            model = TransFuse_S(opt)
        elif opt.encoder=='Swinfuse':#si
            from .SwinFuse import Swinfuse   
            model = Swinfuse(opt)
        elif opt.encoder=='MCNN_h+VIT_n-CL':#si 'MCNN_h+VIT_n'+contrastive learning
            from .MCNNh_VITn_CL import MCNNHeavy_VITnaive_CL  
            model = MCNNHeavy_VITnaive_CL(opt)
        elif opt.encoder=='SCNN_h+VIT_n-CL':#si 'SCNN_h+VIT_n-CL'+ contrastive learning
            from .SCNNh_VITn_CL import SharedCNN_VITNaive_CL  
            model = SharedCNN_VITNaive_CL(opt)
        elif opt.encoder=='MCNN_h+VIT_n-T1T2':#si
            from .MCNNh_VITn_T1T2 import MultiCNNHeavy_VITNaive    
            model = MultiCNNHeavy_VITNaive(opt)
        elif opt.encoder=='MCNN_h+VIT_s-T1T2':#si
            from .MCNNh_VITs_T1T2 import MultiCNNHeavy_VITsingle   
            model = MultiCNNHeavy_VITsingle(opt)
        elif opt.encoder=='MCNN_h+VIT_m-T1T2':#si
            from .MCNNh_VITm_T1T2 import MultiCNNHeavy_VITmultiple   
            model = MultiCNNHeavy_VITmultiple(opt)
    else:
        assert any(opt.encoder==name for name in encoderOption)
        assert any(opt.decoder==name for name in decoderOption)
        if opt.encoder=='VIT_n':#si
            from .VITn import VITNaive    
            model = VITNaive(opt)
        elif opt.encoder=='VIT_s':#si
            from .VITs import VITSingle    
            model = VITSingle(opt)
        elif opt.encoder=='VIT_m':#si
            from .VITm import VITMultiple    
            model = VITMultiple(opt)
        elif opt.encoder=='Unet':#si
            from .netMisc import UnetMonai
            model = UnetMonai(opt)
        elif opt.encoder=='MCNN_h':#si
            from .MCNNh import MultiCNNHeavy
            model = MultiCNNHeavy(opt)
        elif opt.encoder=='MCNN_h+VIT-backbone':#si
            from .MCNNh_VIT_Backbone import MultiCNNHeavy
            model = MultiCNNHeavy(opt)
        elif opt.encoder=='SegResNet':#si
            from .netMisc import SegResNetModel
            model = SegResNetModel(opt)
        elif opt.encoder=='SegResNetVAE':#si
            from .netMisc import SegResNetVAEModel
            model = SegResNetVAEModel(opt)
        elif opt.encoder=='UNETR':#si
            from .netMisc import UNETRModel
            model = UNETRModel(opt)
        elif opt.encoder=='SwinTrans3D':#si
            from .SwinTrans3D import SwinTransformer3D  
            model = SwinTransformer3D(opt)
        elif opt.encoder=='SwinTrans3DSimple':#no
            from .SwinTrans3DS import SwinTransformer3DSimple  
            model = SwinTransformer3DSimple(opt) 
        elif opt.encoder=='VIT_s-T1T2':#si
            from .ViTs_T1T2 import VITSingle    
            model = VITSingle(opt)
        elif opt.encoder=='VIT_m-T1T2':#no
            from .ViTm_T1T2 import VITMultiple    
            model = VITMultiple(opt)
        elif opt.encoder=='ConVnext-UNet':#si
            from .ConvNeXt_Unet import ConvNeXt_Unet
            model = ConvNeXt_Unet(opt)
      


    print("model [%s] was created" % (model.name()))
    return model.init_net(model)
