def create_model(opt):
    model = None
    encoderOption=['CNN_h+VIT_n','MCNN_h+VIT_n','MCNN_h+VIT_s','MCNN_h+VIT_m','Unet','SegResNetVAE','SegResNet','UNETR',
                   'SwinTrans3D','nnFormer','Transfuse','Swinfuse','MCNN_h',
                   'MCNN_h+VIT_cv','MedNeXt-b','MedNeXt-s','MedNeXt-m','MedNeXt-l',
                   'nnUNetPlain']
        
    assert any(opt.encoder==name for name in encoderOption)
    if opt.encoder=='CNN_h+VIT_n':#si
        from .CNNh_VITn import CNNHeavy_VITNaive   
        model = CNNHeavy_VITNaive(opt)
    ########### multi ecnoder CNN + VIT #################
    elif opt.encoder=='MCNN_h+VIT_n':#si
        from .MCNNh_VITn import MultiCNNHeavy_VITNaive    
        model = MultiCNNHeavy_VITNaive(opt)
    elif opt.encoder=='MCNN_h+VIT_s':#si
        from .MCNNh_VITs import MultiCNNHeavy_VITsingle   
        model = MultiCNNHeavy_VITsingle(opt)
    elif opt.encoder=='MCNN_h+VIT_m':#TO CHECK
        from .MCNNh_VITm import MultiCNNHeavy_VITmultiple   
        model = MultiCNNHeavy_VITmultiple(opt)
    elif opt.encoder=='MCNN_h+VIT_cv':#TO CHECK
        from .MCNNh_VITcv import MultiCNNHeavy_VITCrossVit   
        model = MultiCNNHeavy_VITCrossVit(opt)
        #############SOTA####################################"
    elif opt.encoder=='nnFormer':#TO CHECK
        from .nnFormer import nnformer   
        model = nnformer(opt)
    elif opt.encoder=='Transfuse':#TO CHECK
        from .TransFuse import TransFuse_S   
        model = TransFuse_S(opt)
    elif opt.encoder=='Swinfuse':#TO CHECK
        from .SwinFuse import Swinfuse   
        model = Swinfuse(opt)
    ###################### SOTA VIT#####################
    elif opt.encoder=='UNETR':#si
        from .UNETR_DeepSupervision import UNETR_DeepSupervision
        model = UNETR_DeepSupervision(opt) # deep supervision
    elif opt.encoder=='SwinTrans3D':#si
        from .Swin3D_DeepSupervision import Swin3D_DeepSupervision
        model = Swin3D_DeepSupervision(opt) # deep supervision
    ###################### MedNeXt #####################
    elif opt.encoder=='MedNeXt-b':
        from .mednextv1.create_mednext_v1 import create_mednextv1_base  
        model = create_mednextv1_base(opt,num_input_channels=opt.input_nc, num_classes=opt.output_nc) 
    elif opt.encoder=='MedNeXt-m':
        from .mednextv1.create_mednext_v1 import create_mednextv1_medium  
        model = create_mednextv1_medium(opt,num_input_channels=opt.input_nc, num_classes=opt.output_nc)
    elif opt.encoder=='MedNeXt-s':
        from .mednextv1.create_mednext_v1 import create_mednextv1_small  
        model = create_mednextv1_small(opt,num_input_channels=opt.input_nc, num_classes=opt.output_nc) 
    elif opt.encoder=='MedNeXt-l':
        from .mednextv1.create_mednext_v1 import create_mednextv1_large  
        model = create_mednextv1_large(opt,num_input_channels=opt.input_nc, num_classes=opt.output_nc)  
        ################SOTA CNN######################
    elif opt.encoder=='Unet':#TO CHECK deep super
        from .netMisc import UnetMonai
        model = UnetMonai(opt)
    elif opt.encoder=='MCNN_h':#TO CHECK deep super
        from .MCNNh import MultiCNNHeavy
        model = MultiCNNHeavy(opt)
    elif opt.encoder=='nnUNetPlain':#TO CHECK deep super
        from .nnUNet import nnUNetPlain
        model = nnUNetPlain(opt)
    elif opt.encoder=='SegResNet':#TO CHECK deep super
        from .netMisc import SegResNetModel
        model = SegResNetModel(opt)
    elif opt.encoder=='SegResNetVAE':#TO CHECK deep super
        from .netMisc import SegResNetVAEModel
        model = SegResNetVAEModel(opt)

    print("model [%s] was created" % (model.name()))
    return model.init_net(model)
