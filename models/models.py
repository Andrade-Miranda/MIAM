def create_model(opt):
    model = None
    encoderOption=['CNN_h+VIT_n','MCNN_h+VIT_n','MCNN_h+VIT_s','MCNN_h+VIT_m','Unet','VIT_n','VIT_s','VIT_m','SegResNetVAE','SegResNet','UNETR','SwinTrans3DSimple','SwinTrans3D','nnFormer',
                   'SCNN_h+VIT_n','MCNN_l+VIT_n','SCNN_h+VIT_s','SCNN_h+VIT_m','Transfuse','Swinfuse','MCNN_h+VIT_n-CL','SCNN_h+VIT_n-CL',
                    'MCNN_h','ConVnext-UNet','MCNN_h+VIT_cv','VIT_m0','MedNeXt-b','MedNeXt-s','MedNeXt-m','MedNeXt-l','nnUNetPlain','SwinTrans3D_DS']
        
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
    #############shared modality encoder with different layer normalization################
    elif opt.encoder=='SCNN_h+VIT_n':#TO CHECK
        from .SCNNh_VITn import SharedCNN_VITNaive   
        model = SharedCNN_VITNaive(opt)
    elif opt.encoder=='SCNN_h+VIT_s':#TO CHECK
        from .SCNNh_VITs import SharedCNN_VITSimple  
        model = SharedCNN_VITSimple(opt)
    elif opt.encoder=='SCNN_h+VIT_m':#TO CHECK
        from .SCNNh_VITm import SharedCNN_VITMultiple  
        model = SharedCNN_VITMultiple(opt)
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
        ###############Contrastive##################################"
    elif opt.encoder=='MCNN_h+VIT_n-CL':#TO CHECK 'MCNN_h+VIT_n'+contrastive learning
        from .MCNNh_VITn_CL import MCNNHeavy_VITnaive_CL  
        model = MCNNHeavy_VITnaive_CL(opt)
    elif opt.encoder=='SCNN_h+VIT_n-CL':#TO CHECK 'SCNN_h+VIT_n-CL'+ contrastive learning
        from .SCNNh_VITn_CL import SharedCNN_VITNaive_CL  
        model = SharedCNN_VITNaive_CL(opt)
    ###################################NO HYBRID##############################
    elif opt.encoder=='VIT_n':#TO CHECK
        from .VITn import VITNaive    
        model = VITNaive(opt)
    elif opt.encoder=='VIT_s':#TO CHECK
        from .VITs import VITSingle    
        model = VITSingle(opt)
    elif opt.encoder=='VIT_m':#TO CHECK si use crossvit only for two modalities
        from .VITm import VITMultiple    
        model = VITMultiple(opt)
    elif opt.encoder=='VIT_m0':#TO CHECK si old version multiples modalities
        from .VITm0 import VITMultiple    
        model = VITMultiple(opt)
        ###################### SOTA VIT#####################
    elif opt.encoder=='UNETR':#si
        from .UNETR_DeepSupervision import UNETR_DeepSupervision
        model = UNETR_DeepSupervision(opt) # deep supervision
    elif opt.encoder=='SwinTrans3D':#si
        from .Swin3D_DeepSupervision import Swin3D_DeepSupervision
        model = Swin3D_DeepSupervision(opt) # deep supervision
    elif opt.encoder=='SwinTrans3DSimple':#TO CHECK
        from .SwinTrans3D_SE import SwinTransformer3DSimple  
        model = SwinTransformer3DSimple(opt) 
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
    elif opt.encoder=='ConVnext-UNet':#TO CHECK deep super si modify to have all convnext
        from .ConvNeXt_Unet import ConvNeXt_Unet
        model = ConvNeXt_Unet(opt)
    elif opt.encoder=='SegResNet':#TO CHECK deep super
        from .netMisc import SegResNetModel
        model = SegResNetModel(opt)
    elif opt.encoder=='SegResNetVAE':#TO CHECK deep super
        from .netMisc import SegResNetVAEModel
        model = SegResNetVAEModel(opt)

    print("model [%s] was created" % (model.name()))
    return model.init_net(model)
