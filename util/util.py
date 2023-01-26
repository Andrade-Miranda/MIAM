from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import os
import collections
import nibabel as nib
#from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.segmentation import mark_boundaries



# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
#################################################################
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (2, 3,1, 0)) + 1) / 2.0 * 255.0
    return image_numpy[:,:,0,:].astype(imtype)

def tensor2seg(image_tensor,Nclasses,imtype=np.uint8):
    if Nclasses==1:
        threshold=0.5# normalmente debe ser 0.5
        logit_threshold = torch.tensor (threshold / (1 - threshold)).log()
        image = image_tensor.data > logit_threshold
    else:
        image=torch.max(image_tensor.data,dim=1,keepdim=True)[1]#torch.max returns both the max values as well as the indices. solo tomo indices
        
    image_numpy = image.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (2, 3,1, 0))
    imag=image_numpy[:,:,0,:]
    
    if  Nclasses==1:
        imag[imag==1]= 255
    else:
        for i in range(Nclasses):
            imag[imag==i]= ((255*i)/(Nclasses-1)) 
        
    return imag

def tensor2label(image_tensor,Nclasses,imtype=np.uint8):        
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (2, 3,1, 0))
    imag=image_numpy[:,:,0,:]
    
    if  Nclasses==1:
        imag[imag==1]= 255
    else:
        for i in range(Nclasses):
            imag[imag==i]= ((255*i)/(Nclasses-1)) 
        
    return imag

def tensor2nii(image_tensor, imtype=np.uint8):
    
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (2,3,1,0))
    
    return image_numpy[:,:,0,0]


def tensor2SegNii(image_tensor, numLabel=3,imtype=np.uint8):
    
    if type(image_tensor) is tuple:
        image_tensor=torch.max(image_tensor[0],dim=1,keepdim=True)[1]
    else:
        if numLabel==1:
            threshold=0.5# normalmente debe ser 0.5
            logit_threshold = torch.tensor (threshold / (1 - threshold)).log()
            image = image_tensor.data > logit_threshold
        else:
            image=torch.max(image_tensor.data,dim=1,keepdim=True)[1]#torch.max returns both the max values as well as the indices. solo tomo indices
    image_numpy = image.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (2,3,1,0))
    img=image_numpy[:,:,0,0].astype(imtype)
    imagenew=np.zeros(np.shape(image_numpy[:,:,0,0])).astype(imtype) # lineas extras por problemas en colores
    
    if numLabel==3:
        imagenew[img[:,:]==0]=0
        imagenew[img[:,:]==1]=1
        imagenew[img[:,:]==1]=2
    else:
        imagenew=img
        
    return imagenew
##########################################################################



""" ------------EXTRA  FUCNTION LOAD and save variables as json files------------------- """
import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def saveVariablesJSON(opt,ListTensorData,listTensorName,filename='LatentSpace.json'):
    # Serialization
    Data = {}
    
    for i,key,value in enumerate(listTensorName,ListTensorData):
        Data[key]=value.detach().numpy()
    
    filepath=os.path.join(opt.out_dir,filename)
    with open(filepath, 'w') as f:
        json.dump(Data,f,cls=NumpyArrayEncoder)

def LoadVariablesJSON(jsonFile):
    decodedArrays = json.loads(jsonFile)

    finalNumpyArray = numpy.asarray(decodedArrays["array"])
    print("NumPy Array")
    print(finalNumpyArray)
"""-------------------------------------------------------------"""   









""" ------------EXTRA  FUCNTION FOR NETWORK DIAGNOSIS------------------- """
def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)
    

def print_network(net):
    n_parameters = 0
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(net)
    print('Total number of trainables parameters: %d' % n_parameters)
    
def save_model(epoch,model,optimizer,loss,loss_scaler,lr_scheduler,metric,PATH):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': loss,
            'metric':metric,
            },PATH)
"""-------------------------------------------------------------"""   
    



"""----------------------Create directories--------------------------"""   
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
"""-------------------------------------------------------------"""   


""" ------------Save Nii------------------- """
def save_Nii(image_numpy, fname_out,imgpath_original):
    reference_nifti_loaded=nib.load(imgpath_original)
    nib.save(nib.Nifti1Image(image_numpy, None, reference_nifti_loaded.header), fname_out)
    

def save_image(image_numpy, image_path):
    if (len(image_numpy.shape)>2):
        image_pil = Image.fromarray(image_numpy[:,:,0])
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
""" ------------------------------------------------------------------------------- """




"""-----------------------TO CHECK----------------------------"""   
def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))
"""----------------------------------------------------------------"""   


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
##########################################################################
def one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    if labels.dim() <= 0:
        raise AssertionError("labels should have dim of 1 or more.")

    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels

def LossReduction(reduction="mean"):
    """
    See also:
        - :py:class:`monai.losses.dice.DiceLoss`
        - :py:class:`monai.losses.dice.GeneralizedDiceLoss`
        - :py:class:`monai.losses.focal_loss.FocalLoss`
        - :py:class:`monai.losses.tversky.TverskyLoss`
    """
    
    return reduction
##########################################################################        
        
        
        
        
        
        


#####EXTRA Pierre-HENRI#######################################################
def normalization_imgs(imgs):
    ''' centering and reducing data structures '''
    imgs = imgs.astype(np.float32, copy=False)
    mean = np.mean(imgs) # mean for data centering
    std = np.std(imgs) # std for data normalization
    if np.int32(std) != 0:
        imgs -= mean
        imgs /= std
    return imgs

def normalization_masks(imgs_masks):
    imgs_masks = imgs_masks.astype(np.float32, copy=False)
    imgs_masks /= 255.
    imgs_masks = imgs_masks.astype(np.uint8)
    return imgs_masks
    
def boundaries(img, pred, groundtruth):
    img = rescale_intensity(img, in_range=(np.min(img),np.max(img)), out_range=(0,1))
    if type(pred) == np.ndarray and type(groundtruth) == np.ndarray:
        out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)
        out = mark_boundaries(out, pred, color=(1, 0, 0), background_label=2)
    else:
        if type(pred) == np.ndarray:
            out = mark_boundaries(img, pred, color=(1, 0, 0), background_label=2)
        if type(groundtruth) == np.ndarray:
            out = mark_boundaries(img, groundtruth, color=(0, 1, 0), background_label=4)            
    return out  

def mask_zero(mask):
    return nib.Nifti1Image(np.zeros(shape=mask.shape).astype(np.uint8), affine=mask.affine, header=mask.header)
################################################################################