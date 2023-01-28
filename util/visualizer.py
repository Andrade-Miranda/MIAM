import numpy as np
import os
import ntpath
import time
from . import util
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from util.tsne import tsne
import torch
import math


class VisualPlots():
    def __init__(self, opt):
        self.opt=opt

    def imshow_results(self,data,outputs,epoch,numberCases=2,slices=65):
        fig=plt.figure(figsize=(8, 8))
        columns = numberCases
        rows = 4
        
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            if i in range(columns):
                plt.title(data['keys'][i])
                img=data["image"][i,0,slices,:,:].detach().numpy()
                plt.imshow(img,cmap='gray')
            elif i in range(columns,columns*2):
                img=data["image"][i-numberCases,1,slices,:,:].detach().numpy()
                plt.imshow(img,cmap='gray')
            elif i in range(columns*2,columns*3):
                img=data["label"][i-(numberCases*2),0,slices,:,:].detach().numpy()
                plt.imshow(img)
            else:
                img=torch.argmax(outputs, dim=1).detach().numpy()[i-8,slices,:,:]
                plt.imshow(img)
            
        plt.show()
        plt.savefig(os.path.join(self.opt.out_dir,'PartialResults_'+str(slices)+'_'+str(epoch)+'.pdf'))                
            
    def save_Loss_MetricsBrats(self, epoch_loss_values,val_loss_values, metric_values_tumor, best_metric_epoch,best_metric,metric_values_tc,metric_values_wt,metric_values_et,val_interval):
        
        plt.figure("Loss and Dice", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Train and val Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        z = val_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, color="red",label='train')
        plt.plot(x, z, color="blue",label='val')
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values_tumor))]
        y = metric_values_tumor
        plt.xlabel("epoch")
        plt.ylabel("Dice")
        plt.plot(x, y, color="green")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.show()
        #plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_LossVsDice.pdf'))
        plt.savefig(os.path.join(self.opt.out_dir,'LossVsDice.pdf'))


        plt.figure("train", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("Val Mean Dice TC")
        x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
        y = metric_values_tc
        plt.xlabel("epoch")
        plt.plot(x, y, color="blue")
        plt.subplot(1, 3, 2)
        plt.title("Val Mean Dice WT")
        x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
        y = metric_values_wt
        plt.xlabel("epoch")
        plt.plot(x, y, color="brown")
        plt.subplot(1, 3, 3)
        plt.title("Val Mean Dice ET")
        x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
        y = metric_values_et
        plt.xlabel("epoch")
        plt.plot(x, y, color="purple")
        plt.show()
        #plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_TC-WT-ET_.pdf'))
        plt.savefig(os.path.join(self.opt.out_dir,'TC-WT-ET_.pdf'))
        

    def save_Loss_MetricsAMOS22(self, epoch_loss_values,metric_total,metric_values_organs,best_metric_epoch,best_metric,
                          val_interval,organs={'0':'Background','1':'spleen','2':'right kidney', 
                                  '3':'left kidney','4':'gallbladder','5':'esophagus','6':'liver','7':'stomach', 
                                  '8':'aorta','9':'inferior vena cava','10':'pancreas','11':'right adrenal gland',
                                  '12':'left adrenal gland','13':'duodenum','14':'bladder', '15':'prostate/uterus'}):
        
        plt.figure("Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y, color="red")
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_total))]
        y = metric_total
        plt.xlabel("epoch")
        plt.plot(x, y, color="green")
        plt.show()
        plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_LossVsVal.pdf'))
        
        fig=plt.figure(figsize=(12, 16))
        columns = 6
        rows = int(math.ceil(len(metric_values_organs[0])/6))        
        for i in range(len(metric_values_organs[0])):
            fig.add_subplot(rows, columns, i+1)
            plt.title(organs[str(i)])
            x = [val_interval * (j + 1) for j in range(len(metric_values_organs))]
            y=[]
            for numMetrics in metric_values_organs:
                y.append(numMetrics[i].cpu().detach().numpy())
            plt.xlabel("epoch")
            plt.plot(x, y, color="red")
            
        plt.show()
        plt.savefig(os.path.join(self.opt.out_dir,'Dice_per_organs.pdf'))



    def save_Loss_Metrics(self, epoch_loss_values,val_loss_values, metric_values_tumor, val_interval):
        
        plt.figure("Loss and Dice", (12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, color="red")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.subplot(1, 3, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values_tumor))]
        y = metric_values_tumor
        plt.xlabel("epoch")
        plt.ylabel("Dice")
        plt.plot(x, y, color="green")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.subplot(1, 3, 3)
        plt.title("Epoch Average Train and val Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        z = val_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, color="red",label='train')
        plt.plot(x, z, color="blue",label='val')
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.show()
        #plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_LossVsDice.pdf'))
        plt.savefig(os.path.join(self.opt.out_dir,'LossVsDice.pdf'))
        
    
        
    def save_Loss_MetricsHektor(self, epoch_loss_values,val_loss_values, metric_values_tumor,recall_values_tumor,precision_values_tumor,HDistance, AVgSurfDis, val_interval):
        
        plt.figure("Loss and Dice", (12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, color="red")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.subplot(1, 3, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values_tumor))]
        y = metric_values_tumor
        plt.xlabel("epoch")
        plt.ylabel("Dice")
        plt.plot(x, y, color="green")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.subplot(1, 3, 3)
        plt.title("Epoch Average Train and val Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        z = val_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, color="red",label='train')
        plt.plot(x, z, color="blue",label='val')
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.show()
        #plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_LossVsDice.pdf'))
        plt.savefig(os.path.join(self.opt.out_dir,'LossVsDice.pdf'))
        
        plt.figure("Recall and Precision", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Recall")
        x = [i + 1 for i in range(len(recall_values_tumor))]
        y = recall_values_tumor
        plt.xlabel("epoch")
        plt.ylabel("Recall")
        plt.plot(x, y, color="red")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.subplot(1, 2, 2)
        plt.title("Precision")
        x = [val_interval * (i + 1) for i in range(len(precision_values_tumor))]
        y = precision_values_tumor
        plt.xlabel("epoch")
        plt.ylabel("Precision")
        plt.plot(x, y, color="green")
        plt.yticks(np.arange(0, 1, step=0.1))  # Set label locations.
        plt.show()
        #plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'Recall-Precision.pdf'))
        plt.savefig(os.path.join(self.opt.out_dir,'Recall-Precision.pdf'))     
        
        plt.figure("Hausdorff distance (HD) and Average surface Distance (ASD)", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("HD")
        x = [i + 1 for i in range(len(HDistance))]
        y = HDistance
        plt.xlabel("epoch")
        plt.xlabel("HD")
        plt.plot(x, y, color="red")
        plt.subplot(1, 2, 2)
        plt.title("ASD")
        x = [val_interval * (i + 1) for i in range(len(AVgSurfDis))]
        y = AVgSurfDis
        plt.xlabel("epoch")
        plt.ylabel("ASD")
        plt.plot(x, y, color="green")
        plt.show()
        #plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'Recall-Precision.pdf'))
        plt.savefig(os.path.join(self.opt.out_dir,'HD-ASD.pdf'))  
        
        
    # def save_latentspacePlot(self,epochs,zConv,labels,zVit,option):
    #     # labelsVit=[]
    #     # for j in range(len(zVit)):
    #     #     for i in range(self.opt.batchSize):
    #     #         labelsVit+=[i]*512
        
    #     # if option[0] and option[1]:
    #     #     plt.figure("t-sne Latent space", (12, 6))
    #     #     plt.subplot(1, 2, 1)
    #     #     plt.title("Contrastive Loss")
    #     #     z=torch.cat(zConv)
    #     #     featsize=z.shape[-1]
    #     #     Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
    #     #     plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    #     #     plt.show()
    #     #     plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveCNN.pdf'))
    #     #     plt.close()
            
    #     #     plt.subplot(1, 2, 2)
    #     #     plt.title("PatchNCE")
    #     #     z=torch.cat(zVit)
    #     #     featsize=z.shape[-1]
    #     #     Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
    #     #     plt.scatter(Y[:, 0], Y[:, 1], 20, labelsVit)
    #     #     plt.show()
    #     #     plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveViT.pdf'))
            
    #     if option[0]:
    #         plt.figure("t-sne Contrastive Loss", (12, 6))
    #         z=torch.cat(zConv)
    #         featsize=z.shape[-1]
    #         Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
    #         plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    #         plt.show()
    #         plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveCNN.pdf'))
    #         plt.close()

        
        # elif option[1]:
        #     plt.figure("t-sne PatchNCE", (12, 6))
        #     z=torch.cat(zVit)
        #     featsize=z.shape[-1]
        #     Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
        #     plt.scatter(Y[:, 0], Y[:, 1], 20, labelsVit)
        #     plt.show()
        #     plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveViT.pdf'))
        #     plt.close()

        
# #########EXTRA MONAI#######
# from typing import Optional


# from monai.config.type_definitions import NdarrayOrTensor
# from monai.transforms.croppad.array import SpatialPad
# from monai.transforms.utils import rescale_array
# from monai.transforms.utils_pytorch_numpy_unification import repeat, where
# from monai.utils.type_conversion import convert_data_type, convert_to_dst_type

# from matplotlib.colors import ListedColormap



# def matshow3d(
#     volume,
#     fig=None,
#     title: Optional[str] = None,
#     figsize=(10, 10),
#     frames_per_row: Optional[int] = None,
#     frame_dim: int = -3,
#     vmin=None,
#     vmax=None,
#     every_n: int = 1,
#     interpolation: str = "none",
#     show=False,
#     fill_value=np.nan,
#     margin: int = 1,
#     dtype=np.float32,
#     **kwargs,
# ):
#     """
#     Create a 3D volume figure as a grid of images.

#     Args:
#         volume: 3D volume to display. Higher dimensional arrays will be reshaped into (-1, H, W).
#             A list of channel-first (C, H[, W, D]) arrays can also be passed in,
#             in which case they will be displayed as a padded and stacked volume.
#         fig: matplotlib figure to use. If None, a new figure will be created.
#         title: title of the figure.
#         figsize: size of the figure.
#         frames_per_row: number of frames to display in each row. If None, sqrt(firstdim) will be used.
#         frame_dim: for higher dimensional arrays, which dimension (`-1`, `-2`, `-3`) is moved to the `-3`
#             dim and reshape to (-1, H, W) shape to construct frames, default to `-3`.
#         vmin: `vmin` for the matplotlib `imshow`.
#         vmax: `vmax` for the matplotlib `imshow`.
#         every_n: factor to subsample the frames so that only every n-th frame is displayed.
#         interpolation: interpolation to use for the matplotlib `matshow`.
#         show: if True, show the figure.
#         fill_value: value to use for the empty part of the grid.
#         margin: margin to use for the grid.
#         dtype: data type of the output stacked frames.
#         kwargs: additional keyword arguments to matplotlib `matshow` and `imshow`.

#     See Also:
#         - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
#         - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html

#     Example:

#         >>> import numpy as np
#         >>> import matplotlib.pyplot as plt
#         >>> from monai.visualize import matshow3d
#         # create a figure of a 3D volume
#         >>> volume = np.random.rand(10, 10, 10)
#         >>> fig = plt.figure()
#         >>> matshow3d(volume, fig=fig, title="3D Volume")
#         >>> plt.show()
#         # create a figure of a list of channel-first 3D volumes
#         >>> volumes = [np.random.rand(1, 10, 10, 10), np.random.rand(1, 10, 10, 10)]
#         >>> fig = plt.figure()
#         >>> matshow3d(volumes, fig=fig, title="List of Volumes")
#         >>> plt.show()

#     """
#     vol: np.ndarray = convert_data_type(data=volume, output_type=np.ndarray)[0]  # type: ignore
#     if isinstance(vol, (list, tuple)):
#         # a sequence of channel-first volumes
#         if not isinstance(vol[0], np.ndarray):
#             raise ValueError("volume must be a list of arrays.")
#         pad_size = np.max(np.asarray([v.shape for v in vol]), axis=0)
#         pad = SpatialPad(pad_size[1:])  # assuming channel-first for item in vol
#         vol = np.concatenate([pad(v) for v in vol], axis=0)
#     else:  # ndarray
#         while len(vol.shape) < 3:
#             vol = np.expand_dims(vol, 0)  # so that we display 1d and 2d as well

#     vol = np.moveaxis(vol, frame_dim, -3)  # move the expected dim to construct frames with `B` or `C` dims
#     if len(vol.shape) > 3:
#         vol = vol.reshape((-1, vol.shape[-2], vol.shape[-1]))
#     vmin = np.nanmin(vol) if vmin is None else vmin
#     vmax = np.nanmax(vol) if vmax is None else vmax

#     # subsample every_n-th frame of the 3D volume
#     vol = vol[:: max(every_n, 1)]
#     if not frames_per_row:
#         frames_per_row = int(np.ceil(np.sqrt(len(vol))))
#     # create the grid of frames
#     cols = max(min(len(vol), frames_per_row), 1)
#     rows = int(np.ceil(len(vol) / cols))
#     width = [[0, cols * rows - len(vol)]] + [[margin, margin]] * (len(vol.shape) - 1)
#     vol = np.pad(vol.astype(dtype, copy=False), width, mode="constant", constant_values=fill_value)
#     im = np.block([[vol[i * cols + j] for j in range(cols)] for i in range(rows)])

#     # figure related configurations
#     if fig is None:
#         fig = plt.figure(tight_layout=True)
#     if not fig.axes:
#         fig.add_subplot(111)
#     ax = fig.axes[0]
#     ax.matshow(im, vmin=vmin, vmax=vmax, interpolation=interpolation, **kwargs)
#     ax.axis("off")
#     if title is not None:
#         ax.set_title(title)
#     if figsize is not None:
#         fig.set_size_inches(figsize)
#     if show:
#         plt.show()
#     return fig, im


# def blend_images(
#     image: NdarrayOrTensor, label: NdarrayOrTensor, alpha: float = 0.5, cmap: str = "hsv", rescale_arrays: bool = True
# ):
#     """
#     Blend an image and a label. Both should have the shape CHW[D].
#     The image may have C==1 or 3 channels (greyscale or RGB).
#     The label is expected to have C==1.

#     Args:
#         image: the input image to blend with label data.
#         label: the input label to blend with image data.
#         alpha: when blending image and label, `alpha` is the weight for the image region mapping to `label != 0`,
#             and `1 - alpha` is the weight for the label region that `label != 0`, default to `0.5`.
#         cmap: specify colormap in the matplotlib, default to `hsv`, for more details, please refer to:
#             https://matplotlib.org/2.0.2/users/colormaps.html.
#         rescale_arrays: whether to rescale the array to [0, 1] first, default to `True`.

#     """

#     if label.shape[0] != 1:
#         raise ValueError("Label should have 1 channel")
#     if image.shape[0] not in (1, 3):
#         raise ValueError("Image should have 1 or 3 channels")
#     # rescale arrays to [0, 1] if desired
#     if rescale_arrays:
#         image = rescale_array(image)
#         label = rescale_array(label)
#     # convert image to rgb (if necessary) and then rgb
#     if image.shape[0] == 1:
#         image = repeat(image, 3, axis=0)

#     def get_label_rgb(color: str, label: NdarrayOrTensor):
#         # make the color map:
#         _cmap = [cm.get_cmap(ListedColormap([color[0],color[i+1]])) for i in range(len(color)-1)]
#         label_np: np.ndarray
#         label_np, *_ = convert_data_type(label, np.ndarray)  # type: ignore
#         label_rgb_np = [_cmap[i](label_np[0][i]) for i in range(label.shape[1])]
#         label_rgb = np.moveaxis(np.stack(label_rgb_np),-1,1)
#         #label_rgb, *_ = convert_to_dst_type(label_rgb_np, label)
#         return label_rgb[:,:3,:]

#     label_rgb = get_label_rgb(cmap, label)
#     w_image = where(torch.sum(torch.from_numpy(label_rgb),(0,1))==0,1.0,alpha)
#     w_label=torch.zeros(label.shape)
#     for i in [1,0,2]:#orden de concatenacion
#         w_label=where(label[0][i]==1,label_rgb[i],w_label)
#     return w_image * image + w_label 
   
