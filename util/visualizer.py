import numpy as np
import os
import ntpath
import time
from . import util
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util.tsne import tsne
import torch


class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.translation=opt.which_direction
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port = opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'Training')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create Training images directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)


    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([errors[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)
        
        

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)

    def mkdir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

######################NO en USO por el momento################################
    def save_images_to_dir(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_path[0].split('/')

        save_dir = os.path.join(image_dir, 'img_fake_only', full_path_strs[-3], full_path_strs[-2])
        self.mkdir(save_dir)

        label = 'fake_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        save_dir = os.path.join(image_dir, 'img_all', full_path_strs[-3], full_path_strs[-2])
        self.mkdir(save_dir)

        label = 'fake_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir, image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'real_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'real_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'fake_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'rec_A'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)

        label = 'rec_B'
        image_numpy = visuals[label]
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(save_dir,image_name)
        if not os.path.exists(save_path):
            util.save_image(image_numpy, save_path)
 ######################NO en USO por el momento################################



#########save image to the disk during training#############################
############Segmentation Cyclegan Synsegnet#####################################
    def save_current_imagesSegmentation(self, image_dir, visuals, image_path,epochs,epoch_iter):
        short_path = [os.path.splitext(ntpath.basename(i))[0] for i in image_path]
        
        self.mkdir(image_dir)
        print('Save training results in %s...' % image_dir)
        
        columns = 4
        rows = 3
        labels=['Real Image','Real Segmentation','Fake Segmentation']
        fig,big_axes=plt.subplots(figsize=(15, 15),nrows=rows, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            if row==1:
                big_ax.set_title("%s \n" % labels[row-1], fontsize=14)
            else:
                big_ax.set_title("%s" % labels[row-1], fontsize=14)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax._frameon = False
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            if i in range(4):
                plt.title(short_path[i])
                img=visuals['real_A'][:,:,i]
                plt.imshow(img,cmap='gray')
            elif i in range(4,8):
                img=visuals['manual_B'][:,:,i-4]
                plt.imshow(img,cmap='gray')
            else:
                img=visuals['seg_B'][:,:,i-8]
                plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join(image_dir,str(epochs)+'_'+str(epoch_iter)+'_RealVsSegmented.pdf'))
        plt.close()
##########################################################################


# save image to the disk
    def save_current_images(self, image_dir, visuals, A_paths,B_paths,epochs,epoch_iter):
        short_pathA = [os.path.splitext(ntpath.basename(i))[0] for i in A_paths]
        short_pathB = [os.path.splitext(ntpath.basename(i))[0] for i in B_paths]
        
        self.mkdir(image_dir)
        print('Save training results in %s...' % image_dir)
        
        columns = 4
        rows = 3
        labels=['Real Image','Real Segmentation','Fake Segmentation']
        fig,big_axes=plt.subplots(figsize=(15, 15),nrows=rows, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            if row==1:
                big_ax.set_title("%s \n" % labels[row-1], fontsize=14)
            else:
                big_ax.set_title("%s" % labels[row-1], fontsize=14)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax._frameon = False
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            if i in range(4):
                plt.title(short_pathA[i])
                img=visuals['real_A'][:,:,i]
                plt.imshow(img,cmap='gray')
            elif i in range(4,8):
                img=visuals['manual_B'][:,:,i-4]
                plt.imshow(img,cmap='gray')
            else:
                img=visuals['seg_B'][:,:,i-8]
                plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join(image_dir,str(epochs)+'_'+str(epoch_iter)+'_RealVsSegmented.pdf'))
        plt.close()
        
        
        
        labels=['Real A','Fake B','Reconstructed A']
        fig,big_axes=plt.subplots(figsize=(15, 15),nrows=rows, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            if row==1:
                big_ax.set_title("%s \n" % labels[row-1], fontsize=14)
            else:
                big_ax.set_title("%s" % labels[row-1], fontsize=14)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax._frameon = False
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            if i in range(4):
                plt.title(short_pathA[i])
                img=visuals['real_A'][:,:,i]
                plt.imshow(img,cmap='gray')
            elif i in range(4,8):
                img=visuals['fake_B'][:,:,i-4]
                plt.imshow(img,cmap='gray')
            else:
                img=visuals['rec_A'][:,:,i-8]
                plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join(image_dir,str(epochs)+'_'+str(epoch_iter)+'_RealAVsfakeB.pdf'))
        plt.close()
        
        
        labels=['Real B','Fake A','Reconstructed B']
        fig,big_axes=plt.subplots(figsize=(15, 15),nrows=rows, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            if row==1:
                big_ax.set_title("%s \n" % labels[row-1], fontsize=14)
            else:
                big_ax.set_title("%s" % labels[row-1], fontsize=14)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax._frameon = False        
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            if i in range(4):
                plt.title(short_pathB[i])
                img=visuals['real_B'][:,:,i]
                plt.imshow(img,cmap='gray')
            elif i in range(4,8):
                img=visuals['fake_A'][:,:,i-4]
                plt.imshow(img,cmap='gray')
            else:
                img=visuals['rec_B'][:,:,i-8]
                plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join(image_dir,str(epochs)+'_'+str(epoch_iter)+'_RealBVsfakeA.pdf'))
        plt.close()
        

# save image to the disk
#image _dir: directory to save images
#visuals: dictionary with images
#
    def save_current_imagesCyclegan(self, image_dir, visuals, A_paths,B_paths,epochs,epoch_iter):
        short_pathA = [os.path.splitext(ntpath.basename(i))[0] for i in A_paths]
        short_pathB = [os.path.splitext(ntpath.basename(i))[0] for i in B_paths]
        
        self.mkdir(image_dir)
        print('Save training results in %s...' % image_dir)
        
        columns = 4
        rows = 3
                
        labels=['Real A','Fake B','Reconstructed A']
        fig,big_axes=plt.subplots(figsize=(15, 15),nrows=rows, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            if row==1:
                big_ax.set_title("%s \n" % labels[row-1], fontsize=14)
            else:
                big_ax.set_title("%s" % labels[row-1], fontsize=14)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax._frameon = False
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            if i in range(4):
                plt.title(short_pathA[i])
                img=visuals['real_A'][:,:,i]
                plt.imshow(img,cmap='gray')
            elif i in range(4,8):
                img=visuals['fake_B'][:,:,i-4]
                plt.imshow(img,cmap='gray')
            else:
                img=visuals['rec_A'][:,:,i-8]
                plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join(image_dir,str(epochs)+'_'+str(epoch_iter)+'_RealAVsfakeB.pdf'))
        plt.close()
        
        
        labels=['Real B','Fake A','Reconstructed B']
        fig,big_axes=plt.subplots(figsize=(15, 15),nrows=rows, ncols=1, sharey=True)
        for row, big_ax in enumerate(big_axes, start=1):
            if row==1:
                big_ax.set_title("%s \n" % labels[row-1], fontsize=14)
            else:
                big_ax.set_title("%s" % labels[row-1], fontsize=14)
            big_ax.set_xticks([])
            big_ax.set_yticks([])
            big_ax._frameon = False        
        for i in range(columns*rows):
            fig.add_subplot(rows, columns, i+1)
            plt.axis('off')
            if i in range(4):
                plt.title(short_pathB[i])
                img=visuals['real_B'][:,:,i]
                plt.imshow(img,cmap='gray')
            elif i in range(4,8):
                img=visuals['fake_A'][:,:,i-4]
                plt.imshow(img,cmap='gray')
            else:
                img=visuals['rec_B'][:,:,i-8]
                plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join(image_dir,str(epochs)+'_'+str(epoch_iter)+'_RealBVsfakeA.pdf'))
        plt.close()



    def save_cyclegan_images_to_dir(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_path[0].split('/')

        save_dirFAKE = os.path.join(image_dir, full_path_strs[-2],'FakeImg')

        self.mkdir(save_dirFAKE)

        label = 'fake_Img'
        image_numpyImg = visuals[label]
        image_name = '%s_%s.nii.gz' % (name, self.translation)
        save_pathImg= os.path.join(save_dirFAKE,image_name)
        

        if not os.path.exists(save_pathImg):
            util.save_Nii(image_numpyImg, save_pathImg,image_path[0])


    def save_seg_images_to_dir(self, image_dir, visuals, image_path):
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]
        full_path_strs = image_path[0].split('/')

        save_dirFAKE = os.path.join(image_dir, full_path_strs[-2],'FakeImg')
        save_dirSEG = os.path.join(image_dir, full_path_strs[-2],'FakeSeg')

        self.mkdir(save_dirFAKE)
        self.mkdir(save_dirSEG)

        label = 'fake_Seg'
        image_numpySeg = visuals[label]
        image_name = '%s_%s.nii.gz' % (name, label)
        save_pathSeg = os.path.join(save_dirSEG,image_name)
        
        label = 'fake_Img'
        image_numpyImg = visuals[label]
        image_name = '%s_%s.nii.gz' % (name, self.translation)
        save_pathImg= os.path.join(save_dirFAKE,image_name)
        
        
        if not os.path.exists(save_pathSeg):
            util.save_Nii(image_numpySeg, save_pathSeg,image_path[0])
            
        if not os.path.exists(save_pathImg):
            util.save_Nii(image_numpyImg, save_pathImg,image_path[0])



class VisualPlots():
    def __init__(self, opt):
        self.opt=opt            
            
    def save_Loss_Metrics(self, epoch_loss_values, metric_values, best_metric_epoch,best_metric,metric_values_tc,metric_values_wt,metric_values_et,val_interval):
        
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.plot(x, y, color="red")
        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x = [val_interval * (i + 1) for i in range(len(metric_values))]
        y = metric_values
        plt.xlabel("epoch")
        plt.plot(x, y, color="green")
        plt.show()
        plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_LossVsVal.pdf'))


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
        plt.savefig(os.path.join(self.opt.out_dir,str(best_metric_epoch)+'_'+str(best_metric)+'_TC-WT-ET_.pdf'))
        
    def save_Loss_MetricsHektor(self, epoch_loss_values, metric_values_tumor,recall_values_tumor,precision_values_tumor,HDistance, AVgSurfDis, val_interval):
        
        plt.figure("Loss and Dice", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        y = epoch_loss_values
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, y, color="red")
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
        
        
    def save_latentspacePlot(self,epochs,zConv,labels,zVit,option):
        # labelsVit=[]
        # for j in range(len(zVit)):
        #     for i in range(self.opt.batchSize):
        #         labelsVit+=[i]*512
        
        # if option[0] and option[1]:
        #     plt.figure("t-sne Latent space", (12, 6))
        #     plt.subplot(1, 2, 1)
        #     plt.title("Contrastive Loss")
        #     z=torch.cat(zConv)
        #     featsize=z.shape[-1]
        #     Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
        #     plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
        #     plt.show()
        #     plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveCNN.pdf'))
        #     plt.close()
            
        #     plt.subplot(1, 2, 2)
        #     plt.title("PatchNCE")
        #     z=torch.cat(zVit)
        #     featsize=z.shape[-1]
        #     Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
        #     plt.scatter(Y[:, 0], Y[:, 1], 20, labelsVit)
        #     plt.show()
        #     plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveViT.pdf'))
            
        if option[0]:
            plt.figure("t-sne Contrastive Loss", (12, 6))
            z=torch.cat(zConv)
            featsize=z.shape[-1]
            Y = tsne(z.view(-1,featsize).detach().cpu().numpy(), 2, 50, 20.0)
            plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
            plt.show()
            plt.savefig(os.path.join(self.opt.out_dir,str(epochs)+'_ContrastiveCNN.pdf'))
            plt.close()

        
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
   
