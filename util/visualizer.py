import numpy as np
import os
from . import util
import matplotlib.pyplot as plt
#from util.tsne import tsne
import torch
import math
from monai.visualize import blend_images, matshow3d
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay,auc 
from matplotlib.gridspec import GridSpec
import matplotlib.lines as lines
from help_fnct.calibration.uncertainty_helpers import UncertaintyOps
#matplotlib.use('Agg')
from matplotlib.ticker import FixedFormatter


from sklearn.calibration import calibration_curve, CalibrationDisplay
import wandb
#os.environ["WANDB_MODE"]="offline"

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

    def segment_thumbnails(self,image,label,frame_dim,savepath,FigName):
        ret = blend_images(image, label, alpha=0.6, cmap="hsv", rescale_arrays=False)
        fig=matshow3d(
                volume=ret,
                fig=None,
                title='example',
                figsize=(100, 100),
                every_n=1,
                frame_dim=frame_dim,
                channel_dim=0,
                show=False,
                cmap="gray",
                vmin=-1,
                vmax=1,
                )
        plt.savefig(savepath+FigName+'.pdf')                
            
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


class Picai_ResultsPlots():

    def __init__(self, opt,wandb_logger):
        self.opt=opt
        self.wandb_logger=wandb_logger

    def plot_wandb(self,metrics,seg_metrics,calibration_values,UseEvaluator):
        self.y_prob,self.y_test,ECE,_,_,prr,AUC,self.percentile,self.integrated_scores,self.integrated_accuracy,self.scores,self.fitted_accuracy=calibration_values
        self.wandb_logger.log({
                                "AP/imgLevel":metrics.bootstrapMetrics["AP_bootstrap"],
                                "Ranking/imgLevel":(metrics.bootstrapMetrics["AP_bootstrap"]+UseEvaluator["image-level classification"]["1"]["image-level AUC"])/2,
                                "AUROC/imgLevel":UseEvaluator["image-level classification"]["1"]["image-level AUC"],
                               "AUROC/voxelLevel":AUC,
                               "CPM/LesionLevel":metrics.bootstrapMetrics["CPM"],
                               "CCR/imgLevel":UseEvaluator["image-level classification"]["1"]["CCR"],
                               "PRR/imgLevel":metrics.bootstrapMetrics['PRR-imageLevel_bootstrap'],
                               "PRR/LesionLevel":metrics.bootstrapMetrics['PRR-LesionLevel_bootstrap'],
                               "PRR/voxelLevel":prr,
                               "VS/VoxelLevel":UseEvaluator["mean"]["1"]["Volumetric Similarity"],
                               "DSC/VoxelLevel":UseEvaluator["mean"]["1"]["Dice"],
                               "DSC/LesionLevel":metrics.bootstrapMetrics['Dice_avg-LesionLevel_bootstrap'],
                                "ECE-clasf/calibr":self.ECE,
                                "ECE-Seg/calibr":ECE,                              
                                "ROC" : wandb.plot.roc_curve([metrics.case_target[s] for s in metrics.subject_list],
                                                        [[1-metrics.case_pred[s],metrics.case_pred[s]] for s in metrics.subject_list],
                                                        labels=['Benign','Malign'],classes_to_plot=1,
                                                         title='ROC'),
                                "PR":wandb.plot.pr_curve([metrics.case_target[s] for s in metrics.subject_list], 
                                                   [[1-metrics.case_pred[s],metrics.case_pred[s]] for s in metrics.subject_list],
                                                   labels=['Benign','Malign'],classes_to_plot=1,
                                                   title='Precision vs Recall'),
                                "Confusion Matrix Test WB":wandb.plot.confusion_matrix(y_true=[metrics.case_target[s] for s in metrics.subject_list],
                                                    preds=[np.argmax([1-metrics.case_pred[s],metrics.case_pred[s]]) for s in metrics.subject_list],     
                                                    class_names=['Benign','Malign']),
                                "Confusion Matrix": wandb.sklearn.plot_confusion_matrix(y_true=[metrics.case_target[s] for s in metrics.subject_list],
                                                                                 y_pred=[np.argmax([1-metrics.case_pred[s],metrics.case_pred[s]]) for s in metrics.subject_list], 
                                                                                 labels=['Benign','Malign'])                   
                                                   })
        

        #### FROC plots
        data = [[x, y] for (x, y) in zip(list(metrics.bootstrapMetrics["fps_bs_itp_bootstrap"]), list(metrics.bootstrapMetrics["sens_bs_mean_bootstrap"]))]
        table = wandb.Table(data=data, columns=["Average number of false positives per scan", "Sensitivity"])
        self.wandb_logger.log(
            {
            "FROC": wandb.plot.line(table, "Average number of false positives per scan", "Sensitivity", title="FROC performance")
            }
        )           
        #### calibration plots ECE
        prob_true, prob_pred = calibration_curve(self.y_test, self.y_prob, n_bins=15,strategy='uniform')
        random_guessing = np.linspace(0, 1, len(prob_pred))
        # Combine Calibration diagram and Random Guessing in one plot
        self.wandb_logger.log({"Calibration diagram ECE": wandb.plot.line_series(
                                xs=[list(prob_pred)] + [list(random_guessing)],
                                ys=[list(prob_true)] + [list(random_guessing)],
                                keys=['Calibration','Reference'],
                                title="Calibration diagram ECE",
                                xname='Mean Predicted Probability'
                                )})
        ## PRR vs dice plots Voxel-Level
        data = [[x, y] for (x, y) in zip([UseEvaluator["mean"]["1"]["Dice"]],[prr])]
        table = wandb.Table(data=data, columns = ["Dice","PRR"])
        self.wandb_logger.log({"PRR vs Dice - voxel level" : wandb.plot.scatter(table, "Dice","PRR", 
                                 title="PRR vs Dice - voxel level")})
        ## PRR vs dice plots Image-Level
        data = [[x, y] for (x, y) in zip([UseEvaluator["mean"]["1"]["Dice"]],[metrics.bootstrapMetrics['PRR-imageLevel_bootstrap']])]
        table = wandb.Table(data=data, columns = ["Dice","PRR"])
        self.wandb_logger.log({"PRR vs Dice - image level" : wandb.plot.scatter(table,"Dice", "PRR", 
                                 title="PRR vs Dice - image level")})
        ## PRR vs dice plots lesion-Level
        data = [[x, y] for (x, y) in zip([metrics.bootstrapMetrics['Dice_avg-LesionLevel_bootstrap']],[metrics.bootstrapMetrics['PRR-LesionLevel_bootstrap']])]
        table = wandb.Table(data=data, columns = ["Dice","PRR"])
        self.wandb_logger.log({"PRR vs Dice - lesion level" : wandb.plot.scatter(table, "Dice","PRR", 
                                 title="PRR vs Dice - lesion level")})
        ## AUROC vs DICE plots voxel-Level
        data = [[x, y] for (x, y) in zip([UseEvaluator["mean"]["1"]["Dice"]],[AUC])]
        table = wandb.Table(data=data, columns = ["Dice","AUC"])
        self.wandb_logger.log({"AUC vs Dice - voxel level" : wandb.plot.scatter(table, "Dice","AUC", 
                                 title="AUC vs Dice - voxel level")})
        ## AUROC vs DICE plots Image-Level
        data = [[x, y] for (x, y) in zip([UseEvaluator["mean"]["1"]["Dice"]],[UseEvaluator["image-level classification"]["1"]["image-level AUC"]])]
        table = wandb.Table(data=data, columns = ["Dice","AUC"])
        self.wandb_logger.log({"AUC vs Dice - Image level" : wandb.plot.scatter(table, "Dice","AUC", 
                                 title="AUC vs Dice - Image level")})
        ## CPM vs DICE plots Image-Level
        data = [[x, y] for (x, y) in zip([UseEvaluator["mean"]["1"]["Dice"]],[UseEvaluator["image-level classification"]["1"]["image-level AUC"]])]
        table = wandb.Table(data=data, columns = ["Dice","CPM"])
        self.wandb_logger.log({"AUC vs Dice - Image level" : wandb.plot.scatter(table, "Dice", "CPM",
                                 title="CPM vs Dice - Image level")})
        

        #### calibration plots ADA ECE
        #prob_true, prob_pred = calibration_curve(self.y_test, self.y_prob, n_bins=15,strategy='quantile')
        #random_guessing = np.linspace(0, 1, len(prob_pred))
        # Combine Calibration diagram and Random Guessing in one plot
        #self.wandb_logger.log({"Calibration diagram ADA ECE": wandb.plot.line_series(
        #                        xs=[list(prob_pred)] + [list(random_guessing)],
        #                        ys=[list(prob_true)] + [list(random_guessing)],
        #                        keys=['Calibration','Reference'],
        #                        title="Calibration diagram ADA ECE",
        #                        xname='Mean Predicted Probability'
        #                       )})

        #### calibration plots KS test - cumulative score-probability
        # self.wandb_logger.log({"Cumulative Score/Probability vs percentile ": wandb.plot.line_series(
        #                         xs=[list(100.0*self.percentile)] + [list(100.0*self.percentile)],
        #                         ys=[list(self.integrated_scores)] + [list(self.integrated_accuracy)],
        #                         keys=['Cumulative Score','Cumulative Probability'],
        #                         title="Cumulative Score/Probability vs percentile ",
        #                         xname='Percentile'
        #                         )})
        # self.wandb_logger.log({"Cumulative Score/Probability vs Cumulative Score": wandb.plot.line_series(
        #                         xs=[list(self.integrated_scores)] + [list(self.integrated_scores)],
        #                         ys=[list(self.integrated_scores)] + [list(self.integrated_accuracy)],
        #                         keys=['Cumulative Score','Cumulative Probability'],
        #                         title="Cumulative Score/Probability vs Cumulative Score ",
        #                         xname='Cumulative Score'
        #                         )})
        # self.wandb_logger.log({"Score/Probability vs Percentile": wandb.plot.line_series(
        #                         xs=[list(100.0*self.percentile)] + [list(100.0*self.percentile)],
        #                         ys=[list(self.scores)] + [list(self.fitted_accuracy)],
        #                         keys=['Score','Probability'],
        #                         title="Score/Probability vs Percentile",
        #                         xname='Percentile'
        #                         )})   
        # self.wandb_logger.log({"Score/Probability vs Score": wandb.plot.line_series(
        #                         xs=[list(self.scores)] + [list(self.scores)],
        #                         ys=[list(self.scores)] + [list(self.fitted_accuracy)],
        #                         keys=['Score','Probability'],
        #                         title="Score/Probability vs Score",
        #                         xname='Score'                                
        #                         )})
        ####################################################################################                                      
        
        self.wandb_logger.finish() 
    
    def Plot_curves(self,metrics,seg_metrics,calibration_values,UseEvaluator):
        self.plot_PR_ROC_FROC(metrics)
        self.plot_FROC_bootstrap(metrics)
        self.calibrationMetrics=UncertaintyOps
        self.calibration_curves_Clasification(metrics)

        if  self.opt.enable_wandb: 
            self.plot_wandb(metrics,seg_metrics,calibration_values,UseEvaluator)   

    def plot_PR_ROC_FROC(self,metrics):
        # Precision-Recall (PR) curve
        self.precision = metrics.precision
        self.recall = metrics.recall

        # Receiver Operating Characteristic (ROC) curve
        self.tpr = metrics.case_TPR
        self.fpr = metrics.case_FPR

        # Free-Response Receiver Operating Characteristic (FROC) curve
        self.sensitivity = metrics.lesion_TPR
        self.fp_per_case = metrics.lesion_FPR

        # plot Precision-Recall (PR) curve
        self.dispPR = PrecisionRecallDisplay(precision=self.precision, recall=self.recall, average_precision=metrics.AP)
        self.dispPR.plot()
        plt.show()
        plt.savefig(os.path.join(self.opt.outputSoft_dir,'PR.pdf'))
        plt.close()

        # plot Receiver Operating Characteristic (ROC) curve
        f, ax = plt.subplots()
        self.dispROC = RocCurveDisplay(fpr=self.fpr, tpr=self.tpr, roc_auc=metrics.auroc)#,estimator_name=self.opt.name.split('__')[0])
        self.dispROC.plot(ax=ax)
        ax.plot([0,1],[0,1],linestyle='--', color='red')
        plt.show()
        plt.savefig(os.path.join(self.opt.outputSoft_dir,'ROC.pdf'))
        plt.close()

        # plot Free-Response Receiver Operating Characteristic (FROC) curve
        f, ax = plt.subplots()
        self.dispFROC = RocCurveDisplay(fpr=self.fp_per_case, tpr=self.sensitivity)#,roc_auc=metrics.aufroc)
        self.dispFROC.plot(ax=ax)
        ax.set_xlim(0.001, 5.0); ax.set_xscale('log')
        ax.set_xlabel("False positives per case"); ax.set_ylabel("Sensitivity")
        plt.show()
        plt.savefig(os.path.join(self.opt.outputSoft_dir,'FROC.pdf'))
        plt.close()

    def plot_FROC_bootstrap(self,metrics):
        fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = metrics.computeFROC_bootstrap()
        xmin = metrics.FROC_minX
        xmax = metrics.FROC_maxX
            # create FROC graphs
        ax = plt.gca()
        clr = 'b'
        ax.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
        ax.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':') # , label = "lb")
        ax.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':') # , label = "ub")
        ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Average number of false positives per scan')
        ax.set_ylabel('Sensitivity')
        #ax.legend(loc='lower right')
        ax.set_title('FROC performance ')
        
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FixedFormatter([0.125,0.25,0.5,1,2,4,8]))
        
        # set your ticks manually
        ax.xaxis.set_ticks([0.125,0.25,0.5,1,2,4,8])
        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(visible=True, which='both')#plt.grid(b=True, which='both')
        plt.tight_layout()
        plt.savefig(os.path.join(self.opt.outputSoft_dir,"froc_bootstrap.pdf"))
        plt.show()
        plt.close()
        
    def calibration_curves_Clasification(self,metrics):
        y_prob=np.array([[metrics.case_pred[s]] for s in metrics.subject_list])
        y_test=np.array([metrics.case_target[s] for s in metrics.subject_list])
        probs=torch.tensor(np.array([[1-metrics.case_pred[s],metrics.case_pred[s]] for s in metrics.subject_list]))
        self.ECE,self.ADA_ECE,self.ks_test=self.compute_calibrationMetrics(probs, torch.tensor(y_test))
        ECE="{:.2f}".format(self.ECE.item())
        ADA_ECE="{:.2f}".format(self.ADA_ECE.item())

        ############ECE Plot#############################
        self.plot_calibration_curve(y_test, y_prob,strategy='uniform',title='Calibration Curve',file='ECE-clasif',ECE_value=ECE, n_bins=10, ax=None, hist=True)
        ############ADA ECE Plot#############################
        self.plot_calibration_curve(y_test, y_prob,strategy='quantile',title='Calibration Curve',file='ADA_ECE-clasif',ECE_value=ADA_ECE, n_bins=10, ax=None, hist=True)
        ################# KS PLOT ##############################################################
        self.plot_KS_graphs(probs,torch.tensor(y_test),self.opt.outputSoft_dir,"KS_Test-clasif","KS_TEST")

    def plot_calibration_curve(self,y_true, y_prob,strategy,title,file,ECE_value, n_bins=10, ax=None, hist=True):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins,strategy=strategy)
        if ax is None:
            ax = plt.gca()
        if hist:
            ax.hist(y_prob, weights=np.ones_like(y_prob) / len(y_prob), alpha=.4,
               bins=np.maximum(10, n_bins))
        ax.plot([0, 1], [0, 1], ':', c='k')
        ax.plot(prob_pred, prob_true, marker="o")

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title+"\n"+'ECE= '+ ECE_value)
        ax.grid()
        ax.set(aspect='equal')
        plt.show()
        plt.savefig(os.path.join(self.opt.outputSoft_dir,file+'.pdf'))
        plt.close()
        ###### extra plot with bin separated in a new axis:
        ax1 = plt.subplot(2, 1, 1)
        ax1.grid()
        ax1.set_title(title+"\n"+'ECE= '+ ECE_value)
        dispCER_ECE = CalibrationDisplay(prob_true, prob_pred,y_prob)
        ax1.plot(prob_pred, prob_true,marker='o',label='Calibration plots')
        ax1.plot([0,1],[0,1],linestyle='--', color='gray',label='Perfect calibration')
        # Add histogram
        ax2 = plt.subplot(2, 1, 2)
        ax2.hist(dispCER_ECE.y_prob,
            range=(0,1),
            bins=np.maximum(10, n_bins),
            label="",#self.opt.name,
            )
        ax2.set(title="Histogram calibration", xlabel="Mean predicted probability", ylabel="Count")
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.opt.outputSoft_dir,"Histogram_"+file))
        plt.close()
    
    def compute_calibrationMetrics(self,y_prob, y_test):
        ECE=self.calibrationMetrics.ECE(y_prob, y_test, num_bins=10, binning_strategy='equal_size', class_wise=False)
        ADA_ECE=self.calibrationMetrics.ECE(y_prob, y_test, num_bins=10, binning_strategy='equal_population', class_wise=False)
        ks_test=self.calibrationMetrics.ks_test(y_prob, y_test,class_wise=False)

        return ECE,ADA_ECE,ks_test


    def plot_KS_graphs(self,scores, labels, outdir, plotname, title="", spline_method='parabolic', splines=6, showplots=True):

        probs, preds = torch.max(scores, dim=1)
        accs = preds.eq(labels)
        # Change to numpy, then this will work
        scores = self.calibrationMetrics.ensure_numpy(probs)
        labels = self.calibrationMetrics.ensure_numpy(accs)
    
        # Sort the data
        order = np.argsort(scores)
        scores = scores[order]
        labels = labels[order]
    
        # Accumulate and normalize by dividing by num samples
        nsamples = len(scores)
        integrated_scores = np.cumsum(scores) / nsamples
        integrated_accuracy   = np.cumsum(labels) / nsamples
        percentile = np.linspace (0.0, 1.0, nsamples)
        fitted_accuracy, fitted_error = self.calibrationMetrics.compute_accuracy(scores, labels, splines, spline_method)
    
        # Work out the Kolmogorov-Smirnov error
        KS_error_max = np.amax(np.absolute (integrated_scores - integrated_accuracy))
    
        if showplots:
            # Set up the graphs
            f, ax = plt.subplots(1, 4, figsize=(20, 5))
            size = 0.2
            f.suptitle (title+ f"\nKS-error = {str(round(float(KS_error_max),4)*100.0)}%, "
                               f"Probability={str(round(float(integrated_accuracy[-1]),4)*100.0)}%"
                        , fontsize=18, fontweight="bold")
    
            # First graph, (accumualated) integrated_scores and integrated_accuracy vs sample number
            ax[0].plot(100.0*percentile, integrated_scores, linewidth=3, label='Cumulative Score')
            ax[0].plot(100.0*percentile, integrated_accuracy, linewidth=3, label='Cumulative Probability')
            ax[0].set_xlabel("Percentile", fontsize=16, fontweight="bold")
            ax[0].set_ylabel("Cumulative Score / Probability", fontsize=16, fontweight="bold")
            ax[0].legend(fontsize=13)
            ax[0].set_title('(a)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
            ax[0].grid()
    
            # Second graph, (accumualated) integrated_scores and integrated_accuracy versus
            # integrated_scores
            ax[1].plot(integrated_scores, integrated_scores, linewidth=3, label='Cumulative Score')
            ax[1].plot(integrated_scores, integrated_accuracy, linewidth=3,
                       label="Cumulative Probability")
            ax[1].set_xlabel("Cumulative Score", fontsize=16, fontweight="bold")
            # ax[1].set_ylabel("Cumulative Score / Probability", fontsize=12)
            ax[1].legend(fontsize=13)
            ax[1].set_title('(b)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
            ax[1].grid()
    
            # Third graph, scores and accuracy vs percentile
            ax[2].plot(100.0*percentile, scores, linewidth=3, label='Score')
            ax[2].plot(100.0*percentile, fitted_accuracy, linewidth=3, label=f"Probability")
            ax[2].set_xlabel("Percentile", fontsize=16, fontweight="bold")
            ax[2].set_ylabel("Score / Probability", fontsize=16, fontweight="bold")
            ax[2].legend(fontsize=13)
            ax[2].set_title('(c)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
            ax[2].grid()
    
            # Fourth graph,
            # integrated_scores
            ax[3].plot(scores, scores, linewidth=3, label=f"Score")
            ax[3].plot(scores, fitted_accuracy, linewidth=3, label='Probability')
            ax[3].set_xlabel("Score", fontsize=16, fontweight="bold")
            # ax[3].set_ylabel("Score / Probability", fontsize=12)
            ax[3].legend(fontsize=13)
            ax[3].set_title('(d)', y=-size, fontweight="bold", fontsize=16) # increase or decrease y as needed
            ax[3].grid()
            plt.show()
            plt.savefig(os.path.join(outdir, plotname) + '_KS.pdf', bbox_inches="tight")
            plt.close()

        
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
   
