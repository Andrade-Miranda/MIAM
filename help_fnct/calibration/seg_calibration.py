import copy
import os
from typing import Dict, Union, Optional, Sequence, Set
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from medutils.medutils import load_itk, get_gdth_pred_names
#import logging
from tqdm import tqdm
from pathlib import Path
import torch
from help_fnct.calibration.uncertainty_helpers import UncertaintyOps
from sklearn.calibration import calibration_curve, CalibrationDisplay



def type_check(gdth_path: Union[str, pathlib.Path, Sequence, None]=None,
               pred_path: Union[str, pathlib.Path, Sequence, None]=None) -> None:
    if type(gdth_path) is not type(pred_path):  # gdth_path and pred_path should have the same type
        raise Exception(f"gdth_array is {type(gdth_path)} but pred_array is {type(pred_path)}. "
                        f"They should be the same type.")
    assert any(isinstance(gdth_path, tp) for tp in [str, pathlib.Path, Sequence, type(None)])

    if isinstance(gdth_path, Sequence):
        assert any(isinstance(gdth_p, tp) for tp in [str, pathlib.Path] for gdth_p in gdth_path)


def Evaluate_Segcalibration_Folder(
                  gdth_path: Union[str, pathlib.Path, Sequence, None] = None,
                  pred_path: Union[str, pathlib.Path, Sequence, None] = None,
                  mask_path: Union[str, pathlib.Path, Sequence, None] = None,
                  outputpath: Union[str, pathlib.Path, Sequence, None] = None,#mascara to help to select the pixel used to compute calibration, if it is none take the whole volume
                  verbose: bool = True):
    """

    :param labels:  exclude background
    :param gdth_path: absolute path of a directory  or file 
    :param pred_path: absolute path of a directory  or file 
    :return: 
    """
    type_check(gdth_path, pred_path)
    print(f"start to calculate metrics (volume or distance) and write them to csv",flush=True)
    #logging.info('start to calculate metrics (volume or distance) and write them to csv')
    output_list = []

    if gdth_path is not None:
        if os.path.isfile(gdth_path):  # gdth is a file instead of a directory
            gdth_names, pred_names, mask_names = [gdth_path], [pred_path], [mask_path]
        else:
            gdth_names, pred_names = get_gdth_pred_names(gdth_path, pred_path)
            _,mask_names=get_gdth_pred_names(gdth_path, mask_path)

        with tqdm(zip(gdth_names, pred_names,mask_names), disable=not verbose) as pbar:
            for gdth_name, pred_name,mask_name in pbar:
                pbar.set_description(f'Process {os.path.basename(pred_name)} ...')
                gdth, gdth_origin, gdth_spacing = load_itk(gdth_name, require_ori_sp=True)
                pred, pred_origin, pred_spacing = load_itk(pred_name, require_ori_sp=True)
                mask, pred_origin, pred_spacing = load_itk(mask_name, require_ori_sp=True)

                ### take only region inside the prostate
                gdth_inside_mask=gdth[mask>0]
                pred_inside_mask=pred[mask>0]
                ### I CAN INCLUDE A RANDOM SAMPLING INSIDE PROSTATE Number of pixels per image to be chosen at random to evaluate calibration.
                #selected_indices = np.random.choice(len(gdth_inside_mask), size=round(len(gdth_inside_mask)*0.4), replace=False)
                #gdth_inside_mask = gdth_inside_mask[selected_indices]
                #pred_inside_mask = pred_inside_mask[selected_indices]
                ### put one sample after the other to create a big list of size samples x classes (2) [1-probs, probs]
                # Concatenate the results to create a vector
                if 'prob_vector_GT' not in locals():
                    prob_vector_GT = 1-pred_inside_mask
                    prob_vector_Tumor=pred_inside_mask
                    label_vector = gdth_inside_mask
                else:
                    prob_vector_GT = np.concatenate((prob_vector_GT, 1-pred_inside_mask))
                    prob_vector_Tumor = np.concatenate((prob_vector_Tumor, pred_inside_mask))
                    label_vector = np.concatenate((label_vector, gdth_inside_mask))
    
    probs=torch.tensor(np.concatenate((prob_vector_GT[:,None],prob_vector_Tumor[:,None]),axis=1))
    y_test=torch.tensor(label_vector)
    ECE,ADA_ECE,ks_test,prr,AUC,percentile,integrated_scores,integrated_accuracy,scores,fitted_accuracy=calibration_curves_Segmentation(probs,y_test,outputpath)

    return probs[:,-1], y_test,ECE,ADA_ECE,ks_test,prr,AUC,percentile,integrated_scores,integrated_accuracy,scores,fitted_accuracy


def calibration_curves_Segmentation(probs,y_test,outputpath):
    num_bins=15
    ECE_value,ADA_ECE_value,ks_test,prr,AUC=compute_calibrationMetrics(probs, y_test,num_bins=num_bins)
    ECE="{:.2f}".format(ECE_value.item())
    ADA_ECE="{:.2f}".format(ADA_ECE_value.item())

    ############ECE Plot#############################
    plot_calibration_curve(y_test, probs[:,-1],strategy='uniform',title='Calibration Curve',file='ECE-Seg',ECE_value=ECE,outputSoft_dir=outputpath, 
                           n_bins=num_bins, ax=None, hist=False)
    ############ADA ECE Plot#############################
    plot_calibration_curve(y_test, probs[:,-1],strategy='quantile',title='Calibration Curve',file='ADA_ECE-Seg',ECE_value=ADA_ECE,outputSoft_dir=outputpath, 
                           n_bins=num_bins, ax=None, hist=False)
    ################# KS PLOT ##############################################################
    percentile,integrated_scores,integrated_accuracy,scores,fitted_accuracy=plot_KS_graphs(probs,y_test,outputpath,"KS_Test-seg","KS_TEST")
    return ECE_value,ADA_ECE_value,ks_test,prr,AUC,percentile,integrated_scores,integrated_accuracy,scores,fitted_accuracy


def plot_calibration_curve(y_true, y_prob,strategy,title,file,ECE_value,outputSoft_dir, n_bins=10, ax=None, hist=True):
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
    plt.savefig(os.path.join(outputSoft_dir,file+'.pdf'))
    plt.close()
    ###### extra plot with bin separated in a new axis:
    #ax1 = plt.subplot(2, 1, 1)
    #ax1.grid()
    #ax1.set_title(title+"\n"+'ECE= '+ ECE_value)
    #dispCER_ECE = CalibrationDisplay(prob_true, prob_pred,y_prob)
    #ax1.plot(prob_pred, prob_true,marker='o',label='Calibration plots')
    #ax1.plot([0,1],[0,1],linestyle='--', color='gray',label='Perfect calibration')
    # Add histogram
    #ax2 = plt.subplot(2, 1, 2)
    #ax2.hist(dispCER_ECE.y_prob,
    #        range=(0,1),
    #        bins=np.maximum(10, n_bins),
    #        label="",#self.opt.name,
    #        )
    #ax2.set(title="Histogram calibration", xlabel="Mean predicted probability", ylabel="Count")
    #plt.tight_layout()
    #plt.show()
    #plt.savefig(os.path.join(outputSoft_dir,"Histogram_"+file))
    #plt.close()
    
def compute_calibrationMetrics(y_prob, y_test,num_bins):
    calibrationMetrics=UncertaintyOps()
    ECE=calibrationMetrics.ECE(y_prob, y_test, num_bins=num_bins, binning_strategy='equal_size', class_wise=False)
    ADA_ECE=calibrationMetrics.ECE(y_prob, y_test, num_bins=num_bins, binning_strategy='equal_population', class_wise=False)
    ks_test=calibrationMetrics.ks_test(y_prob, y_test,class_wise=False)
    prr = calibrationMetrics.prediction_rejection_ratio(y_test,y_prob,metric='prob', norm_logits=True)
    AUC=calibrationMetrics.auroc(y_test,y_prob[:,1])

    return ECE,ADA_ECE,ks_test,prr,AUC


def plot_KS_graphs(scores, labels, outdir, plotname, title="", spline_method='parabolic', splines=6, showplots=True):
    calibrationMetrics=UncertaintyOps
    probs, preds = torch.max(scores, dim=1)
    accs = preds.eq(labels)
    # Change to numpy, then this will work
    scores = calibrationMetrics.ensure_numpy(probs)
    labels = calibrationMetrics.ensure_numpy(accs)
    
    # Sort the data
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]
    
    # Accumulate and normalize by dividing by num samples
    nsamples = len(scores)
    integrated_scores = np.cumsum(scores) / nsamples
    integrated_accuracy   = np.cumsum(labels) / nsamples
    percentile = np.linspace (0.0, 1.0, nsamples)
    fitted_accuracy, fitted_error = calibrationMetrics.compute_accuracy(scores, labels, splines, spline_method)
    
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

        PRData=pd.DataFrame({'percentile':percentile,'integrated_scores':integrated_scores,
                             'integrated_accuracy':integrated_accuracy,'scores':scores,
                             'fitted_accuracy':fitted_accuracy})
        KStest_file = Path(Path(outdir)) / "KS-test.csv"
        PRData.to_csv(KStest_file, index=False)

        return percentile,integrated_scores,integrated_accuracy,scores,fitted_accuracy


