#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import torch
from scipy.stats import t
import scipy.stats as st
import matplotlib.pyplot as plt
import os


try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.data_utils import PathLike, load_metrics, save_metrics


@dataclass
class Metrics:
    lesion_results: Union[Dict[Hashable, List[Tuple[int, float, float]]], PathLike]
    case_target: Optional[Dict[Hashable, int]] = None
    case_pred: Optional[Dict[Hashable, float]] = None
    case_weight: Optional[Union[Dict[Hashable, float], List[float]]] = None
    lesion_weight: Optional[Dict[Hashable, List[float]]] = None
    bootstrap: bool = False
    thresholds: "Optional[npt.NDArray[np.float64]]" = None
    subject_list: Optional[List[str]] = None
    sort: bool = True

    def __post_init__(self):
        
        if isinstance(self.lesion_results, (str, Path)):
            # load metrics from file
            self.load(self.lesion_results)

        if self.subject_list is None:
            self.subject_list = sorted(list(self.lesion_results))

        if self.case_target is None:
            # derive case-level targets as the maximum lesion-level target
            self.case_target = {
                idx: max([is_lesion for is_lesion, _, _ in case_y_list]) if len(case_y_list) else 0
                for idx, case_y_list in self.lesion_results.items()
            }

        if self.case_pred is None:
            # derive case-level predictions as the maximum lesion-level prediction
            self.case_pred = {
                idx: max([confidence for _, confidence, _ in case_y_list]) if len(case_y_list) else 0
                for idx, case_y_list in self.lesion_results.items()
            }

        if not isinstance(self.case_weight, dict):
            subject_list = list(self.case_target)
            if self.case_weight is None:
                self.case_weight = {idx: 1 for idx in subject_list}
            else:
                self.case_weight = {idx: weight for idx, weight in zip(subject_list, self.case_weight)}

        if self.lesion_weight is None:
            subject_list = sorted(list(self.lesion_results))
            self.lesion_weight = {idx: [1]*len(case_y_list) for idx, case_y_list in self.lesion_results.items()}

        if self.bootstrap:
            self.bootstrap_list=[]
            if self.sort:
                self.reps = 1000
                for i in range(self.reps):
                    bootstrap_list = list(np.random.choice(self.subject_list, len(self.case_pred), replace=True))
                    bootstrap_list = sorted(bootstrap_list)
                    self.bootstrap_list.append(bootstrap_list)
        if self.sort:
            # sort dictionaries
            subject_list = sorted(list(self.lesion_results))
            self.lesion_results = {idx: self.lesion_results[idx] for idx in subject_list}
            self.lesion_weight = {idx: self.lesion_weight[idx] for idx in subject_list}
            self.case_target = {idx: self.case_target[idx] for idx in subject_list}
            self.case_pred = {idx: self.case_pred[idx] for idx in subject_list}
            self.case_weight = {idx: self.case_weight[idx] for idx in subject_list}

    # aggregates
    def calc_auroc(self, subject_list: Optional[List[str]] = None) -> float:
        """Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)"""
        return self.calculate_ROC(subject_list=subject_list)['AUROC']
    
    def calc_aufroc(self, subject_list: Optional[List[str]] = None) -> float:
        """Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)"""
        return self.calculate_ROC(subject_list=subject_list)['AUFROC']

    @property
    def auroc(self) -> float:
        """Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)"""
        return self.calc_auroc()

    @property
    def aufroc(self) -> float:
        """Calculate case-level Area Under the Receiver Operating Characteristic curve (AUROC)"""
        return self.calc_aufroc()

    def calc_AP(self, subject_list: Optional[List[str]] = None) -> float:
        """Calculate Average Precision"""
        return self.calculate_precision_recall(subject_list=subject_list)['AP']

    @property
    def AP(self) -> float:
        """Calculate Average Precision"""
        return self.calc_AP()

    @property
    def num_cases(self) -> int:
        """Calculate the number of cases"""
        return len(self.subject_list)

    @property
    def num_lesions(self) -> int:
        """Calculate the number of ground truth lesions"""
        return sum([is_lesion for is_lesion, *_ in self.lesion_results_flat])

    @property
    def score(self):
        """Calculate the ranking score, as used in the PI-CAI 22 Grand Challenge"""
        return (self.auroc + self.AP) / 2

    # lesion-level results
    def get_lesion_results_flat(self, subject_list: Optional[List[str]] = None):
        """Flatten the per-case lesion evaluation results into a single list"""
        if subject_list is None:
            subject_list = self.subject_list

        return [
            (is_lesion, confidence, overlap)
            for subject_id in subject_list
            for is_lesion, confidence, overlap in self.lesion_results[subject_id]
        ]

    @property
    def lesion_results_flat(self) -> List[Tuple[int, float, float]]:
        """Flatten the per-case y_list"""
        return self.get_lesion_results_flat()

    def get_lesion_weight_flat(self, subject_list: Optional[List[str]] = None) -> List[float]:
        """Retrieve lesion-wise sample weights (for a given subset of cases)"""
        if subject_list is None:
            subject_list = self.subject_list

        # collect lesion weights (and flatten)
        return [weight for subject_id in subject_list for weight in self.lesion_weight[subject_id]]

    @property
    def lesion_weight_flat(self) -> List[float]:
        """Retrieve lesion-wise sample weights (for a given subset of cases)"""
        return self.get_lesion_weight_flat()

    @property
    def precision(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level precision at each threshold"""
        return self.calculate_precision_recall()['precision']

    @property
    def recall(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level recall at each threshold"""
        return self.calculate_precision_recall()['recall']

    @property
    def lesion_TP(self) -> "npt.NDArray[np.float64]":
        """Calculate number of true positive lesion detections at each threshold"""
        return self.calculate_counts()['TP']

    @property
    def lesion_FP(self) -> "npt.NDArray[np.float64]":
        """Calculate number of false positive lesion detections at each threshold"""
        return self.calculate_counts()['FP']

    @property
    def lesion_TPR(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level true positive rate (sensitivity) at each threshold"""
        if self.num_lesions > 0:
            return self.lesion_TP / self.num_lesions
        else:
            return np.array([np.nan] * len(self.lesion_TP))

    @property
    def lesion_FPR(self) -> "npt.NDArray[np.float64]":
        """Calculate lesion-level false positive rate (number of false positives per case) at each threshold"""
        return self.lesion_FP / self.num_cases

    # case-level results
    def calc_case_TPR(self, subject_list: Optional[List[str]] = None) -> "npt.NDArray[np.float64]":
        """Calculate case-level true positive rate (sensitivity) at each threshold"""
        return self.calculate_ROC(subject_list=subject_list)['TPR']

    @property
    def case_TPR(self) -> "npt.NDArray[np.float64]":
        """Calculate case-level true positive rate (sensitivity) at each threshold"""
        return self.calc_case_TPR()

    def calc_case_FPR(self, subject_list: Optional[List[str]] = None) -> "npt.NDArray[np.float64]":
        """Calculate case-level false positive rate (1 - specificity) at each threshold"""
        return self.calculate_ROC(subject_list=subject_list)['FPR']

    @property
    def case_FPR(self) -> "npt.NDArray[np.float64]":
        """Calculate case-level false positive rate (1 - specificity) at each threshold"""
        return self.calc_case_FPR()

    # supporting functions
    def calculate_counts(self, subject_list: Optional[List[str]] = None) -> "Dict[str, npt.NDArray[np.float32]]":
        """
        Calculate lesion-level true positive (TP) detections and false positive (FP) detections as each threshold.
        """
        # flatten y_list (and select cases in subject_list)
        lesion_y_list = self.get_lesion_results_flat(subject_list=subject_list)

        # collect targets and predictions
        y_true: "npt.NDArray[np.float64]" = np.array([target for target, *_ in lesion_y_list])
        y_pred: "npt.NDArray[np.float64]" = np.array([pred for _, pred, *_ in lesion_y_list])

        if self.thresholds is None:
            # collect thresholds for lesion-based analysis
            self.thresholds = np.unique(y_pred)
            self.thresholds[::-1].sort()  # sort thresholds in descending order (inplace)

            # for >10,000 thresholds: resample to 10,000 unique thresholds, while also
            # keeping all thresholds higher than 0.8 and the first 20 thresholds
            if len(self.thresholds) > 10_000:
                rng = np.arange(1, len(self.thresholds), len(self.thresholds)/10_000, dtype=np.int32)
                st = [self.thresholds[i] for i in rng]
                low_thresholds = self.thresholds[-20:]
                self.thresholds = np.array([t for t in self.thresholds if t > 0.8 or t in st or t in low_thresholds])

        # define placeholders
        FP: "npt.NDArray[np.float32]" = np.zeros_like(self.thresholds, dtype=np.float32)
        TP: "npt.NDArray[np.float32]" = np.zeros_like(self.thresholds, dtype=np.float32)

        # for each threshold: count FPs and TPs
        for i, th in enumerate(self.thresholds):
            y_pred_thresholded = (y_pred >= th).astype(int)
            tp = np.sum(y_true*y_pred_thresholded)
            fp = np.sum(y_pred_thresholded - y_true*y_pred_thresholded)

            # update with new point
            FP[i] = fp
            TP[i] = tp

        # extend curve to infinity
        TP[-1] = TP[-2]
        FP[-1] = np.inf

        return {
            'TP': TP,
            'FP': FP,
        }

    def calculate_precision_recall(self, subject_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Precision-Recall curve and calculate average precision (AP).
        """
        # flatten y_list (and select cases in subject_list)
        lesion_y_list = self.get_lesion_results_flat(subject_list=subject_list)

        # collect targets and predictions
        y_true: "npt.NDArray[np.float64]" = np.array([target for target, *_ in lesion_y_list])
        y_pred: "npt.NDArray[np.float64]" = np.array([pred for _, pred, *_ in lesion_y_list])

        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(
            y_true=y_true,
            probas_pred=y_pred,
            sample_weight=self.get_lesion_weight_flat(subject_list=subject_list)
        )

        # set precision to zero at a threshold of "zero", as those lesion
        # candidates are included just to convey the number of lesions to
        # the precision_recall_curve function, and not actual candidates
        precision[:-1][thresholds == 0] = 0

        # calculate average precision using the step function integral
        # The following works because the last entry of precision is
        # guaranteed to be 1, as returned by precision_recall_curve
        # Taken from https://github.com/scikit-learn/scikit-learn/blob/
        # 32f9deaaf27c7ae56898222be9d820ba0fd1054f/sklearn/metrics/_ranking.py#L212
        AP = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)

        return {
            'AP': AP,
            'precision': precision,
            'recall': recall,
            'Best_THR':  thresholds[ix],
        }

    def calculate_ROC(self, subject_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate Receiver Operating Characteristic curve for case-level risk stratification.
        """
        if subject_list is None:
            subject_list = self.subject_list

        fpr, tpr, threshold = roc_curve(
            y_true=[self.case_target[s] for s in subject_list],
            y_score=[self.case_pred[s] for s in subject_list],
            sample_weight=[self.case_weight[s] for s in subject_list],
        )

        auroc = auc(fpr, tpr)
        J=tpr-fpr
        idx=np.argmax(J)
        ###froc### I ommit the last value to avoid infinity problem
        TP=self.calculate_counts(subject_list=subject_list)['TP']
        FP=self.calculate_counts(subject_list=subject_list)['FP']
        num_lesions=sum([is_lesion for is_lesion, *_ in self.get_lesion_results_flat(subject_list=subject_list)])
        sensitivity = TP[:-1] / num_lesions
        fp_per_case = FP[:-1] / self.num_cases
        froc=auc(fp_per_case, sensitivity)#or self.lesion_FPR [:-1],self.lesion_TPR[:-1]
        return {
            'FPR': fpr,
            'TPR': tpr,
            'AUFROC':froc,
            'AUROC': auroc,
            'Best_THR': threshold[idx]
        }

    def prediction_rejection_ratio(self,subject_list: Optional[List[str]] = None, metric='prob', norm_logits=False,level='image', misclassification=None):
         # Based on https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/rejection.py
        # compute area between base_error(1-x) and the rejection curve
        # compute area between base_error(1-x) and the oracle curve
        # take the ratio

        if level=='image':
            labels=torch.tensor([self.case_target[s] for s in subject_list])
            logits_tumor=[self.case_pred[s]  for s in subject_list]
            logits_notumor=[1-self.case_pred[s]  for s in subject_list]            
        else:# lesion level
            data=[]#data=self.get_lesion_results_flat(subject_list=subject_list)
            for subject_id in subject_list:
                if len(self.lesion_results[subject_id])==0:
                    data.append((0,0,1))
                else:
                    for is_lesion, confidence, overlap in self.lesion_results[subject_id]:
                        data.append((is_lesion, confidence, overlap))#self.get_lesion_results_flat(subject_list=subject_list)
            labels=torch.tensor([truelabel[0] for truelabel in data ])
            logits_tumor=[truelabel[1] for truelabel in data]
            logits_notumor=[1-truelabel[1] for truelabel in data]
        logits=torch.tensor(np.concatenate((np.array(logits_notumor)[:,None],np.array(logits_tumor)[:,None]),axis=1))

        # consider only positive or negative for missclassification
        if  misclassification=='Positive': #I only will check missclassification of positive true samples, check when the model is not sure about the positive
            logits = logits[labels == 1, :]
            labels = labels[labels == 1]
        elif  misclassification=='Negative': #I only will check missclassification of negative true samples, check when the model is not sure about the negative
            logits = logits[labels == 0, :]
            labels = labels[labels == 0]
        else:
            pass # evaluate all data, default case
   
         # Get class probabilities
        probs = logits # For maskformer we compute probs directly
     
        if metric == 'prob':
             confidence, preds = torch.max(probs, dim=1) # Take as confidence the probability of the predicted class
             sorted_idx = torch.argsort(confidence, descending = True)
        elif metric == 'entropy':
            probs = probs + 1e-16
            confidence = torch.sum((torch.log(probs) * probs), axis=1) # Negative entropy
            preds = torch.argmax(probs, dim=1)
            sorted_idx = torch.argsort(confidence, descending = False)
         # the rejection plots needs to reject to the right the most uncertain/less confident samples
         # if uncertainty metric, high means reject, sort in ascending uncertainty;
        # if confidence metric, low means reject, sort in descending confidence
        

        # reverse cumulative errors function (rev = from all to first, instead from first error to all)
        self.rev_cum_errors = []
        # fraction of data rejected, to compute a certain value of rev_cum_errors
        self.fraction_data = []

        num_samples = preds.shape[0]
     
        errors = (labels[sorted_idx] != preds[sorted_idx]).float().numpy()
        self.rev_cum_errors = np.cumsum(errors) / num_samples
        self.fraction_data = np.array([float(i + 1) / float(num_samples) * 100.0 for i in range(num_samples)])
     
        base_error = self.rev_cum_errors[-1] # error when all data is taken into account

        # area under the rejection curve (used later to compute area between random and rejection curve)
        auc_uns = 1.0 - auc(self.fraction_data / 100.0, self.rev_cum_errors[::-1] / 100.0) #invert cumulate error

        # random rejection baseline, it's 1 - x line "scaled" and "shifted" to pass through base error and go to 100% rejection
        self.random_rejection = np.asarray(
                 [base_error * (1.0 - float(i) / float(num_samples)) for i in range(num_samples)],
                 dtype=np.float32)
        # area under random rejection, should be 0.5
        auc_rnd = 1.0 - auc(self.fraction_data / 100.0, self.random_rejection / 100.0)

        # oracle curve, the oracle is assumed to commit the base error
        # making the oracle curve commit the base error allows to remove the impact of the base error when computing
        # the ratio of areas
        # line passing through base error at perc_rej = 0, and crossing
        # the line goes from x=0 to x=base_error/100*num_samples <- this is when the line intersects the x axis
        # which means the oracle ONLY REJECTS THE SAMPLES THAT ARE MISCASSIFIED
        # afterwards the function is set to zero
        orc_rejection = np.asarray(
                 [base_error * (1.0 - float(i) / float(base_error / 100.0 * num_samples)) for i in
                  range(int(base_error / 100.0 * num_samples))], dtype=np.float32)
        self.orc = np.zeros_like(self.rev_cum_errors)
        self.orc[0:orc_rejection.shape[0]] = orc_rejection
        auc_orc = 1.0 - auc(self.fraction_data / 100.0, self.orc / 100.0)
         
        # reported from -100 to 100
        rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

        return rejection_ratio
    
    
    def DiceLesion(self,subject_list: Optional[List[str]] = None):
        data=[lesions[2] for lesions in self.get_lesion_results_flat(subject_list=subject_list) if lesions[2]>0]

        # Mean and standard error of the mean
        mean_value = np.mean(data)
        std_value = np.std(data)

        return {'mean':mean_value,"SD":std_value}
    
    def compute_confidenceInterval(self,data,confidence_level=0.95):
        # Confidence level (e.g., 95% confidence interval)
        confidence_level = confidence_level
        # Degrees of freedom (N - 1 for a sample, N for a population)
        degrees_of_freedom = len(data) - 1
        # Mean and standard error of the mean
        mean_value = np.mean(data)
        std_error = st.sem(data)
        # Compute the confidence interval
        confidence_interval = t.interval(confidence_level, degrees_of_freedom, loc=mean_value, scale=std_error)

        return {"mean":mean_value,"CI":confidence_interval,"std":std_error}
    
    def computeFROC_bootstrap(self, confidence = 0.95):
    
        fps_lists = []
        sens_lists = []
        thresholds_lists = []
        sens_mean,sens_lb,sens_up=[],[],[]
        # plot settings
        self.FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
        self.FROC_maxX = 8 # Maximum value of x-axis of FROC curve
        numberOfBootstrapSamples=len(self.bootstrap_list)
        for i in range(numberOfBootstrapSamples):
            print ('computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples))
            ###froc###
            TP=self.calculate_counts(subject_list=self.bootstrap_list[i])['TP']
            FP=self.calculate_counts(subject_list=self.bootstrap_list[i])['FP']
            sens = TP / sum([is_lesion for is_lesion, *_ in self.get_lesion_results_flat(subject_list=self.bootstrap_list[i])])
            fps = FP / self.num_cases
    
            fps_lists.append(fps)
            sens_lists.append(sens)
            #thresholds_lists.append(thresholds)

        # compute statistic
        all_fps = np.linspace(self.FROC_minX, self.FROC_maxX, num=10000)
    
        # Then interpolate all FROC curves at this points
        interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
        for i in range(numberOfBootstrapSamples):
            interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
        # compute mean and CI
        mean_value = np.mean(interp_sens,axis=0)
        std_error = st.sem(interp_sens,axis=0)
        # Compute the confidence interval
        confidence_interval = t.interval(confidence, len(interp_sens) - 1, loc=mean_value, scale=std_error)
        
        sens_mean,sens_lb,sens_up=mean_value,confidence_interval[0],confidence_interval[1]

        return all_fps, sens_mean, sens_lb, sens_up
    
    def Plot_PRR_Image_Lesion(self, path: str):

        for i in ['image','lesion']:
            PRRImageBoot=self.prediction_rejection_ratio(subject_list=self.subject_list,level=i)
            #plot rejection ratio plot
            plt.plot(self.fraction_data, self.orc, lw=2)
            plt.fill_between(self.fraction_data, self.orc, self.random_rejection, alpha=0.5)
            plt.plot(self.fraction_data, self.rev_cum_errors[::-1], lw=2)
            plt.fill_between(self.fraction_data, self.rev_cum_errors[::-1], self.random_rejection, alpha=0.0)
            plt.plot(self.fraction_data, self.random_rejection, 'k--', lw=2)
            plt.legend(['Oracle', 'Uncertainty', 'Random'])
            plt.xlabel('Percentage of predictions rejected to oracle')
            plt.ylabel('Classification Error (%)')
            plt.savefig(os.path.join(path,'Rejection-Curve-oracle_'+i+'.png'), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close()

            plt.plot(self.fraction_data, self.orc, lw=2)
            plt.fill_between(self.fraction_data, self.orc, self.random_rejection, alpha=0.0)
            plt.plot(self.fraction_data, self.rev_cum_errors[::-1], lw=2)
            plt.fill_between(self.fraction_data, self.rev_cum_errors[::-1], self.random_rejection, alpha=0.5)
            plt.plot(self.fraction_data, self.random_rejection, 'k--', lw=2)
            plt.legend(['Oracle', 'Uncertainty', 'Random'])
            plt.xlabel('Percentage of predictions rejected to oracle')
            plt.ylabel('Classification Error (%)')
            plt.savefig(os.path.join(path,'Rejection-Curve-uncertainty_'+i+'.png'), bbox_inches='tight', dpi=300)
            # plt.show()
            plt.close()
    
                
    @property
    def version(self):
        return "1.5.x"

    def as_dict(self):
        return {
            # aggregates
            "auroc": self.auroc,
            "AP": self.AP,
            "num_cases": self.num_cases,
            "num_lesions": self.num_lesions,
            "picai_eval_version": self.version,

            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,

            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }
    

    def fullBootstrap(self):
        aurocBoot,APBoot,aufrocBoot,PRRImageBoot,PRRLesionBoot,Dice_avgBoot=[],[],[],[],[],[]
        for i in range(self.reps):
            aurocBoot.append(self.calculate_ROC(subject_list=self.bootstrap_list[i])['AUROC'])
            APBoot.append(self.calculate_precision_recall(subject_list=self.bootstrap_list[i])['AP'])
            aufrocBoot.append(self.calculate_ROC(subject_list=self.bootstrap_list[i])['AUFROC'])
            PRRImageBoot.append(self.prediction_rejection_ratio(subject_list=self.bootstrap_list[i],level='image'))
            PRRLesionBoot.append(self.prediction_rejection_ratio(subject_list=self.bootstrap_list[i],level='lesion'))
            Dice_avgBoot.append(self.DiceLesion(subject_list=self.bootstrap_list[i])['mean'])
        fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up=self.computeFROC_bootstrap()
        self.bootstrapMetrics={
            # aggregates
            "auroc_bootstrap": np.mean(aurocBoot),
            "AP_bootstrap": np.mean(APBoot),
            "Ranking_bootstrap": (np.mean(aurocBoot)+np.mean(APBoot))/2,
            "aufroc_bootstrap":np.mean(aufrocBoot),
            "PRR-imageLevel_bootstrap":np.mean(PRRImageBoot),
            "PRR-LesionLevel_bootstrap":np.mean(PRRLesionBoot),
            "Dice_avg-LesionLevel_bootstrap":np.mean(Dice_avgBoot),
            "auroc-CI_bootstrap": self.compute_confidenceInterval(aurocBoot)['CI'],
            "AP-CI_bootstrap": self.compute_confidenceInterval(APBoot)['CI'],
            "aufroc-CI_bootstrap":self.compute_confidenceInterval(aufrocBoot)['CI'],
            "PRR-imageLevel-CI_bootstrap":self.compute_confidenceInterval(PRRImageBoot)['CI'],
            "PRR-LesionLevel-CI_bootstrap":self.compute_confidenceInterval(PRRLesionBoot)['CI'],
            "Dice_avg-LesionLevel-CI_bootstrap":self.compute_confidenceInterval(Dice_avgBoot)['CI'],
            "fps_bs_itp_bootstrap":fps_bs_itp,
            "sens_bs_mean_bootstrap":sens_bs_mean,
            "sens_bs_lb_bootstrap":sens_bs_lb,
            "sens_bs_up_bootstrap":sens_bs_up,
            "fps-1/8":sens_bs_mean[np.round(fps_bs_itp,3)==1/8][0],
            "fps-1/4":sens_bs_mean[np.round(fps_bs_itp,3)==1/4][0],
            "fps-1/2":sens_bs_mean[np.round(fps_bs_itp,3)==1/2][0],
            "fps-1":sens_bs_mean[np.round(fps_bs_itp,3)==1][0],
            "fps-2":sens_bs_mean[np.round(fps_bs_itp,3)==2][0],
            "fps-4":sens_bs_mean[np.round(fps_bs_itp,3)==4][0],
            #"fps-8":sens_bs_mean[np.round(fps_bs_itp,3)==8][0],
            "CPM":(sens_bs_mean[np.round(fps_bs_itp,3)==1/8][0]+sens_bs_mean[np.round(fps_bs_itp,3)==1/4][0]+
                          sens_bs_mean[np.round(fps_bs_itp,3)==1/2][0]+sens_bs_mean[np.round(fps_bs_itp,3)==1][0]+
                          sens_bs_mean[np.round(fps_bs_itp,3)==2][0]+sens_bs_mean[np.round(fps_bs_itp,3)==4][0])/6,
            "fps-1/8_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==1/8][0],sens_bs_up[np.round(fps_bs_itp,3)==1/8][0]],
            "fps-1/4_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==1/4][0],sens_bs_up[np.round(fps_bs_itp,3)==1/4][0]],
            "fps-1/2_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==1/2][0],sens_bs_up[np.round(fps_bs_itp,3)==1/2][0]],
            "fps-1_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==1][0],sens_bs_up[np.round(fps_bs_itp,3)==1][0]],
            "fps-2_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==2][0],sens_bs_up[np.round(fps_bs_itp,3)==2][0]],
            "fps-4_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==4][0],sens_bs_up[np.round(fps_bs_itp,3)==4][0]],
            #"fps-8_CI":[sens_bs_lb[np.round(fps_bs_itp,3)==8][0],sens_bs_up[np.round(fps_bs_itp,3)==8][0]],
            "CPM_CI":[(sens_bs_lb[np.round(fps_bs_itp,3)==1/8][0]+sens_bs_lb[np.round(fps_bs_itp,3)==1/4][0]+
                          sens_bs_lb[np.round(fps_bs_itp,3)==1/2][0]+sens_bs_lb[np.round(fps_bs_itp,3)==1][0]+
                          sens_bs_lb[np.round(fps_bs_itp,3)==2][0]+sens_bs_lb[np.round(fps_bs_itp,3)==4][0]+
                          sens_bs_lb[np.round(fps_bs_itp,3)==8][0])/8,(sens_bs_up[np.round(fps_bs_itp,3)==1/8][0]+
                          sens_bs_up[np.round(fps_bs_itp,3)==1/4][0]+sens_bs_up[np.round(fps_bs_itp,3)==1/2][0]+
                          sens_bs_up[np.round(fps_bs_itp,3)==1][0]+sens_bs_up[np.round(fps_bs_itp,3)==2][0])/6]
        }
        return self.bootstrapMetrics

    def full_dict(self):
        return {
            # aggregates
            "auroc": self.auroc,
            "AP": self.AP, #this based on lesion not per case
            "Ranking":self.score,
            "aufroc":self.aufroc,
            "PRR-imageLevel":self.prediction_rejection_ratio(subject_list=self.subject_list,level='image'),
            "PRR-LesionLevel":self.prediction_rejection_ratio(subject_list=self.subject_list,level='lesion'),
            "Dice_avg-LesionLevel":self.DiceLesion(subject_list=self.subject_list)['mean'],
            "num_cases": self.num_cases,
            "num_lesions": self.num_lesions,
            "picai_eval_version": self.version,

            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,
            "precision": self.precision,
            "recall": self.recall,
            'lesion_TPR': self.lesion_TPR,
            'lesion_FPR': self.lesion_FPR,
            "thresholds": self.thresholds,

            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def minimal_dict(self):
        return {
            # lesion-level results
            "lesion_results": self.lesion_results,
            "lesion_weight": self.lesion_weight,

            # case-level results
            "case_pred": self.case_pred,
            "case_target": self.case_target,
            "case_weight": self.case_weight,
        }

    def save(self, path: PathLike):
        """Save metrics to file (including aggregates)"""
        save_metrics(metrics=self.as_dict(), file_path=path)

    def save_full(self, path: PathLike):
        """Save metrics to file (including derived metrics)"""
        save_metrics(metrics=self.full_dict(), file_path=path)
    
    def save_fullBootstrap(self, path: PathLike):
        """Save metrics to file (including derived metrics)"""
        save_metrics(metrics=self.fullBootstrap(), file_path=path)

    def save_minimal(self, path: PathLike):
        """Save metrics to file (minimal required metrics)"""
        save_metrics(metrics=self.minimal_dict(), file_path=path)

    def load(self, path: PathLike):
        """Load metrics from file"""
        metrics = load_metrics(path)

        # parse metrics
        self.case_target = {idx: int(float(val)) for idx, val in metrics['case_target'].items()}
        self.case_pred = {idx: float(val) for idx, val in metrics['case_pred'].items()}
        self.case_weight = {idx: float(val) for idx, val in metrics['case_weight'].items()}
        self.lesion_weight = {idx: [float(val) for val in weights] for idx, weights in metrics['lesion_weight'].items()}
        self.lesion_results = {
            idx: [
                (int(float(is_lesion)), float(confidence), float(overlap))
                for (is_lesion, confidence, overlap) in lesion_results_case
            ]
            for idx, lesion_results_case in metrics['lesion_results'].items()
        }

    def __str__(self) -> str:
        return f"Metrics(auroc={self.auroc:.2%}, AP={self.AP:.2%}, {self.num_cases} cases, {self.num_lesions} lesions)"

    def __repr__(self) -> str:
        return self.__str__()
