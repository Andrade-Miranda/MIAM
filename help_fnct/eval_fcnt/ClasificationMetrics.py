import argparse
import json
import os
from pathlib import Path
import pandas as pd


from picai_eval import evaluate_folder
from report_guided_annotation import extract_lesion_candidates

from help_fnct.calibration.seg_calibration import Evaluate_Segcalibration_Folder
from util.visualizer import Picai_ResultsPlots


# acquire and parse input and output paths
parser = argparse.ArgumentParser(description='Command Line Arguments')
parser.add_argument("-i", "--input", type=str, required=True,
                    help="Path to folder with model predicitons (detection maps)")
parser.add_argument("-l", "--labels", type=str, required=True,
                    help="Path to folder with labels (defaults to input folder if unspecified)")
parser.add_argument("-l", "--labels", type=str, required=True,
                    help="Path to folder with labels (defaults to input folder if unspecified)")
parser.add_argument("-o", "--output", type=str, default="metrics.json",
                    help="Path to store metrics file, relative to the input folder.")
parser.add_argument("-s", "--subject_list", type=str, required=False,
                    help="Path to subject list, relative to the input folder. The subject list " +
                         "may be stored as json list, or json dictionary with 'subject_list' as parameter.")
parser.add_argument("--pred_extensions", type=str, nargs="+", required=False,
                    help="List of allowed file formats for detection maps." +
                         "Default: .npz, .npy, .nii.gz, .nii, .mha and .mhd")
parser.add_argument("--label_extensions", type=str, nargs="+", required=False,
                    help="List of allowed file formats for annotations." +
                         "Default: .nii.gz, .nii, .mha, .mhd, .npz and .npy")
parser.add_argument("--y_det_postprocess_func", type=str, required=False,
                    help="Post-processing function for detection maps. Available: `extract_lesion_candidates`")
parser.add_argument("--y_det_postprocess_kwargs", type=str, required=False,
                    help='Post-processing arguments for detection maps. E.g.: `{"threshold": "dynamic"}`')
args = parser.parse_args()

if args.labels is None:
    args.labels = args.input
args.output = os.path.join(args.input, args.output)
if args.subject_list is not None:
    args.subject_list = os.path.join(args.input, args.subject_list)
    with open(args.subject_list) as fp:
        args.subject_list = json.load(fp)
    if isinstance(args.subject_list, dict):
        args.subject_list = args.subject_list['subject_list']

if args.y_det_postprocess_func is not None:
    if args.y_det_postprocess_func == "extract_lesion_candidates":
        if args.y_det_postprocess_kwargs is None:
            args.y_det_postprocess_kwargs = {}
        else:
            args.y_det_postprocess_kwargs = json.loads(args.y_det_postprocess_kwargs)
        args.y_det_postprocess_func = lambda pred: extract_lesion_candidates(pred, **args.y_det_postprocess_kwargs)[0]
    else:
        raise ValueError(f"Received unsupported post-processing function: {args.y_det_postprocess_func}")

print(f"""
    PICAI Evaluation
    Model predictions path: {args.input}
    Labels path: {args.labels}
    Output Metrics Path: {args.output}
""")

assert os.path.exists(args.input), f"Input folder does not exist at {args.input}!"
assert os.path.exists(os.path.dirname(args.output)), f"Output folder does not exist at {args.output}!"


if isinstance(args.subject_list, (str, Path)):
    with open(args.subject_list) as fp:
        subject_list = json.load(fp)["data"]

# calculate metrics
clasif_metrics = evaluate_folder(y_det_dir=Path(args.input),
                          y_true_dir=Path(args.label),
                          subject_list=subject_list,
                          bootstrap = True,
                          y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred,threshold=0.5)[0],#lambda pred: extract_lesion_candidates(pred,threshold="dynamic")[0],
                          detection_map_postfixes=[""],
                          #min_overlap=0.01,
                          label_postfixes=[""],
                          num_parallel_calls=1,overlap_func = 'DSC'
                        )

# perform classification metrics without bootstrapping
clasif_metrics.save_full(Path(args.output) / "metrics.json")
#save precision recall
PRData=pd.DataFrame({'precision':clasif_metrics.precision,'recall':clasif_metrics.recall})
PRData_file = Path(Path(args.output)) / "PRData.csv"
PRData.to_csv(PRData_file, index=False)
#save ROC
ROCData=pd.DataFrame({'TPR':clasif_metrics.case_TPR,'FPR':clasif_metrics.case_FPR})
ROCData_file = Path(Path(args.output)) / "ROCData.csv"
ROCData.to_csv(ROCData_file, index=False)
#save FROC
FROCData=pd.DataFrame({'sensitivity':clasif_metrics.lesion_TPR,'fp_per_case':clasif_metrics.lesion_FPR})
FROCData_file = Path(Path(args.output)) / "FROCData.csv"
FROCData.to_csv(FROCData_file, index=False)

#sumary
sumary=pd.DataFrame({'AP':[clasif_metrics.AP],'Ranking':[clasif_metrics.score],'AUROC':[clasif_metrics.auroc],'FROC':[clasif_metrics.aufroc]})
sumary_file = Path(Path(args.output)) / "sumary.csv"
sumary.to_csv(sumary_file, index=False)
# save classification with bootstrapping
clasif_metrics.save_fullBootstrap(Path(args.output) / "metrics_bootstrap.json")

#calibration segmentation
calibration_values=Evaluate_Segcalibration_Folder(
                                gdth_path=args.label,
                                pred_path=args.input,
                                mask_path=args.postpro_dir,
                                outputpath=args.output)
#sumary calibration and misclassification
_,_,ECE,ADA_ECE,ks_test,prr,AUC,_,_,_,_,_=calibration_values
sumaryCalib=pd.DataFrame({'ECE':[ECE.item()],'ADA_ECE':[ADA_ECE.item()],'ks_test':[ks_test],'PRR-voxel':[prr],'AUC-voxel':[AUC]})
sumary_file = Path(Path(args.output)) / "sumary_calibration.csv"
sumaryCalib.to_csv(sumary_file, index=False)