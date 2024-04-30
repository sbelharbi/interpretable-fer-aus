#!/usr/bin/env bash 
 
# Activate your virtual environment.
# ----------------------------------------
CONDA_BASE=$(conda info --base) 
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate fer
# ----------------------------------------

# ==============================================================================
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

rootdir=/absolute/path/to/here  # e.g. /home/user_name/code/facial-expression-recognition


python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/CAM/align_atten_to_heatmap_True/AffectNet-resnet50-CAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/CAM/align_atten_to_heatmap_False/AffectNet-resnet50-CAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/deit_tscam_small_patch16_224/STD_CL/TSCAM/align_atten_to_heatmap_True/AffectNet-deit_tscam_small_patch16_224-TSCAM-GAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/deit_tscam_small_patch16_224/STD_CL/TSCAM/align_atten_to_heatmap_False/AffectNet-deit_tscam_small_patch16_224-TSCAM-GAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/apvit/STD_CL/APVIT/align_atten_to_heatmap_True/AffectNet-apvit-APVIT-NONE-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/apvit/STD_CL/APVIT/align_atten_to_heatmap_False/AffectNet-apvit-APVIT-NONE-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/GradCam/align_atten_to_heatmap_True/AffectNet-resnet50-GradCam-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/GradCam/align_atten_to_heatmap_False/AffectNet-resnet50-GradCam-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/GradCAMpp/align_atten_to_heatmap_True/AffectNet-resnet50-GradCAMpp-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/GradCAMpp/align_atten_to_heatmap_False/AffectNet-resnet50-GradCAMpp-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/LayerCAM/align_atten_to_heatmap_True/AffectNet-resnet50-LayerCAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/LayerCAM/align_atten_to_heatmap_False/AffectNet-resnet50-LayerCAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/WILDCAT/align_atten_to_heatmap_True/AffectNet-resnet50-WILDCAT-WildCatCLHead-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/WILDCAT/align_atten_to_heatmap_False/AffectNet-resnet50-WILDCAT-WildCatCLHead-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/ACoL/align_atten_to_heatmap_True/AffectNet-resnet50-ACoL-ACOL-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/ACoL/align_atten_to_heatmap_False/AffectNet-resnet50-ACoL-ACOL-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/PRM/align_atten_to_heatmap_True/AffectNet-resnet50-PRM-PRM-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/PRM/align_atten_to_heatmap_False/AffectNet-resnet50-PRM-PRM-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/ADL/align_atten_to_heatmap_True/AffectNet-resnet50-ADL-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/ADL/align_atten_to_heatmap_False/AffectNet-resnet50-ADL-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/CutMIX/align_atten_to_heatmap_True/AffectNet-resnet50-CutMIX-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/AffectNet/resnet50/STD_CL/CutMIX/align_atten_to_heatmap_False/AffectNet-resnet50-CutMIX-WGAP-cp_best-boxv2_False 

python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/CAM/align_atten_to_heatmap_True/RAF-DB-resnet50-CAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/CAM/align_atten_to_heatmap_False/RAF-DB-resnet50-CAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/deit_tscam_small_patch16_224/STD_CL/TSCAM/align_atten_to_heatmap_True/RAF-DB-deit_tscam_small_patch16_224-TSCAM-GAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/deit_tscam_small_patch16_224/STD_CL/TSCAM/align_atten_to_heatmap_False/RAF-DB-deit_tscam_small_patch16_224-TSCAM-GAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/apvit/STD_CL/APVIT/align_atten_to_heatmap_True/RAF-DB-apvit-APVIT-NONE-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/apvit/STD_CL/APVIT/align_atten_to_heatmap_False/RAF-DB-apvit-APVIT-NONE-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/GradCam/align_atten_to_heatmap_True/RAF-DB-resnet50-GradCam-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/GradCam/align_atten_to_heatmap_False/RAF-DB-resnet50-GradCam-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/GradCAMpp/align_atten_to_heatmap_True/RAF-DB-resnet50-GradCAMpp-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/GradCAMpp/align_atten_to_heatmap_False/RAF-DB-resnet50-GradCAMpp-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/LayerCAM/align_atten_to_heatmap_True/RAF-DB-resnet50-LayerCAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/LayerCAM/align_atten_to_heatmap_False/RAF-DB-resnet50-LayerCAM-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/WILDCAT/align_atten_to_heatmap_True/RAF-DB-resnet50-WILDCAT-WildCatCLHead-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/WILDCAT/align_atten_to_heatmap_False/RAF-DB-resnet50-WILDCAT-WildCatCLHead-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/ACoL/align_atten_to_heatmap_True/RAF-DB-resnet50-ACoL-ACOL-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/ACoL/align_atten_to_heatmap_False/RAF-DB-resnet50-ACoL-ACOL-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/PRM/align_atten_to_heatmap_True/RAF-DB-resnet50-PRM-PRM-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/PRM/align_atten_to_heatmap_False/RAF-DB-resnet50-PRM-PRM-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/ADL/align_atten_to_heatmap_True/RAF-DB-resnet50-ADL-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/ADL/align_atten_to_heatmap_False/RAF-DB-resnet50-ADL-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/CutMIX/align_atten_to_heatmap_True/RAF-DB-resnet50-CutMIX-WGAP-cp_best-boxv2_False 
python eval.py --cudaid 0 --split test --checkpoint_type best --exp_path $rootdir/shared-trained-models/FG_FER/RAF-DB/resnet50/STD_CL/CutMIX/align_atten_to_heatmap_False/RAF-DB-resnet50-CutMIX-WGAP-cp_best-boxv2_False 

