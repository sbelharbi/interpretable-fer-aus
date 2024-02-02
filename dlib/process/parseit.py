# Sel-contained-as-possible module handles parsing the input using argparse.
# handles seed, and initializes some modules for reproducibility.

import os
from os.path import dirname, abspath, join, basename, expanduser, normpath
import sys
import argparse
from copy import deepcopy
import warnings
import subprocess
import fnmatch
import glob
import shutil
import datetime as dt

import yaml
import munch
import numpy as np
import torch
import torch.distributed as dist

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.configure import config
from dlib.utils.tools import get_root_wsol_dataset
from dlib.utils import reproducibility

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import is_cc
from dlib.utils.shared import get_tag_device
from dlib.utils.tools import get_tag
from dlib.utils.tools import build_heatmap_folder
from dlib.utils.tools import get_heatmap_tag


def mkdir(fd):
    if not os.path.isdir(fd):
        os.makedirs(fd, exist_ok=True)


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def find_files_pattern(fd_in_, pattern_):
    assert os.path.exists(fd_in_), "Folder {} does not exist.".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def null_str(v):
    if v in [None, '', 'None']:
        return 'None'

    if isinstance(v, str):
        return v

    raise NotImplementedError(f"{v}, type: {type(v)}")


class Dict2Obj(object):
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def get_args(args: dict, eval: bool = False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--cudaid", type=str, default=None,
                        help="cuda id: '0,1,2,3'")
    parser.add_argument("--MYSEED", type=str, default=None, help="Seed.")
    parser.add_argument("--debug_subfolder", type=str, default=None,
                        help="Name of subfold for debugging. Default: ''.")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Name of the dataset.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes in the dataset.")

    parser.add_argument("--crop_size", type=int, default=None,
                        help="Crop size (int) of the patches in training.")
    parser.add_argument("--resize_size", type=int, default=None,
                        help="Resize image into this size before processing.")

    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Max epoch.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Training batch size (optimizer).")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Evaluation batch size.")
    parser.add_argument("--valid_freq_mb", type=float, default=None,
                        help="Minibatch frequency to validate: -1., ]0, 1[ .")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for dataloader multi-proc.")
    parser.add_argument("--exp_id", type=str, default=None, help="Exp id.")
    parser.add_argument("--verbose", type=str2bool, default=None,
                        help="Verbosity (bool).")
    parser.add_argument("--fd_exp", type=str, default=None,
                        help="Relative path to exp folder.")

    # ==========================================================================
    #                            STD_CL
    # ==========================================================================
    parser.add_argument("--std_cl_w_style", type=str, default=None,
                        help="STD_CL loss: how to use per-class weight for "
                             "unbalanced classes.")

    # ==========================================================================
    #                    LAYERWISE ATTENTION ALIGNMENT TO HEATMAP
    # ==========================================================================
    parser.add_argument("--align_atten_to_heatmap", type=str2bool, default=None,
                        help="Use or not the loss that aligns layerwise "
                             "attention to a heatmap.")
    parser.add_argument("--align_atten_to_heatmap_type_heatmap", type=str,
                        default=None,
                        help="Type of heatmap: landmarks, or action units.")
    parser.add_argument("--align_atten_to_heatmap_normalize", type=str2bool,
                        default=None,
                        help="Normalize or not the heatmap.")
    parser.add_argument("--align_atten_to_heatmap_aus_seg_full",
                        type=str2bool, default=None,
                        help="Use full image when landmarks fail. applied "
                             "only for HEATMAP_AUNITS_LEARNED_SEG.")
    parser.add_argument("--align_atten_to_heatmap_jaw", type=str2bool,
                        default=None,
                        help="Show jaw or not for heatmap from landmarks.")
    parser.add_argument("--align_atten_to_heatmap_elb", type=str2bool,
                        default=None,
                        help="Use or not ELB for alignment when using cosine.")
    parser.add_argument("--align_atten_to_heatmap_lndmk_variance", type=float,
                        default=None,
                        help="Variance of the Gaussian that"
                             " generate a heatmap from a landmark.")
    parser.add_argument("--align_atten_to_heatmap_layers", type=str,
                        default=None,
                        help="id layer (int) where to apply this loss. Many "
                             "layers can be indicated. Must be separated by "
                             "'-'.")
    parser.add_argument("--align_atten_to_heatmap_align_type", type=str,
                        default=None,
                        help="How to estimate the layerwise attention from "
                             "the feature maps.")
    parser.add_argument("--align_atten_to_heatmap_norm_att", type=str,
                        default=None,
                        help="How to normalize the attention.")
    parser.add_argument("--align_atten_to_heatmap_p", type=float,
                        default=None,
                        help="Percentage of feature maps to randomly select "
                             "when using random attention estimation. ]0, 1.]")
    parser.add_argument("--align_atten_to_heatmap_q", type=float,
                        default=None,
                        help="Percentage of feature maps to select "
                             "when using a percentage of maps to estimate "
                             "attention estimation. ]0, 1.]. we use the same "
                             "maps always.")
    parser.add_argument("--align_atten_to_heatmap_loss", type=str,
                        default=None,
                        help="Loss how to align attention and a heatmap.")
    parser.add_argument("--align_atten_to_heatmap_lambda", type=float,
                        default=None,
                        help="Lambda of this loss.")
    parser.add_argument("--align_atten_to_heatmap_scale_to", type=str,
                        default=None,
                        help="How to scale tensors.")
    parser.add_argument("--align_atten_to_heatmap_start_ep", type=int,
                        default=None,
                        help="Epoch when to start applying this loss.")
    parser.add_argument("--align_atten_to_heatmap_end_ep", type=int,
                        default=None,
                        help="Epoch when to stop applying this loss.")
    parser.add_argument("--align_atten_to_heatmap_folder", type=str,
                        default=None,
                        help="Path to the precomputed heatmaps.")
    parser.add_argument("--align_atten_to_heatmap_use_precomputed",
                        type=str2bool, default=None,
                        help="Allow or not the usage of precomputed heatmaps.")
    parser.add_argument("--align_atten_to_heatmap_use_self_atten",
                        type=str2bool, default=None,
                        help="Mask or not layer features via self-attention at "
                             "layer where the "
                             "alignment is done.")

    # random data augmentation using heatmap over input image. heatmap is
    # estimated either from facial landmarks or facial action units.
    # ==========================================================================
    #                     MASKOUTFER DATA AUGMENTATION
    #                           TRAIN / EVAL SET
    # ==========================================================================

    # Train
    parser.add_argument("--train_daug_mask_img_heatmap", type=str2bool, default=None,
                        help="Apply or not random data augmentation over "
                             "input image using heatmap of facial landmarks "
                             "or facial action units. Over a set for training.")
    parser.add_argument("--train_daug_mask_img_heatmap_type", type=str,
                        default=None,
                        help="type of heat map: action units or landmarks.")
    parser.add_argument("--train_daug_mask_img_heatmap_bg_filler", type=str,
                        default=None,
                        help="how to fill the background.")
    parser.add_argument("--train_daug_mask_img_heatmap_p", type=float,
                        default=None,
                        help="Probability to apply this transformation [0, 1].")
    parser.add_argument("--train_daug_mask_img_heatmap_gauss_sigma", type=float,
                        default=None,
                        help="Variance of the Gaussian blur for background "
                             "filler if gaussian. > 0.")
    parser.add_argument("--train_daug_mask_img_heatmap_dilation", type=str2bool,
                        default=None,
                        help="Apply or not dilation over roi estimated from "
                             "the heatmap.")
    parser.add_argument("--train_daug_mask_img_heatmap_radius", type=int,
                        default=None,
                        help="Radius of the structural element (disk). > 0")
    parser.add_argument("--train_daug_mask_img_heatmap_normalize", type=str2bool,
                        default=None,
                        help="Normalize or not the heatmap.")
    parser.add_argument("--train_daug_mask_img_heatmap_aus_seg_full",
                        type=str2bool, default=None,
                        help="Use full image when landmarks fail. applied "
                             "only for HEATMAP_AUNITS_LEARNED_SEG.")
    parser.add_argument("--train_daug_mask_img_heatmap_jaw", type=str2bool,
                        default=None,
                        help="Show jaw or not for heatmap from landmarks.")
    parser.add_argument("--train_daug_mask_img_heatmap_lndmk_variance", type=float,
                        default=None,
                        help="Variance of the Gaussian that"
                             " generate a heatmap from a landmark.")
    parser.add_argument("--train_daug_mask_img_heatmap_folder", type=str,
                        default=None,
                        help="Path to the precomputed heatmaps.")
    parser.add_argument("--train_daug_mask_img_heatmap_use_precomputed",
                        type=str2bool, default=None,
                        help="Allow or not the usage of precomputed heatmaps.")

    # Eval
    parser.add_argument("--eval_daug_mask_img_heatmap", type=str2bool,
                        default=None,
                        help="Apply or not fixed data augmentation over "
                             "input image using heatmap of facial landmarks "
                             "or facial action units. Over a set for a set "
                             "for evaluation.")
    parser.add_argument("--eval_daug_mask_img_heatmap_type", type=str,
                        default=None,
                        help="type of heat map: action units or landmarks.")
    parser.add_argument("--eval_daug_mask_img_heatmap_bg_filler", type=str,
                        default=None,
                        help="how to fill the background.")
    parser.add_argument("--eval_daug_mask_img_heatmap_gauss_sigma", type=float,
                        default=None,
                        help="Variance of the Gaussian blur for background "
                             "filler if gaussian. > 0.")
    parser.add_argument("--eval_daug_mask_img_heatmap_dilation", type=str2bool,
                        default=None,
                        help="Apply or not dilation over roi estimated from "
                             "the heatmap.")
    parser.add_argument("--eval_daug_mask_img_heatmap_radius", type=int,
                        default=None,
                        help="Radius of the structural element (disk). > 0")
    parser.add_argument("--eval_daug_mask_img_heatmap_normalize",
                        type=str2bool,
                        default=None,
                        help="Normalize or not the heatmap.")
    parser.add_argument("--eval_daug_mask_img_heatmap_aus_seg_full",
                        type=str2bool, default=None,
                        help="Use full image when landmarks fail. applied "
                             "only for HEATMAP_AUNITS_LEARNED_SEG.")
    parser.add_argument("--eval_daug_mask_img_heatmap_jaw", type=str2bool,
                        default=None,
                        help="Show jaw or not for heatmap from landmarks.")
    parser.add_argument("--eval_daug_mask_img_heatmap_lndmk_variance",
                        type=float,
                        default=None,
                        help="Variance of the Gaussian that"
                             " generate a heatmap from a landmark.")
    parser.add_argument("--eval_daug_mask_img_heatmap_folder", type=str,
                        default=None,
                        help="Path to the precomputed heatmaps.")
    parser.add_argument("--eval_daug_mask_img_heatmap_use_precomputed",
                        type=str2bool, default=None,
                        help="Allow or not the usage of precomputed heatmaps.")

    # orthog. features
    parser.add_argument("--free_orth_ft", type=str2bool, default=None,
                        help="Apply or not free orthog. ft loss.")
    parser.add_argument("--free_orth_ft_lambda", type=float, default=None,
                        help="free_orth_ft: lambda loss.")
    parser.add_argument("--free_orth_ft_start_ep", type=int, default=None,
                        help="free_orth_ft: start epoch.")
    parser.add_argument("--free_orth_ft_end_ep", type=int, default=None,
                        help="free_orth_ft: end epoch.")
    parser.add_argument("--free_orth_ft_elb", type=str2bool, default=None,
                        help="free_orth_ft: use or not ELB.")
    parser.add_argument("--free_orth_ft_layers", type=str, default=None,
                        help="free_orth_ft: layers to be used.")
    parser.add_argument("--free_orth_ft_same_cl", type=str2bool, default=None,
                        help="free_orth_ft: apply to samples with same "
                             "classes.")
    parser.add_argument("--free_orth_ft_diff_cl", type=str2bool, default=None,
                        help="free_orth_ft: apply to samples with different "
                             "classes.")

    parser.add_argument("--guid_orth_ft", type=str2bool, default=None,
                        help="Apply or not free orthog. ft loss.")
    parser.add_argument("--guid_orth_ft_lambda", type=float, default=None,
                        help="guid_orth_ft: lambda loss.")
    parser.add_argument("--guid_orth_ft_start_ep", type=int, default=None,
                        help="guid_orth_ft: start epoch.")
    parser.add_argument("--guid_orth_ft_end_ep", type=int, default=None,
                        help="guid_orth_ft: end epoch.")
    parser.add_argument("--guid_orth_ft_elb", type=str2bool, default=None,
                        help="guid_orth_ft: use or not ELB.")
    parser.add_argument("--guid_orth_ft_layers", type=str, default=None,
                        help="guid_orth_ft: layers to be used.")

    # high entropy over probability.
    parser.add_argument("--high_entropy", type=str2bool, default=None,
                        help="High entropy regularizer over probabilities: "
                             "apply or not.")
    parser.add_argument("--high_entropy_type", type=str, default=None,
                        help="high_entropy: type.")
    parser.add_argument("--high_entropy_lambda", type=float, default=None,
                        help="high_entropy: lambda loss.")
    parser.add_argument("--high_entropy_a", type=float, default=None,
                        help="high_entropy: alpha ]0, 1[.")
    parser.add_argument("--high_entropy_start_ep", type=int, default=None,
                        help="high_entropy: start epoch.")
    parser.add_argument("--high_entropy_end_ep", type=int, default=None,
                        help="high_entropy: end epoch.")

    # constraint on score
    parser.add_argument("--con_scores", type=str2bool, default=None,
                        help="Apply or not constraint over class scores "
                             "loss.")
    parser.add_argument("--con_scores_lambda", type=float, default=None,
                        help="Con. scores: lambda loss.")
    parser.add_argument("--con_scores_min", type=float, default=None,
                        help="Con. scores: minimum difference (>= 0).")
    parser.add_argument("--con_scores_start_ep", type=int, default=None,
                        help="Con. scores: start epoch.")
    parser.add_argument("--con_scores_end_ep", type=int, default=None,
                        help="Con. scores: end epoch.")

    # train data sampler
    parser.add_argument("--data_weighted_sampler", type=str2bool, default=None,
                        help="Train data sampler: weighted or not.")
    parser.add_argument("--data_weighted_sampler_w", type=str, default=None,
                        help="Train data sampler: weighting style.")
    parser.add_argument("--data_weighted_sampler_per_cl", type=str,
                        default=None,
                        help="Train data sampler: style to sample balanced "
                             "samples PER-CLASS.")

    # self-cost sensitive loss.
    parser.add_argument("--s_cost_s", type=str2bool, default=None,
                        help="s_cost_s: use or not self-cost sensitive loss.")
    parser.add_argument("--s_cost_s_lambda", type=float, default=None,
                        help="s_cost_s: lambda.")
    parser.add_argument("--s_cost_s_start_ep", type=int, default=None,
                        help="s_cost_s: start epoch.")
    parser.add_argument("--s_cost_s_end_ep", type=int, default=None,
                        help="s_cost_s: end epoch.")
    parser.add_argument("--s_cost_s_apply_to", type=str, default=None,
                        help="s_cost_s: logits or probs.")
    parser.add_argument("--s_cost_s_norm", type=str, default=None,
                        help="s_cost_s: how to normalize scores.")
    parser.add_argument("--s_cost_s_confusion_func", type=str, default=None,
                        help="s_cost_s: confusion function.")
    parser.add_argument("--s_cost_s_topk", type=int, default=None,
                        help="s_cost_s: topk scores.")
    parser.add_argument("--s_cost_s_reduction", type=str, default=None,
                        help="s_cost_s: loss reduction per sample.")

    # Cross-entropy classification
    parser.add_argument("--ce", type=str2bool, default=None,
                        help="CE: use or not cross-entropy for classification.")
    parser.add_argument("--ce_lambda", type=float, default=None,
                        help="CE: lambda.")
    parser.add_argument("--ce_start_ep", type=int, default=None,
                        help="CE: start epoch.")
    parser.add_argument("--ce_end_ep", type=int, default=None,
                        help="CE: end epoch.")
    parser.add_argument("--ce_label_smoothing", type=float, default=None,
                        help="CE: label smoothing.")

    # do_segmentation over action units.
    parser.add_argument("--aus_seg", type=str2bool, default=None,
                        help="Apply or not segmentation loss (BCE) over "
                             "action units output.")
    parser.add_argument("--aus_seg_lambda", type=float, default=None,
                        help="aus_seg: lambda.")
    parser.add_argument("--aus_seg_start_ep", type=int, default=None,
                        help="aus_seg: start epoch.")
    parser.add_argument("--aus_seg_end_ep", type=int, default=None,
                        help="aus_seg: end epoch.")
    parser.add_argument("--aus_seg_heatmap_type", type=str, default=None,
                        help="Heatmap type.")
    parser.add_argument("--aus_seg_normalize", type=str2bool, default=None,
                        help="Normalize or not the heatmap.")
    parser.add_argument("--aus_seg_aus_seg_full",
                        type=str2bool, default=None,
                        help="Use full image when landmarks fail. applied "
                             "only for HEATMAP_AUNITS_LEARNED_SEG.")
    parser.add_argument("--aus_seg_jaw", type=str2bool, default=None,
                        help="Show jaw or not for heatmap from landmarks.")
    parser.add_argument("--aus_seg_lndmk_variance", type=float, default=None,
                        help="Variance of the Gaussian that"
                             " generate a heatmap from a landmark.")
    parser.add_argument("--aus_seg_folder", type=str, default=None,
                        help="Path to the precomputed heatmaps.")
    parser.add_argument("--aus_seg_use_precomputed", type=str2bool,
                        default=None,
                        help="Allow or not the usage of precomputed heatmaps.")

    # MSE classification
    parser.add_argument("--mse", type=str2bool, default=None,
                        help="MSE: use or not MSE for classification.")
    parser.add_argument("--mse_lambda", type=float, default=None,
                        help="MSE: lambda.")
    parser.add_argument("--mse_start_ep", type=int, default=None,
                        help="MSE: start epoch.")
    parser.add_argument("--mse_end_ep", type=int, default=None,
                        help="MSE: end epoch.")

    # attention size loss
    parser.add_argument("--att_sz", type=str2bool, default=None,
                        help="att_sz: use or not attention size constraint.")
    parser.add_argument("--att_sz_lambda", type=float, default=None,
                        help="att_sz: lambda.")
    parser.add_argument("--att_sz_start_ep", type=int, default=None,
                        help="att_sz: start epoch.")
    parser.add_argument("--att_sz_end_ep", type=int, default=None,
                        help="att_sz: end epoch.")
    parser.add_argument("--att_sz_bounds", type=str, default=None,
                        help="att_sz: size bounds.")

    # attention entropy loss
    parser.add_argument("--att_ent_sz", type=str2bool, default=None,
                        help="att_ent_sz: use or not attention entropy loss.")
    parser.add_argument("--att_ent_lambda", type=float, default=None,
                        help="att_ent_sz: lambda.")
    parser.add_argument("--att_ent_start_ep", type=int, default=None,
                        help="att_ent_sz: start epoch.")
    parser.add_argument("--att_ent_end_ep", type=int, default=None,
                        help="att_ent_sz: end epoch.")

    # MAE classification
    parser.add_argument("--mae", type=str2bool, default=None,
                        help="MAE: use or not MAE for classification.")
    parser.add_argument("--mae_lambda", type=float, default=None,
                        help="MAE: lambda.")
    parser.add_argument("--mae_start_ep", type=int, default=None,
                        help="MAE: start epoch.")
    parser.add_argument("--mae_end_ep", type=int, default=None,
                        help="MAE: end epoch.")

    # weights sparsity
    parser.add_argument("--w_sparsity", type=str2bool, default=None,
                        help="w_sparsity: use/not weights sparsity loss.")
    parser.add_argument("--w_sparsity_lambda", type=float, default=None,
                        help="w_sparsity: lambda.")
    parser.add_argument("--w_sparsity_start_ep", type=int, default=None,
                        help="w_sparsity: start epoch.")
    parser.add_argument("--w_sparsity_end_ep", type=int, default=None,
                        help="w_sparsity: end epoch.")

    # Orthogonality for the weight of the linear classifier (output net)
    parser.add_argument("--ortho_lw", type=str2bool, default=None,
                        help="ortho_lw: use/not weights sparsity loss.")
    parser.add_argument("--ortho_lw_method", type=str, default=None,
                        help="ortho_lw: method type of orthogonality.")
    parser.add_argument("--ortho_lw_spec_iter", type=int, default=None,
                        help="ortho_lw: number of iterations for spectral "
                             "norm estimation for the method SRIP.")
    parser.add_argument("--ortho_lw_lambda", type=float, default=None,
                        help="ortho_lw: lambda.")
    parser.add_argument("--ortho_lw_start_ep", type=int, default=None,
                        help="ortho_lw: start epoch.")
    parser.add_argument("--ortho_lw_end_ep", type=int, default=None,
                        help="ortho_lw: end epoch.")

    # linear features sparsity: input of the linea layer at the net output.
    parser.add_argument("--sparse_lf", type=str2bool, default=None,
                        help="sparse_lf: use/not sparsity loss over linear "
                             "feature.")
    parser.add_argument("--sparse_lf_lambda", type=float, default=None,
                        help="sparse_lf: lambda.")
    parser.add_argument("--sparse_lf_start_ep", type=int, default=None,
                        help="sparse_lf: start epoch.")
    parser.add_argument("--sparse_lf_end_ep", type=int, default=None,
                        help="sparse_lf: end epoch.")
    parser.add_argument("--sparse_lf_method", type=str, default=None,
                        help="sparse_lf: method name.")
    parser.add_argument("--sparse_lf_p", type=float, default=None,
                        help="sparse_lf: p.")
    parser.add_argument("--sparse_lf_c", type=float, default=None,
                        help="sparse_lf: c.")
    parser.add_argument("--sparse_lf_use_elb", type=str2bool, default=None,
                        help="sparse_lf: use of ELB.")
    parser.add_argument("--sparse_lf_average_it", type=str2bool, default=None,
                        help="sparse_lf: average or not.")

    # linear weight sparsity: weight of the linea layer at the net output.
    parser.add_argument("--sparse_lw", type=str2bool, default=None,
                        help="sparse_lw: use/not sparsity loss over linear "
                             "weight.")
    parser.add_argument("--sparse_lw_lambda", type=float, default=None,
                        help="sparse_lw: lambda.")
    parser.add_argument("--sparse_lw_start_ep", type=int, default=None,
                        help="sparse_lw: start epoch.")
    parser.add_argument("--sparse_lw_end_ep", type=int, default=None,
                        help="sparse_lw: end epoch.")
    parser.add_argument("--sparse_lw_method", type=str, default=None,
                        help="sparse_lw: method name.")
    parser.add_argument("--sparse_lw_p", type=float, default=None,
                        help="sparse_lw: p.")
    parser.add_argument("--sparse_lw_c", type=float, default=None,
                        help="sparse_lw: c.")
    parser.add_argument("--sparse_lw_use_elb", type=str2bool, default=None,
                        help="sparse_lw: use of ELB.")
    parser.add_argument("--sparse_lw_average_it", type=str2bool, default=None,
                        help="sparse_lw: average or not.")

    # Sparse APVIT attention..
    parser.add_argument("--sparse_at", type=str2bool, default=None,
                        help="sparse_at: use/not sparsity loss over APVIT "
                             "attention.")
    parser.add_argument("--sparse_at_lambda", type=float, default=None,
                        help="sparse_at: lambda.")
    parser.add_argument("--sparse_at_start_ep", type=int, default=None,
                        help="sparse_at: start epoch.")
    parser.add_argument("--sparse_at_end_ep", type=int, default=None,
                        help="sparse_at: end epoch.")
    parser.add_argument("--sparse_at_method", type=str, default=None,
                        help="sparse_at: method name.")
    parser.add_argument("--sparse_at_p", type=float, default=None,
                        help="sparse_at: p.")
    parser.add_argument("--sparse_at_c", type=float, default=None,
                        help="sparse_at: c.")
    parser.add_argument("--sparse_at_use_elb", type=str2bool, default=None,
                        help="sparse_at: use of ELB.")
    parser.add_argument("--sparse_at_average_it", type=str2bool, default=None,
                        help="sparse_at: average or not.")


    # Multi-class focal loss for classification
    parser.add_argument("--mtl_focal", type=str2bool, default=None,
                        help="Multi-class focal loss: yes/no.")
    parser.add_argument("--mtl_focal_lambda", type=float, default=None,
                        help="MTL-focal: lambda.")
    parser.add_argument("--mtl_focal_start_ep", type=int, default=None,
                        help="MTL-focal: start epoch.")
    parser.add_argument("--mtl_focal_end_ep", type=int, default=None,
                        help="MTL-focal: end epoch.")
    parser.add_argument("--mtl_focal_alpha", type=float, default=None,
                        help="MTL-focal: Alpha.")
    parser.add_argument("--mtl_focal_gamma", type=float, default=None,
                        help="MTL-focal: Gamma.")

    # Curriculum learning
    parser.add_argument("--curriculum_l", type=str2bool, default=None,
                        help="curriculum_l: use/not curriculum learning.")
    parser.add_argument("--curriculum_l_type", type=str, default=None,
                        help="curriculum_l: type of curriculum learning.")
    parser.add_argument("--curriculum_l_epoch_rate", type=int, default=None,
                        help="curriculum_l: epoch rate to update CL.")
    parser.add_argument("--curriculum_l_cl_class_order", type=str, default=None,
                        help="curriculum_l: curriculum of classes.")


    # ======================================================================
    #                      WSOL
    # ======================================================================
    parser.add_argument('--data_root', default=None,
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default=None)
    parser.add_argument('--mask_root', default=None,
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool,
                        default=None,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=None,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')
    parser.add_argument('--cam_curve_interval', type=float, default=None,
                        help='CAM curve interval')
    parser.add_argument('--multi_contour_eval', type=str2bool, default=None)
    parser.add_argument('--multi_iou_eval', type=str2bool, default=None)
    parser.add_argument('--box_v2_metric', type=str2bool, default=None)
    parser.add_argument('--eval_checkpoint_type', type=str, default=None)
    # ======================================================================
    #                      OPTIMIZER
    # ======================================================================
    # opt0: optimizer for the model.
    parser.add_argument("--opt__name_optimizer", type=str, default=None,
                        help="Name of the optimizer 'sgd', 'adam'.")
    parser.add_argument("--opt__lr", type=float, default=None,
                        help="Learning rate (optimizer)")
    parser.add_argument("--opt__momentum", type=float, default=None,
                        help="Momentum (optimizer)")
    parser.add_argument("--opt__dampening", type=float, default=None,
                        help="Dampening for Momentum (optimizer)")
    parser.add_argument("--opt__nesterov", type=str2bool, default=None,
                        help="Nesterov or not for Momentum (optimizer)")
    parser.add_argument("--opt__weight_decay", type=float, default=None,
                        help="Weight decay (optimizer)")
    parser.add_argument("--opt__beta1", type=float, default=None,
                        help="Beta1 for adam (optimizer)")
    parser.add_argument("--opt__beta2", type=float, default=None,
                        help="Beta2 for adam (optimizer)")
    parser.add_argument("--opt__eps_adam", type=float, default=None,
                        help="eps for adam (optimizer)")
    parser.add_argument("--opt__amsgrad", type=str2bool, default=None,
                        help="amsgrad for adam (optimizer)")
    parser.add_argument("--opt__lr_scheduler", type=str2bool, default=None,
                        help="Whether to use or not a lr scheduler")
    parser.add_argument("--opt__name_lr_scheduler", type=str, default=None,
                        help="Name of the lr scheduler")
    parser.add_argument("--opt__gamma", type=float, default=None,
                        help="Gamma of the lr scheduler. (mystep)")
    parser.add_argument("--opt__last_epoch", type=int, default=None,
                        help="Index last epoch to stop adjust LR(mystep)")
    parser.add_argument("--opt__min_lr", type=float, default=None,
                        help="Minimum allowed value for lr.")
    parser.add_argument("--opt__t_max", type=float, default=None,
                        help="T_max, maximum epochs to restart. (cosine)")
    parser.add_argument("--opt__step_size", type=int, default=None,
                        help="Step size for lr scheduler.")
    parser.add_argument("--opt__lr_classifier_ratio", type=float, default=None,
                        help="Multiplicative factor for the classifier head "
                             "learning rate.")
    parser.add_argument("--opt__clipgrad", type=float, default=None,
                        help="Gradient clip norm.")

    # ======================================================================
    #                              MODEL
    # ======================================================================
    parser.add_argument("--arch", type=str, default=None,
                        help="model's name.")
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Name of the backbone")
    parser.add_argument("--large_maps", type=str2bool, default=None,
                        help="Large or small output maps.")
    parser.add_argument("--in_channels", type=int, default=None,
                        help="Input channels number.")
    parser.add_argument("--strict", type=str2bool, default=None,
                        help="strict mode for loading weights.")
    parser.add_argument("--encoder_weights", type=str, default=None,
                        help="Pre-trained weights.")
    parser.add_argument("--path_pre_trained", type=str, default=None,
                        help="Absolute/relative path to file of weights.")
    parser.add_argument("--support_background", type=str2bool, default=None,
                        help="use or not 1 extra plan for background cams.")
    parser.add_argument("--scale_in", type=float, default=None,
                        help="How much to scale the input.")

    parser.add_argument("--freeze_cl", type=str2bool, default=None,
                        help="whether or not to freeze the classifier.")
    parser.add_argument("--folder_pre_trained_cl", type=str, default=None,
                        help="NAME of folder containing classifier's "
                             "weights.")
    parser.add_argument("--spatial_dropout", type=float, default=None,
                        help="2d dropout at the last feature layer ([0, 1.]).")

    parser.add_argument("--apvit_k", type=int, default=None,
                        help="k for apvit.")
    parser.add_argument("--apvit_r", type=float, default=None,
                        help="r for apvit.")
    parser.add_argument("--apvit_attn_method", type=str, default=None,
                        help="apvit: attention type.")
    parser.add_argument("--apvit_normalize_att", type=str2bool, default=None,
                        help="apvit: normalize/not attention.")
    parser.add_argument("--apvit_apply_self_att", type=str2bool, default=None,
                        help="apvit: apply/not attention to features.")
    parser.add_argument("--apvit_hid_att_dim", type=int, default=None,
                        help="apvit: size of hidden layer for attention.")

    parser.add_argument("--do_segmentation", type=str2bool, default=None,
                        help="If true, the classifier encoder is equipped "
                             "with a simple segmentation head. It is used to "
                             "learnt o predict action units heatmaps.")

    # ======================================================================
    #                    CLASSIFICATION SPATIAL POOLING
    # ======================================================================
    parser.add_argument("--method", type=str, default=None,
                        help="Name of method.")
    parser.add_argument("--spatial_pooling", type=str, default=None,
                        help="Name of spatial pooling for classification.")
    # ======================================================================
    #                        WILDCAT POOLING
    # ======================================================================

    parser.add_argument("--wc_alpha", type=float, default=None,
                        help="Alpha (classifier, wildcat)")
    parser.add_argument("--wc_kmax", type=float, default=None,
                        help="Kmax (classifier, wildcat)")
    parser.add_argument("--wc_kmin", type=float, default=None,
                        help="Kmin (classifier, wildcat)")
    parser.add_argument("--wc_dropout", type=float, default=None,
                        help="Dropout (classifier, wildcat)")
    parser.add_argument("--wc_modalities", type=int, default=None,
                        help="Number of modalities (classifier, wildcat)")

    parser.add_argument("--lse_r", type=float, default=None,
                        help="LSE r pooling.")

    # cutmix
    parser.add_argument('--cutmix_beta', type=float, default=None,
                        help='CUTMIX beta.')
    parser.add_argument('--cutmix_prob', type=float, default=None,
                        help='CUTMIX. probablity to do it over a minibatch.')

    # ACoL
    parser.add_argument('--acol_drop_threshold', type=float, default=None,
                        help='Float. threshold for ACOL.')

    # PRM
    parser.add_argument('--prm_ks', type=int, default=None,
                        help='PRM: kernel size.')
    parser.add_argument('--prm_st', type=int, default=None,
                        help='PRM: kernel stride.')

    # ADL
    parser.add_argument('--adl_drop_rate', type=float, default=None,
                        help='Float.drop-rate for ADL.')
    parser.add_argument('--adl_drop_threshold', type=float, default=None,
                        help='Float. threshold for ADL.')

    # ======================================================================
    #                        CLASSIFICATION HEAD
    # ======================================================================
    parser.add_argument("--dense_dims", type=null_str, default=None,
                        help="dense layers at the classification head. max 2. "
                             "eg. '1024-512'.")
    parser.add_argument("--dense_dropout", type=float, default=None,
                        help="Dropout at dense layers at the classification "
                             "head.")

    # ======================================================================
    #                         EXTRA - MODE
    # ======================================================================

    parser.add_argument("--seg_mode", type=str, default=None,
                        help="Segmentation mode.")
    parser.add_argument("--task", type=str, default=None,
                        help="Type of the task.")
    parser.add_argument("--master_selection_metric", type=str, default=None,
                        help="Model selection metric over validation set.")
    parser.add_argument("--multi_label_flag", type=str2bool, default=None,
                        help="Whether the dataset is multi-label.")
    # ======================================================================
    #                         ELB
    # ======================================================================
    parser.add_argument("--elb_init_t", type=float, default=None,
                        help="Init t for elb.")
    parser.add_argument("--elb_max_t", type=float, default=None,
                        help="Max t for elb.")
    parser.add_argument("--elb_mulcoef", type=float, default=None,
                        help="Multi. coef. for elb..")

    # ======================================================================
    #                         CONSTRAINTS
    # ======================================================================
    parser.add_argument("--crf_fc", type=str2bool, default=None,
                        help="CRF over fcams flag.")
    parser.add_argument("--crf_lambda", type=float, default=None,
                        help="Lambda for crf flag.")
    parser.add_argument("--crf_sigma_rgb", type=float, default=None,
                        help="sigma rgb of crf flag.")
    parser.add_argument("--crf_sigma_xy", type=float, default=None,
                        help="sigma xy for crf flag.")
    parser.add_argument("--crf_scale", type=float, default=None,
                        help="scale factor for crf flag.")
    parser.add_argument("--crf_start_ep", type=int, default=None,
                        help="epoch start crf loss.")
    parser.add_argument("--crf_end_ep", type=int, default=None,
                        help="epoch end crf loss. use -1 for end training.")

    parser.add_argument("--entropy_fc", type=str2bool, default=None,
                        help="Entropy over fcams flag.")
    parser.add_argument("--entropy_fc_lambda", type=float, default=None,
                        help="lambda for entropy over fcams flag.")

    parser.add_argument("--max_sizepos_fc", type=str2bool, default=None,
                        help="Max size pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_lambda", type=float, default=None,
                        help="lambda for max size low pos fcams flag.")
    parser.add_argument("--max_sizepos_fc_start_ep", type=int, default=None,
                        help="epoch start maxsz loss.")
    parser.add_argument("--max_sizepos_fc_end_ep", type=int, default=None,
                        help="epoch end maxsz. -1 for end training.")

    # ordinal classification:
    # - mean
    parser.add_argument("--oc_mean", type=str2bool, default=None,
                        help="Use/not ordinal mean loss.")
    parser.add_argument("--oc_mean_lambda", type=float, default=None,
                        help="lambda for oc_mean.")
    parser.add_argument("--oc_mean_epsilon", type=float, default=None,
                        help="Epsilon for oc_mean.")
    parser.add_argument("--oc_mean_elb", type=str2bool, default=None,
                        help="Use/not ELB for oc_mean.")
    parser.add_argument("--oc_mean_start_ep", type=int, default=None,
                        help="epoch start oc_mean loss.")
    parser.add_argument("--oc_mean_end_ep", type=int, default=None,
                        help="epoch end oc_mean. -1 for end training.")

    # - variance
    parser.add_argument("--oc_var", type=str2bool, default=None,
                        help="Use/not ordinal variance loss.")
    parser.add_argument("--oc_var_lambda", type=float, default=None,
                        help="lambda for oc_var.")
    parser.add_argument("--oc_var_epsilon", type=float, default=None,
                        help="Epsilon for oc_var.")
    parser.add_argument("--oc_var_elb", type=str2bool, default=None,
                        help="Use/not ELB for oc_var.")
    parser.add_argument("--oc_var_start_ep", type=int, default=None,
                        help="epoch start oc_var loss.")
    parser.add_argument("--oc_var_end_ep", type=int, default=None,
                        help="epoch end oc_var. -1 for end training.")

    # - unimodality via inequalities
    parser.add_argument("--oc_unim_inq", type=str2bool, default=None,
                        help="Use/not ordinal variance via unimodality via "
                             "inequalities loss.")
    parser.add_argument("--oc_unim_inq_lambda", type=float, default=None,
                        help="lambda for oc_unim_inq.")
    parser.add_argument("--oc_unim_inq_type", type=str, default=None,
                        help="Epsilon for oc_unim_inq.")
    parser.add_argument("--oc_unim_inq_start_ep", type=int, default=None,
                        help="epoch start oc_unim_inq loss.")
    parser.add_argument("--oc_unim_inq_end_ep", type=int, default=None,
                        help="epoch end oc_unim_inq. -1 for end training.")


    parser.add_argument("--im_rec", type=str2bool, default=None,
                        help="image reconstruction flag.")
    parser.add_argument("--im_rec_lambda", type=float, default=None,
                        help="Lambda for image reconstruction.")
    parser.add_argument("--im_rec_elb", type=str2bool, default=None,
                        help="use/not elb for image reconstruction.")

    parser.add_argument("--sl_fc", type=str2bool, default=None,
                        help="Self-learning over fcams.")
    parser.add_argument("--sl_fc_lambda", type=float, default=None,
                        help="Lambda for self-learning fcams.")
    parser.add_argument("--sl_start_ep", type=int, default=None,
                        help="Start epoch for self-learning fcams.")
    parser.add_argument("--sl_end_ep", type=int, default=None,
                        help="End epoch for self-learning fcams.")
    parser.add_argument("--sl_min", type=int, default=None,
                        help="MIN for self-learning fcams.")
    parser.add_argument("--sl_max", type=int, default=None,
                        help="MAX for self-learning fcams.")
    parser.add_argument("--sl_ksz", type=int, default=None,
                        help="Kernel size for dilation for self-learning "
                             "fcams.")
    parser.add_argument("--sl_min_p", type=float, default=None,
                        help="Percentage of pixels to be considered "
                             "background to sample from.")
    parser.add_argument("--sl_fg_erode_k", type=int, default=None,
                        help="Kernel size of erosion for foreground.")
    parser.add_argument("--sl_fg_erode_iter", type=int, default=None,
                        help="Number of time to perform erosion over "
                             "foreground.")
    parser.add_argument("--sl_min_ext", type=int, default=None,
                        help="MIN extent for self-learning fcams.")
    parser.add_argument("--sl_max_ext", type=int, default=None,
                        help="MAX extent for self-learning fcams.")
    parser.add_argument("--sl_block", type=int, default=None,
                        help="Size of the blocks for self-learning fcams.")

    parser.add_argument("--seg_ignore_idx", type=int, default=None,
                        help="Ignore index for segmentation.")
    parser.add_argument("--amp", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "training.")
    parser.add_argument("--amp_eval", type=str2bool, default=None,
                        help="Whether to use automatic mixed precision for "
                             "inference.")
    # DDP
    parser.add_argument("--local_rank", type=int, default=None,
                        help='DDP. Local rank. Set too zero if you are using '
                             'one node. not CC().')
    parser.add_argument("--local_world_size", type=int, default=None,
                        help='DDP. Local world size: number of gpus per node. '
                             'Not CC().')

    parser.add_argument('--init_method', default=None,
                        type=str,
                        help='DDP. init method. CC().')
    parser.add_argument('--dist_backend', default=None, type=str,
                        help='DDP. Distributed backend. CC()')
    parser.add_argument('--world_size', type=int, default=None,
                        help='DDP. World size. CC().')

    input_parser = parser.parse_args()

    def warnit(name, vl_old, vl):
        """
        Warn that the variable with the name 'name' has changed its value
        from 'vl_old' to 'vl' through command line.
        :param name: str, name of the variable.
        :param vl_old: old value.
        :param vl: new value.
        :return:
        """
        if vl_old != vl:
            print("Changing {}: {}  -----> {}".format(name, vl_old, vl))
        else:
            print("{}: {}".format(name, vl_old))

    attributes = input_parser.__dict__.keys()

    for k in attributes:
        val_k = getattr(input_parser, k)
        if k in args.keys():
            if val_k is not None:
                warnit(k, args[k], val_k)
                args[k] = val_k
            else:
                warnit(k, args[k], args[k])

        elif k in args['model'].keys():  # try model
            if val_k is not None:
                warnit('model.{}'.format(k), args['model'][k], val_k)
                args['model'][k] = val_k
            else:
                warnit('model.{}'.format(k), args['model'][k],
                       args['model'][k])

        elif k in args['optimizer'].keys():  # try optimizer 0
            if val_k is not None:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k], val_k)
                args['optimizer'][k] = val_k
            else:
                warnit(
                    'optimizer.{}'.format(k), args['optimizer'][k],
                    args['optimizer'][k]
                )
        else:
            raise ValueError("Key {} was not found in args. ..."
                             "[NOT OK]".format(k))

    # add the current seed to the os env. vars. to be shared across this
    # process.
    # this seed is expected to be local for this process and all its
    # children.
    # running a parallel process will not have access to this copy not
    # modify it. Also, this variable will not appear in the system list
    # of variables. This is the expected behavior.
    # TODO: change this way of sharing the seed through os.environ. [future]
    # the doc mentions that the above depends on `putenv()` of the
    # platform.
    # https://docs.python.org/3.7/library/os.html#os.environ
    os.environ['MYSEED'] = str(args["MYSEED"])
    max_seed = (2 ** 32) - 1
    msg = f"seed must be: 0 <= {int(args['MYSEED'])} <= {max_seed}"
    assert 0 <= int(args['MYSEED']) <= max_seed, msg

    args['outd'], args['subpath'] = outfd(Dict2Obj(args), eval=eval)
    args['outd_backup'] = args['outd']
    if is_cc():
        _tag = '{}__{}'.format(
            basename(normpath(args['outd'])), '{}'.format(
                np.random.randint(low=0, high=10000000, size=1)[0]))
        args['outd'] = join(os.environ["SLURM_TMPDIR"], _tag)
        mkdir(args['outd'])

    cmdr = not constants.OVERRUN
    cmdr &= not eval
    if is_cc():
        cmdr &= os.path.isfile(join(args['outd_backup'], 'passed.txt'))
        os.makedirs(join(os.environ["SCRATCH"], constants.SCRATCH_COMM),
                    exist_ok=True)
    else:
        cmdr &= os.path.isfile(join(args['outd'], 'passed.txt'))
    if cmdr:
        warnings.warn('EXP {} has already been done. EXITING.'.format(
            args['outd']))
        sys.exit(0)

    args['scoremap_paths'] = configure_scoremap_output_paths(Dict2Obj(args))

    if args['box_v2_metric']:
        args['multi_contour_eval'] = True
        args['multi_iou_eval'] = True
    else:
        args['multi_contour_eval'] = False
        args['multi_iou_eval'] = False

    if args['model']['freeze_cl']:
        tag = get_tag(Dict2Obj(args),
                      checkpoint_type=args['eval_checkpoint_type'])
        args['model']['folder_pre_trained_cl'] = join(
            root_dir, 'pretrained', tag)

        assert os.path.isdir(args['model']['folder_pre_trained_cl'])

    # path to pre-computed heatmaps: always needed.
    _cnd_ = args['align_atten_to_heatmap_use_precomputed']
    _cnd_ &= (args['align_atten_to_heatmap_folder'] == '')
    if _cnd_:
        args['align_atten_to_heatmap_folder'] = build_heatmap_folder(
            args, constants.ALIGN_ATTEN_HEATMAP, False)

    _cnd_ = args['train_daug_mask_img_heatmap_use_precomputed']
    _cnd_ &= (args['train_daug_mask_img_heatmap_folder'] == '')
    if _cnd_:
        args['train_daug_mask_img_heatmap_folder'] = build_heatmap_folder(
            args, constants.TRAIN_HEATMAP, False)

    _cnd_ = args['eval_daug_mask_img_heatmap_use_precomputed']
    _cnd_ &= (args['eval_daug_mask_img_heatmap_folder'] == '')
    if _cnd_:
        args['eval_daug_mask_img_heatmap_folder'] = build_heatmap_folder(
            args, constants.EVAL_HEATMAP, False)

    _cnd_ = args['model']['do_segmentation']
    _cnd_ &= args['aus_seg']
    _cnd_ &= (args['aus_seg_folder'] == '')
    if _cnd_:
        args['aus_seg_folder'] = build_heatmap_folder(
            args, constants.AUS_SEGM, False)

        # if is_cc():
        #     baseurl_sc = "{}/datasets/wsol-done-right".format(
        #         os.environ["SCRATCH"])
        #     scratch_path = join(baseurl_sc, '{}.tar.gz'.format(tag))
        #
        #     if os.path.isfile(scratch_path):
        #         slurm_dir = get_root_wsol_dataset()
        #         cmds = [
        #             'cp {} {} '.format(scratch_path, slurm_dir),
        #             'cd {} '.format(slurm_dir),
        #             'tar -xf {}'.format('{}.tar.gz'.format(tag))
        #         ]
        #         cmdx = " && ".join(cmds)
        #         print("Running bash-cmds: \n{}".format(
        #             cmdx.replace("&& ", "\n")))
        #         subprocess.run(cmdx, shell=True, check=True)
        #
        #         assert os.path.isdir(join(slurm_dir, tag))
        #         path_heatmaps = join(slurm_dir, tag)
        #
        #     else:
        #         raise ValueError(f"{scratch_path} not found.")
        # else:
        #     baseurl = get_root_wsol_dataset()
        #     path_heatmaps = join(baseurl, tag)
        #
        # assert os.path.isdir(path_heatmaps), path_heatmaps
        # args['align_atten_to_heatmap_folder'] = path_heatmaps


    if args['task'] == constants.F_CL:
        for split in constants.SPLITS:
            tag = get_tag(Dict2Obj(args),
                          checkpoint_type=args['eval_checkpoint_type'])
            tag += '_cams_{}'.format(split)

            if is_cc():
                baseurl_sc = "{}/datasets/wsol-done-right".format(
                    os.environ["SCRATCH"])
                scratch_path = join(baseurl_sc, '{}.tar.gz'.format(tag))

                if os.path.isfile(scratch_path):
                    slurm_dir = get_root_wsol_dataset()
                    cmds = [
                        'cp {} {} '.format(scratch_path, slurm_dir),
                        'cd {} '.format(slurm_dir),
                        'tar -xf {}'.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                    assert os.path.isdir(join(slurm_dir, tag))
                    args['std_cams_folder'][split] = join(slurm_dir, tag)

            else:
                path_cams = join(root_dir, constants.DATA_CAMS, tag)
                cndx = not os.path.isdir(path_cams)
                cndx &= os.path.isfile('{}.tar.gz'.format(path_cams))
                if cndx:
                    cmds_untar = [
                        'cd {} '.format(join(root_dir, constants.DATA_CAMS)),
                        'tar -xf {} '.format('{}.tar.gz'.format(tag))
                    ]
                    cmdx = " && ".join(cmds_untar)
                    print("Running bash-cmds: \n{}".format(
                        cmdx.replace("&& ", "\n")))
                    subprocess.run(cmdx, shell=True, check=True)

                if os.path.isdir(path_cams):
                    args['std_cams_folder'][split] = path_cams

    # DDP. ---------------------------------------------------------------------
    ngpus_per_node = torch.cuda.device_count()

    if is_cc():  # multiple nodes. each w/ multiple gpus.
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

        available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES').split(',')
        current_device = int(available_gpus[local_rank])
        torch.cuda.set_device(current_device)

        args['rank'] = rank
        args['local_rank'] = local_rank
        args['is_master'] = ((local_rank == 0) and (rank == 0))
        args['c_cudaid'] = current_device
        args['is_node_master'] = (local_rank == 0)

    else:  # single machine w/ multiple gpus.
        args['world_size'] = ngpus_per_node
        args['is_master'] = args['local_rank'] == 0
        args['is_node_master'] = args['local_rank'] == 0
        torch.cuda.set_device(args['local_rank'])
        args['c_cudaid'] = args['local_rank']
        args['world_size'] = ngpus_per_node

    # --------------------------------------------------------------------------

    reproducibility.set_to_deterministic(seed=int(args["MYSEED"]), verbose=True)

    args_dict = deepcopy(args)
    args = Dict2Obj(args)
    # sanity check ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    valid_freq_mb = args.valid_freq_mb
    assert isinstance(valid_freq_mb, float), type(valid_freq_mb)
    assert (valid_freq_mb == -1) or (0 < valid_freq_mb < 1), valid_freq_mb

    if args.align_atten_to_heatmap_type_heatmap == \
            constants.HEATMAP_AUNITS_LEARNED_SEG:
        assert args.align_atten_to_heatmap_use_precomputed

    if args.train_daug_mask_img_heatmap_type == \
            constants.HEATMAP_AUNITS_LEARNED_SEG:
        assert args.train_daug_mask_img_heatmap_use_precomputed

    if args.eval_daug_mask_img_heatmap_type == \
            constants.HEATMAP_AUNITS_LEARNED_SEG:
        assert args.eval_daug_mask_img_heatmap_use_precomputed

    if args.aus_seg_heatmap_type == constants.HEATMAP_AUNITS_LEARNED_SEG:
        assert args.aus_seg_use_precomputed

    if args.aus_seg:
        assert args.model['do_segmentation']

    if args.model['do_segmentation']:
        assert args.aus_seg

    if args.align_atten_to_heatmap:
        assert args.align_atten_to_heatmap_type_heatmap in \
               constants.HEATMAP_TYPES, args.align_atten_to_heatmap_type_heatmap

    if args.train_daug_mask_img_heatmap:
        v = args.train_daug_mask_img_heatmap_type
        assert v in constants.HEATMAP_TYPES, v
        v = args.train_daug_mask_img_heatmap_bg_filler
        assert v in constants.BG_FILLERS, v

    if args.eval_daug_mask_img_heatmap:
        v = args.eval_daug_mask_img_heatmap_type
        assert v in constants.HEATMAP_TYPES, v
        v = args.eval_daug_mask_img_heatmap_bg_filler
        assert v in constants.BG_FILLERS, v


    assert args.master_selection_metric in constants.METRICS, \
        args.master_selection_metric
    assert args.std_cl_w_style in constants.CLW, args.std_cl_w_style

    if args.sparse_at:
        msg = f"{args.model['encoder_name']} | {constants.APVIT}"
        assert args.model['encoder_name'] == constants.APVIT, msg

    assert args.optimizer['opt__clipgrad'] >= 0, args.optimizer['opt__clipgrad']
    assert isinstance(args.optimizer['opt__clipgrad'], float), type(
        args.optimizer['opt__clipgrad'])

    pretrain = constants.PRETRAINED + ['None']
    assert args.model['encoder_weights'] in pretrain, args.model[
        'encoder_weights']

    if args.model['encoder_weights'] == constants.VGGFACE2:
        msg = f"pretrained weight of {constants.VGGFACE2} available only for " \
              f"{constants.RESNET50} at the moment."
        assert args.model['encoder_name'] == constants.RESNET50, msg

    spatial_dropout = args.model['spatial_dropout']
    assert isinstance(spatial_dropout, float), spatial_dropout
    assert 0. <= spatial_dropout <= 1., spatial_dropout

    assert args.spatial_pooling == constants.METHOD_2_POOLINGHEAD[args.method]
    assert args.model['encoder_name'] in constants.BACKBONES
    
    assert not args.multi_label_flag
    assert args.seg_mode == constants.BINARY_MODE

    if isinstance(args.resize_size, int):
        if isinstance(args.crop_size, int):
            assert args.resize_size >= args.crop_size

    # todo
    assert args.model['scale_in'] > 0.
    assert isinstance(args.model['scale_in'], float)

    if args.task == constants.STD_CL:
        assert not args.model['freeze_cl']
        assert args.model['folder_pre_trained_cl'] in [None, '', 'None']

    used_constraints = [args.sl_fc,
                        args.crf_fc,
                        args.entropy_fc,
                        args.max_sizepos_fc]

    if args.task == constants.STD_CL:
        assert not any(used_constraints)

    # assert args.resize_size == constants.SZ256
    # assert args.crop_size == constants.SZ224

    f"{args.crop_size} | {args.resize_size}"
    assert args.crop_size <= args.resize_size, msg

    if args.task == constants.F_CL:
        assert any(used_constraints)
        assert args.model['arch'] == constants.UNETFCAM

        assert args.eval_checkpoint_type == constants.BEST

    assert args.model['arch'] in constants.ARCHS

    assert not args.im_rec

    return args, args_dict


def configure_scoremap_output_paths(args):
    scoremaps_root = join(args.outd, 'scoremaps')
    scoremaps = mch()
    for split in (constants.TRAINSET, constants.VALIDSET, constants.TESTSET):
        scoremaps[split] = join(scoremaps_root, split)
        if not os.path.isdir(scoremaps[split]):
            os.makedirs(scoremaps[split], exist_ok=True)
    return scoremaps


def outfd(args, eval=False):

    tag = [('id', args.exp_id),
           ('tsk', args.task),
           ('ds', args.dataset),
           ('mth', args.method),
           ('spooling', args.spatial_pooling),
           ('sd', args.MYSEED),
           ('ecd', args.model['encoder_name']),
           ('epx', args.max_epochs),
           ('bsz', args.batch_size),
           ('lr', args.optimizer['opt__lr']),
           ('box_v2_metric', args.box_v2_metric),
           ('amp', args.amp),
           ('amp_eval', args.amp_eval)
           ]

    tag = [(el[0], str(el[1])) for el in tag]
    tag = '-'.join(['_'.join(el) for el in tag])

    if args.task == constants.F_CL:
        # todo: add hyper-params.
        tag2 = []

        if args.sl_fc:
            tag2.append(("sl_fc", 'yes'))

        if args.crf_fc:
            tag2.append(("crf_fc", 'yes'))

        if args.entropy_fc:
            tag2.append(("entropy_fc", 'yes'))

        if args.max_sizepos_fc:
            tag2.append(("max_sizepos_fc", 'yes'))

        if tag2:
            tag2 = [(el[0], str(el[1])) for el in tag2]
            tag2 = '-'.join(['_'.join(el) for el in tag2])
            tag = "{}-{}".format(tag, tag2)

    parent_lv = "exps"
    if args.debug_subfolder not in ['', None, 'None']:
        parent_lv = join(parent_lv, args.debug_subfolder)

    subfd = join(args.dataset, args.model['encoder_name'], args.task,
                 args.method)
    _root_dir = root_dir
    if is_cc():
        _root_dir = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER)

    subpath = join(parent_lv,
                   subfd,
                   tag)
    if not eval:
        OUTD = join(_root_dir,
                    subpath
                    )
    else:
        OUTD = join(_root_dir, args.fd_exp)

    OUTD = expanduser(OUTD)

    if not os.path.exists(OUTD):
        os.makedirs(OUTD, exist_ok=True)

    return OUTD, subpath


def wrap_sys_argv_cmd(cmd: str, pre):
    splits = cmd.split(' ')
    el = splits[1:]
    pairs = ['{} {}'.format(i, j) for i, j in zip(el[::2], el[1::2])]
    pro = splits[0]
    sep = ' \\\n' + (len(pre) + len(pro) + 2) * ' '
    out = sep.join(pairs)
    return "{} {} {}".format(pre, pro, out)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    for fld in flds_files:
        files = glob.iglob(os.path.join(root_dir, fld, "*"))
        subfd = join(dest, fld) if fld != "." else dest
        if not os.path.exists(subfd):
            os.makedirs(subfd, exist_ok=True)

        for file in files:
            if file.endswith(exts):
                if os.path.isfile(file):
                    shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def amp_log(args: object):
    _amp = False
    if args.amp:
        DLLogger.log(fmsg('AMP: activated'))
        _amp = True

    if args.amp_eval:
        DLLogger.log(fmsg('AMP_EVAL: activated'))
        _amp = True

    if _amp:
        tag = get_tag_device(args=args)
        if 'P100' in get_tag_device(args=args):
            DLLogger.log(fmsg('AMP [train: {}, eval: {}] is ON but your GPU {} '
                              'does not seem to have tensor cores. Your code '
                              'may experience slowness. It is better to '
                              'deactivate AMP.'.format(args.amp,
                                                       args.amp_eval, tag)))

def log_path_precomputed_heatmaps(args: object):
    if args.align_atten_to_heatmap:
        if args.align_atten_to_heatmap_use_precomputed:
            msg = f"Precomputed heatmaps are set to be used from " \
                  f"{args.align_atten_to_heatmap_folder}."
            DLLogger.log(fmsg(msg))
        else:
            DLLogger.log(fmsg("Heatmaps will be computed on the fly."))


def parse_input(eval=False):
    """
    Parse the input.
    and
    initialize some modules for reproducibility.
    """
    parser = argparse.ArgumentParser()

    if not eval:
        parser.add_argument("--dataset", type=str,
                            help="Dataset name: {}.".format(constants.DATASETS))
        input_args, _ = parser.parse_known_args()
        args: dict = config.get_config(input_args.dataset)
        args, args_dict = get_args(args)

        if is_cc():
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.init_method,
                                    world_size=args.world_size,
                                    rank=args.rank)
        else:
            dist.init_process_group(backend=args.dist_backend)

        group = dist.group.WORLD
        group_size = torch.distributed.get_world_size(group)
        args_dict['distributed'] = group_size > 1
        assert group_size == args_dict['world_size']
        args.distributed = group_size > 1
        assert group_size == args.world_size

        log_backends = [
            ArbJSONStreamBackend(Verbosity.VERBOSE,
                                 join(args.outd, "log.json")),
            ArbTextStreamBackend(Verbosity.VERBOSE,
                                 join(args.outd, "log.txt")),
        ]

        if args.verbose:
            log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))

        if not args.is_master:
            dist.barrier()
            DLLogger.init_arb(backends=log_backends, master_pid=0)
        else:
            DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())
            dist.barrier()

        DLLogger.log(fmsg("Start time: {}".format(args.t0)))

        amp_log(args=args)
        log_path_precomputed_heatmaps(args=args)

        __split = constants.TRAINSET
        if os.path.isdir(args.std_cams_folder[__split]):
            msg = 'Will be using PRE-computed cams for split {} from ' \
                  '{}'.format(__split, args.std_cams_folder[__split])
            warnings.warn(msg)
            DLLogger.log(msg)
        else:
            msg = 'Will RE-computed cams for split {}.'.format(__split)
            if args.task == constants.F_CL:
                warnings.warn(msg)
                DLLogger.log(msg)

        outd = args.outd

        if args.is_master:
            if not os.path.exists(join(outd, "code/")):
                os.makedirs(join(outd, "code/"), exist_ok=True)

            with open(join(outd, "code/config.yml"), 'w') as fyaml:
                yaml.dump(args_dict, fyaml)

            with open(join(outd, "config.yml"), 'w') as fyaml:
                yaml.dump(args_dict, fyaml)

            str_cmd = wrap_sys_argv_cmd(" ".join(sys.argv), "time python")
            with open(join(outd, "code/cmd.sh"), 'w') as frun:
                frun.write("#!/usr/bin/env bash \n")
                frun.write(str_cmd)

            copy_code(join(outd, "code/"), compress=True, verbose=False)
        dist.barrier()
    else:

        raise NotImplementedError

        parser.add_argument("--fd_exp", type=str,
                            help="relative path to the exp folder.")
        input_args, _ = parser.parse_known_args()
        _root_dir = root_dir
        if is_cc():
            _root_dir = join(os.environ["SCRATCH"], 'fcam')

        fd = join(_root_dir, input_args.fd_exp)

        yaml_file = join(fd, 'config.yaml')
        with open(yaml_file, 'r') as fy:
            args = yaml.safe_load(fy)

        args, args_dict = get_args(args, eval)

    DLLogger.flush()
    return args, args_dict
