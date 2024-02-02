import warnings
import sys
import os
from os.path import dirname, abspath, join, basename
from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from torch.optim import SGD
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.learning import lr_scheduler as my_lr_scheduler

from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import count_nb_params
from dlib.configure import constants
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag
from dlib.utils.shared import format_dict_2_str
from dlib.utils.shared import move_state_dict_to_device
from dlib.utils.shared import cl_w_tech1
from dlib.utils.shared import cl_w_tech2

import dlib
from dlib.create_models.core import create_model
from dlib.create_models.core import create_apvit

from dlib.losses.elb import ELB
from dlib import losses


import dlib.dllogger as DLLogger


__all__ = [
    'get_loss',
    'get_pretrainde_classifier',
    'get_model',
    'get_optimizer'
]


def get_encoder_d_c(encoder_name):
    if encoder_name in [constants.VGG16]:
        vgg_encoders = dlib.encoders.vgg_encoders
        encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        decoder_channels = (256, 128, 64)
    else:
        encoder_depth = 5
        decoder_channels = (256, 128, 64, 32, 16)

    return encoder_depth, decoder_channels


def get_loss(args):
    masterloss = losses.MasterLoss(cuda_id=args.c_cudaid)
    support_background = args.model['support_background']
    multi_label_flag = args.multi_label_flag
    assert not multi_label_flag

    # image classification loss
    if args.task == constants.STD_CL:
        if args.ce:
            cl_loss = losses.CrossEntropyLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.ce_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.ce_start_ep,
                end_epoch=args.ce_end_ep
            )
            cl_loss.set_it(ce_label_smoothing=args.ce_label_smoothing)
            masterloss.add(cl_loss)

        # Orthogonality for weight of the linear classifier (output net)
        if args.ortho_lw:

            ortho_lw_loss = losses.OrthoLinearWeightLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.ortho_lw_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.ortho_lw_start_ep,
                end_epoch=args.ortho_lw_end_ep,
                elb=nn.Identity()
            )

            ortho_lw_loss.set_it(method=args.ortho_lw_method,
                                 spec_iter=args.ortho_lw_spec_iter
                                 )

            masterloss.add(ortho_lw_loss)

        # sparse linear features.
        if args.sparse_lf:
            use_elb = args.sparse_lf_use_elb
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            sparse_lf_loss = losses.SparseLinearFeaturesLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.sparse_lf_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.sparse_lf_start_ep,
                end_epoch=args.sparse_lf_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            sparse_lf_loss.set_it(method=args.sparse_lf_method,
                                  p=args.sparse_lf_p,
                                  c=args.sparse_lf_c,
                                  use_elb=use_elb,
                                  average_it=args.sparse_lf_average_it
                                  )

            masterloss.add(sparse_lf_loss)

        # sparse linear weight.
        if args.sparse_lw:
            use_elb = args.sparse_lw_use_elb
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            sparse_lw_loss = losses.SparseLinClassifierWeightsLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.sparse_lw_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.sparse_lw_start_ep,
                end_epoch=args.sparse_lw_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            sparse_lw_loss.set_it(method=args.sparse_lw_method,
                                  p=args.sparse_lw_p,
                                  c=args.sparse_lw_c,
                                  use_elb=use_elb,
                                  average_it=args.sparse_lw_average_it
                                  )

            masterloss.add(sparse_lw_loss)

        # sparse APVIT attention.
        if args.sparse_at:
            use_elb = args.sparse_at_use_elb
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            sparse_at_loss = losses.SparseApvitAttentionLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.sparse_at_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.sparse_at_start_ep,
                end_epoch=args.sparse_at_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            sparse_at_loss.set_it(method=args.sparse_at_method,
                                  p=args.sparse_at_p,
                                  c=args.sparse_at_c,
                                  use_elb=use_elb,
                                  average_it=args.sparse_at_average_it
                                  )

            masterloss.add(sparse_at_loss)

        if args.s_cost_s:
            self_cost_sensitive_loss = losses.SelfCostSensitiveLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.s_cost_s_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.s_cost_s_start_ep,
                end_epoch=args.s_cost_s_end_ep
            )
            self_cost_sensitive_loss.set_it(
                apply_to=args.s_cost_s_apply_to,
                n_cls=args.num_classes, norm=args.s_cost_s_norm,
                confusion_func=args.s_cost_s_confusion_func,
                top_k=args.s_cost_s_topk,
                reduction=args.s_cost_s_reduction
            )
            masterloss.add(self_cost_sensitive_loss)

        if args.high_entropy:
            h_ent_loss = losses.HighEntropy(
                cuda_id=args.c_cudaid,
                lambda_=args.high_entropy_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.high_entropy_start_ep,
                end_epoch=args.high_entropy_end_ep
            )
            h_ent_loss.set_it(type_reg=args.high_entropy_type,
                              alpha=args.high_entropy_a)
            masterloss.add(h_ent_loss)

        if args.mse:
            mse_loss = losses.MeanSquaredErrorLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.mse_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.mse_start_ep,
                end_epoch=args.mse_end_ep
            )
            mse_loss.set_it(n_cls=args.num_classes)
            masterloss.add(mse_loss)

        if args.mae:
            mae_loss = losses.MeanAbsoluteErrorLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.mae_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.mae_start_ep,
                end_epoch=args.mae_end_ep
            )
            mae_loss.set_it(n_cls=args.num_classes)
            masterloss.add(mae_loss)

        if args.w_sparsity:
            ws_loss = losses.WeightsSparsityLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.w_sparsity_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.w_sparsity_start_ep,
                end_epoch=args.w_sparsity_end_ep
            )

            masterloss.add(ws_loss)

        if args.mtl_focal:
            mtl_f = losses.MultiClassFocalLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.mtl_focal_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.mtl_focal_start_ep,
                end_epoch=args.mtl_focal_end_ep
            )
            mtl_f.set_it(alpha_focal=args.mtl_focal_alpha,
                         gamma_focal=args.mtl_focal_gamma
                         )
            masterloss.add(mtl_f)

        # Layerwise attention alignment with heatmaps

        if args.align_atten_to_heatmap:
            use_elb = args.align_atten_to_heatmap_elb
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            align_loss = losses.AlignToHeatMap(
                cuda_id=args.c_cudaid,
                lambda_=args.align_atten_to_heatmap_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.align_atten_to_heatmap_start_ep,
                end_epoch=args.align_atten_to_heatmap_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            align_loss.set_it(alignment=args.align_atten_to_heatmap_align_type,
                              p=args.align_atten_to_heatmap_p,
                              q=args.align_atten_to_heatmap_q,
                              atten_layers=args.align_atten_to_heatmap_layers,
                              loss_type=args.align_atten_to_heatmap_loss,
                              scale_to=args.align_atten_to_heatmap_scale_to,
                              use_elb=use_elb,
                              norm_att=args.align_atten_to_heatmap_norm_att
                              )

            masterloss.add(align_loss)

        if args.aus_seg:
            assert args.model['do_segmentation']
            aus_seg_loss = losses.AunitsSegmentationLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.aus_seg_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.aus_seg_start_ep,
                end_epoch=args.aus_seg_end_ep
            )
            masterloss.add(aus_seg_loss)


        if args.oc_mean:
            use_elb = args.oc_mean_elb
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            oc_mean_loss = losses.OrdinalMeanLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.oc_mean_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.oc_mean_start_ep,
                end_epoch=args.oc_mean_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            oc_mean_loss.set_it(use_elb=use_elb, eps=args.oc_mean_epsilon)
            masterloss.add(oc_mean_loss)

        if args.oc_var:
            use_elb = args.oc_var_elb
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            oc_var_loss = losses.OrdinalVarianceLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.oc_var_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.oc_var_start_ep,
                end_epoch=args.oc_var_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            oc_var_loss.set_it(use_elb=use_elb, eps=args.oc_var_epsilon)
            masterloss.add(oc_var_loss)


        if args.oc_unim_inq:
            elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            oc_unimod_ineq_loss = losses.OrdIneqUnimodLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.oc_unim_inq_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.oc_unim_inq_start_ep,
                end_epoch=args.oc_unim_inq_end_ep,
                elb=deepcopy(elb)
            )

            oc_unimod_ineq_loss.set_it(data_type=args.oc_unim_inq_type)
            masterloss.add(oc_unimod_ineq_loss)


        if args.con_scores:
            elb = ELB(init_t=args.elb_init_t,
                      max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            con_scores_loss = losses.ConstraintScoresLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.con_scores_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.con_scores_start_ep,
                end_epoch=args.con_scores_end_ep,
                elb=deepcopy(elb)
            )

            con_scores_loss.set_it(n_cls=args.num_classes,
                                   con_scores_min=args.con_scores_min)

            masterloss.add(con_scores_loss)

        # orthogonal features

        if args.free_orth_ft:
            use_elb = args.free_orth_ft_elb
            elb = ELB(init_t=args.elb_init_t,
                      max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            free_orth_ft_loss = losses.FreeOrthFeatures(
                cuda_id=args.c_cudaid,
                lambda_=args.free_orth_ft_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.free_orth_ft_start_ep,
                end_epoch=args.free_orth_ft_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            free_orth_ft_loss.set_it(use_layers=args.free_orth_ft_layers,
                                  same_cl=args.free_orth_ft_same_cl,
                                  diff_cl=args.free_orth_ft_diff_cl,
                                  use_elb=use_elb
                                  )

            masterloss.add(free_orth_ft_loss)


        if args.guid_orth_ft:
            use_elb = args.guid_orth_ft_elb
            elb = ELB(init_t=args.elb_init_t,
                      max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            guid_orth_ft_loss = losses.GuidedOrthFeatures(
                cuda_id=args.c_cudaid,
                lambda_=args.guid_orth_ft_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.guid_orth_ft_start_ep,
                end_epoch=args.guid_orth_ft_end_ep,
                elb=deepcopy(elb) if use_elb else nn.Identity()
            )

            guid_orth_ft_loss.set_it(use_layers=args.guid_orth_ft_layers,
                                     n_cls=args.num_classes,
                                     use_elb=use_elb
                                     )

            masterloss.add(guid_orth_ft_loss)

        if args.att_sz:

            elb = ELB(init_t=args.elb_init_t,
                      max_t=args.elb_max_t,
                      mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

            atte_sz_loss = losses.AttentionSizeLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.att_sz_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.att_sz_start_ep,
                end_epoch=args.att_sz_end_ep,
                elb=deepcopy(elb)
            )

            bounds = args.att_sz_bounds
            bounds = bounds.split('*')
            assert len(bounds) == 2, bounds

            low_b = float(bounds[0])
            up_b = float(bounds[1])
            atte_sz_loss.set_it(low_b=low_b, up_b=up_b)

            masterloss.add(atte_sz_loss)


        if args.att_ent_sz:

            atte_ent_loss = losses.LowEntropyAttentionLoss(
                cuda_id=args.c_cudaid,
                lambda_=args.att_ent_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.att_ent_start_ep,
                end_epoch=args.att_ent_end_ep
            )

            masterloss.add(atte_ent_loss)

    # fcams
    elif args.task == constants.F_CL:

        if not args.model['freeze_cl']:
            masterloss.add(losses.ClLoss(
                cuda_id=args.c_cudaid,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        elb = ELB(init_t=args.elb_init_t, max_t=args.elb_max_t,
                  mulcoef=args.elb_mulcoef).cuda(args.c_cudaid)

        if args.im_rec:
            masterloss.add(
                losses.ImgReconstruction(
                    cuda_id=args.c_cudaid,
                    lambda_=args.im_rec_lambda,
                    elb=deepcopy(elb) if args.sr_elb else nn.Identity(),
                    support_background=support_background,
                    multi_label_flag=multi_label_flag)
            )

        if args.crf_fc:
            masterloss.add(losses.ConRanFieldFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.crf_lambda,
                sigma_rgb=args.crf_sigma_rgb,
                sigma_xy=args.crf_sigma_xy,
                scale_factor=args.crf_scale,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.crf_start_ep,
                end_epoch=args.crf_end_ep
            ))

        if args.entropy_fc:
            masterloss.add(losses.EntropyFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.entropy_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag))

        if args.max_sizepos_fc:
            masterloss.add(losses.MaxSizePositiveFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.max_sizepos_fc_lambda,
                elb=deepcopy(elb), support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.max_sizepos_fc_start_ep,
                end_epoch=args.max_sizepos_fc_end_ep
            ))

        if args.sl_fc:
            sl_fcam = losses.SelfLearningFcams(
                cuda_id=args.c_cudaid,
                lambda_=args.sl_fc_lambda,
                support_background=support_background,
                multi_label_flag=multi_label_flag,
                start_epoch=args.sl_start_ep, end_epoch=args.sl_end_ep,
                seg_ignore_idx=args.seg_ignore_idx
            )

            masterloss.add(sl_fcam)

        assert len(masterloss.n_holder) > 1
    else:
        raise NotImplementedError

    if (args.task != constants.STD_CL) and args.align_atten_to_heatmap:
        raise NotImplementedError

    masterloss.check_losses_status()
    masterloss.cuda(args.c_cudaid)

    # init class weights.
    masterloss.init_class_weights(n_cls=args.num_classes,
                                  w=get_base_cl_weights(args),
                                  style=args.std_cl_w_style)

    DLLogger.log(message="Train loss: {}".format(masterloss))
    return masterloss

def get_base_cl_weights(args):
    n_cls = args.num_classes

    if args.std_cl_w_style in [constants.CLWNONE, constants.CLWADAPTIVE]:
        return None

    elif args.std_cl_w_style in [constants.CLWFIXEDTECH1,
                                 constants.CLWFIXEDTECH2,
                                 constants.CLWMIXED
                                 ]:
        path_stats = join(root_dir, args.metadata_root, 'per_class_weight.yaml')
        with open(path_stats, 'r') as f:
            stats = yaml.safe_load(f)
        assert isinstance(stats, dict), type(stats)
        ks = len(list(stats.keys()))
        assert ks == n_cls, f"{ks} {n_cls}"
        w = [1. for _ in range(n_cls)]
        for k in stats:
            w[k] = stats[k]

        # tech 1: CLWADAPTIVE unnecessary...
        if args.std_cl_w_style in [constants.CLWFIXEDTECH1,
                                   constants.CLWADAPTIVE]:

            w = cl_w_tech1(w=w, n_cls=n_cls)

        # tech 2:
        elif args.std_cl_w_style == constants.CLWFIXEDTECH2:
            w = cl_w_tech2(w=w)

        else:
            raise NotImplementedError(args.std_cl_w_style)


        return w


def get_aux_params(args):
    """
    Prepare the head params.
    :param args:
    :return:
    """
    assert args.spatial_pooling in constants.SPATIAL_POOLINGS
    return {
        "pooling_head": args.spatial_pooling,
        "classes": args.num_classes,
        "modalities": args.wc_modalities,
        "kmax": args.wc_kmax,
        "kmin": args.wc_kmin,
        "alpha": args.wc_alpha,
        "dropout": args.wc_dropout,
        "dense_dims": args.dense_dims,
        "dense_dropout": args.dense_dropout,
        "support_background": args.model['support_background'],
        "r": args.lse_r,
        'encoder_name': args.model['encoder_name'],
        'acol_drop_threshold': args.acol_drop_threshold,
        'prm_ks': args.prm_ks,
        'prm_st': args.prm_st
    }


def get_pretrainde_classifier(args):
    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    aux_params = None

    apply_self_atten: bool = False
    atten_layers_id: list = None

    do_segmentation = args.model['do_segmentation']

    if do_segmentation:
        assert args.method not in [constants.METHOD_APVIT,
                                   constants.METHOD_TSCAM], args.method

    cndx = args.align_atten_to_heatmap
    cndx &= args.align_atten_to_heatmap_use_self_atten

    if cndx:
        # supports only resnetx. not transformers.
        assert p.encoder_name in [constants.RESNET18, constants.RESNET34,
                   constants.RESNET50, constants.RESNET101,
                   constants.RESNET152], p.encoder_name

        layers = args.align_atten_to_heatmap_layers
        assert isinstance(layers, str), type(str)
        z = layers.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, f"{len(z)} | {layers}"
        for i in z:
            assert i > 0, i  # layers count starts from 0. but layer 0 is not
        # allowed to be used for alignment. it holds the input image.
        # we dont check here the conformity to the maximum allowed value.
        # we do that in the forward. the maximum allowed value will be
        # determined automatically based on the length of the feature
        # holder.

        atten_layers_id: list = z
        apply_self_atten = cndx

    if args.method == constants.METHOD_APVIT:
        _encoder = args.model['encoder_name']
        assert _encoder == constants.APVIT, f"{_encoder} |" \
                                            f" {constants.APVIT}"
        assert not do_segmentation

        model = instance_apvit(args)

    elif args.method == constants.METHOD_TSCAM:
        assert not do_segmentation

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=args.method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            num_classes=args.num_classes
        )

    elif args.method == constants.METHOD_ADL:
        aux_params = get_aux_params(args)
        model = create_model(
            task=constants.STD_CL,
            arch=constants.STDCLASSIFIER,
            method=args.method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=p.in_channels,
            encoder_depth=encoder_depth,
            scale_in=p.scale_in,
            large_maps=p.large_maps,
            use_adl=True,
            adl_drop_rate=args.adl_drop_rate,
            adl_drop_threshold=args.adl_drop_threshold,
            apply_self_atten=apply_self_atten,
            atten_layers_id=atten_layers_id,
            do_segmentation=do_segmentation,
            aux_params=aux_params
        )

    else:
        aux_params = get_aux_params(args)
        model = create_model(
            task=constants.STD_CL,
            arch=constants.STDCLASSIFIER,
            method=args.method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=p.in_channels,
            encoder_depth=encoder_depth,
            scale_in=p.scale_in,
            large_maps=p.large_maps,
            use_adl=False,
            apply_self_atten=apply_self_atten,
            atten_layers_id=atten_layers_id,
            do_segmentation=do_segmentation,
            aux_params=aux_params
        )

    DLLogger.log("PRETRAINED CLASSIFIER `{}` was created. "
                 "Nbr.params: {}".format(model, count_nb_params(model)))
    log = "Arch: {}\n" \
          "encoder_name: {}\n" \
          "encoder_weights: {}\n" \
          "classes: {}\n" \
          "aux_params: \n{}\n" \
          "scale_in: {}\n" \
          "freeze_cl: {}\n" \
          "img_range: {} \n" \
          "".format(p.arch,
                    p.encoder_name,
                    encoder_weights, classes,
                    format_dict_2_str(
                        aux_params) if aux_params is not None else None,
                    p.scale_in, p.freeze_cl, args.img_range
                    )
    DLLogger.log(log)

    path_cl = args.model['folder_pre_trained_cl']
    assert path_cl not in [None, 'None', '']

    msg = "You have asked to set the classifier " \
          " from {} .... [OK]".format(path_cl)
    warnings.warn(msg)
    DLLogger.log(msg)

    tag = get_tag(args)

    if path_cl.endswith(os.sep):
        source_tag = basename(path_cl[:-1])
    else:
        source_tag = basename(path_cl)

    assert tag == source_tag

    all_w = torch.load(join(path_cl, 'model.pt'),
                       map_location=get_cpu_device())

    if args.model['encoder_name'] == constants.APVIT:
        model.load_state_dict(all_w, strict=True)

    elif args.method in [constants.METHOD_TSCAM]:
        model.load_state_dict(all_w, strict=True)

    else:
        encoder_w = all_w['encoder']
        classification_head_w = all_w['classification_head']

        model.encoder.super_load_state_dict(encoder_w, strict=True)
        model.classification_head.load_state_dict(
            classification_head_w, strict=True)

        if do_segmentation:
            segmentation_head_w = all_w['segmentation_head']
            model.segmentation_head.load_state_dict(
                segmentation_head_w, strict=True)

    # old style:
    # encoder_w = torch.load(join(path_cl, 'encoder.pt'),
    #                        map_location=get_cpu_device())
    # model.encoder.super_load_state_dict(encoder_w, strict=True)
    #
    # header_w = torch.load(join(path_cl, 'classification_head.pt'),
    #                       map_location=get_cpu_device())
    # model.classification_head.load_state_dict(header_w, strict=True)


    # if args.model['freeze_cl']:
    #     assert args.task == constants.F_CL
    #     assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']
    #
    #     model.freeze_classifier()
    #     model.assert_cl_is_frozen()

    model.eval()
    return model

def instance_apvit(args):
    encoder = args.model['encoder_name']
    assert encoder == constants.APVIT, f"{encoder} | {constants.APVIT}"
    return create_apvit(k=args.model['apvit_k'],
                        r=args.model['apvit_r'],
                        num_classes=args.num_classes,
                        dense_dims=args.dense_dims,
                        attn_method=args.model['apvit_attn_method'],
                        normalize_att=args.model['apvit_normalize_att'],
                        apply_self_att=args.model['apvit_apply_self_att'],
                        hid_att_dim=args.model['apvit_hid_att_dim'],
                        )


def get_model(args, eval=False):

    p = Dict2Obj(args.model)

    encoder_weights = p.encoder_weights
    if encoder_weights == "None":
        encoder_weights = None

    classes = args.num_classes
    encoder_depth, decoder_channels = get_encoder_d_c(p.encoder_name)

    aux_params = None

    apply_self_atten: bool = False
    atten_layers_id: list = None

    cndx = args.align_atten_to_heatmap
    cndx &= args.align_atten_to_heatmap_use_self_atten

    if cndx:
        # supports only resnetx. not transformers.
        assert p.encoder_name in [constants.RESNET18, constants.RESNET34,
                                  constants.RESNET50, constants.RESNET101,
                                  constants.RESNET152], p.encoder_name

        layers = args.align_atten_to_heatmap_layers
        assert isinstance(layers, str), type(str)
        z = layers.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, f"{len(z)} | {layers}"
        for i in z:
            assert i > 0, i  # layers count starts from 0. but layer 0 is not
        # allowed to be used for alignment. it holds the input image.
        # we dont check here the conformity to the maximum allowed value.
        # we do that in the forward. the maximum allowed value will be
        # determined automatically based on the length of the feature
        # holder.

        atten_layers_id: list = z
        apply_self_atten = cndx

    do_segmentation = args.model['do_segmentation']
    if do_segmentation:
        assert args.task == constants.STD_CL, args.task
        assert args.method not in [constants.METHOD_APVIT,
                                   constants.METHOD_TSCAM], args.method

    if args.task == constants.STD_CL:

        if args.method == constants.METHOD_APVIT:
            _encoder = args.model['encoder_name']
            assert _encoder == constants.APVIT, f"{_encoder} |" \
                                                f" {constants.APVIT}"
            assert not do_segmentation

            model = instance_apvit(args)

        elif args.method == constants.METHOD_TSCAM:
            assert not do_segmentation

            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                num_classes=args.num_classes
            )

        elif args.method == constants.METHOD_ADL:
            aux_params = get_aux_params(args)
            model = create_model(
                task=constants.STD_CL,
                arch=constants.STDCLASSIFIER,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                encoder_depth=encoder_depth,
                scale_in=p.scale_in,
                large_maps=p.large_maps,
                use_adl=True,
                adl_drop_rate=args.adl_drop_rate,
                adl_drop_threshold=args.adl_drop_threshold,
                apply_self_atten=apply_self_atten,
                atten_layers_id=atten_layers_id,
                do_segmentation=do_segmentation,
                aux_params=aux_params
            )

        else:
            aux_params = get_aux_params(args)
            model = create_model(
                task=args.task,
                arch=p.arch,
                method=args.method,
                encoder_name=p.encoder_name,
                encoder_weights=encoder_weights,
                in_channels=p.in_channels,
                encoder_depth=encoder_depth,
                scale_in=p.scale_in,
                spatial_dropout=p.spatial_dropout,
                large_maps=p.large_maps,
                apply_self_atten=apply_self_atten,
                atten_layers_id=atten_layers_id,
                do_segmentation=do_segmentation,
                aux_params=aux_params
            )
    elif args.task == constants.F_CL:
        aux_params = get_aux_params(args)

        assert args.seg_mode == constants.BINARY_MODE
        seg_h_out_channels = 2

        model = create_model(
            task=args.task,
            arch=p.arch,
            method=args.method,
            encoder_name=p.encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=p.in_channels,
            seg_h_out_channels=seg_h_out_channels,
            scale_in=p.scale_in,
            large_maps=p.large_maps,
            aux_params=aux_params,
            freeze_cl=p.freeze_cl,
            im_rec=args.im_rec,
            img_range=args.img_range
        )
    else:
        raise NotImplementedError

    DLLogger.log(f"`{model}` was created. Nbr.params: {count_nb_params(model)}")

    log = "Arch: {}\n" \
          "task: {}\n" \
          "encoder_name: {}\n" \
          "encoder_weights: {}\n" \
          "classes: {}\n" \
          "aux_params: \n{}\n" \
          "scale_in: {}\n" \
          "freeze_cl: {}\n" \
          "im_rec: {}\n" \
          "img_range: {} \n" \
          "".format(p.arch, args.task, p.encoder_name,
                    encoder_weights, classes,
                    format_dict_2_str(
                        aux_params) if aux_params is not None else None,
                    p.scale_in, p.freeze_cl, args.im_rec, args.img_range
                    )
    DLLogger.log(log)
    DLLogger.log(model.get_info_nbr_params())

    path_file = args.model['path_pre_trained']
    if path_file not in [None, 'None']:
        msg = "You have asked to load a specific pre-trained " \
              "model from {} .... [OK]".format(path_file)
        warnings.warn(msg)
        DLLogger.log(msg)
        pre_tr_state = torch.load(path_file, map_location=get_cpu_device())
        model.load_state_dict(pre_tr_state, strict=args.model['strict'])

    path_cl = args.model['folder_pre_trained_cl']
    if path_cl not in [None, 'None', '']:
        assert args.task == constants.F_CL

        msg = "You have asked to set the classifier " \
              " from {} .... [OK]".format(path_cl)
        warnings.warn(msg)
        DLLogger.log(msg)

        tag = get_tag(args)

        if path_cl.endswith(os.sep):
            source_tag = basename(path_cl[:-1])
        else:
            source_tag = basename(path_cl)

        assert tag == source_tag

        all_w = torch.load(join(path_cl, 'model.pt'),
                           map_location=get_cpu_device())

        if args.method in [constants.METHOD_TSCAM, constants.METHOD_APVIT]:
            model.load_state_dict(all_w, strict=True)

        else:

            encoder_w = all_w['encoder']
            classification_head_w = all_w['classification_head']

            model.encoder.super_load_state_dict(encoder_w, strict=True)
            model.classification_head.load_state_dict(
                classification_head_w, strict=True)

            if do_segmentation:
                segmentation_head_w = all_w['segmentation_head']
                model.segmentation_head.load_state_dict(
                    segmentation_head_w, strict=True)


        # old style.
        # encoder_w = torch.load(join(path_cl, 'encoder.pt'),
        #                        map_location=get_cpu_device())
        # model.encoder.super_load_state_dict(encoder_w, strict=True)
        #
        # header_w = torch.load(join(path_cl, 'classification_head.pt'),
        #                       map_location=get_cpu_device())
        # model.classification_head.load_state_dict(header_w, strict=True)

    if args.model['freeze_cl']:
        assert args.task == constants.F_CL
        assert args.model['folder_pre_trained_cl'] not in [None, 'None', '']

        model.freeze_classifier()
        model.assert_cl_is_frozen()

    if eval:
        assert os.path.isdir(args.outd)
        tag = get_tag(args, checkpoint_type=args.eval_checkpoint_type)
        path = join(args.outd, tag)
        cpu_device = get_cpu_device()

        if args.task == constants.STD_CL:

            path = join(path, 'model.pt')
            all_w = torch.load(path, map_location=get_cpu_device())

            if args.method in [constants.METHOD_TSCAM,
                                 constants.METHOD_APVIT]:
                model.load_state_dict(all_w, strict=True)

            else:
                encoder_w = all_w['encoder']
                classification_head_w = all_w['classification_head']

                model.encoder.super_load_state_dict(encoder_w, strict=True)
                model.classification_head.load_state_dict(
                    classification_head_w, strict=True)

                if do_segmentation:
                    segmentation_head_w = all_w['segmentation_head']
                    model.segmentation_head.load_state_dict(
                        segmentation_head_w, strict=True)

            # old style:
            # weights = torch.load(join(path, 'encoder.pt'),
            #                      map_location=cpu_device)
            # model.encoder.super_load_state_dict(weights, strict=True)
            #
            # weights = torch.load(join(path, 'classification_head.pt'),
            #                      map_location=cpu_device)
            # model.classification_head.load_state_dict(weights, strict=True)

        elif args.task == constants.F_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=cpu_device)
            model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=cpu_device)
            model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=cpu_device)
            model.segmentation_head.load_state_dict(weights, strict=True)
            if model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=cpu_device)
                model.reconstruction_head.load_state_dict(weights, strict=True)
        else:
            raise NotImplementedError

        msg = "EVAL-mode. Reset model weights to: {}".format(path)
        warnings.warn(msg)
        DLLogger.log(msg)

        model.eval()

    return model


def standardize_otpmizers_params(optm_dict):
    """
    Standardize the keys of a dict for the optimizer.
    all the keys starts with 'optn[?]__key' where we keep only the key and
    delete the initial.
    the dict should not have a key that has a dict as value. we do not deal
    with this case. an error will be raise.

    :param optm_dict: dict with specific keys.
    :return: a copy of optm_dict with standardized keys.
    """
    msg = "'optm_dict' must be of type dict. found {}.".format(type(optm_dict))
    assert isinstance(optm_dict, dict), msg
    new_optm_dict = deepcopy(optm_dict)
    loldkeys = list(new_optm_dict.keys())

    for k in loldkeys:
        if k.startswith('opt'):
            msg = "'{}' is a dict. it must not be the case." \
                  "otherwise, we have to do a recursive thing....".format(k)
            assert not isinstance(new_optm_dict[k], dict), msg

            new_k = k.split('__')[1]
            new_optm_dict[new_k] = new_optm_dict.pop(k)

    return new_optm_dict


def _get_model_params_for_opt(args, model):
    hparams = deepcopy(args.optimizer)
    hparams = standardize_otpmizers_params(hparams)
    hparams = Dict2Obj(hparams)

    if args.task == constants.F_CL or (
            args.model['encoder_name'] == constants.APVIT
    ) or (args.method == constants.METHOD_TSCAM):
        return [
            {'params': model.parameters(), 'lr': hparams.lr}
        ]

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['features.'],
        'resnet': ['layer4.', 'fc.'],
        'inception': ['Mixed', 'Conv2d_1', 'Conv2d_2',
                      'Conv2d_3', 'Conv2d_4'],
    }

    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    architecture = args.model['encoder_name']
    assert architecture in constants.BACKBONES

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['encoder.features.'],  # features
        'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
        'inception': ['encoder.Mixed', 'encoder.Conv2d_1', 'encoder.Conv2d_2',
                      'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
    }

    param_features = []
    param_classifiers = []

    def param_features_substring_list(arch):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if arch.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(arch))

    for name, parameter in model.named_parameters():

        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_features.append(parameter)
            elif architecture in  [constants.RESNET18, constants.RESNET34,
                                  constants.RESNET50, constants.RESNET101,
                                  constants.RESNET152]:
                param_classifiers.append(parameter)
            else:
                raise NotImplementedError
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                param_classifiers.append(parameter)
            elif architecture in [constants.RESNET18, constants.RESNET34,
                                  constants.RESNET50, constants.RESNET101,
                                  constants.RESNET152]:
                param_features.append(parameter)
            else:
                raise NotImplementedError

    return [
            {'params': param_features, 'lr': hparams.lr},
            {'params': param_classifiers,
             'lr': hparams.lr * hparams.lr_classifier_ratio}
    ]


def get_optimizer(args, model):
    """Instantiate an optimizer.
    Input:
        args: object. Contains the configuration of the exp that has been
        read from the yaml file.
        mode: a pytorch model with parameters.

    Output:
        optimizer: a pytorch optimizer.
        lrate_scheduler: a pytorch learning rate scheduler (or None).
    """
    hparams = deepcopy(args.optimizer)
    hparams = standardize_otpmizers_params(hparams)
    hparams = Dict2Obj(hparams)

    op_col = {}

    params = _get_model_params_for_opt(args, model)

    op_name = hparams.name_optimizer
    assert op_name in constants.OPTIMIZERS, f"{op_name}, {constants.OPTIMIZERS}"

    if op_name == constants.SGD:
        optimizer = SGD(params=params,
                        momentum=hparams.momentum,
                        dampening=hparams.dampening,
                        weight_decay=hparams.weight_decay,
                        nesterov=hparams.nesterov)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['momentum'] = hparams.momentum
        op_col['dampening'] = hparams.dampening
        op_col['weight_decay'] = hparams.weight_decay
        op_col['nesterov'] = hparams.nesterov

    elif op_name == "adam":
        optimizer = Adam(params=params,
                         betas=(hparams.beta1, hparams.beta2),
                         eps=hparams.eps_adam,
                         weight_decay=hparams.weight_decay,
                         amsgrad=hparams.amsgrad)
        op_col['optim_name'] = hparams.name_optimizer
        op_col['lr'] = hparams.lr
        op_col['beta1'] = hparams.beta1
        op_col['beta2'] = hparams.beta2
        op_col['weight_decay'] = hparams.weight_decay
        op_col['amsgrad'] = hparams.amsgrad
    else:
        raise ValueError("Unsupported optimizer `{}` .... "
                         "".format(args.optimizer["name"]))

    if hparams.lr_scheduler:
        if hparams.name_lr_scheduler == "step":
            lrate_scheduler = lr_scheduler.StepLR(optimizer,
                                                  step_size=hparams.step_size,
                                                  gamma=hparams.gamma,
                                                  last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "cosine":
            lrate_scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hparams.t_max,
                eta_min=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['T_max'] = hparams.T_max
            op_col['eta_min'] = hparams.eta_min
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "mystep":
            lrate_scheduler = my_lr_scheduler.MyStepLR(
                optimizer,
                step_size=hparams.step_size,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch,
                min_lr=hparams.min_lr)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['step_size'] = hparams.step_size
            op_col['gamma'] = hparams.gamma
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "mycosine":
            lrate_scheduler = my_lr_scheduler.MyCosineLR(
                optimizer,
                coef=hparams.coef,
                max_epochs=hparams.max_epochs,
                min_lr=hparams.min_lr,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['coef'] = hparams.coef
            op_col['max_epochs'] = hparams.max_epochs
            op_col['min_lr'] = hparams.min_lr
            op_col['last_epoch'] = hparams.last_epoch

        elif hparams.name_lr_scheduler == "multistep":
            lrate_scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hparams.milestones,
                gamma=hparams.gamma,
                last_epoch=hparams.last_epoch)
            op_col['name_lr_scheduler'] = hparams.name_lr_scheduler
            op_col['milestones'] = hparams.milestones
            op_col['gamma'] = hparams.gamma
            op_col['last_epoch'] = hparams.last_epoch

        else:
            raise ValueError("Unsupported learning rate scheduler `{}` .... "
                             "[NOT OK]".format(
                                hparams.name_lr_scheduler))
    else:
        lrate_scheduler = None

    DLLogger.log("Optimizer:\n{}".format(format_dict_2_str(op_col)))

    return optimizer, lrate_scheduler


def run_estimate_cl_weights():
    dataset = constants.RAFDB

    for style in [constants.CLWFIXEDTECH1, constants.CLWFIXEDTECH2]:
        args = {
            'num_classes': constants.NUMBER_CLASSES[dataset],
            'metadata_root': join(constants.RELATIVE_META_ROOT, dataset),
            'std_cl_w_style': style
        }
        args = Dict2Obj(args)

        print(f"Style: {style} // dataset: {dataset}")
        print(get_base_cl_weights(args))


if __name__ == "__main__":
    run_estimate_cl_weights()