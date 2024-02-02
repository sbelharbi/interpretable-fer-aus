import os
import sys
from os.path import join, dirname, abspath
import datetime as dt

import munch


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.tools import chunk_it
from dlib.utils.tools import get_root_wsol_dataset

__all__ = ["get_config"]


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_data_paths(args, dsname=None):
    if dsname is None:
        dsname = args['dataset']

    train = val = test = join(args['data_root'], dsname)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def get_nbr_bucket(ds: str) -> int:
    nbr_chunks = constants.NBR_CHUNKS_TR[ds]
    out = chunk_it(list(range(nbr_chunks)), constants.BUCKET_SZ)
    return len(list(out))


def get_config(ds):
    assert ds in constants.DATASETS, ds

    args = {
        # ======================================================================
        #                               GENERAL
        # ======================================================================
        "MYSEED": 0,  # Seed for reproducibility. int in [0, 2**32 - 1].
        "cudaid": '0',  # str. cudaid. form: '0,1,2,3' for cuda devices.
        "debug_subfolder": '',  # subfolder used for debug. if '', we do not
        # consider it.
        "dataset": ds,  # name of the dataset.
        "num_classes": constants.NUMBER_CLASSES[ds],  # Total number of classes.
        "crop_size": constants.SZ224,  # int. size of cropped patch.
        "resize_size": constants.SZ256,  # int. size to which the image
        # is resized before cropping.
        "batch_size": 32,  # the batch size for training.
        "eval_batch_size": 32,  # evaluation batch size.
        "valid_freq_mb": -1.,  # float. percentage of total trainset to process
        # before perform an internal validation [evaluation over validation
        # set] (within an epoch). This is done in addition to the mandatory
        # validation at each epoch. This allows to do model selection when
        # dataset is large such as Affecnet. Accepted values: -1., floats in ]0,
        # 1[. if it is in ]0, 1[, it is a percentage of total train
        # minibatches. Use the value -1. to turn it off.
        "num_workers": 5,  # number of workers for dataloader of the trainset.
        "exp_id": "123456789",  # exp id. random number unique for the exp.
        "verbose": True,  # if true, we print messages in stdout.
        'fd_exp': None,  # relative path to folder where the exp.
        'abs_fd_exp': None,  # absolute path to folder where the exp.
        'best_epoch': 0,  # int. best epoch.
        'img_range': constants.RANGE_TANH,  # range of the image values after
        # normalization either in [0, 1] or [-1, 1]. see constants.
        't0': dt.datetime.now(),  # approximate time of starting the code.
        'tend': None,  # time when this code ends.
        'running_time': None,  # the time needed to run the entire code.
        'ds_chunkable': (constants.NBR_CHUNKS_TR[ds] != -1),  # whether the
        # trainset is chunked or not. only ilsvrc is chunked. if you want to
        # turn off this completely, set it to False.
        'nbr_buckets': get_nbr_bucket(ds),  # number of train bucket. applied
        # only for chunkable datasets.
        # ======================================================================
        #                      WSOL DONE RIGHT
        # ======================================================================
        "data_root": get_root_wsol_dataset(),  # absolute path to data parent
        # folder.
        "metadata_root": constants.RELATIVE_META_ROOT,  # path to metadata.
        # contains splits.
        "mask_root": get_root_wsol_dataset(),  # path to masks.
        "proxy_training_set": False,  # efficient hyper-parameter search with
        # a proxy training set. true/false.
        "std_cams_folder": mch(train='', val='', test=''),  # folders where
        # cams of std_cl are stored to be used for f_cl training. typicaly,
        # we store only training. this is an option since f_cl can still
        # compute the std_cals. but, storing them making their access fast
        # to avoid re-computing them every time during training. the exact
        # location will be determined during parsing the input. this is
        # optional. if we do not find this folder, we recompute the cams.
        "num_val_sample_per_class": 0,  # number of full_supervision
        # validation sample per class. 0 means: use all available samples.
        'cam_curve_interval': .4,  # CAM curve interval. sota: 0.001. not
        # useful for FER.
        'multi_contour_eval': True,  # Bounding boxes are extracted from all
        # contours in the thresholded score map. You can use this feature by
        # setting multi_contour_eval to True (default). Otherwise,
        # bounding boxes are extracted from the largest connected
        # component of the score map.
        'multi_iou_eval': True,
        'iou_threshold_list': [50],
        'box_v2_metric': False,
        'eval_checkpoint_type': constants.BEST,  # just for
        # stand-alone inference. during training+inference, we evaluate both.
        # Necessary s well for the task F_CL during training to select the
        # init-model-weights-classifier.
        # ======================================================================
        #                      CURRICULUM LEARNING
        # ======================================================================
        "curriculum_l": False,  # if true, CL is applied over trainset.
        "curriculum_l_type": constants.CURRICULUM_CLASS,  # type of CL. see
        # constants.CURRICULUM_TYPES
        "curriculum_l_epoch_rate": 1,  # how often to update the curriculum.
        # [epoch].
        "curriculum_l_cl_class_order": '',  # str. for type:
        # CURRICULUM_CLASS. indicates the order of classes. separators: * to
        # separate between two updates, - to separate between classes in the
        # same update. e.g.: 1-2*6-5-4*3 means: [[1, 2], [6, 5, 4],
        # [3]]. initial CL starts with [1, 2]. then, in the next update,
        # we add [6, 5, 4]. The last update will add [3].
        # ======================================================================
        #                      VISUALISATION OF REGIONS OF INTEREST
        # ======================================================================
        "alpha_visu": 100,  # transparency alpha for cams visualization. low is
        # opaque (matplotlib).
        "height_tag": 60,  # the height of the margin where the tag is written.
        # ======================================================================
        #                             OPTIMIZER (n0)
        #                            TRAIN THE MODEL
        # ======================================================================
        "optimizer": {  # the optimizer
            # ==================== SGD =======================
            "opt__name_optimizer": "sgd",  # str name. 'sgd', 'adam'
            "opt__lr": 0.001,  # Initial learning rate.
            "opt__momentum": 0.9,  # Momentum.
            "opt__dampening": 0.,  # dampening.
            "opt__weight_decay": 1e-4,  # The weight decay (L2) over the
            # parameters.
            "opt__nesterov": True,  # If True, Nesterov algorithm is used.
            # ==================== ADAM =========================
            "opt__beta1": 0.9,  # beta1.
            "opt__beta2": 0.999,  # beta2
            "opt__eps_adam": 1e-08,  # eps. for numerical stability.
            "opt__amsgrad": False,  # Use amsgrad variant or not.
            # ========== LR scheduler: how to adjust the learning rate. ========
            "opt__lr_scheduler": True,  # if true, we use a learning rate
            # scheduler.
            # ========> MyStepLR: override torch.optim.lr_scheduler.StepLR
            "opt__name_lr_scheduler": "mystep",  # str name.
            "opt__step_size": 40,  # Frequency of which to adjust the lr.
            "opt__gamma": 0.1,  # the update coefficient: lr = gamma * lr.
            "opt__last_epoch": -1,  # the index of the last epoch where to stop
            # adjusting the LR.
            "opt__min_lr": 1e-7,  # minimum allowed value for lr.
            "opt__t_max": 100,  # T_max for cosine schedule.
            "opt__lr_classifier_ratio": 10.,  # Multiplicative factor on the
            # classifier layer (head) learning rate.
            "opt__clipgrad": 0.0,  # norm for gradient clip.
        },
        # ======================================================================
        #                              MODEL
        # ======================================================================
        "model": {
            "arch": constants.UNETFCAM,  # name of the model.
            # see: constants.nets.
            "encoder_name": constants.RESNET50,  # backbone for task of SEG.
            "encoder_weights": constants.IMAGENET,
            # pretrained weights or 'None'.
            "in_channels": 3,  # number of input channel.
            "large_maps": False,  # boo. if true, models oiutput maps of
            # 28x28. else, 7x7.
            "path_pre_trained": None,
            # None, `None` or a valid str-path. if str,
            # it is the absolute/relative path to the pretrained model. This can
            # be useful to resume training or to force using a filepath to some
            # pretrained weights.
            "strict": True,  # bool. Must be always be True. if True,
            # the pretrained model has to have the exact architecture as this
            # current model. if not, an error will be raise. if False, we do the
            # best. no error will be raised in case of mismatch.
            "support_background": False,  # useful for classification tasks
            # only:
            # std_cl, f_cl only. if true, an additional cam is used for the
            # background. this does not change the number of global
            # classification logits. irrelevant for segmentation task.
            "scale_in": 1.,  # float > 0.  how much to scale
            # the input image to not overflow the memory. This scaling is done
            # inside the model on the same device as the model.
            "freeze_cl": False,  # applied only for task F_CL. if true,
            # the classifier (encoder + head) is frozen.
            "folder_pre_trained_cl": None,
            # NAME of folder containing weights of
            # classifier. it must be in in 'pretrained' folder.
            # used in combination with `freeze_cl`. the folder contains
            # encoder.pt, head.pt weights of the encoder and head. the base name
            # of the folder is a tag used to make sure of compatibility between
            # the source (source of weights) and target model (to be frozen).
            # You do not need to set this parameters if `freeze_cl` is true.
            # we set it automatically when parsing the parameters.
            "spatial_dropout": 0.0,  # perform 2d dropout at the last feature
            # layer.
            "apvit_k": 160,  # int. k for apvit. max 196.
            "apvit_r": 0.9,  # float. r for apvit. ]0, 1].
            "apvit_attn_method": constants.ATT_SUM_ABS_1,
            "apvit_normalize_att": False,  # self-normalize/not the attention.
            "apvit_apply_self_att": False,  # apply attention to features.
            "apvit_hid_att_dim": 128,  # when using learnable attention: size
            # of the hidden layer.
            'do_segmentation': False,  # if true, we add a small segmentation
            # head to the classifier to perform binary segmentation with
            # single output map. this is used for action units heatmap
            # estimation.
        },
        # ======================================================================
        #                        MASKOUTFER DATA AUGMENTATION
        #                   ACTION UNITS/LANDMARKS-GUIDED DATA AUGMENTATION
        #               USE ACTION UNITS TO AUGMENT IMAGES VIA MASKING.
        # Train and eval set.
        # ======================================================================
        # Train -- apply data aug. over trainset -------------------------------
        "train_daug_mask_img_heatmap": False,  # bool. random standalone data
        # augmentation. masks the input image using a heatmap from either
        # facial action units or facial landmarks. this is independent of
        # align_atten_to_heatmap. applied only over trainset.
        "train_daug_mask_img_heatmap_type": constants.HEATMAP_AUNITS_LNMKS,
        # type of heatmap. See constants.HEATMAP_TYPES.
        "train_daug_mask_img_heatmap_bg_filler": constants.BG_F_BLACK,  # how to
        # fill the background. see constants.BG_FILLERS.
        "train_daug_mask_img_heatmap_p": 0.5,  # probability to apply this
        # transformation to a sample. [0, 1]. float.
        "train_daug_mask_img_heatmap_gauss_sigma": 30.,
        "train_daug_mask_img_heatmap_dilation": False,  # apply dilation over
        # roi mask determined by the heatmap.
        "train_daug_mask_img_heatmap_radius": 1,  # int. radius of the kernel
        # disk for dilation. int. > 0.
        "train_daug_mask_img_heatmap_normalize": False,  # bool. if ture,
        # heatmap is normalized into [0, 1] via min max.
        "train_daug_mask_img_heatmap_aus_seg_full": False,  # bool. if ture,
        # for the type 'HEATMAP_AUNITS_LEARNED_SEG', when the facial landmark
        # has failed, we show the full image (no action units are applied).
        # if false, we use the model prediction for this case.
        "train_daug_mask_img_heatmap_jaw": False,  # show jaw or not for
        # HEATMAP_LNDMKS.
        "train_daug_mask_img_heatmap_lndmk_variance": 64.,  # Multivariate normal
        # distribution (2d) is used to model a landmark in a heatmap.
        # Covariance matrix is variance * Identity.
        "train_daug_mask_img_heatmap_use_precomputed": False,  # if true, we use
        # pre-computed heatmaps. path stored in align_atten_to_heatmap_folder.
        # must be true for the type 'HEATMAP_AUNITS_LEARNED_SEG'.
        "train_daug_mask_img_heatmap_folder": '',  # path where the
        # pre-computed heatmaps (of landmarks or action units) are located. if
        # not set we set it automatically. we set a convention for the name
        # of the folder, and where it should be.  if we do not find the files,
        # we compute the heatmaps. to allow or prevent to use pre-computed
        # heatmaps, use 'train_daug_mask_img_heatmap_use_precomputed'.


        # Eval: apply data aug. over a set for evaluation. e.g. testset,
        # validset -------------------------------------------------------------
        "eval_daug_mask_img_heatmap": False,  # bool. standalone data
        # augmentation. masks the input image using a heatmap from either
        # facial action units or facial landmarks. this is independent of
        # align_atten_to_heatmap. applied only over a set for evaluation. it
        # is deterministic.
        "eval_daug_mask_img_heatmap_type": constants.HEATMAP_AUNITS_LNMKS,
        # type of heatmap. See constants.HEATMAP_TYPES.
        "eval_daug_mask_img_heatmap_bg_filler": constants.BG_F_BLACK,  # how to
        # fill the background. see constants.BG_FILLERS.
        "eval_daug_mask_img_heatmap_gauss_sigma": 30.,
        "eval_daug_mask_img_heatmap_dilation": False,  # apply dilation over
        # roi mask determined by the heatmap.
        "eval_daug_mask_img_heatmap_radius": 1,  # int. radius of the kernel
        # disk for dilation. int. > 0.
        "eval_daug_mask_img_heatmap_normalize": False,  # bool. if ture,
        # heatmap is normalized into [0, 1] via min max.
        "eval_daug_mask_img_heatmap_aus_seg_full": False,  # bool. if ture,
        # for the type 'HEATMAP_AUNITS_LEARNED_SEG', when the facial landmark
        # has failed, we show the full image (no action units are applied).
        # if false, we use the model prediction for this case.
        "eval_daug_mask_img_heatmap_jaw": False,  # show jaw or not for
        # HEATMAP_LNDMKS.
        "eval_daug_mask_img_heatmap_lndmk_variance": 64.,
        # Multivariate normal
        # distribution (2d) is used to model a landmark in a heatmap.
        # Covariance matrix is variance * Identity.
        "eval_daug_mask_img_heatmap_use_precomputed": False,  # if true, we use
        # pre-computed heatmaps. path stored in align_atten_to_heatmap_folder.
        # must be true for the type 'HEATMAP_AUNITS_LEARNED_SEG'.
        "eval_daug_mask_img_heatmap_folder": '',  # path where the
        # pre-computed heatmaps (of landmarks or action units) are located. if
        # not set we set it automatically. we set a convention for the name
        # of the folder, and where it should be.  if we do not find the files,
        # we compute the heatmaps. to allow or prevent to use pre-computed
        # heatmaps, use 'eval_daug_mask_img_heatmap_use_precomputed'.
        # ======================================================================
        # ======================================================================

        # ======================================================================
        #                         DATA SAMPLER: UNBALANCE CLASSES
        # ======================================================================
        "data_weighted_sampler": False,  # bool. if true, train samples are
        # drawn by considering the class weights.
        "data_weighted_sampler_w": constants.CLWFIXEDTECH1,  # technique to
        # estimate class weights.
        "data_weighted_sampler_per_cl": constants.PER_CL_NONE,  # str. see
        # constants.DATA_SAMPLER_PER_CL. Weighted sampler lead to
        # over-sampling of under-represented samples --> duplicate of
        # samples. This is less effective at mini-batch-samples especially
        # when there is a large unbalance in classes. To avoid samples
        # duplication in one epoch, we consider randomly sampling n samples per
        # class (uniform sampling) at each epoch.
        # n: number of samples in the class the most under-representative.
        # when this is set to 'PER_CL_NONE', it is not applied and weighted
        # sampling is used instead. This has effect when set to 'PER_CL_MIN_CL'.
        # if it is set to 'PER_CL_MIN_CL', and there is a large class
        # unbalance, this will reduce the number of samples seen per-epoch to
        # n * nbr_classes. therefore, over-represented samples will not be
        # fully covered since we always sample only n samples from them. in
        # this case, we recommend using larger max epochs to benefit more
        # from over-represented classes.
        # ======================================================================
        #                    LAYERWISE ATTENTION LOSS
        #                      ONLY FOR std_CL TASK.
        # ======================================================================
        # align layerwise attention to some heat map.
        "align_atten_to_heatmap": False,  # use or not layerwise attention
        # alignment with a heatmap.
        "align_atten_to_heatmap_type_heatmap": constants.HEATMAP_LNDMKS,
        # type of the heatmap. see constants.HEATMAP_TYPES.
        "align_atten_to_heatmap_normalize": False,  # bool. if ture, heatmap
        # is normalized into [0, 1] via min max.
        "align_atten_to_heatmap_aus_seg_full": False,  # bool. if ture,
        # for the type 'HEATMAP_AUNITS_LEARNED_SEG', when the facial landmark
        # has failed, we show the full image (no action units are applied).
        # if false, we use the model prediction for this case.
        "align_atten_to_heatmap_jaw": False,  # show jaw or not for
        # HEATMAP_LNDMKS.
        "align_atten_to_heatmap_elb": False,  # if true, elb will be use for
        # alignment. applied only when align_atten_to_heatmap_type_heatmap
        # cosine similarity:  constants.A_COSINE, constants.A_STD_COSINE.
        "align_atten_to_heatmap_lndmk_variance": 64.,  # Multivariate normal
        # distribution (2d) is used to model a landmark in a heatmap.
        # Covariance matrix is variance * Identity.
        'align_atten_to_heatmap_layers': '5',  # which layers should be used
        # for alignment. see constants.MAX_NBR_LAYERS. several layers can be
        # considered via '-'. eg: '3-4-5'. Layers count starts from 0 (not
        # 1). But allowed layers are >= 1. the first layer holds the same
        # input image (cant be used for alignment).
        "align_atten_to_heatmap_align_type": constants.ALIGN_AVG,
        # how attention is estimated from features. see constants.ALIGNMENTS.
        "align_atten_to_heatmap_norm_att": constants.NORM_NONE,  # how to
        # normalize attention. see: constants.NORMS_ATTENTION.
        "align_atten_to_heatmap_p": 1.,  # Bernoulli dist. param p. used when
        # 'align_type' is random. Represents the percentage of feature maps
        # used to estimate the per-layer attention map. p% random feature
        # maps will be selected at each iteration for each sample.
        "align_atten_to_heatmap_q": 1.,  # percentage of maps to be used.
        # we select fixed q% of maps for all samples and at each
        # iteration and compute their average to estimate the attention.
        # applicable only for the case 'ALIGN_AVG_FIXED_Q'.
        "align_atten_to_heatmap_loss": constants.A_COSINE,  # alignment loss.
        # see constants.ALIGN_LOSSES
        "align_atten_to_heatmap_lambda": 1.,  # lambda for this loss.
        "align_atten_to_heatmap_start_ep": 0,  # epoch when to start this loss.
        "align_atten_to_heatmap_end_ep": -1,  # epoch when to stop using this
        # loss. -1: never stop.
        "align_atten_to_heatmap_scale_to": constants.SCALE_TO_ATTEN,  # scale
        # attention to heatmap size or, scale heatmap to attention size.
        "align_atten_to_heatmap_folder": '',  # path where the
        # pre-computed heatmaps (of landmarks or action units) are located. if
        # not set we set it automatically. we set a convention for the name
        # of the folder, and where it should be.  if we do not find the files,
        # we compute the heatmaps. to allow or prevent to use pre-computed
        # heatmaps, use 'align_atten_to_heatmap_use_precomputed'.
        "align_atten_to_heatmap_use_precomputed": False,  # if true, we use
        # pre-computed heatmaps. path stored in align_atten_to_heatmap_folder.
        # must be true for the type 'HEATMAP_AUNITS_LEARNED_SEG'.
        "align_atten_to_heatmap_use_self_atten": False,  # if true,
        # for each selected layer 'align_atten_to_heatmap_layers', we compute
        # its attention (as the avg of the feature maps), and perform masking
        # of the features (multiplication), before going to the next layer.
        # ======================================================================
        #                   LAYERWISE ORTHOGONAL FEATURES
        # ======================================================================
        # free orthogonality: no constraint.
        "free_orth_ft": False,  # use/not free (unguided) ortho ft.
        "free_orth_ft_lambda": 1.,  # float. lambda.
        "free_orth_ft_start_ep": 0,  # start epoch
        "free_orth_ft_end_ep": -1,  # end epoch
        "free_orth_ft_elb": False,  # if true ELB is used. else, penalty.
        "free_orth_ft_layers": '5',  # layers to apply to this loss.
        # see constants.MAX_NBR_LAYERS + max 2. we can add up to 2 dense layers
        # for the classification head.
        # several layers can be
        # considered via '-'. eg: '3-4-5'. Layers count starts from 0 (not
        # 1). But allowed layers are >= 1. the first layer holds the same
        # input image (cant be used for alignment).
        "free_orth_ft_same_cl": True,  # apply or not for samples with same
        # class.
        "free_orth_ft_diff_cl": False,  # apply or not for samples with
        # different classes. free_orth_ft_same_cl or free_orth_ft_diff_cl
        # must be true.
        # Guided orthogonality: feature basis are predefined.
        "guid_orth_ft": False,  # use/not free (unguided) ortho ft.
        "guid_orth_ft_lambda": 1.,  # float. lambda.
        "guid_orth_ft_start_ep": 0,  # start epoch
        "guid_orth_ft_end_ep": -1,  # end epoch
        "guid_orth_ft_elb": False,  # if true ELB is used. else, penalty.
        "guid_orth_ft_layers": '5',  # layers to apply to this loss.
        # see constants.MAX_NBR_LAYERS + max 2. we can add up to 2 dense
        # layer in the classification head. several layers
        # can be considered via '-'. eg: '3-4-5'. Layers count starts from 0
        # (not 1). But allowed layers are >= 1. the first layer holds the same
        # input image (cant be used for alignment).
        # ======================================================================
        #                    High Entropy probs.
        # ======================================================================
        "high_entropy": False,  # bool. use or not max entropy of probability
        # distribution of classes. This aims to regularize the entropy and
        # push it to be high making the model less certain (reduce overfitting).
        "high_entropy_type": constants.MAX_ENTROPY,  # str. type of high
        # entropy. see constants.HIGH_ENTROPY_REGS.
        "high_entropy_lambda": 1.,  # lambda.
        "high_entropy_a": 0.,  # float. ]0, 1[. alpha for general max entropy
        "high_entropy_start_ep": 0,  # int. start epoch to apply this loss.
        "high_entropy_end_ep": -1,  # int. when to stop this loss. -1: never.
        # ======================================================================
        #                    Score constraint loss
        # ======================================================================
        "con_scores": False,  # bool. use or not constraint on per-class
        # score. if true, score (logit) of true class is constrained to be
        # greater as possible than any other classes. USE ELB.
        "con_scores_lambda": 1.,  # lambda.
        "con_scores_min": 0.,  # float. >= 0. minim difference allowed.
        "con_scores_start_ep": 0,  # int. start epoch to apply this loss.
        "con_scores_end_ep": -1,  # int. when to stop this loss. -1: never.
        # ======================================================================
        #                    Cross Entropy loss
        # ======================================================================
        "ce": False,  # Cross entropy for image classification.
        "ce_lambda": 1.,  # lambda for cross-entropy.
        "ce_start_ep": 0,  # epoch when to start ce loss.
        "ce_end_ep": -1,  # epoch when to stop using ce loss. -1: never stop.
        "ce_label_smoothing": 0.0,  # label smoothing. [0., 1.]

        "aus_seg": False,  # BCE for action units segmentation task. if true,
        # do_segmentation must be true. BCEWithLogitsLoss is used within.
        # mainly, this is made to train to segment action units heatmaps. but
        # you can segment other things: landmarks heatmap.
        "aus_seg_lambda": 1.,  # lambda of this loss.
        "aus_seg_start_ep": 0,  # epoch when to start this loss.
        "aus_seg_end_ep": -1,  # epoch when to stop using this loss. -1: never
        # stop.
        "aus_seg_heatmap_type": constants.HEATMAP_AUNITS_LNMKS,
        # type of heatmap. See constants.HEATMAP_TYPES.
        "aus_seg_normalize": False,  # bool. if ture,
        # heatmap is normalized into [0, 1] via min max.
        "aus_seg_aus_seg_full": False,  # bool. if ture,
        # for the type 'HEATMAP_AUNITS_LEARNED_SEG', when the facial landmark
        # has failed, we show the full image (no action units are applied).
        # if false, we use the model prediction for this case.
        "aus_seg_jaw": False,  # show jaw or not for
        # HEATMAP_LNDMKS.
        "aus_seg_lndmk_variance": 64.,
        # Multivariate normal
        # distribution (2d) is used to model a landmark in a heatmap.
        # Covariance matrix is variance * Identity.
        "aus_seg_use_precomputed": False,  # if true, we use
        # pre-computed heatmaps. path stored in align_atten_to_heatmap_folder.
        # must be true for the type 'HEATMAP_AUNITS_LEARNED_SEG'.
        "aus_seg_folder": '',  # path where the
        # pre-computed heatmaps (of landmarks or action units) are located. if
        # not set we set it automatically. we set a convention for the name
        # of the folder, and where it should be.  if we do not find the files,
        # we compute the heatmaps. to allow or prevent to use pre-computed
        # heatmaps, use 'aus_seg_use_precomputed'.

        # mse loss for classification.
        "mse": False,  # MSE loss for image classification.
        "mse_lambda": 1.,  # lambda for mse.
        "mse_start_ep": 0,  # epoch when to start mse loss.
        "mse_end_ep": -1,  # epoch when to stop using mse loss. -1: never stop.

        # attention size loss.
        "att_sz": False,  # apply attention size loss or not.
        "att_sz_lambda": 1.,  # lambda of loss.
        "att_sz_start_ep": 0,  # epoch when to start this loss.
        "att_sz_end_ep": -1,  # epoch when to stop using this loss. -1: never
        # stop.
        "att_sz_bounds": '0*1',  # lower and upper bound of size. low in [0,
        # 1[. up in ]0, 1]. separate bounds by '*'.

        # low entropy over attention loss.
        "att_ent_sz": False,  # apply atention low entropy loss or not.
        "att_ent_lambda": 1.,  # lambda of loss.
        "att_ent_start_ep": 0,  # epoch when to start this loss.
        "att_ent_end_ep": -1,  # epoch when to stop using this loss. -1: never
        # stop.

        # mean absolute error loss for classification.
        "mae": False,  # MAE loss for image classification.
        "mae_lambda": 1.,  # lambda for mae.
        "mae_start_ep": 0,  # epoch when to start mae loss.
        "mae_end_ep": -1,  # epoch when to stop using mae loss. -1: never stop.


        # multi-class focal oss
        "mtl_focal": False,  # Multi-class focal loss for image classification.
        "mtl_focal_lambda": 1.,  # lambda for cross-entropy.
        "mtl_focal_start_ep": 0,  # epoch when to start ce loss.
        "mtl_focal_end_ep": -1,  # epoch when to stop using ce loss. -1: never stop.
        "mtl_focal_alpha": 1.0,  # alpha focal loss > 0.
        "mtl_focal_gamma": 0.0,  # gamma focal loss >= 0.

        # weight of linear classifier: orthogonality.
        "ortho_lw": False,  # Orthogonality of the weight of the linear
        # classifier.
        "ortho_lw_method": constants.ORTHOG_SOFT,  # orthogonality tech. see
        # constants.ORTHOG_TECHS.
        "ortho_lw_spec_iter": 5,  # int. number of iterations of power
        # iteration. techn to estimate spectral norm when the method is SRIP.
        "ortho_lw_lambda": 1.,  # lambda.
        "ortho_lw_start_ep": 0,  # epoch when to start w_sparsity loss.
        "ortho_lw_end_ep": -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # weight sparsity loss
        "w_sparsity": False,  # Weight sparsity (l1 norm).
        "w_sparsity_lambda": 1.,  # lambda.
        "w_sparsity_start_ep": 0,  # epoch when to start w_sparsity loss.
        "w_sparsity_end_ep": -1,  # epoch when to stop using w_sparsity
        # loss. -1: never stop.

        # linear features sparsity: input of the linea layer at the net output.
        "sparse_lf": False,  # sparse linear features
        "sparse_lf_lambda": 1.,  # lambda.
        "sparse_lf_method": constants.SPARSE_L1,  # lambda.
        "sparse_lf_p": 1.,  # percentage of features to apply this loss to. (
        # top p% low values. p ]0, 1])
        "sparse_lf_c": 0.,  # norm <= c. c >= 0.
        "sparse_lf_use_elb": False,  # use ELB or not.
        "sparse_lf_average_it": False,  # average the loss by the p% of
        # features. does not apply for norm inf.
        "sparse_lf_start_ep": 0,  # epoch when to start w_sparsity loss.
        "sparse_lf_end_ep": -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # linear weight sparsity: weight of the linea layer at the net output.
        "sparse_lw": False,  # sparse linear weight
        "sparse_lw_lambda": 1.,  # lambda.
        "sparse_lw_method": constants.SPARSE_L1,  # lambda.
        "sparse_lw_p": 1.,  # percentage of features to apply this loss to. (
        # top p% low values. p ]0, 1])
        "sparse_lw_c": 0.,  # norm <= c. c >= 0.
        "sparse_lw_use_elb": False,  # use ELB or not.
        "sparse_lw_average_it": False,  # average the loss by the p% of
        # features. does not apply for norm inf.
        "sparse_lw_start_ep": 0,  # epoch when to start w_sparsity loss.
        "sparse_lw_end_ep": -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # Sparse APVIT attention
        "sparse_at": False,  # sparse APVIT attention.
        "sparse_at_lambda": 1.,  # lambda.
        "sparse_at_method": constants.SPARSE_L1,  # lambda.
        "sparse_at_p": 1.,  # percentage of features to apply this loss to. (
        # top p% low values. p ]0, 1])
        "sparse_at_c": 0.,  # norm <= c. c >= 0.
        "sparse_at_use_elb": False,  # use ELB or not.
        "sparse_at_average_it": False,  # average the loss by the p% of
        # features. does not apply for norm inf.
        "sparse_at_start_ep": 0,  # epoch when to start w_sparsity loss.
        "sparse_at_end_ep": -1,  # epoch when to stop using this loss.
        # -1: never stop.

        # self-cost sensitive.
        "s_cost_s": False,  # self-cost sensitive.
        "s_cost_s_lambda": 1.,  # lambda for cross-entropy.
        "s_cost_s_start_ep": 0,  # epoch when to start ce loss.
        "s_cost_s_end_ep": -1,  # epoch when to stop using this loss. -1:
        # never stop.
        "s_cost_s_apply_to": constants.LOGITS,  # apply to logits/probs.
        "s_cost_s_norm": constants.NORM_SCORES_MAX,  # how to normalize
        # scores. see constants.NORM_SCORES
        "s_cost_s_confusion_func": constants.EXP_CONF_FUNC,  # confusion
        # function. see constants.CONFUSION_FUNCS
        "s_cost_s_topk": -1,  # int. apply loss to top k classes (ordered
        # from high to low loss). -1: take all. topk starts from 1.
        "s_cost_s_reduction": constants.REDUCE_MEAN,  # str. how to reduce
        # the loss for single sample: constants.REDUCTIONS.

        # ======================================================================
        #                    STD_CL TECH.
        # ======================================================================
        "std_cl_w_style": constants.CLWNONE,  # how to use per-class weight
        # to unbalanced classes issue. see constants.CLW.
        # ======================================================================
        #                    CLASSIFICATION SPATIAL POOLING
        # ======================================================================
        "method": constants.METHOD_WILDCAT,
        "spatial_pooling": constants.WILDCATHEAD,
        # ======================================================================
        #                        SPATIAL POOLING:
        #                            WILDCAT
        # ======================================================================
        "wc_modalities": 5,
        "wc_kmax": 0.5,
        "wc_kmin": 0.1,
        "wc_alpha": 0.6,
        "wc_dropout": 0.0,

        # CUTMIX.
        "cutmix_beta": 1.,  # float. hyper-parameter of beta distribution.
        "cutmix_prob": 1.,  # probability to perform cutmix over full minibatch.

        # acol
        'acol_drop_threshold': 0.1,  # float. threshold value. better [0, 1]

        # PRM
        'prm_ks': 3,  # kernel size for peak stimulation.
        'prm_st': 1,  # kernel stride.

        # ADL
        "adl_drop_rate": 0.4,  # float. [0, 1]. how often to do drop mask.
        "adl_drop_threshold": 0.1,  # float. [0, 1]. percentage of maximum
        # intensity: val > adl_drop_threshold * maximum_intensity.

        # ======================================================================
        # Dense layers at the classification head.
        "dense_dims": '',  # max 2 dims. '1024-512' to specify 2 dense layers
        # with dims: 1024, and 512. if none, set it to '' or 'None', or None.
        # for resnet max is 2 layers. for apvit, is not limited.
        "dense_dropout": 0.,  # float. dropout at dense layer at the
        # classification head.
        # ================== LSE pooling
        "lse_r": 10.,  # r for logsumexp pooling.
        # ======================================================================
        #                          Segmentation mode
        # ======================================================================
        "seg_mode": constants.BINARY_MODE,
        # SEGMENTATION mode: bin only always.
        "task": constants.STD_CL,  # task: standard classification,
        # full classification (FCAM).
        "master_selection_metric": constants.CL_ACCURACY_MTR,  # model
        # selection metric over validation.
        "multi_label_flag": False,
        # whether the dataset has multi-label or not.
        # ======================================================================
        #                          ELB
        # ======================================================================
        "elb_init_t": 1.,  # used for ELB.
        "elb_max_t": 10.,  # used for ELB.
        "elb_mulcoef": 1.01,  # used for ELB.
        # ======================================================================
        #                            CONSTRAINTS:
        #                     'SuperResolution', sr
        #                     'ConRanFieldFcams', crf_fc
        #                     'EntropyFcams', entropy_fc
        #                     'PartUncerknowEntropyLowCams', partuncertentro_lc
        #                     'PartCertKnowLowCams', partcert_lc
        #                     'MinSizeNegativeLowCams', min_sizeneg_lc
        #                     'MaxSizePositiveLowCams', max_sizepos_lc
        #                     'MaxSizePositiveFcams' max_sizepos_fc
        # ======================================================================
        "max_epochs": 150,  # number of training epochs.
        # -----------------------  FCAM
        "sl_fc": False,  # use self-learning over fcams.
        "sl_fc_lambda": 1.,  # lambda for self-learning over fcams
        "sl_start_ep": 0,  # epoch when to start sl loss.
        "sl_end_ep": -1,  # epoch when to stop using sl loss. -1: never stop.
        "sl_min": 10,  # int. number of pixels to be considered
        # background (after sorting all pixels).
        "sl_max": 10,  # number of pixels to be considered
        # foreground (after sorting all pixels).
        "sl_min_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_min from.
        "sl_max_ext": 1000,  # the extent of pixels to consider for selecting
        # sl_max.
        "sl_block": 1,  # size of the block. instead of selecting from pixel,
        # we allow initial selection from grid created from blocks of size
        # sl_blockxsl_block. them, for each selected block, we select a random
        # pixel. this helps selecting from fare ways regions. if you don't want
        # to use blocks, set this to 1 where the selection is done directly over
        # pixels without passing through blocks.
        "sl_ksz": 1,  # int, kernel size for dilation around the pixel. must be
        # odd number.
        'sl_min_p': .2,  # percentage of pixels to be used for background
        # sampling. percentage from entire image size.
        'sl_fg_erode_k': 11,  # int. size of erosion kernel to clean foreground.
        'sl_fg_erode_iter': 1,  # int. number of erosions for foreground.
        # ----------------------- FCAM
        "crf_fc": False,  # use or not crf over fcams.  (penalty)
        "crf_lambda": 2.e-9,  # crf lambda
        "crf_sigma_rgb": 15.,
        "crf_sigma_xy": 100.,
        "crf_scale": 1.,  # scale factor for input, segm.
        "crf_start_ep": 0,  # epoch when to start crf loss.
        "crf_end_ep": -1,  # epoch when to stop using crf loss. -1: never stop.
        # ======================================================================
        # ======================================================================
        #                                EXTRA
        # ======================================================================
        # ======================================================================
        # ----------------------- FCAM
        "entropy_fc": False,  # use or not the entropy over fcams. (penalty)
        "entropy_fc_lambda": 1.,
        # -----------------------  FCAM
        "max_sizepos_fc": False,  # use absolute size (unsupervised) over all
        # fcams. (elb)
        "max_sizepos_fc_lambda": 1.,
        "max_sizepos_fc_start_ep": 0,  # epoch when to start maxsz loss.
        "max_sizepos_fc_end_ep": -1,  # epoch when to stop using mxsz loss. -1:
        # never stop.
        # ----------------------------------------------------------------------
        #                         Ordinal classification (OC)
        # ----------------------------------------------------------------------
        # mean oc: (expected label - true label)** 2 <= eps.
        "oc_mean": False,  # use or not min expected class.
        "oc_mean_lambda": 1.,  # lambda for oc_mean.
        "oc_mean_epsilon": 0.01,  # (expected label - true label)**2 <= eps.
        "oc_mean_elb": False,  # use or not elb.
        "oc_mean_start_ep": 0,  # epoch when to start pc_mean loss.
        "oc_mean_end_ep": -1,  # epoch when to stop using oc_mean loss. -1:
        # never stop.

        # variance oc: variance label <= eps.
        "oc_var": False,  # use or not min expected class.
        "oc_var_lambda": 1.,  # lambda for oc_mean.
        "oc_var_epsilon": 0.01,  # label variance <= eps.
        "oc_var_elb": False,  # use or not elb.
        "oc_var_start_ep": 0,  # epoch when to start pc_mean loss.
        "oc_var_end_ep": -1,  # epoch when to stop using oc_mean loss. -1:
        # never stop.

        # Unimodality via inequalities oc:  via. elb.
        "oc_unim_inq": False,  # use or not.
        "oc_unim_inq_lambda": 1.,  # lambda.
        "oc_unim_inq_type": constants.LOGITS,  # apply over logits or probs
        "oc_unim_inq_start_ep": 0,  # epoch when to start this loss.
        "oc_unim_inq_end_ep": -1,  # epoch when to stop using this loss. -1:
        # never stop.
        # ----------------------------------------------------------------------
        # ----------------------- NOT USED
        # ------------------------------- GENERIC
        "im_rec": False,  # image reconstruction loss.
        "im_rec_lambda": 1.,
        "im_rec_elb": False,  # use or not elb for image reconstruction.
        # ----------------------------- NOT USED
        # ----------------------------------------------------------------------
        # ======================================================================
        # ======================================================================
        # ======================================================================

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ GENERIC
        'seg_ignore_idx': -255,  # ignore index for segmentation alignment.
        'amp': False,  # if true, use automatic mixed-precision for training
        'amp_eval': False,  # if true, amp is used for inference.
        # ======================================================================
        #                             DDP:
        # NOT CC(): means single machine.  CC(): multiple nodes.
        # ======================================================================
        'local_rank': 0,  # int. for not CC(). must be 0 if just one node.
        'local_world_size': 1,  # int. for not CC(). number of gpus to use.
        'rank': 0,  # int. global rank. useful for CC(). 0 otherwise. will be
        # set automatically.
        'init_method': '',  # str. CC(). init method. needs to be defined.
        # will be be determined automatically.
        'dist_backend': constants.NCCL,  # str. CC() or not CC(). distributed
        # backend.
        'world_size': 1,  # init. CC(). total number of gpus. will be
        # determined automatically.
        'is_master': False,  # will be set automatically if this process is
        # the master.
        'is_node_master': False,  # will be set auto. true if this process is
        # has local rank = 0.
        'c_cudaid': 0,  # int. current cuda id. auto-set.
        'distributed': False,  # bool. auto-set. indicates whether we are
        # using more than 1 gpu. This will help differentiate when accessing to
        # model.attributes when it is wrapped with a ddp and when not.
    }

    pre = constants.FORMAT_DEBUG.split('_')[0]
    dsname = args['dataset']
    if dsname.startswith(pre):
        dsname = dsname.replace('{}_'.format(pre), '')

    args['data_paths'] = configure_data_paths(args, dsname)
    args['metadata_root'] = join(args['metadata_root'], args['dataset'])

    openimg_ds = constants.OpenImages
    if openimg_ds.startswith(pre):
        openimg_ds = dsname.replace('{}_'.format(pre), '')
    args['mask_root'] = join(args['mask_root'], openimg_ds)

    data_cams = join(root_dir, constants.DATA_CAMS)
    if not os.path.isdir(data_cams):
        os.makedirs(data_cams, exist_ok=True)

    return args


if __name__ == '__main__':
    args = get_config(constants.RAFDB)
    print(args['metadata_root'])
