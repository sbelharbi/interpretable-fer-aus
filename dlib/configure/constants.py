# possible tasks
STD_CL = "STD_CL"  # standard classification using only the encoder features.
# ouputs: logits, cams.
F_CL = 'F_CL'  # standard classification but using the decoder features.
# outputs: logits, cams.
SEG = "SEGMENTATION"  # standard supervised segmentation. outputs:
# segmentation masks.

TASKS = [STD_CL, F_CL, SEG]


# name of the classifier head (pooling operation)
WILDCATHEAD = "WildCatCLHead"

GAP = 'GAP'
WGAP = 'WGAP'
ACOL = 'ACOL'
MAXPOOL = 'MaxPool'
LSEPOOL = 'LogSumExpPool'
PRM = 'PRM'
NONEPOOL = 'NONE'

SPATIAL_POOLINGS = [WILDCATHEAD, GAP, ACOL, WGAP, MAXPOOL, LSEPOOL, PRM,
                    NONEPOOL]

# methods
METHOD_WILDCAT = 'WILDCAT'  # pooling: WILDCATHEAD
METHOD_GAP = 'GAP'  # pooling: GAP

METHOD_MAXPOOL = 'MaxPOL'  # pooling: MAXPOOL
METHOD_LSE = 'LogSumExp'  # pooling: logsumexp.

# -- all methods below use WGAP.

METHOD_CAM = 'CAM'
METHOD_SCORECAM = 'ScoreCAM'
METHOD_SSCAM = 'SSCAM'
METHOD_ISCAM = 'ISCAM'

METHOD_GRADCAM = 'GradCam'
METHOD_GRADCAMPP = 'GradCAMpp'
METHOD_SMOOTHGRADCAMPP = 'SmoothGradCAMpp'
METHOD_XGRADCAM = 'XGradCAM'
METHOD_LAYERCAM = 'LayerCAM'

# new wsol methods.
METHOD_CUTMIX = 'CutMIX'

METHOD_ADL = 'ADL'
METHOD_ACOL = 'ACoL'

METHOD_PRM = 'PRM'
METHOD_TSCAM = 'TSCAM'
METHOD_APVIT = 'APVIT'


METHODS = [METHOD_WILDCAT,
           METHOD_GAP,
           METHOD_MAXPOOL,
           METHOD_LSE,
           METHOD_CAM,
           METHOD_SCORECAM,
           METHOD_SSCAM,
           METHOD_ISCAM,
           METHOD_GRADCAM,
           METHOD_GRADCAMPP,
           METHOD_SMOOTHGRADCAMPP,
           METHOD_XGRADCAM,
           METHOD_LAYERCAM,
           METHOD_CUTMIX,
           METHOD_ACOL,
           METHOD_PRM,
           METHOD_TSCAM,
           METHOD_ADL,
           METHOD_APVIT
           ]

METHOD_2_POOLINGHEAD = {
        METHOD_WILDCAT: WILDCATHEAD,
        METHOD_GAP: GAP,
        METHOD_MAXPOOL: MAXPOOL,
        METHOD_LSE: LSEPOOL,
        METHOD_CAM: WGAP,
        METHOD_SCORECAM: WGAP,
        METHOD_SSCAM: WGAP,
        METHOD_ISCAM: WGAP,
        METHOD_GRADCAM: WGAP,
        METHOD_GRADCAMPP: WGAP,
        METHOD_SMOOTHGRADCAMPP: WGAP,
        METHOD_XGRADCAM: WGAP,
        METHOD_LAYERCAM: WGAP,
        METHOD_CUTMIX: WGAP,
        METHOD_ACOL: ACOL,
        METHOD_PRM: PRM,
        METHOD_TSCAM: GAP,
        METHOD_ADL: WGAP,
        METHOD_APVIT: NONEPOOL
    }

METHOD_REQU_GRAD = {
        METHOD_WILDCAT: False,
        METHOD_GAP: False,
        METHOD_MAXPOOL: False,
        METHOD_LSE: False,
        METHOD_CAM: False,
        METHOD_SCORECAM: False,
        METHOD_SSCAM: False,
        METHOD_ISCAM: False,
        METHOD_GRADCAM: True,
        METHOD_GRADCAMPP: True,
        METHOD_SMOOTHGRADCAMPP: True,
        METHOD_XGRADCAM: True,
        METHOD_LAYERCAM: True,
        METHOD_CUTMIX: False,
        METHOD_ACOL: False,
        METHOD_PRM: False,
        METHOD_TSCAM: False,
        METHOD_ADL: False,
        METHOD_APVIT: False
}

METHOD_LITERAL_NAMES = {
        METHOD_WILDCAT: 'WILDCAT',
        METHOD_GAP: 'GAP',
        METHOD_MAXPOOL: 'MaxPool',
        METHOD_LSE: 'LSEPool',
        METHOD_CAM: 'CAM',
        METHOD_SCORECAM: 'ScoreCAM',
        METHOD_SSCAM: 'SSCAM',
        METHOD_ISCAM: 'ISCAM',
        METHOD_GRADCAM: 'GradCAM',
        METHOD_GRADCAMPP: 'GradCam++',
        METHOD_SMOOTHGRADCAMPP: 'Smooth-GradCAM++',
        METHOD_XGRADCAM: 'XGradCAM',
        METHOD_LAYERCAM: 'LayerCAM',
        METHOD_CUTMIX: 'CutMix',
        METHOD_ACOL: 'ACol',
        METHOD_PRM: 'PRM',
        METHOD_TSCAM: 'TS-CAM',
        METHOD_ADL: 'ADL',
        METHOD_APVIT: 'APViT'
}
# datasets mode
DS_TRAIN = "TRAIN"  # for training
DS_EVAL = "EVAL"  # for evaluation.

SET_MODES = [DS_TRAIN, DS_EVAL]

# Tags for samples
L = 0  # Labeled samples

samples_tags = [L]  # list of possible sample tags.

# pixel-wise supervision:
ORACLE = "ORACLE"  # provided by an oracle.
SELF_LEARNED = "SELF-LEARNED"  # self-learned.
VOID = "VOID"  # None

# segmentation modes.
#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


# pretraining
IMAGENET = "imagenet"
VGGFACE2 = 'VGGFace2'
PRETRAINED = [IMAGENET, VGGFACE2]

# archs
STDCLASSIFIER = "STDClassifier"
TSCAMCLASSIFIER = 'TSCAMClassifier'
APVITCLASSIFIER = 'APVITClassifier'

UNETFCAM = 'UnetFCAM'  # USED

UNET = "Unet"
UNETPLUPLUS = "UnetPlusPlus"
MANET = "MAnet"
LINKNET = "Linknet"
FPN = "FPN"
PSPNET = "PSPNet"
DEEPLABV3 = "DeepLabV3"
DEEPLABV3PLUS = "DeepLabV3Plus"
PAN = "PAN"

ARCHS = [STDCLASSIFIER, TSCAMCLASSIFIER, APVITCLASSIFIER, UNETFCAM]

# std cld method to arch.
STD_CL_METHOD_2_ARCH = {
    METHOD_WILDCAT: STDCLASSIFIER,
    METHOD_GAP: STDCLASSIFIER,
    METHOD_MAXPOOL: STDCLASSIFIER,
    METHOD_LSE: STDCLASSIFIER,
    METHOD_CAM: STDCLASSIFIER,
    METHOD_SCORECAM: STDCLASSIFIER,
    METHOD_SSCAM: STDCLASSIFIER,
    METHOD_ISCAM: STDCLASSIFIER,
    METHOD_GRADCAM: STDCLASSIFIER,
    METHOD_GRADCAMPP: STDCLASSIFIER,
    METHOD_SMOOTHGRADCAMPP: STDCLASSIFIER,
    METHOD_XGRADCAM: STDCLASSIFIER,
    METHOD_LAYERCAM: STDCLASSIFIER,
    METHOD_ACOL: STDCLASSIFIER,
    METHOD_CUTMIX: STDCLASSIFIER,
    METHOD_PRM: STDCLASSIFIER,
    METHOD_TSCAM: TSCAMCLASSIFIER,
    METHOD_ADL: STDCLASSIFIER,
    METHOD_APVIT: APVITCLASSIFIER
}

# ecnoders

APVIT = 'apvit'  # https://arxiv.org/pdf/2212.05463.pdf

#  resnet
RESNET18 = 'resnet18'
RESNET34 = 'resnet34'
RESNET50 = 'resnet50'
RESNET101 = 'resnet101'
RESNET152 = 'resnet152'

# vgg
VGG16 = 'vgg16'

# inceptionv3
INCEPTIONV3 = 'inceptionv3'

# DEIT TSCAM
DEIT_TSCAM_BASE_P16_224 = 'deit_tscam_base_patch16_224'
DEIT_TSCAM_SMALL_P16_224 = 'deit_tscam_small_patch16_224'
DEIT_TSCAM_TINY_P16_224 = 'deit_tscam_tiny_patch16_224'

BACKBONES = [RESNET18,
             RESNET34,
             RESNET50,
             RESNET101,
             RESNET152,
             VGG16,
             INCEPTIONV3,
             APVIT,
             DEIT_TSCAM_SMALL_P16_224,
             DEIT_TSCAM_BASE_P16_224,
             DEIT_TSCAM_TINY_P16_224
             ]

TSCAM_BACKBONES = [DEIT_TSCAM_SMALL_P16_224,
                   DEIT_TSCAM_BASE_P16_224,
                   DEIT_TSCAM_TINY_P16_224]
# ------------------------------------------------------------------------------

# datasets
DEBUG = False


ILSVRC = "ILSVRC"
CUB = "CUB"
OpenImages = 'OpenImages'
RAFDB = 'RAF-DB'
AFFECTNET = 'AffectNet'

FORMAT_DEBUG = 'DEBUG_{}'
if DEBUG:
    RAFDB = FORMAT_DEBUG.format(RAFDB)
    AFFECTNET = FORMAT_DEBUG.format(AFFECTNET)
    CUB = FORMAT_DEBUG.format(CUB)
    ILSVRC = FORMAT_DEBUG.format(ILSVRC)
    OpenImages = FORMAT_DEBUG.format(OpenImages)


DATASETS = [RAFDB, AFFECTNET, CUB, ILSVRC, OpenImages]

BUCKET_SZ = 8

NBR_CHUNKS_TR = {
    'ILSVRC': 30 * 8,  # 30 *8: ~5k per chunk
    'DEBUG_ILSVRC': 3 * 8,  # 3 *8: ~5k per chunk.
    # no chunking:
    'RAF-DB': -1,
    'AffectNet': -1,
    'CUB': -1,
    'DEBUG_CUB': -1,
    'OpenImages': -1,
    'DEBUG_OpenImages': -1,
}

RELATIVE_META_ROOT = './folds/wsol-done-right-splits'

NUMBER_CLASSES = {
    ILSVRC: 1000,
    CUB: 200,
    OpenImages: 100,
    RAFDB: 7,
    AFFECTNET: 7
}

CROP_SIZE = 224
RESIZE_SIZE = 256

SZ224 = 224
SZ256 = 256
SZ112 = 112

# ================= check points
LAST = 'last'
BEST = 'best'

# ==============================================================================

# Colours
COLOR_WHITE = "white"
COLOR_BLACK = "black"

# backbones.

# =================================================
NCOLS = 80  # tqdm ncols.

# stages:
STGS_TR = "TRAIN"
STGS_EV = "EVAL"


# datasets:
TRAINSET = 'train'
VALIDSET = 'val'
TESTSET = 'test'

SPLITS = [TRAINSET, VALIDSET, TESTSET]

# image range: [0, 1] --> Sigmoid. [-1, 1]: TANH
RANGE_TANH = "tanh"
RANGE_SIGMOID = 'sigmoid'

# ==============================================================================
# cams extractor
TRG_LAYERS = {
            RESNET50: 'encoder.layer4.2.relu3',
            RESNET18: 'encoder.layer4.2.relu3',
            RESNET34: 'encoder.layer4.2.relu3',
            RESNET101: 'encoder.layer4.2.relu3',
            RESNET152: 'encoder.layer4.2.relu3',
            VGG16: 'encoder.relu',
            INCEPTIONV3: 'encoder.SPG_A3_2b.2',
            APVIT: 'None'
        }
FC_LAYERS = {
    RESNET50: 'classification_head.fc',
    RESNET18: 'classification_head.fc',
    RESNET34: 'classification_head.fc',
    RESNET101: 'classification_head.fc',
    RESNET152: 'classification_head.fc',
    VGG16: 'classification_head.fc',
    INCEPTIONV3: 'classification_head.fc',
    APVIT: 'None'
}

# EXPs
OVERRUN = False

# cam_curve_interval: for bbox. use high interval for validation (not test).
# high number of threshold slows down the validation because of
# `cv2.findContours`. this gets when cams are bad leading to >1k contours per
# threshold. default evaluation: .001.
VALID_FAST_CAM_CURVE_INTERVAL = .004

# data: name of the folder where cams will be stored.
DATA_CAMS = 'data_cams'

FULL_BEST_EXPS = 'full_best_exps'

# DDP
NCCL = 'nccl'
GLOO = 'gloo'

# scratch folder
SCRATCH_FOLDER = 'fer-action-u'

# CC: communitation folder
SCRATCH_COMM = f'{SCRATCH_FOLDER}/communication'


# ===
# optimizers
SGD = 'sgd'
ADAM = 'adam'
OPTIMIZERS = [SGD, ADAM]

# unbalanced classes
CLWNONE = 'cl_w_none'  # no class weights are used. Uniform.
CLWFIXEDTECH1 = 'cl_w_fixed_tech1'  # use cl. w. estimated from trainset before
# training. this shifts the model's focus to minority classes.
# computed via: wi = (1/wi) * (totcal/nclasses).
CLWFIXEDTECH2 = 'cl_w_fixed_tech2'  # use cl. w. estimated from trainset before
# training. this shifts the model's focus to minority classes.
# computed via: wi = max_w / wi.
CLWADAPTIVE = 'cl_w_adaptive'  # use cl.c. estimated at each epoch the per
# class error over trainset periodically (each epoch). for each class,
# we compute the amount of error per class. this shifts the model's focus to
# classes with high errors.
CLWMIXED = 'cl_w_mixed'  # it is a mix between cl_w_fixed + cl_w_adaptive.
# this shifts the model's focus to minority classes, and classes with high
# error.
CLW = [CLWNONE, CLWFIXEDTECH1, CLWFIXEDTECH2, CLWADAPTIVE, CLWMIXED]

# data sampler class weights: how to estimate them.
DATA_SAMPLER_CLW = [CLWFIXEDTECH1, CLWFIXEDTECH2]

# how many samples to sample per cl:
# PER_CL_NONE: do not apply.
# PER_CL_MIN_CL: look for the class the most under-represented and its number
# of samples as n.

PER_CL_NONE = 'per_cl_none'
PER_CL_MIN_CL = 'per_cl_min_cl'
DATA_SAMPLER_PER_CL = [PER_CL_NONE, PER_CL_MIN_CL]

# Metrics
LOCALIZATION_MTR = 'localization'
CL_ACCURACY_MTR = 'cl_accuracy'
CL_CONFMTX_MTR = 'cl_confusion_matrix'
AU_COSINE_MTR = 'au_cosine'  # cosine(layer_attention or cam,
# action_untis_heatmap)
MSE_MTR = 'mse'  # mean squared error.
MAE_MTR = 'mae'  # mean absolute error.

# segmentation measures
DICE_MTR = 'dice'
IOU_MTR = 'iou'

SEG_PERCLMTX_MTR = 'seg_per_cl_matrix'

METRICS = [LOCALIZATION_MTR,
           CL_ACCURACY_MTR,
           CL_CONFMTX_MTR,
           AU_COSINE_MTR,
           MSE_MTR,
           MAE_MTR,
           DICE_MTR,
           IOU_MTR,
           SEG_PERCLMTX_MTR
           ]

# Heatmap alignment types
ALIGN_AVG_HALF_LEFT = 'avg_half_left'  # average of half left of features
ALIGN_AVG_HALF_RIGHT = 'avg_half_right'  # average of half right of features
ALIGN_AVG = 'avg'  # average of all features
ALIGN_RANDOM_AVG = 'random_avg'  # randomly select p% of features and average
# them.
ALIGN_AVG_FIXED_Q = 'avg_fixed_q'  # align the average of fixed q% of the maps.
# this is different than 'ALIGN_RANDOM_AVG' in the sens the we use always the
# same q% maps. however, 'ALIGN_RANDOM_AVG' select randomly p% of maps.
ALIGNMENTS = [ALIGN_AVG_HALF_LEFT, ALIGN_AVG_HALF_RIGHT, ALIGN_AVG,
              ALIGN_RANDOM_AVG, ALIGN_AVG_FIXED_Q]

# number of main layers: can be used to extract layerwise attention alignment.
MAX_NBR_LAYERS = {
    RESNET50: 6,
    RESNET18: 6,
    RESNET34: 6,
    RESNET101: 6,
    RESNET152: 6,
    VGG16: 4,
    INCEPTIONV3: 6,
    APVIT: 1  # irrelevant.
}

# align attention to heatmap loss
A_L1 = 'l1'
A_L2 = 'l2'
A_COSINE = 'cosine'
A_STD_COSINE = 'std_cosine'  # attention is first standardized: (att - mean)/std
# todo: l1 smooth.
A_KL = 'kl'

ALIGN_LOSSES = [A_L1, A_L2, A_COSINE, A_STD_COSINE, A_KL]

# types of heatmaps
HEATMAP_LNDMKS = 'heatmap_landmarks'  # heatmap estimated from landmarks
HEATMAP_AUNITS_LNMKS = 'heatmap_action_units_from_landmarks'  # heatmap
# estimated from action units estimated from landmarks. the action units are
# selected based on the sample class.
HEATMAP_GENERIC_AUNITS_LNMKS = 'heatmap_generic_action_units_from_landmarks'
# class agnostic action units (takes all possible action units without
# looking to the class in the opposite to heatmap_action_units_from_landmarks
# that consider the class). This can be helpful when using action units at
# test time.
HEATMAP_PER_CLASS_AUNITS_LNMKS = 'heatmap_per_class_action_units_from_landmarks'
# considers a heatmap of action units per class.

HEATMAP_AUNITS_LEARNED_SEG = 'heatmap_action_units_learned_by_segmentation'
# action units heatmaps learned by segmentation

HEATMAP_TYPES = [HEATMAP_LNDMKS,
                 HEATMAP_AUNITS_LNMKS,
                 HEATMAP_GENERIC_AUNITS_LNMKS,
                 HEATMAP_PER_CLASS_AUNITS_LNMKS,
                 HEATMAP_AUNITS_LEARNED_SEG
                 ]

# which size to scale to:
SCALE_TO_ATTEN = 'scale_to_attention'  # scale heatmap to attention map size.
SCALE_TO_HEATM = 'scale_to_heatmap'  # scale attention to heatmap size.

SCALE_TO = [SCALE_TO_ATTEN, SCALE_TO_HEATM]


# attention normalization
NORM_NONE = 'norm_none'  # no normalization
NORM_SOFTMAX = 'norm_softmax'  # softmax the map.
NORMS_ATTENTION = [NORM_SOFTMAX, NORM_NONE]

# EXPRESSIONS
SURPRISE = 'Surprise'
FEAR = 'Fear'
DISGUST = 'Disgust'
HAPPINESS = 'Happiness'
SADNESS = 'Sadness'
ANGER = 'Anger'
NEUTRAL = 'Neutral'

EXPRESSIONS = [SURPRISE, FEAR, DISGUST, SADNESS, HAPPINESS, ANGER, NEUTRAL]

# curriculum types
CURRICULUM_CLASS = 'curriculum_class'  # update: add new class.
CURRICULUM_RANDOM = 'curriculum_random'  # update: add radom samples.

CURRICULUM_TYPES = [CURRICULUM_CLASS, CURRICULUM_RANDOM]

# Entropy regularizer
KL_UNIFORM = 'kl_uniform'
MAX_ENTROPY = 'max_entropy'
GEN_ENTROPY = 'generalized_entropy'  # https://arxiv.org/abs/2005.00820

HIGH_ENTROPY_REGS = [KL_UNIFORM, MAX_ENTROPY, GEN_ENTROPY]

# pretrained weights' folder name in the root code.

PRETRAINED_WEIGHTS_DIR = 'pretrained-weights'
FOLDER_PRETRAINED_IMAGENET = 'pretrained-imgnet'  # todo: redundant.


# Logits vs probs
LOGITS = 'logits'
PROBS = 'probs'

DATA_TYPE = [LOGITS, PROBS]

# Normalize
NORM_SCORES_NONE = 'none'
NORM_SCORES_MAX = 'max'
NORM_SCORES_SUM = 'sum'
NORM_SCORES_MEAN = 'mean'

NORM_SCORES = [NORM_SCORES_NONE, NORM_SCORES_MAX, NORM_SCORES_SUM,
               NORM_SCORES_MEAN
               ]

# family function to compute confusion
LINEAR_CONF_FUNC = 'linear'
EXP_CONF_FUNC = 'exp'

CONFUSION_FUNCS = [LINEAR_CONF_FUNC, EXP_CONF_FUNC]

# reduction
REDUCE_SUM = 'sum'
REDUCE_MEAN = 'mean'

REDUCTIONS = [REDUCE_SUM, REDUCE_MEAN]

# sparsity loss
SPARSE_L1 = 'l1'
SPARSE_L2 = 'l2'
SPARSE_INF = 'inf'

SPARSE_TECHS = [SPARSE_L1, SPARSE_L2, SPARSE_INF]

# orthogonality
ORTHOG_SOFT = 'orthog_soft'  # soft orthogonality (WtW - I)
ORTHOG_D_SOFT = 'orthog_d_soft'  # double soft
ORTHOG_MC = 'orthog_mc'  # Mutual coherence
ORTHOG_SRIP = 'orthog_srip'  # Spectral Restricted Isometry Property

ORTHOG_TECHS = [ORTHOG_SOFT, ORTHOG_D_SOFT, ORTHOG_MC, ORTHOG_SRIP]

# APVIT attention type.
ATT_LA = 'LA'
ATT_SUM = 'SUM'
ATT_SUM_ABS_1 = 'SUM_ABS_1'
ATT_SUM_ABS_2 = 'SUM_ABS_2'
ATT_MAX = 'MAX'
ATT_MAX_ABS_1 = 'MAX_ABS_1'
ATT_RAND = 'Random'

ATT_MEAN = 'mean_ATT'
ATT_PARAM_ATT = 'PARAM_ATT'  # parametric
ATT_PARAM_G_ATT = 'PARAM_G_ATT'  # parametric (gated attention)

ATT_METHODS = [ATT_LA, ATT_SUM, ATT_SUM_ABS_1, ATT_SUM_ABS_2, ATT_MAX,
               ATT_MAX_ABS_1, ATT_RAND,
               ATT_MEAN,
               ATT_PARAM_ATT, ATT_PARAM_G_ATT]

# data augmentation using action units or landmarks over input image via masking
# how to fill the background:
BG_F_BLACK = 'bg_f_black'  # black pixels
BG_F_IM_AVG = 'bg_f_im_avg'  # the average of the same image
BG_F_GAUSSIAN_BLUR = 'bg_f_gaussian_blur'  # blur the background.

BG_FILLERS = [BG_F_BLACK, BG_F_IM_AVG, BG_F_GAUSSIAN_BLUR]

# helper keys:
TRAIN_HEATMAP = 'train_daug_mask_img_heatmap'
EVAL_HEATMAP = 'eval_daug_mask_img_heatmap'
ALIGN_ATTEN_HEATMAP = 'align_atten_to_heatmap'
AUS_SEGM = 'aus_seg'

HEATMAP_KEYS = [TRAIN_HEATMAP,
                EVAL_HEATMAP,
                ALIGN_ATTEN_HEATMAP,
                AUS_SEGM
                ]

PRECOMPUTED = {
    ALIGN_ATTEN_HEATMAP: 'align_atten_to_heatmap_use_precomputed',
    TRAIN_HEATMAP: 'train_daug_mask_img_heatmap_use_precomputed',
    EVAL_HEATMAP: 'eval_daug_mask_img_heatmap_use_precomputed',
    AUS_SEGM: 'aus_seg_use_precomputed'
}

FOLDER_HEATMAP = {
    ALIGN_ATTEN_HEATMAP: 'align_atten_to_heatmap_folder',
    TRAIN_HEATMAP: 'train_daug_mask_img_heatmap_folder',
    EVAL_HEATMAP: 'eval_daug_mask_img_heatmap_folder',
    AUS_SEGM: 'aus_seg_folder'
}

# Average image pixel of trainsets
AVG_IMG_PIXEL_TRAIN_RAF_DB = [95, 106, 131]  # RGB
AVG_IMG_PIXEL_TRAIN_AFFECTNET = [99, 113, 144]  # AFFECTNET

AVG_IMG_PIXEL_TRAINSETS = {
    RAFDB: AVG_IMG_PIXEL_TRAIN_RAF_DB,
    AFFECTNET: AVG_IMG_PIXEL_TRAIN_AFFECTNET
}