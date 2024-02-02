import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from dlib.losses.jaccard import JaccardLoss
from dlib.losses.dice import DiceLoss
from dlib.losses.focal import FocalLoss
from dlib.losses.lovasz import LovaszLoss
from dlib.losses.soft_bce import SoftBCEWithLogitsLoss
from dlib.losses.soft_ce import SoftCrossEntropyLoss

from dlib.losses.core import MasterLoss
from dlib.losses.core import CrossEntropyLoss
from dlib.losses.core import MeanAbsoluteErrorLoss
from dlib.losses.core import MeanSquaredErrorLoss
from dlib.losses.core import MultiClassFocalLoss
from dlib.losses.core import ImgReconstruction
from dlib.losses.core import SelfLearningFcams
from dlib.losses.core import ConRanFieldFcams
from dlib.losses.core import EntropyFcams
from dlib.losses.core import MaxSizePositiveFcams
from dlib.losses.core import WeightsSparsityLoss

from dlib.losses.align_to_heatmap import AlignToHeatMap
from dlib.losses.align_to_heatmap import AunitsSegmentationLoss

from dlib.losses.orth_features import FreeOrthFeatures
from dlib.losses.orth_features import GuidedOrthFeatures
from dlib.losses.orth_features import OrthoLinearWeightLoss

from dlib.losses.ordinal import OrdinalMeanLoss
from dlib.losses.ordinal import OrdinalVarianceLoss
from dlib.losses.ordinal import OrdIneqUnimodLoss

from dlib.losses.constrain_scores import ConstraintScoresLoss

from dlib.losses.cost_sensitive import SelfCostSensitiveLoss

from dlib.losses.entropy_reg import HighEntropy

from dlib.losses.sparsity import SparseLinearFeaturesLoss
from dlib.losses.sparsity import SparseLinClassifierWeightsLoss
from dlib.losses.sparsity import SparseApvitAttentionLoss
from dlib.losses.sparsity import AttentionSizeLoss
from dlib.losses.sparsity import LowEntropyAttentionLoss

