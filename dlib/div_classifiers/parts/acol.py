"""
Original repository: https://github.com/xiaomengyc/ACoL
"""
import sys
from os.path import dirname, abspath, join

root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(root_dir)

import torch
import torch.nn as nn

from dlib.div_classifiers.parts.util import get_attention
from dlib.div_classifiers.parts.util import normalize_tensor
from dlib.configure import constants

__all__ = ['AcolBase']


class AcolBase(nn.Module):
    def __init__(self,
                 encoder_name: str,
                 num_classes: int,
                 drop_threshold: float = 0.1,
                 expansion: int = 1,
                 support_background: bool = False
                 ):
        super(AcolBase, self).__init__()

        assert encoder_name.startswith('resnet') or encoder_name.startswith(
            'vgg') or encoder_name.startswith('inception'), encoder_name
        self.encoder_name = encoder_name

        assert expansion >= 1, expansion
        assert isinstance(expansion, int), type(expansion)
        self.expansion = expansion
        self.support_background = support_background

        assert num_classes >= 1, num_classes
        assert isinstance(num_classes, int), type(num_classes)
        self.num_classes = num_classes

        classes = self.num_classes
        if self.support_background:
            classes = classes + 1

        assert isinstance(drop_threshold, float), type(drop_threshold)
        self.drop_threshold = drop_threshold  # better [0, 1].

        self.classifier_A = None
        self.classifier_B = None
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._create_cl_a_b(classes, expansion)

        self.cams = None

    def _create_cl_a_b(self, num_classes: int, expansion: int = 1):

        if self.encoder_name in [constants.RESNET50,
                                 constants.RESNET18,
                                 constants.RESNET34,
                                 constants.RESNET101,
                                 constants.RESNET152]:

            self.classifier_A = nn.Sequential(
                nn.Conv2d(512 * expansion, 1024, 3, 1, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, 1024, 3, 1, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, num_classes, 1, 1, padding=0)
            )
            self.classifier_B = nn.Sequential(
                nn.Conv2d(512 * expansion, 1024, 3, 1, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, 1024, 3, 1, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(1024, num_classes, 1, 1, padding=0)
            )

        else:
            raise NotImplementedError(self.encoder_name)  # todo: vgg,
            # inception.

    def correct_cl_logits(self, logits):
        if self.support_background:
            return logits[:, 1:]
        else:
            return logits

    def forward(self,
                features: torch.Tensor,
                labels: torch.Tensor = None
                ):

        assert labels is not None
        cams_a, logits = self._branch(features=features,
                                      classifier=self.classifier_A)

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention = get_attention(feature=cams_a, label=labels)
        erased_feature = _erase_attention(
            features=features, attention=attention,
            drop_threshold=self.drop_threshold)
        cams_b, logits_b = self._branch(features=erased_feature,
                                        classifier=self.classifier_B)

        logits_a = logits

        normalized_a = normalize_tensor(cams_a.detach().clone())
        normalized_b = normalize_tensor(cams_b.detach().clone())
        cams = torch.max(normalized_a, normalized_b)
        self.cams = cams.detach()

        return logits_a, logits_b

    def _branch(self, features, classifier):
        cams = classifier(features)
        logits = self.avgpool(cams).flatten(1)
        logits = self.correct_cl_logits(logits)
        return cams, logits

    def flush(self):
        self.cams = None


def _erase_attention(features, attention, drop_threshold):
    b, _, h, w = attention.size()
    pos = torch.ge(attention, drop_threshold)
    mask = attention.new_ones((b, 1, h, w))
    mask[pos.data] = 0.
    erased_feature = features * mask
    return erased_feature


def get_loss(output_dict, gt_labels, **kwargs):
    return nn.CrossEntropyLoss()(output_dict['logits'], gt_labels.long()) + \
           nn.CrossEntropyLoss()(output_dict['logit_b'], gt_labels.long())
