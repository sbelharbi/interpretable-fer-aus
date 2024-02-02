import sys
from os.path import dirname, abspath

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.div_classifiers.parts.acol import AcolBase

__all__ = ['GAP', 'ACOL', 'WGAP', 'MaxPool', 'LogSumExpPool', 'PRM']


class _BasicPooler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 classes: int,
                 support_background: bool = False,
                 r: float = 10.,
                 modalities: int = 5,
                 kmax: float = 0.5,
                 kmin: float = None,
                 alpha: float = 0.6,
                 dropout: float = 0.0,
                 dense_dropout: float = 0.0,
                 dense_dims: str = '',
                 encoder_name: str = constants.RESNET50,
                 encoder_expansion: int = 1,
                 acol_drop_threshold: float = 0.1,
                 prm_ks: int = 3,
                 prm_st: int = 1
                 ):
        super(_BasicPooler, self).__init__()

        self.cams = None
        self.in_channels = in_channels
        self.classes = classes
        self.support_background = support_background

        self.encoder_name = encoder_name
        assert encoder_expansion >= 1, encoder_expansion
        assert isinstance(encoder_expansion, int), type(encoder_expansion)
        self.encoder_expansion = encoder_expansion

        assert isinstance(acol_drop_threshold, float), type(acol_drop_threshold)
        self.acol_drop_threshold = acol_drop_threshold

        # logsumexp
        self.r = r
        # wildcat
        self.modalities = modalities
        self.kmax = kmax
        self.kmin = kmin
        self.alpha = alpha
        self.dropout = dropout

        self.dense_dropout = dense_dropout
        z = []
        if dense_dims not in ['', 'None', None]:
            z = dense_dims.split('-')  # accept only 2 max dense layer: d1-d2

        z = [int(i) for i in z]
        assert len(z) <= 2, f"{len(z)}: {z}"
        self.d1 = None
        self.d2 = None

        if len(z) == 1:
            self.d1 = z[0]

        elif len(z) == 2:
            self.d1 = z[0]
            self.d2 = z[1]

        elif len(z) == 0:
            pass
        else:
            raise NotImplementedError(z)


        self.features = []
        self.linear_features = None
        # self.linear_w = None

        # prm
        assert isinstance(prm_ks, int)
        assert prm_ks > 0
        assert isinstance(prm_st, int)
        assert prm_st > 0
        self.prm_ks = prm_ks
        self.prm_st = prm_st

        self.name = 'null-name'

    @property
    def linear_w(self):
        return None

    @property
    def builtin_cam(self):
        return True

    def assert_x(self, x):
        assert isinstance(x, torch.Tensor)
        assert x.ndim == 4

    def correct_cl_logits(self, logits):
        if self.support_background:
            return logits[:, 1:]
        else:
            return logits

    def get_nbr_params(self):
        return sum([p.numel() for p in self.parameters()])

    def free_mem(self):
        self.cams = None

    def flush(self):
        self.features = []
        self.linear_features = None
        # self.linear_w = None

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={})'.format(
            self.__class__.__name__, self.in_channels, self.classes,
            self.support_background)


class GAP(_BasicPooler):
    """ https://arxiv.org/pdf/1312.4400.pdf """
    def __init__(self, **kwargs):
        super(GAP, self).__init__(**kwargs)
        self.name = 'GAP'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor = None
                ) -> torch.Tensor:
        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()
        logits = self.pool(out).flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits


class ACOL(_BasicPooler):
    """ https://arxiv.org/pdf/1804.06962.pdf
     "Adversarial Complementary Learning for Weakly Supervised Object
     Localization."
     """
    def __init__(self, **kwargs):
        super(ACOL, self).__init__(**kwargs)
        self.name = 'ACOL'

        classes = self.classes
        if self.support_background:
            classes = classes + 1


        self.pool = nn.AdaptiveAvgPool2d(1)
        self.acol_pooler = AcolBase(encoder_name=self.encoder_name,
                                    num_classes=classes,
                                    drop_threshold=self.acol_drop_threshold,
                                    expansion=self.encoder_expansion,
                                    support_background=self.support_background
                                    )

        self.logits_b = None

    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        self.assert_x(x)

        logits_a, logits_b = self.acol_pooler(features=x, labels=labels)
        self.cams = self.acol_pooler.cams.detach().clone()
        logits = logits_a  # already corrected.
        self.logits_b = logits_b

        return logits

    def flush(self):
        super(ACOL, self).flush()
        self.acol_pooler.flush()
        self.logits_b = None


class WGAP(_BasicPooler):
    """ https://arxiv.org/pdf/1512.04150.pdf """
    def __init__(self, **kwargs):
        super(WGAP, self).__init__(**kwargs)
        self.name = 'WGAP'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # self.fc = nn.Linear(self.in_channels, classes)

        d = self.in_channels

        self.fc1 = None
        self.fc2 = None

        # self.linear_w = None


        if self.d1 is not None:
            self.fc = nn.Linear(d, self.d1)

            if self.d2 is not None:
                self.fc1 = nn.Linear(self.d1, self.d2)
                self.fc2 = nn.Linear(self.d2, classes)

                # self.linear_w = self.fc2.weight

            else:
                self.fc1 = nn.Linear(self.d1, classes)
                # self.linear_w = self.fc1.weight

        else:
            self.fc = nn.Linear(d, classes)
            # self.linear_w = self.fc.weight

        self.lin_dropout = nn.Dropout(self.dense_dropout)

    @property
    def linear_w(self):
        if self.d1 is not None:

            if self.d2 is not None:
                return self.fc2.weight

            else:
                return self.fc1.weight

        else:
            return self.fc.weight


    @property
    def builtin_cam(self):
        return False

    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:

        # pre_logit = self.avgpool(x)
        # pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        # logits = self.fc(pre_logit)
        #
        # logits = self.correct_cl_logits(logits)

        self.features = []
        self.linear_features = None
        # self.linear_w = None

        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        self.linear_features = pre_logit
        # self.linear_w = self.fc.weight

        logits = self.fc(pre_logit)


        if self.fc1 is not None:
            logits = F.relu(logits, inplace=False)

            self.features.append(logits)
            self.linear_features = logits

            logits = self.lin_dropout(logits)

            logits = self.fc1(logits)

            # self.linear_w = self.fc1.weight

            if self.fc2 is not None:
                logits = F.relu(logits, inplace=False)

                self.features.append(logits)
                self.linear_features = logits

                logits = self.lin_dropout(logits)
                logits = self.fc2(logits)

                # self.linear_w = self.fc2.weight

            else:
                pass

        else:
            pass

        logits = self.correct_cl_logits(logits)

        return logits


class MaxPool(_BasicPooler):
    def __init__(self, **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.name = 'MaxPool'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor = None
                ) -> torch.Tensor:

        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()
        logits = self.pool(out).flatten(1)

        logits = self.correct_cl_logits(logits)
        return logits


class LogSumExpPool(_BasicPooler):
    """ https://arxiv.org/pdf/1411.6228.pdf """
    def __init__(self, **kwargs):
        super(LogSumExpPool, self).__init__(**kwargs)
        self.name = 'LogSumExpPool'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:

        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()
        m = self.maxpool(out)
        out = self.avgpool((self.r * (out - m)).exp()).log().mul(1./self.r) + m

        logits = out.flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits

    def __repr__(self):
        return '{}(in_channels={}, classes={}, support_background={}, ' \
               'r={})'.format(self.__class__.__name__, self.in_channels,
                              self.classes, self.support_background, self.r)


class PRM(_BasicPooler):
    """
    https://arxiv.org/pdf/1804.00880.pdf
    Peak Response Map pooling (PRM)
    'Weakly Supervised Instance Segmentation using Class Peak Response'.
    """
    def __init__(self, **kwargs):
        super(PRM, self).__init__(**kwargs)
        self.name = 'PRM'

        classes = self.classes
        if self.support_background:
            classes = classes + 1

        self.conv = nn.Conv2d(self.in_channels, out_channels=classes,
                              kernel_size=1)
        self.maxpool = nn.MaxPool2d(kernel_size=self.prm_ks, stride=self.prm_st)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,
                x: torch.Tensor,
                labels: torch.Tensor = None) -> torch.Tensor:
        self.assert_x(x)

        out = self.conv(x)
        self.cams = out.detach()

        out = self.maxpool(out)

        logits = self.pool(out).flatten(1)
        logits = self.correct_cl_logits(logits)

        return logits


if __name__ == '__main__':
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed

    set_seed(0)
    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    b, c, h, w = 3, 1024, 8, 8
    classes = 5
    x = torch.rand(b, c, h, w).to(DEVICE)

    for support_background in [True, False]:
        for cl in [GAP, WGAP, MaxPool, LogSumExpPool, PRM]:
            instance = cl(in_channels=c, classes=classes,
                          support_background=support_background)
            instance.to(DEVICE)
            announce_msg('TEsting {}'.format(instance))
            out = instance(x)
            if instance.builtin_cam:
                print('x: {}, cam: {}, logitcl shape: {}, logits: {}'.format(
                    x.shape, instance.cams.shape, out.shape, out))
            else:
                print('x: {}, logitcl shape: {}, logits: {}'.format(
                    x.shape, out.shape, out))
