import sys
from os.path import dirname, abspath
from typing import List

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.elementary import ElementaryLoss
from dlib.losses.entropy import Entropy
from dlib.crf.dense_crf_loss import DenseCRFLoss
from dlib.configure import constants

__all__ = [
    'MasterLoss',
    'CrossEntropyLoss',
    'MeanAbsoluteErrorLoss',
    'MeanSquaredErrorLoss',
    'WeightsSparsityLoss',
    'MultiClassFocalLoss',
    'ImgReconstruction',
    'SelfLearningFcams',
    'ConRanFieldFcams',
    'EntropyFcams',
    'MaxSizePositiveFcams'
]


class ClLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ClLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(reduction="mean").to(self._device)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(ClLoss, self).forward(epoch=epoch)

        if self.cl_w_updated:
            self.loss = nn.CrossEntropyLoss(weight=self.per_class_weights,
                                            reduction="mean").to(self._device)
            self.cl_w_updated = False

        if not self.is_on():
            return self._zero

        return self.loss(input=cl_logits, target=glabel) * self.lambda_


class WeightsSparsityLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(WeightsSparsityLoss, self).__init__(**kwargs)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(WeightsSparsityLoss, self).forward(epoch=epoch)

        assert model is not None

        if not self.is_on():
            return self._zero

        l1loss = self._zero
        for w in model.parameters():
            l1loss = l1loss + torch.linalg.norm(w.view(-1), ord=1)

        return l1loss * self.lambda_


class CrossEntropyLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)

        self.ce_label_smoothing: float = 0.0

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = False

    def set_it(self, ce_label_smoothing: float):
        assert isinstance(ce_label_smoothing, float), type(ce_label_smoothing)
        assert 0 <= ce_label_smoothing <= 1., ce_label_smoothing

        self.ce_label_smoothing = ce_label_smoothing

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=self.ce_label_smoothing
        ).to(self._device)

        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(CrossEntropyLoss, self).forward(epoch=epoch)

        if self.cl_w_updated:
            self.loss = nn.CrossEntropyLoss(weight=self.per_class_weights,
                                            reduction="mean").to(self._device)
            self.cl_w_updated = False

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        # cutmix
        if self.cutmix_holder is not None:
            assert isinstance(self.cutmix_holder, list)
            assert len(self.cutmix_holder) == 3
            target_a, target_b, lam = self.cutmix_holder
            loss = (self.loss(cl_logits, target_a) * lam + self.loss(
                cl_logits, target_b) * (1. - lam))

            return loss

        # acol
        if hasattr(model.classification_head, 'logits_b'):
            l_a = self.loss(input=cl_logits, target=glabel) * self.lambda_
            cl_logits_b = model.classification_head.logits_b
            l_b = self.loss(input=cl_logits_b, target=glabel) * self.lambda_

            return l_a + l_b

        # standard

        return self.loss(input=cl_logits, target=glabel) * self.lambda_


class MeanAbsoluteErrorLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MeanAbsoluteErrorLoss, self).__init__(**kwargs)

        self.n_cls: int = 1

        self.loss = nn.L1Loss(reduction="mean").to(self._device)


        self.already_set = False

    def set_it(self, n_cls: int):
        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls
        self.n_cls = n_cls

        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(MeanAbsoluteErrorLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        _onehot = F.one_hot(glabel, num_classes=self.n_cls)
        _probs = F.softmax(cl_logits, dim=1)

        return self.loss(input=_probs, target=_onehot.float()) * self.lambda_


class MeanSquaredErrorLoss(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MeanSquaredErrorLoss, self).__init__(**kwargs)

        self.n_cls: int = 1

        self.loss = nn.MSELoss(reduction="mean").to(self._device)


        self.already_set = False

    def set_it(self, n_cls: int):
        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls
        self.n_cls = n_cls

        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(MeanSquaredErrorLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        _onehot = F.one_hot(glabel, num_classes=self.n_cls)
        _probs = F.softmax(cl_logits, dim=1)
        return self.loss(input=_probs, target=_onehot.float()) * self.lambda_


class MultiClassFocalLoss(ElementaryLoss):
    """
    Adaptation of focal loss for multi-class classification.
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, **kwargs):
        super(MultiClassFocalLoss, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(weight=None,
                                        reduction="none").to(self._device)
        self.alpha_focal = 1.
        self.gamma_focal = 0.0
        self.already_set = False

    def set_it(self, alpha_focal: float, gamma_focal: float):
        assert isinstance(alpha_focal, float), type(alpha_focal)
        assert alpha_focal > 0., alpha_focal

        assert isinstance(gamma_focal, float), type(gamma_focal)
        assert gamma_focal >= 0., gamma_focal

        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal

        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(MultiClassFocalLoss, self).forward(epoch=epoch)

        if self.cl_w_updated:
            self.loss = nn.CrossEntropyLoss(weight=self.per_class_weights,
                                            reduction="none").to(self._device)
            self.cl_w_updated = False

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        ce_loss = self.loss(input=cl_logits, target=glabel)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha_focal * ((1 - pt) ** self.gamma_focal) *
                      ce_loss).mean()

        return self.lambda_ * focal_loss


class ImgReconstruction(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ImgReconstruction, self).__init__(**kwargs)

        self.loss = nn.MSELoss(reduction="none").to(self._device)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(ImgReconstruction, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        n = x_in.shape[0]
        loss = self.elb(self.loss(x_in, im_recon).view(n, -1).mean(
            dim=1).view(-1, ))
        return self.lambda_ * loss.mean()


class SelfLearningFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(SelfLearningFcams, self).__init__(**kwargs)

        self.loss = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.seg_ignore_idx).to(self._device)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(SelfLearningFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag

        return self.loss(input=fcams, target=seeds) * self.lambda_


class ConRanFieldFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(ConRanFieldFcams, self).__init__(**kwargs)

        self.loss = DenseCRFLoss(
            weight=self.lambda_, sigma_rgb=self.sigma_rgb,
            sigma_xy=self.sigma_xy, scale_factor=self.scale_factor
        ).to(self._device)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(ConRanFieldFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero
        
        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        return self.loss(images=raw_img, segmentations=fcams_n)


class EntropyFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(EntropyFcams, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(EntropyFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert fcams.ndim == 4
        bsz, c, h, w = fcams.shape

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        probs = fcams_n.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c)

        return self.lambda_ * self.loss(probs).mean()


class MaxSizePositiveFcams(ElementaryLoss):
    def __init__(self, **kwargs):
        super(MaxSizePositiveFcams, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(MaxSizePositiveFcams, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert not self.multi_label_flag  # todo

        if fcams.shape[1] > 1:
            fcams_n = F.softmax(fcams, dim=1)
        else:
            fcams_n = F.sigmoid(fcams)
            fcams_n = torch.cat((1. - fcams_n, fcams_n), dim=1)

        n = fcams.shape[0]
        loss = None
        for c in [0, 1]:
            bl = fcams_n[:, c].view(n, -1).sum(dim=-1).view(-1, )
            if loss is None:
                loss = self.elb(-bl)
            else:
                loss = loss + self.elb(-bl)

        return self.lambda_ * loss * (1./2.)


class MasterLoss(nn.Module):
    def __init__(self, cuda_id: int, name=None):
        super().__init__()
        self._name = name

        self.losses = []
        self.l_holder = []
        self.n_holder = [self.__name__]
        self._device = torch.device(cuda_id)

    def add(self, loss_: ElementaryLoss):
        self.losses.append(loss_)
        self.n_holder.append(loss_.__name__)

    def update_t(self):
        for loss in self.losses:
            loss.update_t()

    def init_class_weights(self,
                           n_cls: int,
                           w: torch.Tensor = None,
                           style: str = constants.CLWNONE):
        for loss in self.losses:
            loss.init_class_weights(n_cls=n_cls,
                                    w=w,
                                    style=style)

    def update_per_class_weights(self, w: torch.Tensor):
        for loss in self.losses:
            loss.update_per_class_weights(w=w)

    def set_cutmix_holder(self, cutmix_holder: List):
        for loss in self.losses:
            assert isinstance(loss, ElementaryLoss)
            loss.set_cutmix_holder(cutmix_holder)

    def reset_cutmix_holder(self):
        for loss in self.losses:
            assert isinstance(loss, ElementaryLoss)
            loss.reset_cutmix_holder()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name

    def forward(self, **kwargs):
        assert self.losses != []

        self.l_holder = []
        for loss in self.losses:

            self.l_holder.append(loss(**kwargs).to(self._device))



        loss = sum(self.l_holder)
        self.l_holder = [loss] + self.l_holder

        # security.
        self.reset_cutmix_holder()
        return loss

    def to_device(self):
        for loss in self.losses:
            loss.to(self._device)

    def check_losses_status(self):
        print('-' * 60)
        print('Losses status:')

        for i, loss in enumerate(self.losses):
            if hasattr(loss, 'is_on'):
                print(self.n_holder[i+1], ': ... ',
                      loss.is_on(),
                      "({}, {})".format(loss.start_epoch, loss.end_epoch))
        print('-' * 60)

    def __str__(self):
        return "{}():".format(
            self.__class__.__name__, ", ".join(self.n_holder))


if __name__ == "__main__":
    from dlib.utils.reproducibility import set_seed
    set_seed(seed=0)
    b, c = 10, 4
    cudaid = 0
    torch.cuda.set_device(cudaid)

    loss = MasterLoss(cuda_id=cudaid)
    print(loss.__name__, loss, loss.l_holder, loss.n_holder)
    loss.add(SelfLearningFcams(cuda_id=cudaid))
    for l in loss.losses:
        print(l, isinstance(l, SelfLearningFcams))

    for e in loss.n_holder:
        print(e)

