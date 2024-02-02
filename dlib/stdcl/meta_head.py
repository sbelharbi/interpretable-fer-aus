import sys
from os.path import dirname, abspath
from typing import Optional, Union, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import STDClModel

from dlib import poolings

from dlib.configure import constants


__all__ = ['MetaHead']


class MetaHead(nn.Module):
    def __init__(self,
                 meta_classes: dict,
                 pooling_head: str,
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
                 dense_dims: str = ''
                 ):
        super(MetaHead, self).__init__()

        # meta_classes = {
        # 0: [0, 1],
        # 1: [2, 3],
        # 2: [4, 5, 6]
        # }
        assert isinstance(meta_classes, dict), type(meta_classes)
        tmp_cls = []
        for k in meta_classes:
            msg = f"{k} | {type(meta_classes[k])}"
            assert isinstance(meta_classes[k], list), msg
            assert isinstance(k, int), f"{k} | {type(k)}"
            tmp_cls = tmp_cls + meta_classes[k]

        self.n_meta_classes = len(list(meta_classes.keys()))
        self.meta_classes = meta_classes

        assert isinstance(classes, int), type(classes)
        assert classes > 0, classes
        assert classes == len(tmp_cls), f"{classes} | {len(tmp_cls)}"

        self.n_classes = classes # real number of classes.
        self.features = []

        # meta head
        self.meta_classifier = poolings.__dict__[pooling_head](
            in_channels=in_channels,
            classes=self.n_meta_classes,
            support_background=support_background,
            r=r,
            modalities=modalities,
            kmax=kmax,
            kmin=kmin,
            alpha=alpha,
            dropout=dropout,
            dense_dropout=dense_dropout,
            dense_dims=dense_dims

        )

        # leaf classifiers
        self.leaf_classifiers = nn.ModuleList([
            poolings.__dict__[pooling_head](
                in_channels=in_channels,
                classes=len(list(meta_classes[k])),
                support_background=support_background,
                r=r,
                modalities=modalities,
                kmax=kmax,
                kmin=kmin,
                alpha=alpha,
                dropout=dropout,
                dense_dropout=dense_dropout,
                dense_dims=dense_dims

            )
            for k in meta_classes
        ])

    def flush(self):
        self.features = []

        if hasattr(self.meta_classifier, 'flush'):
            self.meta_classifier.flush()

        for m in self.leaf_classifiers:
            if hasattr(m, 'flush'):
                m.flush()

    def get_real_class_logits_non_diff(self,
                                       meta_output: torch.Tensor,
                                       leaf_outputs: list
                                       ) -> torch.Tensor:

        assert meta_output.ndim == 2, meta_output.ndim
        bsz, _ = meta_output.shape
        for o in leaf_outputs:
            assert o.ndim == 2, o.ndim
            assert o.shape[0] == bsz, f"{o.shape[0]} | {bsz}"

        out_scores = torch.zeros((bsz, self.n_classes), requires_grad=False,
                                 device=meta_output.device,
                                 dtype=meta_output.dtype
                                 )


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 4, x.ndim

        meta_output = self.meta_classifier(x)

        leaf_outputs = [m(x) for m in self.leaf_classifiers]