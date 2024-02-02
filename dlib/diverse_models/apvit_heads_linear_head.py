import sys
from os.path import dirname, abspath
from functools import partial
import os
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


__all__ = ['LinearClsHead', 'MultiLinearClsHead']


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base head.

    """

    def __init__(self):
        super(BaseHead, self).__init__()

    def init_weights(self):
        pass

    def forward_train(self, x, gt_label, **kwargss):
        pass


class ClsHead(BaseHead):
    """classification head.
    """

    def __init__(self):
        super(ClsHead, self).__init__()

    def get_linear_w(self):
        raise NotImplementedError



class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self,
                 num_classes: int,
                 in_channels: int = 768
                 ):
        super(LinearClsHead, self).__init__()

        assert isinstance(num_classes, int), type(num_classes)
        assert num_classes > 0, num_classes

        assert isinstance(in_channels, int), type(in_channels)
        assert in_channels > 0, in_channels

        self.in_channels = in_channels
        self.num_classes = num_classes

        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        # normal_init(self.fc, mean=0, std=0.01, bias=0)
        constant_init(self.fc, val=0, bias=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_score = self.fc(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        return cls_score

    def get_linear_w(self):
        return self.fc.weight


class MultiLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
    """  # noqa: W605

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_channels: list):
        super().__init__()

        assert isinstance(num_classes, int), type(num_classes)
        assert num_classes > 0, num_classes

        assert isinstance(in_channels, int), type(in_channels)
        assert in_channels > 0, in_channels

        assert isinstance(hidden_channels, list), type(hidden_channels)
        assert len(hidden_channels) > 0, hidden_channels

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels

        self.net = None
        self.fc2 = None

        self._init_layers()

    @staticmethod
    def _build_hidden_layer(in_c: int, outc: int) -> list:
        return [nn.Linear(in_c, outc),
                nn.BatchNorm1d(outc),
                nn.ReLU(inplace=False)
                ]

    def _init_layers(self):
        l = []
        in_c = self.in_channels
        for h in self.hidden_channels:
            l = l + self._build_hidden_layer(in_c=in_c, outc=h)
            in_c = h

        self.fc2 = nn.Linear(in_c, self.num_classes)
        self.net = torch.nn.Sequential(*l)


    def init_weights(self):
        constant_init(self.fc2, val=0, bias=0)

    def extract_feat(self, img):
        x = self.net(img)
        cls_score = self.fc2(x)
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return cls_score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        cls_score = self.fc2(x)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        return cls_score

    def get_linear_w(self):
        return self.fc2.weight


def test_linearclshead():

    c = 7
    in_c = 768

    model = LinearClsHead(num_classes=c,
                          in_channels=in_c
                          )
    print("Testing {}".format(model.__class__.__name__))
    model.eval()
    print("Num. parameters: {}".format(
        sum([p.numel() for p in model.parameters()])))
    cuda_id = "0"
    DEVICE = torch.device(f'cuda:{cuda_id}')

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)

    b = 32

    x = torch.randn(b, in_c)
    x = x.to(DEVICE)
    out = model(x)
    print(out.shape, c)


def test_multilinearckshead():

    c = 7
    in_c = 768
    h_c = 256

    model = MultiLinearClsHead(num_classes=c,
                               in_channels=in_c,
                               hidden_channels=[512, 256, 128]
                               )
    print("Testing {}".format(model.__class__.__name__))
    model.eval()
    print("Num. parameters: {}".format(
        sum([p.numel() for p in model.parameters()])))
    cuda_id = "0"
    DEVICE = torch.device(f'cuda:{cuda_id}')

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)

    b = 32

    x = torch.randn(b, in_c)
    x = x.to(DEVICE)
    out = model(x)
    print(out.shape, c)


if __name__ == "__main__":

    # test_linearclshead()
    test_multilinearckshead()