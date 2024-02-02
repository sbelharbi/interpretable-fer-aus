import sys
import os
import time
from os.path import dirname, abspath, join
import datetime as dt

import numpy as np
import torch
import torch.nn as nn

from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

__all__ = ['DenseCRFLoss']


class DenseCRFLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        """
        Init. function.
        :param weight: float. It is Lambda for the crf loss.
        :param sigma_rgb: float. sigma for the bilateheral filtering (
        appearance kernel): color similarity.
        :param sigma_xy: float. sigma for the bilateral filtering
        (appearance kernel): proximity.
        :param scale_factor: float. ratio to scale the image and
        segmentation. Helpful to control the computation (speed) / precision.
        """
        super(DenseCRFLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

    def forward(self, images, segmentations):
        """
        Forward loss.
        Image and segmentation are scaled with the same factor.

        :param images: torch tensor of the image (values in [0, 255]). shape
        N*C*H*W. DEVICE: CPU.
        :param segmentations: softmaxed logits. cuda.
        :return: loss score (scalar).
        """
        raise NotImplementedError
        val = 0.0

        return val

    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )


def test_DenseCRFLoss():
    import time

    from dlib.utils.reproducibility import set_seed
    from dlib.utils.shared import announce_msg

    from torch.profiler import profile, record_function, ProfilerActivity

    seed = 0
    cuda = "0"
    print("cuda:{}".format(cuda))
    print("DEVICE BEFORE: ", torch.cuda.current_device())
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    set_seed(seed=seed)
    n, h, w = 32, 244, 244
    scale_factor = 1.
    img = torch.randint(
        low=0, high=256,
        size=(n, 3, h, w), dtype=torch.float, device=DEVICE,
        requires_grad=False).cpu()
    nbr_cl = 2
    segmentations = torch.rand(size=(n, nbr_cl, h, w), dtype=torch.float,
                               device=DEVICE,
                               requires_grad=True)

    loss = DenseCRFLoss(weight=1e-7,
                        sigma_rgb=15.,
                        sigma_xy=100.,
                        scale_factor=scale_factor
                        ).to(DEVICE)
    announce_msg("testing {}".format(loss))
    set_seed(seed=seed)
    if nbr_cl > 1:
        softmax = nn.Softmax(dim=1)
    else:
        softmax = nn.Sigmoid()

    print(img.sum(), softmax(segmentations).sum())

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with autocast(enabled=False):
        z = loss(images=img, segmentations=softmax(segmentations))
    torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print('time op: {}'.format(elapsed_time_ms))

    print('Time ({} x {} : scale: {}: N: {}): TIME_ABOVE'.format(
        h, w, scale_factor, n))
    tx = time.perf_counter()
    z.backward()
    print('backward {}'.format(time.perf_counter() - tx))
    print('Loss: {} {} (nbr_cl: {})'.format(z, z.dtype, nbr_cl))


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    for i in range(3):
        test_DenseCRFLoss()
