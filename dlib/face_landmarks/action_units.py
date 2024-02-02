import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple
import argparse
import math
import time
import datetime as dt


import numpy as np
import torch
import cv2
import yaml
import matplotlib.pyplot as plt
import tqdm
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('nbAgg')


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import reformat_id
from dlib.utils.shared import configure_metadata
from dlib.utils.shared import get_image_ids
from dlib.utils.shared import get_class_labels
from dlib.utils.shared import get_landmarks
from dlib.utils.tools import get_root_wsol_dataset

from dlib.configure import constants

# ref: Automatic Analysis of Facial Actions: A Survey.
# Brais Martinez; Michel F. Valstar; Bihan Jiang; Maja Pantic. 2017.
# https://ieeexplore.ieee.org/document/7990582

# code reference: https://github.com/rakutentech/FAU_CVPR2021/blob/main
# /Prepare_data.py

__all__ = ['build_all_action_units']


AU = {0: 'Inner Brow Raiser',
      1: 'Outer Brow Raiser',
      2: 'Brow Lowerer',
      3: 'Upper Lid Raiser',
      4: 'Cheek Raiser',
      5: 'Lid Tightener',
      6: 'Nose Wrinkler',
      7: 'Upper Lip Raiser',
      8: 'Lip Corner Puller',
      9: 'Dimpler',
      10: 'Lip Corner Depressor',
      11: 'Chin Raiser',
      12: 'Lip Stretcher',
      13: 'Lip Tightener',
      14: 'Lip pressor',
      15: 'Lips Part',
      16: 'Jaw Drop',
      17: 'Eyes Closed',
      18: 'Mouth Stretcher',
      19: 'Lower Lip Depressor'
      }

EXP2AU = {
    constants.ANGER: [2, 3, 5, 7, 11, 13, 14, 15, 16],
    constants.DISGUST: [6, 3, 19, 11, 15, 16],
    constants.FEAR: [0, 1, 2, 3, 12, 15, 16, 18],
    constants.HAPPINESS: [4, 8, 15],
    constants.SADNESS: [0, 2, 4, 10, 11],
    constants.SURPRISE:[0, 1, 3, 16, 18]
}

# GENERIC_AUS = [0, 1, 4, 8, 15]
# GENERIC_AUS = [4, 8, 15]
GENERIC_AUS = sorted(list(AU.keys()))

ALL_AUS = sorted(list(AU.keys()))


def plot_action_units_ellipsoid(au: int,
                                h: int,
                                w: int,
                                lndmks: list,
                                ) -> Tuple[np.ndarray, bool]:
    assert isinstance(lndmks, list), type(lndmks)
    assert len(lndmks) == 68, len(lndmks)
    # lndmks: [(x, y), ....]: x: width, y: height.


    att_map = np.zeros((h, w)) + 1e-4
    cp = att_map.copy()
    col = (255, 255, 255)
    a = 0  # angle
    s = 0  # start angle
    e = 360  # end angle
    f = cv2.FILLED

    if au == 0:  # AU 0: inner brow raiser
        l_x1, l_y1 = lndmks[20]
        r_x2, r_y2 = lndmks[23]
        major = round(w / 8)
        minor = round(h/10)
        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 1:  # AU 1: Outer Brow Raiser

        l_x1, l_y1 = lndmks[18]
        r_x2, r_y2 = lndmks[25]
        major = round(w / 8)
        minor = round(h / 10)

        cv2.ellipse(att_map, (l_x1, l_y1), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, r_y2), (major, minor), a, s, e, col, f)

    elif au == 2:  # AU 2: Brow Lowerer
        l_x, l_y = lndmks[19]
        r_x, r_y = lndmks[24]
        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        # major = abs(int((r_x - l_x) / 2.))
        # minor = abs(int((r_y - l_y) / 2.))
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 10

        if minor == 0:
            minor = 10

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 3:  # AU 3: Upper Lid Raiser
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 - l_y1) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 - l_y2) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 4:  # AU 4: Cheek Raiser
        l_x1, l_y1 = lndmks[41]
        r_x1, r_y1 = lndmks[46]

        x = l_x1  # - round(w / 10)
        y = l_y1 + round(h / 6)
        major = round(w / 10)
        minor = round(h / 10)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = r_x1   # - round(w / 10)
        y = r_y1 + round(h / 6)
        major = round(w / 10)
        minor = round(h / 10)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 5: # AU 5: Lid Tightener
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)
        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)
        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 6:  # AU 6: Nose Wrinkler
        l_x1, l_y1 = lndmks[29]
        r_x1, r_y1 = lndmks[31]
        r_x2, r_y2 = lndmks[35]

        cv2.ellipse(att_map, (r_x1, l_y1), (20, 20), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x2, l_y1), (20, 20), a, s, e, col, f)

    elif au == 7:  # AU 7: Upper Lip Raiser
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[50]

        r_x1, r_y1 = lndmks[52]
        r_x2, r_y2 = lndmks[54]

        x = int((l_x1 + l_x1) / 2)
        y = int((l_y2 + l_y2) / 2)

        cv2.ellipse(att_map, (x, y), (20, 20), a, s, e, col, f)

        x = int((r_x1 + r_x1) / 2)
        y = int((r_y2 + r_y2) / 2)

        cv2.ellipse(att_map, (x, y), (20, 20), a, s, e, col, f)

    elif au == 8:  # AU 8: Lip corner puller.
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 9:  # AU 9: Dimpler.
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        l_x = max(l_x - 20 , 0)
        r_x = max(min((r_x + 20, w)), min((r_x + 10, w)))

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 10:  # AU 10 == AU 8: Lip Corner Depressor
        l_x, l_y = lndmks[48]
        r_x, r_y = lndmks[54]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    elif au == 11:  # AU 11: Chin Raiser
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[8]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 12:  # AU 12: Lip Stretcher
        l_x1, l_y1 = lndmks[48]
        l_x2, l_y2 = lndmks[6]


        r_x1, r_y1 = lndmks[50]
        r_x2, r_y2 = lndmks[10]

        x = int((l_x1 + l_x2) / 2)
        y = int((l_y1 + l_y2) / 2)
        major = 20
        minor = 20

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((r_x1 + r_x2) / 2)
        y = int((r_y1 + r_y2) / 2)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 13:  # AU 13: Lip Tightener
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)


    elif au == 14:  # AU 14: Lip pressor == AU 13.
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)

        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 15:  # AU 15: Lips Part

        t_x, t_y = lndmks[51]
        b_x, b_y = lndmks[57]

        major = 25
        minor = 10

        cv2.ellipse(att_map, (t_x, t_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (b_x, b_y), (major, minor), a, s, e, col, f)

    elif au == 16:  # AU 16: Jaw Drop
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 17:  # AU 17: Eyes Closed
        l_x1, _ = lndmks[36]
        _, l_y1 = lndmks[38]
        r_x1, _ = lndmks[39]
        _, r_y1 = lndmks[41]
        l_x2, _ = lndmks[42]
        _, l_y2 = lndmks[44]
        r_x2, _ = lndmks[45]
        _, r_y2 = lndmks[47]

        x = int((l_x1 + r_x1) / 2)
        y = int((l_y1 + r_y1) / 2)

        major = max(int((r_x1 - l_x1) / 2), 0)
        minor = max(int((r_y1 + 10 - l_y1 + 10) / 2), 0)

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

        x = int((l_x2 + r_x2) / 2)
        y = int((l_y2 + r_y2) / 2)

        major = max(int((r_x2 - l_x2) / 2), 0)
        minor = max(int((r_y2 + 10 - l_y2 + 10) / 2), 0)
        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 18:  # AU 18: Mouth stretcher == AU 16.
        l_x, _ = lndmks[48]
        r_x, _ = lndmks[54]
        _, l_y = lndmks[51]
        _, r_y = lndmks[57]

        x = int((l_x + r_x) / 2)
        y = int((l_y + r_y) / 2)
        major = max(int((r_x - l_x) / 2), 0)
        minor = max(int((r_y - l_y) / 2), 0)

        if major == 0:
            major = 5

        if minor == 0:
            minor = 5

        cv2.ellipse(att_map, (x, y), (major, minor), a, s, e, col, f)

    elif au == 19:  # AU 19: Lower lip depressor
        l_x, l_y = lndmks[59]
        r_x, r_y = lndmks[55]

        major = 20
        minor = 20

        cv2.ellipse(att_map, (l_x, l_y), (major, minor), a, s, e, col, f)
        cv2.ellipse(att_map, (r_x, r_y), (major, minor), a, s, e, col, f)

    is_roi = ((att_map - cp).sum() > 0)  # sometimes ellipse may draw outside
    # image leading to empty heatmap. this heatmap should be flagged.


    if is_roi:
        att_map = cv2.resize(att_map, dsize=(28, 28))

    else:  # flag invalid heatmaps.
        att_map = np.zeros((28, 28)) + np.inf

    return att_map, is_roi


def build_all_action_units(lndmks: list,
                           h: int,
                           w: int,
                           cl: str,
                           aus_type: str
                           ) -> np.ndarray:
    """
    Build heatmaps for action units related to the expression 'cl'.
    Supports: 68 landmarks (https://b2633864.smushcdn.com/2633864/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg?size=630x508&lossy=1&strip=1&webp=1)
    :param lndmks: list of landmarks [(x1, y1), ....]. x: width, y: height.
    :param h: int. image height.
    :param w: int. image width.
    :param cl: str. class name.
    :param p: float. Bernoulli dist. parameter p to take value 1.
    :return: np.ndarray. action units heatmaps for the request class expression.
    """
    assert isinstance(lndmks, list), type(lndmks)
    assert len(lndmks) == 68, len(lndmks)

    assert cl in constants.EXPRESSIONS, cl

    # perform checking internally: stand-alone function.
    if aus_type == constants.HEATMAP_AUNITS_LNMKS:
        if cl == constants.NEUTRAL or lndmks[0][0] == np.inf:
            return np.zeros((1, h, w)) + np.inf

    elif aus_type == constants.HEATMAP_GENERIC_AUNITS_LNMKS:
        if lndmks[0][0] == np.inf:
            return np.zeros((1, h, w)) + np.inf

    else:
        raise NotImplementedError(aus_type)

    lndmks = [(int(z[0]), int(z[1])) for z in lndmks]

    assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS], aus_type

    if aus_type == constants.HEATMAP_AUNITS_LNMKS:

        assert cl in EXP2AU, cl
        aus = EXP2AU[cl]

    elif aus_type == constants.HEATMAP_GENERIC_AUNITS_LNMKS:
        aus = GENERIC_AUS

    else:
        raise NotImplementedError(aus_type)

    n_au = len(aus)
    full_att_map = np.zeros((n_au, h, w))

    if lndmks[0][0] == np.inf:  # invalid landmarks.
        return np.zeros((1, h, w)) + np.inf

    rois = []
    holder = []
    failed = 0

    for i, au in enumerate(aus):
        att_map, is_roi = plot_action_units_ellipsoid(
            au=au, h=h, w=w, lndmks=lndmks)

        if is_roi:
            att_map = cv2.blur(att_map, (3, 3))
            attmap_resized = cv2.resize(att_map, dsize=(w, h))
            full_att_map[i, :, :] = attmap_resized
            holder.append(attmap_resized)

        else:
            failed += 1


        rois.append(is_roi)

    if sum(rois) == 0:  # ignore. failed to draw rois inside image.
        return np.zeros((1, h, w)) + np.inf

    if failed > 0:
        full_att_map = np.array(holder)  # nbr_roi, h, w


    return full_att_map.astype(np.float32)


def show_all_action_untis(img: np.ndarray, aus_maps: np.ndarray, wfp: str):
    assert img.ndim ==3, img.ndim
    assert aus_maps.ndim == 4, aus_maps.ndim
    n = aus_maps.shape[1]
    ncols = 6
    nrows = math.ceil((n + 1) / float(ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    axes[0, 0].imshow(img)

    k = 0
    for i in range(nrows):
        for j in range(ncols):
            if i == j == 0:
                continue

            if k >= n:
                axes[i, j].set_visible(False)
                continue
            axes[i, j].imshow(aus_maps[0, k, :, :])
            axes[i, j].text(
                3, 40, f"AU-{k}",
                fontsize=7,
                bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8}
                    )

            k += 1

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()



def fast_draw_landmarks(img: np.ndarray,
                        heatmap: np.ndarray,
                        x_h: list,
                        y_w: list,
                        wfp
                        ):

    alpha = 1.
    markersize = 4.5
    lw = 1.
    color = 'r'
    markeredgecolor = 'red'

    height, width = img.shape[:2]
    # just for plotting
    # avoid plotting right over thr border. dots spill outside the image.
    x_h = [max(min(x, height - 1), 0) for x in x_h]
    y_w = [max(min(y, width - 1), 0) for y in y_w]

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)
    axes[0, 0].imshow(img[:, :, ::-1])

    nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

    # close eyes and mouths
    plot_close = lambda i1, i2: axes[0, 0].plot([x_h[i1], x_h[i2]],
                                                [y_w[i1], y_w[i2]],
                                                color=color,
                                                lw=lw,
                                                alpha=alpha - 0.1
                                                )
    plot_close(41, 36)
    plot_close(47, 42)
    plot_close(59, 48)
    plot_close(67, 60)

    for ind in range(len(nums) - 1):
        l, r = nums[ind], nums[ind + 1]
        axes[0, 0].plot(x_h[l:r], y_w[l:r], color=color, lw=lw,
                        alpha=alpha - 0.1)

        axes[0, 0].plot(x_h[l:r], y_w[l:r], marker='o', linestyle='None',
                        markersize=markersize, color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)

    axes[0, 1].imshow(heatmap)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()


def fast_draw_heatmap(img: np.ndarray,
                      heatmap: np.ndarray,
                      cl: str,
                      wfp: str,
                      binary_roi: np.ndarray = None,
                      img_msk_black: np.ndarray = None,
                      img_msk_avg: np.ndarray = None,
                      img_msk_blur: np.ndarray = None
                      ):

    alpha = 1.
    markersize = 4.5
    lw = 1.
    color = 'r'
    markeredgecolor = 'red'

    height, width = img.shape[:2]

    ncols = 3

    if binary_roi is not None:
        ncols += 1

    if img_msk_black is not None:
        ncols += 1

    if img_msk_avg is not None:
        ncols += 1

    if img_msk_blur is not None:
        ncols += 1

    fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False)

    fontsize = 7 if ncols == 2 else 3

    i = 0
    axes[0, i].imshow(img[:, :, ::-1])

    i += 1

    axes[0, i].imshow(img[:, :, ::-1])
    axes[0, i].imshow(heatmap, alpha=0.7)
    axes[0, i].text(
        3, 40, cl,
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor':'none'}
    )

    i += 1


    axes[0, i].imshow(heatmap, alpha=1.0)
    axes[0, i].text(
        3, 40, cl,
        fontsize=fontsize,
        bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8, 'edgecolor': 'none'}
    )

    i += 1

    if binary_roi is not None:
        axes[0, i].imshow(binary_roi.astype(np.uint8) * 255, cmap='gray')
        axes[0, i].text(
            3, 40, 'ROI',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1

    if img_msk_black is not None:
        axes[0, i].imshow(img_msk_black[:, :, ::-1])
        axes[0, i].text(
            3, 40, 'Black masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1

    if img_msk_avg is not None:
        axes[0, i].imshow(img_msk_avg[:, :, ::-1])
        axes[0, i].text(
            3, 40, 'Average masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1

    if img_msk_blur is not None:
        axes[0, i].imshow(img_msk_blur[:, :, ::-1])
        axes[0, i].text(
            3, 40, 'Gaussian blur masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        i += 1



    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()

def switch_key_val_dict(d: dict) -> dict:
    out = dict()
    for k in d:
        assert d[k] not in out, 'more than 1 key with same value. wrong.'
        out[d[k]] = k

    return out


def get_avg_image_pixel(ds: str) -> np.ndarray:
    import cv2
    from dlib.utils.tools import get_root_wsol_dataset
    # from dlib.datasets.wsol_loader import RandomImMaskViaHeatMap

    split = constants.TRAINSET

    announce_msg(f"Computing the average pixel of trainset. DS: {ds}...")

    metadata_root = join(root_dir, f"folds/wsol-done-right-splits/{ds}/{split}")
    metadata = configure_metadata(metadata_root)
    landmarks = get_landmarks(metadata)
    labels = get_class_labels(metadata)

    folds_path = join(root_dir,
                      join(root_dir, f"folds/wsol-done-right-splits/{ds}")
                      )
    path_class_id = join(folds_path, 'class_id.yaml')
    with open(path_class_id, 'r') as fcl:
        cl_int = yaml.safe_load(fcl)

    n = len(list(landmarks.keys()))

    baseurl = get_root_wsol_dataset()

    all_avg = 0.0
    img_type = None

    for im_id in tqdm.tqdm(landmarks, total=n, ncols=80):
        path = join(baseurl, ds, im_id)
        assert os.path.isfile(path), path

        img = cv2.imread(path)

        avg = img.mean(0).mean(0).reshape((1, 1, 3))
        avg = avg.clip(0, 255).astype(img.dtype)

        all_avg += avg

        img_type = img.dtype

    all_avg = all_avg / float(n)
    all_avg = all_avg.clip(0, 255).astype(img_type)

    return all_avg


def test_heatmap_from_action_units(ds: str,
                                   split: str,
                                   aus_type: str,
                                   avg_train_pixel: np.ndarray,
                                   clean: bool
                                   ):
    import cv2
    from dlib.utils.tools import get_root_wsol_dataset
    # from dlib.datasets.wsol_loader import RandomImMaskViaHeatMap

    assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS], aus_type

    announce_msg(f"Test of plotting heatmaps from landmarks. "
                 f"DS: {ds}, Split: {split}, AUS type: {aus_type}...")

    metadata_root = join(root_dir, f"folds/wsol-done-right-splits/{ds}/{split}")
    metadata = configure_metadata(metadata_root)
    landmarks = get_landmarks(metadata)
    labels = get_class_labels(metadata)

    folds_path = join(root_dir,
                      join(root_dir, f"folds/wsol-done-right-splits/{ds}")
                      )
    path_class_id = join(folds_path, 'class_id.yaml')
    with open(path_class_id, 'r') as fcl:
        cl_int = yaml.safe_load(fcl)

    int_cl = switch_key_val_dict(cl_int)

    n = len(list(landmarks.keys()))

    baseurl = get_root_wsol_dataset()
    variance = 64.
    fdout = join(root_dir, f'data/debug/out/{aus_type}/{ds}')
    fdout_all_au = join(root_dir, f'data/debug/out/all-{aus_type}/{ds}')

    if os.path.isdir(fdout) and clean:
        print(f" deleting {fdout}")
        os.system(f"rm -r {fdout}")

    if os.path.isdir(fdout_all_au) and clean:
        print(f" deleting {fdout_all_au}")
        os.system(f"rm -r {fdout_all_au}")

    os.makedirs(fdout, exist_ok=True)
    os.makedirs(fdout_all_au, exist_ok=True)

    h = constants.SZ224
    w = constants.SZ224
    cov = np.zeros((2, 2), dtype=np.float32)
    np.fill_diagonal(cov, variance)
    normalize = True
    scale = 16.
    show_local = False
    failed_inf = 0

    jj = 0

    for im_id in tqdm.tqdm(landmarks, total=n, ncols=80):

        cl: int = labels[im_id]
        cl: str = int_cl[cl]

        if aus_type == constants.HEATMAP_AUNITS_LNMKS:
            if cl == constants.NEUTRAL:
                continue

        if not show_local:
            path = join(baseurl, ds, im_id)
            assert os.path.isfile(path), path

            img = cv2.imread(path)

        _h, _w = img.shape[:2]
        h = _h
        w = _w

        lndmks = landmarks[im_id]
        # to int
        if lndmks[0][0] == np.inf:
            continue

        lndmks = [(int(z[0]), int(z[1])) for z in lndmks]
        t0 = time.perf_counter()
        heatmaps = build_all_action_units(lndmks=lndmks,
                                          h=h,
                                          w=w,
                                          cl=cl,
                                          aus_type=aus_type
                                          )
        # n_au, h, w

        if np.isinf(heatmaps).sum() > 0:
            failed_inf += 1

            continue

        # debug:
        # show_all_action_untis(img=img, aus_maps=heatmap,
        #                       wfp=join(fdout_all_au,
        #                                f'{reformat_id(im_id)}.jpg'))

        # visualize.
        heatmap = heatmaps.max(axis=0, initial=-1)  # h, w
        assert heatmap.shape == (h, w), f"{heatmap.shape} {(h, w)}"

        if normalize:
            _min = heatmap.min()
            _max = heatmap.max()
            _deno = _max - _min

            if _deno == 0:
                _deno = 1.
            heatmap = (heatmap - _min) / _deno

        heatmap = heatmap.astype(np.float32)

        t1 = time.perf_counter()
        print(f"Time: {t1 - t0} (s)")

        assert np.isnan(heatmaps).sum() == 0, 'nan'
        assert np.isinf(heatmaps).sum() == 0, 'inf'

        # ----------------------------------------------------------------------
        binary_aus_roi = None

        otsu_thresh = threshold_otsu(heatmap)
        binary_aus_roi = (heatmap >= otsu_thresh).astype(np.float32)

        radius_dilate = 0

        if radius_dilate > 0:
            binary_aus_roi = binary_dilation(binary_aus_roi,
                                             footprint=disk(
                                                 radius=radius_dilate,
                                                 dtype=np.float32,
                                                 strict_radius=True
                                             )
                                             ).astype(np.float32)

        img_msk_black = get_masked_img_black(img, binary_aus_roi)
        img_msk_avg = get_masked_img_avg(img, binary_aus_roi, avg_train_pixel)
        img_mask_blur = get_masked_img_blur(img, binary_aus_roi, sigma=30)

        # test data transformer: wsol_loader.RandomImMaskViaHeatMap()

        # roi_dilation = True
        # roi_dilation_radius = 9
        #
        # t_black = RandomImMaskViaHeatMap(p=1., bg_filler=constants.BG_F_BLACK,
        #                                  roi_dilation=roi_dilation,
        #                                  roi_dilation_radius=roi_dilation_radius)
        #
        # t_avg = RandomImMaskViaHeatMap(p=1., bg_filler=constants.BG_F_IM_AVG,
        #                                roi_dilation=roi_dilation,
        #                                roi_dilation_radius=roi_dilation_radius)
        # t_gaus_blur = RandomImMaskViaHeatMap(
        #     p=1.,
        #     bg_filler=constants.BG_F_GAUSSIAN_BLUR,
        #     bg_filler_gauss_sigma=30.,
        #     roi_dilation=roi_dilation,
        #     roi_dilation_radius=roi_dilation_radius)

        # img_msk_black, _, _, _, _ = t_black(
        #     Image.fromarray(img, 'RGB'), None, None,None,
        #     torch.from_numpy(heatmap).unsqueeze(0))
        # img_msk_black = np.array(img_msk_black)
        #
        # img_msk_avg, _, _, _, _ = t_avg(
        #     Image.fromarray(img, 'RGB'), None, None, None,
        #     torch.from_numpy(heatmap).unsqueeze(0))
        # img_msk_avg = np.array(img_msk_avg)
        #
        # img_mask_blur, _, _, _, _ = t_gaus_blur(
        #     Image.fromarray(img, 'RGB'), None, None, None,
        #     torch.from_numpy(heatmap).unsqueeze(0))
        # img_mask_blur = np.array(img_mask_blur)

        # ----------------------------------------------------------------------

        if not show_local:
            fast_draw_heatmap(
                img, heatmap, cl,
                wfp=join(fdout, f'{reformat_id(im_id)}.jpg'),
                binary_roi=None,
                img_msk_black=None,
                img_msk_avg=None,
                img_msk_blur=None
            )

        # Show
        if show_local:
            plt.imshow(heatmap)
            plt.show()
            interp = interpolate_via_pytorch(heatmap, int(h / scale),
                                             int(w / scale))
            plt.imshow(interp)
            plt.show()
            break

        jj += 1

        if jj == 5:
            pass

    announce_msg(f"End test of plotting heatmaps from landmarks. "
                 f"DS: {ds}, Split: {split}, AUS type: {aus_type}...")
    msg = f"Ds: {ds}. Split: {split}. AUs type: {aus_type}.\n"
    msg += f"Total: {n}\nSuccess: {n - failed_inf}\nFailed: {failed_inf}."
    announce_msg(msg)


def get_masked_img_black(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    assert img.ndim == 3, img.ndim
    assert roi.ndim == 2, roi.ndim

    _roi = np.expand_dims(roi.copy(), axis=2)

    return (img * _roi).astype(img.dtype)


def get_masked_img_avg(img: np.ndarray,
                       roi: np.ndarray,
                       avg_train_pixel: np.ndarray
                       ) -> np.ndarray:

    assert img.ndim == 3, img.ndim
    assert avg_train_pixel.ndim == 3, avg_train_pixel.ndim  # 1, 1, 3.
    assert avg_train_pixel.shape == (1, 1, 3), avg_train_pixel.shape
    assert roi.ndim == 2, roi.ndim

    _roi = np.expand_dims(roi.copy(), axis=2)

    # avg = img.mean(0).mean(0).reshape((1, 1, 3))
    # avg = avg.clip(0, 255).astype(img.dtype)
    avg = avg_train_pixel.astype(img.dtype)
    _im_avg = img * 0.0 + avg

    new_img = img * _roi + _im_avg * (1. - _roi)

    return new_img.astype(img.dtype)


def get_masked_img_blur(img: np.ndarray, roi: np.ndarray,
                        sigma: float) -> np.ndarray:
    assert img.ndim == 3, img.ndim
    assert roi.ndim == 2, roi.ndim
    assert sigma > 0, sigma

    _roi = np.expand_dims(roi.copy(), axis=2)

    _blurred_img = gaussian(img, sigma=sigma, preserve_range=True,
                            channel_axis=-1)

    new_img = img * _roi + _blurred_img * (1. - _roi)

    return new_img.astype(img.dtype)


def interpolate_via_pytorch(x: np.ndarray,
                            new_h: int,
                            new_w: int
                            ) -> np.ndarray:
    a = torch.from_numpy(x)
    assert a.ndim == 2, a.ndim
    a = a.unsqueeze(0).unsqueeze(0)  # 1, 1, h, w.
    new_a = F.interpolate(input=a,
                          size=(new_h, new_w),
                          mode='bicubic'
                          )
    new_a = new_a.squeeze()
    return new_a.numpy().astype(x.dtype)


def landmarks_to_heatmap(h: int,
                         w: int,
                         lndmks: list,
                         cov: np.ndarray,
                         normalize: bool
                         ) -> np.ndarray:
    return np.ones((h, w), dtype=np.float32)

def store_heatmaps_of_action_units(ds: str,
                                   split: str,
                                   clean: bool,
                                   aus_type: str
                                   ):

    from dlib.utils.tools import get_root_wsol_dataset
    from dlib.utils.tools import get_heatmap_tag
    from dlib.configure.config import get_config
    from dlib.utils.tools import Dict2Obj
    from dlib.utils.shared import reformat_id

    assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                        constants.HEATMAP_PER_CLASS_AUNITS_LNMKS
                        ], aus_type

    announce_msg(f"Storing heatmaps from landmarks. "
                 f"DS: {ds}, Split: {split}, AUS type: {aus_type}...")

    metadata_root = join(root_dir, f"folds/wsol-done-right-splits/{ds}/{split}")
    metadata = configure_metadata(metadata_root)
    landmarks = get_landmarks(metadata)
    labels = get_class_labels(metadata)

    folds_path = join(root_dir,
                      join(root_dir, f"folds/wsol-done-right-splits/{ds}")
                      )
    path_class_id = join(folds_path, 'class_id.yaml')
    with open(path_class_id, 'r') as fcl:
        cl_int = yaml.safe_load(fcl)

    int_cl = switch_key_val_dict(cl_int)

    l_classes = []
    int_classes = sorted(list(int_cl.keys()), reverse=False)
    for k in int_classes:
        l_classes.append(int_cl[k])

    n = len(list(landmarks.keys()))

    # config
    normalize = True
    args = get_config(ds)
    args = Dict2Obj(args)
    args.align_atten_to_heatmap_type_heatmap = aus_type
    args.align_atten_to_heatmap_normalize = normalize
    tag = get_heatmap_tag(args, key=constants.ALIGN_ATTEN_HEATMAP)

    baseurl = get_root_wsol_dataset()
    outdir = join(baseurl, tag)
    if os.path.isdir(outdir) and clean:
        print(f" deleting {outdir}")
        os.system(f"rm -r {outdir}")

    os.makedirs(outdir, exist_ok=True)
    print(f"Destination: {outdir}")

    h, w = constants.SZ224, constants.SZ224

    failed_hmap = np.zeros((1, h, w)) + np.inf
    nbr_failure = 0

    for im_id in tqdm.tqdm(landmarks, total=n, ncols=80):

        cl: int = labels[im_id]
        cl: str = int_cl[cl]

        lndmks = landmarks[im_id]

        path_img = join(baseurl, ds, im_id)

        if aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS]:

            heatmap, failed = get_single_class_aus_heatmap(
                aus_type=aus_type,
                lndmks=lndmks,
                cl=cl,
                path_img=path_img,
                failed_hmap=failed_hmap
            )

            mtx = heatmap

        elif aus_type == constants.HEATMAP_PER_CLASS_AUNITS_LNMKS:
            heatmaps, failed = get_all_per_class_aus_heatmap(
                aus_type=aus_type,
                lndmks=lndmks,
                path_img=path_img,
                l_classes=l_classes,
                failed_hmap=failed_hmap
            )

            mtx = heatmaps

        else:
            raise NotImplementedError(aus_type)


        path_out = join(outdir, f"{reformat_id(im_id)}.npy")
        # mtx: h, w. or c, h, w.
        np.save(path_out, mtx, allow_pickle=False, fix_imports=True)

        nbr_failure += failed

    announce_msg(f"End storing heatmaps from landmarks. "
                 f"DS: {ds}, Split: {split}, AUS type: {aus_type}...")

    # log
    folder = f'data/debug/out/{aus_type}'
    os.makedirs(folder, exist_ok=True)

    log_file = join(
        root_dir, f'{folder}/'
                  f'log-failure-ds-{ds}-aus_type-{aus_type}-split-{split}.txt')

    msg = f"Dataset: {ds}. Split: {split}. AUs type: {aus_type}.\n"
    msg += f"Total: {n}\nSuccess: {n - nbr_failure}\nFailed: {nbr_failure}.\n"
    with open(log_file, 'w') as fx:
        fx.write(msg)

    announce_msg(msg)


def get_all_per_class_aus_heatmap(aus_type: str,
                                  lndmks: list,
                                  path_img: str,
                                  l_classes: list,
                                  failed_hmap: np.ndarray
                                  ) -> Tuple[np.ndarray, bool]:

    assert aus_type == constants.HEATMAP_PER_CLASS_AUNITS_LNMKS, aus_type
    assert failed_hmap.ndim == 3, failed_hmap.ndim  # h, w

    assert isinstance(l_classes, list), type(l_classes)
    n = len(l_classes)
    assert n > 0, n

    failed_hmap2 = failed_hmap.squeeze(0)

    default_full_fail_heatmap = np.repeat(np.expand_dims(failed_hmap2, axis=0),
                                          n, axis=0)  # n, h, w

    if lndmks[0][0] == np.inf:
        failed = True
        return default_full_fail_heatmap, failed

    assert os.path.isfile(path_img), path_img
    img = cv2.imread(path_img)

    fail_collect = []
    _aus_type = constants.HEATMAP_AUNITS_LNMKS
    heatmap = None

    for cl in l_classes:

        assert isinstance(cl, str), type(cl)

        _heatmap, _failed = get_single_class_aus_heatmap(aus_type=_aus_type,
                                                         lndmks=lndmks,
                                                         cl=cl,
                                                         path_img=path_img,
                                                         failed_hmap=failed_hmap,
                                                         img=img
                                                         )  # h, w
        fail_collect.append(_failed)

        assert _heatmap.ndim == 2, _heatmap.ndim

        _heatmap = np.expand_dims(_heatmap, axis=0)
        if heatmap is None:
            heatmap = _heatmap

        else:
            heatmap = np.concatenate((heatmap, _heatmap), axis=0)

    return heatmap, all(fail_collect)


def get_single_class_aus_heatmap(aus_type: str,
                                 lndmks: list,
                                 cl: str,
                                 path_img: str,
                                 failed_hmap: np.ndarray,
                                 img: np.ndarray = None
                                 ) -> Tuple[np.ndarray, bool]:
    assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS], aus_type

    assert failed_hmap.ndim == 3, failed_hmap.ndim

    failed = False

    if aus_type == constants.HEATMAP_AUNITS_LNMKS:
        if cl == constants.NEUTRAL or lndmks[0][0] == np.inf:
            failed = True

    elif aus_type == constants.HEATMAP_GENERIC_AUNITS_LNMKS:
        if lndmks[0][0] == np.inf:
            failed = True

    else:
        raise NotImplementedError(aus_type)

    if not failed:
        assert isinstance(cl, str), type(cl)

        if img is None:
            assert os.path.isfile(path_img), path_img
            img = cv2.imread(path_img)

        _h, _w = img.shape[:2]
        h = _h
        w = _w

        lndmks = [(int(z[0]), int(z[1])) for z in lndmks]
        heatmaps = build_all_action_units(lndmks=lndmks, h=h, w=w, cl=cl,
                                          aus_type=aus_type
                                          )  # z, h, w

        # failure:
        if np.isinf(heatmaps).sum() > 0:
            failed = True

    if failed:
        heatmaps = failed_hmap

    # store only the final map.
    heatmap = heatmaps.max(axis=0, initial=-1)  # h, w

    return heatmap, failed


def build_heatmaps_from_action_units(ds: str, aus_type: str):

    if aus_type == constants.HEATMAP_GENERIC_AUNITS_LNMKS:
        folder = f'data/debug/out/{aus_type}'
        os.makedirs(folder, exist_ok=True)

        log_file_generic_aus = join(
            root_dir, f'{folder}/ds-log-store-{aus_type}.txt')
        dump_stats_generic_aus_into_file(log_file_generic_aus)

    i = 0
    for split in [constants.TRAINSET, constants.VALIDSET, constants.TESTSET]:
        clean = (i == 0)
        store_heatmaps_of_action_units(ds=ds,
                                       split=split,
                                       clean=clean,
                                       aus_type=aus_type
                                       )

        i += 1

def dump_stats_generic_aus_into_file(file_path: str):
    with open(file_path, 'a') as fout:
        msg = dt.datetime.now().strftime('%m %d %Y:%H %M %S %f')
        fout.write(f"{msg}: Generic AUS: {GENERIC_AUS}. \n")


def visualize_aus(ds: str, aus_type: str):

    assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS], aus_type

    if aus_type == constants.HEATMAP_GENERIC_AUNITS_LNMKS:
        folder = f'data/debug/out/{aus_type}'
        os.makedirs(folder, exist_ok=True)

        log_file_generic_aus = join(
            root_dir, f'{folder}/ds-log-visual-{aus_type}.txt')
        dump_stats_generic_aus_into_file(log_file_generic_aus)

    avg_train_pixel = get_avg_image_pixel(ds)

    i = 0
    for split in [constants.TESTSET,
                  # constants.VALIDSET,
                  # constants.TRAINSET
                  ]:
        clean = (i == 0)
        test_heatmap_from_action_units(ds=ds,
                                       split=split,
                                       aus_type=aus_type,
                                       avg_train_pixel=avg_train_pixel,
                                       clean=clean
                                       )

        i += 1


if __name__ == "__main__":
    _ACTION_VIS = 'visualize'
    _ACTION_STORE = 'store'
    _ACTIONS = [_ACTION_STORE, _ACTION_VIS]

    from dlib.utils.shared import str2bool
    from dlib.utils.shared import announce_msg

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default=_ACTION_VIS,
                        help="action: visualize or store aus maps.")
    parser.add_argument("--aus_type", type=str,
                        default=constants.HEATMAP_AUNITS_LNMKS,
                        help="Action units type.")
    parser.add_argument("--dataset", type=str,
                        default=constants.RAFDB,
                        required=True,
                        help="Dataset name: raf-db, affectnet.")

    args = parser.parse_args()
    aus_type = args.aus_type
    assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                        constants.HEATMAP_PER_CLASS_AUNITS_LNMKS], aus_type

    ds_name = args.dataset
    action = args.action
    assert action in _ACTIONS, action
    assert ds_name in [constants.RAFDB, constants.AFFECTNET], ds_name

    ds = ds_name

    if action == _ACTION_VIS:

        assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                            constants.HEATMAP_GENERIC_AUNITS_LNMKS], aus_type

        visualize_aus(ds, aus_type)

    elif action == _ACTION_STORE:
        build_heatmaps_from_action_units(ds, aus_type)

    else:
        raise NotImplementedError(args.action)
