import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple
import argparse


import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('nbAgg')

from synergy3DMM import SynergyNet



# from FaceBoxes import FaceBoxes
# from utils.render import render

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.tools import get_root_wsol_dataset

# dir_synergy3dmm = join(root_dir, 'SynergyNet')
# sys.path.append(dir_synergy3dmm)

from SynergyNet.utils.ddfa import ToTensor, Normalize
from SynergyNet.utils.inference import predict_sparseVert, draw_landmarks
# from SynergyNet.model_building import SynergyNet

from dlib.utils.shared import reformat_id

from dlib.configure import constants


def fast_draw_landmarks(img: np.ndarray,
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

    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=True)
    plt.imshow(img[:, :, ::-1])

    nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

    # close eyes and mouths
    plot_close = lambda i1, i2: plt.plot([x_h[i1], x_h[i2]],
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
        plt.plot(x_h[l:r], y_w[l:r], color=color, lw=lw,
                 alpha=alpha - 0.1)

        plt.plot(x_h[l:r], y_w[l:r], marker='o', linestyle='None',
                 markersize=markersize, color=color,
                 markeredgecolor=markeredgecolor, alpha=alpha)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()


def init_run():

    name= 'test_0002'
    img_path = join(root_dir, f'data/debug/input/faces/{name}.jpg')
    assert os.path.isfile(img_path), img_path

    img = cv2.imread(img_path)
    use_cuda = True
    if use_cuda:
        cuda_id = 0
        torch.cuda.set_device(cuda_id)
        device = torch.device(f'cuda:{cuda_id}')
    else:
        device = torch.device('cpu')

    model = SynergyNet().to(device)
    t0 = time.perf_counter()
    lmk3d, mesh, pose = model.get_all_outputs(img)
    t1 = time.perf_counter()
    print(f"Time: {t1 - t0} s (CUDA: {use_cuda})")
    # lmk3d: list of n (faces).
    # each element is np array, float64: 3, 68 shape. x, y, z.
    # coordinates: x: height, y: width.
    print(lmk3d, type(lmk3d), len(lmk3d), type(lmk3d[0]), lmk3d[0].shape,
          lmk3d[0].dtype)

    outd = join(root_dir, f'data/debug/input/faces-landmarks')
    os.makedirs(outd, exist_ok=True)
    x = lmk3d[0][0, :].tolist()
    y = lmk3d[0][1, :].tolist()
    # draw_landmarks(img, lmk3d, wfp=f'{outd}/{name}.jpg')
    fast_draw_landmarks(img, x_h=x, y_w=y, wfp=f'{outd}/{name}.jpg')


def run_split(ds_name: str, split: str, plot_landmarks: bool):
    ims_id_path = join(root_dir, constants.RELATIVE_META_ROOT, ds_name, split,
                       'image_ids.txt')
    assert os.path.isfile(ims_id_path), ims_id_path

    baseurl = get_root_wsol_dataset()
    store_draw_lnd_im = join(root_dir, f'data/debug/out/landmarks/{ds_name}')

    if plot_landmarks:
        os.makedirs(store_draw_lnd_im, exist_ok=True)
        print(f"Will store drawn images with landmarks in {store_draw_lnd_im}")

    image_ids = []

    with open(ims_id_path, 'r') as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))

    out_landmark_path = join(root_dir, constants.RELATIVE_META_ROOT, ds_name,
                             split, 'landmarks.txt')
    log_f = join(root_dir, constants.RELATIVE_META_ROOT, ds_name, split,
                'log-landmarks.txt')

    lndmks = []
    null_lnds = [np.inf for _ in range(68)]

    nbr_failed = 0
    nbr_multiple_faces = 0
    nbr_success = 0
    neg_coord = 0
    model = SynergyNet()

    for im_id in tqdm.tqdm(image_ids, total=len(image_ids), ncols=80):
        path = join(baseurl, ds_name, im_id)
        assert os.path.isfile(path)

        img = cv2.imread(path)

        lmk3d, mesh, pose = model.get_all_outputs(img)
        # lmk3d: list of n (faces).
        # each element is np array, float64: 3, 68 shape. x, y, z.
        # coordinates: x: width, y: height.

        success = True
        if len(lmk3d) > 0:
            x = lmk3d[0][0, :].tolist()
            y = lmk3d[0][1, :].tolist()

            nbr_multiple_faces += (len(lmk3d) > 1)
            nbr_success += 1
            neg_coord += ((lmk3d[0][0] < 0)).sum() + ((lmk3d[0][1] < 0)).sum()
        else:
            x = null_lnds
            y = null_lnds

            nbr_failed += 1
            success = False


        lndmks.append([im_id, x, y])

        if plot_landmarks and success:
            fast_draw_landmarks(img, x_h=x, y_w=y,
                                wfp=join(store_draw_lnd_im,
                                         f'{reformat_id(im_id)}.jpg')
                                )

    with open(out_landmark_path, 'w') as fout:
        idens = []
        for im_id, x, y in lndmks:
            s = x + y
            s = [str(i) for i in s]
            s = ','.join(s)
            assert im_id not in idens, im_id
            idens.append(im_id)
            fout.write(f'{im_id},{s}\n')

    msg = f"Dataset: {ds_name}. Split: {split}.\n" \
          f"Total samples: {len(image_ids)}\n" \
          f"NBR success: {nbr_success}\n" \
          f"NBR failed: {nbr_failed}\n" \
          f"NBR multiple faces: {nbr_multiple_faces}\n" \
          f"NBR negative coordinates: {neg_coord}."

    with open(log_f, 'w') as fout:
        fout.write(msg)

    print(f"{msg} \nDone with split {split}.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", type=str2bool, default=False, required=True,
                        help="Draw or not landmarks.")
    parser.add_argument("--dataset", type=str,
                        default=constants.RAFDB,
                        required=True,
                        help="Dataset name: raf-db, affectnet.")

    args = parser.parse_args()
    ds_name = args.dataset
    draw = args.draw
    assert ds_name in [constants.RAFDB, constants.AFFECTNET], ds_name
    assert isinstance(draw, bool), type(draw)

   # init_run()

    print(f"Processing dataset (STORE/Draw[{draw}]): {ds_name}")
    for split in [constants.VALIDSET, constants.TRAINSET, constants.TESTSET]:
        print(f"Split: {split}")
        run_split(ds_name=ds_name, split=split, plot_landmarks=draw)
    # ==========================================================================