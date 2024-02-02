import sys
from os.path import dirname, abspath
from typing import Optional, Union, List, Tuple

import tqdm
import torch
from pykeops.torch import LazyTensor

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import STDClModel

from dlib import poolings

from dlib.configure import constants
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg


__all__ = ['Knn']


class Knn(object):
    def __init__(self, k: int, cudaid: int):
        super(Knn, self).__init__()

        assert isinstance(k, int), type(k)
        assert k > 0, k

        self.k = k
        assert isinstance(cudaid, int), type(cudaid)
        assert cudaid >= 0, cudaid
        self.cudaid = cudaid

        self.train_fts = None
        self.train_lbls = None

        self.model = None
        self.log = ''

        self.already_set = False
    def set_k(self, k: int):
        assert isinstance(k, int), type(k)
        assert k > 0, k

        self.k = k

    def set_it(self, model, train_loader):

        self.log = ''

        device = torch.device(f'cuda:{self.cudaid}')
        self.model = model.to(device)
        self.model.eval()

        loader = train_loader

        fts = None
        lbls = None
        print('Building trainset...')

        for i, (images, targets, _, _, _, _, _) in tqdm.tqdm(
                enumerate(loader), ncols=80, total=len(loader)):
            images = images.to(device)
            targets = targets.to(device)
            self.model(images)
            c_ft = self.model.linear_features.detach()

            if fts is None:
                fts = c_ft
                lbls = targets
            else:
                fts = torch.cat([fts, c_ft], dim=0)
                lbls = torch.cat([lbls, targets], dim=0)

        assert fts.ndim == 2, fts.ndim
        assert lbls.ndim == 1, lbls.ndim
        assert fts.shape[0] == lbls.shape[0], f"{fts.shape[0]} | " \
                                              f"{lbls.shape[0]}"
        self.train_fts = fts
        self.train_lbls = lbls


        self.log = 'Building trainset is done.\n'
        self.log += f"Train samples: {fts.shape}. \n"
        self.log += f"Train labels: {lbls.shape} \n"

        self.already_set = True

    def prepapre_eval_set(self, eval_loader
                          ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        assert self.already_set

        device = torch.device(f'cuda:{self.cudaid}')
        loader = eval_loader

        fts = None
        lbls = None
        all_ids = []
        print('Building eval set...')

        for i, (images, targets, images_id, _, _, _, _) in tqdm.tqdm(
                enumerate(loader), ncols=80, total=len(loader)):
            images = images.to(device)
            targets = targets.to(device)
            self.model(images)
            c_ft = self.model.linear_features.detach()

            if fts is None:
                fts = c_ft
                lbls = targets
                all_ids = list(images_id)
            else:
                fts = torch.cat([fts, c_ft], dim=0)
                lbls = torch.cat([lbls, targets], dim=0)
                all_ids = all_ids + list(images_id)

        assert fts.ndim == 2, fts.ndim
        assert lbls.ndim == 1, lbls.ndim
        assert fts.shape[0] == lbls.shape[0], f"{fts.shape[0]} | " \
                                              f"{lbls.shape[0]}"

        self.log = 'Building eval set is done.\n'
        self.log += f"Eval samples: {fts.shape}. \n"
        self.log += f"Eval labels: {lbls.shape} \n"

        return fts, lbls, all_ids

    def evaluate(self, eval_loader) -> Tuple[torch.Tensor, torch.Tensor, List]:
        assert self.already_set

        vl_fts, vl_lbls, vl_ids = self.prepapre_eval_set(eval_loader)
        return self.evaluate_prepared_data(vl_fts, vl_lbls, vl_ids)

    def evaluate_prepared_data(self,
                               vl_fts: torch.Tensor,
                               vl_lbls: torch.Tensor,
                               vl_ids: List
                               )  -> Tuple[torch.Tensor, torch.Tensor, List]:

        assert self.already_set

        tr_fts = self.train_fts
        tr_lbls = self.train_lbls

        msg = f"{vl_fts.shape[1]} | {tr_fts.shape[1]}"
        assert vl_fts.shape[1] == tr_fts.shape[1], msg

        k = self.k

        x_i = LazyTensor(vl_fts[:, None, :])  # (n_evl, 1, d) eval set
        x_j = LazyTensor(tr_fts[None, :, :])  # (1, n_tr, d) train set
        d_ij = ((x_i - x_j) ** 2).sum(-1)  # (n_vl, n_tr) symbolic matrix of
        # squared L2 distances.

        ind_knn = d_ij.argKmin(k, dim=1)  # Samples <-> Dataset, (n_evl, k)
        lab_knn = tr_lbls[ind_knn]  # (n_evl, k) array of integers in [0,9]
        y_knn, _ = lab_knn.mode()  # Compute the most likely label

        assert y_knn.shape == vl_lbls.shape

        return y_knn, vl_lbls, vl_ids


def run_knn():
    cuda_id = 0
    device = torch.device(f'cuda:{cuda_id}')
    k = 1

    model = Knn(k=k, cudaid=cuda_id)


if __name__ == "__main__":
    run_knn()
