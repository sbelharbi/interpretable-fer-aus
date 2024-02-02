import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.tools import check_box_convention

_ALIGN_PATH = join(root_dir, 'face_evoLVe/applications/align')
sys.path.append(_ALIGN_PATH)

from face_evoLVe.applications.align.detector import detect_faces
from face_evoLVe.applications.align.align_trans import get_reference_facial_points
from face_evoLVe.applications.align.align_trans import warp_and_crop_face


__all__ = ['FaceEvolveAlign']


class FaceEvolveAlign(object):
    """
    Crop and align faces using https://github.com/ZhaoJ9014/face.evoLVe.
    """
    def __init__(self,
                 out_size: int = constants.SZ256,
                 verbose: bool = False,
                 no_warnings: bool = False
                 ):

        if no_warnings:
            warnings.filterwarnings("ignore")


        assert isinstance(out_size, int), type(out_size)
        assert out_size > 0, out_size

        self.out_size = out_size
        self.success = False

        assert isinstance(verbose, bool), type(verbose)
        self.verbose = verbose

        scale = out_size / 112.
        self.reference = get_reference_facial_points(
            default_square=True) * scale

    def _reset_success(self):
        self.success = False

    @staticmethod
    def bb_iou(box_a, box_b):
        # intersection_over_union
        # ref: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        assert isinstance(box_a, np.ndarray), type(box_a)
        assert isinstance(box_b, np.ndarray), type(box_b)

        assert box_a.ndim == 1, box_a.ndim
        assert box_b.ndim == 1, box_b.ndim
        assert box_a.shape == box_b.shape
        assert box_a.size == 4, box_a.size


        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])

        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)

        box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
        box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def get_closest_to_proposal(self,
                                bbxoes: np.ndarray,
                                p: np.ndarray) -> int:

        assert isinstance(bbxoes, np.ndarray), type(bbxoes)
        assert isinstance(p, np.ndarray), type(p)

        assert bbxoes.ndim == 2, bbxoes.ndim
        assert p.ndim == 1, p.ndim
        assert bbxoes.shape[1] == 4, bbxoes.shape[1]
        assert p.size == 4, p.size

        iou = []
        for i in range(bbxoes.shape[0]):
            iou.append(self.bb_iou(bbxoes[i], p))

        iou = np.array(iou)
        return iou.argmax()


    def align(self,
              img_path: str,
              proposed_bbx: np.ndarray = None
              ) -> np.ndarray:

        self._reset_success()

        input_img = Image.open(img_path, 'r').convert('RGB')
        success = True
        face = None

        try:  # Handle exception
            cwd = os.getcwd()
            os.chdir(_ALIGN_PATH)  # paths of weights are hardcoded (relative)
            bounding_boxes, landmarks = detect_faces(input_img)
            os.chdir(cwd)
            # bounding_boxes: np.ndarray (n, 5): n number of bbox.
            # 4 items:
            # 5th item: face score.

            n = len(landmarks)  # nbr faces.

            if n > 0:
                check_box_convention(bounding_boxes[:, :4], 'x0y0x1y1',
                                     tolerate_neg=True)

            if n == 0:  # if there is none.
                success = False

            elif n == 1:  # if there is one.
                success = True
                i = 0

            elif n > 1:  # if there are many faces
                success = True
                if proposed_bbx is not None:
                    i = self.get_closest_to_proposal(bounding_boxes[:, :4],
                                                     proposed_bbx)

                else:
                    i = bounding_boxes[:, -1].argmax()
                    assert i == 0, i



            if success:
                facial5points = [[landmarks[i][j], landmarks[i][j + 5]] for j
                                 in range(5)]
                warped_face = warp_and_crop_face(
                    np.array(input_img),
                    facial5points,
                    self.reference,
                    crop_size=(self.out_size, self.out_size)
                )
                # <class 'numpy.ndarray'> [0, 255] (sz, sz, 3) uint8
                face: np.ndarray = warped_face

        except Exception as e:
            success = False
            print('error ', repr(e))

        self.success = success

        if success:
            assert face is not None
            return face

        else:
            if self.verbose:
                print(f"Failed at {img_path}")
            if proposed_bbx is not None:
                assert proposed_bbx.ndim == 1, proposed_bbx.ndim
                assert proposed_bbx.size == 4, proposed_bbx.size

                left, upper, right, lower = proposed_bbx

                original_crop = input_img.crop((left, upper, right, lower))
                return np.array(original_crop).astype(np.uint8)

            else:
                return np.array(input_img).astype(np.uint8)


def test_FaceEvolveAlign():
    aligner = FaceEvolveAlign(out_size=constants.SZ256, verbose=True,
                              no_warnings=True)
    paths = [join(root_dir, 'data/debug/input/test_0006.jpg'),
             join(root_dir, 'data/debug/input/test_0038.jpg'),
             join(root_dir, 'data/debug/input/test_0049.jpg'),
             join(root_dir, 'data/debug/input/test_0067.jpg')
    ]

    for img_path in paths:
        face = aligner.align(img_path)
        plt.imshow(face)
        plt.axis("off")
        plt.show()
        print(f"Sucess-----------> {aligner.success}")

    from dlib.utils.shared import find_files_pattern
    paths = find_files_pattern(join(root_dir, 'data/debug/input/faces'),
                               "*.jpg")

    for img_path in paths:
        face = aligner.align(img_path)
        plt.imshow(face)
        plt.axis("off")
        plt.show()
        print(f"Sucess-----------> {aligner.success}")




if __name__ == "__main__":
    test_FaceEvolveAlign()