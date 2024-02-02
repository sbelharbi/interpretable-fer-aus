import os
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple


import numpy as np
import matplotlib.pyplot as plt

if "HOST_XXX" in os.environ.keys():
    if os.environ['HOST_XXX'] == 'gsys':
        # CUDA_VISIBLE_DEVICES="" python dlib/face_landmarks/face_align.py
        import tensorflow
        path = join(os.environ['CUSTOM_CUDNN'],
                    'cudnn-10.1-linux-x64-v7.6.0.64/lib64/libcudnn.so')
        tensorflow.load_library(path)
        tensorflow.config.set_visible_devices([], 'GPU')
        print(path, 'Tensorflow has been loaded early, '
                    'and gpu-usage has been disabled')


from deepface import DeepFace
from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants


__all__ = ['FaceAlign']


class FaceAlign(object):
    """
    Align faces using MTCNN or Retinaface.
    Try first with MTCNN if it fails, try Retinaface.
    Failed cases:
    - found more than 2 faces.
    - found no faces.
    if failure, we return the input image.
    """
    def __init__(self, out_size: int = constants.SZ256, verbose: bool = False):
        assert isinstance(out_size, int), type(out_size)
        assert out_size > 0, out_size

        self.out_size = out_size
        self.detector_backends = ["mtcnn", "retinaface"]
        self.success = False

        assert isinstance(verbose, bool), type(verbose)
        self.verbose = verbose

    def _reset_success(self):
        self.success = False

    def align(self, img_path: str, proposed_bbx: np.ndarray = None) -> np.ndarray:
        self._reset_success()

        input_img = Image.open(img_path, 'r').convert('RGB')
        input_img = np.array(input_img)  # h, w, 3. uint8.
        success = True
        face = None

        for detector_backend in self.detector_backends:
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=img_path,
                    detector_backend=detector_backend,
                    target_size=(self.out_size, self.out_size),
                    align=True,
                    grayscale=False
                )

                face = face_objs[0]['face']
                # number faces > 1 --> uncertain. discard it.
                success = (len(face_objs) == 1)
            except:
                success = False

            if success:
                break

        self.success = success

        # face: h, w, 3.
        if success:
            assert face is not None
            return (face * 255).astype(np.uint8)

        else:
            if self.verbose:
                print(f"Failed at {img_path}")

            return input_img.astype(np.uint8)


def test_FaceAlign():
    aligner = FaceAlign(out_size=constants.SZ256, verbose=True)
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


def test_alignment(img_path: str):
    detector_backends = ["opencv", "ssd", "mtcnn", "retinaface"]
    input_img = Image.open(img_path, 'r').convert('RGB')
    print(type(input_img))
    img_np = np.array(input_img)
    print(img_np.shape, img_np.dtype, img_np.min(), img_np.max())



    # extract faces
    for detector_backend in detector_backends:
        try:
            face_objs = DeepFace.extract_faces(
                img_path=img_path, detector_backend=detector_backend,
                target_size=(256, 256)
            )

            for face_obj in face_objs:
                face = face_obj["face"]
                print(detector_backend, len(face_objs), face_obj['confidence'])
                print(type(face), face.shape, face.dtype, face.min(), face.max())
                face = face * 255
                face = face.astype(np.uint8)
                plt.imshow(face)
                plt.axis("off")
                plt.show()
                print("-----------")
        except:
            print(detector_backend, 'didnt find a face.')


if __name__ == "__main__":

    # path_img = join(root_dir, 'data/debug/input/test_0006.jpg')
    path_img = join(root_dir, 'data/debug/input/test_0038.jpg')
    path_img = join(root_dir, 'data/debug/input/test_0049.jpg')
    # path_img = join(root_dir, 'data/debug/input/test_0067.jpg')
    # test_alignment(path_img)

    test_FaceAlign()


