import os
import sys
import time
from os.path import dirname, abspath, join, basename
from typing import Optional, Union, Tuple

import cv2
import facealignment

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

# require python 3.8.

if __name__ == "__main__":
    s= 256

    paths = [join(root_dir, 'data/debug/input/test_0006.jpg'),
             join(root_dir, 'data/debug/input/test_0038.jpg'),
             join(root_dir, 'data/debug/input/test_0049.jpg'),
             join(root_dir, 'data/debug/input/test_0067.jpg')
             ]

    # Instantiate FaceAlignmentTools class
    tool = facealignment.FaceAlignmentTools()

    for im_path in paths:
        print(f"processing {im_path}")
        assert os.path.isfile(im_path), im_path
        bs = basename(im_path)
        single_face = cv2.imread(im_path)

        # MTCNN need RGB instead of CV2-BGR images
        single_face = cv2.cvtColor(single_face, cv2.COLOR_BGR2RGB)


        # Align image with single face

        aligned_img = tool.align(single_face, dsize=(s, s),
                                 allow_multiface=False)
        print(type(aligned_img))
        if aligned_img is not None:
            print(aligned_img.shape, aligned_img.dtype, aligned_img.max())
            screen_img = cv2.hconcat([single_face, aligned_img])
            screen_img = cv2.cvtColor(screen_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(bs, screen_img)

        else:
            print(f"Failed to process {im_path}")
        # cv2.imshow("Aligned Example Image", screen_img)
        # cv2.waitKey(0)

