import math
import random
import copy
import csv
import sys
import os
from os.path import join, dirname, abspath, basename

import numpy as np
import tqdm
import yaml
from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants

CROP_SIZE = constants.SZ224
SEED = 0

# Align face:
ALIGN_DEEP_FACE = 'deep_face'  # https://github.com/serengil/deepface
ALIGN_FACE_EVOLVE = 'face_evolve'  # https://github.com/ZhaoJ9014/face.evoLVe


def switch_key_val_dict(d: dict) -> dict:
    out = dict()
    for k in d:
        assert d[k] not in out, 'more than 1 key with same value. wrong.'
        out[d[k]] = k

    return out


def per_cl_stats(l: list) -> dict:
    out = {}
    l = np.array(l)
    classes = np.unique(l)  # sorted, increasing order.
    print(f"Unique classes {classes}")

    for k in classes:
        c = 0
        for e in l:
            if e == k:
                c += 1

        out[int(k)] = float(c)

    return out


def print_args_info(args: object):
    msg = f"Dataset: {args.dataset} \n" \
          f"VL%: {args.folding['vl']} \n"
    # if args.crop_align:
    msg += f"crop_align: {args.crop_align} \n"
    msg += f"align_type: {args.align_type} \n"
    msg += f"align: {args.align} \n"
    msg += f"bbx_upscale: {args.bbx_upscale} \n"

    # if args.use_original_align:
    msg += f"use_original_align: {args.use_original_align} \n"

    msg += f"test_set_as_validset: {args.test_set_as_validset}"
    print(msg)


def class_balance_stat(l: list, name: str, int_to_str: dict) -> str:
    msg = f"{80 * '*'} \n"
    msg += f"Set: {name} (total: {len(l)})\n"

    for k in int_to_str:
        c = 0
        for e in l:
            if e == k:
                c += 1

        msg += f"{int_to_str[k]}: {c} samples. \n"

    return msg


def dump_set(data: list, outfd: str):
    # data [[img_id(str), cl(int)]].

    print(f"Dumping in {outfd}")
    outfolds = outfd

    os.makedirs(outfolds, exist_ok=True)

    img_ids_fids = open(join(outfolds, 'image_ids.txt'), 'w')
    cl_labels_fids = open(join(outfolds, 'class_labels.txt'), 'w')
    img_size_fids = open(join(outfolds, 'image_sizes.txt'), 'w')
    loc_fids = open(join(outfolds, 'localization.txt'), 'w')

    idens = []
    z = constants.SZ224 - 1
    bbox = [0, 0, 223, 223]

    for iden, cl in data:
        assert iden not in idens, iden

        idens.append(iden)

        # class_labels.txt
        # <path>,<integer_class_label>
        # path/to/image1.jpg,0
        # path/to/image2.jpg,1
        # path/to/image3.jpg,1

        img_ids_fids.write(f'{iden}\n')

        # class_labels.txt
        # <path>,<integer_class_label>
        # path/to/image1.jpg,0
        # path/to/image2.jpg,1
        # path/to/image3.jpg,1

        cl_labels_fids.write(f'{iden},{cl}\n')

        # image_sizes.txt
        # <path>,<w>,<h>
        # path/to/image1.jpg,500,300
        # path/to/image2.jpg,1000,600
        # path/to/image3.jpg,500,300

        img_size_fids.write(f'{iden},{z},{z}\n')

        # localization.txt
        # <path>,<x0>,<y0>,<x1>,<y1>
        # path/to/image1.jpg,156,163,318,230
        # path/to/image1.jpg,23,12,101,259
        # path/to/image2.jpg,143,142,394,248
        # path/to/image3.jpg,28,94,485,303

        loc_fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')

    img_ids_fids.close()
    cl_labels_fids.close()
    img_size_fids.close()
    loc_fids.close()


    # image_ids.txt
    # <path>
    # path/to/image1.jpg
    # path/to/image2.jpg
    # path/to/image3.jpg

    # with open(join(outfolds, 'image_ids.txt'), 'w') as fids:
    #     idens = []
    #     for iden, cl in data:
    #         if iden not in idens:
    #             idens.append(iden)
    #             fids.write(f'{iden}\n')

    # class_labels.txt
    # <path>,<integer_class_label>
    # path/to/image1.jpg,0
    # path/to/image2.jpg,1
    # path/to/image3.jpg,1

    # with open(join(outfolds, 'class_labels.txt'), 'w') as fids:
    #     idens = []
    #     for iden, cl in data:
    #         if iden not in idens:
    #             idens.append(iden)
    #             fids.write(f'{iden},{cl}\n')

    # image_sizes.txt
    # <path>,<w>,<h>
    # path/to/image1.jpg,500,300
    # path/to/image2.jpg,1000,600
    # path/to/image3.jpg,500,300

    # z = constants.SZ224 - 1
    #
    # with open(join(outfolds, 'image_sizes.txt'), 'w') as fids:
    #     idens = []
    #     for iden, cl in data:
    #         if iden not in idens:
    #             idens.append(iden)
    #             fids.write(f'{iden},{z},{z}\n')

    # localization.txt
    # <path>,<x0>,<y0>,<x1>,<y1>
    # path/to/image1.jpg,156,163,318,230
    # path/to/image1.jpg,23,12,101,259
    # path/to/image2.jpg,143,142,394,248
    # path/to/image3.jpg,28,94,485,303

    # bbox = [0, 0, 223, 223]
    # with open(join(outfolds, 'localization.txt'), 'w') as fids:
    #     for iden, cl in data:
    #         fids.write(f'{iden},{",".join([str(z) for z in bbox])}\n')