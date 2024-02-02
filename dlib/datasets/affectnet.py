"""
Create csv file of AFFECTNET dataset.
"""
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

from dlib.datasets.csv_tools import get_stats
from dlib.face_landmarks import FaceAlign
from dlib.face_landmarks.faceevolve_align import FaceEvolveAlign


from dlib.utils.reproducibility import set_seed
from dlib.datasets.default_labels_order import ORDERED_EMOTIONS

from dlib.utils.tools import Dict2Obj
from dlib.configure import constants
from dlib.utils.shared import announce_msg
from dlib.utils.tools import get_root_wsol_dataset
from dlib.utils.shared import find_files_pattern

from dlib.datasets.ds_shared import CROP_SIZE
from dlib.datasets.ds_shared import SEED
from dlib.datasets.ds_shared import ALIGN_DEEP_FACE
from dlib.datasets.ds_shared import ALIGN_FACE_EVOLVE
from dlib.datasets.ds_shared import class_balance_stat
from dlib.datasets.ds_shared import dump_set
from dlib.datasets.ds_shared import print_args_info
from dlib.datasets.ds_shared import per_cl_stats
from dlib.datasets.ds_shared import switch_key_val_dict

from dlib.datasets.default_labels_order import ORDERED_EMOTIONS

from dlib.utils.tools import check_box_convention


# Original assignment:
# 0: Neutral,
# 1: Happiness,
# 2: Sadness,
# 3: Surprise,
# 4: Fear,
# 5: Disgust,
# 6: Anger
origin_cl_int_to_txt = {
        0: constants.NEUTRAL,
        1: constants.HAPPINESS,
        2: constants.SADNESS,
        3: constants.SURPRISE,
        4: constants.FEAR,
        5: constants.DISGUST,
        6: constants.ANGER
    }


def split_AFFECTNET(args):
    """
    Create a validation/train sets in AFFECNET dataset.
    Test set is provided.
    """
    crop_align = args.crop_align
    align = args.align
    align_type = args.align_type
    use_original_align = args.use_original_align
    test_set_as_validset = args.test_set_as_validset
    already_cropped_aligned = args.already_cropped_aligned
    ds = args.dataset

    assert not (crop_align and use_original_align)
    if align:
        assert crop_align

    print_args_info(args)

    ordered_labels = ORDERED_EMOTIONS[ds]

    cl_int_to_txt = switch_key_val_dict(ordered_labels)

    if crop_align:
        final_id_tag = "our_aligned"
        bbx_upscale = args.bbx_upscale
        assert 0. <= bbx_upscale <= 1., bbx_upscale
        if align_type == ALIGN_DEEP_FACE:
            aligner = FaceAlign(out_size=CROP_SIZE, verbose=False)

        elif align_type == ALIGN_FACE_EVOLVE:
            aligner = FaceEvolveAlign(out_size=CROP_SIZE,
                                      verbose=False,
                                      no_warnings=True)

        else:
            raise NotImplementedError(align_type)

    elif already_cropped_aligned:
        final_id_tag = "our_aligned"

    else:
        assert use_original_align

    if use_original_align:
        final_id_tag = ""

    baseurl = args.baseurl
    sample_cl, sample_id = [], []
    sample_id_cl = dict()

    log = f"{40 * '*'} Log {40 * '*'} \nDataset: {ds}. \n"
    # images + cl.
    with open(join(baseurl, ds, "all_exp_annotation.txt"),
              "r") as fcl:
        content = fcl.readlines()
        for el in content:
            el = el.rstrip("\n\r")
            idcl, cl = el.split(",")
            cl = int(cl)

            sample_id.append(idcl)
            f = join(baseurl, ds, sample_id[-1])
            assert os.path.isfile(f), f
            sample_cl.append(cl)
            print(f"{f} >>> {cl}")
            assert idcl not in sample_id_cl, idcl
            sample_id_cl[idcl] = cl

    # --------------------------------------------------------------------------
    origin_cl_int_to_txt = {v: k for k, v in ordered_labels.items()}
    # --------------------------------------------------------------------------

    if crop_align and not already_cropped_aligned:

        outfd_align = join(baseurl, ds, final_id_tag)
        os.makedirs(outfd_align, exist_ok=True)

        print('Aligning faces...')
        list_failed_align = []

        for id_ in tqdm.tqdm(sample_id, total=len(sample_id), ncols=80):

            img_path = join(baseurl, ds, id_)
            crop_path = img_path  # image already cropped.

            if not crop_align:
                break

            # with open(f, 'r') as fin:
            #     line = fin.readline()
            #     # print(f, line)
            #     out = line.strip('\n').split(' ')
            #     x0, y0, x1, y1 = out[:4]
            #     x0 = int(max(float(x0), 0))
            #     y0 = int(max(float(y0), 0))
            #     x1 = int(max(float(x1), 0))
            #     y1 = int(max(float(y1), 0))
            #     bbox_xyxy = np.array([x0, y0, x1, y1])
            #     # x: width, y: height.
            #     check_box_convention(bbox_xyxy, 'x0y0x1y1')
            #
            #     widths = bbox_xyxy[2] - bbox_xyxy[0]
            #     heights = bbox_xyxy[3] - bbox_xyxy[1]
            #
            #     half_w = int(widths * bbx_upscale / 2.)
            #     half_h = int(heights * bbx_upscale / 2.)
            #
            #     b = basename(f).split('.')[0].replace('_boundingbox', '')
            #     b = f"{b}.{args.img_extension}"
            #     img_path = join(baseurl, ds, pre_id, b)
            #     left, upper, right, lower = y0, x0, y1, x1
            #
            #     assert os.path.isfile(img_path), img_path
            #     img = Image.open(img_path, 'r').convert('RGB')
            #     img_w, img_h = img.size
            #
            #     original_crop = img.crop((left, upper, right, lower))
            #
            #     # stretch bbox.
            #
            #     left = max(0, left - half_h)
            #     upper = max(0, upper - half_w)
            #     right = min(right + half_h, img_h - 1)
            #     lower = min(lower + half_w, img_w - 1)
            #
            #     crop = img.crop((left, upper, right, lower))
            #     crop = crop.resize((CROP_SIZE, CROP_SIZE),
            #                        resample=Image.Resampling.LANCZOS)
            #     crop_path = join(outfd_crop, b)
            #     crop.save(crop_path)

            bbox_xyxy = None
            img = Image.open(img_path, 'r').convert('RGB')
            original_crop = img.copy()
            crop = img.copy()
            crop = crop.resize((CROP_SIZE, CROP_SIZE),
                               resample=Image.Resampling.LANCZOS)

            # align face:
            if align:
                if align_type == ALIGN_DEEP_FACE:
                    aligned_face = aligner.align(crop_path)

                elif align_type == ALIGN_FACE_EVOLVE:
                    aligned_face = aligner.align(img_path,
                                                 proposed_bbx=bbox_xyxy
                                                 )
                    if (not aligner.success) and (bbox_xyxy is not None):
                        aligned_face = Image.fromarray(aligned_face)

                        aligned_face = aligned_face.resize(
                            (CROP_SIZE, CROP_SIZE),
                            resample=Image.Resampling.LANCZOS
                        )
                        aligned_face = np.array(aligned_face).astype(
                            np.uint8)

                    elif (not aligner.success) and (bbox_xyxy is None):
                        aligned_face = original_crop.resize(
                            (CROP_SIZE, CROP_SIZE),
                            resample=Image.Resampling.LANCZOS
                        )
                        aligned_face = np.array(aligned_face).astype(
                            np.uint8)

            else:
                aligned_face = np.array(crop)

            face = Image.fromarray(aligned_face)

            align_path = join(outfd_align, id_)
            os.makedirs(dirname(align_path), exist_ok=True)

            face.save(align_path)
            if not aligner.success and align:
                list_failed_align.append(crop_path)

        # info failure to align
        print(f"Nbr failed to face-align: {len(list_failed_align)}:")
        for s in list_failed_align:
            print(s)
        print(80 * "=")
        log += f"Nbr failed to face-align: {len(list_failed_align)} \n"

    # pair imd_id, img_cl.
    all_tr = dict()
    tst_set = dict()
    for s in sample_id_cl:
        if args.use_original_align:
            new_id = s
            pz = join(baseurl, ds, new_id)
            assert os.path.isfile(pz), pz
        else:
            new_id = f"{final_id_tag}/{s}"
            pz = join(baseurl, ds, new_id)
            assert os.path.isfile(pz), pz

        assert new_id not in all_tr, new_id
        assert new_id not in tst_set, new_id
        if s.startswith('train_set'):
            all_tr[new_id] = sample_id_cl[s]

        elif s.startswith('val_set'):
            tst_set[new_id] = sample_id_cl[s]

        else:
            raise ValueError(s)

    nbr_all_tr = len(list(all_tr.keys()))
    nbr_tst = len(list(tst_set.keys()))
    msg = f"full train set {nbr_all_tr}. tst set: {nbr_tst}. samples. \n"
    print(msg)
    log += msg

    set_seed(SEED)
    if test_set_as_validset:

        vl_set = tst_set.copy()
        tr_set = all_tr.copy()

        # set classes to start from 0 instead of 1.
        # Update: classes already start from 0. no need to correct
        for k in vl_set:
            vl_set[k] = vl_set[k]

        for k in tr_set:
            tr_set[k] = tr_set[k]

    else:

        # split valid and train sets.
        tr_set, vl_set = dict(), dict()
        for cl in origin_cl_int_to_txt:
            tmp = []
            for k in all_tr:
                if all_tr[k] == cl:
                    tmp.append(k)
            n = int(args.folding['vl'] * len(tmp) / 100.)
            assert n > 0, n
            for i in range(100):
                random.shuffle(tmp)

            # cl = cl - 1  # start from 0.
            # Update: classes already start from 0. no need to correct
            for i in range(len(tmp)):
                if i < n:
                    k = tmp[i]
                    vl_set[k] = cl
                else:
                    k = tmp[i]
                    tr_set[k] = cl

    nbr_tr = len(list(tr_set))
    nbr_vl = len(list(vl_set))

    msg = f"train set {nbr_tr}. valid set {nbr_vl}. tst set: {nbr_tst}."
    print(msg)
    log += f"{msg} \n"

    # start from 0 labels for test set
    # Update: classes already start from 0. no need to correct
    for k in tst_set:
        tst_set[k] = tst_set[k]

    # dump data
    list_train = [[k, tr_set[k]] for k in tr_set]
    list_vl = [[k, vl_set[k]] for k in vl_set]
    list_test = [[k, tst_set[k]] for k in tst_set]

    for i in range(1000):
        random.shuffle(list_train)

    relative_p = join(root_dir, constants.RELATIVE_META_ROOT, ds)
    os.makedirs(relative_p, exist_ok=True)

    classes_id = dict()
    for cl in origin_cl_int_to_txt:
        name_cl = origin_cl_int_to_txt[cl]
        classes_id[name_cl] = cl
        # Update: classes already start from 0. no need to correct

    with open(join(relative_p, "class_id.yaml"), 'w') as f:
        yaml.dump(classes_id, f)

    for split in constants.SPLITS:
        os.makedirs(join(relative_p, split), exist_ok=True)

        with open(join(relative_p, split, "class_id.yaml"), 'w') as f:
            yaml.dump(classes_id, f)

    dump_set(list_train, join(relative_p, constants.TRAINSET))
    dump_set(list_vl, join(relative_p, constants.VALIDSET))
    dump_set(list_test, join(relative_p, constants.TESTSET))

    tr_vl = list_train + list_vl

    # log
    log += f"Percentage valid from train: {args.folding['vl']} %. \n"

    # if args.crop_align:
    log += f"crop_align: {args.crop_align}. \n"
    log += f"align_type: {args.align_type} \n"
    log += f"align: {args.align}. \n"
    log += f"Upscale box (%): {args.bbx_upscale}. \n"

    # if args.use_original_align:
    log += f"use_original_align: {args.use_original_align}. \n"

    # if args.test_set_as_validset:
    log += f"test_set_as_validset: {args.test_set_as_validset}. \n"

    msg = class_balance_stat([k[1] for k in list_train],
                             constants.TRAINSET,
                             int_to_str=origin_cl_int_to_txt)
    log += msg

    msg = class_balance_stat([k[1] for k in list_vl],
                             constants.VALIDSET,
                             int_to_str=origin_cl_int_to_txt)
    log += msg

    msg = class_balance_stat([k[1] for k in list_test],
                             constants.TESTSET,
                             int_to_str=origin_cl_int_to_txt)
    log += msg

    msg = class_balance_stat([k[1] for k in tr_vl],
                             'Train + valid',
                             int_to_str=origin_cl_int_to_txt)
    log += msg

    if test_set_as_validset:
        all_samples = list_train + list_test

    else:
        all_samples = list_train + list_vl + list_test

    msg = class_balance_stat([k[1] for k in all_samples],
                             'All data',
                             int_to_str=origin_cl_int_to_txt)
    log += msg

    with open(join(relative_p, "log.txt"), 'w') as f:
        f.write(log)

    print(log)
    out_stat = per_cl_stats(l=[k[1] for k in tr_vl])
    with open(join(relative_p, "per_class_weight.yaml"), 'w') as f:
        yaml.dump(out_stat, f)

    print_args_info(args)
    print(f"Done splitting {ds} dataset.")


def separate_only_7_valid_emotions():

    ds = constants.AFFECTNET

    baseurl = get_root_wsol_dataset()
    ds_folder_original = join(baseurl, f"{ds}_original")
    assert ds_folder_original, ds_folder_original
    ds_folder_cleaned = join(baseurl, ds)
    if os.path.isdir(ds_folder_cleaned):
        os.system(f'rm -r {ds_folder_cleaned}')

    os.makedirs(ds_folder_cleaned, exist_ok=True)

    subsets = ['train_set', 'val_set']
    data = dict()
    stats = dict()

    l = []
    for s in subsets:

        lfiles = find_files_pattern(
            join(ds_folder_original, f'{s}/annotations'), '*.npy')

        stats[s] = {'original': len(lfiles), 'filtered': 0}

        # keep expressions only + neutral.

        for f in tqdm.tqdm(lfiles, total=len(lfiles), ncols=80):
            b = basename(f)
            if b.endswith('_exp.npy'):
                exp = int(np.load(f).reshape((1,))[0])
                # original order:
                # 0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise,
                # 4: Fear, 5: Disgust, 6: Anger

                # unconsidered:
                # 7: Contempt,
                # [8: None, 9: Uncertain, 10: No-Face] only large version.
                assert 0 <= exp <= 10, exp

                if exp > 6:  #  ignore after anger.
                    continue

                # convert to our class coding.
                cl_str = origin_cl_int_to_txt[exp]
                exp = ORDERED_EMOTIONS[ds][cl_str]

                name = b.split('_')[0]
                img_path = join(ds_folder_original,  f'{s}/images/{name}.jpg')
                assert os.path.isfile(img_path), img_path

                id_sample: str = img_path.replace(ds_folder_original, '')
                if id_sample.startswith(os.sep):
                    id_sample = id_sample[1:]

                assert id_sample not in data, id_sample
                data[id_sample] = {'img_path': img_path, 'label': exp}

                stats[s]['filtered'] += 1

                l.append(exp)

    # dump + copy
    fout = open(join(ds_folder_cleaned, 'all_exp_annotation.txt'), 'w')
    print('Copying clean data ... ')
    for k in tqdm.tqdm(data, ncols=80, total=len(list(data.keys()))):
        img_path = data[k]['img_path']

        label = data[k]['label']
        fout.write(f'{k},{label}\n')
        dest = join(ds_folder_cleaned, k)
        os.makedirs(dirname(dest), exist_ok=True)
        os.system(f"cp {img_path} {dest}")

    fout.close()

    print(stats)
    n = stats['train_set']['filtered'] + stats['val_set']['filtered']
    print(f"Total data copied: {n} samples.")
    print(f'Unique classes: {np.unique(np.array(l))}')
    # {'train_set': {'original': 1150604, 'filtered': 283901},
    # 'val_set': {'original': 15996, 'filtered': 3500}}
    # Total data copied: 287401 samples.
    # Unique classes: [0 1 2 3 4 5 6]


def split_crop_align_AFFECTNET(root_main: str,
                               crop_align: bool,
                               align: bool,
                               align_type: str,
                               use_original_align: bool,
                               test_set_as_validset: bool,
                               already_cropped_aligned: bool
                               ):
    """
    AFFETCNET.

    :param root_main: str. absolute path to folder containing main_old.py.
    :return:
    """
    # ===============
    # Reproducibility
    # ===============

    # ===========================

    set_seed(SEED)

    # ===========================
    ds = constants.AFFECTNET

    assert align_type in [ALIGN_DEEP_FACE, ALIGN_FACE_EVOLVE], align_type

    announce_msg("Processing dataset: {}".format(ds))

    args = {"baseurl": get_root_wsol_dataset(),
            "folding": {"vl": 1},  # 1% of trainset will be used for validset.
            "dataset": ds,
            "fold_folder": join(root_main, f"folds/{ds}"),
            "img_extension": "jpg",
            "nbr_splits": 1,  # how many times to perform the k-folds over
            # the available train samples.
            "path_encoding": join(root_main,
                                  f"folds/{ds}/encoding-origine.yaml"),
            "nbr_classes": None,  # Keep only 5 random classes. If you want
            # to use the entire dataset, set this to None.
            "bbx_upscale": .30,  # percentage to increase width and height of
            # original bbox (they usually crop relevant part of the face).
            # half of the percentage will be added to each direction.
            "crop_align": crop_align,  # do or not crop/align.
            "align": align,  # if true, images are aligned after beeing
            # cropped. goes with 'crop_align'.
            "align_type": align_type,  # str. type of face aligner.
            "already_cropped_aligned": already_cropped_aligned,  # if true,
            # we do not crop and align. we use existing files.
            "use_original_align": use_original_align,  # use or not the
            # original aligned images. if true, no crop align is used.
            "test_set_as_validset": test_set_as_validset  # if true,
            # the validset is testset.
            }
    args["nbr_folds"] = math.ceil(100. / args["folding"]["vl"])
    set_seed(SEED)
    split_AFFECTNET(Dict2Obj(args))

if __name__ == "__main__":
    root_dir = dirname(dirname(dirname(abspath(__file__))))
    sys.path.append(root_dir)

    ds = constants.AFFECTNET
    baseurl = get_root_wsol_dataset()
    ds_folder = join(baseurl, ds)
    if not os.path.isdir(ds_folder):
        separate_only_7_valid_emotions()

    # do not crop/align. use original. -----------------------------------------
    crop_align = False
    align = False  # controls crop_align
    already_cropped_aligned = False

    use_original_align = True
    # --------------------------------------------------------------------------

    align_type = ALIGN_FACE_EVOLVE
    test_set_as_validset = True

    if crop_align and not already_cropped_aligned:
        assert not use_original_align
        resp = input("you want to align faces[time consuming]? y/n?: ")
        if resp == 'y':
            print('Ok. will do.')
        else:
            print('Ok. Exiting. Fix it...')
            sys.exit()

    if align:
        assert crop_align


    if use_original_align:
        assert not crop_align

    if already_cropped_aligned:
        assert not crop_align
        assert not use_original_align

    split_crop_align_AFFECTNET(root_main=root_dir,
                               crop_align=crop_align,
                               align=align,
                               align_type=align_type,
                               use_original_align=use_original_align,
                               test_set_as_validset=test_set_as_validset,
                               already_cropped_aligned=already_cropped_aligned
                               )