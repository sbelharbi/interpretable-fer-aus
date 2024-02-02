# divers classifiers.
import sys
from os.path import dirname, abspath

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)


from dlib.div_classifiers.deit import deit_tscam_base_patch16_224
from dlib.div_classifiers.deit import deit_tscam_tiny_patch16_224
from dlib.div_classifiers.deit import deit_tscam_small_patch16_224


from dlib.configure import constants

models = dict()

models[constants.METHOD_TSCAM] = dict()
models[constants.METHOD_TSCAM][
    constants.DEIT_TSCAM_TINY_P16_224] = deit_tscam_tiny_patch16_224
models[constants.METHOD_TSCAM][
    constants.DEIT_TSCAM_SMALL_P16_224] = deit_tscam_small_patch16_224
models[constants.METHOD_TSCAM][
    constants.DEIT_TSCAM_BASE_P16_224] = deit_tscam_base_patch16_224

