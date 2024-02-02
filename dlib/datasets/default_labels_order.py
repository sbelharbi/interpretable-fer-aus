"""
Default order of emotions
"""
import sys
from os.path import join, dirname, abspath

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants

ORDERED_EMOTIONS = {

    constants.RAFDB: {
        constants.NEUTRAL: 0,
        constants.HAPPINESS: 1,
        constants.SURPRISE: 2,
        constants.FEAR: 3,
        constants.ANGER: 4,
        constants.DISGUST: 5,
        constants.SADNESS: 6
    },
    constants.AFFECTNET: {
        constants.NEUTRAL: 0,
        constants.HAPPINESS: 1,
        constants.SURPRISE: 2,
        constants.FEAR: 3,
        constants.ANGER: 4,
        constants.DISGUST: 5,
        constants.SADNESS: 6
    }
}