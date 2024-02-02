import os
import sys
from os.path import dirname, abspath, join


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.face_landmarks.face_align import FaceAlign
