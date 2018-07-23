# -*- coding: utf-8 -*-
"""
    Training configurations
"""

import numpy as np
import pickle
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

#PATH RELATED
__C.PATH_TO_SAVE_MODELS = '/DATA/tf_output/models'

# Training phase
__C.TRAIN = edict()

# ========================= Learning Configs ==========================
# Dropout ratio
__C.TRAIN.DROUPOUT_RATIO = 0.2
__C.TRAIN.LOSS_TYPE = 'nll'

# ========================= Learning Configs ==========================

__C.TRAIN.WEIGHT_DECAY = 5e-4

__C.TRAIN.CLIP_GRADIENT = 20

__C.TRAIN.MINIBATCH = 64

__C.TRAIN.MAX_STEPS = 12000

__C.TRAIN.NUM_GPUS = 1

# Initial learning rate
__C.TRAIN.INITIAL_LEARNING_RATE = 0.001

__C.TRAIN.NUM_EPOCHS_PER_DECAY = 30

__C.TRAIN.LEARNING_RATE_DECAY_FACTOR = 0.1

__C.TRAIN.SGD_MOMENTUM = 0.9

# number of segments to split one video
__C.TRAIN.SEGMENT_NUM = 3

__C.TRAIN.USE_DROPPING = True

__C.TRAIN.MODE = 'rgb'
__C.TRAIN.INPUT_CHS = 3

# label size
__C.TRAIN.LABEL_SIZE = {
    'kinetics': 400,
    'kinetics600': 600,
    'ucf': 101
}

# epoch size
__C.TRAIN.EPOCH_SIZE = {
    'kinetics': {
        'training': 235205,
        'validation': 19377
        },
    'kinetics600': {
        'training': 389987,
        'validation': 29779
        },
    'ucf': {
        'training': 9537,
        'validation': 3783
    }
}


__C.TEST = edict()
__C.TEST.MINIBATCH = 8
__C.TEST.SEGMENT_NUM = 25
__C.TEST.CROP_NUM = 10

# input image size
__C.IMAGE_WIDTH_PRE = 320
__C.IMAGE_HEIGHT_PRE = 256
__C.IMAGE_WIDTH = 224
__C.IMAGE_HEIGHT = 224

__C.IMAGE_SHORT_MIN = 256
__C.IMAGE_SHORT_MAX = 320

