from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
from easydict import EasyDict as edict

config = edict()

config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''
config.VERBOSE = False
config.TAG = ''

# CUDNN related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# PRVG related params
config.PRVG = edict()
config.PRVG.PARALLEL_REGRESSION_MODULE = edict()
config.PRVG.PARALLEL_REGRESSION_MODULE.NAME = ''
config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS = None

config.PRVG.LSTM = edict()
config.PRVG.LSTM.NAME = ''
config.PRVG.LSTM.PARAMS = None

config.PRVG.INPUT_SIZE = 500

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.NAME = ''
config.DATASET.MODALITY = ''
config.DATASET.VIS_INPUT_TYPE = ''
config.DATASET.NO_VAL = False
config.DATASET.BIAS = 0
config.DATASET.NUM_SAMPLE_CLIPS = 256
config.DATASET.SPLIT = ''
config.DATASET.NORMALIZE = False
config.DATASET.RANDOM_SAMPLING = False

# train
config.TRAIN = edict()
config.TRAIN.LR = 0.001
config.TRAIN.WEIGHT_DECAY = 0
config.TRAIN.FACTOR = 0.8
config.TRAIN.PATIENCE = 20
config.TRAIN.MAX_EPOCH = 20
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.SHUFFLE = True
config.TRAIN.CONTINUE = False

config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10

def _update_dict(cfg, value):
    for k, v in value.items():
        if k in cfg:
            if k == 'PARAMS':
                cfg[k] = v
            elif isinstance(v, dict):
                _update_dict(cfg[k],v)
            else:
                cfg[k] = v
        else:
            raise ValueError("{} not exist in config.py".format(k))

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(config[k], v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
