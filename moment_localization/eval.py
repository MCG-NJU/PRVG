from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import pprint
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import datasets
import models
from core.config import config, update_config
from core.engine import Engine
from core.utils import AverageMeter
from core import eval
from core.utils import create_logger
import models.loss as loss
import math

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--dataDir', help='data path', type=str)
    parser.add_argument('--modelDir', help='model path', type=str)
    parser.add_argument('--logDir', help='log path', type=str)
    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    parser.add_argument('--tag', help='tags shown in log', type=str)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.dataDir:
        config.DATA_DIR = args.dataDir
    if args.modelDir:
        config.MODEL_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.verbose:
        config.VERBOSE = args.verbose
    if args.tag:
        config.TAG = args.tag


if __name__ == '__main__':

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(config, args.cfg, config.TAG)
    logger.info('\n' + pprint.pformat(args))
    logger.info('\n' + pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    dataset_name = config.DATASET.NAME
    model_name = config.MODEL.NAME

    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
    model.load_state_dict(model_checkpoint)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    def iterator(split):
        if split == 'train':
            dataloader = DataLoader(train_dataset,
                                    batch_size=config.TRAIN.BATCH_SIZE,
                                    shuffle=config.TRAIN.SHUFFLE,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'val':
            dataloader = DataLoader(val_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'test':
            dataloader = DataLoader(test_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        elif split == 'train_no_shuffle':
            dataloader = DataLoader(eval_train_dataset,
                                    batch_size=config.TEST.BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=config.WORKERS,
                                    pin_memory=False,
                                    collate_fn=datasets.dense_collate_fn)
        else:
            raise NotImplementedError

        return dataloader


    def network(sample):
        textual_input = sample['batch_word_vectors']
        visual_input = [b.cuda() for b in sample['batch_vis_input']]  # for new_tsn
        timestamps = sample['batch_timestamps']
        sample_sentences = sample['batch_sample_sentences']

        prediction = model(visual_input, textual_input, sample_sentences)

        boxes = prediction['pred_boxes']
        attn_output_weights = prediction['attn_output_weights']

        loss_value, loss_attn, loss_bbox, loss_giou = getattr(loss, config.LOSS.NAME)(boxes, timestamps, config.LOSS.PARAMS,
                                                                           attn_output_weights)
        return loss_value, boxes, loss_attn, loss_bbox, loss_giou



    def on_test_start(state):
        state['loss_meter'] = AverageMeter()
        state['batch_timestamps'] = []
        state['boxes'] = []
        if config.VERBOSE:
            if state['split'] == 'train':
                state['progress_bar'] = tqdm(total=math.ceil(len(train_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'val':
                state['progress_bar'] = tqdm(total=math.ceil(len(val_dataset) / config.TEST.BATCH_SIZE))
            elif state['split'] == 'test':
                state['progress_bar'] = tqdm(total=math.ceil(len(test_dataset) / config.TEST.BATCH_SIZE))
            else:
                raise NotImplementedError


    def on_test_forward(state):
        if config.VERBOSE:
            state['progress_bar'].update(1)
        state['loss_meter'].update(state['loss'].item(), 1)

        for b in state['sample']['batch_timestamps']:
            if len(state['batch_timestamps']) == 0:
                state['batch_timestamps'] = b
            else:
                state['batch_timestamps'] = np.vstack((state['batch_timestamps'], b))
        for b in state['output']:
            b = b.cpu().detach().numpy()
            if len(state['boxes']) == 0:
                state['boxes'] = b
            else:
                state['boxes'] = np.vstack((state['boxes'], b))


    def on_test_end(state):
        state['Rank@N,mIoU@M'], state['miou'] = eval.eval_predictions(state['boxes'], state['batch_timestamps'],
                                                                      verbose=False)
        if config.VERBOSE:
            state['progress_bar'].close()


    engine = Engine()
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    test_state = engine.test(network,
                iterator('test'),
                split='test')

    loss_message = ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
    test_state['loss_meter'].reset()
    test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                      'performance on testing set')
    table_message = '\n' + test_table

    message = loss_message + table_message + '\n'
    logger.info(message)

# export CUDA_VISIBLE_DEVICES=0 python eval.py --verbose --cfg ../experiments/activitynet/acnet_test.yaml
# export CUDA_VISIBLE_DEVICES=1 python eval.py --verbose --cfg ../experiments/tacos/tacos_test.yaml