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

    train_dataset = getattr(datasets, dataset_name)('train')
    if config.TEST.EVAL_TRAIN:
        eval_train_dataset = getattr(datasets, dataset_name)('train')
    if not config.DATASET.NO_VAL:
        val_dataset = getattr(datasets, dataset_name)('val')
    test_dataset = getattr(datasets, dataset_name)('test')

    model = getattr(models, model_name)()
    if config.MODEL.CHECKPOINT and config.TRAIN.CONTINUE:
        model_checkpoint = torch.load(config.MODEL.CHECKPOINT)
        model.load_state_dict(model_checkpoint)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999),
                           weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.AdamW(model.parameters(), lr=config.TRAIN.LR, betas=(0.9, 0.999),
    #                        weight_decay=config.TRAIN.WEIGHT_DECAY)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, momentum=0.9, weight_decay=config.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.TRAIN.FACTOR,
                                                     patience=config.TRAIN.PATIENCE, verbose=config.VERBOSE)


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


    def on_start(state):
        state['loss_meter'] = AverageMeter()
        state['test_interval'] = int(len(train_dataset) / config.TRAIN.BATCH_SIZE * config.TEST.INTERVAL)
        state['t'] = 1
        model.train()
        if config.VERBOSE:
            state['progress_bar'] = tqdm(total=state['test_interval'])


    def on_forward(state):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        state['loss_meter'].update(state['loss'].item(), 1)


    def on_update(state):  # Save All
        if config.VERBOSE:
            state['progress_bar'].update(1)

        if state['t'] % state['test_interval'] == 0:
        # if state['t'] / state['test_interval'] >= 20 and state['t'] % state['test_interval'] == 0: # for tacos
            model.eval()
            if config.VERBOSE:
                state['progress_bar'].close()

            loss_message = '\niter: {} train loss {:.4f}'.format(state['t'], state['loss_meter'].avg)
            table_message = ''
            if config.TEST.EVAL_TRAIN:
                train_state = engine.test(network, iterator('train_no_shuffle'), 'train')
                train_table = eval.display_results(train_state['Rank@N,mIoU@M'], train_state['miou'],
                                                   'performance on training set')
                table_message += '\n' + train_table
            if not config.DATASET.NO_VAL:
                val_state = engine.test(network, iterator('val'), 'val')
                state['scheduler'].step(-val_state['loss_meter'].avg)       # 根据val的结果调整学习率
                loss_message += ' val loss {:.4f}'.format(val_state['loss_meter'].avg)
                val_state['loss_meter'].reset()
                val_table = eval.display_results(val_state['Rank@N,mIoU@M'], val_state['miou'],
                                                 'performance on validation set')
                table_message += '\n' + val_table

            test_state = engine.test(network, iterator('test'), 'test')
            loss_message += ' test loss {:.4f}'.format(test_state['loss_meter'].avg)
            test_state['loss_meter'].reset()
            test_table = eval.display_results(test_state['Rank@N,mIoU@M'], test_state['miou'],
                                              'performance on testing set')
            table_message += '\n' + test_table

            message = loss_message + table_message + '\n'
            logger.info(message)

            cfg_name = args.cfg
            cfg_name = os.path.basename(cfg_name).split('.yaml')[0]
            saved_model_filename = os.path.join(config.MODEL_DIR, config.DATASET.NAME,
                                                'tsn_pl', cfg_name)
            if not os.path.exists(saved_model_filename):
                print('Make directory %s ...' % saved_model_filename)
                os.makedirs(saved_model_filename)
            saved_model_filename = os.path.join(
                saved_model_filename,
                'iter{:06d}.pkl'.format(state['t'])
            )

            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), saved_model_filename)
            else:
                torch.save(model.state_dict(), saved_model_filename)

            if config.VERBOSE:
                state['progress_bar'] = tqdm(total=state['test_interval'])
            model.train()
            state['loss_meter'].reset()


    def on_end(state):
        if config.VERBOSE:
            state['progress_bar'].close()


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
    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_update'] = on_update
    engine.hooks['on_end'] = on_end
    engine.hooks['on_test_start'] = on_test_start
    engine.hooks['on_test_forward'] = on_test_forward
    engine.hooks['on_test_end'] = on_test_end
    engine.train(network,
                 iterator('train'),
                 maxepoch=config.TRAIN.MAX_EPOCH,
                 optimizer=optimizer,
                 scheduler=scheduler)


# export CUDA_VISIBLE_DEVICES=0 python main.py --verbose --cfg ../experiments/activitynet/acnet.yaml
# export CUDA_VISIBLE_DEVICES=1 python main.py --verbose --cfg ../experiments/tacos/tacos.yaml