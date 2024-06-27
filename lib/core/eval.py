import json
import argparse
import numpy as np
import torch
from terminaltables import AsciiTable

from core.config import config, update_config


def iou(pred, gt):  # require pred and gt is numpy
    inter_left = np.maximum(pred[:, 0], gt[:, 0])
    inter_right = np.minimum(pred[:, 1], gt[:, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0], gt[:, 0])
    union_right = np.maximum(pred[:, 1], gt[:, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    return overlap


def rank(pred, gt):
    return pred.index(gt) + 1


def eval(pred, gt):
    '''
    RECALL: 1
    TIOU: 0.3,0.5,0.7
    '''

    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    eval_result = np.zeros(len(tious))

    overlap = iou(pred, gt)
    num = len(pred)

    for i, t in enumerate(tious):
        eval_result[i] = np.count_nonzero(overlap > t) * 1.0 / num
    miou = np.mean(overlap)
    return eval_result.reshape(-1, 1), miou


def eval_predictions(segments, data, verbose=True):
    eval_result, miou = eval(segments, data)
    if verbose:
        print(display_results(eval_result, miou, ''))

    return eval_result, miou


def display_results(eval_result, miou, title=None):
    tious = [float(i) for i in config.TEST.TIOU.split(',')] if isinstance(config.TEST.TIOU, str) else [config.TEST.TIOU]
    recalls = [int(i) for i in config.TEST.RECALL.split(',')] if isinstance(config.TEST.RECALL, str) else [
        config.TEST.RECALL]

    display_data = [['Rank@{},mIoU@{}'.format(i, j) for i in recalls for j in tious] + ['mIoU']]
    eval_result = eval_result * 100
    miou = miou * 100
    display_data.append(['{:.02f}'.format(eval_result[j][i]) for i in range(len(recalls)) for j in range(len(tious))]
                        + ['{:.02f}'.format(miou)])
    table = AsciiTable(display_data, title)
    for i in range(len(tious) * len(recalls)):
        table.justify_columns[i] = 'center'
    return table.table


def parse_args():
    parser = argparse.ArgumentParser(description='Train localization network')

    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    parser.add_argument('--verbose', default=False, action="store_true", help='print progress bar')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.verbose:
        config.VERBOSE = args.verbose
