import os
import json
import h5py

import torch
import torch.utils.data as data
import torchtext
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F

import numpy as np
import random

from core.config import config
from core.eval import iou
from . import new_tsn_sample, tsn_sample


class ActivityNet(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache='../glove')
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(ActivityNet, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        # val_1.json is renamed as val.json, val_2.json is renamed as test.json

        with open(os.path.join('../data/ActivityNet/', '{}.json'.format(split)), 'r') as f:
            annotations = json.load(f)
        anno_pairs = []
        for vid, video_anno in annotations.items():
            duration = video_anno['duration']
            flag = True
            for timestamp in video_anno['timestamps']:
                if timestamp[0] >= timestamp[1]:
                    flag = False
            if not flag:
                continue
            anno_pairs.append(
                {
                    'video': vid,
                    'duration': duration,
                    'sentences': video_anno['sentences'],
                    'timestamps': video_anno['timestamps']
                }
            )
        self.annotations = anno_pairs

    def __getitem__(self, index):
        video_id = self.annotations[index]['video']
        duration = self.annotations[index]['duration']
        tot_sentence = len(self.annotations[index]['sentences'])

        P = min(9, tot_sentence + 1)
        num_sentence = np.random.randint(1, P)
        if num_sentence > tot_sentence:
            num_sentence = tot_sentence
        idx_sample = random.sample(range(tot_sentence), num_sentence)
        idx_sample.sort()
        if self.split == 'train':
            sentence_sample = [self.annotations[index]['sentences'][idx] for idx in idx_sample]
            timestamps_sample = [self.annotations[index]['timestamps'][idx] for idx in idx_sample]
        else:
            sentence_sample = self.annotations[index]['sentences']
            timestamps_sample = self.annotations[index]['timestamps']
        word_vectors_list = []
        for sentence in sentence_sample:
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()],
                                     dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs)  # word_vectors (seq, 300)
            word_vectors_list.append(word_vectors)

        visual_input = self.get_video_features(video_id)

        visual_input = new_tsn_sample(visual_input, 256)

        for i in range(len(timestamps_sample)):
            timestamps_sample[i][0] /= duration
            timestamps_sample[i][1] /= duration
        timestamps_sample = np.array(timestamps_sample)

        item = {
            'visual_input': visual_input,
            'anno_idx': index,
            'timestamps': timestamps_sample,
            'word_vectors': word_vectors_list,
            'duration': duration,
        }
        return item

    def __len__(self):
        return len(self.annotations)

    def get_video_features(self, vid):
        assert config.DATASET.VIS_INPUT_TYPE == 'c3d'
        with h5py.File(os.path.join(self.data_dir, 'sub_activitynet_v1-3.c3d.hdf5'), 'r') as f:
            if vid[-1] == ')':
                vid = vid[:-3]
            features = torch.from_numpy(f[vid]['c3d_features'][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        return features