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
from . import new_tsn_sample, new_tsn_sample_test, tsn_sample


class TACoS(data.Dataset):
    vocab = torchtext.vocab.pretrained_aliases["glove.6B.300d"](cache='../glove')
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, split):
        super(TACoS, self).__init__()

        self.vis_input_type = config.DATASET.VIS_INPUT_TYPE
        self.data_dir = config.DATA_DIR
        self.split = split

        self.annotations = np.load(os.path.join('../data/TACoS/', '{}.npy'.format(split)), allow_pickle=True)

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
        else:  # val和test模式不采样
            sentence_sample = self.annotations[index]['sentences']
            timestamps_sample = self.annotations[index]['timestamps']
        word_vectors_list = []
        for sentence in sentence_sample:
            word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 400000) for w in sentence.split()],
                                     dtype=torch.long)
            word_vectors = self.word_embedding(word_idxs)  # word_vectors (seq, 300)
            word_vectors_list.append(word_vectors)

        visual_input = self.get_video_features(video_id)

        if self.split == 'train':
            visual_input = new_tsn_sample(visual_input, 512)
        else:
            visual_input = new_tsn_sample_test(visual_input, 512)

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
        with h5py.File(os.path.join(self.data_dir, 'tall_c3d_features.hdf5'), 'r') as f:
            if vid[-1] == ')':
                vid = vid.split('(')[0]
            features = torch.from_numpy(f[vid][:])
        if config.DATASET.NORMALIZE:
            features = F.normalize(features, dim=1)
        return features