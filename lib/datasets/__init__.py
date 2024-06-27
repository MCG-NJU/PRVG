import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from core.config import config


def tsn_sample(visual_input):
    num_sample_clips = config.DATASET.NUM_SAMPLE_CLIPS
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(),torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            sample_idx = torch.randint(s_idx, e_idx + 1, (1,))
            new_visual_input.append(torch.mean(visual_input[sample_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


def new_tsn_sample(visual_input, num_sample_clips):
    num_clips = visual_input.shape[0]
    while num_sample_clips > num_clips:
        num_sample_clips /= 2
    num_sample_clips = int(num_sample_clips)
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            sample_idx = torch.randint(s_idx, e_idx + 1, (1,))
            new_visual_input.append(torch.mean(visual_input[sample_idx],dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input

def new_tsn_sample_test(visual_input, num_sample_clips):
    num_clips = visual_input.shape[0]
    while num_sample_clips > num_clips:
        num_sample_clips /= 2
    num_sample_clips = int(num_sample_clips)
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


def dense_collate_fn(batch):
    bs = len(batch)
    visual_clips, visul_emd = batch[0]['visual_input'].shape
    batch_anno_idxs = [b['anno_idx'] for b in batch]

    # turn a list of tensor to tensor
    # batch_vis_feats = torch.cat([b['visual_input'] for b in batch]).reshape(bs, visual_clips, visul_emd).float()  # for tsn
    batch_vis_feats = [b['visual_input'].float() for b in batch]  # for new_tsn
    batch_duration = [b['duration'] for b in batch]
    batch_timestamps = [b['timestamps'] for b in batch]
    batch_word_vectors = []
    batch_sample_sentences = []

    for b in batch:
        seq_lens = []
        word_vectors_list = b['word_vectors']
        for word_vectors in word_vectors_list:
            seq_lens.append(len(word_vectors))
        word_vectors_list = pad_sequence(word_vectors_list, batch_first=True)
        batch_word_vectors.append(word_vectors_list)
        batch_sample_sentences.append(seq_lens)

    batch_data = {
        'batch_anno_idxs': batch_anno_idxs,
        'batch_duration': batch_duration,
        'batch_timestamps': batch_timestamps,
        'batch_vis_input': batch_vis_feats,
        'batch_word_vectors': batch_word_vectors,
        'batch_sample_sentences': batch_sample_sentences
    }
    return batch_data




from datasets.activitynet import ActivityNet
from datasets.tacos import TACoS
