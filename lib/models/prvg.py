import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import math

from core.config import config
from .parallel_regression_modules.parallel_regression import build_parallel_regression

class ClipAvgPool(nn.Module):

    def __init__(self):
        super(ClipAvgPool, self).__init__()
        input_size = 4096
        hidden_size = 512
        kernel_size = 4
        stride = 4
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        return vis_h


class PRVG(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_queries = config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.NUM_QUERIES
        self.parallel_regression = build_parallel_regression()
        hidden_dim = self.parallel_regression.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        txt_input_size = config.PRVG.LSTM.PARAMS.TXT_INPUT_SIZE
        txt_hidden_size = config.PRVG.LSTM.PARAMS.TXT_HIDDEN_SIZE
        self.lstm = nn.LSTM(txt_input_size, txt_hidden_size // 2 if config.PRVG.LSTM.PARAMS.BIDIRECTIONAL else txt_hidden_size,
                            num_layers=config.PRVG.LSTM.PARAMS.NUM_LAYERS, bidirectional=config.PRVG.LSTM.PARAMS.BIDIRECTIONAL,
                            batch_first=True)
        self.visual_linear = nn.Linear(config.PRVG.INPUT_SIZE, config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.HIDDEN_DIM)  # adjust the dim of visual input

    def forward(self, visual_inputs, textual_inputs, sample_sentences):
        outputs_coord = []
        ouputs_attn_weights = []

        visual_inputs = [self.visual_linear(visual_inputs[i]) for i in range(len(visual_inputs))]

        for i in range(len(textual_inputs)):
            textual_input = pack_padded_sequence(textual_inputs[i].to(visual_inputs[0].device), sample_sentences[i], batch_first=True, enforce_sorted=False)
            textual_input, _ = self.lstm(textual_input)
            textual_input, _ = pad_packed_sequence(textual_input, batch_first=True)
            sample_sentences_idx = torch.tensor(sample_sentences[i]) - 1
            encoded_textual_input = torch.zeros((textual_input.shape[0], textual_input.shape[2])).to(visual_inputs[0].device)
            for j in range(len(sample_sentences_idx)):
                encoded_textual_input[j] = textual_input[j, sample_sentences_idx[j], :]

            textual_input = positional_encoding(encoded_textual_input).unsqueeze(0)
            visual_input = positional_encoding(visual_inputs[i]).unsqueeze(0)

            hs, attn_output_weights = self.parallel_regression(visual_input.permute(1, 0, 2), textual_input.permute(1, 0, 2))
            hs = hs.permute(1, 0, 2)

            attn_output_weights = attn_output_weights.squeeze(0)
            output_coord = self.bbox_embed(hs).squeeze(0).sigmoid()  # [s, e]

            outputs_coord.append(output_coord)
            ouputs_attn_weights.append(attn_output_weights)

        out = {'pred_boxes': outputs_coord, 'attn_output_weights': ouputs_attn_weights}

        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def positional_encoding(input):
    seq_len = input.shape[0]
    emb_size = input.shape[1]
    den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
    pos = torch.arange(0, seq_len).reshape(seq_len, 1)
    pos_embedding = torch.zeros((seq_len, emb_size))
    pos_embedding[:, 0::2] = torch.sin(pos * den)
    pos_embedding[:, 1::2] = torch.cos(pos * den)
    return input + pos_embedding.to(input.device)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.pos_embedding = pos_embedding

    def forward(self, token_embedding: Tensor):
        return token_embedding + self.pos_embedding[:token_embedding.size(0), :]

