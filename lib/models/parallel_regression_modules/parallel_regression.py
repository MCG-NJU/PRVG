import copy
from typing import Optional
from core.config import config

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ParallelRegression(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_cre_layers: int = 1,
                 num_lmd_layers: int = 1, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5):
        super(ParallelRegression, self).__init__()

        text_cre_layer = CRELayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps)
        visual_cre_layer = CRELayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps)
        text_cre_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        visual_cre_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.text_cre = CRE(text_cre_layer, num_cre_layers, text_cre_norm)
        self.visual_cre = CRE(visual_cre_layer, num_cre_layers, visual_cre_norm)

        lmd_layer = LMDLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps)
        lmd_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.lmd = LMD(lmd_layer, num_lmd_layers, lmd_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        memory = self.visual_cre(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        tgt = self.text_cre(tgt, mask=tgt_mask, src_key_padding_mask=tgt_key_padding_mask)
        output, attn_output_weights = self.lmd(tgt, memory, memory_mask=memory_mask,
                                                   memory_key_padding_mask=memory_key_padding_mask)
        return output, attn_output_weights

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class CRE(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(CRE, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class LMD(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(LMD, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        output = tgt

        for mod in self.layers:
            output, attn_output_weights = mod(output, memory,
                                              memory_mask=memory_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output_weights


class CRELayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        super(CRELayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CRELayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class LMDLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        super(LMDLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(LMDLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2, attn_output_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_output_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def build_parallel_regression():
    return ParallelRegression(
        d_model=config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.HIDDEN_DIM,
        dropout=config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.DROPOUT,
        nhead=config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.NHEADS,
        dim_feedforward=config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.DIM_FEEDFORWAD,
        num_cre_layers=config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.CRE_LAYERS,
        num_lmd_layers=config.PRVG.PARALLEL_REGRESSION_MODULE.PARAMS.LMD_LAYERS
    )


class simpleMLP(nn.Module):
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