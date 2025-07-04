# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/transformer.py
"""
Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        num_queries=256,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before, num_queries
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        num_queries=256
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_c = nn.MultiheadAttention(num_queries, nhead // 2, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)      

        # self.linear3 = nn.Linear(2*d_model, dim_feedforward)
        # self.linear4 = nn.Linear(dim_feedforward, d_model)
        # self.activation2 = _get_activation_fn(activation)

        # self.linear3 = nn.Linear(2*d_model, dim_feedforward)
        # self.dropout5 = nn.Dropout(dropout)
        # self.dropout6 = nn.Dropout(dropout)
        # self.linear4 = nn.Linear(dim_feedforward, d_model)
        # self.activation2 = _get_activation_fn(activation)


        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        # self.dropout5 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        
        # # self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt_s = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        # shortcut = tgt
        tgt = tgt + self.dropout1(tgt_s)
        tgt = self.norm4(tgt)
        


        # self-attention 2 channel 
        # 串联
        q_c = k_c = self.with_pos_embed(tgt, query_pos).permute(2, 1, 0)
        tgt3 = self.self_attn_c(q_c, k_c, value=tgt.permute(2, 1, 0), attn_mask=None, key_padding_mask=None)[0].permute(2, 1, 0)    

        # 并联
        # q_c = k_c = q.permute(2, 1, 0)
        # tgt3 = self.self_attn_c(q_c, k_c, value=shortcut.permute(2, 1, 0), attn_mask=None, key_padding_mask=None)[0].permute(2, 1, 0)    

        # 这里 shortcut 的输入应该是上一段的输出，即tgt2
        # tgt3 = shortcut + self.dropout4(tgt3)
        # tgt = self.norm4(tgt3)   
        # tgt3 = self.norm4(tgt3)   
         
        tgt2 = tgt + self.dropout4(tgt3)
        tgt = self.norm1(tgt2)

        # tgt = torch.cat([tgt2, tgt3], dim=-1)
        # tgt = self.linear4(self.dropout5(self.activation2(self.linear3(tgt3))))
        # 这样做是为了可以复用mask_former的代码
        # tgt = tgt2 + tgt3

        # # cross-attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]

        tgt2 = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt2)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # 添加一路额外的self-attention 和 MLP
        # tgt3 = self.linear4(self.dropout5(self.activation2(self.linear3(tgt3))))
        # tgt = self.linear4(self.activation2(self.linear3(tgt))) + self.dropout3(tgt2)
        
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


if __name__ == "__main__":
    import math

    class PositionEmbeddingSine(nn.Module):
        def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
            super().__init__()
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale

        def forward(self, x, mask=None):
            if mask is None:
                mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            return pos
    
    hidden_dim = 256
    dropout = 0
    nheads = 8
    dim_feedforward = 2048
    enc_layers = 0
    dec_layers = 8
    pre_norm = False
    deep_supervision = False
    Query_num = 200
    batch_size = 2
    h = 8
    w = 8
    src = torch.rand(batch_size, hidden_dim, h, w)

    transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )
    pos_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    pos = pos_emb(src)
    query_embed = nn.Embedding(Query_num, hidden_dim)
    trans = transformer(src, None, query_embed.weight, pos)