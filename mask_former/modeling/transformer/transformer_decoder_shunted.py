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

import math
from torch.nn.init import trunc_normal_

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, sr_ratio=1, Query_num=200):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)

        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act1_1 = nn.ReLU6()
            self.act1_2 = nn.ReLU6()
            self.act2_1 = nn.ReLU6()
            self.act2_2 = nn.ReLU6()
            self.softmax2 = nn.Softmax(dim=-1)
            if sr_ratio==8:
                self.sr1_1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.sr1_2 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1_1 = nn.LayerNorm(dim)
                self.norm1_2 = nn.LayerNorm(dim)
                self.sr2_1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.sr2_2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2_1 = nn.LayerNorm(dim)
                self.norm2_2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1_1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.sr1_2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1_1 = nn.LayerNorm(dim)
                self.norm1_2 = nn.LayerNorm(dim)
                self.sr2_1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.sr2_2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2_1 = nn.LayerNorm(dim)
                self.norm2_2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1_1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.sr1_2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1_1 = nn.LayerNorm(dim)
                self.norm1_2 = nn.LayerNorm(dim)
                self.sr2_1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.sr2_2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2_1 = nn.LayerNorm(dim)
                self.norm2_2 = nn.LayerNorm(dim)
            self.k1 = nn.Linear(dim, dim//2, bias=qkv_bias)
            self.v1 = nn.Linear(dim, dim//2, bias=qkv_bias)
            self.k2 = nn.Linear(dim, dim //2, bias=qkv_bias)
            self.v2 = nn.Linear(dim, dim//2, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def forward(self, x, H, W):
    def forward(self, query,key,value, H, W, attn_mask=None, key_padding_mask=None):
        Q, B, C = query.shape # Q B C
        # N, _, _ = key.shape
        q = self.q(query).reshape(B, Q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() # B head Q C
        if self.sr_ratio > 1:
            key_ = key.permute(0, 2, 1).reshape(B, C, H, W) # HW, B, C
            value_ = value.permute(0, 2, 1).reshape(B, C, H, W)
            key_1 = self.act1_1(self.norm1_1(self.sr1_1(key_).reshape(B, C, -1).permute(0, 2, 1).contiguous()))
            value_1 = self.act1_2(self.norm1_2(self.sr1_2(value_).reshape(B, C, -1).permute(0, 2, 1).contiguous()))
            
            key_2 = self.act2_1(self.norm2_1(self.sr2_1(key_).reshape(B, C, -1).permute(0, 2, 1).contiguous()))
            value_2 = self.act2_2(self.norm2_2(self.sr2_2(value_).reshape(B, C, -1).permute(0, 2, 1).contiguous()))
            
            k1 = self.k1(key_1.contiguous()).reshape(B, -1, 1, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()[0]
            v1 = self.v1(value_1.contiguous()).reshape(B, -1, 1, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()[0]
            k2 = self.k2(key_2.contiguous()).reshape(B, -1, 1, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()[0]
            v2 = self.v2(value_2.contiguous()).reshape(B, -1, 1, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()[0] #B head N C
            
            attn1 = (q[:, :self.num_heads//2].contiguous() @ k1.transpose(-2, -1).contiguous()) * self.scale
            # attn1 = attn1.softmax(dim=-1)
            attn1 = self.softmax(attn1)
            # attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).contiguous().reshape(B, -1, C//2).transpose(1, 2).contiguous().view(B,C//2, H//self.sr_ratio, W//self.sr_ratio).contiguous()).\
                view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous()
            x1 = (attn1 @ v1).transpose(1, 2).contiguous().reshape(Q, B, C//2)
            attn2 = (q[:, self.num_heads // 2:].contiguous() @ k2.transpose(-2, -1).contiguous()) * self.scale
            attn2 = self.softmax2(attn2)
            # attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                    transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio).contiguous()).\
                view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous()
            x2 = (attn2 @ v2).transpose(1, 2).contiguous().reshape(Q, B, C//2)

            x = torch.cat([x1,x2], dim=-1).contiguous()
        else:
            k = self.k(key).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(value).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous() #B head N C
#             print("the size of v is {}".format(v.size()))

            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
            attn = self.softmax(attn)
            # attn = self.attn_drop(attn)  # B head Q N   B head N C

            v1 = v + self.local_conv(value.permute(0, 2, 1).contiguous().reshape(B, C, H, W)).reshape(B, -1 , self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            x = (attn @ v1).transpose(1, 2).contiguous().reshape(Q, B, C) 
        x = self.proj(x.contiguous())
        # x = self.proj_drop(x)

        return x


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
        num_queries=200,
    ):
        super().__init__()
        if num_decoder_layers == 8:
            nheads = [2, 4, 4, 8, 8, 8, 8, 16]
            sr_ratios = [8, 4, 4, 2, 2, 2, 2, 1]
        elif num_decoder_layers == 6:
            if nhead == 8:
                # sr_ratios = 2
                nheads = [2, 4, 4, 8, 8, 16]
                sr_ratios = [8, 4, 4, 2, 2, 1]
            # else:
            #     nheads = [2, 4, 8, 8, 8, 16]
            #     sr_ratios = [8, 4, 2, 2, 2, 1]            
            

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.ModuleList([TransformerDecoderLayer(
            d_model, nheads[i], dim_feedforward, dropout, activation, normalize_before, sr_ratios[i], num_queries
        ) for i in range(num_decoder_layers)])        
        # decoder_layer = TransformerDecoderLayer(
        #     d_model, nhead, dim_feedforward, dropout, activation, normalize_before, sr_ratios, num_queries
        # )
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
            tgt, memory, h, w, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed
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
        self.layers = decoder_layer
        # self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        height,
        width,
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
                height,
                width,
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
        sr_ratio=1,
        num_queries=200
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = Attention(d_model, nhead, sr_ratio=sr_ratio, Query_num=num_queries)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        height,
        width,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            H=height,
            W=width,
        )
        # tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
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
        height,
        width,
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
            height,
            width,
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


