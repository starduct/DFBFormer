# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.SSA import Block

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer


def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


# class DWConv(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super().__init__()
#         # 也相当于分组为1的分组卷积
#         self.depth_conv = nn.Conv2d(in_channels=in_ch,
#                                     out_channels=in_ch,
#                                     kernel_size=3,
#                                     stride=1,
#                                     padding=1,
#                                     groups=in_ch)
#         self.point_conv = nn.Conv2d(in_channels=in_ch,
#                                     out_channels=out_ch,
#                                     kernel_size=1,
#                                     stride=1,
#                                     padding=0,
#                                     groups=1)
#     def forward(self,input):
#         out = self.depth_conv(input)
#         out = self.point_conv(out)
#         return out


class OutputConv(nn.Module):
    def __init__(
        self,
        inp_dim,
        oup_dim,
        kernel_size,
        stride,
        padding,
        bias,
        norm,
        activation,
    ):
        super().__init__()
        self.inp_dim = inp_dim
        # self.hidden_dim = oup_dim * 2
        self.oup_dim = oup_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.norm = norm
        # self.hidden_norm = None
        self.activation = activation
        # self.conv1 = nn.conv2d(
        #             self.inp_dim,
        #             self.inp_dim,
        #             kernel_size = 1,
        #             stride = 1,
        #             padding = 0,
        #             groups = self.inp_dim,
        #         )
        self.conv2 = nn.Conv2d(self.inp_dim, 
                               self.inp_dim, 
                               self.kernel_size, 
                               stride =self.stride, 
                               padding=self.padding, 
                               groups=self.inp_dim) 
        self.conv111_0 = nn.Conv2d(self.inp_dim, self.inp_dim, (1,11), padding=(0, 5), groups=self.inp_dim)
        self.conv111_1 = nn.Conv2d(self.inp_dim, self.inp_dim, (11,1), padding=(5, 0), groups=self.inp_dim)
        self.conv17_0 = nn.Conv2d(self.inp_dim, self.inp_dim, (1,7), padding=(0, 3), groups=self.inp_dim)
        self.conv17_1 = nn.Conv2d(self.inp_dim, self.inp_dim, (7,1), padding=(3, 0), groups=self.inp_dim)
        self.bn = nn.BatchNorm2d(self.inp_dim, momentum=0.1)

        self.conv3 = Conv2d(
            self.inp_dim,
            self.oup_dim,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = self.bias,
            norm = self.norm,
            activation = self.activation,
        )

    def forward(self, x):
        # y0 = self.conv1(x)
        y = self.conv2(x)
        y1 = self.conv17_0(x)
        y2 = self.conv17_1(x)
        y3 = self.conv111_0(x)
        y4 = self.conv111_1(x)
        out = y + y1 + y2 + y3 + y4 + x
        out = self.bn(out)
        return self.conv3(out)


@SEM_SEG_HEADS_REGISTRY.register()
class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这个是最后的一个层，应该是res5使用的代码，这里需要加强
                output_norm = get_norm(norm, conv_dim)
                output_conv = OutputConv(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = Conv2d(
                #     in_channels,
                #     conv_dim,
                #     kernel_size=3,
                #     stride=1,
                #     padding=1,
                #     bias=use_bias,
                #     norm=output_norm,
                #     activation=F.relu,
                # )
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = OutputConv(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = Conv2d(
                #     conv_dim,
                #     conv_dim,
                #     kernel_size=3,
                #     stride=1,
                #     padding=1,
                #     bias=use_bias,
                #     norm=output_norm,
                #     activation=F.relu,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


@SEM_SEG_HEADS_REGISTRY.register()
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), transformer_encoder_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)
