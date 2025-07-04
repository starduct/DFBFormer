# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.position_encoding import PositionEmbeddingSine
from ..transformer.transformer import TransformerEncoder, TransformerEncoderLayer
# from .S3_DSConv_pro import DSConv_pro
from .S3_DSConv import DSConv


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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class DSC2Block(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias, output_norm, device, kernel_size=3, extend_scope=1.0, if_offset=True):
        """
        初始化函数还不完善，后续需要把这几个参数写到config里面
        version=2，调整模块的输入部分，并增加conv数量
        """
        super(DSC2Block, self).__init__()
        # 用来调整channel数量的模块可以换换
        # 如何统一输入是后面几个版本的主要差异了
        # 这个版本先用一个在原模块前的卷积快来修改inp
        # shortcut可以换点东西上去,inp=?, oup=oup

        # self.conv0 = Conv2d(
        #     in_ch,
        #     out_ch*3,
        #     kernel_size=1,
        #     stride=1,
        #     padding=1,
        #     bias=use_bias,
        #     norm=output_norm,
        #     activation=F.relu,
        # )
        # 这里没太明白是哪里出了问题，上下的差异在后面的那些参数上，可能就多了norm和relu，而且下面的shortcut也是同样的代码
        self.conv0 = nn.Conv2d(in_ch, 3*out_ch, kernel_size=1, stride=1)
        self.conv00 = Conv2d(
            out_ch*3,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        self.conv1 = Conv2d(
            3 * out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

        weight_init.c2_xavier_fill(self.conv0)
        weight_init.c2_xavier_fill(self.conv00)
        weight_init.c2_xavier_fill(self.conv1)

        self.conv0x = DSConv(
            out_ch*3,
            out_ch,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device=device,
        )
        self.conv0y = DSConv(
            out_ch*3,
            out_ch,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device=device,
        )

        self.shortcut = Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

    def forward(self, x):
        # block0
        # 这里可能有问题，如果inp和oup的channel不一致，那么dw卷积无法使用，dw无法修改channel数量
        x_ = self.conv0(x)
        # in -> out * 3

        # 下面这几层的输入是3倍于输出的channel数量
        x_00_0 = self.conv00(x_)  # 后面自带BN和ReLu
        x_0x_0 = self.conv0x(x_)  # 后面自带GN和ReLu
        x_0y_0 = self.conv0y(x_)  # 后面自带GN和ReLu

        # 后面自带BN和ReLu
        x_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        # 3*out -> out

        x_shortcut = self.shortcut(x)
        # in -> out

        if x_1.shape == x_shortcut.shape:
            oup = x_1 + x_shortcut
        else:
            oup = x_1

        return oup


class MaskDSC2Block(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias=None, output_norm=None, device=None, kernel_size=3, extend_scope=1.0, if_offset=True):
        """
        初始化函数还不完善，后续需要把这几个参数写到config里面
        version=2，调整模块的输入部分，并增加conv数量
        """
        super(MaskDSC2Block, self).__init__()
        # 用来调整channel数量的模块可以换换
        # 如何统一输入是后面几个版本的主要差异了
        # 这个版本先用一个在原模块前的卷积快来修改inp
        # shortcut可以换点东西上去,inp=?, oup=oup

        # self.conv0 = Conv2d(
        #     in_ch,
        #     out_ch*3,
        #     kernel_size=1,
        #     stride=1,
        #     padding=1,
        #     bias=use_bias,
        #     norm=output_norm,
        #     activation=F.relu,
        # )
        # 这里没太明白是哪里出了问题，上下的差异在后面的那些参数上，可能就多了norm和relu，而且下面的shortcut也是同样的代码
        self.conv0 = nn.Conv2d(in_ch, 3*out_ch, kernel_size=1, stride=1)
        self.conv00 = Conv2d(
            out_ch*3,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        self.conv1 = Conv2d(
            3 * out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

        weight_init.c2_xavier_fill(self.conv0)
        weight_init.c2_xavier_fill(self.conv00)
        weight_init.c2_xavier_fill(self.conv1)

        self.conv0x = DSConv(
            out_ch*3,
            out_ch,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device=device,
        )
        self.conv0y = DSConv(
            out_ch*3,
            out_ch,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device=device,
        )

        self.shortcut = Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

    def forward(self, x):
        # block0
        # 这里可能有问题，如果inp和oup的channel不一致，那么dw卷积无法使用，dw无法修改channel数量
        x_ = self.conv0(x)
        # in -> out * 3

        # 下面这几层的输入是3倍于输出的channel数量
        x_00_0 = self.conv00(x_)  # 后面自带BN和ReLu
        x_0x_0 = self.conv0x(x_)  # 后面自带GN和ReLu
        x_0y_0 = self.conv0y(x_)  # 后面自带GN和ReLu

        # 后面自带BN和ReLu
        x_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        # 3*out -> out

        x_shortcut = self.shortcut(x)
        # in -> out

        if x_1.shape == x_shortcut.shape:
            oup = x_1 + x_shortcut
        else:
            oup = x_1

        # x_1 = self.conv1(x_00_0)
        return oup


class DSCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias, output_norm, device, kernel_size=3, extend_scope=1.0, if_offset=True):
        """
        初始化函数还不完善，后续需要把这几个参数写到config里面
        version=1，该版本直接修改了basedecoder，效果下降，现已放弃
        """
        super(DSCBlock, self).__init__()
        self.conv00 = Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        self.conv1 = Conv2d(
            3*out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

        weight_init.c2_xavier_fill(self.conv00)
        weight_init.c2_xavier_fill(self.conv1)

        self.conv0x = DSConv(
            in_ch,
            out_ch,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device=device,
        )
        self.conv0y = DSConv(
            in_ch,
            out_ch,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device=device,
        )

    def forward(self, x):
        # block0
        # 这里可能有问题，如果inp和oup的channel不一致，那么dw卷积无法使用，dw无法修改channel数量
        x_00_0 = self.conv00(x)
        x_0x_0 = self.conv0x(x)
        x_0y_0 = self.conv0y(x)
        x_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        # x_1 = self.conv1(x_00_0)
        return x_1


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
        device: torch.device
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
        # starting from "res2" to "res5"，分辨率从大到小
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]  # 通道数量从小到大

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里是分辨率最小的一层，那后面有点不对劲啊？
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSCBlock(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
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
                # output_conv = DSCBlock(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]  # 这里变成分辨率4321
        self.output_convs = output_convs[::-1]  # 分辨率自大到小4321

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
        ret["device"] = cfg.MODEL.DEVICE
        return ret

    def forward_features(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):  # 从分辨率最小的开始
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)  # 第一次循环会生成y
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + \
                    F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), None

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning(
            "Calling forward() may cause unpredicted behavior of PixelDecoder module.")
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
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

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
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
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
                y = cur_fpn + \
                    F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
        return self.mask_features(y), transformer_encoder_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning(
            "Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


@SEM_SEG_HEADS_REGISTRY.register()
class DSCDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=3
                )
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=3
                )
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


@SEM_SEG_HEADS_REGISTRY.register()
class DSCDecoderMask(BasePixelDecoder):
    # 在最后输出mask的位置使用dsc2block
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC2Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                #     kernel_size=3
                # )
                weight_init.c2_xavier_fill(output_conv)
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

                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                #     kernel_size=3
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = MaskDSC2Block(
            conv_dim,
            mask_dim,
            device=device,
            kernel_size=5
        )
        # weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class DSCDecoderKernelSize(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层，换参数之后显存爆了，先把输出层的dscblock关了
                output_norm = get_norm(norm, conv_dim)
                # output_conv = Conv2d(
                #                     in_channels,
                #                     conv_dim,
                #                     kernel_size=3,
                #                     stride=1,
                #                     padding=1,
                #                     bias=use_bias,
                #                     norm=output_norm,
                #                     activation=F.relu,
                #                 )
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=5
                )
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=5
                )
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


class DSC3Block(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias, output_norm, device, kernel_size=9, extend_scope=1.0, if_offset=True):
        """
        初始化函数还不完善，后续需要把这几个参数写到config里面
        version=3，dsc模块前提，类convnext结构，切换dw卷积，这个的想法是减少参数量，模块参数有问题，换下面的plus版本
        """
        super(DSC3Block, self).__init__()
        # 用来调整channel数量的模块可以换换
        # 如何统一输入是后面几个版本的主要差异了
        # 这个版本先保持inp不变，后通过1x1卷积放缩，将channel对齐到oup

        self.conv00 = Conv2d(
            in_ch,
            in_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=in_ch,
            bias=use_bias,
            norm=output_norm,
            # activation=F.relu,
        )

        self.conv0x = DSConv(
            in_ch,
            in_ch,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device=device,
        )
        self.conv0y = DSConv(
            in_ch,
            in_ch,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device=device,
        )

        # 下面这个函数是关键位置，让前面的输出加到一起，这里出问题，kernel size错了
        self.conv1 = Conv2d(
            in_ch,
            4 * out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            activation=F.relu,
        )
        self.conv2 = nn.Conv2d(4*out_ch, out_ch, kernel_size=1, stride=1)

        weight_init.c2_xavier_fill(self.conv2)
        weight_init.c2_xavier_fill(self.conv00)
        weight_init.c2_xavier_fill(self.conv1)

    def forward(self, x):
        # shortcut前提
        x_ = x
        # in -> in

        # 下面这几层的输入是1倍于输入的channel数量 inp -> inp
        x_00_0 = self.conv00(x)  # 后面有激活函数 inp
        oup = x_00_0 + x_
        x_0x_0 = self.conv0x(x)  # 后面有激活函数 inp
        oup = oup + x_0x_0
        x_0y_0 = self.conv0y(x)  # 后面有激活函数 inp
        oup = oup + x_0y_0

        x_1 = self.conv1(oup)
        # inp -> 4*out

        x_2 = self.conv2(x_1)
        # 4*oup -> oup

        return x_2


class DSC3BlockPlus(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias, output_norm, device, kernel_size=9, extend_scope=1.0, if_offset=True):
        """
        初始化函数还不完善，后续需要把这几个参数写到config里面
        version=3.1，dsc模块前提，类convnext结构，切换dw卷积，这个的想法是减少参数量，修复上面block的问题，另外调了下结构
        """
        super(DSC3Block, self).__init__()
        # 用来调整channel数量的模块可以换换
        # 如何统一输入是后面几个版本的主要差异了
        # 这个版本先保持inp不变，后通过1x1卷积放缩，将channel对齐到oup

        self.conv00 = Conv2d(
            in_ch,
            in_ch,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=in_ch,
            bias=use_bias,
            norm=output_norm,
            # activation=F.relu,
        )

        self.conv0x = DSConv(
            in_ch,
            in_ch,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device=device,
        )
        self.conv0y = DSConv(
            in_ch,
            in_ch,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device=device,
        )

        # 下面这个函数是关键位置，让前面的输出加到一起
        self.conv1 = Conv2d(
            3 * in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            # padding=1,
            bias=use_bias,
            activation=F.relu,
        )
        self.conv2 = Conv2d(in_ch, out_ch, kernel_size=1,
                            stride=1, activation=F.relu,)

        weight_init.c2_xavier_fill(self.conv00)
        weight_init.c2_xavier_fill(self.conv1)
        weight_init.c2_xavier_fill(self.conv2)

    def forward(self, x):
        # shortcut前提
        x_ = x
        # in -> in

        # 下面这几层的输入是1倍于输入的channel数量 inp -> inp
        x_00_0 = self.conv00(x)  # 后面有激活函数 inp
        x_0x_0 = self.conv0x(x)  # 后面有激活函数 inp
        x_0y_0 = self.conv0y(x)  # 后面有激活函数 inp

        x_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        # 3 * inp -> oup

        x_2 = self.conv2(x_)
        # inp -> oup

        return x_1 + x_2


@SEM_SEG_HEADS_REGISTRY.register()
class DSC2Decoder(BasePixelDecoder):
    # 这个用的是dsc3block，不对
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, in_channels)
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
                output_conv = DSC3Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
                output_conv = DSC3Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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


@SEM_SEG_HEADS_REGISTRY.register()
class BottomDSCDecoder(BasePixelDecoder):
    """
    只在输出层位置使用（最大分辨率区域有transformer）目前设置为Kernel size = 5
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=5
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
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


@SEM_SEG_HEADS_REGISTRY.register()
class Bottom3DSCDecoder(BasePixelDecoder):
    """
    第一层不用，在后面的输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 改成判断位，这段一定是后面靠近输出的位置
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0:
                # 最前面一层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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


@SEM_SEG_HEADS_REGISTRY.register()
class Bottom3DSCDecoderKernelSize(BasePixelDecoder):
    """
    第一层不用，在后面的输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 改成判断位，这段一定是后面靠近输出的位置
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=5,
                )
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0:
                # 最前面一层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=5,
                )
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


@SEM_SEG_HEADS_REGISTRY.register()
class Bottom2DSCDecoder(BasePixelDecoder):
    """
    第一层不用，在后面的输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这段一定是后面靠近输出的位置，使用dscblock
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0 or idx == 1:
                # 最前面两层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
        # 这里是最后统一到输出channel上
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class Bottom24DSCDecoder(BasePixelDecoder):
    """
    第一层不用，在后面的输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这段一定是后面靠近输出的位置，使用dscblock
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = DSC2Block(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
                # weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0 or idx == 2:
                # 1,3两层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
        # 这里是最后统一到输出channel上
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class Bottom23DSCDecoder(BasePixelDecoder):
    """
    第一层不用，在后面的输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这段一定是后面靠近输出的位置，不使用dscblock
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC2Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0:
                # 1层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
        # 这里是最后统一到输出channel上
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class Layer2DSCDecoder(BasePixelDecoder):
    """
    更正为feature map倒数第二大的层
    只在第二层用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这段一定是后面靠近输出的位置，不使用dscblock
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC2Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0 or idx == 2:
                # 1,3两层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
        # 这里是最后统一到输出channel上
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class FrontDSCDecoder(BasePixelDecoder):
    """
    只在非输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC3Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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


@SEM_SEG_HEADS_REGISTRY.register()
class Layer3DSCDecoder(BasePixelDecoder):
    """
    更正为feature map倒数第三大的层，feature map 第二小的层
    只在第三层用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这段一定是后面靠近输出的位置，不使用dscblock
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC2Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0 or idx == 1:
                # 1,2两层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
        # 这里是最后统一到输出channel上
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class Layer3DSCDecoderKernelSize(BasePixelDecoder):
    """
    更正为feature map倒数第三大的层，feature map 第二小的层
    只在第三层用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这段一定是后面靠近输出的位置，不使用dscblock
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC2Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)

            # 除了最后一层以外，设置不同的模块
            elif idx == 0 or idx == 1:
                # 1,2两层不使用dscblock
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=5
                )
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
        # 这里是最后统一到输出channel上
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)


@SEM_SEG_HEADS_REGISTRY.register()
class Front1DSCDecoder(BasePixelDecoder):
    """
    只在非输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC3Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            elif idx > 0:
                # 中间层
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
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
                # output_conv = DSC2Block(
                #     conv_dim,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                # )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
            else:
                # 第一层
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                # 这里需要加强
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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


@SEM_SEG_HEADS_REGISTRY.register()
class Front3DSCDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                # output_conv = DSC2Block(
                #     in_channels,
                #     conv_dim,
                #     use_bias=use_bias,
                #     output_norm=output_norm,
                #     device=device,
                #     kernel_size=3
                # )
                weight_init.c2_xavier_fill(output_conv)
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
                output_conv = DSC2Block(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                    kernel_size=3
                )
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


class ReParaBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bias, output_norm, device, kernel_size=3, extend_scope=1.0, if_offset=True):
        """
        初始化函数还不完善，后续需要把这几个参数写到config里面
        version=3，重参数化模块
        """
        super(ReParaBlock, self).__init__()
        # 用来调整channel数量的模块可以换换
        # 如何统一输入是后面几个版本的主要差异了
        # 这个版本先用一个在原模块前的卷积快来修改inp
        # shortcut可以换点东西上去,inp=?, oup=oup

        # self.conv0 = Conv2d(
        #     in_ch,
        #     out_ch*3,
        #     kernel_size=1,
        #     stride=1,
        #     padding=1,
        #     bias=use_bias,
        #     norm=output_norm,
        #     activation=F.relu,
        # )
        # 这里没太明白是哪里出了问题，上下的差异在后面的那些参数上，可能就多了norm和relu，而且下面的shortcut也是同样的代码
        self.conv0 = nn.Conv2d(in_ch, 3*out_ch, kernel_size=1, stride=1)
        self.conv00 = Conv2d(
            out_ch*3,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        self.conv1 = Conv2d(
            3 * out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

        weight_init.c2_xavier_fill(self.conv0)
        weight_init.c2_xavier_fill(self.conv00)
        weight_init.c2_xavier_fill(self.conv1)

        self.conv0x = DSConv(
            out_ch*3,
            out_ch,
            kernel_size,
            extend_scope,
            0,
            if_offset,
            device=device,
        )
        self.conv0y = DSConv(
            out_ch*3,
            out_ch,
            kernel_size,
            extend_scope,
            1,
            if_offset,
            device=device,
        )

        self.shortcut = Conv2d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )

    def forward(self, x):
        # block0
        # 这里可能有问题，如果inp和oup的channel不一致，那么dw卷积无法使用，dw无法修改channel数量
        x_ = self.conv0(x)
        # in -> out * 3

        # 下面这几层的输入是3倍于输出的channel数量
        x_00_0 = self.conv00(x_)
        x_0x_0 = self.conv0x(x_)
        x_0y_0 = self.conv0y(x_)

        x_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
        # 3*out -> out

        x_shortcut = self.shortcut(x)
        # in -> out

        if x_1.shape == x_shortcut.shape:
            oup = x_1 + x_shortcut
        else:
            oup = x_1

        # x_1 = self.conv1(x_00_0)
        return oup


@SEM_SEG_HEADS_REGISTRY.register()
class ReParaDecoder(BasePixelDecoder):
    """
    只在非输出层位置使用（最大分辨率区域有transformer）
    """
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        device: torch.device
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim,
                         mask_dim=mask_dim, norm=norm, device=device)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        # starting from "res2" to "res5"
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                # 这里需要加强，这个位置应该是最后的输出层
                output_norm = get_norm(norm, conv_dim)
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
                output_conv = ReParaBlock(
                    in_channels,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
                output_conv = ReParaBlock(
                    conv_dim,
                    conv_dim,
                    use_bias=use_bias,
                    output_norm=output_norm,
                    device=device,
                )
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
