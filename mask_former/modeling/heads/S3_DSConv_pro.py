# -*- coding: utf-8 -*-
import os
import torch
from torch import nn
import einops
# import fvcore.nn.weight_init as weight_init

"""Dynamic Snake Convolution Module"""


class DSConv_pro(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str = "cuda",
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        # channel划分为4个一组进行groupnorm
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )


    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled


# Code for testing the DSConv
if __name__ == '__main__':
    import numpy as np


    class EncoderConv(nn.Module):
        def __init__(self, in_ch, out_ch, use_bias, output_norm, kernel_size):
            super(EncoderConv, self).__init__()
            # self.conv = Conv2d(
            #             in_ch,
            #             out_ch,
            #             kernel_size=kernel_size,
            #             stride=1,
            #             padding=1,
            #             bias=use_bias,
            #             norm=output_norm,
            #             activation=F.relu,
            #         )
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.gn = nn.GroupNorm(out_ch // 4, out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.gn(x)
            x = self.relu(x)
            return x
            return self.conv(x)


    class DecoderConv(nn.Module):
        def __init__(self, in_ch, out_ch, use_bias, output_norm, kernel_size):
            super(DecoderConv, self).__init__()
            # self.conv = Conv2d(
            #     in_ch,
            #     out_ch,
            #     kernel_size=kernel_size,
            #     stride=1,
            #     padding=1,
            #     bias=use_bias,
            #     norm=output_norm,
            #     activation=F.relu,
            # )
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.gn = nn.GroupNorm(out_ch // 4, out_ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.conv(x)
            x = self.gn(x)
            x = self.relu(x)
            return x
            # return self.conv(x)
        

    class DSCBlock(nn.Module):
        def __init__(self, in_ch, out_ch, use_bias, output_norm, device, kernel_size=3, extend_scope=1.0, if_offset=True):
            """
            初始化函数还不完善，后续需要把这几个参数写到config里面
            """
            super(DSCBlock, self).__init__()
            self.conv00 = EncoderConv(in_ch, out_ch, use_bias, output_norm, kernel_size)
            self.conv1 = DecoderConv(3*out_ch, out_ch, use_bias, output_norm, kernel_size)
            self.conv0x = DSConv_pro(
                in_ch,
                out_ch,
                kernel_size,
                extend_scope,
                0,
                if_offset,
                device=device,
            )
            self.conv0y = DSConv_pro(
                in_ch,
                out_ch,
                kernel_size,
                extend_scope,
                1,
                if_offset,
                device=device,
            )
            # weight_init.c2_xavier_fill(self.conv0x)
            # weight_init.c2_xavier_fill(self.conv0y)
            # weight_init.c2_xavier_fill(self.conv1)
            # weight_init.c2_xavier_fill(self.conv00)


        def forward(self, x):
            # block0
            x_00_0 = self.conv00(x)
            print(x_00_0.shape)
            x_0x_0 = self.conv0x(x)
            x_0y_0 = self.conv0y(x)
            tmp = torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1)
            print(tmp.shape)
            x_1 = self.conv1(tmp)
            print(x_1.shape)
            return x_1



    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = np.random.rand(1, 256, 20, 20)
    # A = np.ones(shape=(3, 2, 2, 3), dtype=np.float32)
    # print(A)
    A = A.astype(dtype=np.float32)
    A = torch.from_numpy(A)
    # print(A.shape)
    # conv0 = DSConv_pro(
    #     in_channels = 256,
    #     out_channels = 256,
    #     kernel_size = 3,
    #     extend_scope = 1.0,
    #     morph = 0,
    #     if_offset = True,
    #     device=device)
    # if torch.cuda.is_available():
    #     A = A.to(device)
    #     conv0 = conv0.to(device)
    # out = conv0(A)
    # print(out.shape)
    # print(conv0)
    # print(out)


    norm = ""
    use_bias = norm == ""
    output_norm = None
    # print(A.shape)
    conv0 = DSCBlock(256, 256, True, output_norm, device, kernel_size=5, extend_scope=1.0, if_offset=True)
    if torch.cuda.is_available():
        A = A.to(device)
        conv0 = conv0.to(device)
    out = conv0(A)
    print(out.shape)
    print(conv0)
