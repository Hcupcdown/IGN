import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock

# -*- coding: utf-8 -*-
# Original copyright:
# The copy right is under the MIT license.
# MANNER (https://github.com/winddori2002/MANNER) / author: winddori2002


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn   = nn.InstanceNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x
    
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        return x * self.sigmoid(x)

class DepthwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=False):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                              groups=in_channels, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x
    
class PointwiseConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        
        x = self.conv(x)
        
        return x


class ResConBlock(nn.Module):
    """
    Residual Conformer block.
        in_channels  :  in channel in encoder and decoder.
        kernel_size  :  kernel size for depthwise convolution.
        growth1      :  expanding channel size and reduce after GLU.
        growth2      :  decide final channel size in the block, encoder for 2, decoder for 1/2.
    """
    def __init__(self, in_channels, kernel_size=31, growth1=2, growth2=2):
        super().__init__()
        
        out_channels1 = int(in_channels*growth1)
        out_channels2 = int(in_channels*growth2)
        self.point_conv1 = nn.Sequential(
                           PointwiseConv(in_channels, out_channels1, stride=1, padding=0, bias=True),
                           nn.InstanceNorm1d(out_channels1), nn.GLU(dim=1))
        self.depth_conv  = nn.Sequential(
                           DepthwiseConv(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
                           nn.InstanceNorm1d(in_channels), Swish())
        self.point_conv2 = nn.Sequential(
                           PointwiseConv(in_channels, out_channels2, stride=1, padding=0, bias=True),
                           nn.InstanceNorm1d(out_channels2), Swish())
        self.conv     = BasicConv(out_channels2, out_channels2, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_channels, out_channels2, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        
        out = self.point_conv1(x)
        out = self.depth_conv(out)
        out = self.point_conv2(out)
        out = self.conv(out)
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
    
class ResBlock(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return x + self.model(x)

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=False,
                 bias=True):
        super().__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        
        self.bn   = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x_in):
        
        x = self.conv(x_in)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
            
        return x + x_in

class Conv2dBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int = 3,
                 stride:int = 1,
                 padding:int = 1,
                 dilation:int = 1,
                 groups:int = 1,
                 bias:bool = False,
                 relu:bool = True,
                 bn:bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              dilation=dilation,
                              groups=groups, 
                              bias=bias)
        self.bn   = nn.BatchNorm2d(out_channels,
                                   eps=1e-5,
                                   momentum=0.01,
                                   affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Encoder(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 layer,
                 depth,
                 g1=2,
                 g2=1):
        super().__init__()
        
        self.layer      = layer
        self.depth      = depth
        self.down_conv  = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size, stride),
                                        nn.InstanceNorm1d(in_channels),
                                        nn.ReLU())
        self.conv_block = ResConBlock(in_channels, growth1=g1, growth2=g2)

    def forward(self, x):
        x = self.down_conv(x)        
        x = self.conv_block(x)

        return x

class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, layer, depth, g1=2, g2=1):
        super().__init__()
        
        self.layer      = layer
        self.depth      = depth
        self.conv_block = ResConBlock(in_channels, growth1=g1, growth2=g2)
        self.up_conv    = nn.Sequential(nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
                                        nn.InstanceNorm1d(out_channels),
                                        nn.ReLU())
    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.up_conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, 
                 in_channels:int=513,
                 hidden:int=513,
                 depth:int=6,
                 kernel_size:int=4,
                 stride:int=2,
                 device:str="cuda:0"):
        super().__init__()

        self.kernel_size  = kernel_size
        self.stride = stride
        self.depth = depth
        self.device = device
        in_convs = [nn.Conv1d(in_channels=in_channels,
                             out_channels=hidden,
                             kernel_size=3,
                             stride=1,
                             padding=1)]
        out_convs = []
        for i in range(depth):
            in_convs.append(Conv1dBlock(in_channels=hidden,
                                        out_channels=hidden,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            out_convs.append(Conv1dBlock(in_channels=hidden,
                                         out_channels=hidden,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1))
        out_convs.append(nn.Sequential(nn.Conv1d(in_channels=hidden,
                                   out_channels=in_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1),
                                   nn.Sigmoid()))
        self.in_conv = nn.Sequential(*in_convs)
        self.out_conv = nn.Sequential(*out_convs)
        self.length = None
        self.flag = 0


        for i in range(1):
            setattr(self, f"conformer_{i}", ConformerBlock(dim=hidden, ff_mult=4, ff_dropout=0.1, conv_dropout=0.1))

    def forward(self, x):
        """
        input X : [B, C, T]
        output X: [B, C, T]
        """
        length = x.shape[-1]
        self.cal_padding(length)
        x = F.pad(x, (0, self.length - length))
        x = self.in_conv(x)

        x = x.permute(0,2,1).contiguous()
        for i in range(1):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()


        x = self.out_conv(x) 
        x = x[..., :length]
        
        return x


    def cal_padding(self, length):
        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        self.length =  int(length)


class ConditionUNet(nn.Module):
    def __init__(self, 
                 in_channels:int=513,
                 hidden:int=513,
                 depth:int=6,
                 kernel_size:int=4,
                 stride:int=2,
                 device:str="cuda:0"):
        super().__init__()

        self.kernel_size  = kernel_size
        self.stride = stride
        self.depth = depth
        self.device = device
        in_convs = [nn.Conv1d(in_channels=in_channels,
                             out_channels=hidden,
                             kernel_size=3,
                             stride=1,
                             padding=1)]
        condition_in_convs = [nn.Conv1d(in_channels=in_channels,
                             out_channels=hidden,
                             kernel_size=3,
                             stride=1,
                             padding=1)]
        out_convs = []
        for i in range(depth):
            in_convs.append(Conv1dBlock(in_channels=hidden,
                                        out_channels=hidden,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            condition_in_convs.append(Conv1dBlock(in_channels=hidden,
                                        out_channels=hidden,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1))
            out_convs.append(Conv1dBlock(in_channels=hidden,
                                         out_channels=hidden,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1))
        out_convs.append(nn.Sequential(nn.Conv1d(in_channels=hidden,
                                   out_channels=in_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1),
                                   nn.Sigmoid()))
        self.in_conv = nn.Sequential(*in_convs)
        self.condition_in_conv = nn.Sequential(*condition_in_convs)
        self.out_conv = nn.Sequential(*out_convs)
        self.length = None
        self.fusion_layear = nn.Sequential(
            nn.Conv1d(hidden*2, hidden, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(hidden),
            nn.ReLU()
        )
        self.flag = 0

        for i in range(1):
            setattr(self, f"conformer_{i}", ConformerBlock(dim=hidden, ff_mult=4, ff_dropout=0.1, conv_dropout=0.1))

    def forward(self, x, condition):
        """
        input X : [B, C, T]
        output X: [B, C, T]
        """
        length = x.shape[-1]
        self.cal_padding(length)
        x = F.pad(x, (0, self.length - length))
        condition = F.pad(condition, (0, self.length - length))
        x = self.in_conv(x)
        condition = self.condition_in_conv(condition)
        x = torch.cat([x, condition], dim=1)
        x = self.fusion_layear(x)

        x = x.permute(0,2,1).contiguous()
        for i in range(1):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()


        x = self.out_conv(x) 
        x = x[..., :length]
        
        return x


    def cal_padding(self, length):
        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        self.length =  int(length)