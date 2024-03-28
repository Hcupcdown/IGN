import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock


class BasicConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False
                ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        if bn:
            self.bn = nn.InstanceNorm1d(out_channels,
                                        eps=1e-5,
                                        momentum=0.01,
                                        affine=True)
        else:
            self.bn = None
        
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

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 bias=False
                ):
        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, 
                              groups=in_channels,
                              stride=stride,
                              padding=padding,
                              bias=bias
                            )

    def forward(self, x):
        
        x = self.conv(x)
        
        return x
    
class PointwiseConv(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding=0,
                 bias=True
                ):
        super(PointwiseConv, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=stride,
                              padding=padding,
                              bias=bias)

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
    def __init__(self, in_channels, kernel_size=3, growth1=2, growth2=1):
        super().__init__()
        
        # out_channels1 = int(in_channels*growth1)
        # out_channels2 = int(in_channels*growth2)
        # self.point_conv1 = nn.Sequential(
        #                                  PointwiseConv(in_channels,
        #                                                out_channels1,
        #                                                stride=1,
        #                                                padding=0,
        #                                                bias=True),
        #                                  nn.InstanceNorm1d(out_channels1),
        #                                  nn.GLU(dim=1))
        # self.depth_conv  = nn.Sequential(DepthwiseConv(in_channels,
        #                                                in_channels,
        #                                                kernel_size,
        #                                                stride=1,
        #                                                padding=(kernel_size - 1) // 2),
        #                                  nn.InstanceNorm1d(in_channels),
        #                                  Swish()
        #                                 )
        # self.point_conv2 = nn.Sequential(PointwiseConv(in_channels,
        #                                                out_channels2,
        #                                                stride=1,
        #                                                padding=0,
        #                                                bias=True),
        #                                  nn.InstanceNorm1d(out_channels2),
        #                                  Swish()
        #                                 )
        self.conv = BasicConv(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              relu=True,
                              bn=False)
        self.shortcut = BasicConv(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  relu=False,
                                  bn=False)

    def forward(self, x):
        
        # out = self.point_conv1(x)
        # out = self.depth_conv(out)
        # out = self.point_conv2(out)
        out = self.conv(x)
        out1 = self.shortcut(x)
        out = out + out1
        out = F.relu(out)
        
        return out

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden,
                 depth
                ):
        super().__init__()
        in_conv_list = [nn.Conv1d(in_channels=in_channels,
                                  out_channels=hidden,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)]
        for i in range(depth):
            in_conv_list.append(ResConBlock(in_channels=hidden))
        self.in_conv = nn.Sequential(*in_conv_list)

    def forward(self, x):
        x = self.in_conv(x)
        return x

class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden,
                 depth
                ):
        super().__init__()
        out_conv_list = []
        for i in range(depth):
            out_conv_list.append(ResConBlock(in_channels=hidden))
        out_conv_list.append(nn.Sequential(nn.Conv1d(in_channels=hidden,
                                   out_channels=in_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1),
                                   nn.Sigmoid()))

        self.out_conv = nn.Sequential(*out_conv_list)

    def forward(self, x):
        x = self.out_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, 
                 in_channels:int=1,
                 hidden:int=512,
                 depth:int=6,
                 attn_num:int=5,):
        super().__init__()


        self.depth = depth
        self.attn_num = attn_num

        self.in_conv = Encoder(in_channels=in_channels,
                               hidden=hidden,
                               depth=depth)
        self.out_conv = Decoder(in_channels=in_channels,
                                hidden=hidden,
                                depth=depth)

        for i in range(attn_num):
            setattr(self, f"conformer_{i}", ConformerBlock(dim=hidden,
                                                           ff_mult=4,
                                                           ff_dropout=0.1,
                                                           conv_dropout=0.1))

    def forward(self, x):
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, T].

        Returns:
            torch.Tensor: Output tensor with shape [B, C, T].
        """
        x = self.in_conv(x)

        x = x.permute(0,2,1).contiguous()
        for i in range(self.attn_num):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()

        x = self.out_conv(x) 
        
        return x


class ConditionUNet(UNet):
    def __init__(self, 
                 in_channels:int=1,
                 hidden:int=512,
                 depth:int=6,
                 attn_num:int=5
                ):
        super().__init__(in_channels, hidden, depth, attn_num)

        self.condition_conv = Encoder(in_channels=in_channels,
                                      hidden=hidden,
                                      depth=depth)
        self.fusion_conv = BasicConv(in_channels=hidden*2,
                                     out_channels=hidden,
                                     kernel_size=1,
                                     stride=1,
                                     relu=True,
                                     bias=True)

    def forward(self, x, condition):
        """
        Forward pass of the UNet model.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, T].
            condition (torch.Tensor): Condition tensor with shape [B, C, T].

        Returns:
            torch.Tensor: Output tensor with shape [B, C, T].
        """
        x = self.in_conv(x)
        condition = self.condition_conv(condition)
        x = torch.cat([x, condition], dim=1)
        x = self.fusion_conv(x)

        x = x.permute(0,2,1).contiguous()
        for i in range(1):
            x = getattr(self, f"conformer_{i}")(x) + x
        x = x.permute(0,2,1).contiguous()

        x = self.out_conv(x) 
        
        return x