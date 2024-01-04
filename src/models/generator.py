import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerBlock


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out

class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )
        self.freq_conformer = ConformerBlock(
            dim=num_channel,
            dim_head=num_channel // 4,
            heads=4,
            conv_kernel_size=31,
            attn_dropout=0.2,
            ff_dropout=0.2,
        )

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r
        
    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        
        return out

class Decoder(nn.Module):
    def __init__(self, num_channel=64):
        super(Decoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2))
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 1, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x

class TSCNet(nn.Module):
    def __init__(self,
                 num_channel:int=64,
                 depth:int=4
        ):
        super(TSCNet, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, num_channel, (1, 1), (1, 1)),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=num_channel)
        encoders =[]
        decoders = []
        fusion = []
        for i in range(depth):
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(num_channel, num_channel, (1, 3), (1, 2), padding=(0, 1)),
                    nn.InstanceNorm2d(num_channel, affine=True),
                    nn.PReLU(num_channel))
            )
            decoders.append(
                SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
            )

        decoders.reverse()
        self.encoder_list = nn.ModuleList(encoders)
        self.decoder_list = nn.ModuleList(decoders)
        self.TSCB = TSCB(num_channel=num_channel)

        self.conv = nn.Conv2d(num_channel, 1, (1, 1))

    def forward(self, radar):
        radar = self.conv_1(radar)
        radar = self.dilated_dense(radar)
        skip = []
        for encoder in self.encoder_list:
            skip.append(radar)
            radar = encoder(radar)

        radar = self.TSCB(radar) + radar
        for decoder in self.decoder_list:
            radar = decoder(radar)
            skip_out = skip.pop()
            radar = radar[...,:skip_out.size(-2),:skip_out.size(-1)]
        radar = torch.sigmoid(radar)*skip_out
        radar = self.conv(radar)
        return radar
