import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

##################### 2D ###############################
class ChannelAttention_2D(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        if self.unsqueeze==True:
            res=res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class SpatialAttention_2D(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=1, dia_val=2, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=2, dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        if self.unsqueeze==True:
            res=res.expand_as(x)
        return res


class BAMBlock_2D(nn.Module):

    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca=ChannelAttention_2D(channel=channel,reduction=reduction,unsqueeze=True)
        self.sa=SpatialAttention_2D(channel=channel,reduction=reduction,dia_val=dia_val,unsqueeze=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out.contiguous()

##################### 3D ###############################
class ChannelAttention_3D(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        if self.unsqueeze==True:
            res=res.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class SpatialAttention_3D(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=1, dia_val=2, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv3d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm3d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv3d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=2, dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm3d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv3d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        if self.unsqueeze==True:
            res=res.expand_as(x)
        return res


class BAMBlock_3D(nn.Module):

    def __init__(self, channel=512, reduction=16, dia_val=2):
        super().__init__()
        self.ca=ChannelAttention_3D(channel=channel,reduction=reduction,unsqueeze=True)
        self.sa=SpatialAttention_3D(channel=channel,reduction=reduction,dia_val=dia_val,unsqueeze=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out.contiguous()


if __name__ == '__main__':
    input = torch.randn(50, 512, 3, 3)
    bam = BAMBlock_2D(channel=512,  reduction=16, dia_val=2)
    output = bam(input)
    print(output.shape)

    input = torch.randn(50, 512, 3, 3, 3)
    bam = BAMBlock_3D(channel=512,  reduction=16, dia_val=2)
    output = bam(input)
    print(output.shape)