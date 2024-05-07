import torch
from torch import nn

##################### 2D ###############################
class ChannelAttention_2D(nn.Module):
    def __init__(self, channel, reduction=16, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        if self.unsqueeze==True:
            output = output.expand_as(x)
        return output.contiguous()


class SpatialAttention_2D(nn.Module):
    def __init__(self, kernel_size=7, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        if self.unsqueeze==True:
            output = output.expand_as(x)
        return output.contiguous()


class CBAMBlock_2D(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention_2D(channel=channel, reduction=reduction)
        self.sa = SpatialAttention_2D(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return (out + residual).contiguous()

##################### 3D ###############################
class ChannelAttention_3D(nn.Module):
    def __init__(self, channel, reduction=16, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        if self.unsqueeze==True:
            output = output.expand_as(x)
        return output.contiguous()


class SpatialAttention_3D(nn.Module):
    def __init__(self, kernel_size=7, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        if self.unsqueeze==True:
            output = output.expand_as(x)
        return output.contiguous()


class CBAMBlock_3D(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ca = ChannelAttention_3D(channel=channel, reduction=reduction)
        self.sa = SpatialAttention_3D(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return (out + residual).contiguous()

if __name__ == '__main__':

    input=torch.randn(50, 512, 4, 4)
    kernel_size=input.shape[2]
    cbam = CBAMBlock_2D(channel=512, reduction=16, kernel_size=11)
    output=cbam(input)
    print(output.shape)

    input=torch.randn(50, 512, 4, 4, 4)
    kernel_size=input.shape[2]
    cbam = CBAMBlock_3D(channel=512, reduction=16, kernel_size=11)
    output=cbam(input)
    print(output.shape)
