from torch import nn
import torch

class SEAttention_2D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention_2D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # return x*y.contiguous()
        return y.contiguous()


class SEAttention_3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEAttention_3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _, _  = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # return x*y.contiguous()
        return y.contiguous()

if __name__ == '__main__':
    input=torch.randn(50, 512, 8, 8)
    SE = SEAttention_2D(channel=512, reduction=16)
    output = SE(input)
    print(output.shape)

    input=torch.randn(50, 512, 8, 8, 8)
    SE = SEAttention_3D(channel=512, reduction=16)
    output = SE(input)
    print(output.shape)
