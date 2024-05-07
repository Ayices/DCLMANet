import torch
from torch import nn
import math

##################### 2D ###############################
class ECAAttention_2D(nn.Module):

    def __init__(self, channel=512, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log2(channel)+b)/gamma)
        self.kernel_size = t if t%2 else t+1
        self.kernel_size = 1
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1, 1, kernel_size=self.kernel_size,padding=int(self.kernel_size/2))
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.gap(x).view(b, c, 1, 1) #bs,c,1,1
        y=y.squeeze(-1).permute(0, 2, 1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0, 2, 1).unsqueeze(-1) #bs, c, 1, 1
        return y

##################### 3D ###############################
class ECAAttention_3D(nn.Module):

    def __init__(self, channel=512, gamma=2, b=1):
        super().__init__()
        t = int(abs(math.log2(channel) + b) / gamma)
        self.kernel_size = t if t % 2 else t + 1
        self.kernel_size = 1
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=int(self.kernel_size / 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.gap(x).view(b, c, 1, 1)  # bs,c,1,1,1
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c,1
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # bs,c,1,1
        return y


if __name__ == '__main__':
    input=torch.randn(50, 512, 3, 3)
    eca = ECAAttention_2D(channel=512)
    output=eca(input)
    print(output.shape)

    input=torch.randn(50, 512, 3, 3, 3)
    eca = ECAAttention_3D(channel=512)
    output=eca(input)
    print(output.shape)
