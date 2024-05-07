from torch import nn
import torch

class MS_CAM_2D(nn.Module):

    def __init__(self, channel=64, reduction=4):
        super(MS_CAM_2D, self).__init__()
        inter_channels = int(channel // reduction)

        self.local_att = nn.Sequential(
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        # return x * wei
        return wei


class MS_CAM_3D(nn.Module):

    def __init__(self, channel=64, reduction=4):
        super(MS_CAM_3D, self).__init__()
        inter_channels = int(channel // reduction)

        self.local_att = nn.Sequential(
            nn.Conv3d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channel, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        # return x * wei
        return wei



if __name__ == '__main__':
    input=torch.randn(50, 512, 8, 8)
    SE = MS_CAM_2D(channel=512, reduction=4)
    output=SE(input)
    print(output.shape)

    input=torch.randn(50, 512, 8, 8, 8)
    SE = MS_CAM_3D(channel=512, reduction=4)
    output=SE(input)
    print(output.shape)