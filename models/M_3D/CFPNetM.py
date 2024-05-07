import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_padding(kernel_size, dilation_rate, stride):
    # 计算有效的卷积核尺寸
    effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
    # 计算填充
    padding = (effective_kernel_size - stride) // 2
    return padding

class Conv3dBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, stride, padding='same', activation='relu'):
        super(Conv3dBN, self).__init__()
        if dilation_rate != 1:
            padding = calculate_padding(kernel_size, dilation_rate, stride)
        else:
            padding = kernel_size // 2

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation_rate, groups=1, bias=False)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        return x

class CFPModule(nn.Module):
    def __init__(self, in_channels, filters, d_size):
        super(CFPModule, self).__init__()
        self.x_inp_conv = Conv3dBN(in_channels, filters // 4, 1, 1, 1)

        # Path 1
        self.x_1_1 = Conv3dBN(filters // 4, filters // 16, 3, 1, 1)
        self.x_1_2 = Conv3dBN(filters // 16, filters // 16, 3, 1, 1)
        self.x_1_3 = Conv3dBN(filters // 16, filters // 8, 3, 1, 1)

        # Path 2
        self.x_2_1 = Conv3dBN(filters // 4, filters // 16, 3, d_size // 4 + 1, 1)
        self.x_2_2 = Conv3dBN(filters // 16, filters // 16, 3, d_size // 4 + 1, 1)
        self.x_2_3 = Conv3dBN(filters // 16, filters // 8, 3, d_size // 4 + 1, 1)

        # Path 3
        self.x_3_1 = Conv3dBN(filters // 4, filters // 16, 3, d_size // 2 + 1, 1, 'keru')
        self.x_3_2 = Conv3dBN(filters // 16, filters // 16, 3, d_size // 2 + 1, 1, 'keru')
        self.x_3_3 = Conv3dBN(filters // 16, filters // 8, 3, d_size // 2 + 1, 1, 'keru')

        # Path 4
        self.x_4_1 = Conv3dBN(filters // 4, filters // 16, 3, d_size + 1, 1, 'triple')
        self.x_4_2 = Conv3dBN(filters // 16, filters // 16, 3, d_size + 1, 1, 'triple')
        self.x_4_3 = Conv3dBN(filters // 16, filters // 8, 3, d_size + 1, 1, 'triple')

        # Batch Normalization Layers
        self.bn_o_1 = nn.InstanceNorm3d(filters // 4)
        self.bn_o_2 = nn.InstanceNorm3d(filters // 4)
        self.bn_o_3 = nn.InstanceNorm3d(filters // 4)
        self.bn_o_4 = nn.InstanceNorm3d(filters // 4)
        self.output_bn = nn.InstanceNorm3d(filters)

        # Final Convolution
        self.final_conv = Conv3dBN(filters, filters, 1, 1, 1, padding='valid')

    def forward(self, x):
        x_inp = self.x_inp_conv(x)

        # Path 1
        x_1_1 = self.x_1_1(x_inp)
        x_1_2 = self.x_1_2(x_1_1)
        x_1_3 = self.x_1_3(x_1_2)

        # Path 2
        x_2_1 = self.x_2_1(x_inp)
        x_2_2 = self.x_2_2(x_2_1)
        x_2_3 = self.x_2_3(x_2_2)

        # Path 3
        x_3_1 = self.x_3_1(x_inp)
        x_3_2 = self.x_3_2(x_3_1)
        x_3_3 = self.x_3_3(x_3_2)

        # Path 4
        # 路径 4
        x_4_1 = self.x_4_1(x_inp)
        x_4_2 = self.x_4_2(x_4_1)
        x_4_3 = self.x_4_3(x_4_2)

        # 连接不同路径的输出
        o_1 = torch.cat((x_1_1, x_1_2, x_1_3), dim=1)
        o_2 = torch.cat((x_2_1, x_2_2, x_2_3), dim=1)
        o_3 = torch.cat((x_1_1, x_3_2, x_3_3), dim=1)
        o_4 = torch.cat((x_1_1, x_4_2, x_4_3), dim=1)

        # 应用实例归一化
        o_1 = self.bn_o_1(o_1)
        o_2 = self.bn_o_2(o_2)
        o_3 = self.bn_o_3(o_3)
        o_4 = self.bn_o_4(o_4)

        # 累加不同路径的输出
        ad1 = o_1
        ad2 = ad1 + o_2
        ad3 = ad2 + o_3
        ad4 = ad3 + o_4

        # 将所有路径的输出合并
        output = torch.cat((ad1, ad2, ad3, ad4), dim=1)
        output = self.output_bn(output)

        # 应用最后的卷积层
        output = self.final_conv(output)

        # 最终添加输入和输出
        output = output + x

        return output

class CFPNetM(nn.Module):
    def __init__(self, channels):
        super(CFPNetM, self).__init__()

        # 初始化网络层
        self.conv1 = Conv3dBN(channels, 32, kernel_size=3, dilation_rate=1, stride=2)
        self.conv2 = Conv3dBN(32, 32, kernel_size=3, dilation_rate=1, stride=1)
        self.conv3 = Conv3dBN(32, 32, kernel_size=3, dilation_rate=1, stride=1)

        # 注入层
        self.injection_1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(32),
            nn.ReLU()
        )

        self.opt_Cat = Conv3dBN(36, 64, kernel_size=3, dilation_rate=1, stride=2)
        self.opt_Cat2 = Conv3dBN(132, 128, kernel_size=3, dilation_rate=1, stride=2)

        # CFP模块
        self.cfp1 = CFPModule(64, 64, d_size=2)
        self.cfp2 = CFPModule(64, 64, d_size=2)

        # 第二次注入层
        self.injection_2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(64),
            nn.ReLU()
        )

        # 更多的CFP模块
        self.cfp3 = CFPModule(128, 128, d_size=4)
        self.cfp4 = CFPModule(128, 128, d_size=4)
        self.cfp5 = CFPModule(128, 128, d_size=8)
        self.cfp6 = CFPModule(128, 128, d_size=8)
        self.cfp7 = CFPModule(128, 128, d_size=16)
        self.cfp8 = CFPModule(128, 128, d_size=16)

        # 第三次注入层
        self.injection_3 = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(128),
            nn.ReLU()
        )

        # 上采样层
        self.up1 = nn.ConvTranspose3d(260, 128, kernel_size=2, stride=2, padding=0)
        self.up2 = nn.ConvTranspose3d(260, 64, kernel_size=2, stride=2, padding=0)
        self.up3 = nn.ConvTranspose3d(100, 32, kernel_size=2, stride=2, padding=0)

        # 最终的卷积层
        self.final_conv = Conv3dBN(32, 4, kernel_size=1, dilation_rate=1, stride=1, activation='sigmoid', padding='valid')

        self.Conv_1x1 = nn.Conv3d(4, 4,kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_3 = self.conv3(x_2)

        injection_1 = self.injection_1(x)
        opt_cat_1 = torch.cat((x_3, injection_1), dim=1)

        opt_cat_1_0 = self.opt_Cat(opt_cat_1)
        cfp_1 = self.cfp1(opt_cat_1_0)
        cfp_2 = self.cfp2(cfp_1)

        injection_2 = self.injection_2(injection_1)
        opt_cat_2 = torch.cat((cfp_2, opt_cat_1_0, injection_2), dim=1)

        opt_cat_2_0 = self.opt_Cat2(opt_cat_2)
        cfp_3 = self.cfp3(opt_cat_2_0)
        cfp_4 = self.cfp4(cfp_3)
        cfp_5 = self.cfp5(cfp_4)
        cfp_6 = self.cfp6(cfp_5)
        cfp_7 = self.cfp7(cfp_6)
        cfp_8 = self.cfp8(cfp_7)

        injection_3 = self.injection_3(injection_2)
        opt_cat_3 = torch.cat((cfp_8, opt_cat_2_0, injection_3), dim=1)

        # 上采样并连接
        up1_out = self.up1(opt_cat_3)
        up1_cat = torch.cat((up1_out, opt_cat_2), dim=1)

        up2_out = self.up2(up1_cat)
        up2_cat = torch.cat((up2_out, opt_cat_1), dim=1)

        up3_out = self.up3(up2_cat)

        # 最后的卷积层
        output = self.final_conv(up3_out)

        x_out = self.Conv_1x1(output)

        return F.softmax(x_out, 1)

        # return output




def debug():
    data = torch.rand([2, 4, 128, 128, 128])
    data = data.to('cuda:4')
    model = CFPNetM(channels=4)
    model.to('cuda:4')
    out = model(data)
    print(out.size())


if __name__ == '__main__':
    debug()