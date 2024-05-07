import torch
import torch.nn as nn
from .Attention.Attention_compare_2D import Attention_2D_unique as AMF2M
from .DWT_IDWT_layer import DWT_3D, IDWT_3D, DWT_3D_tiny

class Cov3x3_BN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1,
                 padding = 1, dilation = 1, groups = 8, bias = True,
                 padding_mode = 'zeros', with_BN = True):
        super(Cov3x3_BN, self).__init__()
        self.with_BN = with_BN
        self.cov = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride, padding = padding, dilation = dilation, groups = groups, bias = bias)
        self.bn = nn.BatchNorm3d(num_features = out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        if self.with_BN:
            return self.relu(self.bn(self.cov(input)))
        else:
            return self.relu(self.cov(input))

class Threshold_HFC(torch.nn.Module):
    def __init__(self, la = 0.25):
        super(Threshold_HFC, self).__init__()
        self.threshold = nn.Hardshrink(lambd = la)
    def forward(self, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        return self.threshold(LLH), self.threshold(LHL), self.threshold(LHH), \
               self.threshold(HLL), self.threshold(HLH), self.threshold(HHL), self.threshold(HHH)

class basicNet(torch.nn.Module):
    """
    构建用于神经元图像分割的 3D UNet 对于每个模态单独使用
    它的编码器下采样使用 3D DWT、解码器上采样使用 3D IDWT, 高频分量经过滤波处理
    """
    def __init__(self, in_channels=1, num_class = 2, with_BN = True, channel_width = 4, wavename = 'haar', la = 0.25):
        super(basicNet, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = in_channels, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D(wavename = wavename)
        self.threshold_1 = Threshold_HFC(la = la)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D(wavename = wavename)
        self.threshold_2 = Threshold_HFC(la = la)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D(wavename = wavename)
        self.threshold_3 = Threshold_HFC(la = la)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D(wavename = wavename)
        self.threshold_4 = Threshold_HFC(la = la)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_52 = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.upsampling_4 = IDWT_3D(wavename = wavename)
        # 4 * 16 * 16
        self.cov3d_41_de = Cov3x3_BN(in_channels = 16 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_3 = IDWT_3D(wavename = wavename)
        # 8 * 32 * 32
        self.cov3d_31_de = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_2 = IDWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_21_de = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.upsampling_1 = IDWT_3D(wavename = wavename)
        # 16 * 64 * 64
        self.cov3d_11_de = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_de = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

        if num_class == 1:
            self.Softmax_layer = nn.Sigmoid()
        else:
            self.Softmax_layer = nn.Softmax(dim=1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        output, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.downsampling_1(output_1)
        LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.threshold_1(LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        output, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.downsampling_2(output_2)
        LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.threshold_2(LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        output, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.downsampling_3(output_3)
        LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.threshold_3(LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        output, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.downsampling_4(output_4)
        LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.threshold_4(LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(output, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4), dim = 1)))

        output = self.upsampling_3(output, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3), dim = 1)))

        output = self.upsampling_2(output, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2), dim = 1)))

        output = self.upsampling_1(output, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1), dim = 1)))

        output = self.cov_final(output)
        return self.Softmax_layer(output)


def up(x):
    return nn.functional.interpolate(x, scale_factor=2)


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv_decod_block(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        act_fn,
    )
    return model


class MixedFusion_Block(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )

        # revised in 09/09/2019.
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim,  kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, x1, x2, xx):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(torch.cat((out1, xx), dim=1))

        return out2


class MixedFusion_Block0(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn):
        super(MixedFusion_Block0, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(in_dim), act_fn, )
        # self.layer1 = nn.Sequential(nn.Conv2d(in_dim*3, in_dim, kernel_size=1),nn.BatchNorm2d(in_dim),act_fn,)
        self.layer2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(out_dim), act_fn, )

    def forward(self, x1, x2):
        # multi-style fusion
        fusion_sum = torch.add(x1, x2)  # sum
        fusion_mul = torch.mul(x1, x2)

        modal_in1 = torch.reshape(x1, [x1.shape[0], 1, x1.shape[1], x1.shape[2], x1.shape[3]])
        modal_in2 = torch.reshape(x2, [x2.shape[0], 1, x2.shape[1], x2.shape[2], x2.shape[3]])
        modal_cat = torch.cat((modal_in1, modal_in2), dim=1)
        fusion_max = modal_cat.max(dim=1)[0]

        out_fusion = torch.cat((fusion_sum, fusion_mul, fusion_max), dim=1)

        out1 = self.layer1(out_fusion)
        out2 = self.layer2(out1)
        return out2


class FAModule(nn.Module):
    def __init__(self, in_dim):
        super(FAModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch1 = nn.Sequential(nn.Conv2d(in_dim, int(in_dim / 4), 1), self.relu)
        self.conv_branch2 = nn.Sequential(nn.Conv2d(in_dim, int(in_dim / 2), 1), self.relu,
                                          nn.Conv2d(int(in_dim / 2), int(in_dim / 4), 3, 1, 1), self.relu)
        self.conv_branch3 = nn.Sequential(nn.Conv2d(in_dim, int(in_dim / 4), 1), self.relu,
                                          nn.Conv2d(int(in_dim / 4), int(in_dim / 4), 5, 1, 2), self.relu)
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(in_dim, int(in_dim / 4), 1), self.relu)

    def forward(self, x):
        # aggregation
        x_branch1 = self.conv_branch1(x)
        x_branch2 = self.conv_branch2(x)
        x_branch3 = self.conv_branch3(x)
        x_branch4 = self.conv_branch4(x)

        x = torch.cat((x_branch1, x_branch2, x_branch3, x_branch4), dim=1)
        return x

class pre_layer(nn.Module):
    def __init__(self, filters_in, filters_out, act_fn):
        super(pre_layer, self).__init__()

        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=filters_in, out_channels=filters_in,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(filters_in),
        )

        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=filters_in, out_channels=filters_out,
                      kernel_size=3,
                      padding=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(filters_out),
        )

        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=filters_out, out_channels=filters_out,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(filters_out),
            act_fn,
        )

    def forward(self, x):
        conv1 = self.conv1_block(x)
        conv2 = self.conv2_block(conv1)
        conv3 = self.conv3_block(conv2)
        return conv3


class AMF2M_Block(nn.Module):

    def __init__(self, in_dim, out_dim, act_fn, img_size=[4, 4]):
        super(AMF2M_Block, self).__init__()

        fusion_num = 2
        kernel_size = 5
        reduction_ca = 4
        reduction_ca_Attention = 4  # 外层自适应通道注意力融合的reduction
        reduction_sa_Attention = 4  # 外层自适应空间注意力融合的reduction
        Attention_type = 'CBAM'

        self.fisionblocks = AMF2M(channel=in_dim, img_size=img_size,
                                  reduction_ca_Attention=reduction_ca_Attention,  # 外层自适应通道注意力融合的reduction
                                  reduction_sa_Attention=reduction_sa_Attention,  # 外层自适应空间注意力融合的reduction
                                  fusion_num=fusion_num,
                                  kernel_size=kernel_size,
                                  reduction_ca=reduction_ca,
                                  Attention_type=Attention_type)

        self.FAModule = FAModule(in_dim)
        # self.layer = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(out_dim), act_fn, )
        self.layer = pre_layer(in_dim * 2, out_dim, act_fn)


    def forward(self, x1, x2, xx):
        input = []
        input.append(x1)
        input.append(x2)
        input = torch.stack(input, 0)
        out_fusion = self.fisionblocks(input)
        out_fusion_FA = self.FAModule(out_fusion)
        out = self.layer(torch.cat((out_fusion_FA, xx), dim=1))
        return out


class AMF2M_Block0(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, img_size=[4, 4]):
        super(AMF2M_Block0, self).__init__()
        fusion_num = 2
        kernel_size = 5
        reduction_ca = 4
        reduction_ca_Attention = 4  # 外层自适应通道注意力融合的reduction
        reduction_sa_Attention = 4  # 外层自适应空间注意力融合的reduction
        Attention_type = 'CBAM'

        self.fisionblocks = AMF2M(channel=in_dim, img_size=img_size,
                                  reduction_ca_Attention=reduction_ca_Attention,  # 外层自适应通道注意力融合的reduction
                                  reduction_sa_Attention=reduction_sa_Attention,  # 外层自适应空间注意力融合的reduction
                                  fusion_num=fusion_num,
                                  kernel_size=kernel_size,
                                  reduction_ca=reduction_ca,
                                  Attention_type=Attention_type)
        self.FAModule = FAModule(in_dim)
        # self.layer = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        #                            nn.BatchNorm2d(out_dim), act_fn, )
        self.layer = pre_layer(in_dim, out_dim, act_fn)

    def forward(self, x1, x2):
        input = []
        input.append(x1)
        input.append(x2)
        input = torch.stack(input, 0)
        out_fusion = self.fisionblocks(input)
        out_fusion_FA = self.FAModule(out_fusion)
        out = self.layer(out_fusion_FA)
        return out


##############################################
# define our models
class Multi_modal_generator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super(Multi_modal_generator, self).__init__()

        self.in_dim = input_nc
        self.out_dim = ngf
        self.final_out_dim = output_nc

        act_fn = nn.LeakyReLU(0.2, inplace=True)
        # act_fn = nn.ReLU()

        act_fn2 = nn.ReLU(inplace=True)  # nn.ReLU()

        # ~~~ Encoding Paths ~~~~~~ #
        # Encoder (Modality 1)

        #######################################################################
        # Encoder **Modality 1
        #######################################################################
        self.down_1_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
        )
        self.pool_1_0 = maxpool()

        self.down_2_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
        )
        self.pool_2_0 = maxpool()

        self.down_3_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
        )
        self.pool_3_0 = maxpool()

        #######################################################################
        # Encoder **Modality 2
        #######################################################################
        self.down_1_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim), act_fn,
        )
        self.pool_1_1 = maxpool()

        self.down_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 2), act_fn,
        )
        self.pool_2_1 = maxpool()

        self.down_3_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 2, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 4), act_fn,
        )
        self.pool_3_1 = maxpool()

        #######################################################################
        # fusion layer
        #######################################################################
        # down 1st layer
        self.down_fu_1 = AMF2M_Block0(self.out_dim, self.out_dim * 2, act_fn, img_size=[120, 120])
        self.pool_fu_1 = maxpool()

        self.down_fu_2 = AMF2M_Block(self.out_dim * 2, self.out_dim * 4, act_fn, img_size=[60, 60])
        self.pool_fu_2 = maxpool()

        self.down_fu_3 = AMF2M_Block(self.out_dim * 4, self.out_dim * 4, act_fn, img_size=[30, 30])
        self.pool_fu_3 = maxpool()

        # down 4th layer
        self.down_fu_4 = nn.Sequential(
            nn.Conv2d(in_channels=self.out_dim * 4, out_channels=self.out_dim * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_dim * 8), act_fn, )

        # ~~~ Decoding Path ~~~~~~ #
        self.deconv_1_0 = conv_decod_block(self.out_dim * 8, self.out_dim * 4, act_fn2)
        self.deconv_2_0 = AMF2M_Block(self.out_dim * 4, self.out_dim * 2, act_fn2, img_size=[30, 30])
        self.deconv_3_0 = AMF2M_Block(self.out_dim * 2, self.out_dim * 1, act_fn2, img_size=[60, 60])
        self.deconv_4_0 = AMF2M_Block(self.out_dim * 1, self.out_dim, act_fn2, img_size=[120, 120])
        self.deconv_5_0 = conv_decod_block(self.out_dim * 1, self.out_dim, act_fn2)
        self.out        = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim
        # self.out = nn.Sequential(nn.Conv2d(int(self.out_dim), 1, kernel_size=3, stride=1, padding=1),
        #                          nn.Sigmoid())  # self.final_out_dim

        # Modality 1
        self.deconv_1_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_2_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_3_1 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_4_1 = conv_decod_block(self.out_dim * 2, self.out_dim * 2, act_fn2)
        self.deconv_5_1 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_6_1 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out1       = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim
        # self.out1 = nn.Sequential(nn.Conv2d(int(self.out_dim), 1, kernel_size=3, stride=1, padding=1),
        #                           nn.Sigmoid())  # self.final_out_dim
        # modality 2
        self.deconv_1_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_2_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 4, act_fn2)
        self.deconv_3_2 = conv_decod_block(self.out_dim * 4, self.out_dim * 2, act_fn2)
        self.deconv_4_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 2, act_fn2)
        self.deconv_5_2 = conv_decod_block(self.out_dim * 2, self.out_dim * 1, act_fn2)
        self.deconv_6_2 = conv_decod_block(self.out_dim * 1, int(self.out_dim), act_fn2)
        self.out2       = nn.Sequential(nn.Conv2d(int(self.out_dim),1, kernel_size=3, stride=1, padding=1),nn.Tanh()) #  self.final_out_dim
        # self.out2 = nn.Sequential(nn.Conv2d(int(self.out_dim), 1, kernel_size=3, stride=1, padding=1),
        #                           nn.Sigmoid())  # self.final_out_dim

    def forward(self, inputs):
        # ############################# #
        i0 = inputs[:, 0:1, :, :]
        i1 = inputs[:, 1:2, :, :]

        # -----  First Level --------
        down_1_0 = self.down_1_0(i0)
        down_1_1 = self.down_1_1(i1)

        # -----  Second Level --------
        # input_2nd = torch.cat((down_1_0,down_1_1,down_1_2,down_1_3),dim=1)
        # Max-pool
        down_1_0m = self.pool_1_0(down_1_0)
        down_1_1m = self.pool_1_1(down_1_1)

        down_2_0 = self.down_2_0(down_1_0m)
        down_2_1 = self.down_2_1(down_1_1m)

        # -----  Third Level --------
        # Max-pool
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_1(down_2_1)

        down_3_0 = self.down_3_0(down_2_0m)
        down_3_1 = self.down_3_1(down_2_1m)

        # Max-pool
        down_3_0m = self.pool_3_0(down_3_0)
        down_3_1m = self.pool_3_1(down_3_1)

        # ----------------------------------------
        # fusion layer
        down_fu_1 = self.down_fu_1(down_1_0m, down_1_1m)
        down_fu_1m = self.pool_fu_1(down_fu_1)

        down_fu_2 = self.down_fu_2(down_2_0m, down_2_1m, down_fu_1m)
        down_fu_2m = self.pool_fu_2(down_fu_2)

        down_fu_3 = self.down_fu_3(down_3_0m, down_3_1m, down_fu_2m)
        down_fu_4 = self.down_fu_4(down_fu_3)

        # latents     = self.down_fu_4(output_atten)

        #######################################################################
        # ~~~~~~ Decoding
        deconv_1_0 = self.deconv_1_0(down_fu_4)
        deconv_2_0 = self.deconv_2_0(down_3_0m, down_3_1m, deconv_1_0)
        deconv_3_0 = self.deconv_3_0(down_2_0m, down_2_1m, up(deconv_2_0))
        deconv_4_0 = self.deconv_4_0(down_1_0m, down_1_1m, up(deconv_3_0))
        deconv_5_0 = self.deconv_5_0(up(deconv_4_0))
        output = self.out(deconv_5_0)

        # modality 1
        deconv_1_1 = self.deconv_1_1((down_3_0m))
        deconv_2_1 = self.deconv_2_1(up(deconv_1_1))
        deconv_3_1 = self.deconv_3_1((deconv_2_1))
        deconv_4_1 = self.deconv_4_1(up(deconv_3_1))
        deconv_5_1 = self.deconv_5_1((deconv_4_1))
        deconv_6_1 = self.deconv_6_1(up(deconv_5_1))
        output1 = self.out(deconv_6_1)

        # modality 2
        deconv_1_2 = self.deconv_1_2((down_3_1m))
        deconv_2_2 = self.deconv_2_2(up(deconv_1_2))
        deconv_3_2 = self.deconv_3_2((deconv_2_2))
        deconv_4_2 = self.deconv_4_2(up(deconv_3_2))
        deconv_5_2 = self.deconv_5_2((deconv_4_2))
        deconv_6_2 = self.deconv_6_2(up(deconv_5_2))
        output2 = self.out(deconv_6_2)

        return output, output1, output2