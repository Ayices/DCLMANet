import torch
import torch.nn as nn
# from Attention.Attention_compare_3D import Attention_3D_unique as AMF2M
# from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D, DWT_3D_tiny
from .Attention.Attention_compare_3D import Attention_3D_unique as AMF2M
from .DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D, DWT_3D_tiny

class Cov3x3_BN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1,
                 padding = 1, dilation = 1, groups = 1, bias = True,
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
    def __init__(self, in_channels=1, num_class = 2, with_BN = True, channel_width = 8, wavename = 'haar', la = 0.25):
        super(basicNet, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = in_channels, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D(wavename = wavename)
        self.threshold_1 = Threshold_HFC(la = la)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width * 8, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D(wavename = wavename)
        self.threshold_2 = Threshold_HFC(la = la)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width * 8, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D(wavename = wavename)
        self.threshold_3 = Threshold_HFC(la = la)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width * 8, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D(wavename = wavename)
        self.threshold_4 = Threshold_HFC(la = la)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width * 8, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
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
        # 1, W, H, L
        output_1_D = self.cov3d_12_en(self.cov3d_11_en(input))
        # 8, W, H, L
        LLL_1, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.downsampling_1(output_1_D)
        LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.threshold_1(LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        # 8, W/2, H/2, L/2
        output_1 = torch.cat((LLL_1, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1), 1)
        # 64, W/2, H/2, L/2

        output_2_D = self.cov3d_22_en(self.cov3d_21_en(output_1))
        # 16,  W/2, H/2, L/2
        LLL_2, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.downsampling_2(output_2_D)
        LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.threshold_2(LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        # 16,  W/4, H/4, L/4
        output_2 = torch.cat((LLL_2, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2), 1)
        # 128, W/4, H/4, L/4

        output_3_D = self.cov3d_32_en(self.cov3d_31_en(output_2))
        # 32, W/4, H/4, L/4
        LLL_3, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.downsampling_3(output_3_D)
        LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.threshold_3(LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        # 32, W/8, H/8, L/8
        output_3 = torch.cat((LLL_3, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3), 1)
        # 256, W/8, H/8, L/8

        output_4_D = self.cov3d_42_en(self.cov3d_41_en(output_3))
        # 64, W/8, H/8, L/8
        LLL_4, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.downsampling_4(output_4_D)
        LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.threshold_4(LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        # 64, W/16, H/16, L/16
        output_4 = torch.cat((LLL_4, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4), 1)
        # 512, W/16, H/16, L/16

        output_5_D = self.cov3d_52(self.cov3d_51(output_4))
        # 128, W/16, H/16, L/16

        output_4_U = self.upsampling_4(output_5_D, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        output_4_U_C = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output_4_U, output_4_D), dim = 1)))
        # 32, W/8, H/8, L/8

        output_3_U = self.upsampling_3(output_4_U_C, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        output_3_U_C = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output_3_U, output_3_D), dim = 1)))

        output_2_U  = self.upsampling_2(output_3_U_C, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        output_2_U_C = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output_2_U, output_2_D), dim = 1)))

        output_1_U = self.upsampling_1(output_2_U_C, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        output_1_U_C = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output_1_U, output_1_D), dim = 1)))

        output = self.cov_final(output_1_U_C)

        out_put = []
        out_put.append(output_1_D)
        out_put.append(output_2_D)
        out_put.append(output_3_D)
        out_put.append(output_4_D)
        out_put.append(output_5_D)

        return out_put, self.Softmax_layer(output)


class RAFSNet(torch.nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_class=2,
                 channel_width=8,
                 multimodal_num=2,
                 Attention_type='SE',
                 wavename='haar',
                 la=0.25,
                 with_BN=True):

        super(RAFSNet, self).__init__()
        self.multimodal_num = multimodal_num
        self.basicNets = nn.ModuleList([])
        self.fisionblocks = nn.ModuleList([])
        for basicNet_num in range(self.multimodal_num):
            self.basicNets.append(basicNet(in_channels=in_channels,
                                           num_class=num_class,
                                           channel_width=channel_width,
                                           wavename=wavename,
                                           la=la,
                                           with_BN=with_BN))

        in_dim = [1*channel_width, 2*channel_width, 4*channel_width, 8*channel_width, 8*channel_width]
        min_kernel_size = 5
        kernel_size = [5*int(min_kernel_size), 4*int(min_kernel_size), 3*int(min_kernel_size),
                       2*int(min_kernel_size), 1*int(min_kernel_size)]
        multimodal_num_list = [multimodal_num, multimodal_num+1, multimodal_num+1, multimodal_num+1, multimodal_num+1]

        # 解码器一共有五层
        for basiclayer in range(5):
            self.fisionblocks.append(AMF2M(
                                            channel=in_dim[basiclayer],
                                            reduction_ca_Attention=4,  # 外层自适应通道注意力融合的reduction
                                            fusion_num=multimodal_num_list[basiclayer],
                                            kernel_size=kernel_size[basiclayer],
                                            reduction_ca=4,
                                            Attention_type=Attention_type
                                            )
                                    )
        # 32 * 128 * 128
        self.cov3d_11_en = Cov3x3_BN(in_channels = in_channels, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_12_en = Cov3x3_BN(in_channels = 1 * channel_width, out_channels = 1 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_1 = DWT_3D(wavename = wavename)
        self.threshold_1 = Threshold_HFC(la = la)
        # 16 * 64 * 64
        self.cov3d_21_en = Cov3x3_BN(in_channels = 1 * channel_width * 8, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_22_en = Cov3x3_BN(in_channels = 2 * channel_width, out_channels = 2 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_2 = DWT_3D(wavename = wavename)
        self.threshold_2 = Threshold_HFC(la = la)
        # 8 * 32 * 32
        self.cov3d_31_en = Cov3x3_BN(in_channels = 2 * channel_width * 8, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_32_en = Cov3x3_BN(in_channels = 4 * channel_width, out_channels = 4 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_3 = DWT_3D(wavename = wavename)
        self.threshold_3 = Threshold_HFC(la = la)
        #  4 * 16 * 16
        self.cov3d_41_en = Cov3x3_BN(in_channels = 4 * channel_width * 8, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.cov3d_42_en = Cov3x3_BN(in_channels = 8 * channel_width, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
        self.downsampling_4 = DWT_3D(wavename = wavename)
        self.threshold_4 = Threshold_HFC(la = la)

        # 2 * 8 * 8
        self.cov3d_51 = Cov3x3_BN(in_channels = 8 * channel_width * 8, out_channels = 8 * channel_width, kernel_size = 3, padding = 1, with_BN = with_BN)
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

    def forward(self, in_list):
        b, c, w, h, l = in_list.size()
        basic_features_list1 = []
        basic_features_list2 = []
        basic_features_list3 = []
        basic_features_list4 = []
        basic_features_list5 = []
        basic_out_list = []
        for basicNet_num in range(self.multimodal_num):
            basic_features, basic_out = self.basicNets[basicNet_num](in_list[:, basicNet_num, :, :, :].view(b, 1, w, h, l))
            basic_features_list1.append(basic_features[0])
            basic_features_list2.append(basic_features[1])
            basic_features_list3.append(basic_features[2])
            basic_features_list4.append(basic_features[3])
            basic_features_list5.append(basic_features[4])
            basic_out_list.append(basic_out)

        output_fusion_1_D = self.fisionblocks[0](torch.stack(basic_features_list1, 0))
        # 8, W, H, L
        LLL_1, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.downsampling_1(output_fusion_1_D)
        LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1 = self.threshold_1(LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        # 8, W/2, H/2, L/2
        output_1 = torch.cat((LLL_1, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1), 1)
        # 64, W/2, H/2, L/2

        output_2_D = self.cov3d_22_en(self.cov3d_21_en(output_1))
        # 16,  W/2, H/2, L/2
        output_2_D_B = torch.cat((torch.stack(basic_features_list2), output_2_D.unsqueeze(0)), 0)
        output_fusion_2_D = self.fisionblocks[1](output_2_D_B)

        LLL_2, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.downsampling_2(output_fusion_2_D)
        LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2 = self.threshold_2(LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        # 16,  W/4, H/4, L/4
        output_2 = torch.cat((LLL_2, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2), 1)
        # 128, W/4, H/4, L/4

        output_3_D = self.cov3d_32_en(self.cov3d_31_en(output_2))

        output_3_D_B = torch.cat((torch.stack(basic_features_list3), output_3_D.unsqueeze(0)), 0)
        output_fusion_3_D = self.fisionblocks[2](output_3_D_B)

        # 32, W/4, H/4, L/4
        LLL_3, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.downsampling_3(output_fusion_3_D)
        LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3 = self.threshold_3(LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        # 32, W/8, H/8, L/8
        output_3 = torch.cat((LLL_3, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3), 1)
        # 256, W/8, H/8, L/8

        output_4_D = self.cov3d_42_en(self.cov3d_41_en(output_3))
        output_4_D_B = torch.cat((torch.stack(basic_features_list4), output_4_D.unsqueeze(0)), 0)
        output_fusion_4_D = self.fisionblocks[3](output_4_D_B)

        # 64, W/8, H/8, L/8
        LLL_4, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.downsampling_4(output_fusion_4_D)
        LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4 = self.threshold_4(LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        # 64, W/16, H/16, L/16
        output_4 = torch.cat((LLL_4, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4), 1)
        # 512, W/16, H/16, L/16

        output_5_D = self.cov3d_52(self.cov3d_51(output_4))
        output_5_D_B = torch.cat((torch.stack(basic_features_list5), output_5_D.unsqueeze(0)), 0)
        output_fusion_5_D = self.fisionblocks[4](output_5_D_B)
        # 128, W/16, H/16, L/16

        output_4_U = self.upsampling_4(output_fusion_5_D, LLH_4, LHL_4, LHH_4, HLL_4, HLH_4, HHL_4, HHH_4)
        output_4_U_C = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output_4_U, output_fusion_4_D), dim = 1)))
        # 32, W/8, H/8, L/8

        output_3_U = self.upsampling_3(output_4_U_C, LLH_3, LHL_3, LHH_3, HLL_3, HLH_3, HHL_3, HHH_3)
        output_3_U_C = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output_3_U, output_fusion_3_D), dim = 1)))

        output_2_U  = self.upsampling_2(output_3_U_C, LLH_2, LHL_2, LHH_2, HLL_2, HLH_2, HHL_2, HHH_2)
        output_2_U_C = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output_2_U, output_fusion_2_D), dim = 1)))

        output = self.upsampling_1(output_2_U_C, LLH_1, LHL_1, LHH_1, HLL_1, HLH_1, HHL_1, HHH_1)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_fusion_1_D), dim = 1)))

        output = self.cov_final(output)
        return basic_out_list, self.Softmax_layer(output)



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


if __name__ == '__main__':
    """
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 3D DWT、解码器上采样使用 3D IDWT, 高频分量经过滤波处理
    """
    from datetime import datetime

    # input = torch.rand(size=(2, 1, 64, 64, 64)).float().cuda()
    # models = basicNet(in_channels=1, num_class=4, channel_width=8).cuda()
    # print('====================================')
    # start = datetime.now()
    # output = models(input)
    # print('Neuron_WaveSNet_V4')
    # print(output.size())
    # stop = datetime.now()
    # print('tooking {} secs'.format(stop - start))
    # print('====================================')

    input = torch.rand(size=(2, 2, 96, 96, 96)).float().cuda()
    model = RAFSNet(in_channels=1, num_class=1, channel_width=8, multimodal_num=2).cuda()
    print('====================================')
    start = datetime.now()
    basic_out_list, output = model(input)
    print('Neuron_WaveSNet_V4')
    print(output.size())
    stop = datetime.now()
    print('tooking {} secs'.format(stop - start))
    print('====================================')
