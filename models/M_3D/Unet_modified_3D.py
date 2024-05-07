import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class supplement_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(supplement_conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv3d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm3d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class Attention_block_supplement(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_supplement, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm3d(F_int)
            nn.InstanceNorm3d(F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm3d(F_int)
            nn.InstanceNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm3d(1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm3d(F_int)
            nn.InstanceNorm3d(F_int),
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm3d(F_int)
            nn.InstanceNorm3d(F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm3d(1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(U_Net, self).__init__()

        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=base_channel)
        self.Conv2 = conv_block(ch_in=base_channel, ch_out=base_channel*2)
        self.Conv3 = conv_block(ch_in=base_channel*2, ch_out=base_channel*4)
        self.Conv4 = conv_block(ch_in=base_channel*4, ch_out=base_channel*8)
        self.Conv5 = conv_block(ch_in=base_channel*8, ch_out=base_channel*16)

        self.Up5 = up_conv(ch_in=base_channel*16, ch_out=base_channel*8)
        self.Up_conv5 = conv_block(ch_in=base_channel*16, ch_out=base_channel*8)

        self.Up4 = up_conv(ch_in=base_channel*8, ch_out=base_channel*4)
        self.Up_conv4 = conv_block(ch_in=base_channel*8, ch_out=base_channel*4)

        self.Up3 = up_conv(ch_in=base_channel*4, ch_out=base_channel*2)
        self.Up_conv3 = conv_block(ch_in=base_channel*4, ch_out=base_channel*2)

        self.Up2 = up_conv(ch_in=base_channel*2, ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=base_channel*2, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.active(d1)

class U_Net_basechannel(nn.Module):
    def __init__(self,img_ch=4, output_ch=4, base_channel=64):
        super(U_Net_basechannel,self).__init__()
        
        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=base_channel)
        self.Conv2 = conv_block(ch_in=base_channel,ch_out=2*base_channel)
        self.Conv3 = conv_block(ch_in=2*base_channel,ch_out=4*base_channel)
        self.Conv4 = conv_block(ch_in=4*base_channel,ch_out=8*base_channel)
        self.Conv5 = conv_block(ch_in=8*base_channel,ch_out=16*base_channel)

        self.Up5 = up_conv(ch_in=16*base_channel,ch_out=8*base_channel)
        self.Up_conv5 = conv_block(ch_in=16*base_channel, ch_out=8*base_channel)

        self.Up4 = up_conv(ch_in=8*base_channel,ch_out=4*base_channel)
        self.Up_conv4 = conv_block(ch_in=8*base_channel, ch_out=4*base_channel)
        
        self.Up3 = up_conv(ch_in=4*base_channel,ch_out=2*base_channel)
        self.Up_conv3 = conv_block(ch_in=4*base_channel, ch_out=2*base_channel)
        
        self.Up2 = up_conv(ch_in=2*base_channel,ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=2*base_channel, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = nn.Sigmoid()
        self.active = nn.Softmax(dim=1)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.active(d1)


class ME_Net(nn.Module):
    def __init__(self, img_ch=4, output_ch=4, base_channel=64):
        super(ME_Net, self).__init__()

        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=base_channel)
        self.Conv2 = conv_block(ch_in=base_channel, ch_out=2 * base_channel)
        self.Conv3 = conv_block(ch_in=2 * base_channel, ch_out=4 * base_channel)
        self.Conv4 = conv_block(ch_in=4 * base_channel, ch_out=8 * base_channel)
        self.Conv5 = conv_block(ch_in=8 * base_channel, ch_out=16 * base_channel)

        self.Up5 = up_conv(ch_in=16 * base_channel, ch_out=8 * base_channel)
        self.Up_conv5 = conv_block(ch_in=16 * base_channel, ch_out=8 * base_channel)

        self.Up4 = up_conv(ch_in=8 * base_channel, ch_out=4 * base_channel)
        self.Up_conv4 = conv_block(ch_in=8 * base_channel, ch_out=4 * base_channel)

        self.Up3 = up_conv(ch_in=4 * base_channel, ch_out=2 * base_channel)
        self.Up_conv3 = conv_block(ch_in=4 * base_channel, ch_out=2 * base_channel)

        self.Up2 = up_conv(ch_in=2 * base_channel, ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=2 * base_channel, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = nn.Sigmoid()
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.active(d1)

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, depth)
        out = self.gamma*out + x
        return out

class PAM_Module(nn.Module):
    """ Depth attention module"""
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, depth)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, depth)
        out = self.gamma * out + x
        return out

class Neuron_UNet_My(nn.Module):
    """
    UCP_Net
    构建用于神经元图像分割的 3D UNet
    它的编码器下采样使用 max-pooling、解码器上采样使用 反卷积
    """
    def __init__(self, in_channels = 4, num_class = 4,channel_width = 6):
        super(Neuron_UNet_My, self).__init__()
        # 32 * 128 * 128
        self.cov3d_11_en = conv_block(ch_in = in_channels, ch_out = 1 * channel_width)
        self.cov3d_12_en = conv_block(ch_in = 1 * channel_width, ch_out = 1 * channel_width)
        self.cov3d_13_en = conv_block(ch_in=1 * channel_width, ch_out=1 * channel_width)
        self.cam1 = CAM_Module(1 * channel_width)
        self.dam1 = PAM_Module(1 * channel_width)
        self.downsampling_1 = nn.MaxPool3d(kernel_size = 2)
        # 16 * 64 * 64
        self.cov3d_21_en = conv_block(ch_in = 1 * channel_width, ch_out = 2 * channel_width)
        self.cov3d_22_en = conv_block(ch_in = 2 * channel_width, ch_out = 2 * channel_width)
        self.cov3d_23_en = conv_block(ch_in = 2 * channel_width, ch_out = 2 * channel_width)
        self.cam2 = CAM_Module(2 * channel_width)
        self.dam2 = PAM_Module(2 * channel_width)
        self.downsampling_2 = nn.MaxPool3d(kernel_size = 2)
        # 8 * 32 * 32
        self.cov3d_31_en = conv_block(ch_in = 2 * channel_width, ch_out = 4 * channel_width)
        self.cov3d_32_en = conv_block(ch_in = 4 * channel_width, ch_out = 4 * channel_width)
        self.cov3d_33_en = conv_block(ch_in = 4 * channel_width, ch_out = 4 * channel_width)
        self.cam3 = CAM_Module(4 * channel_width)
        self.dam3 = PAM_Module(4 * channel_width)
        self.downsampling_3 = nn.MaxPool3d(kernel_size = 2)
        #  4 * 16 * 16
        self.cov3d_41_en = conv_block(ch_in = 4 * channel_width, ch_out = 8 * channel_width)
        self.cov3d_42_en = conv_block(ch_in = 8 * channel_width, ch_out = 8 * channel_width)
        self.cov3d_43_en = conv_block(ch_in = 8 * channel_width, ch_out = 8 * channel_width)
        self.cam4 = CAM_Module(8 * channel_width)
        self.dam4 = PAM_Module(8 * channel_width)
        self.downsampling_4 = nn.MaxPool3d(kernel_size = 2)

        # 2 * 8 * 8
        self.cov3d_51 = conv_block(ch_in = 8 * channel_width, ch_out = 8 * channel_width)
        self.cov3d_52 = conv_block(ch_in = 8 * channel_width, ch_out = 8 * channel_width)

        self.upsampling_4 = nn.ConvTranspose3d(in_channels = 8 * channel_width, out_channels = 8 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 4 * 16 * 16
        self.cov3d_41_de = conv_block(ch_in = 16 * channel_width, ch_out = 8 * channel_width)
        self.cov3d_42_de = conv_block(ch_in = 8 * channel_width, ch_out = 4 * channel_width)
        self.upsampling_3 = nn.ConvTranspose3d(in_channels = 4 * channel_width, out_channels = 4 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 8 * 32 * 32
        self.cov3d_31_de = conv_block(ch_in = 8 * channel_width, ch_out = 4 * channel_width)
        self.cov3d_32_de = conv_block(ch_in = 4 * channel_width, ch_out = 2 * channel_width)
        self.upsampling_2 = nn.ConvTranspose3d(in_channels = 2 * channel_width, out_channels = 2 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 16 * 64 * 64
        self.cov3d_21_de = conv_block(ch_in = 4 * channel_width, ch_out = 2 * channel_width)
        self.cov3d_22_de = conv_block(ch_in = 2 * channel_width, ch_out = 1 * channel_width)
        self.upsampling_1 = nn.ConvTranspose3d(in_channels = 1 * channel_width, out_channels = 1 * channel_width,
                                               kernel_size = 1, stride = 2, output_padding = 1, padding = 0)
        # 16 * 64 * 64
        self.cov3d_11_de = conv_block(ch_in = 2 * channel_width, ch_out = 1 * channel_width)
        self.cov3d_12_de = conv_block(ch_in = 1 * channel_width, ch_out = 1 * channel_width)

        self.cov_final = nn.Conv3d(in_channels = 1 * channel_width, out_channels = num_class, kernel_size = 1)

        if num_class == 1:
            self.Softmax_layer = nn.Sigmoid()
        else:
            self.Softmax_layer = nn.Softmax(dim=1)

    def forward(self, input):
        output_1 = self.cov3d_12_en(self.cov3d_11_en(input))
        # output_1a = self.cam1(output_1)
        output_1a = self.cam1(output_1) + self.dam1(output_1)
        output_1a = self.cov3d_13_en(output_1a)
        output = self.downsampling_1(output_1)

        output_2 = self.cov3d_22_en(self.cov3d_21_en(output))
        # output_2a = self.cam2(output_2)
        output_2a = self.cam2(output_2) + self.dam2(output_2)
        output_2a = self.cov3d_23_en(output_2a)
        output = self.downsampling_2(output_2)

        output_3 = self.cov3d_32_en(self.cov3d_31_en(output))
        # output_3a = self.cam3(output_3)
        output_3a = self.cam3(output_3) + self.dam3(output_3)
        output_3a = self.cov3d_33_en(output_3a)
        output = self.downsampling_3(output_3)

        output_4 = self.cov3d_42_en(self.cov3d_41_en(output))
        # output_4a = self.cam4(output_4)
        output_4a = self.cam4(output_4) + self.dam4(output_4)
        output_4a = self.cov3d_43_en(output_4a)
        output = self.downsampling_4(output_4)

        output = self.cov3d_52(self.cov3d_51(output))

        output = self.upsampling_4(input = output)
        output = self.cov3d_42_de(self.cov3d_41_de(torch.cat((output, output_4a), dim = 1)))

        output = self.upsampling_3(input = output)
        output = self.cov3d_32_de(self.cov3d_31_de(torch.cat((output, output_3a), dim = 1)))

        output = self.upsampling_2(input = output)
        output = self.cov3d_22_de(self.cov3d_21_de(torch.cat((output, output_2a), dim = 1)))

        output = self.upsampling_1(input = output)
        output = self.cov3d_12_de(self.cov3d_11_de(torch.cat((output, output_1a), dim = 1)))

        output = self.cov_final(output)

        return self.Softmax_layer(output)


class Fusion_model_basechannel(nn.Module):
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(Fusion_model_basechannel, self).__init__()

        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv1_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv1_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv1_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv1_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.Conv2_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv2_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv2_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv2_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv2_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.Conv3_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv3_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv3_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv3_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv3_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.Conv4_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv4_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv4_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv4_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv4_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.ConvFusion = nn.Conv3d(in_channels= 16 * base_channel, out_channels = img_ch, kernel_size=3, padding=1)
        self.softmaxFusion = nn.Softmax(dim=1)
        self.ConvReweightedFeatures = nn.Conv3d(in_channels=16*base_channel,out_channels=16*base_channel,kernel_size=1,padding=0)

        self.Up5 = up_conv(ch_in=16 * base_channel, ch_out=8 * base_channel)
        self.Up_conv5 = conv_block(ch_in=16 * base_channel, ch_out=8 * base_channel)

        self.Up4 = up_conv(ch_in=8 * base_channel, ch_out=4 * base_channel)
        self.Up_conv4 = conv_block(ch_in=8 * base_channel, ch_out=4 * base_channel)

        self.Up3 = up_conv(ch_in=4 * base_channel, ch_out=2 * base_channel)
        self.Up_conv3 = conv_block(ch_in=4 * base_channel, ch_out=2 * base_channel)

        self.Up2 = up_conv(ch_in=2 * base_channel, ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=2 * base_channel, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = nn.Sigmoid()
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        img1, img2, img3, img4 = x[:,0,:,:,:].unsqueeze(dim=1) , x[:,1,:,:,:].unsqueeze(dim=1), x[:,2,:,:,:].unsqueeze(dim=1), x[:,3,:,:,:].unsqueeze(dim=1)


        img1_1 = self.Conv1_1(img1)

        img1_2 = self.Maxpool(img1_1)
        img1_2 = self.Conv1_2(img1_2)

        img1_3 = self.Maxpool(img1_2)
        img1_3 = self.Conv1_3(img1_3)

        img1_4 = self.Maxpool(img1_3)
        img1_4 = self.Conv1_4(img1_4)

        img1_5 = self.Maxpool(img1_4)
        img1_5 = self.Conv1_5(img1_5)


        img2_1 = self.Conv2_1(img2)

        img2_2 = self.Maxpool(img2_1)
        img2_2 = self.Conv2_2(img2_2)

        img2_3 = self.Maxpool(img2_2)
        img2_3 = self.Conv2_3(img2_3)

        img2_4 = self.Maxpool(img2_3)
        img2_4 = self.Conv2_4(img2_4)

        img2_5 = self.Maxpool(img2_4)
        img2_5 = self.Conv2_5(img2_5)



        img3_1 = self.Conv3_1(img3)

        img3_2 = self.Maxpool(img3_1)
        img3_2 = self.Conv3_2(img3_2)

        img3_3 = self.Maxpool(img3_2)
        img3_3 = self.Conv3_3(img3_3)

        img3_4 = self.Maxpool(img3_3)
        img3_4 = self.Conv3_4(img3_4)

        img3_5 = self.Maxpool(img3_4)
        img3_5 = self.Conv3_5(img3_5)


        img4_1 = self.Conv4_1(img4)

        img4_2 = self.Maxpool(img4_1)
        img4_2 = self.Conv4_2(img4_2)

        img4_3 = self.Maxpool(img4_2)
        img4_3 = self.Conv4_3(img4_3)

        img4_4 = self.Maxpool(img4_3)
        img4_4 = self.Conv4_4(img4_4)

        img4_5 = self.Maxpool(img4_4)
        img4_5 = self.Conv4_5(img4_5)

        img_concat = torch.cat((img1_5,img2_5,img3_5,img4_5),dim=1)
        img_Attentionmap = self.softmaxFusion(self.ConvFusion(img_concat))

        img1_add_attention = img1_5 * img_Attentionmap[:, 0, :, :, :].unsqueeze(dim=1)
        img2_add_attention = img2_5 * img_Attentionmap[:, 1, :, :, :].unsqueeze(dim=1)
        img3_add_attention = img3_5 * img_Attentionmap[:, 2, :, :, :].unsqueeze(dim=1)
        img4_add_attention = img4_5 * img_Attentionmap[:, 3, :, :, :].unsqueeze(dim=1)

        img_concat_add_attention = torch.cat((img1_add_attention,img2_add_attention,img3_add_attention,img4_add_attention),dim=1)
        x5 = self.ConvReweightedFeatures(img_concat_add_attention)

        x4 = torch.cat((img1_4,img2_4,img3_4,img4_4), dim=1)
        x3 = torch.cat((img1_3, img2_3, img3_3, img4_3), dim=1)
        x2 = torch.cat((img1_2, img2_2, img3_2, img4_2), dim=1)
        x1 = torch.cat((img1_1,img2_1,img3_1,img4_1), dim=1)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.active(d1)

class FusionUCP_model_basechannel(nn.Module):
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(FusionUCP_model_basechannel, self).__init__()

        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv1_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv1_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv1_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv1_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.cam1_1 = CAM_Module(1 * base_channel // 4)
        self.dam1_1 = PAM_Module(1 * base_channel // 4)
        self.cam1_2 = CAM_Module(2 * base_channel // 4)
        self.dam1_2 = PAM_Module(2 * base_channel // 4)
        self.cam1_3 = CAM_Module(4 * base_channel // 4)
        self.dam1_3 = PAM_Module(4 * base_channel // 4)
        self.cam1_4 = CAM_Module(8 * base_channel // 4)
        self.dam1_4 = PAM_Module(8 * base_channel // 4)

        self.cdconv1_1 = conv_block(ch_in=1 * base_channel // 4, ch_out=1 * base_channel // 4)
        self.cdconv1_2 = conv_block(ch_in=2 * base_channel // 4, ch_out=2 * base_channel // 4)
        self.cdconv1_3 = conv_block(ch_in=4 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.cdconv1_4 = conv_block(ch_in=8 * base_channel // 4, ch_out=8 * base_channel // 4)

        self.Conv2_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv2_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv2_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv2_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv2_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.cam2_1 = CAM_Module(1 * base_channel // 4)
        self.dam2_1 = PAM_Module(1 * base_channel // 4)
        self.cam2_2 = CAM_Module(2 * base_channel // 4)
        self.dam2_2 = PAM_Module(2 * base_channel // 4)
        self.cam2_3 = CAM_Module(4 * base_channel // 4)
        self.dam2_3 = PAM_Module(4 * base_channel // 4)
        self.cam2_4 = CAM_Module(8 * base_channel // 4)
        self.dam2_4 = PAM_Module(8 * base_channel // 4)

        self.cdconv2_1 = conv_block(ch_in=1 * base_channel // 4, ch_out=1 * base_channel // 4)
        self.cdconv2_2 = conv_block(ch_in=2 * base_channel // 4, ch_out=2 * base_channel // 4)
        self.cdconv2_3 = conv_block(ch_in=4 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.cdconv2_4 = conv_block(ch_in=8 * base_channel // 4, ch_out=8 * base_channel // 4)

        self.Conv3_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv3_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv3_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv3_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv3_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.cam3_1 = CAM_Module(1 * base_channel // 4)
        self.dam3_1 = PAM_Module(1 * base_channel // 4)
        self.cam3_2 = CAM_Module(2 * base_channel // 4)
        self.dam3_2 = PAM_Module(2 * base_channel // 4)
        self.cam3_3 = CAM_Module(4 * base_channel // 4)
        self.dam3_3 = PAM_Module(4 * base_channel // 4)
        self.cam3_4 = CAM_Module(8 * base_channel // 4)
        self.dam3_4 = PAM_Module(8 * base_channel // 4)

        self.cdconv3_1 = conv_block(ch_in=1 * base_channel // 4, ch_out=1 * base_channel // 4)
        self.cdconv3_2 = conv_block(ch_in=2 * base_channel // 4, ch_out=2 * base_channel // 4)
        self.cdconv3_3 = conv_block(ch_in=4 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.cdconv3_4 = conv_block(ch_in=8 * base_channel // 4, ch_out=8 * base_channel // 4)

        self.Conv4_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv4_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv4_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv4_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv4_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.cam4_1 = CAM_Module(1 * base_channel // 4)
        self.dam4_1 = PAM_Module(1 * base_channel // 4)
        self.cam4_2 = CAM_Module(2 * base_channel // 4)
        self.dam4_2 = PAM_Module(2 * base_channel // 4)
        self.cam4_3 = CAM_Module(4 * base_channel // 4)
        self.dam4_3 = PAM_Module(4 * base_channel // 4)
        self.cam4_4 = CAM_Module(8 * base_channel // 4)
        self.dam4_4 = PAM_Module(8 * base_channel // 4)

        self.cdconv4_1 = conv_block(ch_in=1 * base_channel // 4, ch_out=1 * base_channel // 4)
        self.cdconv4_2 = conv_block(ch_in=2 * base_channel // 4, ch_out=2 * base_channel // 4)
        self.cdconv4_3 = conv_block(ch_in=4 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.cdconv4_4 = conv_block(ch_in=8 * base_channel // 4, ch_out=8 * base_channel // 4)

        self.ConvFusion = nn.Conv3d(in_channels= 16 * base_channel, out_channels = img_ch, kernel_size=3, padding=1)
        self.softmaxFusion = nn.Softmax(dim=1)
        self.ConvReweightedFeatures = nn.Conv3d(in_channels=16*base_channel,out_channels=16*base_channel,kernel_size=1,padding=0)

        self.Up5 = up_conv(ch_in=16 * base_channel, ch_out=8 * base_channel)
        self.Up_conv5 = conv_block(ch_in=16 * base_channel, ch_out=8 * base_channel)

        self.Up4 = up_conv(ch_in=8 * base_channel, ch_out=4 * base_channel)
        self.Up_conv4 = conv_block(ch_in=8 * base_channel, ch_out=4 * base_channel)

        self.Up3 = up_conv(ch_in=4 * base_channel, ch_out=2 * base_channel)
        self.Up_conv3 = conv_block(ch_in=4 * base_channel, ch_out=2 * base_channel)

        self.Up2 = up_conv(ch_in=2 * base_channel, ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=2 * base_channel, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = nn.Sigmoid()
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        img1, img2, img3, img4 = x[:,0,:,:,:].unsqueeze(dim=1) , x[:,1,:,:,:].unsqueeze(dim=1), x[:,2,:,:,:].unsqueeze(dim=1), x[:,3,:,:,:].unsqueeze(dim=1)


        img1_1 = self.Conv1_1(img1)

        img1_2 = self.Maxpool(img1_1)
        img1_2 = self.Conv1_2(img1_2)
        img_cat_1_1 = img1_1 + self.cam1_1(img1_1) + self.dam1_1(img1_1)

        img1_3 = self.Maxpool(img1_2)
        img1_3 = self.Conv1_3(img1_3)
        img_cat_1_2 = img1_2 + self.cam1_2(img1_2) + self.dam1_2(img1_2)

        img1_4 = self.Maxpool(img1_3)
        img1_4 = self.Conv1_4(img1_4)
        img_cat_1_3 = img1_3 + self.cam1_3(img1_3) + self.dam1_3(img1_3)

        img1_5 = self.Maxpool(img1_4)
        img1_5 = self.Conv1_5(img1_5)
        img_cat_1_4 = img1_4 + self.cam1_4(img1_4) + self.dam1_4(img1_4)

        img2_1 = self.Conv2_1(img2)

        img2_2 = self.Maxpool(img2_1)
        img2_2 = self.Conv2_2(img2_2)
        img_cat_2_1 = img2_1 + self.cam2_1(img2_1) + self.dam2_1(img2_1)

        img2_3 = self.Maxpool(img2_2)
        img2_3 = self.Conv2_3(img2_3)
        img_cat_2_2 = img2_2 + self.cam2_2(img2_2) + self.dam2_2(img2_2)

        img2_4 = self.Maxpool(img2_3)
        img2_4 = self.Conv2_4(img2_4)
        img_cat_2_3 = img2_3 + self.cam2_3(img2_3) + self.dam2_3(img2_3)

        img2_5 = self.Maxpool(img2_4)
        img2_5 = self.Conv2_5(img2_5)
        img_cat_2_4 = img2_4 + self.cam2_4(img2_4) + self.dam2_4(img2_4)

        img3_1 = self.Conv3_1(img3)

        img3_2 = self.Maxpool(img3_1)
        img3_2 = self.Conv3_2(img3_2)
        img_cat_3_1 = img3_1 + self.cam3_1(img3_1) + self.dam3_1(img3_1)

        img3_3 = self.Maxpool(img3_2)
        img3_3 = self.Conv3_3(img3_3)
        img_cat_3_2 = img3_2 + self.cam3_2(img3_2) + self.dam3_2(img3_2)

        img3_4 = self.Maxpool(img3_3)
        img3_4 = self.Conv3_4(img3_4)
        img_cat_3_3 = img3_3 + self.cam3_3(img3_3) + self.dam3_3(img3_3)

        img3_5 = self.Maxpool(img3_4)
        img3_5 = self.Conv3_5(img3_5)
        img_cat_3_4 = img3_4 + self.cam3_4(img3_4) + self.dam3_4(img3_4)

        img4_1 = self.Conv4_1(img4)

        img4_2 = self.Maxpool(img4_1)
        img4_2 = self.Conv4_2(img4_2)
        img_cat_4_1 = img4_1 + self.cam4_1(img4_1) + self.dam4_1(img4_1)

        img4_3 = self.Maxpool(img4_2)
        img4_3 = self.Conv4_3(img4_3)
        img_cat_4_2 = img4_2 + self.cam4_2(img4_2) + self.dam4_2(img4_2)

        img4_4 = self.Maxpool(img4_3)
        img4_4 = self.Conv4_4(img4_4)
        img_cat_4_3 = img4_3 + self.cam4_3(img4_3) + self.dam4_3(img4_3)

        img4_5 = self.Maxpool(img4_4)
        img4_5 = self.Conv4_5(img4_5)
        img_cat_4_4 = img4_4 + self.cam4_4(img4_4) + self.dam4_4(img4_4)

        img_concat = torch.cat((img1_5,img2_5,img3_5,img4_5),dim=1)
        img_Attentionmap = self.softmaxFusion(self.ConvFusion(img_concat))

        img1_add_attention = img1_5 + img1_5 * img_Attentionmap[:, 0, :, :, :].unsqueeze(dim=1)
        img2_add_attention = img2_5 + img2_5 * img_Attentionmap[:, 1, :, :, :].unsqueeze(dim=1)
        img3_add_attention = img3_5 + img3_5 * img_Attentionmap[:, 2, :, :, :].unsqueeze(dim=1)
        img4_add_attention = img4_5 + img4_5 * img_Attentionmap[:, 3, :, :, :].unsqueeze(dim=1)

        img_concat_add_attention = torch.cat((img1_add_attention,img2_add_attention,img3_add_attention,img4_add_attention),dim=1)
        x5 = self.ConvReweightedFeatures(img_concat_add_attention)

        x4 = torch.cat((img_cat_1_4,img_cat_2_4,img_cat_3_4,img_cat_4_4), dim=1)
        x3 = torch.cat((img_cat_1_3, img_cat_2_3, img_cat_3_3, img_cat_4_3), dim=1)
        x2 = torch.cat((img_cat_1_2, img_cat_2_2, img_cat_3_2, img_cat_4_2), dim=1)
        x1 = torch.cat((img_cat_1_1,img_cat_2_1,img_cat_3_1,img_cat_4_1), dim=1)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.active(d1)


class FusionTry_model_basechannel(nn.Module):
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(FusionTry_model_basechannel, self).__init__()

        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv1_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv1_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv1_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv1_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.Conv2_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv2_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv2_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv2_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv2_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.Conv3_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv3_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv3_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv3_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv3_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.Conv4_1 = conv_block(ch_in=img_ch // 4, ch_out=base_channel // 4)
        self.Conv4_2 = conv_block(ch_in=base_channel // 4, ch_out=2 * base_channel // 4)
        self.Conv4_3 = conv_block(ch_in=2 * base_channel // 4, ch_out=4 * base_channel // 4)
        self.Conv4_4 = conv_block(ch_in=4 * base_channel // 4, ch_out=8 * base_channel // 4)
        self.Conv4_5 = conv_block(ch_in=8 * base_channel // 4, ch_out=16 * base_channel // 4)

        self.ConvFusion = nn.Conv3d(in_channels= 16 * base_channel, out_channels = img_ch, kernel_size=3, padding=1)
        self.softmaxFusion = nn.Softmax(dim=1)
        self.ConvReweightedFeatures = nn.Conv3d(in_channels=16*base_channel,out_channels=16*base_channel,kernel_size=1,padding=0)

        self.Up5 = up_conv(ch_in=16 * base_channel, ch_out=8 * base_channel)
        self.Up_conv5 = conv_block(ch_in=16 * base_channel, ch_out=8 * base_channel)

        self.Up4 = up_conv(ch_in=8 * base_channel, ch_out=4 * base_channel)
        self.Up_conv4 = conv_block(ch_in=8 * base_channel, ch_out=4 * base_channel)

        self.Up3 = up_conv(ch_in=4 * base_channel, ch_out=2 * base_channel)
        self.Up_conv3 = conv_block(ch_in=4 * base_channel, ch_out=2 * base_channel)

        self.Up2 = up_conv(ch_in=2 * base_channel, ch_out=base_channel)
        self.Up_conv2 = conv_block(ch_in=2 * base_channel, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel, output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = nn.Sigmoid()
        self.active = nn.Softmax(dim=1)

    def forward(self, x):
        # encoding path
        img1, img2, img3, img4 = x[:,0,:,:,:].unsqueeze(dim=1) , x[:,1,:,:,:].unsqueeze(dim=1), x[:,2,:,:,:].unsqueeze(dim=1), x[:,3,:,:,:].unsqueeze(dim=1)


        img1_1 = self.Conv1_1(img1)

        img1_2 = self.Maxpool(img1_1)
        img1_2 = self.Conv1_2(img1_2)

        img1_3 = self.Maxpool(img1_2)
        img1_3 = self.Conv1_3(img1_3)

        img1_4 = self.Maxpool(img1_3)
        img1_4 = self.Conv1_4(img1_4)

        img1_5 = self.Maxpool(img1_4)
        img1_5 = self.Conv1_5(img1_5)


        img2_1 = self.Conv2_1(img2)

        img2_2 = self.Maxpool(img2_1)
        img2_2 = self.Conv2_2(img2_2)

        img2_3 = self.Maxpool(img2_2)
        img2_3 = self.Conv2_3(img2_3)

        img2_4 = self.Maxpool(img2_3)
        img2_4 = self.Conv2_4(img2_4)

        img2_5 = self.Maxpool(img2_4)
        img2_5 = self.Conv2_5(img2_5)



        img3_1 = self.Conv3_1(img3)

        img3_2 = self.Maxpool(img3_1)
        img3_2 = self.Conv3_2(img3_2)

        img3_3 = self.Maxpool(img3_2)
        img3_3 = self.Conv3_3(img3_3)

        img3_4 = self.Maxpool(img3_3)
        img3_4 = self.Conv3_4(img3_4)

        img3_5 = self.Maxpool(img3_4)
        img3_5 = self.Conv3_5(img3_5)


        img4_1 = self.Conv4_1(img4)

        img4_2 = self.Maxpool(img4_1)
        img4_2 = self.Conv4_2(img4_2)

        img4_3 = self.Maxpool(img4_2)
        img4_3 = self.Conv4_3(img4_3)

        img4_4 = self.Maxpool(img4_3)
        img4_4 = self.Conv4_4(img4_4)

        img4_5 = self.Maxpool(img4_4)
        img4_5 = self.Conv4_5(img4_5)

        img_concat = torch.cat((img1_5,img2_5,img3_5,img4_5),dim=1)
        img_Attentionmap = self.softmaxFusion(self.ConvFusion(img_concat))

        img1_add_attention = img1_5 * img_Attentionmap[:, 0, :, :, :].unsqueeze(dim=1)
        img2_add_attention = img2_5 * img_Attentionmap[:, 1, :, :, :].unsqueeze(dim=1)
        img3_add_attention = img3_5 * img_Attentionmap[:, 2, :, :, :].unsqueeze(dim=1)
        img4_add_attention = img4_5 * img_Attentionmap[:, 3, :, :, :].unsqueeze(dim=1)

        img_concat_add_attention = torch.cat((img1_add_attention,img2_add_attention,img3_add_attention,img4_add_attention),dim=1)
        x5 = self.ConvReweightedFeatures(img_concat_add_attention)

        x4 = torch.cat((img1_4,img2_4,img3_4,img4_4), dim=1)
        x3 = torch.cat((img1_3, img2_3, img3_3, img4_3), dim=1)
        x2 = torch.cat((img1_2, img2_2, img3_2, img4_2), dim=1)
        x1 = torch.cat((img1_1,img2_1,img3_1,img4_1), dim=1)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return self.active(d1)


class R2U_Net(nn.Module):
    def __init__(self,img_ch=4,output_ch=4,t=2, base_channel=16):
        super(R2U_Net,self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=base_channel,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=1*base_channel,ch_out=2*base_channel,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=2*base_channel,ch_out=4*base_channel,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=4*base_channel,ch_out=8*base_channel,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=8*base_channel,ch_out=16*base_channel,t=t)
        

        self.Up5 = up_conv(ch_in=16*base_channel,ch_out=8*base_channel)
        self.Up_RRCNN5 = RRCNN_block(ch_in=16*base_channel, ch_out=8*base_channel,t=t)
        
        self.Up4 = up_conv(ch_in=8*base_channel,ch_out=4*base_channel)
        self.Up_RRCNN4 = RRCNN_block(ch_in=8*base_channel, ch_out=4*base_channel,t=t)
        
        self.Up3 = up_conv(ch_in=4*base_channel,ch_out=2*base_channel)
        self.Up_RRCNN3 = RRCNN_block(ch_in=4*base_channel, ch_out=2*base_channel,t=t)
        
        self.Up2 = up_conv(ch_in=2*base_channel,ch_out=base_channel)
        self.Up_RRCNN2 = RRCNN_block(ch_in=2*base_channel, ch_out=base_channel,t=t)

        self.Conv_1x1 = nn.Conv3d(base_channel,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return F.softmax(d1, dim=1)


class AttU_Net(nn.Module):
    def __init__(self,img_ch=4, output_ch=4, base_channel=16):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=base_channel)
        self.Conv2 = conv_block(ch_in=base_channel,ch_out=base_channel*2)
        self.Conv3 = conv_block(ch_in=base_channel*2,ch_out=base_channel*4)
        self.Conv4 = conv_block(ch_in=base_channel*4,ch_out=base_channel*8)
        self.Conv5 = conv_block(ch_in=base_channel*8,ch_out=base_channel*16)

        self.Up5 = up_conv(ch_in=base_channel*16,ch_out=base_channel*8)
        self.Att5 = Attention_block(F_g=base_channel*8,F_l=base_channel*8,F_int=base_channel*4)
        self.Up_conv5 = conv_block(ch_in=base_channel*16, ch_out=base_channel*8)

        self.Up4 = up_conv(ch_in=base_channel*8,ch_out=base_channel*4)
        self.Att4 = Attention_block(F_g=base_channel*4,F_l=base_channel*4,F_int=base_channel*2)
        self.Up_conv4 = conv_block(ch_in=base_channel*8, ch_out=base_channel*4)
        
        self.Up3 = up_conv(ch_in=base_channel*4,ch_out=base_channel*2)
        self.Att3 = Attention_block(F_g=base_channel*2,F_l=base_channel*2,F_int=base_channel)
        self.Up_conv3 = conv_block(ch_in=base_channel*4, ch_out=base_channel*2)
        
        self.Up2 = up_conv(ch_in=base_channel*2,ch_out=base_channel)
        self.Att2 = Attention_block(F_g=base_channel,F_l=base_channel,F_int=base_channel//2)
        self.Up_conv2 = conv_block(ch_in=base_channel*2, ch_out=base_channel)

        self.Conv_1x1 = nn.Conv3d(base_channel,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

#         return d1
        return F.softmax(d1, dim = 1)


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=4, output_ch=4,base_channel=16, t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=base_channel,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=base_channel,ch_out=base_channel*2,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=base_channel*2,ch_out=base_channel*4,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=base_channel*4,ch_out=base_channel*8,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=base_channel*8,ch_out=base_channel*16,t=t)
        

        self.Up5 = up_conv(ch_in=base_channel*16,ch_out=base_channel*8)
        self.Att5 = Attention_block(F_g=base_channel*8,F_l=base_channel*8,F_int=base_channel*4)
        self.Up_RRCNN5 = RRCNN_block(ch_in=base_channel*16, ch_out=base_channel*8,t=t)
        
        self.Up4 = up_conv(ch_in=base_channel*8,ch_out=base_channel*4)
        self.Att4 = Attention_block(F_g=base_channel*4,F_l=base_channel*4,F_int=base_channel*2)
        self.Up_RRCNN4 = RRCNN_block(ch_in=base_channel*8, ch_out=base_channel*4,t=t)
        
        self.Up3 = up_conv(ch_in=base_channel*4,ch_out=base_channel*2)
        self.Att3 = Attention_block(F_g=base_channel*2,F_l=base_channel*2,F_int=base_channel)
        self.Up_RRCNN3 = RRCNN_block(ch_in=base_channel*4, ch_out=base_channel*2,t=t)
        
        self.Up2 = up_conv(ch_in=base_channel*2,ch_out=base_channel)
        self.Att2 = Attention_block(F_g=base_channel,F_l=base_channel,F_int=base_channel//2)
        self.Up_RRCNN2 = RRCNN_block(ch_in=base_channel*2, ch_out=base_channel,t=t)

        self.Conv_1x1 = nn.Conv3d(base_channel,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return F.softmax(d1, dim=1)

if __name__ == '__main__':
###########################################################################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 1, 64, 64, 64).cuda()
    net = U_Net(img_ch=1, output_ch=2).cuda()
    out = net(img)
    print(out.shape)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 1, 32, 32, 32).cuda()
    net = R2U_Net(img_ch=1, output_ch=2).cuda()
    out = net(img)
    print(out.shape)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 1, 32, 32, 32).cuda()
    net = AttU_Net(img_ch=1, output_ch=2).cuda()
    out = net(img)
    print(out.shape)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 1, 32, 32, 32).cuda()
    net = R2AttU_Net(img_ch=1, output_ch=2).cuda()
    out = net(img)
    print(out.shape)
