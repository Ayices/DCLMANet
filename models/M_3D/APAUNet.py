# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import torch
from torch import nn
import torch.nn.functional as F
from DrawColorMap import SaveHeatMapWithBackgroundMap as Draw

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
                
def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


class MixConv(nn. Module):
    def __init__(self, inp, oup):
        super(MixConv, self).__init__()

        self.groups = oup // 4
        in_channel = inp // 4
        out_channel = oup // 4

        self.dwconv1 = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.dwconv2 = nn.Conv2d(in_channel, out_channel, 5, padding=2)
        self.dwconv3 = nn.Conv2d(in_channel, out_channel, 7, padding=3)
        self.dwconv4 = nn.Conv2d(in_channel, out_channel, 9, padding=4)

        self.pwconv = nn.Sequential(
            nn.InstanceNorm2d(oup),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup, oup, 1),
            nn.InstanceNorm2d(oup),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        a, b, c, d = torch.split(x, self.groups, dim=1)
        a = self.dwconv1(a)
        b = self.dwconv1(b)
        c = self.dwconv1(c)
        d = self.dwconv1(d)

        out = torch.cat([a, b, c, d], dim=1)
        out = self.pwconv(out)

        return out

class InnerBlock(nn.Module):
    def __init__(self, dim, kernel_size, project_dim=2):
        super(InnerBlock, self).__init__()

        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.InstanceNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.InstanceNorm3d(dim)
        )

        self.bn = nn.InstanceNorm3d(dim)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.seconv1 = nn.Conv3d(dim, attn_chs, 1)
        self.seLayerNorm = nn.LayerNorm(attn_chs)
        self.seReLU = nn.ReLU(inplace=True)
        self.seconv2 = nn.Conv3d(attn_chs, self.radix*dim, 1)
        
    def forward(self, x):
        k = torch.mean(x, self.project_dim) + torch.max(x, self.project_dim)[0]
        k = self.key_embed(k)
        q = torch.mean(x, self.project_dim) + torch.max(x, self.project_dim)[0]
        qk = torch.cat([q, k], dim=1)

        w = self.embed(qk)
        w = w.unsqueeze(self.project_dim)
        fill_shape = w.shape[-1]
        repeat_shape = [1,1,1,1,1]
        repeat_shape[self.project_dim] = fill_shape
        w = w.repeat(repeat_shape)
        
        v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)

        x_attn = self.seconv1(x_gap)
        x_attn = x_attn.squeeze(2).squeeze(2).squeeze(2)
        x_attn = self.seLayerNorm(x_attn)
        x_attn = self.seReLU(x_attn).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x_attn = self.seconv2(x_attn)

        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        
        return out.contiguous()


class InnerTransBlock(nn.Module):
    '''
    parameters: 
        x: low-resolution features from decoder
        y: high-resolution features from encoder 
    '''
    def __init__(self, dim, kernel_size, project_dim=2):
        super(InnerTransBlock, self).__init__()
        
        # current output dimension for decoder
        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size
        
        # kxk group convolution
        self.key_embed = nn.Sequential(
            nn.ConvTranspose2d(2*dim, dim, kernel_size=2, stride=2, groups=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2        
        # two sequential 1x1 convolution 
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.InstanceNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups= dim // share_planes, num_channels = dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.ConvTranspose3d(2*dim, dim, kernel_size=2, stride=2, padding=0, dilation=1, bias=False),
            nn.InstanceNorm3d(dim)
        )

        self.bn = nn.InstanceNorm3d(dim)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)

        self.seconv1 = nn.Conv3d(dim, attn_chs, 1)
        self.seLayerNorm = nn.LayerNorm(attn_chs)
        self.seReLU = nn.ReLU(inplace=True)
        self.seconv2 = nn.Conv3d(attn_chs, self.radix * dim, 1)

    def forward(self, x, y):
        '''
            x: [B,C,H,W,D]
            y: [B,C/2,2H,2W,2D]
        '''
        k = torch.max(x, self.project_dim)[0] + torch.mean(x, self.project_dim)
        k = self.key_embed(k)  
        q = torch.max(y, self.project_dim)[0] + torch.mean(y, self.project_dim)
        qk = torch.cat([q, k], dim=1) 

        w = self.embed(qk)  
        w = w.unsqueeze(self.project_dim)
        fill_shape = w.shape[-1]
        repeat_shape = [1,1,1,1,1]
        repeat_shape[self.project_dim] = fill_shape
        w = w.repeat(repeat_shape)
        
        v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        y = y.view(B, C, 1, H, W, D)
        y = torch.cat([y, v], dim=2)

        y_gap = y.sum(dim=2)
        y_gap = y_gap.mean((2, 3, 4), keepdim=True)

        y_attn = self.seconv1(y_gap)
        y_attn = y_attn.squeeze(2).squeeze(2).squeeze(2)
        y_attn = self.seLayerNorm(y_attn)
        y_attn = self.seReLU(y_attn).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        y_attn = self.seconv2(y_attn)

        y_attn = y_attn.view(B, C, self.radix)
        y_attn = F.softmax(y_attn, dim=2)
        out = (y * y_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        
        return out.contiguous()


class conv_block_APA(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_APA,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
class doubelconv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(doubelconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class APA_H(nn.Module):
    def __init__(self, in_ch, out_ch, transpose=False):
        super(APA_H, self).__init__()
        if transpose is True:
            self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1, padding=0),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True))
            self.attention = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=2)
        else:
            self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, padding=0),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True))
            self.attention = InnerBlock(dim=out_ch, kernel_size=3, project_dim=2)

    def forward(self, input, high_res_input=None):
        attn = input
        if high_res_input is not None:  
            attn = self.attention(attn, high_res_input)  # [B,C,W,D]
            return attn
        else:
            attn = self.attention(attn)  # [B,C,H,W,D]
            return attn

class APA_W(nn.Module):
    def __init__(self, in_ch, out_ch, transpose=False):
        super(APA_W, self).__init__()
        if transpose is True:
            self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1, padding=0),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True))
            self.attention = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=3)
        else:
            self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, padding=0),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True))
            self.attention = InnerBlock(dim=out_ch, kernel_size=3, project_dim=3)
        
    def forward(self, input, high_res_input=None):
        attn = input
        if high_res_input is not None:
            attn = self.attention(attn, high_res_input)  # [B,C,H,D] 
            return attn
        else:
            attn = self.attention(attn)  # [B,C,H,D]
            return attn
        
class APA_D(nn.Module):
    def __init__(self, in_ch, out_ch, transpose=False):
        super(APA_D, self).__init__()
        if transpose is True:
            self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1, padding=0),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True))
            self.attention = InnerTransBlock(dim=out_ch, kernel_size=3, project_dim=4)
        else:
            self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, padding=0),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True))
            self.attention = InnerBlock(dim=out_ch, kernel_size=3, project_dim=4)
        
    def forward(self, input, high_res_input=None): 
        attn = input 
        if high_res_input is not None:
            attn = self.attention(attn, high_res_input)  
            return attn
        else:
            attn = self.attention(attn) 
            return attn

class MPABlock_SECOND(nn.Module):
    def __init__(self, base_channel):
        super(MPABlock_SECOND, self).__init__()

        self.base_channel = base_channel

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2 * base_channel, base_channel // factor, 1, bias=False),
            nn.InstanceNorm3d(base_channel // factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channel // factor, base_channel, kernel_size=1),
            nn.GroupNorm(num_groups=base_channel // share_planes, num_channels=base_channel)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(base_channel, base_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.InstanceNorm3d(base_channel)
        )

        self.bn = nn.InstanceNorm3d(base_channel)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(base_channel * self.radix // reduction_factor, 32)
        self.seconv1 = nn.Conv3d(base_channel, attn_chs, 1)
        self.seLayerNorm = nn.LayerNorm(attn_chs)
        self.seReLU = nn.ReLU(inplace=True)
        self.seconv2 = nn.Conv3d(attn_chs, self.radix * base_channel, 1)

    def forward(self, x, y):
        k = x
        q = y
        qk = torch.cat([q, k], dim=1)
        w = self.embed(qk)

        v = self.conv1x1(x)

        w = F.softmax(w, dim=1)

        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)


        #x_gap = v + x
        x_gap = x.sum(dim=2)

        x_gap = x_gap.mean((2, 3, 4), keepdim=True)

        x_attn = self.seconv1(x_gap)
        x_attn = x_attn.squeeze(2).squeeze(2).squeeze(2)
        x_attn = self.seLayerNorm(x_attn)
        x_attn = self.seReLU(x_attn).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x_attn = self.seconv2(x_attn)

        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)

        return out.contiguous()

class MPABlock(nn.Module):
    def __init__(self, base_channel):
        super(MPABlock, self).__init__()

        self.base_channel = base_channel

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2 * base_channel, base_channel // factor, 1, bias=False),
            nn.InstanceNorm3d(base_channel // factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channel // factor, base_channel, kernel_size=1),
            nn.GroupNorm(num_groups=base_channel // share_planes, num_channels=base_channel)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(base_channel, base_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.InstanceNorm3d(base_channel)
        )

        self.bn = nn.InstanceNorm3d(base_channel)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(base_channel * self.radix // reduction_factor, 32)
        self.seconv1 = nn.Conv3d(base_channel, attn_chs, 1)
        self.seLayerNorm = nn.LayerNorm(attn_chs)
        self.seReLU = nn.ReLU(inplace=True)
        self.seconv2 = nn.Conv3d(attn_chs, self.radix * base_channel, 1)

    def forward(self, x, y):
        k = x
        q = y
        qk = torch.cat([q, k], dim=1)
        w = self.embed(qk)

        v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)


        #x_gap = v + x
        x_gap = x.sum(dim=2)

        x_gap = x_gap.mean((2, 3, 4), keepdim=True)

        x_attn = self.seconv1(x_gap)
        x_attn = x_attn.squeeze(2).squeeze(2).squeeze(2)
        x_attn = self.seLayerNorm(x_attn)
        x_attn = self.seReLU(x_attn).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x_attn = self.seconv2(x_attn)

        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)

        return out.contiguous()



class MPABlock_test(nn.Module):
    def __init__(self, base_channel):
        super(MPABlock_test, self).__init__()

        self.base_channel = base_channel

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2 * base_channel, base_channel // factor, 1, bias=False),
            nn.InstanceNorm3d(base_channel // factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channel // factor, base_channel, kernel_size=1),
            nn.GroupNorm(num_groups=base_channel // share_planes, num_channels=base_channel)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(base_channel, base_channel, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.InstanceNorm3d(base_channel)
        )

        self.bn = nn.InstanceNorm3d(base_channel)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(base_channel * self.radix // reduction_factor, 32)
        self.seconv1 = nn.Conv3d(base_channel, attn_chs, 1)
        self.seLayerNorm = nn.LayerNorm(attn_chs)
        self.seReLU = nn.ReLU(inplace=True)
        self.seconv2 = nn.Conv3d(attn_chs, self.radix * base_channel, 1)

        self.num = 1

    def forward(self, x, y, test):
        k = x
        q = y
        qk = torch.cat([q, k], dim=1)
        w = self.embed(qk)

        v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        # v_test = v.sum(dim=1, keepdim=True)
        # vtest = F.interpolate(v_test, size=[128, 128, 128])
        # vtest = vtest.squeeze(dim=0)
        # vtest = vtest.transpose(1, 3)
        # #
        # test = test.squeeze(dim=0)
        # test = test.transpose(1, 3)
        # Draw(test[:, 64, :, :].unsqueeze(dim=1).detach().cpu().numpy(), alpha=0.5,
        #                input_list=[vtest[:, 64, :, :].unsqueeze(dim=1).detach().cpu().numpy()], name_list=['attention_map'],
        #                root='/home/public/1', num=self.num)



        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)


        #x_gap = v + x
        x_gap = x.sum(dim=2)

        x_gap = x_gap.mean((2, 3, 4), keepdim=True)

        x_attn = self.seconv1(x_gap)
        x_attn = x_attn.squeeze(2).squeeze(2).squeeze(2)
        x_attn = self.seLayerNorm(x_attn)
        x_attn = self.seReLU(x_attn).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x_attn = self.seconv2(x_attn)

        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        # test = test.squeeze(dim=0)
        # test = test.transpose(1, 3)
        # for i in range(64):
        #     # v_test = out.sum(dim=1, keepdim=True)
        #     v_test = out[:,i,:,:,:].unsqueeze(dim=1)
        #     vtest = F.interpolate(v_test, size=[128, 128, 128],mode='nearest')
        #     vtest = vtest.squeeze(dim=0)
        #     vtest = vtest.transpose(1, 3)
        #
        #     Draw(test[:, 64, :, :].unsqueeze(dim=1).detach().cpu().numpy(), alpha=0.5,
        #          input_list=[vtest[:, 64, :, :].unsqueeze(dim=1).detach().cpu().numpy()], name_list=['attention_map'],
        #          root='/home/public/1', num=self.num)
        #     self.num = self.num + 1


        return out.contiguous()

class MPA_test(nn.Module):
    def __init__(self, in_ch):
        super(MPA_test, self).__init__()
        self.attention = MPABlock_test(base_channel=in_ch)

    def forward(self, input, AddAttention,test):
        attn = self.attention(input, AddAttention,test)
        return attn

class MPA_SECOND(nn.Module):
    def __init__(self, in_ch):
        super(MPA_SECOND, self).__init__()
        self.attention = MPABlock_SECOND(base_channel=in_ch)

    def forward(self, input, AddAttention):
        attn = self.attention(input, AddAttention)
        return attn

class MPA(nn.Module):
    def __init__(self, in_ch):
        super(MPA, self).__init__()
        self.attention = MPABlock(base_channel=in_ch)

    def forward(self, input, AddAttention):
        attn = self.attention(input, AddAttention)
        return attn

class APAEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, last_layer=False):
        super(APAEncoder, self).__init__()
        self.last_layer = last_layer
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True)
            )        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.sh_att = APA_H(in_ch, in_ch)
        self.sw_att = APA_W(in_ch, in_ch)
        self.sd_att = APA_D(in_ch, in_ch)
        
    def forward(self, input):  
        feat = self.conv1(input)
        sh_attn = self.sh_att(feat)
        sw_attn = self.sw_att(feat)
        sd_attn = self.sd_att(feat)
        attn = (sh_attn * self.beta[0] + sw_attn * self.beta[1] + sd_attn * self.beta[2]) / self.beta.sum()
        attn = self.conv2(attn)
        if self.last_layer is True:
            return attn
        else:
            return self.Maxpool(attn)

class APADecoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APADecoder, self).__init__()
        self.conv = conv_block_APA(in_ch, out_ch)
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.sh_att = APA_H(in_ch, out_ch, transpose=True)
        self.sw_att = APA_W(in_ch, out_ch, transpose=True)
        self.sd_att = APA_D(in_ch, out_ch, transpose=True)
        
    def forward(self, input, high_res_input):  
        sh_attn = self.sh_att(input, high_res_input)
        sw_attn = self.sw_att(input, high_res_input)
        sd_attn = self.sd_att(input, high_res_input)
        attn = (sh_attn * self.beta[0] + sw_attn * self.beta[1] + sd_attn * self.beta[2]) / self.beta.sum()
        return self.conv(attn)

class APAUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APAUNet, self).__init__()

        self.base_channel = 8
        self.stem =  doubelconv_block(in_ch, self.base_channel)
        self.encoder1 = APAEncoder(self.base_channel, self.base_channel*2)
        self.encoder2 = APAEncoder(self.base_channel*2, self.base_channel*4)
        self.encoder3 = APAEncoder(self.base_channel*4, self.base_channel*8)
        self.encoder4 = APAEncoder(self.base_channel*8, self.base_channel*16)
        self.encoder5 = APAEncoder(self.base_channel*16, self.base_channel*16, last_layer=True)

        self.decoder1 = APADecoder(self.base_channel*16,self.base_channel*8)
        self.decoder2 = APADecoder(self.base_channel*8,self.base_channel*4)
        self.decoder3 = APADecoder(self.base_channel*4,self.base_channel*2)
        self.decoder4 = APADecoder(self.base_channel*2,self.base_channel)
        
        self.tail = nn.Conv3d(self.base_channel, out_ch, 1)
        self.active = nn.Softmax(dim=1)

        self.apply(self.initializer)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.encoder1(c1)
        c3 = self.encoder2(c2)
        c4 = self.encoder3(c3)
        c5 = self.encoder4(c4)
        c6 = self.encoder5(c5)
        c7 = self.decoder1(c6,c4)
        c8 = self.decoder2(c7,c3)
        c9 = self.decoder3(c8,c2)
        c10 = self.decoder4(c9,c1)
        out = self.tail(c10)
        return self.active(out)


class APAUNet_Init(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APAUNet_Init, self).__init__()

        self.base_channel = 16
        self.initializer = InitWeights_He(1e-2)
        self.stem = doubelconv_block(in_ch, self.base_channel)
        self.encoder1 = APAEncoder(self.base_channel, self.base_channel * 2)
        self.encoder2 = APAEncoder(self.base_channel * 2, self.base_channel * 4)
        self.encoder3 = APAEncoder(self.base_channel * 4, self.base_channel * 8)
        self.encoder4 = APAEncoder(self.base_channel * 8, self.base_channel * 16)
        self.encoder5 = APAEncoder(self.base_channel * 16, self.base_channel * 16, last_layer=True)

        self.decoder1 = APADecoder(self.base_channel * 16, self.base_channel * 8)
        self.decoder2 = APADecoder(self.base_channel * 8, self.base_channel * 4)
        self.decoder3 = APADecoder(self.base_channel * 4, self.base_channel * 2)
        self.decoder4 = APADecoder(self.base_channel * 2, self.base_channel)

        self.tail = nn.Conv3d(self.base_channel, out_ch, 1)
        self.active = nn.Softmax(dim=1)

        self.apply(self.initializer)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.encoder1(c1)
        c3 = self.encoder2(c2)
        c4 = self.encoder3(c3)
        c5 = self.encoder4(c4)
        c6 = self.encoder5(c5)
        c7 = self.decoder1(c6, c4)
        c8 = self.decoder2(c7, c3)
        c9 = self.decoder3(c8, c2)
        c10 = self.decoder4(c9, c1)
        out = self.tail(c10)
        return self.active(out)

if __name__ == '__main__':
    model = APAUNet(1, 1).eval()

    x = torch.ones((1, 1, 48, 48, 48))
    y = model(x)

    print(y.shape)