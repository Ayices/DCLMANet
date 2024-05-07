import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        all_top_indices_z = [0, 0, 0, 0]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
        mapper_z = all_top_indices_z[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y, mapper_z

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, depth, mapper_x, mapper_y, mapper_z, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y) == len(mapper_z)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, depth, mapper_x, mapper_y, mapper_z, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 5, 'x must been 5` dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3, 4])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, tile_size_z, mapper_x, mapper_y, mapper_z, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y, tile_size_z)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y, w_z) in enumerate(zip(mapper_x, mapper_y, mapper_z)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    for t_z in range(tile_size_z):
                        dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y, t_z] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y) * self.build_filter(t_z, w_z, tile_size_z)

        return dct_filter

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, dct_d, reduction=4, freq_sel_method='top1'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction  #4
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.dct_d = dct_d

        mapper_x, mapper_y, mapper_z = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)    #与method后的数字大小相同
        mapper_x = [temp_x * (dct_h // 8) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 8) for temp_y in mapper_y]
        mapper_z = [temp_z * (dct_d // 8) for temp_z in mapper_z]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, dct_d, mapper_x, mapper_y, mapper_z, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w ,d= x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w or d != self.dct_d:
            x_pooled = torch.nn.functional.adaptive_avg_pool3d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1, 1)
        return x * y.expand_as(x)


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)
        c2wh = dict([(16, 128), (32, 64), (64, 32), (128, 16), (256,8)])
        # c2wh = dict([(8, 128), (16, 64), (32, 32), (64, 16), (128, 8)])
        self.att = MultiSpectralAttentionLayer(channel, c2wh[channel], c2wh[channel], c2wh[channel], reduction=reduction, freq_sel_method ='top1')
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv3d(channel, channel, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # y = self.avg_pool(x)
        y = self.att(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class En_CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(En_CAB, self).__init__()
        modules_body1 = []
        modules_body1.append(conv(n_feat, 2*n_feat, kernel_size, bias=bias))
        modules_body = []
        modules_body.append(act)
        modules_body.append(conv(2*n_feat, 2*n_feat, kernel_size, bias=bias))

        self.CA = CALayer(2*n_feat, reduction, bias=bias)
        self.body1 = nn.Sequential(*modules_body1)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body1(x)
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class De_CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(De_CAB, self).__init__()
        modules_body1 = []
        modules_body1.append(conv(2*n_feat, 2*n_feat, kernel_size, bias=bias))
        modules_body = []
        modules_body.append(act)
        modules_body.append(conv(2*n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(2*n_feat, reduction, bias=bias)
        self.body1 = nn.Sequential(*modules_body1)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body1(x)
        res = self.CA(x)
        res += x
        res = self.body(x)
        return res

class Skip_CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(Skip_CAB, self).__init__()
        modules_body1 = []
        modules_body1.append(conv(2*n_feat, 2*n_feat, kernel_size, bias=bias))
        modules_body = []
        modules_body.append(act)
        modules_body.append(conv(2*n_feat, 2*n_feat, kernel_size, bias=bias))

        self.CA = CALayer(2*n_feat, reduction, bias=bias)
        self.body1 = nn.Sequential(*modules_body1)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body1(x)
        res = self.CA(x)
        res += x
        res = self.body(x)
        return res

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(Encoder, self).__init__()

        self.encoder_level1 = [En_CAB(n_feat, kernel_size, reduction, bias=bias, act=act)]
        self.encoder_level2 = [En_CAB(n_feat * 2, kernel_size, reduction, bias=bias, act=act)]
        self.encoder_level3 = [En_CAB(n_feat * 4, kernel_size, reduction, bias=bias, act=act)]
        self.encoder_level4 = [En_CAB(n_feat * 8, kernel_size, reduction, bias=bias, act=act)]
        self.encoder_level5 = [En_CAB(n_feat * 16, kernel_size, reduction, bias=bias, act=act)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level4 = nn.Sequential(*self.encoder_level4)
        self.encoder_level5 = nn.Sequential(*self.encoder_level5)

        self.down12 = DownSample(n_feat*2)
        self.down23 = DownSample(n_feat*4)
        self.down34 = DownSample(n_feat*8)
        self.down45 = DownSample(n_feat*16)


    def forward(self, x):
        enc1 = self.encoder_level1(x)

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)

        x = self.down34(enc3)

        enc4 = self.encoder_level4(x)

        x = self.down45(enc4)

        enc5 = self.encoder_level5(x)

        return [enc1, enc2, enc3, enc4, enc5]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(Decoder, self).__init__()

        self.decoder_level1 = [De_CAB(n_feat,   kernel_size, reduction, bias=bias, act=act)]
        self.decoder_level2 = [De_CAB(n_feat*2, kernel_size, reduction, bias=bias, act=act)]
        self.decoder_level3 = [De_CAB(n_feat*4, kernel_size, reduction, bias=bias, act=act)]
        self.decoder_level4 = [De_CAB(n_feat*8, kernel_size, reduction, bias=bias, act=act)]
        self.decoder_level5 = [De_CAB(n_feat*16,kernel_size, reduction, bias=bias, act=act)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level4 = nn.Sequential(*self.decoder_level4)
        self.decoder_level5 = nn.Sequential(*self.decoder_level5)

        self.skip_attn1 = Skip_CAB(n_feat , kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = Skip_CAB(n_feat * 2, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn3 = Skip_CAB(n_feat * 4, kernel_size, reduction, bias=bias, act=act)
        self.skip_attn4 = Skip_CAB(n_feat * 8, kernel_size, reduction, bias=bias, act=act)

        self.up21 = SkipUpSample(n_feat*2)
        self.up32 = SkipUpSample(n_feat*4)
        self.up43 = SkipUpSample(n_feat*8)
        self.up54 = SkipUpSample(n_feat*16)

    def forward(self, outs):
        enc1, enc2, enc3, enc4, enc5 = outs

        dec5 = self.decoder_level5(enc5)

        x = self.up54(dec5, self.skip_attn4(enc4))
        dec4 = self.decoder_level4(x)

        x = self.up43(dec4, self.skip_attn3(enc3))
        dec3 = self.decoder_level3(enc3)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1,dec2,dec3,dec4,dec5]

##########################################################################
##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.MaxPool3d(kernel_size=2,stride=2),
                                  nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
                                nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)

        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        diffZ = y.size()[4] - x.size()[4]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        x = x + y
        return x


##########################################################################
class UNet_Fca(nn.Module):
    def __init__(self, in_c=4, out_c=4, n_feat=16, kernel_size=3, reduction=4, bias=True):
        super(UNet_Fca, self).__init__()

        act=nn.PReLU()
        self.conv1 = conv(in_c, n_feat, kernel_size, bias=bias)
        self.encoder = Encoder(n_feat, kernel_size, reduction, act, bias)
        self.decoder = Decoder(n_feat, kernel_size, reduction, act, bias)
        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)
        self.active = nn.Softmax(dim=1)

    def forward(self, img):
        x = self.conv1(img)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.tail(x[0])
        return self.active(x)
