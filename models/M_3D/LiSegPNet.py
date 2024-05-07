import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride= 1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        ch_in = 3,
        num_classes = 1000,
        zero_init_residual = False,
        groups = 1,
        width_per_group = 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
        **kwargs
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(ch_in, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.numFeaturesList = [
            64, 
            64 * block.expansion, 
            128 * block.expansion,
            256 * block.expansion,
            512 * block.expansion
            ]
        self.num_features = 512 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def features(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)

        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def wide_resnet50_2( **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], width_per_group = 64 * 2, **kwargs)

def wide_resnet101_2( **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], width_per_group = 64 * 2, **kwargs)

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
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class RCB(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(RCB,self).__init__()
        self.Conv0 = conv_block(ch_in, ch_out)
        self.channel_expand = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 1, 1, 0),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x_conv = self.Conv0(x)
        x_in = self.channel_expand(x)
        return x_conv + x_in

class MTA_Block(nn.Module):
    def __init__(self, ch_in):
        super(MTA_Block,self).__init__()
        self.ConvC = nn.Sequential(
            nn.Conv3d(2, 1, 7, 1, 3),
            nn.InstanceNorm3d(1)
            )
        self.ConvH = nn.Sequential(
            nn.Conv3d(2, 1, 7, 1, 3),
            nn.InstanceNorm3d(1)
            )
        self.ConvW = nn.Sequential(
            nn.Conv3d(2, 1, 7, 1, 3),
            nn.InstanceNorm3d(1)
            )
        self.ConvD = nn.Sequential(
            nn.Conv3d(2, 1, 7, 1, 3),
            nn.InstanceNorm3d(1)
        )
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch_in * 4, ch_in, 1, 1, 0),
            # nn.BatchNorm2d(ch_in),
            nn.InstanceNorm3d(ch_in),
            nn.ReLU()
            )

    def forward(self,x):
        B, C, H, W, D = x.size()
        x_c = x.permute(0, 2, 3, 4, 1).reshape(B, H*W*D, C)
        x_ca = self.avgpool(x_c)
        x_cm = self.maxpool(x_c)
        x_c = torch.cat([x_ca, x_cm], -1)
        x_c = x_c.permute(0, 2, 1).reshape(B, 2, H, W, D)
        a_c = self.ConvC(x_c)
        a_c = self.sigmoid(a_c)
        x_c = x * a_c
        
        x_h = x.permute(0, 1, 3, 4, 2).reshape(B, C*W*D, H)
        x_ha = self.avgpool(x_h)
        x_hm = self.maxpool(x_h)
        x_h = torch.cat([x_ha, x_hm], -1)
        x_h = x_h.permute(0, 2, 1).reshape(B, 2, C, W, D)
        a_h = self.ConvH(x_h).permute(0, 2, 1, 3, 4)
        a_h = self.sigmoid(a_h)
        x_h = x * a_h
        
        x_w = x.permute(0, 1, 2, 4, 3).reshape(B, C*H*D, W)
        x_wa = self.avgpool(x_w)
        x_wm = self.maxpool(x_w)
        x_w = torch.cat([x_wa, x_wm], -1)
        x_w = x_w.permute(0, 2, 1).reshape(B, 2, C, H ,D)
        a_w = self.ConvH(x_w).permute(0, 2, 3, 1, 4)
        a_w = self.sigmoid(a_w)
        x_w = x * a_w

        x_d = x.reshape(B, C * H * W, D)
        x_da = self.avgpool(x_d)
        x_dm = self.maxpool(x_d)
        x_d = torch.cat([x_da, x_dm], -1)
        x_d = x_d.permute(0, 2, 1).reshape(B, 2, C, H ,W)
        a_d = self.ConvD(x_d).permute(0, 2, 3, 4, 1)
        a_d = self.sigmoid(a_d)
        x_d = x * a_d
        
        x_o = torch.cat( [x_c, x_h, x_w, x_d], 1 )
        x_o = self.bottleneck( x_o )
        return x_o

class LiSegPNet(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, base_ch = 32):
        super(LiSegPNet,self).__init__()
        
        self.output_ch = output_ch
        self.baseEncoder = resnet50(ch_in = img_ch)
        ch_list = self.baseEncoder.numFeaturesList
        self.rcb_e1 = nn.Sequential(
            nn.Conv3d(img_ch + ch_list[0], base_ch, 1, 1, 0),
            # nn.BatchNorm2d(base_ch),
            nn.InstanceNorm3d(base_ch),
            nn.ReLU(),
            RCB(base_ch, base_ch)
        )
        self.mta_e1 = MTA_Block(base_ch)
        self.rcb_e2 = nn.Sequential(
            nn.Conv3d(base_ch + ch_list[1], base_ch * 2, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 2),
            nn.InstanceNorm3d(base_ch * 2),
            nn.ReLU(),
            RCB(base_ch * 2, base_ch * 2)
        )
        self.mta_e2 = MTA_Block(base_ch * 2)
        self.rcb_e3 = nn.Sequential(
            nn.Conv3d(base_ch * 2 + ch_list[2], base_ch * 4, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 4),
            nn.InstanceNorm3d(base_ch * 4),
            nn.ReLU(),
            RCB(base_ch * 4, base_ch * 4)
        )
        self.mta_e3 = MTA_Block(base_ch * 4)
        self.rcb_e4 = nn.Sequential(
            nn.Conv3d(base_ch * 4 + ch_list[3], base_ch * 8, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 8),
            nn.InstanceNorm3d(base_ch * 8),
            nn.ReLU(),
            RCB(base_ch * 8, base_ch * 8)
        )
        self.mta_e4 = MTA_Block(base_ch * 8)
        
        self.bridge = nn.Sequential(
            nn.Conv3d(base_ch * 8 + ch_list[4], base_ch * 8, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 8),
            nn.InstanceNorm3d(base_ch * 8),
            nn.ReLU()
        )
        
        self.maxpool = nn.MaxPool3d(2, 2, 0)
        
        self.ch_c1 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch, 1, 1, 0),
            # nn.BatchNorm2d(base_ch),
            nn.InstanceNorm3d(base_ch),
            nn.ReLU()
        )
        self.mta_c1 = MTA_Block(base_ch)
        
        self.ch_c2 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 2, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 2),
            nn.InstanceNorm3d(base_ch * 2),
            nn.ReLU()
        )
        self.mta_c2 = MTA_Block(base_ch * 2)
        
        self.ch_c3 = nn.Sequential(
            nn.Conv3d(base_ch * 4, base_ch * 4, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 4),
            nn.InstanceNorm3d(base_ch * 4),
            nn.ReLU()
        )
        self.mta_c3 = MTA_Block(base_ch * 4)
        
        self.ch_c4 = nn.Sequential(
            nn.Conv3d(base_ch * 8, base_ch * 8, 1, 1, 0),
            # nn.BatchNorm2d(base_ch * 8),
            nn.InstanceNorm3d(base_ch * 8),
            nn.ReLU()
        )
        self.mta_c4 = MTA_Block(base_ch * 8)
        
        self.rcb_d1 = RCB(
            base_ch * 2, base_ch
        )
        self.rcb_d2 = RCB(
            base_ch * 4, base_ch
        )
        self.rcb_d3 = RCB(
            base_ch * 8, base_ch * 2
        )
        self.rcb_d4 = RCB(
            base_ch * 16, base_ch * 4
        )        
        
        self.Conv_1x1 = nn.Conv3d(base_ch,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x_res1, x_res2, x_res3, x_res4, x_res5 = self.baseEncoder.features(x)
        
        x_e1 = self.rcb_e1( torch.cat([x, x_res1], 1) )
        
        x_e2 = self.mta_e1(x_e1)
        x_e2 = self.maxpool(x_e2)
        x_e2 = self.rcb_e2( torch.cat([x_e2, x_res2], 1) )
        
        x_e3 = self.mta_e2(x_e2)
        x_e3 = self.maxpool(x_e3)
        x_e3 = self.rcb_e3( torch.cat([x_e3, x_res3], 1) )
        
        x_e4 = self.mta_e3(x_e3)
        x_e4 = self.maxpool(x_e4)
        x_e4 = self.rcb_e4( torch.cat([x_e4, x_res4], 1) )
        
        x_e5 = self.mta_e4(x_e4)
        x_e5 = self.maxpool(x_e5)
        x_e5 = self.bridge( torch.cat([x_e5, x_res5], 1) )
        
        x_d5 = F.interpolate( x_e5, scale_factor = 2 )
        
        x_c4 = self.ch_c4( x_e4 )
        x_c4 = x_d5 + x_c4
        x_c4 = self.mta_c4(x_c4)
        x_d4 = self.rcb_d4( torch.cat([x_c4, x_d5], 1) )
        x_d4 = F.interpolate( x_d4, scale_factor = 2 )
        
        x_c3 = self.ch_c3( x_e3 )
        x_c3 = x_d4 + x_c3
        x_c3 = self.mta_c3(x_c3)
        x_d3 = self.rcb_d3( torch.cat([x_c3, x_d4], 1) )
        x_d3 = F.interpolate( x_d3, scale_factor = 2 )
        
        x_c2 = self.ch_c2( x_e2 )
        x_c2 = x_d3 + x_c2
        x_c2 = self.mta_c2(x_c2)
        x_d2 = self.rcb_d2( torch.cat([x_c2, x_d3], 1) )
        x_d2 = F.interpolate( x_d2, scale_factor = 2 )
        
        x_c1 = self.ch_c1( x_e1 )
        x_c1 = x_d2 + x_c1
        x_c1 = self.mta_c1(x_c1)
        x_d1 = self.rcb_d1( torch.cat([x_c1, x_d2], 1) )
                
        x_out = self.Conv_1x1(x_d1)
        if self.output_ch == 1:
            return F.sigmoid(x_out)
        return F.softmax(x_out, 1)

def debug():
    data = torch.rand( [4, 1, 128, 128] )
    data = data.to('cuda')
    model = LiSegPNet(1, 2, 64)
    model.to('cuda')
    out = model(data)
    print(out.size())

if __name__ == '__main__':
    debug()