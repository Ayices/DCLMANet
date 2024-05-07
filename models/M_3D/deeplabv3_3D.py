import torch
import torch.nn as nn
import torch.nn.functional as F

from .module.resnet_3D import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from .module.aspp_3D import ASPP, ASPP_Bottleneck


class DeepLabV3_3D(nn.Module):
    def __init__(self, num_classes, input_channels, resnet, last_activation=None):
        super(DeepLabV3_3D, self).__init__()
        self.num_classes = num_classes
        self.last_activation = last_activation

        if resnet.lower() == 'resnet18_os16':
            self.resnet = ResNet18_OS16(input_channels)
        
        elif resnet.lower() == 'resnet34_os16':
            self.resnet = ResNet34_OS16(input_channels)
        
        elif resnet.lower() == 'resnet50_os16':
            self.resnet = ResNet50_OS16(input_channels)
        
        elif resnet.lower() == 'resnet101_os16':
            self.resnet = ResNet101_OS16(input_channels)
        
        elif resnet.lower() == 'resnet152_os16':
            self.resnet = ResNet152_OS16(input_channels)
        
        elif resnet.lower() == 'resnet18_os8':
            self.resnet = ResNet18_OS8(input_channels)
        
        elif resnet.lower() == 'resnet34_os8':
            self.resnet = ResNet34_OS8(input_channels)

        if resnet.lower() in ['resnet50_os16', 'resnet101_os16', 'resnet152_os16']:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        else:
            self.aspp = ASPP(num_classes=self.num_classes)

        if self.last_activation.lower() == 'sigmoid':
            self.final_layer = nn.Sigmoid()

        elif self.last_activation.lower() == 'softmax':
            self.final_layer = nn.Softmax(dim=1)

    def forward(self, x):

        h = x.size()[2]
        w = x.size()[3]
        c = x.size()[4]

        feature_map = self.resnet(x)

        output = self.aspp(feature_map)

        output = F.interpolate(output, size=(h, w, c), mode='trilinear', align_corners=True)

        output = self.final_layer(output)
        return output

if __name__ == '__main__':
###########################################################################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 3, 128, 128, 128).cuda()
    net = DeepLabV3_3D(input_channels=3, num_classes=5, resnet='resnet18_os16', last_activation='softmax').cuda()
    out = net(img)
    print(out.shape)