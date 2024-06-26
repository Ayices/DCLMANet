# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.resnet_os import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from module.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        self.resnet = ResNet18_OS8() # NOTE! specify the type of ResNet here
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

        if num_classes == 1:
            self.Softmax_layer = nn.Sigmoid()
        else:
            self.Softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))

        return self.Softmax_layer(output)

if __name__ == '__main__':
###########################################################################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 3, 480, 480).cuda()
    net = DeepLabV3(num_classes=5).cuda()
    out = net(img)
    print(out.shape)
