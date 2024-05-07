import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn import Parameter

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self,img_ch=3, output_ch=1):
        super(U_Net,self).__init__()
        
        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


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

        if self.output_ch == 1:
            return F.sigmoid(d1)
        return F.softmax(d1, 1)


class ArcUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(ArcUNet, self).__init__()

        self.output_ch = output_ch
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.ArcMargin = ArcMarginProduct(64, output_ch, s=64.0, m=0.50, easy_margin=False)

    def forward(self, x, seg_label):
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

        # d1 = self.Conv_1x1(d2)

        d2_p = d2.permute(0, 2, 3, 1)
        d2_flat = d2_p.reshape(d2_p.size(0) * d2_p.size(1) * d2_p.size(2), d2_p.size(3))
        arc_label = seg_label.reshape(seg_label.size(0) * seg_label.size(1) * seg_label.size(2)).long()
        out_arc, cosine = self.ArcMargin(d2_flat, seg_label)

        Predict = cosine.reshape(d2_p.size(0), d2_p.size(1), d2_p.size(2), 2)
        Predict = Predict.permute(0, 3, 1, 2)

        return F.softmax(Predict, 1), out_arc, arc_label

    def predict(self, x):
        with torch.no_grad():
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

            d2_p = d2.permute(0, 2, 3, 1)
            d2_flat = d2_p.reshape(d2_p.size(0) * d2_p.size(1) * d2_p.size(2), d2_p.size(3))
            cosine = self.ArcMargin.predict(d2_flat)

            Predict = cosine.reshape(d2_p.size(0), d2_p.size(1), d2_p.size(2), 2)
            Predict = Predict.permute(0, 3, 1, 2)

            return F.softmax(Predict, 1)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        #  torch.nn.functional.linear(input, weight, bias=None)
        #         # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output, cosine

    def predict(self, input):
        with torch.no_grad():
            return F.linear(F.normalize(input), F.normalize(self.weight))

    def get_features(self, input):
        x = F.normalize(input)
        return x


if __name__ == '__main__':
###########################################################################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    img = torch.randn(2, 3, 128, 128).cuda()
    net = U_Net(img_ch=3, output_ch=5).cuda()
    out = net(img)
    print(out.shape)
