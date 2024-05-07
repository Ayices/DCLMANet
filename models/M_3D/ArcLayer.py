import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn import Parameter


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


class ArcLayer(nn.Module):
    def __init__(self, inchannel=3, outchannel=1):
        super(ArcLayer, self).__init__()
        self.outchannel = outchannel
        self.ArcMargin = ArcMarginProduct(inchannel, outchannel, s=64.0, m=0.50, easy_margin=False)

    def forward(self, x, seg_label):
        # 对于特征A=(Batchsize, 64, 128, 128, 128)，
        # 经过最后一层卷积之后变为seg=(Batchsize, outchannel, 128, 128, 128)
        # 则这里的输入x即为A
        # seg_label为分割标签，需要为onehot之前的 只包含0 1 2 3等数字的矩阵。
        # encoding path
        # (Batchsize, 64, 128, 128, 128)->(Batchsize, 128, 128, 128, 64)
        x_p = x.permute(0, 2, 3, 4, 1)
        # (Batchsize, 64, 128, 128, 128)->(Batchsize, 128 * 128 * 128, 64)
        x_pf = x_p.reshape(-1, x_p.size(4))
        # 分割标签应为 (Batchsize, 128, 128, 128)
        # (Batchsize, 128, 128, 128)->(Batchsize * 128 * 128 * 128)
        arc_label = seg_label.reshape(seg_label.size(0) * seg_label.size(1) *
                                      seg_label.size(2) * seg_label.size(3)).long()
        out_arc, cosine = self.ArcMargin(x_pf, seg_label)
        Predict = cosine.reshape(x_p.size(0), x_p.size(1), x_p.size(2), x_p.size(3), self.outchannel)
        Predict = Predict.permute(0, 4, 1, 2, 3)
        return F.softmax(Predict, 1), out_arc, arc_label

    def predict(self, x):
        with torch.no_grad():
            x_p = x.permute(0, 2, 3, 4, 1)
            # (Batchsize, 64, 128, 128, 128)->(Batchsize, 128 * 128 * 128, 64)
            x_pf = x_p.reshape(-1, x_p.size(4))
            cosine = self.ArcMargin.predict(x_pf)
            Predict = cosine.reshape(x_p.size(0), x_p.size(1), x_p.size(2), x_p.size(3), self.outchannel)
            Predict = Predict.permute(0, 4, 1, 2, 3)
            return F.softmax(Predict, 1)

if __name__ == '__main__':
###########################################################################
    from torch.nn import CrossEntropyLoss
    loss_Cross = CrossEntropyLoss(reduction='mean')
    # loss_DC = SoftDiceLoss(smooth=1e-5, do_bg=False)
    #
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")
    #
    # img = torch.randn(2, 64, 32, 32, 32).cuda()
    # seg_label = torch.randn(2, 32, 32, 32).cuda()
    # Arc = ArcLayer(inchannel=64, outchannel=3).cuda()
    # out, out_arc, arc_label = Arc(img, seg_label)
    #
    # # loss计算过程
    # loss = loss_DC(out, seg_label) + loss_Cross(out_arc, arc_label)
    # print(out.shape)
    #
    # # 测试过程
    # seg = Arc.predict(img)