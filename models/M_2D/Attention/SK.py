import torch
from torch import nn
from collections import OrderedDict

class ChannelAttention_3D(nn.Module):
    def __init__(self, channel, reduction=16, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        if self.unsqueeze==True:
            output = output.expand_as(x)
        return output.contiguous()


class SpatialAttention_3D(nn.Module):
    def __init__(self, kernel_size=7, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        if self.unsqueeze==True:
            output = output.expand_as(x)
        return output.contiguous()


class CBAMAttention_3D(nn.Module):
    def __init__(self, channel=512,  img_size=3, fusion_num=4, kernel_size=49, L=32, reduction=4, reduction_ca=4, reduction_sa=4):
        super().__init__()
        self.img_size = img_size
        self.d = max(L, channel // reduction_ca)
        self.dsa = max(img_size*img_size, (3*img_size*img_size) // reduction_sa)
        self.fusion_num = fusion_num
        self.ca = ChannelAttention_3D(channel=channel, reduction=reduction)
        self.sa = SpatialAttention_3D(kernel_size=kernel_size)

        self.fc = nn.Linear(channel, self.d)
        self.fc2 = nn.Linear(3 * self.img_size * self.img_size, self.dsa)
        self.fcas = nn.ModuleList([])
        self.fsas = nn.ModuleList([])
        for i in range(fusion_num):
            self.fcas.append(nn.Linear(self.d, channel))
            self.fsas.append(nn.Linear(self.dsa, 3*self.img_size * self.img_size))

        self.sigmoid = nn.Sigmoid()
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x_input):
        x = x_input.sum(0)
        b, c, w, h, l = x.size()
        ca_out = self.ca(x)

        weights_ca = []
        ca_out_p = self.fc(ca_out.view(b, c))
        for fca in self.fcas:
            weight = self.sigmoid(fca(ca_out_p))
            weights_ca.append(weight.view(b, c, 1, 1, 1))

        weights_ca = torch.stack(weights_ca, 0)

        sa_out = self.sa(x)
        sa_out_fc = self.fc2(sa_out.view(b, w * h * l))

        weights_sa = []
        for fsa in self.fsas:
            weight = self.sigmoid(fsa(sa_out_fc))
            weights_sa.append(weight.view(b, 1, w, h, l))

        weights_sa = torch.stack(weights_sa, 0)

        weights_all_model = weights_ca * weights_sa
        weights_all_model = self.softmax(weights_all_model)
        weights_all_model = weights_all_model.view(self.fusion_num, b, c,  w, h, l)

        out = (x_input * weights_all_model).sum(0)
        return out.contiguous()


class SKAttention_3D(nn.Module):
    def __init__(self, channel=512,kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv3d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm3d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)


    def forward(self, x):
        bs, c, _, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs, c, 1, 1, 1)) #bs,channel
        attention_weughts=torch.stack(weights, 0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V


class iSKAttention_3D(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32, fusion_num=4):
        super().__init__()
        self.d = max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        self.fusion_num = fusion_num
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channel, channel//fusion_num, kernel_size=k, padding=k//2, groups=group)),
                    ('bn', nn.BatchNorm3d(channel//fusion_num)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel//fusion_num, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel//fusion_num))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs, c//self.fusion_num, 1, 1, 1)) #bs,channel
        attention_weughts=torch.stack(weights, 0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V


class iterSKAttention_3D(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=4, group=1, L=32, fusion_num=4):
        super().__init__()
        self.kernels_len = len(kernels)
        self.d = max(L, channel//reduction)
        self.convs = nn.ModuleList([])
        self.fusion_num = fusion_num
        self.attention = CBAMAttention_3D(channel=channel//fusion_num,
                                         img_size=4,
                                         reduction=reduction,
                                         kernel_size=5,
                                         fusion_num=self.kernels_len)
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channel, channel//fusion_num, kernel_size=k, padding=k//2, groups=group)),
                    ('bn', nn.BatchNorm3d(channel//fusion_num)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel//fusion_num, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel//fusion_num))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _, _ = x.size()
        conv_outs = []

        for conv in self.convs:                                        #不同感受野输入，存入同一个列表
            conv_outs.append(conv(x))
        features = torch.stack(conv_outs, 0)                           #不同感受野输出stack在一起

        Attention_out = self.attention(features)                       #不同感受野输出进行注意力加权融合
        Attention_fc=Attention_out.mean(-1).mean(-1).mean(-1)          #加权融合的多感受野特征作为自适应感受野权重的基础
        fc_out_weight = self.fc(Attention_fc)

        weights = []
        for fc in self.fcs:
            weight = fc(fc_out_weight)
            weights.append(weight.view(bs, c//self.fusion_num, 1, 1, 1))
        wild_weights_in = torch.stack(weights, 0)
        wild_weights = self.softmax(wild_weights_in)

        out = (wild_weights*features).sum(0)
        return out


class iterSKAttention_3D_2(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=4, group=1, L=32, fusion_num=4, img_size=4):
        super().__init__()
        self.kernels_len = len(kernels)
        self.d = max(L, channel//reduction)
        self.convs = nn.ModuleList([])
        self.fusion_num = fusion_num
        self.attention = CBAMAttention_3D(channel=channel//fusion_num,
                                         img_size=img_size,
                                         reduction=reduction,
                                         kernel_size=5,
                                         fusion_num=self.kernels_len)
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv3d(channel, channel//fusion_num, kernel_size=k, padding=k//2, groups=group)),
                    ('bn', nn.BatchNorm3d(channel//fusion_num)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel//fusion_num, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel//fusion_num))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _, _ = x.size()
        conv_outs = []

        for conv in self.convs:                                        #不同感受野输入，存入同一个列表
            conv_outs.append(conv(x))
        features = torch.stack(conv_outs, 0)                           #不同感受野输出stack在一起

        Attention_out = self.attention(features)                       #不同感受野输出进行注意力加权融合
        Attention_fc=Attention_out.mean(-1).mean(-1).mean(-1)          #加权融合的多感受野特征作为自适应感受野权重的基础
        fc_out_weight = self.fc(Attention_fc)

        weights = []
        for fc in self.fcs:
            weight = fc(fc_out_weight)
            weights.append(weight.view(bs, c//self.fusion_num, 1, 1, 1))
        wild_weights_in = torch.stack(weights, 0)
        wild_weights = self.softmax(wild_weights_in)

        out = (wild_weights*features).sum(0)
        return out


if __name__ == '__main__':
    input=torch.randn(50, 512 * 4, 3, 4, 4)
    se = iSKAttention_3D(channel=512 * 4, kernels=[1, 3], reduction=8, fusion_num=4)
    output=se(input)
    print(output.shape)