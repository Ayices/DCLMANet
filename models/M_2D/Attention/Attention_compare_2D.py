import torch
from torch import nn
# from .Attention_2D.BAM import ChannelAttention_2D as BAM_ca
# from .Attention_2D.BAM import SpatialAttention_2D as BAM_sa
#
# from .Attention_2D.CBAM import ChannelAttention_2D as CBAM_ca
# from .Attention_2D.CBAM import SpatialAttention_2D as CBAM_sa
#
# from .Attention_2D.ECA import ECAAttention_2D as ECA_ca
# from .Attention_2D.MS_CAM import MS_CAM_2D as MS_CAM_ca_sa
# from .Attention_2D.SE import SEAttention_2D as SE_ca

from BAM import ChannelAttention_2D as BAM_ca
from BAM import SpatialAttention_2D as BAM_sa

from CBAM import ChannelAttention_2D as CBAM_ca
from CBAM import SpatialAttention_2D as CBAM_sa

from ECA import ECAAttention_2D as ECA_ca
from MS_CAM import MS_CAM_2D as MS_CAM_ca_sa
from SE import SEAttention_2D as SE_ca

class Attention_2D_unique(nn.Module):
    def __init__(self,
                 channel=512,
                 img_size=[4, 4],
                 fusion_num=4,
                 L=32,
                 reduction_ca_Attention=4,  #外层自适应通道注意力融合的reduction
                 reduction_sa_Attention=4,  #外层自适应空间注意力融合的reduction
                 reduction_ca=4,
                 kernel_size=7,       #CBAM_sa
                 reduction_sa=4,
                 gamma=2,             #ECA
                 b=1,                 #ECA
                 num_layers=1,        #BAM
                 dia_val=2,           #BAM
                 Attention_type='BAM',
                 ADD_type=1):
        super().__init__()
        self.ADD_type=ADD_type
        self.Attention_type = Attention_type
        self.img_size = img_size
        self.fusion_num = fusion_num
        self.d = max(L, channel // reduction_ca_Attention)
        self.dsa = max(img_size[0], (self.img_size[0]*self.img_size[1]) // reduction_sa_Attention)
        self.avgpool_2D = nn.AdaptiveAvgPool2d((1, 1))
        if self.Attention_type == 'BAM':
            self.sa_flag = True
            self.ca = BAM_ca(channel=channel, reduction=reduction_ca)
            self.sa = BAM_sa(channel=channel, reduction=reduction_sa, num_layers=num_layers, dia_val=dia_val)
        elif self.Attention_type == 'CBAM':
            self.sa_flag = True
            self.ca = CBAM_ca(channel=channel, reduction=reduction_ca)
            self.sa = CBAM_sa(kernel_size=kernel_size)
        elif self.Attention_type == 'ECA':
            self.sa_flag = False
            self.ca = ECA_ca(channel=channel, gamma=gamma, b=b)
        elif self.Attention_type == 'MS_CAM':
            self.sa_flag = False
            self.ca = MS_CAM_ca_sa(channel=channel, reduction=reduction_ca)
        elif self.Attention_type == 'SE':
            self.sa_flag = False
            self.ca = SE_ca(channel=channel, reduction=reduction_ca)

        self.fc = nn.Linear(channel, self.d)
        self.fc2 = nn.Linear(self.img_size[0]*self.img_size[1], self.dsa)
        self.fcas = nn.ModuleList([])
        self.fsas = nn.ModuleList([])
        for i in range(fusion_num):
            self.fcas.append(nn.Linear(self.d, channel))
            self.fsas.append(nn.Linear(self.dsa, self.img_size[0]*self.img_size[1]))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_input):
        x = x_input.sum(0)
        b, c, w, h = x.size()
        ca_out = self.ca(x)
        # 除了BAM,其余的注意力的sigmoid在求通道以及空间注意力的时候已经做了
        if self.Attention_type == 'BAM':
            ca_out = self.sigmoid(ca_out)
        else:
            if self.Attention_type == 'MS_CAM':
                ca_out = self.avgpool_2D(ca_out)

        weights_ca = []
        ca_out_p = self.sigmoid(self.fc(ca_out.view(b, c)))
        for fca in self.fcas:
            weight = self.sigmoid(fca(ca_out_p))
            weights_ca.append(weight.view(b, c, 1, 1))
        weights_ca = torch.stack(weights_ca, 0)

        if self.sa_flag==True:
            # 除了BAM,其余的注意力的sigmoid在求通道以及空间注意力的时候已经做了
            if self.Attention_type == 'BAM':
                sa_out = self.sigmoid(self.sa(x))
            else:
                sa_out = self.sa(x)

            sa_out_fc = self.sigmoid(self.fc2(sa_out.view(b, w * h)))

            weights_sa = []
            for fsa in self.fsas:
                weight = self.sigmoid(fsa(sa_out_fc))
                weights_sa.append(weight.view(b, 1, w, h))

            weights_sa = torch.stack(weights_sa, 0)

            # ADD_type为计算了权重之后与原始模态的结合策略
            # 其中空间注意力与通道注意力相加相乘 对于BAM和CBAM等有计算空间注意力的才有用
            # 1 残差 out=(1+weight)*x weight为空间注意力与通道注意力相加
            # 2 残差 out=(1+weight)*x weight为空间注意力与通道注意力相乘
            # 3 非残差 out=weight*x weight为空间注意力与通道注意力相加
            # 4 非残差 out=weight*x weight为空间注意力与通道注意力相乘
            if self.ADD_type==1:
                # 空间注意力与通道注意力相加
                weights_all_model = weights_ca + weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * (1 + weights_all_model)).sum(0)
            elif self.ADD_type==2:
                # 空间注意力与通道注意力相乘
                weights_all_model = weights_ca * weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * (1 + weights_all_model)).sum(0)
            elif self.ADD_type==3:
                # 空间注意力与通道注意力相加
                weights_all_model = weights_ca + weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * weights_all_model).sum(0)
            elif self.ADD_type==4:
                # 空间注意力与通道注意力相乘
                weights_all_model = weights_ca * weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * weights_all_model).sum(0)
            else:
                print('Wrong ADD_type!')
        else:
            # 空间注意力与通道注意力相加相乘 对于BAM和CBAM等有计算空间注意力的才有用
            if self.ADD_type==1 or self.ADD_type==2:
                # 空间注意力与通道注意力相加
                weights_all_model = self.softmax(weights_ca)
                out = (x_input * (1 + weights_all_model)).sum(0)
            elif self.ADD_type==3 or self.ADD_type==4:
                # 空间注意力与通道注意力相乘
                weights_all_model = self.softmax(weights_ca)
                out = (x_input * weights_all_model).sum(0)
            else:
                print('Wrong ADD_type!')
        return out.contiguous()

class iAttention_2D_unique(nn.Module):
    def __init__(self,
                 channel=512,
                 img_size=[4, 4],
                 fusion_num=4,
                 L=32,
                 reduction_ca_Attention=4,  #外层自适应通道注意力融合的reduction
                 reduction_sa_Attention=4,  #外层自适应空间注意力融合的reduction
                 reduction_ca=4,
                 kernel_size=7,       #CBAM_sa
                 reduction_sa=4,
                 gamma=2,             #ECA
                 b=1,                 #ECA
                 num_layers=1,        #BAM
                 dia_val=2,           #BAM
                 Attention_type='BAM',
                 ADD_type=1):
        super().__init__()
        self.ADD_type = ADD_type
        self.Attention_type = Attention_type
        self.img_size = img_size
        self.fusion_num = fusion_num
        self.d = max(L, channel // reduction_ca_Attention)
        self.dsa = max(img_size[0], (self.img_size[0]*self.img_size[1]) // reduction_sa_Attention)
        self.avgpool_2D = nn.AdaptiveAvgPool2d((1, 1))
        if self.Attention_type == 'BAM':
            self.sa_flag = True
            self.ca = BAM_ca(channel=channel, reduction=reduction_ca)
            self.sa = BAM_sa(channel=channel, reduction=reduction_sa, num_layers=num_layers, dia_val=dia_val)
        elif self.Attention_type == 'CBAM':
            self.sa_flag = True
            self.ca = CBAM_ca(channel=channel, reduction=reduction_ca)
            self.sa = CBAM_sa(kernel_size=kernel_size)
        elif self.Attention_type == 'ECA':
            self.sa_flag = False
            self.ca = ECA_ca(channel=channel, gamma=gamma, b=b)
        elif self.Attention_type == 'MS_CAM':
            self.sa_flag = False
            self.ca = MS_CAM_ca_sa(channel=channel, reduction=reduction_ca)
        elif self.Attention_type == 'SE':
            self.sa_flag = False
            self.ca = SE_ca(channel=channel, reduction=reduction_ca)

        self.fc = nn.Linear(channel, self.d)
        self.fc2 = nn.Linear(self.img_size[0]*self.img_size[1], self.dsa)
        self.fcas = nn.ModuleList([])
        self.fsas = nn.ModuleList([])
        for i in range(fusion_num):
            self.fcas.append(nn.Linear(self.d, channel))
            self.fsas.append(nn.Linear(self.dsa, self.img_size[0]*self.img_size[1]))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

        self.Attention = Attention_2D_unique(
            channel=channel,
            img_size=img_size,
            fusion_num=fusion_num,
            L=L,
            reduction_ca_Attention=reduction_ca_Attention,  # 外层自适应通道注意力融合的reduction
            reduction_sa_Attention=reduction_sa_Attention,  # 外层自适应空间注意力融合的reduction
            reduction_ca=reduction_ca,
            kernel_size=kernel_size,  # CBAM_sa
            reduction_sa=reduction_sa,
            gamma=gamma,  # ECA
            b=b,  # ECA
            num_layers=num_layers,  # BAM
            dia_val=dia_val,  # BAM
            Attention_type=Attention_type,
            ADD_type=ADD_type)

    def forward(self, x_input):
        x = self.Attention(x_input)
        b, c, w, h = x.size()
        ca_out = self.ca(x)

        # 除了BAM,其余的注意力的sigmoid在求通道以及空间注意力的时候已经做了
        if self.Attention_type == 'BAM':
            ca_out = self.sigmoid(ca_out)
        else:
            if self.Attention_type == 'MS_CAM':
                ca_out = self.avgpool_2D(ca_out)

        weights_ca = []
        ca_out_p = self.sigmoid(self.fc(ca_out.view(b, c)))
        for fca in self.fcas:
            weight = self.sigmoid(fca(ca_out_p))
            weights_ca.append(weight.view(b, c, 1, 1))
        weights_ca = torch.stack(weights_ca, 0)

        if self.sa_flag==True:
            # 除了BAM,其余的注意力的sigmoid在求通道以及空间注意力的时候已经做了
            if self.Attention_type == 'BAM':
                sa_out = self.sigmoid(self.sa(x))
            else:
                sa_out = self.sa(x)

            sa_out_fc = self.sigmoid(self.fc2(sa_out.view(b, w * h)))

            weights_sa = []
            for fsa in self.fsas:
                weight = self.sigmoid(fsa(sa_out_fc))
                weights_sa.append(weight.view(b, 1, w, h))

            weights_sa = torch.stack(weights_sa, 0)

            # ADD_type为计算了权重之后与原始模态的结合策略
            # 其中空间注意力与通道注意力相加相乘 对于BAM和CBAM等有计算空间注意力的才有用
            # 1 残差 out=(1+weight)*x weight为空间注意力与通道注意力相加
            # 2 残差 out=(1+weight)*x weight为空间注意力与通道注意力相乘
            # 3 非残差 out=weight*x weight为空间注意力与通道注意力相加
            # 4 非残差 out=weight*x weight为空间注意力与通道注意力相乘
            if self.ADD_type==1:
                # 空间注意力与通道注意力相加
                weights_all_model = weights_ca + weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * (1 + weights_all_model)).sum(0)
            elif self.ADD_type==2:
                # 空间注意力与通道注意力相乘
                weights_all_model = weights_ca * weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * (1 + weights_all_model)).sum(0)
            elif self.ADD_type==3:
                # 空间注意力与通道注意力相加
                weights_all_model = weights_ca + weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * weights_all_model).sum(0)
            elif self.ADD_type==4:
                # 空间注意力与通道注意力相乘
                weights_all_model = weights_ca * weights_sa
                weights_all_model = self.softmax(weights_all_model)
                out = (x_input * weights_all_model).sum(0)
            else:
                print('Wrong ADD_type!')
        else:
            # 空间注意力与通道注意力相加相乘 对于BAM和CBAM等有计算空间注意力的才有用
            if self.ADD_type==1 or self.ADD_type==2:
                # 空间注意力与通道注意力相加
                weights_all_model = self.softmax(weights_ca)
                out = (x_input * (1 + weights_all_model)).sum(0)
            elif self.ADD_type==3 or self.ADD_type==4:
                # 空间注意力与通道注意力相乘
                weights_all_model = self.softmax(weights_ca)
                out = (x_input * weights_all_model).sum(0)
            else:
                print('Wrong ADD_type!')
        return out.contiguous()



if __name__ == '__main__':
    # BAM CBAM ECA MS_CAM SE
    Attention_type = 'SE'
    ADD_type = 4
    input=torch.randn(4, 50, 512, 4, 4)
    Attention_fusion = Attention_2D_unique(channel=512, Attention_type=Attention_type, fusion_num=4, ADD_type=ADD_type)
    output = Attention_fusion(input)
    print(output.shape)

    input=torch.randn(4, 50, 512, 4, 4)
    Attention_fusion = iAttention_2D_unique(channel=512, Attention_type=Attention_type, fusion_num=4, ADD_type=ADD_type)
    output = Attention_fusion(input)
    print(output.shape)