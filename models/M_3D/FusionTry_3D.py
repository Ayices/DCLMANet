import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.M_3D.ArcLayer import ArcLayer
import matplotlib.pyplot as plt
import math
from models.M_3D.Unet_modified_3D import *
from models.M_3D.APAUNet import *
from torch.nn import CrossEntropyLoss
from PIL import Image
import os
import seaborn as sns
import pandas as pd

def Normalization(image):
    image = ((image - image.min()) / (image.max() - image.min()))
    return image

class EmbeddingContrasiveLoss(nn.Module):
    def __init__(self, tempture=0.7, epsion=1e-4):
        super(EmbeddingContrasiveLoss, self).__init__()
        self.tempture = tempture
        self.ep = epsion

    def forward(self, z, pair_mask):
        iner = torch.mm(z, z.transpose(1, 0))
        length_z = torch.sqrt(torch.sum(torch.square(z), 1, keepdim=True))
        length_z = torch.mm(length_z, length_z.transpose(1, 0))

        cos = iner / (length_z + self.ep)
        cos_p = cos.unsqueeze(-1) * pair_mask
        loss = torch.exp(cos_p / self.tempture).sum((0, 1)) \
               / torch.sum(torch.exp(cos / self.tempture))
        loss = -torch.log(loss) / pair_mask.sum((0, 1))
        return loss.mean()


def jigsaw(datas, row_offset=5, col_offset=5):
    dim = len(np.shape(datas))
    if dim == 4:
        num_row, num_col, h, w = np.shape(datas)
        channel = 1
        results = np.zeros([num_row * h + (num_row - 1) * row_offset,
                            num_col * w + (num_col - 1) * col_offset]) + 1
    else:
        num_row, num_col, h, w, channel = np.shape(datas)
        results = np.zeros([num_row * h + (num_row - 1) * row_offset,
                            num_col * w + (num_col - 1) * col_offset,
                            channel]) + 1

    for r in range(num_row):
        for c in range(num_col):
            results[r * row_offset + r * h: r * row_offset + r * h + h,
            c * col_offset + c * w: c * col_offset + c * w + w
            ] = datas[r, c]
    return results

class EmbeddingContrasiveLoss_APA(nn.Module):
    def __init__(self, tempture=0.7, epsion=1e-4):
        super(EmbeddingContrasiveLoss_APA, self).__init__()
        self.tempture = tempture
        self.ep = epsion
        self.num = 0
        self.list_num = 0
        self.img_draw = torch.zeros([1,1,4,4], dtype=torch.float)


    def forward(self, z, pair_mask):
        iner = torch.bmm(z, z.permute(0, 2, 1))
        length_z = torch.sqrt(torch.sum(torch.square(z), 2, keepdim=True))
        length_z = torch.bmm(length_z, length_z.permute(0, 2, 1))

        cos = iner / (length_z + self.ep)
        cos_p = cos.unsqueeze(-1) * pair_mask.unsqueeze(0)


        loss = torch.exp(cos_p / self.tempture).sum((1, 2)) \
               / (torch.exp(cos / self.tempture).sum((1, 2))).unsqueeze(-1)
        loss = -torch.log(loss) / pair_mask.sum((0, 1)).unsqueeze(0)

        return loss.mean(-1).mean()

class EmbeddingContrasiveLoss_CoCo(nn.Module):
    def __init__(self, tempture=0.7, epsion=1e-4):
        super(EmbeddingContrasiveLoss_CoCo, self).__init__()
        self.tempture = tempture
        self.ep = epsion

    def forward(self, z, pair_mask):
        # iner = torch.mm(z, z.transpose(1, 0))
        # length_z = torch.sqrt(torch.sum(torch.square(z), 1, keepdim=True))
        # length_z = torch.mm(length_z, length_z.transpose(1, 0))
        #
        # cos = iner / (length_z + self.ep)
        # cos_p = cos.unsqueeze(-1) * pair_mask
        # loss = torch.exp(cos_p / self.tempture).sum((0, 1)) \
        #        / torch.sum(torch.exp(cos / self.tempture))
        # loss = -torch.log(loss) / pair_mask.sum((0, 1))

        # ----------------------------------- #
        iner = torch.norm(z.unsqueeze(1) - z.unsqueeze(2),p=1, dim=-1) + self.ep

        iner_p = iner.unsqueeze(-1) * pair_mask.unsqueeze(0)

        loss = (iner_p / self.tempture).sum((1, 2)) \
               / ((iner / self.tempture).sum((1, 2))).unsqueeze(-1)
        loss = loss / pair_mask.sum((0, 1)).unsqueeze(0)

        return loss.sum()
        # return loss

import numpy as np


def gen_pair_mask(num_list):
    num_data = np.sum(num_list)
    mask = np.zeros([num_data, num_data, len(num_list)])
    tnum = 0
    for i, num_clas in enumerate(num_list):
        mask[tnum: tnum + num_clas, tnum: tnum + num_clas, i] = 1
        tnum = tnum + num_clas
    return mask


def gen_pair_mask_APA(num_list):
    num_data = np.sum(num_list)
    mask = np.zeros([num_data, num_data, len(num_list)])
    tnum = 0
    for i, num_clas in enumerate(num_list):
        mask[tnum: tnum + num_clas, tnum: tnum + num_clas, i] = 1
        tnum = tnum + num_clas
    return mask

class DCL_MANet(nn.Module):
    # --------------------------------模态对比学习--------------------------------------------#
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(DCL_MANet, self).__init__()

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

        # self.modalembedding = EmbeddingContrasiveLoss()
        self.modalembedding = EmbeddingContrasiveLoss_APA()

        self.ModalAttention1 = Modal_Attention(in_ch=16 * base_channel // 4, out_ch=16 * base_channel // 4)
        self.ModalAttention2 = Modal_Attention(in_ch=16 * base_channel // 4, out_ch=16 * base_channel // 4)
        self.ModalAttention3 = Modal_Attention(in_ch=16 * base_channel // 4, out_ch=16 * base_channel // 4)
        self.ModalAttention4 = Modal_Attention(in_ch=16 * base_channel // 4, out_ch=16 * base_channel // 4)

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

    def forward(self, x, is_train):
        # encoding path
        img1, img2, img3, img4 = x[:, 0, :, :, :].unsqueeze(dim=1), x[:, 1, :, :, :].unsqueeze(dim=1), x[:, 2, :, :, :] \
            .unsqueeze(dim=1), x[:, 3, :, :, :].unsqueeze(dim=1)

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

        # # --------------------------- 对比学习 --------------------------------------#
        # loss_contrastive = 0.0
        # if is_train:
        #     t1_view = img1_5.view(img1_5.size(0), -1)
        #     t1ce_view = img2_5.view(img2_5.size(0), -1)
        #     t2_view = img3_5.view(img3_5.size(0), -1)
        #     flair_view = img4_5.view(img4_5.size(0), -1)
        #
        #     modalconcat = torch.cat((t1_view, t1ce_view, t2_view, flair_view), dim=0)
        #     modalist = [img1_5.size(0), img2_5.size(0), img3_5.size(0), img4_5.size(0)]
        #
        #     modallist_mask = gen_pair_mask(modalist)
        #     modallist_mask = torch.tensor(modallist_mask, dtype=torch.float32, device=img4_5.device,
        #                                   requires_grad=False)
        #
        #     loss_contrastive = self.modalembedding(modalconcat, modallist_mask)

        # -----------------  密集对比学习  -------------------#
        loss_contrastive = 0.0

        B, C, W, H, D = img1_5.shape

        if is_train:
            t1_view = img1_5.view(B, C, -1).permute(2, 0, 1)
            t1ce_view = img2_5.view(B, C, -1).permute(2, 0, 1)
            t2_view = img3_5.view(B, C, -1).permute(2, 0, 1)
            flair_view = img4_5.view(B, C, -1).permute(2, 0, 1)

            # modalconcat = torch.cat((t1_view, t1ce_view, t2_view, flair_view), dim=0)
            modalconcat = torch.cat((t1_view, t1ce_view, t2_view, flair_view), dim=1)

            modalist = [B, B, B, B]
            # modalist = [img1_5.size(0), img2_5.size(0), img3_5.size(0), img4_5.size(0)]

            modallist_mask = gen_pair_mask_APA(modalist)
            modallist_mask = torch.tensor(modallist_mask, dtype=torch.float32, device=img4_5.device,
                                          requires_grad=False)

            loss_contrastive = self.modalembedding(modalconcat, modallist_mask)

        # -----------------  模态注意力  -------------------#
        # flair  t1, t1ce, t2
        img1_5_att = self.ModalAttention1(img1_5, img2_5, img3_5, img4_5)
        img2_5_att = self.ModalAttention2(img2_5, img1_5, img3_5, img4_5)
        img3_5_att = self.ModalAttention3(img3_5, img1_5, img2_5, img4_5)
        img4_5_att = self.ModalAttention4(img4_5, img1_5, img2_5, img3_5)

        x5 = torch.cat((img1_5_att, img2_5_att, img3_5_att, img4_5_att), dim=1)
        # x5 = torch.cat((img1_5, img2_5, img3_5, img4_5), dim=1)
        # -----------------  decoder  -------------------#

        x4 = torch.cat((img1_4, img2_4, img3_4, img4_4), dim=1)
        x3 = torch.cat((img1_3, img2_3, img3_3, img4_3), dim=1)
        x2 = torch.cat((img1_2, img2_2, img3_2, img4_2), dim=1)
        x1 = torch.cat((img1_1, img2_1, img3_1, img4_1), dim=1)

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

        if is_train:
            return self.active(d1), loss_contrastive
        else:
            return self.active(d1)

class Modal_Attention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Modal_Attention, self).__init__()
        self.conv_feat1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_feat2 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_feat3 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )
        self.conv_feat4 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1),
            nn.InstanceNorm3d(in_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.loss_softmax = nn.Softmax(dim=0)
        self.beta = nn.Parameter(torch.ones(3), requires_grad=True)
        self.MPA_Attention = MPA(in_ch)

    def forward(self, modal1, modal2, modal3, modal4):
        modal1_feat = self.conv_feat1(modal1)
        modal2_feat = self.conv_feat2(modal2)
        modal3_feat = self.conv_feat3(modal3)
        modal4_feat = self.conv_feat4(modal4)

        modal2_attn = self.MPA_Attention(modal1_feat, modal2_feat)
        modal3_attn = self.MPA_Attention(modal1_feat, modal3_feat)
        modal4_attn = self.MPA_Attention(modal1_feat, modal4_feat)

        attn_number = self.loss_softmax(self.beta)
        attn = (modal2_attn * attn_number[0] + modal3_attn * attn_number[1] + modal4_attn * attn_number[2]) / attn_number.sum()
        return attn + modal1


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img = torch.randn(2, 4, 128, 128, 128).cuda()
    net = DCL_MANet(img_ch=4, output_ch=4, base_channel=16).cuda()

    out = net(img, is_train=True)
    print(out.shape)
