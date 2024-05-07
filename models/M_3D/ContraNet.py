import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from models.M_3D.ArcLayer import ArcLayer
import matplotlib.pyplot as plt
import math
from models.M_3D.Unet_modified_3D import *
from models.M_3D.APAUNet import *
from models.M_3D.FusionTry_3D import *
from ASCLoss.project_head import ProjHead
from ASCLoss.asc_loss import BinaryDice_xent, ASC_loss


class ProjCoCoLossTry(nn.Module):
    def __init__(self, input_channel, input_pixnum):
        super(ProjCoCoLossTry, self).__init__()
        # self.ProjHead_t1 = ProjHead(input_channel, input_pixnum)
        # self.ProjHead_t1ce = ProjHead(input_channel, input_pixnum)
        # self.ProjHead_t2 = ProjHead(input_channel, input_pixnum)
        # self.ProjHead_flair = ProjHead(input_channel, input_pixnum)
        self.ProjHead = ProjHead(input_channel, input_pixnum)

    def forward(self, t1, t1ce, t2, flair):
        Bf, Cf, Hf, Wf, Df = t1.shape

        # t1 = self.ProjHead_t1(t1)
        # t1ce = self.ProjHead_t1ce(t1ce)
        # t2 = self.ProjHead_t2(t2)
        # flair = self.ProjHead_flair(flair)

        t1 = self.ProjHead(t1)
        t1ce = self.ProjHead(t1ce)
        t2 = self.ProjHead(t2)
        flair = self.ProjHead(flair)

        if Bf != 2:
            print("batchsize is not 2")
            return 0

        t1_neg = torch.cat((t1ce, t2, flair), 0)
        t2_neg = torch.cat((t1, t1ce, flair), 0)
        t1ce_neg = torch.cat((t1, t2, flair), 0)
        flair_neg = torch.cat((t1, t1ce, t2), 0)

        Bn, Cn = t1_neg.shape

        neg_t1_0 = []
        neg_t1_1 = []
        neg_t2_0 = []
        neg_t2_1 = []
        neg_t1ce_0 = []
        neg_t1ce_1 = []
        neg_flair_0 = []
        neg_flair_1 = []

        for j in range(Bn):
            neg_t1_0.append(torch.norm(t1[0] - t1_neg[j], p=1))
            neg_t1_1.append(torch.norm(t1[1] - t1_neg[j], p=1))
            neg_t1ce_0.append(torch.norm(t1ce[0] - t1ce_neg[j], p=1))
            neg_t1ce_1.append(torch.norm(t1ce[1] - t1ce_neg[j], p=1))
            neg_t2_0.append(torch.norm(t2[0] - t2_neg[j], p=1))
            neg_t2_1.append(torch.norm(t2[1] - t2_neg[j], p=1))
            neg_flair_0.append(torch.norm(flair[0] - flair_neg[j], p=1))
            neg_flair_1.append(torch.norm(flair[1] - flair_neg[j], p=1))

        loss_t1_0 = torch.norm(t1[0] - t1[1], p=1) / torch.sum(torch.Tensor(neg_t1_0))
        loss_t1_1 = torch.norm(t1[1] - t1[0], p=1) / torch.sum(torch.Tensor(neg_t1_1))
        loss_t1ce_0 = torch.norm(t1ce[0] - t1ce[1], p=1) / torch.sum(torch.Tensor(neg_t1ce_0))
        loss_t1ce_1 = torch.norm(t1ce[1] - t1ce[0], p=1) / torch.sum(torch.Tensor(neg_t1ce_1))
        loss_t2_0 = torch.norm(t2[0] - t2[1], p=1) / torch.sum(torch.Tensor(neg_t2_0))
        loss_t2_1 = torch.norm(t2[1] - t2[0], p=1) / torch.sum(torch.Tensor(neg_t2_1))
        loss_flair_0 = torch.norm(flair[0] - flair[1], p=1) / torch.sum(torch.Tensor(neg_flair_0))
        loss_flair_1 = torch.norm(flair[1] - flair[0], p=1) / torch.sum(torch.Tensor(neg_flair_1))

        return loss_t1_0 + loss_t1_1 + loss_t1ce_0 + loss_t1ce_1 + loss_t2_0 + loss_t2_1 + loss_flair_0 + loss_flair_1


class ASCLossTry(nn.Module):
    def __init__(self):
        super(ASCLossTry, self).__init__()
        self.ASCLoss = ASC_loss()

    def forward(self, t1, t1ce, t2, flair):
        loss_1 = self.ASCLoss(t1,t1ce)
        loss_2 = self.ASCLoss(t1,t2)
        loss_3 = self.ASCLoss(t1,flair)
        loss_4 = self.ASCLoss(t2,t1ce)
        loss_5 = self.ASCLoss(t2,flair)
        loss_6 = self.ASCLoss(t1ce,flair)

        return loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6


class Fusion_ProjCoCo_model_attention_basechannel(nn.Module):
    # --------------------------------模态对比学习--------------------------------------------#
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(Fusion_ProjCoCo_model_attention_basechannel, self).__init__()

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

        self.modalembedding = ProjCoCoLossTry(input_channel=16 * base_channel // 4,
                                              input_pixnum=16 * base_channel // 4 * 8 * 8 * 8)

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

        if is_train:
            loss_contrastive = self.modalembedding(img2_5, img3_5, img4_5, img1_5)

        # -----------------  模态注意力  -------------------#
        # flair  t1, t1ce, t2
        img1_5_att = self.ModalAttention1(img1_5, img2_5, img3_5, img4_5)
        img2_5_att = self.ModalAttention2(img2_5, img1_5, img3_5, img4_5)
        img3_5_att = self.ModalAttention3(img3_5, img1_5, img2_5, img4_5)
        img4_5_att = self.ModalAttention4(img4_5, img1_5, img2_5, img3_5)

        x5 = torch.cat((img1_5_att, img2_5_att, img3_5_att, img4_5_att), dim=1)
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


class Fusion_ASC_model_attention_basechannel(nn.Module):
    # --------------------------------模态对比学习--------------------------------------------#
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(Fusion_ASC_model_attention_basechannel, self).__init__()

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

        self.modalembedding = ASCLossTry()

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

        # -----------------  面积一致性对比学习  -------------------#
        loss_contrastive = 0.0

        if is_train:
            loss_contrastive = self.modalembedding(img2_5, img3_5, img4_5, img1_5)

        # -----------------  模态注意力  -------------------#
        # flair  t1, t1ce, t2
        img1_5_att = self.ModalAttention1(img1_5, img2_5, img3_5, img4_5)
        img2_5_att = self.ModalAttention2(img2_5, img1_5, img3_5, img4_5)
        img3_5_att = self.ModalAttention3(img3_5, img1_5, img2_5, img4_5)
        img4_5_att = self.ModalAttention4(img4_5, img1_5, img2_5, img3_5)

        x5 = torch.cat((img1_5_att, img2_5_att, img3_5_att, img4_5_att), dim=1)
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

class Fusion_ASCCoCo_model_attention_basechannel(nn.Module):
    # --------------------------------模态对比学习--------------------------------------------#
    def __init__(self, img_ch=4, output_ch=4, base_channel=16):
        super(Fusion_ASCCoCo_model_attention_basechannel, self).__init__()

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

        self.modalembedding1 = ASCLossTry()
        self.modalembedding2 = CoCoLossTry()

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

        # -----------------  对比学习  -------------------#
        loss_contrastive = 0.0

        if is_train:
            loss_contrastive1 = self.modalembedding1(img2_5, img3_5, img4_5, img1_5)
            loss_contrastive2 = self.modalembedding2(img2_5, img3_5, img4_5, img1_5)

            loss_contrastive = loss_contrastive1 + loss_contrastive2

        # -----------------  模态注意力  -------------------#
        # flair  t1, t1ce, t2
        img1_5_att = self.ModalAttention1(img1_5, img2_5, img3_5, img4_5)
        img2_5_att = self.ModalAttention2(img2_5, img1_5, img3_5, img4_5)
        img3_5_att = self.ModalAttention3(img3_5, img1_5, img2_5, img4_5)
        img4_5_att = self.ModalAttention4(img4_5, img1_5, img2_5, img3_5)

        x5 = torch.cat((img1_5_att, img2_5_att, img3_5_att, img4_5_att), dim=1)
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