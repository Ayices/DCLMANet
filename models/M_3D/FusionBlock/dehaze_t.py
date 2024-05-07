import torch.nn as nn
import torch
from models import trans
from loss import smooth_loss
import models.blocks as blocks


def make_model(args):
    device = 'cpu' if args.cpu else 'cuda'
    return DEHAZE_T(img_channels=args.n_colors, t_channels=args.t_channels, n_resblock=args.n_resblock,
                    n_feat=args.n_feat, device=device)


class FusionModule(nn.Module):
    def __init__(self, n_feat, kernel_size=5):
        super(FusionModule, self).__init__()
        print("Creating BRB-Fusion-Module")
        self.block1 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)
        self.block2 = blocks.BinResBlock(n_feat, kernel_size=kernel_size)

    def forward(self, x, y):
        H_0 = x + y
        x_1, y_1, H_1 = self.block1(x, y, H_0)
        x_2, y_2, H_2 = self.block2(x_1, y_1, H_1)
        return H_2

class DEHAZE_T(nn.Module):

    def __init__(self, img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device='cuda'):
        super(DEHAZE_T, self).__init__()
        self.device = device
        self.extra_feat = nn.Sequential(
            nn.Conv2d(img_channels, n_feat, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            blocks.ResBlock(n_feat, n_feat, kernel_size=5, stride=1)
        )
        # self.fusion_feat = FusionModule(n_feat=n_feat, kernel_size=5)
        self.fusion_feat = FusionModule(n_feat=n_feat, kernel_size=5)
        self.conv2=conv(in_channels=32,out_channels=3,kernel_size=1)
        self.trans=HF_Net()
        self.trans_net = trans.TRANS(in_channels=1, out_channels=t_channels,
                                     n_resblock=n_resblock, n_feat=n_feat, feat_in=True)
        self.smooth_loss = smooth_loss.Smooth_Loss()


    def forward(self, x, pre_est_J):
        # 从模糊图像 I(x) 和生成的参考图像 R(x) 中提取特征,利用残差网络ResBlock
        # x.Size([1, 3, 240, 240])
        # x_feat torch.Size([1, 32, 240, 240])
        # pre_Fison_feat torch.Size([1, 32, 240, 240])
        x_feat = self.extra_feat(x)
        pre_est_J_feat = self.extra_feat(pre_est_J)

        # 融合特征,fusioned_feat torch.Size([1, 32, 240, 240])
        fusioned_feat = self.fusion_feat(x_feat, pre_est_J_feat)
        fusioned_feat = self.conv2(fusioned_feat)

        trans_out= self.trans(fusioned_feat)
        return trans_out
