import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

# from UNesT.unest import UNesT

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

class SA_block(nn.Module):
    def __init__(self,F_s,F_q,F_ch):
        super(SA_block,self).__init__()
        sq_ch = np.maximum( 1, F_ch // 2 )
        self.W_S = nn.Sequential(
            nn.Conv3d(F_s, sq_ch, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(sq_ch),
            nn.InstanceNorm3d(sq_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(sq_ch, F_ch, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(F_ch),
            nn.InstanceNorm3d(F_ch)
            )
        
        self.W_Q = nn.Sequential(
            nn.Conv3d(F_q, sq_ch, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(sq_ch),
            nn.InstanceNorm3d(sq_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(sq_ch, F_ch, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(F_ch)
            nn.InstanceNorm3d(F_ch)
        )
                
        # self.softmax = nn.Softmax(2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_s, x_q):
        a_s = self.W_S(x_s)
        # a_s = self.softmax( a_s.reshape(a_s.size(0), a_s.size(1), -1) ).reshape( *a_s.size() )
        a_s = self.sigmoid(a_s)
        v_s = a_s * x_s
        
        a_q = self.W_Q(x_q)
        # a_q = self.softmax( a_q.reshape(a_q.size(0), a_q.size(1), -1) ).reshape( *a_q.size() )
        a_q = self.sigmoid(a_q)
        v_q = a_q * x_q 
        
        return v_s + v_q

class Dense_block(nn.Module):
    def __init__(self, ch_in, ch_out, downsample = True):
        super(Dense_block,self).__init__()
        if downsample:
            self.convIn = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=2,padding=1,bias=True),
                # nn.BatchNorm2d(ch_out),
                nn.InstanceNorm3d(ch_out),
                nn.ReLU(inplace=True)
                )
        else:
            self.convIn = nn.Sequential(
                nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
                # nn.BatchNorm2d(ch_out),
                nn.InstanceNorm3d(ch_out),
                nn.ReLU(inplace=True)
                )
        self.ConvS1 = nn.Sequential(
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )
        self.ConvS2 = nn.Sequential(
            nn.Conv3d(ch_out * 2, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )
        self.ConvS3 = nn.Sequential(
            nn.Conv3d(ch_out * 3, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )
        self.ConvS4 = nn.Sequential(
            nn.Conv3d(ch_out * 4, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )
        self.ConvS5 = nn.Sequential(
            nn.Conv3d(ch_out * 5, ch_out, kernel_size=1,stride=1,padding=0,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        if self.convIn is not None:
            x_in = self.convIn(x)
        else:
            x_in = x
        x_s1 = self.ConvS1(x_in)
        item_list = [x_in, x_s1]
        x_s1 = torch.cat(item_list, 1)
        x_s2 = self.ConvS2(x_s1)
        item_list.append(x_s2)
        x_s2 = torch.cat(item_list, 1)
        x_s3 = self.ConvS3(x_s2)
        item_list.append(x_s3)
        x_s3 = torch.cat(item_list, 1)
        x_s4 = self.ConvS4(x_s3)
        item_list.append(x_s4)
        x_s4 = torch.cat(item_list, 1)
        x_s5 = self.ConvS5(x_s4)        
        return x_s5

class SCR_block(nn.Module):
    def __init__(self, ch_in, ch_out, num_scales = 1):
        super(SCR_block,self).__init__()
        self.ConvAT_1 = nn.Sequential(
            nn.Conv3d(
                ch_in, ch_out,
                3, 1, 1
            ),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(False)
        )
        
        self.ConvAT_list = nn.ModuleList( [] )
        for s in range(1, num_scales):
            self.ConvAT_list.append(
                nn.Sequential(
                    nn.Conv3d(
                        ch_in, ch_out,
                        3, 1, 1 + s, s + 1
                    ),
                    # nn.BatchNorm2d(ch_out),
                    nn.InstanceNorm3d(ch_out),
                    nn.ReLU(False)
                )
            )
            
        self.bottleneck = nn.Sequential(
            nn.Conv3d(
                ch_out * num_scales, ch_out,
                1, 1, 0
            ),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(False)
        )
        
    def forward(self, x, x_list):
        in_list = [x]
        for tx in x_list:
            in_list.append( F.interpolate(tx, x.size()[2:]) )
        x_in = torch.cat(in_list, 1)
        x_AT1 = self.ConvAT_1( x_in )
        out_list = [ x_AT1 ]
        for convAT in self.ConvAT_list:
            tx = convAT(x_in)
            out_list.append(tx)
        x_out = torch.cat( out_list, 1 )
        x_out = self.bottleneck( x_out )
        return x_out

class MSFEF_block(nn.Module):
    def __init__( self, ch_in, ch_out ):
        super(MSFEF_block,self).__init__()
        conv_params = torch.rand([ch_out, ch_in, 3, 3, 3])
        self.conv_params = nn.Parameter(data=conv_params, requires_grad=True)
        self.sa_b12 = SA_block(
            ch_out, ch_out, ch_out
        )
        self.sa_b34 = SA_block(
            ch_out, ch_out, ch_out
        )
        self.sa_b14 = SA_block(
            ch_out, ch_out, ch_out
        )
        
    def forward(self, x):
        x_at1 = F.conv3d(x, self.conv_params, None, 1, 1, 1)
        x_at2 = F.conv3d(x, self.conv_params, None, 1, 2, 2)
        x_sa12 = self.sa_b12(x_at1, x_at2)
        
        x_at3 = F.conv3d(x, self.conv_params, None, 1, 3, 3)
        x_at4 = F.conv3d(x, self.conv_params, None, 1, 4, 4)
        x_sa34 = self.sa_b34(x_at3, x_at4)
        
        x_sa14 = self.sa_b14(x_sa12, x_sa34)
        return x_sa14

class EU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1, base_ch = 16):
        super(EU_Net,self).__init__()
        
        self.output_ch = output_ch

        self.Conv1 = Dense_block(ch_in=img_ch,ch_out=base_ch, downsample=False)
        self.Conv2 = Dense_block(ch_in=base_ch,ch_out=base_ch * 2)
        self.Conv3 = Dense_block(ch_in=base_ch * 2,ch_out=base_ch * 4)
        self.Conv4 = Dense_block(ch_in=base_ch * 4,ch_out=base_ch * 8)
        self.Conv5 = nn.Sequential(
            nn.Conv3d(  base_ch * 8,  base_ch * 16, 3, 2, 1 ),
            # nn.BatchNorm2d(base_ch * 16),
            nn.InstanceNorm3d(base_ch * 16),
            nn.ReLU()
            )
        self.MSFEF = MSFEF_block( base_ch * 16, base_ch * 16)
        
        self.SCR1 = SCR_block( base_ch * (8 + 4 + 2 + 1), base_ch, 4 )
        self.SCR2 = SCR_block( base_ch * (8 + 4 + 2), base_ch * 2, 3 )
        self.SCR3 = SCR_block( base_ch * (8 + 4), base_ch * 4, 2 )
        self.SCR4 = SCR_block( base_ch * 8, base_ch * 8, 1 )
        
        self.UPConv4 = Dense_block(base_ch * (16 + 8), base_ch * 4, False)
        self.UPConv3 = Dense_block(base_ch * 8, base_ch * 2, False)
        self.UPConv2 = Dense_block(base_ch * 4, base_ch, False)
        self.UPConv1 = Dense_block(base_ch * 2, base_ch, False)
        
        self.Conv_1x1 = nn.Conv3d(base_ch,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x_e1 = self.Conv1(x)
        x_e2 = self.Conv2(x_e1)
        x_e3 = self.Conv3(x_e2)
        x_e4 = self.Conv4(x_e3)
        x_e5 = self.Conv5(x_e4)
        x_e5 = self.MSFEF(x_e5)
        
        x_d4 = F.interpolate(x_e5, scale_factor=2)
        x_c4 = self.SCR4(x_e4, [])
        x_d4 = torch.cat([x_d4, x_c4], 1)
        x_d4 = self.UPConv4(x_d4)
        
        x_d3 = F.interpolate(x_d4, scale_factor=2)
        x_c3 = self.SCR3(x_e3, [x_e4])
        x_d3 = torch.cat([x_d3, x_c3], 1)
        x_d3 = self.UPConv3(x_d3)
        
        x_d2 = F.interpolate(x_d3, scale_factor=2)
        x_c2 = self.SCR2(x_e2, [x_e3, x_e4])
        x_d2 = torch.cat([x_d2, x_c2], 1)
        x_d2 = self.UPConv2(x_d2)
        
        x_d1 = F.interpolate(x_d2, scale_factor=2)
        x_c1 = self.SCR1(x_e1, [x_e2, x_e3, x_e4])
        x_d1 = torch.cat([x_d1, x_c1], 1)
        x_d1 = self.UPConv1(x_d1)
        
        x_out = self.Conv_1x1(x_d1)
        if self.output_ch == 1:
            return F.sigmoid(x_out)
        return F.softmax(x_out, 1)

def debug():
    data = torch.rand( [2, 4, 128, 128, 128] )
    data = data.to('cuda:1')
    model = UNesT(in_channels=4, out_channels=4)
    model.to('cuda:1')
    out = model(data)
    print(out.size())

if __name__ == '__main__':
    debug()