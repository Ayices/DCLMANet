import sys
from networkx.algorithms.flow.tests.test_maxflow_large_graph import gen_pyramid
sys.path.append('./')
sys.path.append('../')
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

def Gen_kernel(func_name,kernel_size=3,sigma=1):
    if func_name=='gaussian':
        m=n=(kernel_size-1.)/2.
        y, x= np.ogrid[-m:m+1,-n:n+1]
        h=np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernelsize = 3, sigma = 1):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = Gen_kernel('gaussian', kernelsize, sigma)
#         kernel3x3 = Gen_kernel('gaussian', 3, sigma)
        
        self.kernelsize = kernelsize
        
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(self.channels, -1, -1, -1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
#         kernel3x3 = torch.FloatTensor(kernel3x3).unsqueeze(0).unsqueeze(0)
#         kernel3x3 = kernel3x3.expand(self.channels, -1, -1, -1)
#         self.weight3x3 = nn.Parameter(data=kernel3x3, requires_grad=False)
        
    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=self.kernelsize // 2, groups=self.channels)
#         x = F.conv2d(x, self.weight3x3, padding=1, groups=self.channels)
        return x

class gaussaion_downsampling_block(nn.Module):
    def __init__(self, ch_in, kernelsize = 3, sigma = 1):
        super(gaussaion_downsampling_block,self).__init__()
        self.pre_conv = nn.Sequential(
            GaussianBlurConv(ch_in, kernelsize, sigma),
#             GaussianBlurConv(ch_in, 3, sigma),
        )
         
    def forward(self,x):
        x = self.pre_conv(x)
        return x
    
class gen_pyramids(nn.Module):
    def __init__(self, ch_in = 1, num_scale = 3):
        super(gen_pyramids,self).__init__()
        self.num_scale = num_scale
        self.scales = []
        for i in range(self.num_scale):
            self.scales.append(gaussaion_downsampling_block(ch_in, 3))
         
    def forward(self,x):
        outputs = []
        for i in range(self.num_scale):
            outputs.append(self.sclaes[i](x))
        return outputs
    
    def cuda(self):
        for i in range(self.num_scale):
            self.sclaes[i] = self.sclaes[i].cuda()
    
    def eval(self):
        for i in range(self.num_scale):
            self.sclaes[i] = self.sclaes[i].eval()        

def debug():
    input = torch.zeros([1, 1, 256, 256])
    pyramids = gen_pyramids(1, 4)
    out = pyramids(input)

if __name__ == '__main__':
    debug()