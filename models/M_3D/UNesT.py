import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

# classes

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.channel_expand = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 1, 1, 0),
            # nn.BatchNorm2d(ch_out),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_conv = self.conv(x)
        x_in = self.channel_expand(x)
        return x_conv + x_in

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        # self.attend = nn.Softmax(dim = -1)
        self.attend = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, num_patch, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.pos_embeding = nn.Parameter( torch.zeros(1, num_patch, dim) )
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        x = x + self.pos_embeding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x
    
class UNesT(nn.Module):
    def __init__(self, image_size, patch_size, ch_in = 1, num_classes = 2, heads = 32, base_ch = 32, dim_head = 64):
        super().__init__()
        im_h, im_w, im_d = pair(image_size)
        pa_h, pa_w, pa_d = pair(patch_size)
        self.to_patch_embedding = Rearrange('b c (h p1) (w p2) -> ( b p1 p2 ) ( h w ) c', p1 = pa_h, p2 = pa_w)
                        
        self.conv_c0 = ResBlock(ch_in, base_ch)        
        
        self.conv_e1 = nn.Sequential(
            ConvPool(ch_in, base_ch * 2),
            ConvPool(base_ch * 2, base_ch * 4)
        )
        self.conv_c1_t = ResBlock(base_ch * 4, base_ch * 4)
        tim_h, tim_w = im_h // 4, im_w // 4
        num_patch = (tim_h // pa_h) * (tim_w // pa_w)
        self.trans_e1 = Transformer( num_patch, base_ch * 4, 4, heads, dim_head, base_ch * 8 )
        self.to_latent1 = Rearrange('( b p1 p2 ) ( h w ) c -> b c (h p1) (w p2)', p1 = pa_h, p2 = pa_w, h = tim_h // pa_h)
        self.conv_c1_b = ResBlock(base_ch * 4, base_ch * 4)
        
        self.conv_e2 = ConvPool(base_ch * 4, base_ch * 8)
        self.conv_c2 = ResBlock(base_ch * 8, base_ch * 8)
        tim_h, tim_w = im_h // 8, im_w // 8
        num_patch = (tim_h // pa_h) * (tim_w // pa_w)
        self.trans_e2 = Transformer( num_patch, base_ch * 8, 8, heads, 64, base_ch * 16 )
        self.to_latent2 = Rearrange('( b p1 p2 ) ( h w ) c -> b c (h p1) (w p2)', p1 = pa_h, p2 = pa_w, h = tim_h // pa_h)
        
        self.conv_e3 = ConvPool(base_ch * 8, base_ch * 16)
        self.conv_c3 = ResBlock(base_ch * 16, base_ch * 16)
        tim_h, tim_w = im_h // 16, im_w // 16
        num_patch = (tim_h // pa_h) * (tim_w // pa_w)
        self.trans_e3 = Transformer( num_patch, base_ch * 16, 16, heads, 64, base_ch * 32 )
        self.to_latent3 = Rearrange('( b p1 p2 ) ( h w ) c -> b c (h p1) (w p2)', p1 = pa_h, p2 = pa_w, h = tim_h // pa_h)
        
        self.conv_e4 = nn.Sequential(
            ConvPool(base_ch * 16, base_ch * 16),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, 1, 1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU()
        )
        
        self.conv_d3 = ResBlock( base_ch * (16 + 16), base_ch * 8 )
        self.conv_d2 = ResBlock( base_ch * (8 + 8), base_ch * 4 )
        self.conv_d1_b = ResBlock( base_ch * (4 + 4), base_ch * 4 )
        self.conv_d1_t = ResBlock( base_ch * (4 + 4), base_ch )
        self.conv_d0 = ResBlock( base_ch * 2, base_ch )
        
        self.seg_head = nn.Conv2d(base_ch, num_classes, kernel_size=1,stride=1,padding=0)
        
        
    def forward(self, img):
        x_c0 = self.conv_c0(img)
        x_e1 = self.conv_e1(img)
        
        x_c1_t = self.conv_c1_t(x_e1)
        x_e1 = self.to_patch_embedding(x_e1)
        x_e1 = self.trans_e1(x_e1)
        x_e1 = self.to_latent1(x_e1)
        x_c1_b = self.conv_c1_b(x_e1)
        
        x_e2 = self.conv_e2(x_e1)
        x_e2 = self.to_patch_embedding(x_e2)
        x_e2 = self.trans_e2(x_e2)
        x_e2 = self.to_latent2(x_e2)
        x_c2 = self.conv_c2(x_e2)
        
        x_e3 = self.conv_e3(x_e2)
        x_e3 = self.to_patch_embedding(x_e3)
        x_e3 = self.trans_e3(x_e3)
        x_e3 = self.to_latent3(x_e3)
        x_c3 = self.conv_c3(x_e3)
        
        x_e4 = self.conv_e4(x_e3)
        
        x_d3 = F.interpolate(x_e4, scale_factor=2)
        x_d3 = self.conv_d3(torch.cat([x_c3, x_d3], 1))
        
        x_d2 = F.interpolate(x_d3, scale_factor=2)
        x_d2 = self.conv_d2(torch.cat([x_c2, x_d2], 1))
        
        x_d1_b = F.interpolate(x_d2, scale_factor=2)
        x_d1_b = self.conv_d1_b(torch.cat([x_c1_b, x_d1_b], 1))
        
        x_d1_t = self.conv_d1_t(torch.cat([x_c1_t, x_d1_b], 1))
        
        x_d0 = F.interpolate(x_d1_t, scale_factor=4)
        x_d0 = self.conv_d0(torch.cat([x_c0, x_d0], 1))
        
        out = self.seg_head(x_d0)
        if out.size(1) == 1:
            return F.sigmoid(out)
        return F.softmax(out, 1)

def debug():
    data = torch.rand( [2, 4, 128, 128, 128] )
    data = data.to('cuda')
    model = UNesT(image_size=128, patch_size=8, ch_in=4, num_classes=4, base_ch=16)
    model.to('cuda')
    out = model(data)
    print(out.size())

if __name__ == '__main__':
    debug()