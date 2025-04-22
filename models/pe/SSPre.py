import torch
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F
from einops import rearrange

from .penet_classifier import PENetClassifier

#w ImgEncoder
class ImgEncoder(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.penet = PENetClassifier()
        self.l1 = nn.Linear(2048, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.a1 = nn.ReLU()
        self.d1 = nn.Dropout(0.5)

        self.load_pretrained()

    def load_pretrained(self):
        module = self.penet
        des = 'load_pretrained_ImgEncoder_penet'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/ckpt/penet.pth.tar'
        
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print(des)

    def train(self, mode = True):
        module_f = self.penet
        if mode:
            for module in self.children():
                module.train(mode)
            module_f.eval()
            for param in module_f.parameters():
                param.requires_grad = False
        else:
            for module in self.children():
                module.train(mode)
    
    def forward(self, i):
        i, penet_out = self.penet(i)
        i = nn.AdaptiveAvgPool3d(1)(i)
        i = i.view(i.size(0), -1)
        i = self.l1(i)
        i = self.a1(i)
        i = self.d1(i)
        i = self.l2(i)
        i = self.a1(i)
        i = self.d1(i)
        i = self.l3(i)
        return i
    
#w TabEncoder
class TabEncoder(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.l1 = nn.Linear(49, 128)
        self.l2 = nn.Linear(128, 128)
        self.a1 = nn.ReLU()
        self.d1 = nn.Dropout(0.5)
    
    def forward(self, t):
        t = self.l1(t)
        t = self.a1(t)
        t = self.d1(t)
        t = self.l2(t)
        return t

#w Scale
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self):
        return self.scale.exp()
        
#w CLIP
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i_sh, t_sh, scale, y):
        i_sh = F.normalize(i_sh, dim = -1)
        t_sh = F.normalize(t_sh, dim = -1)
        logit_i2t = i_sh @ t_sh.T * scale
        logit_t2i = logit_i2t.T

        bs = i_sh.shape[0]
        label = y.expand(bs, bs).T.clone()
        for k in range(bs):
            if y[k] == 0:
                for j in range(bs):
                    if label[k][j] == 0:
                        label[k][j] = 1
                    else:
                        label[k][j] = 0

        q = F.softmax(label / (label.sum(dim = -1, keepdim = True)), dim = -1)
        p_i2t = F.log_softmax(logit_i2t, dim = -1)
        p_t2i = F.log_softmax(logit_t2i, dim = -1)
        kl_p1_q = F.kl_div(p_i2t, q, reduction = 'batchmean')
        kl_q_p1 = F.kl_div(q.log(), p_i2t.exp(), reduction = 'batchmean')
        kl_p2_q = F.kl_div(p_t2i, q, reduction = 'batchmean')
        kl_q_p2 = F.kl_div(q.log(), p_t2i.exp(), reduction = 'batchmean')
        return (kl_p1_q + kl_q_p1 + kl_p2_q + kl_q_p2) / 4

#w GEGLU
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)
 
#w FeedForward
def FeedForward(
    dim = 64,
    mult = 4,
    dropout = 0.2):
    #w dim:bs, 128, dim 
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

#w SA
class SA(nn.Module):
    def __init__(
        self,
        head = 4,
        dim = 64,
        dropout = 0.2):
        super().__init__()

        self.head = head
        self.scale = (dim / head) ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.l1 = nn.Linear(dim, dim, bias = False)
        self.d1 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.head
        x = self.norm(x) #w bs, 128, 64

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t:rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        dropped_attn = self.d1(attn)
        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.l1(out) 
        return out, attn
    
#w CA
class CA(nn.Module):
    def __init__(
        self,
        head = 4,
        dim = 64,
        dropout = 0.2):
        super().__init__()

        self.head = head
        self.scale = (dim / head) ** -0.5

        self.norm_i = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        self.to_qkv_i = nn.Linear(dim, dim * 3, bias = False)
        self.to_qkv_t = nn.Linear(dim, dim * 3, bias = False)
        self.to_out_i = nn.Linear(dim, dim, bias = False)
        self.to_out_t = nn.Linear(dim, dim, bias = False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, i, t):
        h = self.head
        i = self.norm_i(i)
        t = self.norm_t(t)

        q_i, k_i, v_i = self.to_qkv_i(i).chunk(3, dim = -1)
        q_t, k_t, v_t = self.to_qkv_t(t).chunk(3, dim = -1)
        q_i, k_i, v_i = map(lambda temp:rearrange(temp, 'b n (h d) -> b h n d', h = h), (q_i, k_i, v_i))
        q_t, k_t, v_t = map(lambda temp:rearrange(temp, 'b n (h d) -> b h n d', h = h), (q_t, k_t, v_t))
        q_i = q_i * self.scale
        q_t = q_t * self.scale

        sim_i2t = einsum('b h i d, b h j d -> b h i j', q_i, k_t)
        attn_i2t = sim_i2t.softmax(dim = -1)
        dropped_attn_i2t = self.dropout(attn_i2t)
        out_i2t = einsum('b h i j, b h j d -> b h i d', dropped_attn_i2t, v_t)
        out_i2t = rearrange(out_i2t, 'b h n d -> b n (h d)', h = h)
        out_i2t = self.to_out_t(out_i2t) #w tab

        sim_t2i = einsum('b h i d, b h j d -> b h i j', q_t, k_i)
        attn_t2i = sim_t2i.softmax(dim = -1)
        dropped_attn_t2i = self.dropout(attn_t2i)
        out_t2i = einsum('b h i j, b h j d -> b h i d', dropped_attn_t2i, v_i)
        out_t2i = rearrange(out_t2i, 'b h n d -> b n (h d)', h = h)
        out_t2i = self.to_out_i(out_t2i) #w img

        return torch.cat([out_t2i,out_i2t],1) #w img,tab
    
#w CASA2
class CASA2(nn.Module):
    def __init__(self):
        super().__init__()

        self.w_i = nn.Parameter(torch.randn(128, 64))
        self.b_i = nn.Parameter(torch.randn(128, 64))
        self.w_t = nn.Parameter(torch.randn(128, 64))
        self.b_t = nn.Parameter(torch.randn(128, 64))

        self.Cross1 = CA()
        self.FF1 = FeedForward()
        self.Self1 = SA()
        self.FF2 = FeedForward()
        self.Cross2 = CA()
        self.FF3 = FeedForward()
        self.Self2 = SA()
        self.FF4 = FeedForward()

    def forward(self, i, t):
        b, n = i.shape
        i = i.reshape(b, n, 1)
        t = t.reshape(b, n, 1)
        i = i * self.w_i + self.b_i
        t = t * self.w_t + self.b_t 

        residual_ = torch.cat([i, t], 1)
        fusion = self.Cross1(i, t)
        fusion = fusion + residual_ 
        residual = fusion
        fusion = self.FF1(fusion)
        fusion = fusion + residual 

        residual = fusion
        fusion, self1_attn = self.Self1(fusion)
        fusion = fusion + residual 
        residual = fusion
        fusion = self.FF2(fusion)
        fusion = fusion + residual 

        i, t = fusion.chunk(2, dim = 1)
        residual = torch.cat([i, t], 1)
        fusion = self.Cross2(i, t)
        fusion = fusion + residual 
        residual = fusion
        fusion = self.FF3(fusion)
        fusion = fusion + residual 

        residual = fusion
        fusion, self2_attn = self.Self2(fusion)
        fusion = fusion + residual 
        residual = fusion
        fusion = self.FF4(fusion)
        fusion = fusion + residual 

        self2_attn = self2_attn.sum(-1).sum(1) #w bs, 4, 256, 16 -> bs, 256 
        fusion = fusion.mean(-1) #w bs, 256, 64 -> bs, 256 
        residual_ = residual_.mean(-1) 

        topk_value, topk_idx = torch.topk(self2_attn, k = 128, dim = 1)
        fusion = fusion.gather(1, topk_idx) #w bs, 128
        residual_ = residual_.gather(1, topk_idx) #w bs, 128
        fusion = fusion + residual_

        return fusion #w bs, 128

#w Share
class Share(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale = Scale()
        self.clip = CLIP()
        self.casa2 = CASA2()

    def forward(self, i_sh, t_sh, y):
        scale = self.scale()
        loss = self.clip(i_sh, t_sh, scale, y)
        fusion = self.casa2(i_sh, t_sh)
        return fusion, loss

#w Separate
class Separate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i_sp, i_sh, t_sh, t_sp):
        loss = (torch.norm(i_sp @ t_sp.T, p = 'fro')+\
              torch.norm(i_sp @ i_sh.T, p = 'fro')+\
              torch.norm(t_sp @ t_sh.T, p = 'fro')) / 3
        return loss

#w SS
class SS(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.img_encoder = ImgEncoder()
        self.tab_encoder = TabEncoder()
        self.i_sp = nn.Linear(128, 128)
        self.i_sh = nn.Linear(128, 128)
        self.t_sh = nn.Linear(128, 128)
        self.t_sp = nn.Linear(128, 128)
    
    def forward(self, i, t):
        i = self.img_encoder(i)
        t = self.tab_encoder(t)
        i_sp = self.i_sp(i)
        i_sh = self.i_sh(i)
        t_sh = self.t_sh(t)
        t_sp = self.t_sp(t)
        return i_sp, i_sh, t_sh, t_sp

#w SSPre
class SSPre(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.ss = SS()
        self.separate = Separate()
        self.share = Share()
        self.c1 = nn.Linear(128, 1)
        self.c2 = nn.Linear(128, 1)
        self.c3 = nn.Linear(128, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

    def forward(self, i, t, y):
        i_sp, i_sh, t_sh, t_sp = self.ss(i, t)
        loss_sp = self.separate(i_sp, i_sh, t_sh, t_sp)
        fusion, loss_sh = self.share(i_sh, t_sh, y)
   
        i_sp_out = self.c1(i_sp)
        sh_out = self.c2(fusion)
        t_sp_out = self.c3(t_sp)
        loss = (loss_sp + loss_sh) / 2
        return loss, i_sp_out, sh_out, t_sp_out
    
    def args_dict(self):
        model_args = {}
        return model_args

#w SSPre_Tab
class SSPre_Tab(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.ss = SS()
        self.share = Share()
        self.t_weight = nn.Parameter(torch.randn(2))
        self.c = nn.Linear(128, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

        self.load_pretrained()

    def load_pretrained(self):
        module = self
        des = 'load_pretrained_SSPre_Tab'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/SSPre_1-40/best.pth.tar'
        
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print(des)

    def train(self, mode = True):
        module1 = self.ss
        module2 = self.share
        if mode:
            for module in self.children():
                module.train(mode)
            module1.eval()
            for param in module1.parameters():
                param.requires_grad = False
            module2.eval()
            for param in module2.parameters():
                param.requires_grad = False
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, i, t, y):
        _, i_sh, t_sh, t_sp = self.ss(i, t)
        fusion, _ = self.share(i_sh, t_sh, y)
        weight = F.softmax(self.t_weight, dim = -1)
        t_fusion = weight[0] * t_sp + weight[1] * fusion
        out = self.c(t_fusion)

        return out, weight
    
    def args_dict(self):
        model_args = {}
        return model_args

#w SSPre_Img
class SSPre_Img(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.ss = SS()
        self.share = Share()
        self.i_weight = nn.Parameter(torch.randn(2))
        self.c = nn.Linear(128, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

        self.load_pretrained()

    def load_pretrained(self):
        module = self
        des = 'load_pretrained_SSPre_Tab'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/SSPre_1-40/best.pth.tar'
        
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print(des)

    def train(self, mode = True):
        module1 = self.ss
        module2 = self.share
        if mode:
            for module in self.children():
                module.train(mode)
            module1.eval()
            for param in module1.parameters():
                param.requires_grad = False
            module2.eval()
            for param in module2.parameters():
                param.requires_grad = False
        else:
            for module in self.children():
                module.train(mode)

    def forward(self, i, t, y):
        _, i_sh, t_sh, t_sp = self.ss(i, t)
        fusion, _ = self.share(i_sh, t_sh, y)
        weight = F.softmax(self.i_weight, dim = -1)
        i_fusion = weight[0] * i_sp + weight[1] * fusion
        out = self.c(i_fusion)

        return out, weight
    
    def args_dict(self):
        model_args = {}
        return model_args

