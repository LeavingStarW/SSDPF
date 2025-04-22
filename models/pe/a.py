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
        logit_t2i = t_sh @ i_sh.T

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

#w Separate
class Separate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i_sp, i_sh, t_sh, t_sp):
        loss = (torch.norm(t_sp @ t_sh.T, p = 'fro')+\
              torch.norm(i_sp @ i_sh.T, p = 'fro')) / 2
        return loss

#w SS
class SS(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.i_sp = nn.Linear(128, 128)
        self.i_sh = nn.Linear(128, 128)
        self.t_sh = nn.Linear(128, 128)
        self.t_sp = nn.Linear(128, 128)
    
    def forward(self, i, t):
        if i is None:
            pass
        else:
            i_sp = self.i_sp(i)
            i_sh = self.i_sh(i)
        if t is None:
            pass
        else:
            t_sh = self.t_sh(t)
            t_sp = self.t_sp(t)
        #w return i_sp, i_sh, t_sh, t_sp
        return i_sp, i_sh 

#w StageOne2
class StageOne2(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.img_encoder = ImgEncoder()
        self.tab_encoder = TabEncoder()
        self.scale1 = Scale()
        self.scale2 = Scale()
        self.clip = CLIP()
        self.ss = SS()
        self.separate = Separate()
        self.c1 = nn.Linear(128, 1)
        self.c2 = nn.Linear(128, 1)
        self.c3 = nn.Linear(128, 1)
        self.c4 = nn.Linear(128, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

    def forward(self, i, t, y):
        i = self.img_encoder(i)
        t = self.tab_encoder(t)
        scale1 = self.scale1()
        loss1 = self.clip(i, t, scale1, y)

        i_sp, i_sh, t_sh, t_sp = self.ss(i, t)
        loss_sp = self.separate(i_sp, i_sh, t_sh, t_sp)
        scale2 = self.scale2()
        loss_sh = self.clip(i_sh, t_sh, scale2, y)
        loss = (loss_sp + loss_sh + loss1) / 3
   
        i_sp_out = self.c1(i_sp)
        i_sh_out = self.c2(i_sh)
        t_sh_out = self.c3(t_sh)
        t_sp_out = self.c4(t_sp)
        
        return loss, i_sp_out, i_sh_out, t_sp_out, t_sh_out
    
    def args_dict(self):
        model_args = {}
        return model_args

#w StageOne2_Img
class StageOne2_Img(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.img_encoder = ImgEncoder()
        self.ss = SS()
        self.separate = Separate()
        self.weight = nn.Parameter(torch.randn(2))
        self.c1 = nn.Linear(128, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

        self.load_pretrained()

    def load_pretrained(self):
        module = self
        des = 'load_pretrained_StageOne2_Img'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/StageOne2_1-5/best.pth.tar'
        
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print(des)

    def forward(self, i, t, y):
        i = self.img_encoder(i)
        i_sp, i_sh = self.ss(i, None)
        weight = F.softmax(self.weight, dim = -1)
        fusion = weight[0] * i_sp + weight[1] * i_sh
        out = self.c1(fusion)
        return out, weight
    
    def args_dict(self):
        model_args = {}
        return model_args

#w StageOne2_Tab
class StageOne2_Tab(nn.Module):
    def __init__(
        self,
        **kwargs):
        super().__init__()

        self.tab_encoder = TabEncoder()
        self.ss = SS()
        self.separate = Separate()
        self.weight = nn.Parameter(torch.randn(2))
        self.c1 = nn.Linear(128, 1)
 
        for m in self.modules():
            if isinstance(m, nn.Linear): 
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

        self.load_pretrained()

    def load_pretrained(self):
        module = self
        des = 'load_pretrained_StageOne2_Tab'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/StageOne2_1-5/best.pth.tar'
        
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print(des)

    def forward(self, i, t, y):
        t = self.tab_encoder(t)
        t_sh, t_sp = self.ss(None, t)
        weight = F.softmax(self.weight, dim = -1)
        fusion = weight[0] * t_sp + weight[1] * t_sh
        out = self.c1(fusion)
        return out
    
    def args_dict(self):
        model_args = {}
        return model_args



