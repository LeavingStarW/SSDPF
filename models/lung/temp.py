import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import sys

from models.models_util import *

from .penet import PENet


#w TabMLP
class TabMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.l1 = nn.Linear(7, 32)
        self.l2 = nn.Linear(32, 32)
        self.a = nn.ReLU()
        self.d = nn.Dropout(0.2)

        self.classifier = nn.Linear(32, 1)

    def forward(self, tab):

        tab = self.l1(tab)
        tab = self.a(tab)
        tab = self.d(tab)
        tab = self.l2(tab)
        out = self.classifier(tab)

        return {
            'fea':tab,
            'out':out,
            'all_loss':-1
        }
    

#w ImgMLP
class ImgMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.penet = PENet()
        self.l1 = nn.Linear(256, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, 32)
        self.a = nn.ReLU()
        self.d = nn.Dropout(0.2)

        self.classifier = nn.Linear(32, 1)


        #w
        module = self.penet
        ckpt_path = '/data2/wangchangmiao/wzp/logs/Lung/PENet/epoch_20.pth.tar'
        load_pretrained_model(module, ckpt_path)
        self.is_load = True

    def train(self, mode = True):
        module_ = self.penet
        if mode:
            for module in self.children():
                module.train(mode)
            module_.eval()
            for param in module_.parameters():
                param.requires_grad = False
        else:
            for module in self.children():
                module.train(mode)
        
    def forward(self, img):

        penet_out = self.penet(img)
        img = penet_out['fea']
        img = nn.AdaptiveAvgPool3d(1)(img)
        img = img.view(img.shape[0], -1)

        img = self.l1(img)
        img = self.a(img)
        img = self.d(img)
        img = self.l2(img)
        img = self.a(img)
        img = self.d(img)
        img = self.l3(img)
        out = self.classifier(img)

        return {
            'fea':img,
            'out':out
        }
    

#w 对比学习需要的Scale
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self):
        return self.scale.exp()
    

#w IntraCL
class IntraCL(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = Scale()

    def forward(self, data, label):
        data = F.normalize(data, dim = -1)
        logit_d2d = data @ data.T * self.scale()
        
        bs = data.shape[0]
        label_ = label.expand(bs, bs).T.clone()
        for i in range(bs):
            if label[i] == 0:
                for j in range(bs):
                    if label_[i][j] == 0:
                        label_[i][j] = 1
                    else:
                        label_[i][j] = 0
        
        p = F.log_softmax(logit_d2d, dim = -1)
        q = F.softmax(label_ / (label_.sum(dim = -1, keepdim = True)), dim = -1)
        loss1 = F.kl_div(p, q, reduction = 'batchmean')
        loss2 = F.kl_div(q.log(), p.exp(), reduction = 'batchmean')
        return (loss1 + loss2) / 2
    

#w InterCL
class InterCL(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.scale = Scale()

    def forward(self, img, tab, label):
        img = F.normalize(img, dim = -1)
        tab = F.normalize(tab, dim = -1)
        logit_i2t = img @ tab.T * self.scale()
        logit_t2i = tab @ img.T * self.scale()
        
        bs = img.shape[0]
        label_ = label.expand(bs, bs).T.clone()
        for i in range(bs):
            if label[i] == 0:
                for j in range(bs):
                    if label_[i][j] == 0:
                        label_[i][j] = 1
                    else:
                        label_[i][j] = 0
        
        p_i2t = F.log_softmax(logit_i2t, dim = -1)
        p_t2i = F.log_softmax(logit_t2i, dim = -1)
        q = F.softmax(label_ / (label_.sum(dim = -1, keepdim = True)), dim = -1)
        
        loss1 = F.kl_div(p_i2t, q, reduction = 'batchmean')
        loss2 = F.kl_div(q.log(), p_i2t.exp(), reduction = 'batchmean')
        loss3 = F.kl_div(p_t2i, q, reduction = 'batchmean')
        loss4 = F.kl_div(q.log(), p_t2i.exp(), reduction = 'batchmean')
        return (loss1 + loss2 + loss3 + loss4) / 4
    

#w JSD
class JSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction = 'none', log_target = True)

    def forward(self, p, q):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        jsd = 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
        return jsd.mean()


#w SP1JSD
class SP1JSD(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.img_encoder = ImgMLP()
        self.tab_encoder = TabMLP()

        self.i_sp_proj = nn.Linear(32, 32)
        self.i_sh_proj = nn.Linear(32, 32)
        self.t_sp_proj = nn.Linear(32, 32)
        self.t_sh_proj = nn.Linear(32, 32)

        self.sp_loss = nn.CosineSimilarity(dim = 1)
        self.sh_loss = JSD()

    def forward(self, img, tab):
        img_encoder_out = self.img_encoder(img)
        img = img_encoder_out['fea']
        tab_encoder_out = self.tab_encoder(tab)
        tab = tab_encoder_out['fea']

        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)

        sp_loss = (self.sp_loss(i_sp, i_sh).abs().mean() + self.sp_loss(t_sp, t_sh).abs().mean()) / 2
        sh_loss = self.sh_loss(i_sh.sigmoid(), t_sh.sigmoid())

        loss = sp_loss + sh_loss

        return {
            'sp_loss':sp_loss,
            'sh_loss':sh_loss,
            'i_sp':i_sp,
            'i_sh':i_sh,
            't_sp':t_sp,
            't_sh':t_sh,
            'all_loss':loss
        }
    

#w 只有个性共性学习，个性1，InterCL
class SSPre2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.img_encoder = ImgMLP()
        self.tab_encoder = TabMLP()

        self.i_sp_proj = nn.Linear(32, 32)
        self.i_sh_proj = nn.Linear(32, 32)
        self.t_sp_proj = nn.Linear(32, 32)
        self.t_sh_proj = nn.Linear(32, 32)

        self.sp_loss = nn.CosineSimilarity(dim = 1)
        self.sh_loss = InterCL()

    def forward(self, img, tab, label):
        img_encoder_out = self.img_encoder(img)
        img = img_encoder_out['fea']
        tab_encoder_out = self.tab_encoder(tab)
        tab = tab_encoder_out['fea']

        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)

        sp_loss = (self.sp_loss(i_sp, i_sh).abs().mean() + self.sp_loss(t_sp, t_sh).abs().mean()) / 2
        sh_loss = self.sh_loss(i_sh, t_sh, label)

        loss = sp_loss + sh_loss

        return {
            'sp_loss':sp_loss,
            'sh_loss':sh_loss,
            'i_sp':i_sp,
            'i_sh':i_sh,
            't_sp':t_sp,
            't_sh':t_sh,
            'all_loss':loss
        }
    

#w MGCL，JSD
class SSPre3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.img_encoder = ImgMLP()
        self.tab_encoder = TabMLP()

        self.i_sp_proj = nn.Linear(32, 32)
        self.i_sh_proj = nn.Linear(32, 32)
        self.t_sp_proj = nn.Linear(32, 32)
        self.t_sh_proj = nn.Linear(32, 32)

        self.sp_loss = nn.CosineSimilarity(dim = 1)
        self.sh_loss = JSD()

        self.intra_cl = IntraCL()
        self.inter_cl = InterCL()
        self.sl_intra_cl = IntraCL()

    def forward(self, img, tab, label):
        img_encoder_out = self.img_encoder(img)
        img = img_encoder_out['fea']
        tab_encoder_out = self.tab_encoder(tab)
        tab = tab_encoder_out['fea']


        #w
        img_intra_cl = self.intra_cl(img, label)
        tab_intra_cl = self.intra_cl(tab, label)
        intra_cl_loss = (img_intra_cl + tab_intra_cl) / 2

        inter_cl_loss = self.inter_cl(img, tab, label)

        fusion = (img + tab) / 2
        sl_intra_cl_loss = self.sl_intra_cl(fusion, label)

        mgcl_loss = intra_cl_loss + inter_cl_loss + sl_intra_cl_loss


        #w
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)

        sp_loss = (self.sp_loss(i_sp, i_sh).abs().mean() + self.sp_loss(t_sp, t_sh).abs().mean()) / 2
        sh_loss = self.sh_loss(i_sh, t_sh)

        ss_loss = sp_loss + sh_loss

        loss = mgcl_loss + ss_loss

        return {
            'intra_loss':intra_cl_loss,
            'inter_loss':inter_cl_loss,
            'sl_loss':sl_intra_cl_loss,
            'sp_loss':sp_loss,
            'sh_loss':sh_loss,
            'i_sp':i_sp,
            'i_sh':i_sh,
            't_sp':t_sp,
            't_sh':t_sh,
            'all_loss':loss
        }
    

#w MGCL，InterCL
class SSPre4(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.img_encoder = ImgMLP()
        self.tab_encoder = TabMLP()

        self.i_sp_proj = nn.Linear(32, 32)
        self.i_sh_proj = nn.Linear(32, 32)
        self.t_sp_proj = nn.Linear(32, 32)
        self.t_sh_proj = nn.Linear(32, 32)

        self.sp_loss = nn.CosineSimilarity(dim = 1)
        self.sh_loss = InterCL()

        self.intra_cl = IntraCL()
        self.inter_cl = InterCL()
        self.sl_intra_cl = IntraCL()

    def forward(self, img, tab, label):
        img_encoder_out = self.img_encoder(img)
        img = img_encoder_out['fea']
        tab_encoder_out = self.tab_encoder(tab)
        tab = tab_encoder_out['fea']


        #w
        img_intra_cl = self.intra_cl(img, label)
        tab_intra_cl = self.intra_cl(tab, label)
        intra_cl_loss = (img_intra_cl + tab_intra_cl) / 2

        inter_cl_loss = self.inter_cl(img, tab, label)

        fusion = (img + tab) / 2
        sl_intra_cl_loss = self.sl_intra_cl(fusion, label)

        mgcl_loss = intra_cl_loss + inter_cl_loss + sl_intra_cl_loss


        #w
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)

        sp_loss = (self.sp_loss(i_sp, i_sh).abs().mean() + self.sp_loss(t_sp, t_sh).abs().mean()) / 2
        sh_loss = self.sh_loss(i_sh, t_sh, label)

        ss_loss = sp_loss + sh_loss

        loss = mgcl_loss + ss_loss

        return {
            'intra_loss':intra_cl_loss,
            'inter_loss':inter_cl_loss,
            'sl_loss':sl_intra_cl_loss,
            'sp_loss':sp_loss,
            'sh_loss':sh_loss,
            'i_sp':i_sp,
            'i_sh':i_sh,
            't_sp':t_sp,
            't_sh':t_sh,
            'all_loss':loss
        }