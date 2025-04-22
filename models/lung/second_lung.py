import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

#w 
from .penet_classifier import PENetClassifier




from .models_util import *






#w
class ImgEncoder(nn.Module):
    def __init__(
            self,
            **kwargs):
        super().__init__()
        #w
        self.penet = PENetClassifier()
        self.l1 = nn.Linear(2048, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 32)
        self.a1 = nn.ReLU()
        self.d1 = nn.Dropout(0.2)
        #w
        self.load_pretrained_model()

    def load_pretrained_model(self):
        module = self.penet
        module_name = 'ImgEncoder.penet'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/ckpt/penet.pth.tar'
        #w
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print('load pretrained model:' + module_name)

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
    
    def forward(self, i):
        i, _ = self.penet(i)
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
    
    def args_dict(self):
        model_args = {}
        return model_args
    




class TabEncoder(nn.Module):
    def __init__(
            self,
            **kwargs):
        super().__init__()
        


        self.l1 = nn.Linear(7, 32)
        self.l2 = nn.Linear(32, 32)
        self.a = nn.ReLU()
        self.d = nn.Dropout(0.2)
    

    def forward(self, t):
        t = self.l1(t)
        t = self.a(t)
        t = self.d(t)
        t = self.l2(t)
        return t
    
    def args_dict(self):
        model_args = {}
        return model_args
    



#w
class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        #w
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self):
        return self.scale.exp()
    


    
#w
class IntraCLoss(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.intra_scale = Scale()

    def forward(self, data, label):
        data = F.normalize(data, dim = -1)
        logit_d2d = data @ data.T * self.intra_scale()
        #w
        bs = data.shape[0]
        label_ = label.expand(bs, bs).T.clone()
        for i in range(bs):
            if label[i] == 0:
                for j in range(bs):
                    if label_[i][j] == 0:
                        label_[i][j] = 1
                    else:
                        label_[i][j] = 0
        #w
        q = F.softmax(label_ / (label_.sum(dim = -1, keepdim = True)), dim = -1)
        p = F.log_softmax(logit_d2d, dim = -1)
        loss1 = F.kl_div(p, q, reduction = 'batchmean')
        loss2 = F.kl_div(q.log(), p.exp(), reduction = 'batchmean')
        return (loss1 + loss2) / 2
    




#w
class InterCLoss(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.inter_scale = Scale()

    def forward(self, img, tab, label):
        img = F.normalize(img, dim = -1)
        tab = F.normalize(tab, dim = -1)
        logit_i2t = img @ tab.T * self.inter_scale()
        logit_t2i = tab @ img.T * self.inter_scale()
        #w
        bs = img.shape[0]
        label_ = label.expand(bs, bs).T.clone()
        for i in range(bs):
            if label[i] == 0:
                for j in range(bs):
                    if label_[i][j] == 0:
                        label_[i][j] = 1
                    else:
                        label_[i][j] = 0
        #w
        q = F.softmax(label_ / (label_.sum(dim = -1, keepdim = True)), dim = -1)
        p_i2t = F.log_softmax(logit_i2t, dim = -1)
        p_t2i = F.log_softmax(logit_t2i,dim = -1)
        loss1 = F.kl_div(p_i2t, q, reduction = 'batchmean')
        loss2 = F.kl_div(q.log(), p_i2t.exp(), reduction = 'batchmean')
        loss3 = F.kl_div(p_t2i, q, reduction = 'batchmean')
        loss4 = F.kl_div(q.log(), p_t2i.exp(), reduction = 'batchmean')
        return (loss1 + loss2 + loss3 + loss4) / 4
    




#w
class SampleLevelCLoss(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.sample_level_scale = Scale()

    def forward(self, fusion, label):
        fusion = F.normalize(fusion, dim = -1)
        logit_f2f = fusion @ fusion.T * self.sample_level_scale()
        #w
        bs = fusion.shape[0]
        label_ = label.expand(bs, bs).T.clone()
        for i in range(bs):
            if label[i] == 0:
                for j in range(bs):
                    if label_[i][j] == 0:
                        label_[i][j] = 1
                    else:
                        label_[i][j] = 0
        #w
        q = F.softmax(label_ / (label_.sum(dim = -1, keepdim = True)), dim = -1)
        p = F.log_softmax(logit_f2f, dim = -1)
        loss1 = F.kl_div(p, q, reduction = 'batchmean')
        loss2 = F.kl_div(q.log(), p.exp(), reduction = 'batchmean')
        return (loss1 + loss2) / 2
    





#w
class StageOne3(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.img_encoder = ImgEncoder()
        self.tab_encoder = TabEncoder()
        self.intra_closs = IntraCLoss()
        self.inter_closs = InterCLoss()
        self.sample_level_closs = SampleLevelCLoss()
        #w
        self.i_sp_proj = nn.Linear(32, 32)
        self.i_sh_proj = nn.Linear(32, 32)
        self.t_sp_proj = nn.Linear(32, 32)
        self.t_sh_proj = nn.Linear(32, 32)
        self.sh_closs = InterCLoss()

    def forward(self, img, tab, label):
        img = self.img_encoder(img)
        tab = self.tab_encoder(tab)
        intra_closs_img = self.intra_closs(img, label)
        intra_closs_tab = self.intra_closs(tab, label)
        intra_closs = (intra_closs_img + intra_closs_tab) / 2
        #w
        inter_closs = self.inter_closs(img, tab, label)
        #w
        fusion = (img + tab) / 2
        sample_level_closs = self.sample_level_closs(fusion, label)
        #w
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)
        loss_sp = (torch.norm(t_sp @ t_sh.T, p = 'fro') + torch.norm(i_sp @ i_sh.T, p = 'fro')) / 2
        loss_sh = self.sh_closs(i_sh, t_sh, label)
        #w return intra_closs, inter_closs, sample_level_closs, loss_sp, loss_sh
        return i_sp, i_sh, t_sh, t_sp
    
    def args_dict(self):
        model_args = {}
        return model_args




#w
class ProImg(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()



        #w 原型相关
        pro_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/data/pro_data.csv'
        pro_data = pd.read_csv(pro_path)
        pro_i_po = np.array(pro_data.iloc[0, 1:])
        pro_i_ne = np.array(pro_data.iloc[1, 1:])
        self.pro_positive = nn.Parameter(torch.tensor(pro_i_po)).float()
        self.pro_negative = nn.Parameter(torch.tensor(pro_i_ne)).float()



        #w
        self.img_encoder = ImgEncoder()
        self.i_sp_proj = nn.Linear(128, 128)
        self.i_sh_proj = nn.Linear(128, 128)
        



        #w 分类器
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        )
        


        #w 加载预训练参数
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/StageOne3_1-20_nj/best.pth.tar'
        load_pretrained_model(self, tar_path)



    def forward(self, img, label):
        #w 属性定义
        bs = img.shape[0]
        device = img.device
        loss = None


        #w 设备问题
        pro_positive = self.pro_positive.to(device)
        pro_negative = self.pro_negative.to(device)



        #w 提取特征和映射
        img = self.img_encoder(img)
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)



        #w label假设是1010，i_p_sp就是第一个和第三个特征
        label = label.squeeze()
        i_p_sp = i_sp[label == 1]
        i_n_sp = i_sp[label == 0]





        #w 分别计算i_sp整体分别的阳性和阴性分数 
        i_p_logit = F.normalize(i_sp, dim = -1) @ F.normalize(pro_positive, dim = -1).T 
        i_n_logit = F.normalize(i_sp, dim = -1) @ F.normalize(pro_negative, dim = -1).T
        #w 确保temp是二维的
        if bs == 1:
            temp = label.unsqueeze(0).unsqueeze(1)
        else:
            temp = label.unsqueeze(1)
        #w i_res_logit代表是否一致
        i_logit = i_p_logit > i_n_logit
        i_res_logit = i_logit == temp.bool().squeeze(1)
        i_res_logit[torch.where(i_p_logit == i_n_logit)[0]] = False
        i_res_logit_int = i_res_logit.int()




        label_p = i_res_logit_int[label == 1]
        #w 有可能i_res_logit全是True或者False
        if label_p.numel() != 0:
            pro_positive = F.normalize(pro_positive, dim = -1)
            p_sp = F.normalize(i_p_sp, dim = -1)
            if len(p_sp.shape) == 3:
                p_sp = p_sp.squeeze(0)
            sim_p = pro_positive @ p_sp.T
            #w 计算损失
            q = F.softmax(label_p / (label_p.sum(dim = -1, keepdim = True) + 1e-10), dim = -1)
            p = F.log_softmax(sim_p, dim = -1)
            loss1_p = F.kl_div(p, q, reduction = 'mean')
            loss2_p = F.kl_div(q.log(), p.exp(), reduction = 'mean')
            loss_p =  (loss1_p + loss2_p) / 2
            loss = loss_p



        label_n = i_res_logit_int[label == 0]
        if label_n.numel() != 0:
            pro_negative = F.normalize(pro_negative, dim = -1)
            n_sp = F.normalize(i_n_sp, dim = -1)
            if len(n_sp.shape) == 3:
                n_sp = n_sp.squeeze(0)
            sim_n = pro_negative @ n_sp.T
            #w 计算损失
            q = F.softmax(label_n / (label_n.sum(dim = -1, keepdim = True) + 1e-10), dim = -1)
            p = F.log_softmax(sim_n, dim = -1)
            loss1_n = F.kl_div(p, q, reduction = 'mean')
            loss2_n = F.kl_div(q.log(), p.exp(), reduction = 'mean')
            loss_n =  (loss1_n + loss2_n) / 2
            if loss is None:
                loss = loss_n
            else:
                loss += loss_n
                loss /= 2



        #w 更新组成成分
        logit_diff = (i_p_logit - i_n_logit).abs()
        features = torch.zeros((bs, 256), device = device)
        for idx, is_bad in enumerate(i_res_logit):
            if not is_bad:
                i_sp_good = i_sp[idx]
                i_sh_good = i_sh[idx]
                #w 调整shape
                i_sp_good = i_sp_good.reshape(-1, 128)
                i_sh_good = i_sh_good.reshape(-1, 128)
                i_good = torch.cat([i_sp_good, i_sh_good], 1)
                features[idx] = i_good
            else:
                i_sp_bad = i_sp[idx]
                i_sh_bad = i_sh[idx]
                #w 调整shape
                prob_sp = np.tanh(-logit_diff[idx].detach().cpu().numpy()) + 1
                i_sp_bad = i_sp_bad.reshape(-1, 128) * prob_sp
                i_sh_bad = i_sh_bad.reshape(-1, 128)
                i_bad = torch.cat([i_sp_bad, i_sh_bad], 1)
                features[idx] = i_bad
        



        #w 最终分类结果
        out = self.classifier(features)



        return out, loss
    


    def args_dict(self):
        model_args = {}
        return model_args







class ProTab(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()

        
        self.tab_encoder = TabEncoder()
        self.t_sp_proj = nn.Linear(128, 128)
        self.t_sh_proj = nn.Linear(128, 128)


        pro_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/data/pro_data.csv'
        pro_data = pd.read_csv(pro_path)
        pro_t_po = np.array(pro_data.iloc[2, 1:])
        pro_t_ne = np.array(pro_data.iloc[3, 1:])
        self.pro_positive = nn.Parameter(torch.tensor(pro_t_po)).float()
        self.pro_negative = nn.Parameter(torch.tensor(pro_t_ne)).float()


        
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/ckpt/StageOne3_1-30.pth.tar'
        load_pretrained_model(self, tar_path)


        self.classifier = nn.Linear(256, 1)



    def forward(self, tab, label):
        bs = tab.shape[0]
        device = tab.device
        loss = None
        pro_positive = self.pro_positive.to(device)
        pro_negative = self.pro_negative.to(device)



        tab = self.tab_encoder(tab)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)
        t_sp_detached = t_sp.detach()




        label = label.squeeze()
        t_p_sp = t_sp_detached[label == 1]
        t_n_sp = t_sp_detached[label == 0]


        

        t_p_logit = F.normalize(t_sp_detached, dim = -1) @ F.normalize(pro_positive, dim = -1).T 
        t_n_logit = F.normalize(t_sp_detached, dim = -1) @ F.normalize(pro_negative, dim = -1).T
        if bs == 1:
            temp = label.unsqueeze(0).unsqueeze(1)
        else:
            temp = label.unsqueeze(1)




        t_logit = t_p_logit > t_n_logit
        t_res_logit = t_logit == temp.bool().squeeze(1)
        t_res_logit[torch.where(t_p_logit == t_n_logit)[0]] = False
        t_res_logit_int = t_res_logit.int()
        #w 除了看是否有阳性样本，还要看阳性样本中有没有质量好的
        label_p = t_res_logit_int[label == 1]
        mask = label_p == 1
        if label_p.numel() != 0 and mask.sum() != 0:
            p_sp = t_p_sp[mask].mean(dim = 0)
            if len(p_sp.shape) == 3:
                p_sp = p_sp.squeeze(0)
            p = F.log_softmax(pro_positive, dim = -1)
            q = F.softmax(p_sp, dim = -1)
            loss_p = F.kl_div(p, q, reduction = 'mean')
            loss = loss_p




        #w 根据上述描述，loss确实可能为空
        label_n = t_res_logit_int[label == 0]
        mask = label_n == 1
        if label_n.numel() != 0 and mask.sum() != 0:
            n_sp = t_n_sp[mask].mean(dim = 0)
            if len(n_sp.shape) == 3:
                n_sp = n_sp.squeeze(0)
            p = F.log_softmax(pro_negative, dim = -1)
            q = F.softmax(n_sp, dim = -1)
            loss_n = F.kl_div(p, q, reduction = 'mean')
            if loss is None:
                loss = loss_n
            else:
                loss += loss_n




        #w 先找质量差的
        mask_fc1 = (t_res_logit == 0).detach()
        mask_fc2 = (t_res_logit == 1).detach()
        features = torch.zeros((t_sp.shape[0], t_sp.shape[1] + t_sh.shape[1]), device = device)
        if mask_fc1.any():
            res_fc1 = torch.cat([t_sp[mask_fc1], t_sh[mask_fc1]], -1)
            features[mask_fc1] = res_fc1
        if mask_fc2.any():
            res_fc2 = torch.cat([t_sp[mask_fc2], t_sh[mask_fc2]], -1)
            features[mask_fc2] = res_fc2
        out = self.classifier(features)
        


        return out, loss
    



    def args_dict(self):
        model_args = {}
        return model_args




#w
class StageOne3_1(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.img_encoder = ImgEncoder()
        self.tab_encoder = TabEncoder()
        #w
        self.i_sp_proj = nn.Linear(128, 128)
        self.i_sh_proj = nn.Linear(128, 128)
        self.t_sp_proj = nn.Linear(128, 128)
        self.t_sh_proj = nn.Linear(128, 128)
        self.sh_closs = InterCLoss()

    def forward(self, img, tab, label):
        img = self.img_encoder(img)
        tab = self.tab_encoder(tab)
        #w
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)
        loss_sp = (torch.norm(t_sp @ t_sh.T, p = 'fro') + torch.norm(i_sp @ i_sh.T, p = 'fro')) / 2
        loss_sh = self.sh_closs(i_sh, t_sh, label)
        return loss_sp, loss_sh
    
    def args_dict(self):
        model_args = {}
        return model_args

#w
class StageOne3_Img(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.img_encoder = ImgEncoder()
        #w
        self.i_sp_proj = nn.Linear(128, 128)
        self.i_sh_proj = nn.Linear(128, 128)
        #w
        self.c = nn.Linear(128, 1)
        self.weight = nn.Parameter(torch.ones([2]))

        self.load_pretrained_model()

    def load_pretrained_model(self):
        module = self
        module_name = 'StageOne3_Img'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/StageOne3/best.pth.tar'
        #w
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print('load pretrained model:' + module_name)

    def forward(self, img, tab, label):
        img = self.img_encoder(img)
        #w
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        #W
        weight = F.softmax(self.weight, dim = -1)
        i = i_sp * weight[0] + i_sh * weight[1]
        out = self.c(i)

        return out, weight
    
    def args_dict(self):
        model_args = {}
        return model_args

#w
class Test(nn.Module):
    def __init__(
            self, 
            **kwargs):
        super().__init__()
        #w
        self.img_encoder = ImgEncoder()
        self.tab_encoder = TabEncoder()
        self.intra_closs = IntraCLoss()
        self.inter_closs = InterCLoss()
        self.sample_level_closs = SampleLevelCLoss()
        #w
        self.i_sp_proj = nn.Linear(128, 128)
        self.i_sh_proj = nn.Linear(128, 128)
        self.t_sp_proj = nn.Linear(128, 128)
        self.t_sh_proj = nn.Linear(128, 128)
        self.sh_closs = InterCLoss()

        self.load_pretrained_model()

    def load_pretrained_model(self):
        module = self
        module_name = 'Test'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/StageOne3/best.pth.tar'
        #w
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print('load pretrained model:' + module_name)

    def forward(self, img, tab, label):
        img = self.img_encoder(img)
        tab = self.tab_encoder(tab)
        intra_closs_img = self.intra_closs(img, label)
        intra_closs_tab = self.intra_closs(tab, label)
        intra_closs = (intra_closs_img + intra_closs_tab) / 2
        #w
        inter_closs = self.inter_closs(img, tab, label)
        #w
        fusion = (img + tab) / 2
        sample_level_closs = self.sample_level_closs(fusion, label)
        #w
        i_sp = self.i_sp_proj(img)
        i_sh = self.i_sh_proj(img)
        t_sp = self.t_sp_proj(tab)
        t_sh = self.t_sh_proj(tab)
        loss_sp = (torch.norm(t_sp @ t_sh.T, p = 'fro') + torch.norm(i_sp @ i_sh.T, p = 'fro')) / 2
        loss_sh = self.sh_closs(i_sh, t_sh, label)
        return i_sp, i_sh, t_sh, t_sp
    
    def args_dict(self):
        model_args = {}
        return model_args

#w
class BEST_IMG(nn.Module):
    def __init__(
            self,
            **kwargs):
        super().__init__()
        #w
        self.img_encoder = ImgEncoder()
        self.fc = nn.Linear(128, 1)
        #w
        self.load_pretrained_model()

    def load_pretrained_model(self):
        module = self
        module_name = 'BEST_IMG'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/BEST/best.pth.tar'
        #w
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print('load pretrained model:' + module_name)

    def train(self, mode = True):
        module_ = self.img_encoder.penet
        if mode:
            for module in self.children():
                module.train(mode)
            module_.eval()
            for param in module_.parameters():
                param.requires_grad = False
        else:
            for module in self.children():
                module.train(mode)
    
    def forward(self, i):
        i = self.img_encoder(i)
        out = self.fc(i)
        return out
    
    def args_dict(self):
        model_args = {}
        return model_args
    
#w
class BEST_TAB(nn.Module):
    def __init__(
            self,
            **kwargs):
        super().__init__()
        #w
        self.tab_encoder = TabEncoder()
        self.fc = nn.Linear(128, 1)
        #w
        self.load_pretrained_model()

    def load_pretrained_model(self):
        module = self
        module_name = 'BEST_TAB'
        tar_path = '/mntcephfs/lab_data/wangcm/wangzhipeng/logs/05/BEST/best.pth.tar'
        #w
        trained_dict = torch.load(tar_path)['model_state']
        model_dict = module.state_dict()
        trained_dict = {k[len('module.'):]:v for k, v in trained_dict.items()}
        trained_dict = {k:v for k, v in trained_dict.items() if k in model_dict}
        model_dict.update(trained_dict)
        module.load_state_dict(model_dict)
        print('load pretrained model:' + module_name)
    
    def forward(self, t):
        t = self.tab_encoder(t)
        out = self.fc(t)
        return out
    
    def args_dict(self):
        model_args = {}
        return model_args