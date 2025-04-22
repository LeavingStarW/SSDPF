import os
import shutil
import torch
import torch.nn as nn



import models



class ModelSaver(object):
    def __init__(self, logs_dir, best_metric, maximize_metric):
        super(ModelSaver, self).__init__()



        self.logs_dir = logs_dir
        self.best_metric_name = best_metric
        self.max_ckpts = 2 #w 保留最后两个训练结果以及最优结果


        
        self.best_metric_value = None
        self.maximize_metric = maximize_metric
        self.ckpt_paths = []



    #w 判断新值是不是最优的
    def _is_best(self, metric_value):
        if metric_value is None:
            return False
        return (self.best_metric_value is None
                or (self.maximize_metric and self.best_metric_value < metric_value)
                or (not self.maximize_metric and self.best_metric_value > metric_value))




    def save(self, epoch, model, optimizer, lr_scheduler, device, metric_value):
        if lr_scheduler is None:
            ckpt_dict = {
                'ckpt_info':{'epoch':epoch, self.best_metric_name:metric_value},
                'model_name':model.__class__.__name__,
                'model_state':model.to('cpu').state_dict(),
                'optimizer':optimizer.state_dict(),
            }
        else:
            ckpt_dict = {
                'ckpt_info':{'epoch':epoch, self.best_metric_name:metric_value},
                'model_name':model.module.__class__.__name__,
                'model_state':model.to('cpu').state_dict(),
                'optimizer':optimizer.state_dict(),
                'lr_scheduler':lr_scheduler.state_dict(),
            }


        model.to(device) #w 感觉可以删除
        ckpt_path = os.path.join(self.logs_dir, 'epoch_{}.pth.tar'.format(epoch))
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_value):
            self.best_metric_value = metric_value
            best_path = os.path.join(self.logs_dir, 'best.pth.tar')
            shutil.copy(ckpt_path, best_path) #w 复制文件一份并改名为best_path并保存

 

        self.ckpt_paths.append(ckpt_path)
        if len(self.ckpt_paths) > self.max_ckpts:
            oldest_ckpt = self.ckpt_paths.pop(0)
            os.remove(oldest_ckpt)



    @classmethod
    def load_model(cls, ckpt_path, gpu_ids):
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location = device)
        #w
        model_fn = models.__dict__[ckpt_dict['model_name']]

        model = model_fn()
        model = nn.DataParallel(model, gpu_ids) #w 貌似每次训练前都会用这个，那第一个ckpt_dict的设置需要改
        
        model.load_state_dict(ckpt_dict['model_state'])
        return model, ckpt_dict['ckpt_info']




    @classmethod
    def load_optimizer(cls, ckpt_path, optimizer, lr_scheduler):
        ckpt_dict = torch.load(ckpt_path)
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
