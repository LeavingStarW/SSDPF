
import models
import torch
import torch.nn as nn
import util
from torch.autograd import Variable

from args import TrainArgParser
from evaluator import ModelEvaluator
from logger import TrainLogger
from saver import ModelSaver

###
import pandas as pd
import numpy as np
import sys
import os

#w
import data_loader
class CTPE:
    def __init__(self, name, is_positive, parser, num_slice, first_appear, avg_bbox, last_appear):
        self.study_num = name
        self.is_positive = is_positive
        self.phase = parser
        self.num_slice = num_slice
        self.first_appear = first_appear
        self.bbox = avg_bbox
        self.last_appear = last_appear

    def __len__(self):
        return self.num_slice



def train(args):
    
    



    if args.ckpt_path:
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
        args.start_epoch = ckpt_info['epoch'] + 1
    else:
        model_fn = models.__dict__[args.model]
        model = model_fn(**vars(args))
        model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)
    model.train()



    parameters = model.parameters() #w 用于初始化优化器
    optimizer = util.get_optimizer(parameters, args) #w 根据参数初始化优化器
    lr_scheduler = util.get_scheduler(optimizer, args)
    if args.ckpt_path:
        ModelSaver.load_optimizer(args.ckpt_path, optimizer, lr_scheduler)
    
    

    #w
    data_loader_fn = data_loader.__dict__[args.data_loader]
    train_loader = data_loader_fn(args, phase = 'train', is_training = True)
    logger = TrainLogger(args, len(train_loader.dataset))
    #w
    val_loader = [data_loader_fn(args, phase = 'val', is_training = False)]


    evaluator = ModelEvaluator(val_loader, args.agg_method, data_tab_path=args.data_tab_path)



    saver = ModelSaver(args.logs_dir, args.best_metric, args.maximize_metric)



    #w 损失函数设置
    loss_bf = util.get_loss_fn('bf')



    #w 辅助变量
    only_val = False
    count = 0
    check_frozen_count = 5 #w count为0和5000的时候打印一次参数
    for_tsne_train = False

    
    
    while not logger.is_finished_training():
        logger.start_epoch()




        #w 每个epoch的损失记录
        loss_record = {}
        train_total_samples = 0




        if not only_val:
            for img, target_dict in train_loader:
                logger.start_iter()


                #w
                if args.dataset == 'DatasetLung':
                    ids = [item for item in target_dict['study_num']]
                    tab=[]
                    table = pd.read_csv(args.data_tab_path)
                    for i in range(len(target_dict['study_num'])):
                        data = table[table['NewPatientID'] == ids[i]].iloc[0, 1:8].astype(np.float32)
                        tab.append(torch.tensor(data, dtype=torch.float32))
                    tab = torch.stack(tab).squeeze(1)



                #w PE
                '''ids = [int(item) for item in target_dict['study_num']]
                tab = []
                table_data = pd.read_csv(args.data_tab_path)
                for idx in ids:
                    data = table_data[table_data['idx'] == idx].iloc[:, 4:].astype(np.float32)
                    tab.append(torch.tensor(np.array(data), dtype = torch.float32))
                tab = torch.stack(tab).squeeze(1)'''
                
                
    
                with torch.set_grad_enabled(True):




                    img = img.to(args.device)
                    tab = tab.to(args.device)
                    label = target_dict['is_abnormal'].to(args.device)
                    if args.dataset == 'DatasetLung':
                        label = label.float().unsqueeze(1)
                    else:
                        label = label.float()


                    #w Lung，你需要确保损失值都是用均值聚合后的
                    output = model(img, tab)
                    loss = output['all_loss']




                    #w update loss_record
                    for k, v in output.items():
                        if k.endswith('_loss') and v != -1:
                            len_sample = img.shape[0]  
                            train_total_samples += len_sample
                            if 'train_' + k not in loss_record:
                                loss_record['train_' + k] = v.item() * len_sample
                            else:
                                loss_record['train_' + k] += v.item() * len_sample
                        elif k.endswith('_loss'):
                            len_sample = img.shape[0]  
                            train_total_samples += len_sample
                            if 'train_' + k not in loss_record:
                                loss_record['train_' + k] = loss_out.item() * len_sample
                            else:
                                loss_record['train_' + k] += loss_out.item() * len_sample







                    #w
                    if count % check_frozen_count == 0:
                        for k, v in model.named_parameters():
                            print(k, v)
                        check_frozen_count *= 1000
                    count += 1




                    #w 三步走
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()



                    ###Log
                    logger.log_iter(img,None,None,loss,optimizer)

                logger.end_iter()
                util.step_scheduler(lr_scheduler,global_step=logger.global_step)
                # util.step_scheduler(D_lr_scheduler,global_step=logger.global_step)



        #w 换成单个样本平均损失
        for k, v in loss_record.items():
            if k.endswith('_loss'):
                loss_record[k] /= train_total_samples


        #w 验证入口
        metric, loss_record = evaluator.evaluate(model, args.device, loss_record)





        saver.save(logger.epoch,model,optimizer,lr_scheduler,args.device,
                    metric_value = loss_record['val_all_loss']) #W metric.get(args.best_metric,None)
        print(metric)




        #w epoch末
        logger.end_epoch(metric, loss_record)



       
        util.step_scheduler(lr_scheduler, metric, epoch=logger.epoch, best_metric=args.best_metric)
        #util.step_scheduler(D_lr_scheduler, metric, epoch=logger.epoch, best_metric=args.best_metric)







#w 程序入口，标准写法
if __name__ == '__main__':


    util.set_spawn_enabled() #w 设置多进程，在io_util里
    parser = TrainArgParser() #w 训练参数类
    args_ = parser.parse_args() #w 参数解析，得到参数

    train(args_) #w 输入参数，执行训练
