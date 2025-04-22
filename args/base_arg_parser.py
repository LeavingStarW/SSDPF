import os
import datetime
import json
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import argparse



import util



class BaseArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description = '02')
        self.is_training = None



        #w Lung
        self.parser.add_argument('--data_dir', type = str, default = '/data2/wangchangmiao/wzp/data/Lung/data1.hdf5') 
        self.parser.add_argument('--logs_dir', type = str, default = '/data2/wangchangmiao/wzp/logs/Lung') 
        self.parser.add_argument('--results_preds_dir', type = str, default = '/data2/wangchangmiao/wzp/results/preds/Lung') 
        self.parser.add_argument('--pkl_path', type = str, default = '/data2/wangchangmiao/wzp/data/Lung/series_list_last_AG4.pkl')
        self.parser.add_argument('--data_tab_path', type = str, default = '/data2/wangchangmiao/wzp/data/Lung/G4_first_last_nor.csv')
        self.parser.add_argument('--init_method', type = str, default = 'kaiming')

        self.parser.add_argument('--model', type = str, default = 'None') 
        self.parser.add_argument('--name', type = str, required = True) 
        self.parser.add_argument('--ckpt_path', type = str, required = False) 
        self.parser.add_argument('--bs', type = int, required = True)
        self.parser.add_argument('--dataset', type = str, required = True, choices = ('pe', 'lung'))
        self.parser.add_argument('--num_slices', type = int, required = True) 

        self.parser.add_argument('--agg_method', type = str, default = 'max') 
        self.parser.add_argument('--resize_shape', type = util.args_to_list, default = '208, 208') 
        self.parser.add_argument('--crop_shape', type = util.args_to_list, default = '192, 192')
        
        self.parser.add_argument('--gpu_ids', type = str, default = '0') 
        self.parser.add_argument('--num_workers', type = int, default = 4) 
        self.parser.add_argument('--deterministic', type = util.str_to_bool, default = True) 
        self.parser.add_argument('--cudnn_benchmark', type = util.str_to_bool, default = False) 
        
        

        





        
    def parse_args(self):
        args = self.parser.parse_args()


        #w 参数完善
        args.is_training = self.is_training
        args.start_epoch = 1




        #w 固定种子
        if args.deterministic:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            cudnn.deterministic = True
            print('固定种子')
        



        #w 完善训练和测试结果保存路径
        date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.is_training: #w 如果是测试
            args.results_preds_dir = os.path.join(args.results_preds_dir, '{}_{}'.format(args.name, date_string))
            os.makedirs(args.results_preds_dir, exist_ok = True)
        else: #w 如果是训练
            logs_dir = os.path.join(args.logs_dir, '{}_{}'.format(args.name, date_string))
            os.makedirs(logs_dir, exist_ok = True)
            args.logs_dir = logs_dir
            with open(os.path.join(args.logs_dir, 'args.json'), 'w') as jf:
                json.dump(vars(args), jf, indent = 4, sort_keys = False)
                jf.write('\n')
            #w 确定选择最佳训练结果的依据，False代表性能和指标结果成反比
            if args.best_metric in ['loss']:
                args.maximize_metric = False
            else:
                args.maximize_metric = True




        #w 先GPU再CPU
        args.gpu_ids = util.args_to_list(args.gpu_ids)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
            #w 关闭卷积优化操作
            cudnn.benchmark = args.cudnn_benchmark
        else:
            args.device = 'cpu'

        


        #w 数据集基础设置
        if args.dataset == 'pe':
            args.dataset = 'DatasetPE'
            args.data_loader = 'CTDataLoader'
            args.loader = 'window' #w 窗口加载
        elif args.dataset == 'lung':
            args.dataset = 'DatasetLung'
            args.data_loader = 'CTDataLoader'
            args.loader = 'window' #w 窗口加载
        
        

        return args