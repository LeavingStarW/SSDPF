import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import util

from datetime import datetime


plt.switch_backend('agg')





class BaseLogger(object):
    def __init__(self, args, dataset_len):

        


        self.bs = args.bs
        self.dataset_len = dataset_len
        self.device = args.device

        self.logs_dir = args.logs_dir if args.is_training else args.results_preds_dir

        


        
        # Current iteration in epoch (i.e., # examples seen in the current epoch)
        self.iter = 0
        # Current iteration overall (i.e., total # of examples seen)
        
        self.iter_start_time = None
        self.epoch_start_time = None


        #w
        self.epoch = args.start_epoch
        def round_down(x, m):
            return int(m * round(float(x) / m))
        self.global_step = round_down((self.epoch - 1) * dataset_len, args.bs)

        #w .log的位置
        self.log_path = os.path.join(self.logs_dir, '{}.log'.format(args.name))






    #w 
    def write(self, message):
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')

        print(message)


    #w epoch末执行的记录损失操作
    def _log_loss(self, loss):
        for k, v in loss.items():
            self.write('[{}:{:.4g}]'.format(k, v))


    #w epoch末执行的记录指标操作
    def _log_metric(self, metric):
        for k, v in metric.items():
            self.write('[{}:{:.4g}]'.format(k, v))



    def start_iter(self):
        raise NotImplementedError

    def end_iter(self):
        raise NotImplementedError

    def start_epoch(self):
        raise NotImplementedError

    def end_epoch(self, metrics):
        raise NotImplementedError
