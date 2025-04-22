import torch
from torch.utils.data import DataLoader
#w
import datasets

class CTDataLoader(DataLoader):
    def __init__(self, args, phase, is_training):
        dataset_fn = datasets.__dict__[args.dataset]
        #w 找到我们定义的dataset类
        dataset = dataset_fn(args, phase, is_training)
        #w
        self.bs = args.bs
        self.phase = phase
        super(CTDataLoader, self).__init__(dataset,
                                            batch_size = args.bs,
                                            shuffle = is_training,
                                            num_workers = args.num_workers,
                                            pin_memory = True)
    def get_series_label(self, series_idx):
        return self.dataset.get_series_label(series_idx)
    #w
    def get_series(self, study_num):
        return self.dataset.get_series(study_num)
