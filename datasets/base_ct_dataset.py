import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset

class BaseCTDataset(Dataset):
    def __init__(self, data_dir, is_training_set):
        self.data_dir = data_dir
        self.is_training_set = is_training_set
        self.pixel_dict = None
    #w
    def __len__(self):
        raise NotImplementedError
    def __getitem__(self, item):
        raise NotImplementedError


    #w 归一化
    def _normalize_raw(self, pixels):
        pixels = pixels.astype(np.float32)
        pixels = (pixels - self.pixel_dict['min_val']) / (self.pixel_dict['max_val'] - self.pixel_dict['min_val'])
        pixels = np.clip(pixels, 0., 1.) - self.pixel_dict['avg_val']
        return pixels