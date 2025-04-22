import cv2
import h5py
import numpy as np
import os
import pickle
import random
import torch
import util

from .base_ct_dataset import BaseCTDataset
from ct.ct_pe_constants import *
from scipy.ndimage.interpolation import rotate


class DatasetPE(BaseCTDataset):
    def __init__(self, args, phase, is_training_set=True):

        super(DatasetPE, self).__init__(args.data_dir, is_training_set=is_training_set)
        self.phase = phase
        self.pe_types = args.pe_types
        #w 两个值基本不变
        self.crop_shape = args.crop_shape
        self.resize_shape = args.resize_shape

        #w
        self.do_hflip = self.is_training_set and args.do_hflip
        self.do_vflip = self.is_training_set and args.do_vflip
        self.do_rotate = self.is_training_set and args.do_rotate
        self.do_jitter = self.is_training_set and args.do_jitter


        #w 这个用在归一化中
        self.pixel_dict = {
            'min_val':CONTRAST_HU_MIN,
            'max_val':CONTRAST_HU_MAX,
            'avg_val':CONTRAST_HU_MEAN,
        }
        self.num_slices = args.num_slices
        


        #w 取样文件，里边是多个被初始化的CTPE类的实例
        with open(args.pkl_path, 'rb') as pkl_file:
            all_ctpes = pickle.load(pkl_file)
        #w 根据phase筛选样本
        self.ctpe_list = [ctpe for ctpe in all_ctpes if self._include_ctpe(ctpe)] 



        #w
        self.window_to_series_idx = []   
        self.series_to_window_idx = []
        window_start = 0
        for i, s in enumerate(self.ctpe_list): 
            #w len(s)是总切片数，每num_slices作为一个窗口组，不足num_slices的也作为一个窗口组
            num_windows = len(s) // self.num_slices + (1 if len(s) % self.num_slices > 0 else 0) 
            self.window_to_series_idx += num_windows * [i]   
            self.series_to_window_idx.append(window_start)
            window_start += num_windows 


        self.pe_types = args.pe_types
        self.do_center_abnormality = self.is_training_set
        self.min_pe_slices = args.min_abnormal_slices
        self.positive_idxs = [i for i in range(len(self.ctpe_list)) if self.ctpe_list[i].is_positive]
        self.abnormal_prob = args.abnormal_prob if self.is_training_set else None






    def _include_ctpe(self, pe):
        
        if pe.phase != self.phase:
            return False
        #w 考虑pe_types 
        if pe.is_positive and pe.type not in self.pe_types:
            return False

        return True
    


    def __len__(self):
        return len(self.window_to_series_idx)
    


    def __getitem__(self, idx):
        ctpe_idx = self.window_to_series_idx[idx]
        ctpe = self.ctpe_list[ctpe_idx]


        #w 这个值是0.3，30%概率将阴性调整为阳性，阳性调整为阳性
        if self.abnormal_prob is not None and random.random() < self.abnormal_prob:
            if not ctpe.is_positive:
                ctpe_idx = random.choice(self.positive_idxs)
                ctpe = self.ctpe_list[ctpe_idx]
            #w 紧接着训练集和验证集做居中取样
            start_idx = self._get_abnormal_start_idx(ctpe, do_center=self.do_center_abnormality)
        #w 70%概率拿到阴性就是阴性，阳性就是阳性
        else:
            start_idx = (idx - self.series_to_window_idx[ctpe_idx]) * self.num_slices













        if self.do_jitter:
            start_idx += random.randint(-self.num_slices // 2, self.num_slices // 2)

        #w 待定
        start_idx = min(max(start_idx, 0), len(ctpe) - self.num_slices)










        volume = self._load_volume(ctpe, start_idx)
        volume = self._transform(volume)

        is_abnormal = torch.tensor([self._is_abnormal(ctpe, start_idx)], dtype=torch.float32)
        target = {'is_abnormal': is_abnormal,
                  'study_num': ctpe.study_num,
                  'slice_idx': start_idx,
                  'series_idx': ctpe_idx}

        return volume, target

    def get_series_label(self, series_idx):
        return float(self.ctpe_list[series_idx].is_positive)

    def get_series(self, study_num):
        for ctpe in self.ctpe_list:
            if ctpe.study_num == study_num:
                return ctpe
        return None

    


    def _get_abnormal_start_idx(self, ctpe, do_center=True):
        abnormal_bounds = (min(ctpe.pe_idxs), max(ctpe.pe_idxs))
        if do_center:
            center_idx = sum(abnormal_bounds) // 2
            start_idx = max(0, center_idx - self.num_slices // 2)
        else:
            start_idx = random.randint(abnormal_bounds[0] - self.num_slices + self.min_pe_slices,
                                       abnormal_bounds[1] - self.min_pe_slices + 1)

        return start_idx
    



    def _load_volume(self, ctpe, start_idx):
 


        with h5py.File(os.path.join(self.data_dir), 'r') as hdf5_fh:
            volume = hdf5_fh[str(ctpe.study_num)][start_idx:start_idx + self.num_slices]

        return volume

    def _is_abnormal(self, ctpe, start_idx):
        
        if ctpe.is_positive:
            abnormal_slices = [i for i in ctpe.pe_idxs if start_idx <= i < start_idx + self.num_slices]
            is_abnormal = len(abnormal_slices) >= self.min_pe_slices
        else:
            is_abnormal = False

        return is_abnormal

    def _crop(self, volume, x1, y1, x2, y2):

        volume = volume[:, y1: y2, x1: x2]

        return volume

    def _rescale(self, volume, interpolation=cv2.INTER_AREA):
        return util.resize_slice_wise(volume, tuple(self.resize_shape), interpolation)

    def _pad(self, volume):
        def add_padding(volume_, pad_value=AIR_HU_VAL):
            num_pad = self.num_slices - volume_.shape[0]
            volume_ = np.pad(volume_, ((0, num_pad), (0, 0), (0, 0)), mode='constant', constant_values=pad_value)

            return volume_

        volume_num_slices = volume.shape[0]

        if volume_num_slices < self.num_slices:
            volume = add_padding(volume, pad_value=AIR_HU_VAL)
        elif volume_num_slices > self.num_slices:
            # Choose center slices
            start_slice = (volume_num_slices - self.num_slices) // 2
            volume = volume[start_slice:start_slice + self.num_slices, :, :]

        return volume

    def _transform(self, inputs):
        # Pad or crop to expected number of slices
        inputs = self._pad(inputs)

        if self.resize_shape is not None:
            inputs = self._rescale(inputs, interpolation=cv2.INTER_AREA)

        if self.crop_shape is not None:
            row_margin = max(0, inputs.shape[-2] - self.crop_shape[-2])
            col_margin = max(0, inputs.shape[-1] - self.crop_shape[-1])
            # Random crop during training, center crop during test inference
            row = random.randint(0, row_margin) if self.is_training_set else row_margin // 2
            col = random.randint(0, col_margin) if self.is_training_set else col_margin // 2
            inputs = self._crop(inputs, col, row, col + self.crop_shape[-1], row + self.crop_shape[-2])

        if self.do_vflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-2)

        if self.do_hflip and random.random() < 0.5:
            inputs = np.flip(inputs, axis=-1)

        if self.do_rotate:
            angle = random.randint(-15, 15)
            inputs = rotate(inputs, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL)

        # Normalize raw Hounsfield Units
        inputs = self._normalize_raw(inputs)

        inputs = np.expand_dims(inputs, axis=0)  # Add channel dimension
        inputs = torch.from_numpy(inputs)

        return inputs

