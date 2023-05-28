#! /Work18/.../conda3_env/env_1 python
# @Time : 2020/8/15 7:37
# @Author : gy, syd
# @File : IEMOCAP.py
import random

import torch
import numpy as np
import h5py
import scipy.io as sio
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

class Transform(object):

    def __call__(self, sample):
        x1 = np.transpose(sample['image'])  # <class 'tuple'>: (230, 32, 128)
        y1 = np.transpose(sample['label'])
        g1 = np.transpose(sample['gender'])
        l1 = np.transpose(sample['length'])
        x_train = x1
        y_train = y1
        g_train = g1
        len_train = l1

        img_rows, img_cols = 700, 129

        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)  # <class 'tuple'>: (230, 1, 32, 128)
        # input_shape = (1, img_rows, img_cols)

        x_train = x_train.astype('float32')
        x_train = Variable(torch.from_numpy(x_train))  # torch.Size([230, 1, 32, 128])

        y_train = torch.squeeze(torch.from_numpy(y_train))  # shape torch.Size([230])
        y_train = y_train.long()
        g_train = torch.squeeze(torch.from_numpy(g_train))  # shape torch.Size([230])
        g_train = g_train.long()
        print(f'input shape:{x_train.shape}')

        return {'image': x_train, 'label': y_train, 'gender': g_train, 'length': len_train}


class IEMOCAP(Dataset):

    def __init__(self, root_dir, experiments = '1', nraws=0, is_all_sample=False,
                 train_or_test='train', transform=None, shuffle=False):

        super(IEMOCAP, self).__init__()
        assert isinstance(nraws, (int, list))
        self.transform = transform
        self.nraws = nraws
        self.shuffle = shuffle
        self.is_all_sample = is_all_sample
        if experiments == '1':
            if (train_or_test == "train"):
                self.f = h5py.File(root_dir + 'train2345.mat', 'r')
                self.image = self.f["train2345_data"]
                self.label = self.f["train2345_label"]
                self.gender = self.f["train2345_gender"]
                self.length = self.f["train2345_len"]
            else:
                self.f = h5py.File(root_dir + 'test1.mat', 'r')
                self.image = self.f["test1_data"]
                self.label = self.f["test1_label"]
                self.gender = self.f["test1_gender"]
                self.length = self.f["test1_len"]
        elif experiments == '2':
            if (train_or_test == "train"):
                self.f = h5py.File(root_dir + 'train1345.mat', 'r')
                self.image = self.f["train1345_data"]
                self.label = self.f["train1345_label"]
                self.gender = self.f["train1345_gender"]
                self.length = self.f["train1345_len"]
            else:
                self.f = h5py.File(root_dir + 'test2.mat', 'r')
                self.image = self.f["test2_data"]
                self.label = self.f["test2_label"]
                self.gender = self.f["test2_gender"]
                self.length = self.f["test2_len"]
        elif experiments == '3':
            if (train_or_test == "train"):
                self.f = h5py.File(root_dir + 'train1245.mat', 'r')
                self.image = self.f["train1245_data"]
                self.label = self.f["train1245_label"]
                self.gender = self.f["train1245_gender"]
                self.length = self.f["train1245_len"]
            else:
                self.f = h5py.File(root_dir + 'test3.mat', 'r')
                self.image = self.f["test3_data"]
                self.label = self.f["test3_label"]
                self.gender = self.f["test3_gender"]
                self.length = self.f["test3_len"]
        elif experiments == '4':
            if (train_or_test == "train"):
                self.f = h5py.File(root_dir + 'train1235.mat', 'r')
                self.image = self.f["train1235_data"]
                self.label = self.f["train1235_label"]
                self.gender = self.f["train1235_gender"]
                self.length = self.f["train1235_len"]
            else:
                self.f = h5py.File(root_dir + 'test4.mat', 'r')
                self.image = self.f["test4_data"]
                self.label = self.f["test4_label"]
                self.gender = self.f["test4_gender"]
                self.length = self.f["test4_len"]
        elif experiments == '5':
            if (train_or_test == "train"):
                self.f = h5py.File(root_dir + 'train1234.mat', 'r')
                self.image = self.f["train1234_data"]
                self.label = self.f["train1234_label"]
                self.gender = self.f["train1234_gender"]
                self.length = self.f["train1234_len"]
            else:
                self.f = h5py.File(root_dir + 'test5.mat', 'r')
                self.image = self.f["test5_data"]
                self.label = self.f["test5_label"]
                self.gender = self.f["test5_gender"]
                self.length = self.f["test5_len"]
        else:
            print('wtf?')

        self.initial()

    def initial(self):
        print(self.image.shape)
        self.sample_len = self.image.shape[2]
        if(self.nraws > self.sample_len):
            self.nraws = self.sample_len
        self.start_raw = 0
        self.end_raw = 0
        self.current_sample = dict()
        self.current_sample_len = self.sample_len if self.is_all_sample else self.nraws

    def next(self):
        self.current_sample.clear()
        if (self.is_all_sample):
            self.current_sample = {'image': self.image, 'label': self.label, 'gender': self.gender, 'length': self.length}
            self.current_sample_len = self.sample_len
        else:
            self.end_raw = self.start_raw + self.nraws
            self.end_raw = self.sample_len if self.end_raw >= self.sample_len else self.end_raw
            self.current_sample = {'image': self.image[:, :, self.start_raw: self.end_raw],
                                   'label': self.label[:, self.start_raw: self.end_raw],
                                   'gender': self.gender[:, self.start_raw: self.end_raw],
                                   'length': self.length[:, self.start_raw: self.end_raw]}
            self.current_sample_len = self.end_raw - self.start_raw
            self.start_raw = self.end_raw

        if self.transform:
            self.current_sample = self.transform(self.current_sample)

    def __len__(self):
        return self.current_sample_len

    def __getitem__(self, idx):
        x_train, y_train , g_train, len_train = self.current_sample['image'][idx], self.current_sample['label'][idx], \
                                                     self.current_sample['gender'][idx], self.current_sample['length'][idx]

        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        g_train = torch.tensor(g_train)
        len_train = torch.tensor(len_train)

        return x_train, y_train, g_train, len_train


if __name__ == '__main__':
    pass
