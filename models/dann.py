#! /Work18/.../conda3_env/env_1 python
# @Time : 2021/10/15 11:09
# @Author : gy
# @File : CNN_RNN.py

import torch
import numpy as np
import math
from torch import optim, nn
from torch.autograd import Function
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class DANN(nn.Module):
    def __init__(self, hidden_size=128, n_layers=1, dp=0.25,
                 bidirectional=True, rnn_cell='lstm', variable_lengths=False):
        super(DANN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dp
        self.variable_lengths = variable_lengths
        self.rnn_input_dims = 2128

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(), #nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),  )#nn.MaxPool2d(2),
        self.flatten = Flatten()
        self.getfclayer = nn.Sequential(
            nn.Dropout(p=self.dropout_p),
            nn.Linear(512, 1024),  # 9728 19456
            nn.ReLU()
        )
        self.centerfc = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.rnn = self.rnn_cell(self.rnn_input_dims, self.hidden_size, self.n_layers, dropout=self.dropout_p,
                                 bidirectional=self.bidirectional)
        self.lanout = nn.Linear(512, 2)
        self.emoout = nn.Linear(512, 4)

    def forward(self, x, lengths, x_t, lengths_t):  # torch.Size([B, 1, 311, 129]) # (B, 1 , D, T)
        len = self.get_seq_lens(lengths)
        x = self.conv(x)  # torch.Size([B, 32, 62, 14])
        x = x.transpose(1, 2).contiguous()
        x_size = x.size()
        x = x.view(x_size[0], x_size[1], x_size[2] * x_size[3])  # (B, T , C* D)
        x = x.transpose(0, 1).contiguous()
        x = nn.utils.rnn.pack_padded_sequence(x, lengths= len, enforce_sorted=False)
        x, h_state = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = x.transpose(0, 1).contiguous()
        mu = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        pooled = torch.cat((mu,std),dim=1)
        x = self.dropout(pooled)
        reversal = ReverseLayerF.apply(x , 1)
        emo = self.emoout(x)
        lan = self.lanout(reversal)

        len_t = self.get_seq_lens(lengths_t)
        t = self.conv(x_t)  # torch.Size([B, 32, 62, 14])
        t = t.transpose(1, 2).contiguous()
        t_size = t.size()
        t = t.view(t_size[0], t_size[1], t_size[2] * t_size[3])  # (B, T , C* D)
        t = t.transpose(0, 1).contiguous()
        t = nn.utils.rnn.pack_padded_sequence(t, lengths=len_t, enforce_sorted=False)
        t, h_state = self.rnn(t)
        t, _ = nn.utils.rnn.pad_packed_sequence(t)
        t = t.transpose(0, 1).contiguous()
        t_mu = torch.mean(t, dim=1)
        t_std = torch.std(t, dim=1)
        t_pooled = torch.cat((t_mu, t_std),dim=1)
        t = self.dropout(t_pooled)
        t = ReverseLayerF.apply(t , 1)
        lan_t = self.lanout(t)

        return emo, lan, lan_t

    def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.squeeze().int()