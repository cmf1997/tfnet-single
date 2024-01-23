#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : modules.py
@Time : 2023/11/09 11:21:40
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['IConv']

# code
class IConv(nn.Module):
    def __init__(self, out_channels, kernel_size, tf_len=39, stride=1, **kwargs):
        super(IConv, self).__init__()
        # h weight matrices 
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, tf_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.stride, self.kernel_size = stride, kernel_size
        self.reset_parameters()

    def forward(self, DNA_x, tf_x, **kwargs):
        bs = DNA_x.shape[0]
        kernel = F.relu(torch.einsum('nld,okl->nodk', tf_x, self.weight))
        outputs = F.conv1d(DNA_x.transpose(1, 2).reshape(1, -1, DNA_x.shape[1]),
                        kernel.contiguous().view(-1, *kernel.shape[-2:]), stride=self.stride, groups=bs)
        return outputs.view(bs, -1, outputs.shape[-1]) + self.bias[:, None]

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)