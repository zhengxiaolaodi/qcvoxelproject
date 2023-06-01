#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import time
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import cal_loss
import torch.optim as optim


def knn(x, k):    ## x is [b,c,n]
    inner = -2*torch.matmul(x.transpose(2, 1), x)    ## [b,n,c]x[b,c,n]=[b,n,n]矩阵乘法
    xx = torch.sum(x**2, dim=1, keepdim=True)   ### 　fxm: [B,C,N]尺寸的数据，按C相加，得到[B,1,N]的数据
    pairwise_distance = -xx - inner - xx.transpose(2, 1)   #由于后一步排序从小到大所以这里都乘-1
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)  ## fxm:　tensor.topk会对最后一维排序,
        #　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　输出[最大k个值,对应的检索序号]，[1]表示输出索引

    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    #idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points   # fxm: cpu测试版

    idx = idx + idx_base

    idx = idx.view(-1)  #fxm 这里的idx是一个向量长度b*n*k,每个b中每个点最近邻的k个点的编号索引
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  ## fxm: 将每个点重复k次，x的维度是[b,n,k,d]
    #                                                                           ,和feature一样
    feature = c((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # b之后按xyz...分(这里维度乘２因为拼接了原值与差值)，再按n分，最后是k
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=22):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)                 ## x维度[b,x,n]
        x = get_graph_feature(x, k=self.k)     ## 维度是[b,2*3,n,k]
        x = self.conv1(x)                      ## 维度[b,64,n,k]
        x1 = x.max(dim=-1, keepdim=False)[0]   ## 维度[b,64,n]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]   ## 维度[b,64,n]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]   ## 维度[b,128,n]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]   ## 维度[b,256,n]

        x = torch.cat((x1, x2, x3, x4), dim=1) ## 维度[b,512,n]
        #print(x.shape)

        x = self.conv5(x)                      ## 维度[b,args.emb_dims,n]
        #print(x.shape)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)    ## 维度[b,args.emb_dims]  相当与在ｎ个点中找个最大的，针对某一深度
        #print(x1.shape)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)             ## 维度[b,2048]
        #print(x.shape)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

if __name__ == "__main__":
    class arg():
        def __init__(self):
            self.k = 20
            self.emb_dims = 2048
            self.dropout = 0.2
    arg = arg()

    class arg_voxel():
        def __init__(self):
            self.k = 10
            self.v_k = 8
            self.dropout = 0.2
            self.emb_dims = 1024
            self.lr = 0.001
            self.epochs = 100
    arg_voxel = arg_voxel()
    #
    model = PointNet(arg, 9).cuda(1)
    x_in = torch.randn(2, 3, 4096).cuda(1)
    x_out = model(x_in)
    label = torch.randn(2, 9).cuda(1)
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    criterion = nn.MSELoss()
    loss = criterion(x_out, label)
    loss.backward()
    opt.step()

    # model = DGCNN(arg)
    # stat(model, (1,2048,3))

    # model = DGCNN_voxel_cpu(arg_voxel)
    # stat(model, (1, 2048, 3))

    #### 测试knn,  get_graph_feature两个函数用cpu, gpu时间差距
    ### for cpu: x = [16,128,2048] time 1.42s; x=[16, 3, 2048] time 0.31s ;x=[16, 3, 17048] time 16.07s
    ### for gpu: x = [16,128,2048] time 0.22s; x=[16, 3, 2048] time 0.21s ;x = [16,128,6048] time 0.24s 点再多显存不够
    # x = torch.randn(16, 128, 6048).cuda()
    # time1 = time.time()
    # x = get_graph_feature(x, k=20)
    # time2 = time.time()
    # print(time2 - time1)



