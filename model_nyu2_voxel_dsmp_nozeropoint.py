#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_voxel_reshape import nyu2_18cls_voxel_dsp220
import time
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import cal_loss, IOStream


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)      ## size [B , N, N]
    xx = torch.sum(x**2, dim=1, keepdim=True)   ### 　fxm: [B,C,N]尺寸的数据，按C相加，得到[B,1,N]的数据
    pairwise_distance = -xx - inner - xx.transpose(2, 1)   #由于后一步排序从小到大所以这里都乘-1  size [B , N, N]
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)  ## fxm:　tensor.topk会对最后一维排序,
        #　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　输出[最大k个值,对应的检索序号]，[1]表示输出索引
    return idx


def knn_nozero(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  ## size [B , N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  ### 　fxm: [B,C,N]尺寸的数据，按C相加，得到[B,1,N]的数据
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 由于后一步排序从小到大所以这里都乘-1  size [B , N, N]

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)  ## fxm:　tensor.topk会对最后一维排序,
    # 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　输出[最大k个值,对应的检索序号]，[1]表示输出索引
    #### blow for abandon [0,0,...,0]padding influence for knn neighbor seaching
    b_s, dim, num = x.shape
    xx_one = torch.ones(b_s, 1, num).cuda()
    xx_large = xx_one * 10000000
    xx_zero_sign = torch.where(xx> 0, xx_one, xx_large)
    pairwise_distance = pairwise_distance* xx_zero_sign
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    ####


    return idx




def get_graph_feature(x, k=10, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points    #[b,1,1]
    #idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points   # fxm: cpu测试版

    idx = idx + idx_base   # [b,n,k]

    idx = idx.view(-1)  #fxm 这里的idx是一个向量长度b*n*k,每个b中每个点最近邻的k个点的编号索引
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]   #[b*n*k,c]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  ## fxm: 将每个点重复k次，x的维度是[b,n,k,d]
    #                                                                           ,和feature一样
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # b之后按xyz...分(这里维度乘２因为拼接了原值与差值)，再按n分，最后是k

    return feature


def get_graph_feature_nozero(x, k=10, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_nozero(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # [b,1,1]
    # idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points   # fxm: cpu测试版

    idx = idx + idx_base  # [b,n,k]

    idx = idx.view(-1)  # fxm 这里的idx是一个向量长度b*n*k,每个b中每个点最近邻的k个点的编号索引

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [b*n*k,c]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  ## fxm: 将每个点重复k次，x的维度是[b,n,k,d]
    #                                                                           ,和feature一样
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()  # b之后按xyz...分(这里维度乘２因为拼接了原值与差值)，再按n分，最后是k

    return feature

class DGCNN_voxel_reshape(nn.Module):
    def __init__(self, args, output_channels):
        super(DGCNN_voxel_reshape, self).__init__()
        self.args = args
        self.k = args.k      ## 指　voxel里最近邻点的个数
        self.v_k = args.v_k  ## 指　voxel特征之间最近邻的个数
        self.point_num = args.point_num   ## 指每个voxel中有多少点
        self.voxel_cls = args.voxel_cls   ## 可以理解成有多少类体素，比如墙面体素，地面体素，圆形体素等等
        self.batch_size = args.batch_size
        self.voxel_num = args.voxel_num
        
        #self.bn1 = nn.BatchNorm2d(32)
        #self.bn2 = nn.BatchNorm2d(32)
        #self.bn3 = nn.BatchNorm2d(64)
        #self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(18, 32, kernel_size=1, bias=False),
                                   #self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32*2, 32, kernel_size=1, bias=False),
                                   #self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
                                   #self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   #self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 1024, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(1024, 1024)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(1024, args.voxel_cls)

        ### blow is conv liner layer of voxel feature
        self.bn_conv6 = nn.BatchNorm2d(512)
        self.bn_conv7 = nn.BatchNorm2d(256)
        self.bn_conv8 = nn.BatchNorm1d(args.emb_dims)
        self.conv6 = nn.Sequential(nn.Conv2d(2*args.voxel_cls, 512, kernel_size=1, bias=False),
                                   self.bn_conv6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(2*512, 256, kernel_size=1, bias=False),
                                   self.bn_conv7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(768, args.emb_dims, kernel_size=1, bias=False),
                                      self.bn_conv8,
                                      nn.LeakyReLU(negative_slope=0.2))
        self.linear4 = nn.Linear(args.emb_dims * 2, 256, bias=False)
        self.bn_line4 = nn.BatchNorm1d(256)
        self.dp4 = nn.Dropout(p=args.dropout)
        self.linear5 = nn.Linear(256, output_channels)

    def forward(self, input, cloud_len_list, voxel_num):

        ###### handling the input,from [b,1883,200,3] to [tt,200,3]
        # len_cloud = cloud_len_list.shape[0]
        # x = torch.zeros((sum(cloud_len_list), 220, 3)).cuda()
        # accout = 0
        # for i in range(len_cloud):
        #     x[accout:accout+cloud_len_list[i], :, :] = input[i, :cloud_len_list[i], :, :]
        #     accout += cloud_len_list[i]
        # del input
        ######
        x = input.view(-1, self.point_num, 9)#x=(680,220,9)
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)                 ## x维度[b,x,n]  b_s = 16 x 169
        x = get_graph_feature_nozero(x, k=self.k)     ## 维度是[b,2*3,n,k]
        x = self.conv1(x)                      ## 维度[b,16,n,k]
        x1 = x.max(dim=-1, keepdim=False)[0]   ## 维度[b,16,n]

        x = get_graph_feature_nozero(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]   ##  [b,16,n]

        x = get_graph_feature_nozero(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]   ##  [b,32,n]

        x = get_graph_feature_nozero(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]   ##  [b,64,n]

        x = torch.cat((x1, x2, x3, x4), dim=1) ##  [b,128,n]

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) ## x dim is (b,c,n), after pooling dim is (b,c,1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1) ## after viewing dim is (b,c)
        ## 为了消除填充的０点的影响，平均池化不能要，最大值池化对０点不敏感可以使用
        x = torch.cat((x1, x2), 1)
        #print(x.shape)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        #voxel_fea = x

        #print(x.shape, 'voxel dim')

        ### blow precessing voxel feature

        # xx = torch.zeros(len_cloud, voxel_num, self.args.voxel_cls).cuda()  # xx means voxel features
        # start_num = 0
        # for xx_i in range(len_cloud):
        #     xx[xx_i, :cloud_len_list[xx_i], :] = x[start_num : start_num + cloud_len_list[xx_i], :]
        #     start_num += cloud_len_list[xx_i]
        #print(xx.shape, 'voxel feature')

        xx = x.view(int(x.size(0)/self.voxel_num), self.voxel_num, -1)#x dim is (b*v, f), split x to (b,v,f)
        x = xx.permute(0, 2, 1)

        x = get_graph_feature_nozero(x, k=self.v_k)
        x = self.conv6(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature_nozero(x1, k=self.v_k)
        x = self.conv7(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2), dim=1)
        #print(x.shape)
        x = self.conv8(x)
        #print(x.shape)
        x1 = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        x = torch.cat((x1, x2), 1)
        #print(x.shape)
        x = F.leaky_relu((self.linear4(x)), negative_slope=0.2)
        x = self.dp4(x)
        x = self.linear5(x)
        #print('out size:', x.shape)

        return x

if __name__ == "__main__":
    class arg():
        def __init__(self):
            self.k = 10
            self.v_k = 20
            self.dropout = 0.5
            self.emb_dims = 256
            self.point_num = 200
            self.lr = 0.001
            self.epochs = 100
            self.voxel_cls = 759
            self.voxel_num = 759
            self.batch_size = 8
    arg = arg()

    model = DGCNN_voxel_reshape(arg, output_channels=18).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
    test_loader = DataLoader(nyu2_18cls_voxel_dsp220('test'), num_workers=8,
                             batch_size=arg.batch_size, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, arg.epochs, eta_min=arg.lr)


    for i, (data, label, cloud_len_list, voxel_seqence) in enumerate(test_loader):
        """
        label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
        因此label的长度是１,也没有shape函数，本质上label是一个列表
        """
        print('%s batch: %s'%(i, data.shape))
        #print(label)
        data = torch.FloatTensor(data).cuda()
        cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
        out = model(data, cloud_len_list, voxel_num=759)
        print(out.shape)
        if i >0:
            print(out.shape)
            print('------------------------')
            break

    ##print(model.parameters)

    ### backforward


        label = torch.LongTensor(label).cuda()
        loss = cal_loss(out, label)
        loss.backward()
        opt.step()


    # test = modelnet40_voxel('test')
    # test_loader = DataLoader(modelnet40_voxel('train'), num_workers=8,
    #                          batch_size=16, shuffle=False, collate_fn=my_collate)
    # for i, (data, label) in enumerate(test_loader):
    #     """
    #     label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
    #     因此label的长度是１,也没有shape函数，本质上label是一个列表
    #     """
    #     if i == 0:
    #         label = label.cuda()
    #         time1 = time.time()
    #         pred = model(data)
    #         print(pred.shape)
    #         time2 = time.time()
    #         print('time cost of forward per bitch: ', time2 - time1)
    #         opt.zero_grad()
    #         loss = criterion(pred, label)
    #         loss.backward()
    #         opt.step()
    #         time3 = time.time()
    #         print('time cost of background per bitch: ', time3 - time2)
    #         break
    #     break


    # xx = torch.randn(1* 169, 3, 1260).cuda()
    # knn(xx, 10)