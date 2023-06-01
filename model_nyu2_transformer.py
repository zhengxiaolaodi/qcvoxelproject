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
import math


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

def positional_adding(x, voxel_sequence, cloud_len_list, d_model=760, max_len=759):
    """
    function: add the voxel sequence values to every elements of voxel features.
    Note: d_model(voxel_dim) must be a even number  ; x size:[b, seq_len, voxel_dim] is same with output new_x
    """
    batch_size = x.shape[0]
    #new_x = torch.zeros(batch_size, x.shape[1], x.shape[2]).cuda()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).cuda()
    #print(div_term)
    for i in range(batch_size):
        pe = torch.zeros(max_len, d_model).cuda()
        position = voxel_sequence[i].unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, cloud_len_list[i]:] = 0
        x[i, 1:, :] = x[i, 1:, :] + pe
    new_x = x
    return new_x

def get_attn_pad_mask(seq_q, voxel_sequence):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    voxel_sequence size: [batch, seq_len]
    '''
    # batch_size, len_q = seq_q.size()
    # batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    batch_size = seq_q.size(0)
    len_q = seq_q.size(1)
    len_k = seq_q.size(1)
    voxel_sequence_plus1 = torch.ones(voxel_sequence.shape[0],voxel_sequence.shape[1]+1).cuda()
    voxel_sequence_plus1[:, 1:] = voxel_sequence
    pad_attn_mask = voxel_sequence_plus1.data.eq(0).unsqueeze(1)   # [batch_size, 1, len_k], True is masked
    # print(pad_attn_mask)
    # print(pad_attn_mask.shape)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]

        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn
        #return enc_outputs, enc_self_attn_mask



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs, voxel_sequence, cloud_len_list):
        '''
        enc_inputs: [batch_size, src_len]  fxm: size is [bathc, src_len, d_model]
        becausae inputs is voxel features not the word token exactly.
        '''
        # enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        enc_outputs = positional_adding(enc_inputs, voxel_sequence, cloud_len_list) # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, voxel_sequence) # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            #enc_self_attns.append(enc_self_attn)
        return enc_outputs#, enc_self_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().cuda()



    def forward(self, enc_inputs, voxel_sequence, cloud_len_list):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        #enc_outputs, enc_self_attns = self.encoder(enc_inputs, voxel_sequence)
        enc_outputs = self.encoder(enc_inputs, voxel_sequence, cloud_len_list)
        return enc_outputs

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

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
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
        self.transformer = Transformer()
        self.classifier = nn.Sequential(
            nn.Linear(args.voxel_cls, args.voxel_cls),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=args.dropout),
            nn.Linear(args.voxel_cls, 18)
        )


    def forward(self, input, cloud_len_list, voxel_sequence):

        ###### handling the input,from [b,1883,200,3] to [tt,200,3]
        # len_cloud = cloud_len_list.shape[0]
        # x = torch.zeros((sum(cloud_len_list), 220, 3)).cuda()
        # accout = 0
        # for i in range(len_cloud):
        #     x[accout:accout+cloud_len_list[i], :, :] = input[i, :cloud_len_list[i], :, :]
        #     accout += cloud_len_list[i]
        # del input
        ######
        x = input.view(-1, self.point_num, 3)
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
        x = x.view(int(x.size(0) / self.voxel_num), self.voxel_num, -1)
        #print(x.shape, 'voxel dim')
        cls_elem = (torch.ones(x.shape[0], 1, x.shape[2])*0.5).cuda()
        x = torch.cat((cls_elem, x ), dim=1)
        #print(x.shape, 'voxel dim')
        x = self.transformer(x, voxel_sequence, cloud_len_list)
        x = self.classifier(x[:, 0, :])
        #print('out size:', x.shape)

        return x

# Transformer Parameters
d_model = 760  # Embedding Size
d_ff = 1024  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 1 # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

if __name__ == "__main__":
    test_loader = DataLoader(nyu2_18cls_voxel_dsp220('test'), num_workers=8,
                             batch_size=2, shuffle=True)
    for i, (data, label, cloud_len_list, voxel_seqence) in enumerate(test_loader):
        print(voxel_seqence)
        print(voxel_seqence.shape)
        xx = torch.randn(2, 760, 760)
        mask = get_attn_pad_mask(xx, voxel_seqence)
        #print(mask)
        break

    # class arg():
    #     def __init__(self):
    #         self.k = 10
    #         self.v_k = 20
    #         self.dropout = 0.5
    #         self.emb_dims = 256
    #         self.point_num = 220
    #         self.lr = 0.001
    #         self.epochs = 100
    #         self.voxel_cls = 760
    #         self.voxel_num = 759
    #         self.batch_size = 8
    # arg = arg()
    #
    # # Transformer Parameters
    # d_model = 760  # Embedding Size
    # d_ff = 2048  # FeedForward dimension
    # d_k = d_v = 64  # dimension of K(=Q), V
    # n_layers = 6  # number of Encoder of Decoder Layer
    # n_heads = 8  # number of heads in Multi-Head Attention
    #
    # model = DGCNN_voxel_reshape(arg, output_channels=18).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2,3])
    # test_loader = DataLoader(nyu2_18cls_voxel_dsp220('test'), num_workers=8,
    #                          batch_size=arg.batch_size, shuffle=True)
    #
    # opt = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-4)
    # scheduler = CosineAnnealingLR(opt, arg.epochs, eta_min=arg.lr)
    #
    #
    # for i, (data, label, cloud_len_list, voxel_seqence) in enumerate(test_loader):
    #     """
    #     label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
    #     因此label的长度是１,也没有shape函数，本质上label是一个列表
    #     """
    #     print('%s batch: %s'%(i, data.shape))
    #     #print(label)
    #     data = torch.FloatTensor(data).cuda()
    #     cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
    #     voxel_seqence = torch.LongTensor(voxel_seqence).cuda()
    #     out = model(data, cloud_len_list, voxel_seqence)
    #     print(out.shape)
    #     if i >0:
    #         print(out.shape)
    #         print('------------------------')
    #         break

    ##print(model.parameters)

    ### backforward


        # label = torch.LongTensor(label).cuda()
        # loss = cal_loss(out, label)
        # loss.backward()
        # opt.step()


