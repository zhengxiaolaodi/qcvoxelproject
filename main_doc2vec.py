#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_voxel_reshape import nyu2_18cls_voxel_dsp220
from model_nobn_doc2vec import DGCNN_voxel_reshape
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import time
import h5py
import json
import gensim

#############nyu2数据
def _init_():
    if not os.path.exists('./model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_doc2vec'):
        os.makedirs('./model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_doc2vec')
    if not os.path.exists('./model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_doc2vec/model_n0_knn_voxel_sequence_vcls759'):
        os.makedirs('./model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_doc2vec/model_n0_knn_voxel_sequence_vcls759')


def train(args, io):
    train_loader = DataLoader(nyu2_18cls_voxel_dsp220('train'), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = DGCNN_voxel_reshape(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_voxel_reshape(args, output_channels=18).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4,7])
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    ##### doc_vec model and corpora index
    gensim_model = gensim.models.doc2vec.Doc2Vec.load('./doc2vec_model/gensim-model-dim800-1')
    with open('word_list.json', 'r') as f:
        word_list = json.load(f)
        f.close()
    #####################################

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    #scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    scheduler_steplr = StepLR(opt, step_size= 10, gamma= 0.8)
    criterion = cal_loss

    ###########################  for prosessing interprut
    # for pre_epoch in range(77):
    #     scheduler_steplr.step()
    # model_path = './model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_enbed/model_n0_knn_voxel_sequence_vcls759/model_nobn_enbed_76.t7'
    # model.load_state_dict(torch.load(model_path))
    ###############################

    best_test_acc = 0
    for epoch in range(args.epochs):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        scheduler_steplr.step()
        currect_lr = scheduler_steplr.get_lr()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()   #对于一些含有BatchNorm，Dropout等层的模型，在训练时使用的forward和验证时使用的forward在计算上不太一样.需要设置
        train_pred = torch.zeros(args.batch_size)
        train_true = torch.zeros(args.batch_size)
        for i, (data, label, cloud_len_list, voxel_sequence) in enumerate(train_loader):
            if i % 200 ==0:
                print(i)
            data_arr = torch.FloatTensor(data).cuda()
            cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
            # data = data.permute(0, 2, 1)    # trans to [b,3,n]
            #voxel_sequence = torch.LongTensor(voxel_sequence).cuda()
            batch_size = args.batch_size
            opt.zero_grad()
            logits = model(data_arr, cloud_len_list, 759, gensim_model, word_list)   #####
            label = torch.LongTensor(label).cuda()
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            # train_true.append(label.cpu().numpy())
            # train_pred.append(preds.detach().cpu().numpy())
            train_true = torch.cat((train_true, label.cpu().view(cloud_len_list.shape[0])), 0)
            train_pred = torch.cat((train_pred, preds.detach().cpu()), 0)
        # train_true = np.concatenate(train_true)
        # train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, lr %6f, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 currect_lr[0],
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true[batch_size:], train_pred[batch_size:]),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true[batch_size:], train_pred[batch_size:]))
        io.cprint(outstr)
        torch.save(model.state_dict(), './model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_doc2vec/model_n0_knn_voxel_sequence_vcls759/model_nobn_doc2vec_%s.t7'%(epoch))




def test(args, io):
    test_loader = DataLoader(nyu2_18cls_voxel_dsp220('test'), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN_voxel_reshape(args, output_channels=18).to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    ##### doc_vec model and corpora index
    gensim_model = gensim.models.doc2vec.Doc2Vec.load('./doc2vec_model/gensim-model-dim800-1')
    with open('word_list.json', 'r') as f:
        word_list = json.load(f)
        f.close()
    #####################################

    for iii in range(200):   #### test the last 50 models
        model_path = './model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_doc2vec/model_n0_knn_voxel_sequence_vcls759/model_nobn_doc2vec_%s.t7'%(iii)
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
        test_acc = 0.0
        count = 0.0

        test_pred = torch.zeros(args.test_batch_size)
        test_true = torch.zeros(args.test_batch_size)
        voxel_fea_all = []
        ground_truth = []
        voxel_num_list = []
        for i, (data, label, cloud_len_list, voxel_sequence) in enumerate(test_loader):

            data_arr = torch.FloatTensor(data).cuda()
            cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
            #voxel_sequence = torch.LongTensor(voxel_sequence).cuda()
            batch_size = args.test_batch_size
            logits = model(data_arr, cloud_len_list, 759, gensim_model, word_list)    ###############

            label = torch.LongTensor(label).cuda()
            preds = logits.max(dim=1)[1]
            count += batch_size
            ###voxel_fea_all.append(voxel_fea.detach().cpu().numpy())
            ground_truth.append(label.cpu().numpy())
            voxel_num_list.append(cloud_len_list.cpu().numpy())
            #print(test_true.shape, label.shape)
            test_true = torch.cat((test_true, label.cpu().view(cloud_len_list.shape[0])), 0)
            test_pred = torch.cat((test_pred, preds.detach().cpu()), 0)

        ####voxel_fea_all = np.concatenate(voxel_fea_all)
        ground_truth = np.concatenate(ground_truth)
        voxel_num_list = np.concatenate(voxel_num_list)
        ###print(voxel_fea_all.shape, ground_truth.shape, voxel_num_list.shape)

        # ff = h5py.File('../data/nyu2_18cls_2/voxel_fea_759_model199_onebn_train.h5', 'w')
        # ff.create_dataset('voxel_fea', data=voxel_fea_all)
        # ff.create_dataset('label', data=ground_truth)
        # ff.create_dataset('voxel_num', data=voxel_num_list)
        # ff.close()

        test_acc = metrics.accuracy_score(test_true[args.test_batch_size:], test_pred[args.test_batch_size:])
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true[args.test_batch_size:], test_pred[args.test_batch_size:])
        outstr = 'Test :%s : test acc: %.6f, test avg acc: %.6f'%(iii, test_acc, avg_per_class_acc)
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='nyu2_18cls_crop_1.6_2_voxel_downsmp', metavar='N'
                        )
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')                       ## 原３２，超显存了
    parser.add_argument('--test_batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')                           ##  原１６
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default= False,        #### train(f) or test(t)
                        help='evaluate the model')
    parser.add_argument('--point_num', type=int, default=220,
                        help='num of points to use')
    parser.add_argument('--voxel_num', type=int, default=759,  ###################
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=10,
                        help='k in voxel')
    parser.add_argument('--v_k', type=int, default=20,
                        help='k for voxel')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=256, metavar='N',      ####################
                        help='Dimension of embeddings')
    parser.add_argument('--voxel_cls', type=int, default=759, metavar='N',     ##################
                        help='classification of voxels')
    parser.add_argument('--model_path', type=str,\
                        default='./model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_enbed/model_n0_knn_voxel_sequence_vcls759/model_enbed_dim3_150~199.t7',\
                        metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('./result/run_nobn_doc2vec.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
