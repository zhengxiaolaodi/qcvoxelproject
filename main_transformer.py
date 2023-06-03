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
from data_voxel_reshape import nyu2_18cls_voxel_dsp220, nyu2_1449_6cls_voxel_dsp200
from vit import DGCNN_voxel_reshape
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import time
import h5py
import json
import gensim


def _init_():
    if not os.path.exists('/data4/zb/model/3D_IKEA/voxelres0.2_downsmp_to200_voxelnum_356_voxeldim356_transformer'):
        os.makedirs('/data4/zb/model/3D_IKEA/voxelres0.2_downsmp_to200_voxelnum_356_voxeldim356_transformer')
    # if not os.path.exists('/data4/zb/model/model_nyu2/sub_22cls/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_transformer/model_n0_knn_voxel_sequence_vcls760'):
    #     os.makedirs('/data4/zb/model/model_nyu2/sub_22cls/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_transformer/model_n0_knn_voxel_sequence_vcls760')


def train(args, io):
    #os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2,3,4,5,6,7"  ##########################   更换主显卡

    train_loader = DataLoader(nyu2_1449_6cls_voxel_dsp200('train'), num_workers=8,
                             batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = DGCNN_voxel_reshape(args).to(device)
    elif args.model == 'dgcnn':
        #model = DGCNN_voxel_reshape(args, output_channels=18).to(device)      #####################################################
        model = DGCNN_voxel_reshape(
            num_classes=6,                                                                #####################################
            dim = 356,
            depth=6,
            heads=16,
            mlp_dim=1024,
            dim_head=64,
            dropout=0.5,  ### for tranformer
            emb_dropout=0.1  ### for embedding
        ).cuda()
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = torch.nn.DataParallel(model, device_ids=[ 0, 1, 2, 3, 5,7])
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    ##############   加载预训练的voxel dgcnn
    model_dict = model.state_dict()
    pretrain_voxel_dgcnn = torch.load(
        '/data4/zb/model/3D_IKEA/pretrain_for_voxel_dgcnn_by_IKEA_6cls_vfh_cluster_label356/model_pretrain_for_voxel356_vfh_nomean_198.t7')
    pretrainning = {k: v for k, v in pretrain_voxel_dgcnn.items() if k in model_dict}
    model_dict.update(pretrainning)
    model.load_state_dict(model_dict)
    ##############


    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        ### model.module name: cls_token, voxel_embedding, transformer, classifer
        opt = optim.Adam([{'params': model.module.cls_token, 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.voxel_embedding.parameters(), 'lr': args.lr*0.01, 'weight_decay':1e-4},    ######################
                          {'params': model.module.transformer.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.classifier.parameters(), 'lr': args.lr, 'weight_decay': 1e-4}
                          ])

    #scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    scheduler_steplr = StepLR(opt, step_size= 10, gamma= 0.8)
    criterion = cal_loss

    ###########################  for prosessing interprut
    # for pre_epoch in range(34):
    #     scheduler_steplr.step()
    # model_path = './model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_transformer/model_n0_knn_voxel_sequence_vcls760/model_transformer_33.t7'
    # model.load_state_dict(torch.load(model_path))
    ###############################

    best_test_acc = 0
    for epoch in range(0, args.epochs, 1):
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
            voxel_sequence = torch.LongTensor(voxel_sequence).cuda()
            batch_size = args.batch_size
            opt.zero_grad()
            logits = model(data_arr, cloud_len_list, voxel_sequence)   #####
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
        torch.save(model.state_dict(), '/data4/zb/model/3D_IKEA/voxelres0.2_downsmp_to200_voxelnum_356_voxeldim356_transformer/model_transformer_%s.t7'%(epoch))





def mytest(args, io):
    test_loader = DataLoader(nyu2_1449_6cls_voxel_dsp200('test'), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda:1" if args.cuda else "cpu")

    #Try to load models
    #model = DGCNN_voxel_reshape(args, output_channels=18).to(device)                     ######################################
    model = DGCNN_voxel_reshape(
        num_classes=9,                                                                       #########################################
        dim=748,                                                         ###########################################
        depth=6,
        heads=16,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.5,  ### for tranformer
        emb_dropout=0.1  ### for embedding
    ).cuda(1)
    model = torch.nn.DataParallel(model, device_ids=[1,2,3,4,5,6,7])

    #####################################

    for iii in [14]:   #### test the last 50 models
        model_path = '/data4/zb/model/sun_voxel/voxelres0.2_downsmp_to200_voxelnum_748_voxeldim748_transformer/model_transformer_%s.t7'%(iii)
        model.load_state_dict(torch.load(model_path))
        model = model.eval()
        test_acc = 0.0
        count = 0.0

        test_pred = torch.zeros(args.test_batch_size)
        test_true = torch.zeros(args.test_batch_size)
        voxel_fea_all = []
        ground_truth = []
        voxel_num_list = []
        frame_pred = []
        pred_list = []
        log_list = []
        for i, (data, label, cloud_len_list, voxel_sequence) in enumerate(test_loader):

            data_arr = torch.FloatTensor(data).cuda()
            cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
            voxel_sequence = torch.LongTensor(voxel_sequence).cuda()
            batch_size = args.test_batch_size
            logits, voxel_fea = model(data_arr, cloud_len_list, voxel_sequence)    ###############

            label = torch.LongTensor(label).cuda()
            preds = logits.max(dim=1)[1]
            count += batch_size

            voxel_fea_all.append(voxel_fea.detach().cpu().numpy())

            ground_truth.append(label.cpu().numpy())
            pred_list.append(preds.cpu().numpy())
            log_list.append(logits.detach().cpu().numpy())

            voxel_num_list.append(cloud_len_list.cpu().numpy())
            frame_pred.append(logits.detach().cpu().numpy())
            #print(test_true.shape, label.shape)
            test_true = torch.cat((test_true, label.cpu().view(cloud_len_list.shape[0])), 0)
            test_pred = torch.cat((test_pred, preds.detach().cpu()), 0)

        voxel_fea_all = np.concatenate(voxel_fea_all)
        #np.savetxt('/home/zb/fxm_voxel_dgcnn/result/nyu2_1449_voxel_fea_%s.log'%(iii), voxel_fea_all)
        ground_truth = np.concatenate(ground_truth)
        pred_list = np.concatenate(pred_list)
        log_list = np.concatenate(log_list)
        # voxel_num_list = np.concatenate(voxel_num_list)
        # frame_pred_arr = np.concatenate(frame_pred)
        np.savetxt('/data4/zb/fxm_dgcnn_data/cls_check/real/real_pred_list_%s.log'%(iii), pred_list)            ####################
        np.savetxt('/data4/zb/fxm_dgcnn_data/cls_check/real/real_ground_truth_%s.log' % (iii), ground_truth)     #####################
        np.savetxt('/data4/zb/fxm_dgcnn_data/cls_check/real/real_logits_%s.log' % (iii), log_list)
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
    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')                       ## 原３２，超显存了
    parser.add_argument('--test_batch_size', type=int, default=14, metavar='batch_size',
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
    parser.add_argument('--eval', type=bool,  default= True,        #### train(f) or test(t)
                        help='evaluate the model')
    parser.add_argument('--point_num', type=int, default=200,
                        help='num of points to use')
    parser.add_argument('--voxel_num', type=int, default=356,  ###################
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=10,
                        help='k in voxel')
    parser.add_argument('--v_k', type=int, default=20,
                        help='k for voxel')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=256, metavar='N',      ####################
                        help='Dimension of embeddings')
    parser.add_argument('--voxel_cls', type=int, default=356, metavar='N',     ##################
                        help='classification of voxels')
    parser.add_argument('--model_path', type=str,\
                        default='/data4/zb/model/model_nyu2/1449_6cls/',\
                        metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('/home/zb/fxm_voxel_dgcnn/result/tmp.log')
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
        mytest(args, io)
