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
from data_4096 import nyu2_4096
from model_nyu2_4096 import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import time


def _init_():
    if not os.path.exists('/data4/zb/model/IKEA_4096'):
        os.makedirs('/data4/zb/model/IKEA_4096')


def train(args, io):
    train_loader = DataLoader(nyu2_4096(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(nyu2_4096(partition='test', num_points=args.num_points), num_workers=8,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda:1" if args.cuda else "cpu")
    print(1111111111111111111111111111)

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args,6).to(1)
    elif args.model == 'dgcnn':
        model = DGCNN(args, 6).cuda(1)                                   ###################################   change the cls number
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model, device_ids=[1,2])
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    scheduler_steplr = StepLR(opt, step_size= 10, gamma= 0.8)
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        scheduler.step()
        currect_lr = scheduler.get_last_lr()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()   #对于一些含有BatchNorm，Dropout等层的模型，在训练时使用的forward和验证时使用的forward在计算上不太一样.需要设置
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.cuda(1), label.cuda(1)#.squeeze()
            data = data.permute(0, 2, 1)    # trans to [b,3,n]
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, lr %6f, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 currect_lr[0],
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()      #
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.cuda(1), label.cuda(1)#.squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), '/data4/zb/model/model_nyu2_4096/model_6cls_pointnet.t7' )


def mytest(args, io):
    test_loader = DataLoader(nyu2_4096(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models

    #model = DGCNN(args, 6).to(device)

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args, 6).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, 6).to(device)                                   ###################################   change the cls number
    else:
        raise Exception("Not implemented")
    print(str(model))











    model = nn.DataParallel(model)
    model_path = '/data4/zb/model/model_nyu2_4096/model_6cls_pointnet.t7'
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    voxel_fea_all = []
    ground_truth = []
    voxel_num_list = []
    frame_pred = []
    pred_list = []
    log_list = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

        ground_truth.append(label.cpu().numpy())
        pred_list.append(preds.cpu().numpy())
        log_list.append(logits.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    ground_truth = np.concatenate(ground_truth)
    pred_list = np.concatenate(pred_list)
    log_list = np.concatenate(log_list)
    np.savetxt('/data4/zb/fxm_dgcnn_data/cls_check/nyu2_1449_4096/dgcnn_pred_list.log', pred_list)  ####################
    np.savetxt('/data4/zb/fxm_dgcnn_data/cls_check/nyu2_1449_4096/dgcnn_ground_truth.log',ground_truth)  #####################
    np.savetxt('/data4/zb/fxm_dgcnn_data/cls_check/nyu2_1449_4096/dgcnn_logits.log', log_list)


    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',                 ############################################
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='sun_4096', metavar='N'
                        )
    parser.add_argument('--batch_size', type=int, default=24, metavar='batch_size',
                        help='Size of batch)')  ## 原３２，超显存了
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')  ##  原１６
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
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
    parser.add_argument('--eval', type=bool,  default=True,                          ##################################     train(f) or test(t)
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
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

    # if not args.eval:
    #     train(args, io)
    # else:
    #     mytest(args, io)
    train(args, io)
    mytest(args, io)
