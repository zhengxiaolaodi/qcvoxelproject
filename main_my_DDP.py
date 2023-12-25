from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='4,5,6,7'
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as Farr
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data_voxel_reshape import sunrgbd_9cls_voxel_multi
from vit_dgcnn_multi import DGCNN_voxel_reshape
#from vit_multipara import DGCNN_voxel_reshape
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import time
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler

def prepare():
    parser =argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='nyu2_18cls_crop_1.6_2_voxel_downsmp', metavar='N'
                        )
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')                       ## 原３２，超显存了
    parser.add_argument('--test_batch_size', type=int, default=2, metavar='batch_size',
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
    parser.add_argument('--eval', type=bool,  default=False,        #### train(f) or test(t)
                        help='evaluate the model')
    parser.add_argument('--point_num', type=int, default=200,
                        help='num of points to use')
    parser.add_argument('--voxel_num', type=int, default=748,  ###################
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=10,
                        help='k in voxel')
    parser.add_argument('--v_k', type=int, default=20,
                        help='k for voxel')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',      ####################
                        help='Dimension of embeddings')
    parser.add_argument("--local_rank", type=int, default=1,
                        help="number of cpu threads to use during batch generation")

    parser.add_argument('--use_mix_precision', default=False,                        #
                        action='store_true', help="whether to use mix precision")

    #################################
    parser.add_argument('--voxel_cls', type=int, default=749, metavar='N',     ##################
                        help='classification of voxels')
    parser.add_argument('--model_path', type=str,\
                        default='./model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_enbed/model_n0_knn_voxel_sequence_vcls759/model_enbed_dim3_150~199.t7',\
                        metavar='N',
                        help='Pretrained model path')
    args=parser.parse_args()
    return args

def get_ddp_generator(seed=3407):
    local_rank = int(os.environ['LOCAL_RANK'])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def init_ddp(local_rank):
    # 有了这一句之后，在转换device的时候直接使用 a=a.cuda()即可，否则要用a=a.cuda(local_rank)
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def train(model, train_loader, criterion, opt, scaler,scheduler_steplr,epoch,io):
    model.train()
    scheduler_steplr.step()
    currect_lr = scheduler_steplr.get_last_lr()
    ####################
    # Train
    ####################pycharm

    train_loss = 0.0
    count = 0.0
    model.train()  # 对于一些含有BatchNorm，Dropout等层的模型，在训练时使用的forward和验证时使用的forward在计算上不太一样.需要设置
    train_pred = torch.zeros(args.batch_size)
    train_true = torch.zeros(args.batch_size)
    for i, (data_02, data_04, data_08, label, cloud_len_list, voxel_sequence, voxel_point_number) in enumerate(
            train_loader):
        if i % 200 == 0:
            print(i)
        data_arr_02 = torch.FloatTensor(data_02).cuda()
        data_arr_04 = torch.FloatTensor(data_04).cuda()
        data_arr_08 = torch.FloatTensor(data_08).cuda()
        cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
        voxel_sequence = torch.LongTensor(voxel_sequence).cuda()
        voxel_point_number = torch.LongTensor(voxel_point_number).cuda()
        # data = data.permute(0, 2, 1)    # trans to [b,3,n]
        # voxel_sequence = torch.LongTensor(voxel_sequence).cuda()
        batch_size = args.batch_size
        opt.zero_grad()
        logits, voxel_fea = model(data_arr_02, data_arr_04, data_arr_08, cloud_len_list, voxel_sequence,
                                  voxel_point_number)
        label = torch.LongTensor(label).cuda()
        loss = criterion(logits, label)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        preds = logits.max(dim=1)[1]
        count += batch_size
        train_loss += loss.item() * batch_size
        # train_true.append(label.cpu().numpy())
        # train_pred.append(preds.detach().cpu().numpy())
        train_true = torch.cat((train_true, label.cpu().view(cloud_len_list.shape[0])), 0)
        train_pred = torch.cat((train_pred, preds.detach().cpu()), 0)
    outstr = 'Train %d, lr %6f, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                     currect_lr[0],
                                                                                     train_loss * 1.0 / count,
                                                                                     metrics.accuracy_score(
                                                                                         train_true[batch_size:],
                                                                                         train_pred[batch_size:]),
                                                                                     metrics.balanced_accuracy_score(
                                                                                         train_true[batch_size:],
                                                                                         train_pred[batch_size:]))
    io.cprint(outstr)
    # torch.save(model.modelstate_dict(),
    #            './model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_thin/model_n0_knn_voxel_sequence_vcls759/model_nobn_9feature_novxlpad_%s.t7' % (
    #                epoch))
    local_rank=int(os.environ['LOCAL_RANK'])
    if local_rank == 0:  ### 防止每个进程都保存一次
        torch.save({
            'model': model.state_dict(),
            'scaler': scaler.state_dict()
        },  './model/nyu2_crop_1.6_2_voxel0.2_downsmp_to220_thin/model_n0_knn_voxel_sequence_vcls759/model_nobn_9feature_novxlpad_%s.t7' % (
                   epoch))
    dist.destroy_process_group()


def main(args,io):
    train_sampler = torch.utils.data.distributed.DistributedSampler(sunrgbd_9cls_voxel_multi('train'))
    train_loader = torch.utils.data.DataLoader(
        sunrgbd_9cls_voxel_multi('train'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=train_sampler)

    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    model = DGCNN_voxel_reshape.cuda()  ### 模型的 forward 方法变了
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  ### 转换模型的 BN 层
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank
                                                            ])
    opt = optim.Adam([{'params': model.module.cls_token1, 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.voxel_embedding1.parameters(), 'lr': args.lr*0.01, 'weight_decay':1e-4},    ######################
                          {'params': model.module.transformer1.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.classifier.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.cls_token2, 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.voxel_embedding2.parameters(), 'lr': args.lr * 0.01,'weight_decay': 1e-4},  ######################
                          {'params': model.module.transformer2.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.classifier2.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.cls_token3, 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.voxel_embedding3.parameters(), 'lr': args.lr * 0.01,'weight_decay': 1e-4},  ######################
                          {'params': model.module.transformer3.parameters(), 'lr': args.lr, 'weight_decay': 1e-4},
                          {'params': model.module.classifier3.parameters(), 'lr': args.lr, 'weight_decay': 1e-4}
                          ])
    scheduler_steplr = StepLR(opt, step_size= 10, gamma= 0.8)
    criterion = cal_loss
    scaler = GradScaler()
    for epoch in range(args.epochs):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        if local_rank == 0:  ### 防止每个进程都输出一次
            print(f'begin training of epoch {epoch + 1}/{args.epochs}')
        train_loader.sampler.set_epoch(epoch)  ### 防止采样出 bug
        train(model, train_loader, criterion, opt, scaler,scheduler_steplr,epoch,io)



if __name__=='__main__':
    args = prepare()
    io = IOStream('./result/run_nobn_novxlpad.log')
    io.cprint(str(args))
    main(args,io)
