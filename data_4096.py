#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""



import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def download():##下载数据
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  ## 当前绝对路径
    DATA_DIR = os.path.join(BASE_DIR, 'data')       ##data数据的路径
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):###加载数据集,h5py库为创建数据集和组
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')   # fxm: data.shape is (b,n,3) n=2048
        label = f['label'][:].astype('int64')   # fxm: label.shape is (b,1)  label:0-39
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_nyu2_4096(mode):
    """
    read the downsample labeled nyu2 data. point nums are 4096
    """
    path = '/data4/zb/fxm_dgcnn_data/nyu2_1449_seq/4096_%s.h5'%(mode)
    f = h5py.File(path, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    return data, label


def load_ik3d_4096(mode):
    """
    read the downsample labeled nyu2 data. point nums are 4096
    """
    path="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_dsp_4096_%s"%(mode)
    #path = '/data4/zb/fxm_dgcnn_data/nyu2_1449_seq/4096_%s.h5'%(mode)
    f = h5py.File(path, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    return data, label



class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        print(self.data.shape[0])
        return self.data.shape[0]

class nyu2_4096(Dataset):   ## train has 918 files, test has 227 file
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_nyu2_4096(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        #print(self.data.shape[0])
        return self.data.shape[0]



class d3ikea_4096(Dataset):   ## train has 918 files, test has 227 file
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ik3d_4096(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        #print(self.data.shape[0])
        return self.data.shape[0]









if __name__ == '__main__':
    train = nyu2_4096(4096)   ## 每个样例有２０４８个点
    test = nyu2_4096(4096, 'train')
    count = 0
    for data, label in test:
        count += 1
        print(data.shape)
        print(label.shape)
    print(count)

    train_loader = DataLoader(nyu2_4096(partition='train', num_points=4096), num_workers=8,
                              batch_size=16, shuffle=True, drop_last=True)

