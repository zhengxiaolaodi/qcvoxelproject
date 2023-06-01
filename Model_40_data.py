import os
import sys
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



def load_data(mode):
    if mode == 'train':
        path = '/data4/zb/data_modelnet40/modelnet40_res0.3_sphere200/'
        all_data = []
        all_label = []
        all_voxel_num = []
        all_voxel_seq = []
        for i in range(5):
            f = h5py.File(path + 'modelnet40_res0.3_sphere200_voxel_169_seq_train%s'%(i), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            voxel_num = f['voxel_num'][:].astype('int64')
            voxel_seq = f['voxel_sequence'][:].astype('int64')
            f.close()

            all_data.append(data)
            all_label.append(label)
            all_voxel_num.append(voxel_num)
            all_voxel_seq.append(voxel_seq)

        all_data = np.concatenate(all_data, axis= 0)
        all_label = np.concatenate(all_label, axis= 0)
        all_voxel_num = np.concatenate(all_voxel_num, axis= 0)
        all_voxel_seq = np.concatenate(all_voxel_seq, axis= 0)

    if mode == 'test':
        path = '/data4/zb/data_modelnet40/modelnet40_res0.3_sphere200/'
        all_data = []
        all_label = []
        all_voxel_num = []
        all_voxel_seq = []
        for i in range(2):
            f = h5py.File(path + 'modelnet40_res0.3_sphere200_voxel_169_seq_test%s'%(i), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            voxel_num = f['voxel_num'][:].astype('int64')
            voxel_seq = f['voxel_sequence'][:].astype('int64')
            f.close()

            all_data.append(data)
            all_label.append(label)
            all_voxel_num.append(voxel_num)
            all_voxel_seq.append(voxel_seq)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_voxel_num = np.concatenate(all_voxel_num, axis=0)
        all_voxel_seq = np.concatenate(all_voxel_seq, axis=0)

    return all_data, all_label, all_voxel_num, all_voxel_seq

def load_data_sphere(mode):
    if mode == 'train':
        path = '/data4/zb/data_modelnet40/modelnet40_res0.3_sphere200/'
        all_data = []
        all_label = []
        all_voxel_num = []
        all_voxel_seq = []
        all_voxel_radius = []
        for i in range(5):
            f = h5py.File(path + 'modelnet40_res0.3_sphere200_voxel_169_seq_train%s'%(i), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            voxel_num = f['voxel_num'][:].astype('int64')
            voxel_seq = f['voxel_sequence'][:].astype('int64')
            voxel_radius = f['voxel_radius'][:].astype('float32')
            f.close()

            all_data.append(data)
            all_label.append(label)
            all_voxel_num.append(voxel_num)
            all_voxel_seq.append(voxel_seq)
            all_voxel_radius.append(voxel_radius)

        all_data = np.concatenate(all_data, axis= 0)
        all_label = np.concatenate(all_label, axis= 0)
        all_voxel_num = np.concatenate(all_voxel_num, axis= 0)
        all_voxel_seq = np.concatenate(all_voxel_seq, axis= 0)
        all_voxel_radius = np.concatenate(all_voxel_radius, axis=0)

    if mode == 'test':
        path = '/data4/zb/data_modelnet40/modelnet40_res0.3_sphere200/'
        all_data = []
        all_label = []
        all_voxel_num = []
        all_voxel_seq = []
        all_voxel_radius = []
        for i in range(2):
            f = h5py.File(path + 'modelnet40_res0.3_sphere200_voxel_169_seq_test%s'%(i), 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            voxel_num = f['voxel_num'][:].astype('int64')
            voxel_seq = f['voxel_sequence'][:].astype('int64')
            voxel_radius = f['voxel_radius'][:].astype('float32')
            f.close()

            all_data.append(data)
            all_label.append(label)
            all_voxel_num.append(voxel_num)
            all_voxel_seq.append(voxel_seq)
            all_voxel_radius.append(voxel_radius)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        all_voxel_num = np.concatenate(all_voxel_num, axis=0)
        all_voxel_seq = np.concatenate(all_voxel_seq, axis=0)
        all_voxel_radius = np.concatenate(all_voxel_radius, axis=0)

    return all_data, all_label, all_voxel_num, all_voxel_seq, all_voxel_radius

class modelnet40_voxel(Dataset):
    """
    data 数组，结构是[b,v,n,3]　v= 169, n= 200
    label　数组，结构是[l1,l2,...,ln]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_seq = load_data(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        voxel_num = self.voxel_num[item]
        voxel_seq = self.voxel_seq[item]
        return pointcloud, label, voxel_num, voxel_seq

    def __len__(self):
        #print(len(self.data))
        return len(self.data)

class modelnet40_voxel_sphere(Dataset):
    """
    data 数组，结构是[b,v,n,3]　v= 169, n= 200
    label　数组，结构是[l1,l2,...,ln]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_seq, self.voxel_radius = load_data_sphere(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        voxel_num = self.voxel_num[item]
        voxel_seq = self.voxel_seq[item]
        voxel_radius = self.voxel_radius[item]
        return pointcloud, label, voxel_num, voxel_seq, voxel_radius

    def __len__(self):
        #print(len(self.data))
        return len(self.data)

def load_voxel_pcd_vfh():
    path = '/data4/zb/fxm_dgcnn_data/modelnet40_res0.5/pretrain/'
    f = h5py.File(path + 'modelnet40_train_voxel_res0.5_delno_vfh_10cloud_per_cls', 'r')
    pcd = f['data'][:].astype('float32')
    label = np.loadtxt(path + 'kmeans_label256_modelnet40_train_voxelres0.5_delno_gasd_10cloud_per_cls', dtype= int)
    return pcd, label

class modelnet40_40cls_10percls_voxel_pcd_vfh_cluster_label(Dataset):
    def __init__(self):
        self.data, self.label = load_voxel_pcd_vfh()

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        #print(len(self.data))
        return len(self.data)


if __name__ == '__main__':
    #train = modelnet40_voxel('train')
    # test = modelnet40_voxel('train')
    # count = 0
    # for data, label in test:
    #     count += 1
    # print(count)

    # test_loader = DataLoader(modelnet40_voxel('train'), num_workers=8,
    #                           batch_size=6, shuffle=False)
    #
    # for i, (data, label, voxel_num, voxel_seq) in enumerate(test_loader):
    #     """
    #     label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
    #     因此label的长度是１,也没有shape函数，本质上label是一个列表
    #     """
    #     if i == 0:
    #         print(data.shape, label.shape, voxel_num.shape, voxel_seq.shape)
    #         break
    #     break

    # test_set = modelnet40_voxel('train')
    # data0 = test_set.__getitem__(0)
    # print(len(data0),len(data0[0]),len(data0[0][0]))

    test_loader = DataLoader(modelnet40_voxel_sphere('train'), num_workers=8,
                             batch_size=16, shuffle=False)

    for i, (data, label,_,v_s,v_r) in enumerate(test_loader):
        """
        label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
        因此label的长度是１,也没有shape函数，本质上label是一个列表
        """
        if i == 0:
            print(data.shape, label.shape, v_s.shape, v_r.shape)
            print(label)

            break
        break
    print(torch.cuda.device_count())
