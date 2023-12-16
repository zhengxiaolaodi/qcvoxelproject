import os
import sys
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time






def load_data_sunrgbd_multi(mode):
    #"/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_02_test"
    path = "/data4/zb/fxm_dgcnn_data/3D_IKEA/"
    f = h5py.File(path + '3DIKEA_6cls_voxel_356_seq_02_%s' % (mode), 'r')
    data_02 = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num_02 = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
    voxel_sequence_02 = f['voxel_sequence'][:].astype('int64')
    voxel_sequence_pointnum_02 = f['voxel_point_number'][:].astype('int64')
    f.close()
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    path = "/data4/zb/fxm_dgcnn_data/3D_IKEA/"
    f = h5py.File(path + '3DIKEA_6cls_voxel_356_seq_04_%s' % (mode), 'r')
    data_04 = f['data'][:].astype('float32')
    voxel_num_04 = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
    voxel_sequence_04 = f['voxel_sequence'][:].astype('int64')
    voxel_sequence_04=voxel_sequence_04.reshape([voxel_sequence_04.shape[0],voxel_sequence_04.shape[1]])
    voxel_sequence_pointnum_04 = f['voxel_point_number'][:].astype('int64')
    f.close()
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    path = "/data4/zb/fxm_dgcnn_data/3D_IKEA/"
    f = h5py.File(path + '3DIKEA_6cls_voxel_356_seq_08_%s' % (mode), 'r')
    data_08 = f['data'][:].astype('float32')
    voxel_num_08 = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
    voxel_sequence_08 = f['voxel_sequence'][:].astype('int64')
    voxel_sequence_08 = voxel_sequence_08.reshape([voxel_sequence_08.shape[0], voxel_sequence_08.shape[1]])
    voxel_sequence_pointnum_08 = f['voxel_point_number'][:].astype('int64')
    f.close()
    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    voxel_num_result=np.concatenate((voxel_num_02,voxel_num_04,voxel_num_08),axis=-1)
    voxel_sequence_result=np.concatenate((voxel_sequence_02,voxel_sequence_04,voxel_sequence_08),axis=-1)
    voxel_pointnum_result=np.concatenate((voxel_sequence_pointnum_02,voxel_sequence_pointnum_04,voxel_sequence_pointnum_08),axis=-1)
    return data_02, data_04, data_08,label,voxel_num_result,voxel_sequence_result,voxel_pointnum_result




class sunrgbd_9cls_voxel_multi(Dataset):#这是提取数据的一个类
    def __init__(self, partition):
        self.data_02, self.data_04, self.data_08,self.label,self.voxel_num_result,self.voxel_sequence_result,self.voxel_pointnum_result = load_data_sunrgbd_multi(partition)
        self.partition = partition
    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        data_02_item = self.data_02[item]
        data_04_item = self.data_04[item]
        data_08_item = self.data_08[item]
        label = self.label[item]
        cloud_len_list = self.voxel_num_result[item]
        voxel_sequence = self.voxel_sequence_result[item]
        voxel_pointnum_result=self.voxel_pointnum_result[item]
        return data_02_item,data_04_item,data_08_item, label, cloud_len_list, voxel_sequence,voxel_pointnum_result
    def __len__(self):
        return len(self.data_02)









def load_data(mode):
    path = '/data4/zb/fxm_dgcnn_data/nyu2_sub_22cls_sequence/'
    if mode == 'train':
        all_data = []
        all_label = []
        all_voxel_num = []
        all_voxel_sequence = []
        for i in range(0,5,1):
            f = h5py.File(path + 'nyu2_sub_22cls_voxel759_dsp220_train_%s.h5'%(i), 'r')
            print(i)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            voxel_num = f['voxel_num'][:].astype('int64')
            voxel_sequence = f['voxel_sequence'][:].astype('int64')
            f.close()
            all_data.append(data)####append，在数据结尾添加
            all_label.append(label)
            all_voxel_num.append(voxel_num)
            all_voxel_sequence.append(voxel_sequence)
        # print(6)
        # all_data = np.concatenate(all_data, axis= 0)##concatenate拼接
        # print(6)
        # all_label = np.concatenate(all_label, axis= 0)
        # print(6)
        # all_voxel_num = np.concatenate(all_voxel_num, axis=0)
        # print(6)
        # all_voxel_sequence = np.concatenate(all_voxel_sequence, axis=0)
        all_label = np.concatenate(all_label, axis= 0)
        all_voxel_num = np.concatenate(all_voxel_num, axis=0)
        all_voxel_sequence = np.concatenate(all_voxel_sequence, axis=0)
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        all_data = np.concatenate(all_data, axis= 0)##concatenate拼接
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    if mode == 'test':
        f = h5py.File(path + 'nyu2_sub_22cls_voxel759_dsp220_test_0_21.h5', 'r')
        all_data = f['data'][:].astype('float32')
        all_label = f['label'][:].astype('int64')
        all_voxel_num = f['voxel_num'][:].astype('int64')
        all_voxel_sequence = f['voxel_sequence'][:].astype('int64')
        f.close()
    print(333)
    return all_data, all_label, all_voxel_num, all_voxel_sequence

def load_data_1449(mode):
    path = '/data4/zb/fxm_dgcnn_data/cls_check/'

    all_data = []
    all_label = []
    all_voxel_num = []
    all_voxel_sequence = []

    f = h5py.File(path + 'real_4cls_voxel_748_seq_%s'%(mode), 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')###astype修改数据类型
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    f.close()

    return data, label, voxel_num, voxel_sequence

def load_data_1449_sphere(mode):
    path = '/data4/zb/fxm_dgcnn_data/nyu2_1449_seq/res0.2_sphere200/'

    all_data = []
    all_label = []
    all_voxel_num = []
    all_voxel_sequence = []

    f = h5py.File(path + 'nyu2_1449_res0.2_sphere200_voxel_728_seq_%s'%(mode), 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    voxel_radius = f['voxel_radius'][:].astype('float32')
    f.close()

    return data, label, voxel_num, voxel_sequence, voxel_radius


def load_voxel_pcd_vfh():
    path = '/data4/zb/fxm_dgcnn_data/nyu2_1449_seq/res0.2_sphere200/'
    f = h5py.File(path + 'nyu2_1449_train_voxel_res0.2_sphere200_vfh_10cloud_per_cls', 'r')
    pcd = f['data'][:].astype('float32')
    label = np.loadtxt(path + 'kmeans_label512_nyu2_1449_train_voxelres0.2_sphere200_vfh_10cloud_per_cls', dtype= int)
    return pcd, label


#######################################suntry

def sun_try(mode):
    path = '/data4/zb/fxm_dgcnn_data/sun_rgbd/'

    all_data = []
    all_label = []
    all_voxel_num = []
    all_voxel_sequence = []
    #f = h5py.File(path + '"/data4/zb/fxm_dgcnn_data/sun_rgbd/sun_9cls_voxel_748_seq_train"%s'%(mode), 'r')
    #f = h5py.File(path + 'sun_9cls_voxel_748_seq_%s' % (mode), 'r')
    #f = h5py.File(path + 'newsun_%s' % (mode), 'r')
    #f = h5py.File(path + 'newsun_6feature_%s' % (mode), 'r')
    f = h5py.File(path + 'SUN_data_%s' % (mode), 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')###astype修改数据类型
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    f.close()

    return data, label, voxel_num, voxel_sequence

class sun_class_try(Dataset):
    """
    data 数组，结构是[b,v,n,3]　v= 169, n= 200
    label　数组，结构是[l1,l2,...,ln]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_sequence = sun_try(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        cloud_len_list = self.voxel_num[item]
        voxel_sequence = self.voxel_sequence[item]
        return pointcloud, label, cloud_len_list, voxel_sequence

    def __len__(self):
        #print(len(self.data))
        return len(self.data)


#######################################suntry




##########################################3dikea
#######################################suntry

def dikea_try(mode):
    path = '/data4/zb/fxm_dgcnn_data/3D_IKEA/'

    all_data = []
    all_label = []
    all_voxel_num = []
    all_voxel_sequence = []
    #f = h5py.File(path + "3DIKEA_6cls_voxel_356_seq_%s"%(mode), 'r')
    #f = h5py.File(path + 'sun_9cls_voxel_748_seq_%s' % (mode), 'r')
    #f = h5py.File(path + 'newsun_%s' % (mode), 'r')
    f = h5py.File(path + 'newsun_6feature_%s' % (mode), 'r')
    #f = h5py.File(path + 'SUN_data_%s' % (mode), 'r')
    data = f['data'][:].astype('float32')
    data=data[:,:,:,3:6]
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')###astype修改数据类型
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    f.close()

    return data, label, voxel_num, voxel_sequence

class dikea_class_try(Dataset):
    """
    data 数组，结构是[b,v,n,3]　v= 169, n= 200
    label　数组，结构是[l1,l2,...,ln]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_sequence = dikea_try(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        cloud_len_list = self.voxel_num[item]
        voxel_sequence = self.voxel_sequence[item]
        return pointcloud, label, cloud_len_list, voxel_sequence

    def __len__(self):
        #print(len(self.data))
        return len(self.data)


#######################################suntry











class nyu2_18cls_voxel_dsp220(Dataset):
    """
    data 数组，结构是[b,v,n,3]　v= 169, n= 200
    label　数组，结构是[l1,l2,...,ln]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_sequence = load_data(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        cloud_len_list = self.voxel_num[item]
        voxel_sequence = self.voxel_sequence[item]
        return pointcloud, label, cloud_len_list, voxel_sequence

    def __len__(self):
        #print(len(self.data))
        return len(self.data)

class nyu2_1449_6cls_voxel_dsp200(Dataset):
    """
    data size: [batch, 789, 200, 3]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_sequence = load_data_1449(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        cloud_len_list = self.voxel_num[item]
        voxel_sequence = self.voxel_sequence[item]
        return pointcloud, label, cloud_len_list, voxel_sequence

    def __len__(self):
        #print(len(self.data))
        return len(self.data)


class nyu2_1449_sphere(Dataset):
    """
    data size: [batch, 789, 200, 3]
    """
    def __init__(self, partition):
        self.data, self.label, self.voxel_num, self.voxel_sequence, self.voxel_radius= load_data_1449_sphere(partition)
        self.partition = partition

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        cloud_len_list = self.voxel_num[item]
        voxel_sequence = self.voxel_sequence[item]
        voxel_radius = self.voxel_radius[item]
        return pointcloud, label, cloud_len_list, voxel_sequence, voxel_radius

    def __len__(self):
        #print(len(self.data))
        return len(self.data)

class nyu2_voxel_pcd_vfh_cluster_label(Dataset):
    def __init__(self):
        self.data, self.label = load_voxel_pcd_vfh()

    def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        #print(len(self.data))
        return len(self.data)


# def load_nyu2_reshape_voxel_domnsmp(mode):
#     path = './'
#     f = h5py.File('%s%s.h5'%(path, mode), 'r')
#     data = f['data'][:].astype('float32')
#     label = f['label'][:].astype('int64')
#     f.close()
#     return data, label

# class nyu2_reshape_voxel_downsmp(Dataset):
#     """
#     data 数组，结构是[b,v,n,3]　v= 1883, n= 200
#     label　数组，结构是[l1,l2,...,ln]
#     """
#     def __init__(self, partition):
#         self.data, self.label = load_nyu2_reshape_voxel_domnsmp(partition)
#         self.partition = partition
#
#     def __getitem__(self, item):   #fxm: 表示取第几个样例，item means index of sample
#         pointcloud = self.data[item]
#         label = self.label[item]
#         return pointcloud, label
#
#     def __len__(self):
#         #print(len(self.data))
#         return len(self.data)

if __name__ == '__main__':
    #train = modelnet40_voxel('train')
    # test = modelnet40_voxel('train')
    # count = 0
    # for data, label in test:
    #     count += 1
    # print(count)

    # test_loader = DataLoader(nyu2_18cls_voxel_dsp220('train'), num_workers=8,
    #                           batch_size=4, shuffle=False)
    #
    # for i, (data, label, cloud_len_list, voxel_sequence) in enumerate(test_loader):
    #     """
    #     label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
    #     因此label的长度是１,也没有shape函数，本质上label是一个列表
    #     """
    #     if i == 0:
    #         print(data.shape, label.shape, cloud_len_list.shape, voxel_sequence.shape)
    #         print(label)
    #         print(cloud_len_list)
    #         print(voxel_sequence)
    #         break
    #     break

    test_loader = DataLoader(nyu2_1449_6cls_voxel_dsp200('test'), num_workers=8,
                              batch_size=16, shuffle=True)

    for i, (data, label,_, _) in enumerate(test_loader):
        """
        label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
        因此label的长度是１,也没有shape函数，本质上label是一个列表
        """
        if i == 0:
            print(data.shape, label.shape)
            print(label)

            break
        break

