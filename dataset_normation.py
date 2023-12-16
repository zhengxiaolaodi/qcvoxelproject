import h5py
import numpy as np
import time

def normalization(value):
    """标准化
    公式：(原始值-均值)/方差
    :return 范围任意，标准化后数据均值为0，标准差为1
    """
    new_value = (value - value.mean()) / value.std()
    return new_value

def data_norma(path,path2,trainortest):
    f = h5py.File(path, "r")
    g=h5py.File(path2,"r")
    label = g['label'][:].astype('int64')
    voxel_num = g['voxel_num'][:].astype('int64')###astype修改数据类型
    voxel_sequence = g['voxel_sequence'][:].astype('int64')
    data = f['data'][:].astype('float32')
    datanew = np.nan_to_num(data)
    for i in range(datanew.shape[0]):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        for j in range(datanew.shape[1]):
            k = 0
            while datanew[i, j, k, 0] != 0 and k < (datanew.shape[2] - 1):
                k = k + 1
            if k != (datanew.shape[2] - 1):
                datanew[i, j, k:datanew.shape[2], :] = 0
            for m in range(3, datanew.shape[3]):
                datanew[i, j, :, m] = normalization(datanew[i, j, :, m])
    h5f = h5py.File('D:\data_'+trainortest+'_result_sunrgbd.h5', 'w')
    h5f.create_dataset('data', data=datanew)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('voxel_num', data=voxel_num)
    h5f.create_dataset('voxel_sequence', data=voxel_sequence)
    h5f.close()

path="D:\BaiduNetdiskDownload\H5数据\data_result_train.h5"
path2="D:\h5_dataset\sun_rgbd\sun_9cls_voxel_748_seq_train"

data_norma(path,path2,"train")

path="D:\BaiduNetdiskDownload\H5数据\data_result_test.h5"
path2="D:\h5_dataset\sun_rgbd\sun_9cls_voxel_748_seq_test"

data_norma(path,path2,"test")