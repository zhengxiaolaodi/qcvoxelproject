import h5py
import numpy as np

# def create_h5py(x ,path,h5_name):
#     g = h5py.File(path, "r")
#     label=g["label"][:].astype('int64')
#     voxel_num=g["voxel_num"][:].astype('int64')
#     voxel_sequence=g["voxel_sequence"][:].astype('int64')
#     g.close()
#     f = h5py.File(h5_name, "w")
#     f.create_dataset("data", data=x)
#     f.create_dataset('label',data=label)
#     f.create_dataset('voxel_num',data=voxel_num)
#     f.create_dataset('voxel_sequence',data=voxel_sequence)
#     f.close
#
#
#
# def read_h5py(h5_name):
#     f = h5py.File(h5_name, "r")
#     for key in f.keys():
#         print(f[key].name)
#         print(f[key].shape)
#     x=f["data"][:].astype('float32')
#     f.close()
#     return x
#
# x=read_h5py("D:\BaiduNetdiskDownload\H5数据/data_result_test.h5")
#
#
#
# path=("D:\\fxm\sun_rgbd/sun_9cls_voxel_748_seq_train")
# h5_name="sun_train"
# create_h5py(x,path,h5_name)




#
# f = h5py.File("/data4/zb/fxm_dgcnn_data/sun_rgbd/sun_train", "r")
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].shape)
#     x = f["data"][:].astype('float32')
# print(1)

x1=np.random.rand(12,748,2)
x2=np.random.rand(12,748,2)
x3=np.random.rand(12,748,2)
x=[x1,x2,x3]
for i_res in range(len(x)):
    print(i_res)