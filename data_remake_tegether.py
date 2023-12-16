import h5py
import numpy as np
import time


# f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_train", 'r')
# data_train = f['data'][:].astype('float32')
# label_train = f['label'][:].astype('int64')
# voxel_num_train = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
# voxel_sequence_train = f['voxel_sequence'][:].astype('int64')
# f.close()
#
# f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_test", 'r')
# data_test = f['data'][:].astype('float32')
# label_test = f['label'][:].astype('int64')
# voxel_num_test = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
# voxel_sequence_test = f['voxel_sequence'][:].astype('int64')
# f.close()
#
# data=np.concatenate((data_train,data_test),axis=0)
# label=np.concatenate((label_train,label_test),axis=0)
# voxel_num=np.concatenate((voxel_num_train,voxel_num_test),axis=0)
# voxel_sequence=np.concatenate((voxel_sequence_train,voxel_sequence_test),axis=0)
#
# f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq", "w")
# f.create_dataset("data", data=data)
# f.create_dataset("label", data=label)
# f.create_dataset("voxel_num", data=voxel_num)
# f.create_dataset("voxel_sequence", data=voxel_sequence)
# f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_0.2", 'r')
data = f['data'][:].astype('float32')
label= f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
voxel_sequence= f['voxel_sequence'][:].astype('int64')
voxel_point_number=f['voxel_point_number'][:].astype('int64')
f.close()
name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_02_train"
h5f = h5py.File(name_result, 'w')
h5f.create_dataset('data', data=data[0:6376])
h5f.create_dataset('label', data=label[0:6376])
h5f.create_dataset('voxel_num', data=voxel_num[0:6376])
h5f.create_dataset('voxel_sequence', data=voxel_sequence[0:6376])
h5f.create_dataset('voxel_point_number', data=voxel_point_number[0:6376])
h5f.close()

name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_02_test"
h5f = h5py.File(name_result, 'w')
h5f.create_dataset('data', data=data[6376:])
h5f.create_dataset('label', data=label[6376:])
h5f.create_dataset('voxel_num', data=voxel_num[6376:])
h5f.create_dataset('voxel_sequence', data=voxel_sequence[6376:])
h5f.create_dataset('voxel_point_number', data=voxel_point_number[6376:])
h5f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_0.4", 'r')
data = f['data'][:].astype('float32')
label= f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
voxel_sequence= f['voxel_sequence'][:].astype('int64')
voxel_point_number=f['voxel_point_number'][:].astype('int64')
f.close()
name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_04_train"
h5f = h5py.File(name_result, 'w')
h5f.create_dataset('data', data=data[0:6376])
h5f.create_dataset('label', data=label[0:6376])
h5f.create_dataset('voxel_num', data=voxel_num[0:6376])
h5f.create_dataset('voxel_sequence', data=voxel_sequence[0:6376])
h5f.create_dataset('voxel_point_number', data=voxel_point_number[0:6376])
h5f.close()

name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_04_test"
h5f = h5py.File(name_result, 'w')
h5f.create_dataset('data', data=data[6376:])
h5f.create_dataset('label', data=label[6376:])
h5f.create_dataset('voxel_num', data=voxel_num[6376:])
h5f.create_dataset('voxel_sequence', data=voxel_sequence[6376:])
h5f.create_dataset('voxel_point_number', data=voxel_point_number[6376:])
h5f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_0.8", 'r')
data = f['data'][:].astype('float32')
label= f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
voxel_sequence= f['voxel_sequence'][:].astype('int64')
voxel_point_number=f['voxel_point_number'][:].astype('int64')
f.close()
name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_08_train"
h5f = h5py.File(name_result, 'w')
h5f.create_dataset('data', data=data[0:6376])
h5f.create_dataset('label', data=label[0:6376])
h5f.create_dataset('voxel_num', data=voxel_num[0:6376])
h5f.create_dataset('voxel_sequence', data=voxel_sequence[0:6376])
h5f.create_dataset('voxel_point_number', data=voxel_point_number[0:6376])
h5f.close()

name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_08_test"
h5f = h5py.File(name_result, 'w')
h5f.create_dataset('data', data=data[6376:])
h5f.create_dataset('label', data=label[6376:])
h5f.create_dataset('voxel_num', data=voxel_num[6376:])
h5f.create_dataset('voxel_sequence', data=voxel_sequence[6376:])
h5f.create_dataset('voxel_point_number', data=voxel_point_number[6376:])
h5f.close()





f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_02_train", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_02_test", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_04_train", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_04_test", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_08_train", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()

f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_08_test", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
    print(f[key].shape)
f.close()

