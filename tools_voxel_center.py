import h5py
import numpy as np
import time




def center_data(data,z):
    center=np.sum(data,axis=1)/z
    result=center-z
    return result

def center_make(x):
    x_center=np.mean(x,axis=-2)
    x_center=np.expand_dims(x_center,axis=-2)
    x_center.repeat(x.shape[-2],axis=-2)
    return x-x_center

path="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_02_train"
f = h5py.File(path, 'r')
data = f['data'][:].astype('float32')
label = f['label'][:].astype('int64')
voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
voxel_sequence = f['voxel_sequence'][:].astype('int64')
voxel_point_number=f['voxel_point_number'][:].astype('int64')
f.close()
result=center_make(data)
print(1)
# for i in range(data.shape[0]):
#     print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
#     for j in range(data.shape[1]):
#         y=data[i,j,:,:]
#         z=voxel_point_number[i,j]
#         result=center_data(y,z)
#
