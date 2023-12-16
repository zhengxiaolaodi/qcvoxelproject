import h5py
import numpy as np
import time

def voxel_point_make(path,name_result):
    f = h5py.File(path, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    f.close()

    voxel_point_number = np.ones([data.shape[0], data.shape[1]])
    vectorc_0 = np.array([0, 0, 0]).reshape([1, 3])

    for i in range(data.shape[0]):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        for j in range(data.shape[1]):
            m = 0
            for k in range(data.shape[2]):
                zzz = data[i, j, k, :].reshape([1, 3])
                if zzz[0, 0] != 0 and zzz[0, 1] != 0 and zzz[0, 2] != 0:
                    m = m + 1
            voxel_point_number[i, j] = m
    h5f = h5py.File(name_result, 'w')
    h5f.create_dataset('data', data=data)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('voxel_num', data=voxel_num)
    h5f.create_dataset('voxel_sequence', data=voxel_sequence)
    h5f.create_dataset('voxel_point_number', data=voxel_point_number)
    h5f.close()


# f = h5py.File("/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq_train", "r")
# for key in f.keys():
#     print(f[key].name)  #获得名称，相当于字典中的key
#     print(f[key].shape)



path="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq"
name_result="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_0.2"
voxel_point_make(path,name_result)


# path="D:\h5_dataset\\3D_IKEA\\3DIKEA_6cls_voxel_356_seq_train"
# name_result="D:\h5_dataset\\3D_IKEA\\3DIKEA_0.2"
# voxel_point_make(path,name_result)
# path="H:\dataset_h5\\3D_IKEA\\3DIKEA_6cls_voxel_356_seq_test"
# name_result='H:\dataset_h5_voxel_pointnumber\\3D_IKEA\\3DIKEA_test_0.2_voxel_number'
# voxel_point_make(path,name_result)
#
# path="H:\dataset_h5\sun_rgbd\\sun_9cls_1987_voxel_748_seq_train"
# name_result="H:\dataset_h5_voxel_pointnumber\sun_rgbd\\sun_9cls_1987_voxel_748_seq_train_0.2_point_number"
# voxel_point_make(path,name_result)
# path="H:\dataset_h5\sun_rgbd\\sun_9cls_1878_voxel_748_seq_test"
# name_result="H:\dataset_h5_voxel_pointnumber\sun_rgbd\\sun_9cls_1878_voxel_748_seq_test_0.2_point_number"
# voxel_point_make(path,name_result)
#
#
#
#
# for i in range(0, 5, 1):
#     path=path+'nyu2_sub_22cls_voxel759_dsp220_train_%s.h5'%(i)
#     name_result='nyu2_sub_22cls_voxel759_dsp220_train_%s_voxel_point_number_res0_2.h5'%(i)
#     voxel_point_make(path, name_result)
#
# path='I:\h5_dataset\\nyu2_sub_22cls_sequence\\nyu2_sub_22cls_voxel759_dsp220_test_0_21_res0_2.h5'
# name_result = 'I:\h5_dataset\\nyu2_sub_22cls_sequence\\nyu2_sub_22cls_voxel759_dsp220_test_0_21_voxel_point_number_res0_2.h5'
# voxel_point_make(path, name_result)