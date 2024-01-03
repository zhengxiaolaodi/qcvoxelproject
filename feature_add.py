import torch
import open3d as o3d
import h5py
import time
import numpy as np
def knn_nozero(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  ## size [B , N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  ### 　fxm: [B,C,N]尺寸的数据，按C相加，得到[B,1,N]的数据
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 由于后一步排序从小到大所以这里都乘-1  size [B , N, N]

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)  ## fxm:　tensor.topk会对最后一维排序,
    # 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　输出[最大k个值,对应的检索序号]，[1]表示输出索引
    #### blow for abandon [0,0,...,0]padding influence for knn neighbor seaching
    b_s, dim, num = x.shape
    xx_one = torch.ones(b_s, 1, num).cpu()
    xx_large = xx_one * 10000000
    xx_zero_sign = torch.where(xx> 0, xx_one, xx_large)
    pairwise_distance = pairwise_distance* xx_zero_sign
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    ####
    return idx


def feature_make(x,y):
    d_ij = torch.sqrt((x[:,:, :, 0]-y[:,:,:,0]) ** 2 + (x[:,:, :, 1]) ** 2 + (x[:,:, :, 2]) ** 2)
    d_ij = d_ij.reshape(d_ij.shape[0], d_ij.shape[1], d_ij.shape[2],1)
    p_ij = x-y
    p_ij=torch.cat((p_ij,d_ij),dim=-1)
    a1_ij = torch.atan2((x[ :,:, :, 2]-y[:,:,:,2]) , (torch.sqrt((x[:,:, :, 0]-y[:,:,:,0])** 2 + (x[:, :, :, 1]-y[:,:,:,1]) ** 2)))
    a2_ij = torch.atan2((x[:, :,:, 1]-y[:,:,:,1]) , (x[:, :,:, 0]-y[:,:,:,0]))
    a1_ij=a1_ij.reshape(a1_ij.shape[0], a1_ij.shape[1], a1_ij.shape[2],1)
    a2_ij=a2_ij.reshape(a2_ij.shape[0], a2_ij.shape[1], a2_ij.shape[2],1)
    a_ij=torch.cat((a1_ij,a2_ij,d_ij),dim=-1)
    result=torch.cat((p_ij,a1_ij,a2_ij),dim=-1)
    return result

def normal_feature(x):
    a=x.shape[0]
    b=x.shape[1]
    c=x.shape[2]
    x=x.reshape(x.shape[0]*x.shape[1],x.shape[2])
    x=np.array(x)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(x)
    o3d.geometry.estimate_normals(pcd, search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normal_point = o3d.utility.Vector3dVector(pcd.normals)
    normal_point_array = np.array(normal_point)
    normal_point_array=normal_point_array.reshape([a,b,c])
    normal_point_array=torch.tensor(normal_point_array)
    return normal_point_array





# def distance_angle_feature_add(m, k=10, idx=None):
#     x=m
#     x=torch.tensor(x)
#     batch_size = x.size(0)
#     num_points = x.size(1)
#     x = x.view(batch_size, -1, num_points)#[2385,3,200]
#     if idx is None:
#         idx = knn_nozero(x, k=k)  # (batch_size, num_points, k)
#     device = torch.device('cpu')
#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # [b,1,1]
#     # idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points   # fxm: cpu测试版
#     idx = idx + idx_base  # [b,n,k]
#     idx = idx.view(-1)  # fxm 这里的idx是一个向量长度b*n*k,每个b中每个点最近邻的k个点的编号索引
#     _, num_dims, _ = x.size()
#     x = x.transpose(2,
#                     1).contiguous()
#     x_tran=np.array(x)# (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
#     feature = x.view(batch_size * num_points, -1)  # [b*n*k,c]
#     feature=feature[idx,:]
#     feature = feature.view(batch_size, num_points, k, num_dims)
#     x2=np.array(x)
#     x1 = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
#     x_arry=np.array(x1)
#     fea_array=np.array(feature)
#     dis_and_angle_feature=feature_make(x,feature).max(dim=-2, keepdim=False)[0].squeeze()
#     xxx=np.array(dis_and_angle_feature)
#     fature=normal_feature(m)
#     result=torch.cat((dis_and_angle_feature,fature),dim=-1)
#     return result

def distance_angle_feature_add(m, k=10, idx=None):
    x=m
    x=torch.tensor(x)
    batch_size = x.size(0)
    num_points = x.size(1)
    x = x.view(batch_size, -1, num_points)#[2385,3,200]
    if idx is None:
        idx = knn_nozero(x, k=k)# (batch_size, num_points, k)\
    device = torch.device('cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # [b,1,1]
    idx = idx + idx_base  # [b,n,k]
    idx = idx.view(-1)  # fxm 这里的idx是一个向量长度b*n*k,每个b中每个点最近邻的k个点的编号索引
    _, num_dims, _ = x.size()
    feature = x.view(batch_size * num_points, -1)  # [b*n*k,c]
    feature=feature[idx,:]
    feature = feature.view(batch_size, num_points, k, num_dims)
    feature = feature_remake(feature)
    feature = torch.tensor(feature)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    dis_and_angle_feature=feature_make(x,feature).max(dim=-2, keepdim=False)[0].squeeze()
    fature=normal_feature(m)
    result=torch.cat((dis_and_angle_feature,fature),dim=-1)
    # result=result_remake(result)
    return result

# def result_remake(result):


def feature_remake(a):
    result=np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b=0
            for k in range(a.shape[2]):
                if (a[i,j,k,0] != 0):
                    result[i,j,b]=a[i,j,k]
                    b=b+1
    return result


def h5_nake(data_name,name_result):
    f = h5py.File(data_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    voxel_num = f['voxel_num'][:].astype('int64')  ###astype修改数据类型
    voxel_sequence = f['voxel_sequence'][:].astype('int64')
    voxel_point_number = f['voxel_point_number'][:].astype('int64')
    f.close()
    y = np.random.rand(data.shape[0], data.shape[1], data.shape[2], 9)
    for i in range(data.shape[0]):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        feature = distance_angle_feature_add(data[i], k=20)
        y[i] = feature
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    h5f = h5py.File(name_result, 'w')
    h5f.create_dataset('y', data=data)
    h5f.create_dataset('label', data=label)
    h5f.create_dataset('voxel_num', data=voxel_num)
    h5f.create_dataset('voxel_sequence', data=voxel_sequence)
    h5f.create_dataset('voxel_point_number', data=voxel_point_number)
    h5f.close()




path="D:/multi_para/3Dikea/3DIKEA_6cls_voxel_356_seq_02_train"
name="D:/multi_para/3Dikea/3DIKEA_6cls_voxel_356_seq_02_9fea_train"
h5_nake(path,name)

path="D:/multi_para/3Dikea/3DIKEA_6cls_voxel_356_seq_04_train"
name="D:/multi_para/3Dikea/3DIKEA_6cls_voxel_356_seq_04_9fea_train"
h5_nake(path,name)

path="D:/multi_para/3Dikea/3DIKEA_6cls_voxel_356_seq_08_train"
name="D:/multi_para/3Dikea/3DIKEA_6cls_voxel_356_seq_08_9fea_train"
h5_nake(path,name)
print(1)