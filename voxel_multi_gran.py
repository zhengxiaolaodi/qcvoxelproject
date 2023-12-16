import open3d as o3d
import h5py
import numpy as np
import time
def get_voxel(scene_pcd, res=0.2):
    scene = np.asarray(scene_pcd)
    max_coordinate = scene.max(axis=0)
    min_coordinate = scene.min(axis=0)
    #print(max_coordinate, min_coordinate)
    # df = pd.DataFrame(scene)
    # df.to_clipboard(index=False, header=False)
    xzy_index = (max_coordinate - min_coordinate)//res
    #print(xzy_index)
    #### 排除空体素的操作
    voxel_num = (int(xzy_index[0]) + 1) * (int(xzy_index[1]) + 1) * (int(xzy_index[2]) + 1)
    #print(voxel_num)
    draw_button = np.zeros(voxel_num)
    # voxel_sequence=np.zeros([voxel_num,1])
    # voxle_coods=np.zeros([voxel_num,4])
    for aa in range(scene.shape[0]):
        voxel_index = (scene[aa] - min_coordinate) // res
        point_voxel_index = voxel_index[0] + voxel_index[1] * (int(xzy_index[0])+1) + voxel_index[2] * (int(xzy_index[0])+1) * (int(xzy_index[1])+1)
        draw_button[int(point_voxel_index)] += 1
    num_k=0
    #### 获得体素的八个点, 编号从下倒上逆时针0123， 4567
    voxel_point = []
    voxel_sequence = []
    voxle_coods=[]
    voxel_line = [[0, 1], [1, 2], [2, 3], [3,0], [4, 5], [5, 6], [6, 7], [7,4], [0, 4], [1, 5], [2, 6], [3, 7]]
    for i in range(int(xzy_index[0])+1):
        for j in range(int(xzy_index[1])+1):
            for k in range(int(xzy_index[2])+1):
                p0 = min_coordinate + np.asarray([i, j, k]) * res
                p1 = p0 + np.asarray([1, 0, 0]) * res
                p2 = p0 + np.asarray([1, 1, 0]) * res
                p3 = p0 + np.asarray([0, 1, 0]) * res
                p4 = p0 + np.asarray([0, 0, 1]) * res
                p5 = p0 + np.asarray([1, 0, 1]) * res
                p6 = p0 + np.asarray([1, 1, 1]) * res
                p7 = p0 + np.asarray([0, 1, 1]) * res
                # m = (i * (int(xzy_index[1]) + 1) * (int(xzy_index[2]) + 1) + j * (int(xzy_index[2]) + 1) + k)
                # voxle_coods[m,:]=m,i,j,k
                if draw_button[int(i+ j*(int(xzy_index[0])+1) +k*( (int(xzy_index[0])+1) * (int(xzy_index[1])+1))) ] > 0:
                    voxel_point.append([p0, p1, p2, p3, p4, p5, p6, p7])
                    voxel_sequence.append(k+j*(int(xzy_index[2]+1))+i*(int(xzy_index[2]+1))*(int(xzy_index[1]+1)))
                    voxle_coods.append([(k+j*(int(xzy_index[2]+1))+i*(int(xzy_index[2]+1))*(int(xzy_index[1]+1))),i,j,k])
                    num_k=num_k+1
    voxel_point = np.asarray(voxel_point, dtype=np.float32)
    voxel_sequence= np.asarray(voxel_sequence, dtype=np.float32)
    voxle_coods= np.asarray(voxle_coods, dtype=np.float32)

    return voxel_point, voxel_line,voxel_sequence,voxle_coods





def data_remake(voxel_point_number,voxel_sequence,voxel_num_new,voxel_coods_all,a, b, data, res):
    data_result = np.zeros([data.shape[0], a, b, data.shape[2]])
    for i in range(data.shape[0]):
        voxel_point, voxel_line, voxel_sequence_new,voxle_coods = get_voxel(data[i], res=res)
        voxel_coods_all[i,0:(voxle_coods.shape[0])]=voxle_coods
        voxel_num_new[i]=voxel_point.shape[0]
        voxel_sequence[i,0:(voxel_sequence_new.shape[0])]=voxel_sequence_new.reshape(voxel_sequence_new.shape[0],1)
        pcd = o3d.geometry.PointCloud()  # 实例化一个pointcloud类
        pcd.points = o3d.utility.Vector3dVector(data[i])  # 给该类传入坐标数据，此时pcd.points已经是一个点云了
        #print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        for j in range(voxel_point.shape[0]):
            center = (sum(voxel_point[j, :, 0]) / 8, sum(voxel_point[j, :, 1]) / 8, sum(voxel_point[j, :, 2]) / 8)
            center = np.array(center).reshape([3, 1])
            size = np.full([3, 1], res)
            crop = o3d.geometry.AxisAlignedBoundingBox(center - 0.5 * size, center + 0.5 * size)
            box_points = pcd.crop(crop).points
            voxelpoints = np.asarray(box_points)
            data_result[i, j, 0:(voxelpoints.shape[0]), :] = voxelpoints
            voxel_point_number[i,j]=voxelpoints.shape[0]
    return data_result, voxel_num_new, voxel_sequence,voxel_coods_all,voxel_point_number





def voxel_remake(path, res,name):
    f = h5py.File(path, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], data.shape[3])
    #data=data[0:100]
    biggest_a = 1
    biggest_b = 1
    for i in range(data.shape[0]):
        print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        #print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        voxel_point, voxel_line , voxel_point_number1,voxle_coods= get_voxel(data[i], res=res)
        if voxel_point.shape[0] >= biggest_a :
            biggest_a = voxel_point.shape[0]
        pcd = o3d.geometry.PointCloud()  # 实例化一个pointcloud类
        pcd.points = o3d.utility.Vector3dVector(data[i])  # 给该类传入坐标数据，此时pcd.points已经是一个点云了
        #print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        for j in range(voxel_point.shape[0]):
            center = (sum(voxel_point[j, :, 0]) / 8, sum(voxel_point[j, :, 1]) / 8, sum(voxel_point[j, :, 2]) / 8)
            center = np.array(center).reshape([3, 1])
            size = np.full([3, 1], res)
            crop = o3d.geometry.AxisAlignedBoundingBox(center - 0.5 * size, center + 0.5 * size)
            box_points = pcd.crop(crop).points
            voxelpoints = np.asarray(box_points)
            if voxelpoints.shape[0] >= biggest_b :
                print(voxelpoints.shape[0])
                print(i)
                biggest_b = voxelpoints.shape[0]
    voxel_coods_all=np.zeros([data.shape[0],biggest_a,4])
    voxel_num=np.zeros([data.shape[0],1])
    voxel_sequence=np.zeros([data.shape[0],biggest_a,1])
    voxel_point_number=np.zeros([data.shape[0],biggest_a])
    data_result, voxel_num, voxel_sequence,voxel_coods,voxel_point_number = data_remake(voxel_point_number,voxel_sequence,voxel_num,voxel_coods_all,biggest_a, biggest_b, data, res)
    h5_make(voxel_point_number,data_result,label,voxel_num,voxel_sequence,voxel_coods,name)



def h5_make(voxel_point_number,data_result,label,voxel_num,voxel_sequence,voxel_coods,name):
    f = h5py.File(name, "w")
    f.create_dataset("data", data=data_result)
    f.create_dataset("label", data=label)
    f.create_dataset("voxel_num", data=voxel_num)
    f.create_dataset("voxel_point_number", data=voxel_point_number)
    f.create_dataset("voxel_sequence", data=voxel_sequence)
    f.create_dataset("voxel_coods", data=voxel_coods)
    f.close()
###############3DIKEA
res=0.4
path="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq"
voxel_remake(path,res,name="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_0.4")
res=0.8
path="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_6cls_voxel_356_seq"
voxel_remake(path,res,name="/data4/zb/fxm_dgcnn_data/3D_IKEA/3DIKEA_0.8")