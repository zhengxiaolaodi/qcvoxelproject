import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from data_voxel_reshape import nyu2_18cls_good_data_voxel_pcd_vfh_cluster_label
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import torch.nn.functional as F
from util import cal_loss, IOStream
# helpers

class pointnet(nn.Module):
    def __init__(self, output_channels=380):
        super(pointnet, self).__init__()
        self.emb_dims = 256     ##########   change the last layer dimension
        self.conv1 = nn.Conv1d(3, 32, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)
        self.linear1 = nn.Linear(self.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class PointNet_voxel_reshape(nn.Module):
    def __init__(self):
        super().__init__()

        self.voxel_embedding = pointnet()

    def forward(self, input):
        ###### handling the input,from [b,1883,200,3] to [tt,200,3]

        ######
        # x = input.view(-1, self.point_num, 3)
        ########  由于每个体素所处的空间位置不同，其值差距大，有必要减去均值，相当于给所有体素平移到原点
        # mean_x = input.mean(dim=1)
        # x = input - mean_x.view(-1, 1, 3)
        ########
        x = self.voxel_embedding(input)
        #print(x.shape, 'voxel dgcnn shape')

        return x

class voxel_dgcnn(nn.Module):
    def __init__(self):
        super(voxel_dgcnn, self).__init__()
        self.k = 10  ## 指　voxel里最近邻点的个数

        self.point_num = 220 ## 指每个voxel中有多少点
        self.voxel_cls = 380  ## 可以理解成有多少类体素，比如墙面体素，地面体素，圆形体素等等. 即voxel_fea的dim
        self.voxel_num = 759
        self.emb_dims = 1024

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                   # self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32 * 2, 32, kernel_size=1, bias=False),
                                   # self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32 * 2, 64, kernel_size=1, bias=False),
                                   # self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   # self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(512 * 2, 1024, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, self.voxel_cls)


    def forward(self, input):

        x = input.view(-1, self.point_num, 3)
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)  ## x维度[b,x,n]  b_s = 16 x 169
        x = get_graph_feature_nozero(x, k=self.k)  ## 维度是[b,2*3,n,k]
        x = self.conv1(x)  ## 维度[b,16,n,k]
        x1 = x.max(dim=-1, keepdim=False)[0]  ## 维度[b,16,n]

        x = get_graph_feature_nozero(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]  ##  [b,16,n]

        x = get_graph_feature_nozero(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]  ##  [b,32,n]

        x = get_graph_feature_nozero(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]  ##  [b,64,n]

        x = torch.cat((x1, x2, x3, x4), dim=1)  ##  [b,128,n]

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  ## x dim is (b,c,n), after pooling dim is (b,c,1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  ## after viewing dim is (b,c)
        ## 为了消除填充的０点的影响，平均池化不能要，最大值池化对０点不敏感可以使用
        x = torch.cat((x1, x2), 1)
        # print(x.shape)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = self.linear2(x)


        return x

class DGCNN_voxel_reshape(nn.Module):
    def __init__(self):
        super().__init__()

        self.voxel_embedding = voxel_dgcnn()

    def forward(self, input):
        ###### handling the input,from [b,1883,200,3] to [tt,200,3]

        ######
        # x = input.view(-1, self.point_num, 3)
        ########  由于每个体素所处的空间位置不同，其值差距大，有必要减去均值，相当于给所有体素平移到原点
        # mean_x = input.mean(dim=1)
        # x = input - mean_x.view(-1, 1, 3)
        ########
        x = self.voxel_embedding(input)
        #print(x.shape, 'voxel dgcnn shape')

        return x

if __name__ == '__main__':
    v = PointNet_voxel_reshape(

    )

    class arg():
        def __init__(self):
            self.lr = 0.001
            self.epochs = 100
            self.batch_size = 16
    arg = arg()

    model = v.cuda()
    print(str(model))
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    test_loader = DataLoader(nyu2_18cls_good_data_voxel_pcd_vfh_cluster_label(), num_workers=8,
                             batch_size=arg.batch_size, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, arg.epochs, eta_min=arg.lr)

    for i, (data, label) in enumerate(test_loader):
        """
        label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
        因此label的长度是１,也没有shape函数，本质上label是一个列表
        """
        print('%s batch: %s' % (i, data.shape))
        # print(label)
        data = torch.FloatTensor(data).cuda().permute(0, 2, 1)

        out = model(data)
        print(out.shape)
        if i > 0:
            print(out.shape)
            print('------------------------')
            break

        label = torch.LongTensor(label).cuda()
        loss = cal_loss(out, label)
        loss.backward()
        opt.step()
    #print(str(v))

