import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from data_voxel_reshape import nyu2_18cls_voxel_dsp220, nyu2_1449_6cls_voxel_dsp200
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import torch.nn.functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def knn_nozero(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  ## size [B , N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  ### 　fxm: [B,C,N]尺寸的数据，按C相加，得到[B,1,N]的数据
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # 由于后一步排序从小到大所以这里都乘-1  size [B , N, N]

    # idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)  ## fxm:　tensor.topk会对最后一维排序,
    # 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　输出[最大k个值,对应的检索序号]，[1]表示输出索引
    #### blow for abandon [0,0,...,0]padding influence for knn neighbor seaching
    b_s, dim, num = x.shape
    xx_one = torch.ones(b_s, 1, num).cuda()
    xx_large = xx_one * 10000000
    xx_zero_sign = torch.where(xx> 0, xx_one, xx_large)
    pairwise_distance = pairwise_distance* xx_zero_sign
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    ####


    return idx

def get_graph_feature_nozero(x, k=10, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_nozero(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # [b,1,1]
    # idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points   # fxm: cpu测试版

    idx = idx + idx_base  # [b,n,k]

    idx = idx.view(-1)  # fxm 这里的idx是一个向量长度b*n*k,每个b中每个点最近邻的k个点的编号索引

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # [b*n*k,c]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  ## fxm: 将每个点重复k次，x的维度是[b,n,k,d]
    #                                                                           ,和feature一样
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()  # b之后按xyz...分(这里维度乘２因为拼接了原值与差值)，再按n分，最后是k

    return feature

def positional_adding(x, voxel_sequence, cloud_len_list, d_model, max_len=748):     ###################################################
    """
    function: add the voxel sequence values to every elements of voxel features.
    Note: d_model(voxel_dim) must be a even number  ; x size:[b, seq_len, voxel_dim] is same with output new_x
    """
    batch_size = x.shape[0]
    #new_x = torch.zeros(batch_size, x.shape[1], x.shape[2]).cuda()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).cuda()
    #print(div_term)
    for i in range(batch_size):
        pe = torch.zeros(max_len, d_model).cuda()
        position = voxel_sequence[i].unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe[:, cloud_len_list[i]:] = 0
        x[i, 1:, :] = x[i, 1:, :] + pe
    new_x = x
    return new_x

def get_attn_pad_mask(seq_q, voxel_sequence, cloud_len_list):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    voxel_sequence size: [batch, seq_len]
    '''
    # batch_size, len_q = seq_q.size()
    # batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    batch_size = voxel_sequence.shape[0]
    voxel_sequence_plus1 = torch.ones(batch_size, voxel_sequence.shape[1]+1).cuda()
    voxel_sequence_plus1[:, 1:] = voxel_sequence

    # voxel_sequence_plus1 = voxel_sequence_plus1.unsqueeze(1).expand(batch_size, len_q, len_k).clone()
    # for i in range(batch_size):
    #     voxel_sequence_plus1[i, cloud_len_list[i]:, :] = 0
    # pad_attn_mask = voxel_sequence_plus1.data.eq(0)

    pad_attn_mask = voxel_sequence_plus1.data.eq(0).unsqueeze(1)   # [batch_size, 1, len_k], True is masked
    # print(pad_attn_mask)
    # print(pad_attn_mask.shape)
    pad_attn_mask = pad_attn_mask.expand(batch_size, seq_q, seq_q).clone()
    #print(pad_attn_mask)
    for i in range(batch_size):
        pad_attn_mask[i, cloud_len_list[i]+1:, :] = True
    return pad_attn_mask  # [batch_size, len_q, len_k]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):   ### dim 就是每个word的长度
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, voxel_sequence=0, cloud_len_list=0):
        qkv = self.to_qkv(x).chunk(3, dim = -1)                                                        #这里有个9参数
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   ## dots size: [b, head=16, seq=760, seq=760]

        ################# 对 dots 这个 attention加一个mask处理
        #mask_map = get_attn_pad_mask(759, voxel_sequence, cloud_len_list)
        #################
        attn = self.attend(dots)   ####  这里没有mask操作。直接将attn和V相乘了

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, voxel_sequence, cloud_len_list):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MyNetwork(nn.Module):
    def __init__(self,N_dim,C_dim):
        super(MyNetwork, self).__init__()
        # self.fc_n=nn.Sequential(nn.Linear(N_dim, N_dim),
        #                        nn.ReLU(),
        #                        nn.Linear(N_dim,N_dim)
        # )
        # self.fc_c=nn.Sequential(nn.Linear(C_dim, C_dim),
        #                        nn.ReLU(),
        #                        nn.Linear(C_dim,C_dim)
        # )
        self.fc1 = nn.Linear(N_dim, N_dim)   # 隐藏层1
        self.fc2 = nn.Linear(N_dim, N_dim)    # 隐藏层2
        self.fc3 = nn.Linear(C_dim, C_dim)    # 隐藏层3
        self.fc4 = nn.Linear(C_dim, C_dim)    # 输出层

        # 使用Xavier初始化来初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x_n=torch.max(x,axis=-1)[0]
        x_c=torch.max(x,axis=-2)[0]
        x_n = torch.relu(self.fc1(x_n))
        x_n = self.fc2(x_n)
        x_c = torch.relu(self.fc3(x_c))
        x_c = self.fc4(x_c)
        x_n=x_n.reshape([x_n.shape[0],x_n.shape[1],1])
        x_c=x_c.reshape([x_c.shape[0],1,x_c.shape[1]])
        return x_c,x_n


def input_remake(input):
    x = torch.as_tensor(x, dtype=torch.float32)
    net = MyNetwork(x.shape[-2], x.shape[-1])
    # x=x.reshape([x.shape[0],x.shape[1],x.shape[2]*x.shape[3]])
    # x_s = np.max(x,axis=-1)
    # x_s=x_s.reshape([x_s.shape[0],x_s.shape[1],1])
    # x_t =np.max(x,axis=-2)
    # x_t=x_t.reshape([x_t.shape[0],1,x_t.shape[1]])
    x_s, x_t = net(x)
    print(x_s.shape)
    print(x_t.shape)
    x_s = x_s.repeat(1, input.shape[-2], 1)
    x_t = x_t.repeat(1, 1, input.shape[-1])
    print(x_s.shape)
    print(x_t.shape)
    y = torch.zeros([x.shape[0], x.shape[1], x.shape[2]])
    for i in range(x.shape[0]):
        y[i, ::] = torch.multiply(x_t[i, :, :], x_s[i, :, :])
    print(y.shape)
    z = torch.mean(x, dim=-1).reshape([x.shape[0], x.shape[1], 1])
    print(z.shape)
    result = torch.cat([y, z], dim=-1)
    print(result.shape)
    return result






class voxel_dgcnn(nn.Module):
    def __init__(self, dim):
        super(voxel_dgcnn, self).__init__()
        self.k = 10  ## 指　voxel里最近邻点的个数

        self.point_num = 200 ## 指每个voxel中有多少点
        self.voxel_cls = dim  ## 可以理解成有多少类体素，比如墙面体素，地面体素，圆形体素等等. 即voxel_fea的dim
        #self.voxel_num = 748
        self.emb_dims = 1024

        # self.bn1 = nn.BatchNorm2d(32)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
                                   # self.bn
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
        self.bn7 = nn.BatchNorm1d(self.voxel_cls)
        self.dp2 = nn.Dropout(p=0.5)

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
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)

        return x

class DGCNN_voxel_reshape(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.voxel_cls = dim
        self.voxel_embedding = voxel_dgcnn(dim)
        #self.pos_embedding = nn.Parameter(torch.randn(1, 759 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   ##　对于分类元素是随机初始化，并参与训练的
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, input, cloud_len_list, voxel_sequence):
        ###### handling the input,from [b,1883,200,3] to [tt,200,3]
        len_cloud = cloud_len_list.shape[0]
        x = torch.zeros((sum(cloud_len_list), input.shape[-2], input.shape[-1])).cuda()                                  ######################################
        accout = 0
        for i in range(len_cloud):
            x[accout:accout + cloud_len_list[i], :, :] = input[i, :cloud_len_list[i], :, :]
            accout += cloud_len_list[i]
        del input
        ######
        # x = input.view(-1, self.point_num, 3)
        ########  由于每个体素所处的空间位置不同，其值差距大，有必要减去均值，相当于给所有体素平移到原点
        # mean_x = x.mean(dim=1)
        # x = x - mean_x.view(-1, 1, 3)
        ########
        x = self.voxel_embedding(x)
        voxel_fea = x
        #print(x.shape, 'voxel dgcnn shape')

        xx = torch.zeros(len_cloud, 356, self.voxel_cls).cuda()                                    # xx means voxel features
        start_num = 0
        for xx_i in range(len_cloud):
            xx[xx_i, :cloud_len_list[xx_i], :] = x[start_num: start_num + cloud_len_list[xx_i], :]
            start_num += cloud_len_list[xx_i]
        x = xx
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   ####  fxm: the extra cls element
        x = torch.cat((cls_tokens, x), dim=1)
        #x += self.pos_embedding[:, :(n + 1)]
        x = positional_adding(x, voxel_sequence, cloud_len_list, d_model=self.voxel_cls, max_len= 356)             ##########################
        x = self.dropout(x)

        x = self.transformer(x, voxel_sequence, cloud_len_list)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        MMM=self.classifier(x)
        return self.classifier(x), voxel_fea

if __name__ == '__main__':
    v = DGCNN_voxel_reshape(
        num_classes=6,
        dim=316,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dim_head=64,
        dropout=0.1,    ### for tranformer
        emb_dropout=0.1  ### for embedding
    )

    class arg():
        def __init__(self):
            self.lr = 0.001
            self.epochs = 100
            self.batch_size = 8
    arg = arg()

    model = v.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0,  1,2,3])
    ##############   加载预训练的voxel dgcnn
    # model_dict = model.state_dict()
    # pretrain_voxel_dgcnn = torch.load('./model/pretrain_for_voxel_dgcnn_by_vfh_cluster_label/model_pretrain_for_voxel380_nomean_198.t7')
    # pretrainning = {k: v for k, v in pretrain_voxel_dgcnn.items() if k in model_dict}
    # model_dict.update(pretrainning)
    # model.load_state_dict(model_dict)
    ##############


    test_loader = DataLoader(nyu2_1449_6cls_voxel_dsp200('train'), num_workers=8,
                             batch_size=arg.batch_size, shuffle=True)

    opt = optim.Adam([{'params': model.module.voxel_embedding.parameters(), 'lr': arg.lr*0.01, 'weight_decay': 1e-4},
                      {'params': model.module.classifier.parameters(), 'lr': arg.lr, 'weight_decay':1e-4} ])
    scheduler = CosineAnnealingLR(opt, arg.epochs, eta_min=arg.lr)

    for i, (data, label, cloud_len_list, voxel_seqence) in enumerate(test_loader):
        """
        label 是　tensor([17,  0, 16,  8, 28,  4, 12, 14,  1, 21,  9, 25, 22, 18, 36,  7])
        因此label的长度是１,也没有shape函数，本质上label是一个列表
        """
        print('%s batch: %s' % (i, data.shape))
        # print(label)
        data = torch.FloatTensor(data).cuda()
        cloud_len_list = torch.LongTensor(cloud_len_list).cuda()
        voxel_seqence = torch.LongTensor(voxel_seqence).cuda()
        out, vxoel_fea = model(data, cloud_len_list, voxel_seqence)
        print(out.shape)
        if i > 0:
            print(out.shape, vxoel_fea.shape)
            print('------------------------')
            break


    #print(str(v))