import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
from data_voxel_reshape import nyu2_18cls_voxel_dsp220
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

def positional_adding(x, voxel_sequence, cloud_len_list, d_model=760, max_len=759):
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
    # for i in range(batch_size):
    #     pad_attn_mask[i, cloud_len_list[i]+1:, :] = True
    return pad_attn_mask  # [batch_size, len_q, len_k]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):   ### dim 就是每个word的长度
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *args):     #####  *args : (voxel_sequence, cloud_len_list)
        return self.fn(self.norm(x), args[0], args[1])

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
    def forward(self, x, voxel_sequence, cloud_len_list):
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

    def forward(self, x, voxel_sequence, cloud_len_list):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   ## dots size: [b, head=16, seq=760, seq=760]

        ################# 对 dots 这个 attention加一个mask处理
        mask_map = get_attn_pad_mask(760, voxel_sequence, cloud_len_list)      #### 760 = cls + 759
        mask_multi_head = mask_map.unsqueeze(1).repeat(1, self.heads, 1, 1)
        #print(mask_multi_head.shape)
        dots = dots.masked_fill_(mask_multi_head, -1e9)

        #################
        attn = self.attend(dots)
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
            x = attn(x, voxel_sequence, cloud_len_list) + x
            x = ff(x, voxel_sequence, cloud_len_list) + x
        return x

class DGCNN_voxel_reshape(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.voxel_embedding = nn.Sequential(
            # 首先将图片划分成一个个patch, 在将patch中每一个像素拉成一维向量送入fc层，获得一个patch的描述符
            Rearrange('b n c -> b (n c)'),
            nn.Linear(660, 760),
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, 759 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   ##　对于分类元素是随机初始化，并参与训练的
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(760, 760),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.2),
            nn.Linear(760, num_classes)
        )

    def forward(self, input, cloud_len_list, voxel_sequence):
        ###### handling the input,from [b,1883,200,3] to [tt,200,3]
        len_cloud = cloud_len_list.shape[0]
        x = torch.zeros((sum(cloud_len_list), 220, 3)).cuda()
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

        xx = torch.zeros(len_cloud, 759, 760).cuda()  # xx means voxel features
        start_num = 0
        for xx_i in range(len_cloud):
            xx[xx_i, :cloud_len_list[xx_i], :] = x[start_num: start_num + cloud_len_list[xx_i], :]
            start_num += cloud_len_list[xx_i]
        x = xx

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   ####  fxm: the extra cls element
        x = torch.cat((cls_tokens, x), dim=1)
        #x += self.pos_embedding[:, :(n + 1)]
        x = positional_adding(x, voxel_sequence, cloud_len_list)
        x = self.dropout(x)

        x = self.transformer(x, voxel_sequence, cloud_len_list)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.classifier(x)

if __name__ == '__main__':
    v = DGCNN_voxel_reshape(
        num_classes=18,
        dim=760,
        depth=6,
        heads=16,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.1,    ### for tranformer
        emb_dropout=0.1  ### for embedding
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
    test_loader = DataLoader(nyu2_18cls_voxel_dsp220('test'), num_workers=8,
                             batch_size=arg.batch_size, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=1e-4)
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
        out = model(data, cloud_len_list, voxel_seqence)
        print(out.shape)
        if i > 0:
            print(out.shape)
            print('------------------------')
            break


    #print(str(v))