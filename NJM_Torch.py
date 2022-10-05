#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   NJM_Torch.py
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/9/30 17:59   zxx      1.0         None
'''

"""
使用torch改写NJM算法
"""


# LN_LSTM部分参考 https://github.com/pytorch/pytorch/issues/11335




import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ngpu=1
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print('DEVICE: ', end='')
print(DEVICE)

class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias)

        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden=None):
        # self.check_forward_input(input)
        if hidden is None:
            hx = input.new_zeros(
                input.size(0),
                self.hidden_size,
                requires_grad=False)
            cx = input.new_zeros(
                input.size(0),
                self.hidden_size,
                requires_grad=False)
        else:
            hx, cx = hidden
        # self.check_forward_hidden(input, hx, '[0]')
        # self.check_forward_hidden(input, cx, '[1]')

        gates = self.ln_ih(F .linear(input, self.weight_ih, self.bias_ih)) \
            + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o = gates[:, :(3 * self.hidden_size)].sigmoid().chunk(3, 1)
        g = gates[:, (3 * self.hidden_size):].tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.ln_ho(cy).tanh()
        return hy, cy


class LayerNormLSTM(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=1,
            bias=True,
            bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, bias=bias)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, bias=bias)
                for layer in range(num_layers)
            ])

    def forward(self, input, hidden=None):
        seq_len, batch_size, hidden_size = input.size()  # supports TxNxH only
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(
                self.num_layers *
                num_directions,
                batch_size,
                self.hidden_size,
                requires_grad=False)
            cx = input.new_zeros(
                self.num_layers *
                num_directions,
                batch_size,
                self.hidden_size,
                requires_grad=False)
        else:
            hx, cx = hidden

        ht = [[None, ] * (self.num_layers * num_directions)] * seq_len
        ct = [[None, ] * (self.num_layers * num_directions)] * seq_len

        if self.bidirectional:
            xs = input
            for l, (layer0, layer1) in enumerate(
                    zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht[t][l0], ct[t][l0] = layer0(x0, (h0, c0))
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht[t][l1], ct[t][l1] = layer1(x1, (h1, c1))
                    h1, c1 = ht[t][l1], ct[t][l1]
                xs = [torch.cat((h[l0], h[l1]), dim=1) for h in ht]
            y = torch.stack(xs)
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])
        else:
            h, c = hx, cx
            for t, x in enumerate(input):
                for l, layer in enumerate(self.hidden0):
                    ht[t][l], ct[t][l] = layer(x, (h[l], c[l]))
                    x = ht[t][l]
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1] for h in ht])
            hy = torch.stack(ht[-1])
            cy = torch.stack(ct[-1])

        return y, (hy, cy)


class NJM(nn.Module):
    def __init__(
            self,
            user_id_N=4630,
            user_attr_M=26991,
            item_id_N=26991,
            item_attr_M=4630,
            embedding_size=10,
            batch_size=128,
            beta=0.01,
            alpha=0.1,
            train_T=11,
            data_name="epinions",
            epoch=20,
            device=DEVICE
    ):
        super(NJM, self).__init__()
        self.train_T = train_T
        self.embedding_size = embedding_size
        self.user_node_N = user_id_N + 1
        self.user_attr_M = user_attr_M + 1
        self.item_node_N = item_id_N + 1
        self.item_attr_M = item_attr_M + 1  # 这个特征维度跟用户数量一样，说明特征记录的是被谁互动过
        self.data_name = data_name
        self.batch_size = batch_size
        self.alphaU = beta
        self.alphaS = alpha
        self.epoch = epoch
        self.device = device

        self._build_model()

        if self.device.type == 'cuda':
            self.cuda()

    def _initialize_weights(self):
        # 论文中定义的一些变量

        weights = {
            # U: user latent preference embedding
            'user_latent': nn.Parameter(
                torch.randn(self.user_node_N, self.train_T, self.embedding_size), requires_grad=True),
            # P: user latent link embedding
            'node_proximity': nn.Parameter(
                torch.randn(self.user_node_N, self.train_T, self.embedding_size), requires_grad=True),
            # T: recommendation transformation factor matrix
            'transformation': nn.Parameter(
                torch.ones(self.user_node_N, self.embedding_size), requires_grad=True),
            # V: item latent attribute embedding
            'item_attr_embeddings': nn.Parameter(
                torch.randn(self.item_attr_M, self.embedding_size), requires_grad=True),
            # 计算偏好时平衡消费因子U和社交因子P，均值为0.5，标准差为0.5
            'consumption_balance': nn.Parameter(
                torch.randn(self.user_node_N) * 0.5 + 0.5, requires_grad=True),
            # 计算好友时平衡消费因子U和社交因子P，均值为0.5，标准差为0.5
            'link_balance': nn.Parameter(
                torch.randn(self.user_node_N) * 0.5 + 0.5, requires_grad=True),
        }
        # 必须要用nn.ParameterDict封装，不然参数没有在模型中注册，不会更新
        self.weights = nn.ParameterDict(weights)

        bias = {
            # objective 部分的公式，表示一些不改变的属性
            'user_static': nn.Parameter(
                torch.ones(self.user_node_N, self.embedding_size) * 0.1, requires_grad=True),
            'item_static': nn.Parameter(
                torch.ones(self.item_node_N, self.embedding_size) * 0.1, requires_grad=True),
            # 用于经过rnn编码的item表示
            'item_out_rating': nn.Parameter(
                torch.ones(self.item_node_N, self.embedding_size) * 0.1, requires_grad=True),
            # 用于右上角跟E相乘后
            'link_mlp_embeddings': nn.Parameter(
                torch.ones(self.user_node_N) * 0.1, requires_grad=True),

        }
        self.bias = nn.ParameterDict(bias)

    def _build_model(self):
        self._initialize_weights()

        # build item lstm
        self._build_item_lstm()
        # build affine after lstm
        # 仿射变换
        self.affine_after_lstm = nn.Linear(
            self.embedding_size, self.embedding_size, bias=False)

        # build rating mlp
        self.rating_mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        # build link mlp
        # 计算E: time-dependent user neighborhood embedding
        self.mlp_for_E = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        # 计算u和p拼接，模型右上角
        self.link_mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.ReLU()
        )

    def _build_item_lstm(self):
        # lstm
        self.item_lstm = LayerNormLSTM(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
        )

    def step(self, ipt, mode='train'):
        if self.device.type == 'cuda':
            for k, v in ipt.items():
                ipt[k] = v.cuda()

        if mode == 'train':
            output = {
                'rating_prediction': [],
                'link_prediction': [],
                'penalize_user_embedding': []
            }


            train_user_id = ipt['user_id_list'].long()
            train_item_id = ipt['spot_id_list'].long()
            train_item_id_list = ipt['spot_id_list_list'].long()
            train_rating_indicator = ipt['rating_indicator_list']
            train_item_attr = ipt['spot_attr_list']
            train_rating_label = ipt['rating_pre_list']
            train_predict_link_label = ipt['link_list']
            train_predict_weight = ipt['link_weight_list']
            # ini_social_vector = ini_social_vector
            # ini_homophily_vector = ini_homophily_vector

            # 在每一时刻的训练结束后，更新好友记录，从而为下一时刻的计算服务
            friend_record = None

            # u: 这个batch中用到的U表示，从U中查找
            user_latent_vector = self.weights['user_latent'][train_user_id]
            # p: 这个batch中用到的P表示，从P中查找
            user_latent_promixity = self.weights['node_proximity'][train_user_id]
            # 消费影响权重
            consumption_weigh = self.weights['consumption_balance'][train_user_id]
            # 社交影响权重
            link_weigh = self.weights['link_balance'][train_user_id]
            # 静态用户消费向量
            b_user = self.bias['user_static'][train_user_id]
            # 静态物品消费向量
            b_item = self.bias['item_static'][train_user_id]

            # 计算物品特征变化，物品的特征就是物品跟哪些人互动过
            # train_item_attr是一个train_T * userN的矩阵，每个值是打分（0-1之间）
            train_item_attr = torch.reshape(train_item_attr, (-1, self.item_attr_M))
            # 得到train_T * d
            item_attr_embed = torch.matmul(train_item_attr, self.weights['item_attr_embeddings'])
            # lstm + affine
            item_rnn_input = torch.reshape(item_attr_embed, (-1, self.train_T, self.embedding_size))
            item_output, item_final_states = self.item_lstm(item_rnn_input)
            item_affine_rating = self.affine_after_lstm(item_output) + self.bias['item_out_rating'][train_item_id_list]
            item_affine_rating = torch.reshape(item_affine_rating, (-1, self.train_T, self.embedding_size))

            for t in range(self.train_T):
                if t == 0:
                    # ------------------------------
                    # 模型的左上角
                    user_latent_self = user_latent_vector[:, t, :]
                    # u和q_hair拼接
                    ## 一开始社会影响为0
                    ini_social_vector = torch.zeros(self.batch_size, self.embedding_size, device=self.device)
                    embed_layer = torch.concat([user_latent_self, ini_social_vector], -1)
                    # 输出u_hat
                    user_output = self.rating_mlp(embed_layer)
                    # ------------------------------

                    # rating prediction
                    rating_prediction = user_output * item_affine_rating[:, t, :] + b_user * b_item
                    rating_prediction = torch.sum(rating_prediction, dim=-1)
                    output['rating_prediction'].append({
                        'rating_prediction': rating_prediction,
                        'train_rating_label_t': train_rating_label[:, t],
                        'train_rating_indicator_t': train_rating_indicator[:, t],
                    })

                    # link prediction
                    ## 一开始同质化影响ini_homophily_matrix为0
                    ini_homophily_matrix = torch.zeros(self.user_node_N, self.embedding_size, device=self.device)
                    ## user_n * 2d,这个是计算右上角的E, 非初始状态下堆叠U和P送入MLP计算
                    user_embedding_matrix = torch.concat(
                        [self.weights['node_proximity'][:, t, :], ini_homophily_matrix], -1)
                    link_embedding_matrix = self.mlp_for_E(user_embedding_matrix)
                    ## 计算右上角的h
                    ini_homophily_vector = torch.zeros(self.batch_size, self.embedding_size, device=self.device)
                    link_embed_layer = torch.concat([user_latent_promixity[:, t, :], ini_homophily_vector], -1)
                    link_embed_layer = self.link_mlp(link_embed_layer)
                    ## 计算输出
                    link_prediction = torch.matmul(link_embed_layer, link_embedding_matrix.t())\
                                      + self.bias['link_mlp_embeddings']
                    output['link_prediction'].append({
                        'link_prediction': link_prediction,
                        'train_predict_weight_t': train_predict_weight[:, t],
                        'train_predict_link_label_t': train_predict_link_label[:, t],

                    })

                    # bsz*userN, 更新好友情况
                    friend_record = train_predict_link_label[:, 0, :]
                else:
                    # 左上角部分
                    ## 公式1
                    ### T和U元素乘法
                    user_friend_latent_matrix = self.weights['transformation']\
                                                * self.weights['user_latent'][:, t - 1, :]
                    ### 计算trust score f
                    #### 当前用户的p乘以整体的p，bsz*userN
                    node_proximity = torch.matmul(
                        user_latent_promixity[:, t - 1, :],
                        self.weights['node_proximity'][:, t - 1, :].t())
                    #### 这里是先过sigmoid然后过邻接矩阵，论文里是反过来
                    trust_score = torch.sigmoid(node_proximity)
                    #### friend_record就是I
                    trust_score = friend_record * trust_score
                    #### 下面进行归一化操作，令行和为1（行不全为0）的情况下
                    all = torch.sum(trust_score, keepdim=True, dim=-1)
                    ##### 这里是防止分母为0
                    all_p = all + 1
                    all = torch.where(all == 0, all_p, all)
                    trust_score = trust_score / all
                    #### 执行f*Q，得到q_hair，维度是bsz*d
                    user_friend_latent_vector = torch.matmul(trust_score, user_friend_latent_matrix)

                    ## 过mlp
                    ### 首先经过权重平衡因子放缩
                    user_latent_self = consumption_weigh * (user_latent_vector[:, t - 1, :].t())
                    user_latent_self = user_latent_self.t()

                    user_friend_latent_vector = (1 - consumption_weigh) * (user_friend_latent_vector.t())
                    user_friend_latent_vector = user_friend_latent_vector.t()
                    ### 拼接u和q_hair放入线性层
                    embed_layer = torch.cat([user_latent_self, user_friend_latent_vector], -1)
                    user_output = self.rating_mlp(embed_layer)

                    ## rating prediction
                    rating_prediction = user_output * item_affine_rating[:, t, :] + \
                                        b_user * b_item
                    rating_prediction = torch.sum(rating_prediction, dim=-1)
                    rating_prediction = torch.sigmoid(rating_prediction)
                    output['rating_prediction'].append({
                        'rating_prediction': rating_prediction,
                        'train_rating_label_t': train_rating_label[:, t],
                        'train_rating_indicator_t': train_rating_indicator[:, t],
                    })

                    # 右上角 link precition
                    ## 这是p
                    node_proximity_by_weight = link_weigh * (user_latent_promixity[:, t - 1, :].t())
                    node_proximity_by_weight = node_proximity_by_weight.t()
                    ## 这是u
                    homo_effect = user_latent_vector[:, t - 1, :]
                    homo_effect_by_weight = (1 - link_weigh) * (homo_effect.t())
                    homo_effect_by_weight = homo_effect_by_weight.t()
                    ## 计算h
                    link_embed_layer = torch.concat([node_proximity_by_weight, homo_effect_by_weight], -1)
                    link_embed_layer = self.link_mlp(link_embed_layer)

                    ## 计算E
                    ### 这是P
                    user_node_matrix = self.weights['link_balance'] * \
                                       (self.weights['node_proximity'][:, t - 1, :].t())
                    user_node_matrix = user_node_matrix.t()
                    ### 这是U
                    user_latent_matrix = (1 - self.weights['link_balance']) * \
                                         (self.weights['user_latent'][:, t - 1, :]).t()
                    user_latent_matrix = user_latent_matrix.t()
                    ### 计算E
                    user_embedding_matrix = torch.concat([user_node_matrix, user_latent_matrix], -1)
                    user_embedding_matrix = self.mlp_for_E(user_embedding_matrix)

                    ## h乘E
                    link_prediction = torch.matmul(link_embed_layer, user_embedding_matrix.t()) + \
                                      self.bias['link_mlp_embeddings']
                    output['link_prediction'].append({
                        'link_prediction': link_prediction,
                        'train_predict_weight_t': train_predict_weight[:, t],
                        'train_predict_link_label_t': train_predict_link_label[:, t],

                    })

                    ## 公式11需要的损失
                    output['penalize_user_embedding'].append({
                        'user_output': user_output,
                        'user_latent_vector_t': user_latent_vector[:, t, :],
                        'user_latent_promixity_t-1': user_latent_promixity[:, t - 1, :],
                        'user_latent_promixity_t': user_latent_promixity[:, t, :],
                    })

                    # 更新friend record
                    friend_record = friend_record + train_predict_link_label[:, t, :]
            return output
        elif mode == 'test':
            pass
        else:
            raise ValueError("Wrong mode!!!")


def test_model():
    from Dataset_torch import Dataset4NJM
    from tqdm import tqdm
    dataset = Dataset4NJM()
    dataset.load_data()
    myloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=True)


    njm = NJM(batch_size=128)
    for idx, batch_data in enumerate(tqdm(myloader)):
        output = njm.step(batch_data)
        # print(output)
        # break

if __name__ == '__main__':
    test_model()