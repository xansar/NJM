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
from Dataset_torch import Dataset4NJM, MyDataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle as pkl



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class NJMLossFunction(nn.Module):
    def __init__(self, config):
        super(NJMLossFunction, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction='none')

        self.alphaS = eval(config['LOSS']['alphaS'])
        self.alphaU = eval(config['LOSS']['alphaU'])

    def forward(self, output):
        loss = 0.
        bsz = output['rating_prediction'][0]['rating_prediction'].shape[0]
        for pred in output['rating_prediction']:
            # rating loss rating_pre_list
            Lu = torch.square(pred['rating_prediction'] - pred['rating_pre_list']) * \
                 pred['rating_indicator']
            loss += torch.sum(Lu)
        if 'link_prediction' in output.keys():
            for pred in output['link_prediction']:
                # prediction loss
                Ls = self.bceloss(pred['link_prediction'], pred['train_predict_link_label_t'].float())
                loss += self.alphaS * torch.sum(pred['train_predict_weight_t'] * Ls)
        if 'penalize_user_embedding' in output.keys():
            for pred in output['penalize_user_embedding']:
                # embedding loss over time
                u_part = torch.sum(torch.square(pred['user_output'] - pred['user_latent_vector_t']))
                p_part = torch.sum(torch.square(pred['user_latent_promixity_t-1'] - pred['user_latent_promixity_t']))
                Lr = self.alphaU * (u_part + p_part)
                loss += Lr

        return loss / bsz

# 参考https://github.com/chenhuaizhen/LayerNorm_LSTM/blob/master/ln-wd-vd-lstm.py
class LayerNormLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, dropout=0.0, bias=True, use_layer_norm=True):
        super().__init__(input_size, hidden_size, bias)
        self.use_layer_norm = use_layer_norm
        if self.use_layer_norm:
            self.ln_ih = nn.LayerNorm(4 * hidden_size)
            self.ln_hh = nn.LayerNorm(4 * hidden_size)
            self.ln_ho = nn.LayerNorm(hidden_size)
        # DropConnect on the recurrent hidden to hidden weight
        self.dropout = dropout

    def forward(self, input, hidden=None):
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        weight_hh = nn.functional.dropout(self.weight_hh, p=self.dropout, training=self.training)
        if self.use_layer_norm:
            gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) \
                     + self.ln_hh(F.linear(hx, weight_hh, self.bias_hh))
        else:
            gates = F.linear(input, self.weight_ih, self.bias_ih) \
                    + F.linear(hx, weight_hh, self.bias_hh)

        i, f, c, o = gates.chunk(4, 1)
        i_ = torch.sigmoid(i)
        f_ = torch.sigmoid(f)
        c_ = torch.tanh(c)
        o_ = torch.sigmoid(o)
        cy = (f_ * cx) + (i_ * c_)
        if self.use_layer_norm:
            hy = o_ * self.ln_ho(torch.tanh(cy))
        else:
            hy = o_ * torch.tanh(cy)
        return hy, cy


class LayerNormLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 weight_dropout=0.0,
                 bias=True,
                 bidirectional=False,
                 use_layer_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # using variational dropout
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1
        self.hidden0 = nn.ModuleList([
            LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                              hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
            for layer in range(num_layers)
        ])

        if self.bidirectional:
            self.hidden1 = nn.ModuleList([
                LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size * num_directions),
                                  hidden_size=hidden_size, dropout=weight_dropout, bias=bias, use_layer_norm=use_layer_norm)
                for layer in range(num_layers)
            ])

    def copy_parameters(self, rnn_old):
        for param in rnn_old.named_parameters():
            name_ = param[0].split("_")
            layer = int(name_[2].replace("l", ""))
            sub_name = "_".join(name_[:2])
            if len(name_) > 3:
                self.hidden1[layer].register_parameter(sub_name, param[1])
            else:
                self.hidden0[layer].register_parameter(sub_name, param[1])

    def forward(self, input, hidden=None, seq_lens=None):
        batch_size, seq_len, _ = input.size()
        seq_lens = [seq_len] * batch_size
        num_directions = 2 if self.bidirectional else 1
        if hidden is None:
            hx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers * num_directions, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht = []
        for i in range(seq_len):
            ht.append([None] * (self.num_layers * num_directions))
        ct = []
        for i in range(seq_len):
            ct.append([None] * (self.num_layers * num_directions))

        seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_size, requires_grad=False)
        if seq_lens != None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1)

        if self.bidirectional:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_ = (torch.cuda.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices_reverse = torch.cuda.LongTensor([0] * batch_size).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, 1, 1, self.hidden_size])
            indices = torch.cat((indices_, indices_reverse), dim=1)
            hy = []
            cy = []
            xs = input
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, 2, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, 2, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for l, (layer0, layer1) in enumerate(zip(self.hidden0, self.hidden1)):
                l0, l1 = 2 * l, 2 * l + 1
                h0, c0, h1, c1 = hx[l0], cx[l0], hx[l1], cx[l1]
                for t, (x0, x1) in enumerate(zip(xs, reversed(xs))):
                    ht_, ct_ = layer0(x0, (h0, c0))
                    ht[t][l0] = ht_ * seq_len_mask[t]
                    ct[t][l0] = ct_ * seq_len_mask[t]
                    h0, c0 = ht[t][l0], ct[t][l0]
                    t = seq_len - 1 - t
                    ht_, ct_ = layer1(x1, (h1, c1))
                    ht[t][l1] = ht_ * seq_len_mask[t]
                    ct[t][l1] = ct_ * seq_len_mask[t]
                    h1, c1 = ht[t][l1], ct[t][l1]

                xs = [torch.cat((h[l0]*dropout_mask[l][0], h[l1]*dropout_mask[l][1]), dim=1) for h in ht]
                ht_temp = torch.stack([torch.stack([h[l0], h[l1]]) for h in ht])
                ct_temp = torch.stack([torch.stack([c[l0], c[l1]]) for c in ct])
                if len(hy) == 0:
                    hy = torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    hy = torch.cat((hy, torch.stack(list(ht_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
                if len(cy) == 0:
                    cy = torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))
                else:
                    cy = torch.cat((cy, torch.stack(list(ct_temp.gather(dim=0, index=indices).squeeze(0)))), dim=0)
            y  = torch.stack(xs)
        else:
            # if use cuda, change 'torch.LongTensor' to 'torch.cuda.LongTensor'
            indices = (torch.cuda.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
                [1, self.num_layers, 1, self.hidden_size])
            h, c = hx, cx
            # Variational Dropout
            if not self.training or self.dropout == 0:
                dropout_mask = input.new_ones(self.num_layers, batch_size, self.hidden_size)
            else:
                dropout_mask = input.new(self.num_layers, batch_size, self.hidden_size).bernoulli_(1 - self.dropout)
                dropout_mask = Variable(dropout_mask, requires_grad=False) / (1 - self.dropout)

            for t, x in enumerate(input.transpose(0, 1)):
                for l, layer in enumerate(self.hidden0):
                    ht_, ct_ = layer(x, (h[l], c[l]))
                    ht[t][l] = ht_ * seq_len_mask[t]
                    ct[t][l] = ct_ * seq_len_mask[t]
                    x = ht[t][l] * dropout_mask[l]
                ht[t] = torch.stack(ht[t])
                ct[t] = torch.stack(ct[t])
                h, c = ht[t], ct[t]
            y = torch.stack([h[-1]*dropout_mask[-1] for h in ht])
            hy = torch.stack(list(torch.stack(ht).gather(dim=0, index=indices).squeeze(0)))
            cy = torch.stack(list(torch.stack(ct).gather(dim=0, index=indices).squeeze(0)))

        return y.transpose(0, 1), (hy, cy)

class NJM(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(NJM, self).__init__()
        self.train_T = eval(config['MODEL']['train_T'])
        self.embedding_size = eval(config['MODEL']['embedding_size'])
        self.user_node_N = eval(config['MODEL']['user_id_N']) + 1
        self.user_attr_M = eval(config['MODEL']['user_attr_M']) + 1
        self.item_node_N = eval(config['MODEL']['item_id_N']) + 1
        self.item_attr_M = eval(config['MODEL']['item_attr_M']) + 1  # 这个特征维度跟用户数量一样，说明特征记录的是被谁互动过
        self.data_name = config['DATA']['data_name']
        self.batch_size = eval(config['DATA']['train_batch_size'])

        self.epoch = eval(config['TRAIN']['epoch'])
        if torch.cuda.is_available() and config['TRAIN']['device'] == 'cuda':
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

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
            # affine after lstm
            'item_rnn_out_rating': nn.Parameter(
                torch.randn(self.embedding_size, self.embedding_size), requires_grad=True),
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

        # build rating mlp
        self.rating_mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_size, self.embedding_size),
            nn.ReLU()
        )
        # build link mlp
        # 计算E: time-dependent user neighborhood embedding
        # 计算u和p拼接，模型右上角
        # 这两项都用link_mlp
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
        # self.item_lstm = nn.LSTM(
        #     input_size=self.embedding_size,
        #     hidden_size=self.embedding_size,
        # )

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
            item_affine_rating = torch.matmul(item_output, self.weights['item_rnn_out_rating']) + \
                                 self.bias['item_out_rating'][train_item_id_list]
            item_affine_rating = torch.reshape(item_affine_rating, (-1, self.train_T, self.embedding_size))

            for t in range(self.train_T):
                if t == 0:
                    # ------------------------------
                    # 模型的左上角
                    user_latent_self = user_latent_vector[:, t, :]
                    # u和q_hair拼接
                    ## 一开始社会影响为0
                    ini_social_vector = torch.zeros(self.batch_size, self.embedding_size, device=self.device)
                    embed_layer = torch.cat([user_latent_self, ini_social_vector], -1)
                    # 输出u_hat
                    user_output = self.rating_mlp(embed_layer)
                    # ------------------------------

                    # rating prediction
                    rating_prediction = user_output * item_affine_rating[:, t, :] + b_user * b_item
                    rating_prediction = torch.sum(rating_prediction, dim=-1)
                    rating_prediction = torch.sigmoid(rating_prediction)
                    output['rating_prediction'].append({
                        'rating_prediction': rating_prediction,
                        'rating_pre_list': train_rating_label[:, t],
                        'rating_indicator': train_rating_indicator[:, t],
                    })

                    # link prediction
                    ## 一开始同质化影响ini_homophily_matrix为0
                    ini_homophily_matrix = torch.zeros(self.user_node_N, self.embedding_size, device=self.device)
                    ## user_n * 2d,这个是计算右上角的E, 非初始状态下堆叠U和P送入MLP计算
                    user_embedding_matrix = torch.cat(
                        [self.weights['node_proximity'][:, t, :], ini_homophily_matrix], -1)
                    link_embedding_matrix = self.link_mlp(user_embedding_matrix)
                    ## 计算右上角的h
                    ini_homophily_vector = torch.zeros(self.batch_size, self.embedding_size, device=self.device)
                    link_embed_layer = torch.cat([user_latent_promixity[:, t, :], ini_homophily_vector], -1)
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
                        'rating_pre_list': train_rating_label[:, t],
                        'rating_indicator': train_rating_indicator[:, t],
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
                    link_embed_layer = torch.cat([node_proximity_by_weight, homo_effect_by_weight], -1)
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
                    user_embedding_matrix = torch.cat([user_node_matrix, user_latent_matrix], -1)
                    user_embedding_matrix = self.link_mlp(user_embedding_matrix)

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
        elif mode == 'evaluate':
            output = {
                'rating_prediction': [],
            }

            test_user_id = ipt['user_id_list'].long()
            test_item_id = ipt['spot_id_list'].long()
            test_item_id_list = ipt['spot_id_list_list']
            test_item_attr = ipt['spot_attr_list']
            test_rating_label = ipt['rating_pre_list']
            test_friend_record = ipt['friend_record_list']

            # item lstm
            test_item_attr = torch.reshape(test_item_attr, (-1, self.item_attr_M))
            test_item_attr_embed = torch.matmul(test_item_attr, self.weights['item_attr_embeddings'])

            test_item_rnn_input = torch.reshape(test_item_attr_embed, (-1, self.train_T + 1, self.embedding_size))
            test_item_output, test_item_final_states = self.item_lstm(test_item_rnn_input)
            test_item_output = test_item_output[:, -1, :]
            test_item_affine_rating = torch.matmul(test_item_output, self.weights['item_rnn_out_rating']) + \
                                      self.bias['item_out_rating'][test_item_id]

            # 左上角
            test_user_friend_latent_matrix = self.weights['user_latent'][:, -1, :] * \
                                             self.weights['transformation']
            ## u
            test_user_latent_vector = self.weights['user_latent'][test_user_id]
            test_user_latent_vector = test_user_latent_vector[:, -1, :]
            ## p
            test_user_latent_promixity = self.weights['node_proximity'][test_user_id]
            ## 计算trust　score
            test_node_proximity = torch.matmul(test_user_latent_promixity[:, -1, :],
                                            self.weights['node_proximity'][:, -1, :].t(),
                                            )
            test_trust_score = torch.sigmoid(test_node_proximity)
            test_trust_score = test_trust_score * test_friend_record

            test_all = torch.sum(test_trust_score, keepdim=True, dim=-1)
            test_all_p = test_all + 1
            test_all = torch.where(test_all == 0, test_all_p, test_all)
            test_trust_score = test_trust_score / test_all
            ## 计算q_hair
            test_user_friend_latent_vector = torch.matmul(
                test_trust_score, test_user_friend_latent_matrix)
            test_consumption_weigh = self.weights['consumption_balance'][test_user_id]
            test_social_factor = test_consumption_weigh * test_user_friend_latent_vector.t()
            test_social_factor = test_social_factor.t()
            ## 过mlp
            test_embed_layer = torch.cat([test_user_latent_vector, test_social_factor], -1)
            test_user_output = self.rating_mlp(test_embed_layer)
            ## rating prediction
            test_b_user = self.bias['user_static'][test_user_id]
            test_b_item = self.bias['item_static'][test_item_id]
            test_rating_prediction = test_user_output * test_item_affine_rating + \
                                     test_b_user * test_b_item

            test_rating_prediction = torch.sum(test_rating_prediction, dim=-1)
            test_rating_prediction = torch.sigmoid(test_rating_prediction)
            output['rating_prediction'].append({
                'rating_prediction': test_rating_prediction,
                'rating_pre_list': test_rating_label,
                'rating_indicator': test_rating_label != 0,
            })

        elif mode == 'evaluate_link':
            # link prediction
            link_test_user_id = ipt['link_test_user_id']

            link_test_link_weigh = self.weights['link_balance'][link_test_user_id]
            ## p
            link_test_user_latent_promixity = self.weights['node_proximity'][link_test_user_id]
            link_test_node_proximity = torch.unsqueeze(link_test_user_latent_promixity[-1, :], 0)
            link_test_node_proximity_by_weight = link_test_link_weigh * link_test_node_proximity.t()
            link_test_node_proximity_by_weight = link_test_node_proximity_by_weight.t()
            ## u
            link_test_user_latent_vector = self.weights['user_latent'][link_test_user_id]
            link_test_homo_effect = torch.unsqueeze(link_test_user_latent_vector[-1, :], 0)
            link_test_homo_effect_by_weight = (1 - link_test_link_weigh) * (link_test_homo_effect.t())
            link_test_homo_effect_by_weight = link_test_homo_effect_by_weight.t()
            ## mlp
            link_test_embed_layer = torch.cat(
                [link_test_node_proximity_by_weight, link_test_homo_effect_by_weight], -1)
            link_test_embed_layer = self.link_mlp(link_test_embed_layer)
            ## E
            ### P
            test_user_node_matrix = self.weights['link_balance'] * (self.weights['node_proximity'][:, -1, :].t())
            test_user_node_matrix = test_user_node_matrix.t()
            ### U
            test_user_latent_matrix = (1 - self.weights['link_balance']) * (self.weights['user_latent'][:, -1, :].t())
            test_user_latent_matrix = test_user_latent_matrix.t()
            test_user_embedding_matrix = torch.cat([test_user_node_matrix, test_user_latent_matrix], -1)
            test_user_embedding_matrix = self.link_mlp(test_user_embedding_matrix)

            link_test_link_prediction = torch.matmul(
                link_test_embed_layer, test_user_embedding_matrix.t()) + self.bias['link_mlp_embeddings']
            link_test_link_prediction = torch.sigmoid(link_test_link_prediction)

            output = {
                'predict_link': link_test_link_prediction,
                'user_node_N': self.user_node_N,
            }
            return output

        else:
            raise ValueError("Wrong mode!!!")
        return output

def test_model():


    batch_size = 128

    model = NJM(batch_size=batch_size)
    loss_func = NJMLossFunction()
    optimizer = optim.Adam(lr=1e-1, weight_decay=1e-3, params=model.parameters())

    # 大约30s
    dataset = Dataset4NJM(mode='train')
    dataset.generate()
    myloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
    # data_tmp = None
    # for idx, batch_data in enumerate(tqdm(myloader)):
        # data_tmp = batch_data
        # with torch.no_grad():
            # optimizer.zero_grad()
            # model.eval()
            # output = model.step(data_tmp, mode='test')
            # test_link_prediction(model)
            # break
            # print(output)
            # loss = loss_func(output)
            # loss.backward()
            # optimizer.step()
    # with open("data/temp_test.pkl", 'wb') as f:
    #     pkl.dump(data_tmp, f)

    # test_set = TestDataset()
    # myloaderloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=2)

    for i in range(1):
        all_loss = 0.
        for idx, data_tmp in enumerate(tqdm(myloader)):
            optimizer.zero_grad()
            model.train()
            output = model.step(data_tmp)
            loss = loss_func(output)
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
        all_loss /= (idx + 1)
        print(all_loss)
    #
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True,
    #     with_modules=True
    # ) as prof:
    #     output = model.step(data_tmp, mode='test')
    #
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    # prof.export_stacks('torch_cpu_stack.json', metric='self_cpu_time_total')
    # prof.export_stacks('torch_cuda_stack.json', metric='self_cuda_time_total')
    # print(prof.table())




if __name__ == '__main__':
    print('DEVICE: ', end='')
    print(DEVICE)
    test_model()