#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Dataset_torch.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/2 19:38   zxx      1.0         None
'''
# 复用了https://github.com/NJMCODE2018/NJM的代码
"""
重构NJM的dataset方法，使用torch实现
"""
import numpy as np
import time,datetime
import random
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

def Normalization_gowalla(inX):
    return 1.0 / (1 + np.exp(-inX))

def Normalization_epinions(x,Max=5.0,Min=0):
    x = (x - Min) / (Max - Min)
    return x

class Dataset4NJM(Dataset):
    def __init__(
            self,
            path='data/',
            mode='train',
            negative=5,
            train_step = 11,
            u_counter = 4630,
            s_counter = 26991,
            data_name='epinions'):
        '''
            Constructor:
                data_name: Name of Dataset
        '''
        self.data_name = data_name
        if mode == 'train' or mode == 'test' or mode == 'debug':
            self.mode = mode
        else:
            raise ValueError("Mode should be train or test!")
        # train_id_list 保存互动记录，[user1, item1], [user1, item2], [user2, item1]...
        self.train_id_list = []
        self.train_rating_list = []
        # [[user1, item1], [user2,item2],...]
        self.test_id_list = []
        # 跟idlist对应，[rating1,rating2,...]
        self.test_rating_list = []
        # 无论是train还是test都用self.data统一存储
        self.data = {}
        # self.train_data = {}
        # self.test_data = {}
        # self.user_dict: {userID: [[item1ID, rating, step], ...],...}
        self.user_dict = {}
        # self.spot_dict: {itemID: [[user1ID, rating, step], ...],...}, 倒排表
        self.spot_dict = {}
        self.user_enum = {}
        self.spot_enum = {}
        # 　self.links: {user1ID: [[user2ID, step], [user3ID, step],...],...}
        # 包含所有社交关系中的用户，采用了单向边的记录法，记录了第一位用户，有可能存在漏掉的用户
        self.links = {}
        # 　self.links_array: {user1ID: [user2ID, user3ID,...],...}
        self.links_array = {}
        self.u_counter = u_counter
        self.s_counter = s_counter
        self.path = path
        self.negative = negative
        # self.train_step 表示用前t轮的数据训练
        self.train_step = train_step


    def generate(self):
        import os
        if os.path.exists("data/train_"+ self.data_name+".pkl"):
            self.load_data()
        else:
            self.get_inter_data()
            if self.mode == 'train':
                self.get_train_data()
            else:
                self.get_test_rating()
                self.get_test_link()

    # from memory_profiler import profile
    #
    # @profile(precision=4, stream=open("memory_profiler.log", "w+"))
    def load_data(self):
        if self.mode == 'train':
            # train_data
            with open("data/train_" + self.data_name+".pkl", 'rb') as f:
                self.data = pickle.load(f)
            print("finish getting train_data...")
        elif self.mode == 'test':
            # test_data
            with open("data/test_rating_"+self.data_name+".pkl", 'rb') as f:
                self.data = pickle.load(f)
            print("finish getting test rating...")

            # test link
            with open("data/test_link_"+self.data_name+".pkl", 'rb') as f:
                test_link = pickle.load(f)
            self.last_pre = test_link['last_pre']
            self.till_record = test_link['till_record']
            print("finish getting test link...")
        elif self.mode == 'debug':
            with open("data/temp.pkl", 'rb') as f:
                self.data = pickle.load(f)
            print("finish getting debug_data...")

    def get_inter_data(self):
        # 保存交互的记录，只用了训练集的数据
        # rating数据集组织：userID itemID rating step
        user_r = {} # 存储用户-物品交互列表，user: [item1, item2...]
        count = 0
        f = open("data/"+self.data_name+".train.rating")
        line = f.readline()
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user_id = int(arr[0])
            spot_id = int(arr[1])

            if user_id in user_r.keys():
                if spot_id not in user_r[user_id]:
                    # train_id_list 保存互动记录，[user1, item1], [user1, item2], [user2, item1]...
                    self.train_id_list.append([user_id, spot_id])
                    count += 1
                    user_r[user_id].append(spot_id)
            else:
                self.train_id_list.append([user_id, spot_id])
                count += 1
                user_r[user_id]= [spot_id]

            line = f.readline()
        f.close()

        # link数据集组织：user1ID user2ID step
        f = open("data/"+self.data_name+".train.link")
        line = f.readline()
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user1 = int(arr[0])
            if user1 not in user_r.keys():
                # 没有在用户-物品互动中出现的user
                # TODO: 下面在干嘛没看懂
                user_r[user1] = [1]
                for i in range(20):
                    self.train_id_list.append([user1, i+1])
            line = f.readline()
        f.close()

        f = open("data/"+self.data_name+".train.rating")
        line = f.readline()
        train_data_num_T = 0
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user_id = int(arr[0])
            spot_id = int(arr[1])
            rating = float(arr[2])
            if self.data_name == 'gowalla':
                rating = Normalization_gowalla(rating)
            else:
                rating = Normalization_epinions(rating) # 将评分映射到0-1
            step = int(arr[3])
            # self.user_dict: {userID: [[item1ID, rating, step], ...],...}
            if user_id in self.user_dict.keys():
                self.user_dict[user_id].append([spot_id,rating, step])
            else:
                self.user_dict[user_id] = [[spot_id,rating, step]]
            # self.spot_dict: {itemID: [[user1ID, rating, step], ...],...}, 倒排表
            if spot_id in self.spot_dict.keys():
                self.spot_dict[spot_id].append([user_id,rating, step])
            else:
                self.spot_dict[spot_id] = [[user_id,rating, step]]

            train_data_num_T += 1
            line = f.readline()
        f.close()


        f = open("data/"+self.data_name+".train.link")
        line = f.readline()
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user1 = int(arr[0])
            user2 = int(arr[1])
            step = int(arr[2])
            #　self.links: {user1ID: [[user2ID, step], [user3ID, step],...],...}
            if user1 not in self.links.keys():
                self.links[user1] = [[user2, step]]
            else:
                self.links[user1].append([user2, step])
            # 　self.links_array: {user1ID: [user2ID, user3ID,...],...}
            if user1 not in self.links_array.keys():
                self.links_array[user1] = [user2]
            else:
                self.links_array[user1].append(user2)
            line = f.readline()
        f.close()


        inter_data = {}
        inter_data['ids'] = self.train_id_list
        inter_data['user_dict'] = self.user_dict
        inter_data['spot_dict'] = self.spot_dict
        inter_data['links'] = self.links
        inter_data['links_array'] = self.links_array

        with open("data/inter_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(inter_data, f)

        print("finish getting inter_data...")

    def get_train_data_by_id(self, id=0):
        res = {}
        # self.train_step 表示用前t轮的数据训练
        # rating_indicator
        rating_indicator = np.zeros(self.train_step, dtype='int32')
        rating_pre = np.zeros(self.train_step, dtype='float32')
        spot_attr = []  # 与这条互动中的物品有过交互的其他用户 [[userID,rating,step],...]
        link_res = []   # 当前用户邻接的好友及step [[userID,step],...]
        link_predict_weight = []    # 包括正负样本 [[user2ID,step,weight],...]
        # self.train_id_list[id]是一条记录: [userID, itemID]
        # user1ID in all_users_set
        if self.train_id_list[id][0]  in self.user_dict.keys():
            for record in self.user_dict[self.train_id_list[id][0]]:
                # record: [itemID,rating,step]
                if record[0] == self.train_id_list[id][1]:
                    # 标记这次互动发生在第几个step
                    rating_pre[record[2]] = record[1]
                    rating_indicator[record[2]] = 1

        if self.train_id_list[id][1] in self.spot_dict.keys():
            for record in self.spot_dict[self.train_id_list[id][1]]:
                # record: [userID,rating,step]
                # 这个循环把当前这条记录中，与用户互动的物品有过交互的其他用户记录下来
                spot_attr.append([record[0], record[1], record[2]])

        # self.links.keys()包含所有数据集中发起边的用户
        if self.train_id_list[id][0] in self.links.keys():
            for record in self.links[self.train_id_list[id][0]]:
                # record: [user2ID,step]
                link_res.append([record[0], record[1]])
                # 所有边权重都看作1
                link_predict_weight.append([record[0], record[1], 1.0])
                for i in range(self.negative):
                    # 负采样，总数为用户的好友数*5
                    link_predict_weight.append([random.randrange(start=1,stop=self.u_counter),record[1] ,1.0/self.negative])


        res['user_id'] = self.train_id_list[id][0]
        res['spot_id'] = self.train_id_list[id][1]
        res['spot_attr'] = spot_attr
        res['rating_pre'] = rating_pre
        res['rating_indicator'] = rating_indicator
        res['link_res'] = link_res
        res['link_predict_weight'] = link_predict_weight
        return res

    def get_train_data(self):
        print("start getting train_data...")
        res_t = {}  # {idx: one_data}
        # self.train_id_list里每条记录就是一次用户-物品互动，其中训练数据中没有出现的用户都被赋予了1-20这二十条记录
        for i in range(len(self.train_id_list)):
            res_one = self.get_train_data_by_id(id = i)
            res_t[i] = res_one
        print("finish getting train_data...")
        self.data = res_t
        with open("data/train_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(res_t, f)

    def get_test_rating(self):
        print("start getting test rating...")
        f = open("data/"+self.data_name+".test.rating")
        line = f.readline()
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user_id = int(arr[0])
            spot_id = int(arr[1])
            rating = float(arr[2])
            if self.data_name == 'gowalla':
                rating = Normalization_gowalla(rating)
            else:
                rating = Normalization_epinions(rating)

            if [user_id, spot_id] not in self.test_id_list:
                self.test_id_list.append([user_id, spot_id])
                self.test_rating_list.append(rating)

            line = f.readline()
        f.close()

        test_data = {}
        # 所有的互动列表，[(user,item),...]
        test_data['ids'] = self.test_id_list
        # 所有的评分列表，[r1，r2,...]
        test_data['rating'] = self.test_rating_list
        # inter_data的数据
        test_data['user_dict'] = self.user_dict
        test_data['spot_dict'] = self.spot_dict
        test_data['links'] = self.links # 这里没有修改links
        self.data = test_data
        print("finish getting test rating...")
        with open("data/test_rating_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(test_data, f)

    def get_test_link(self):
        # 最后一个step的link，{user1:[user2,user3,...],...}
        self.last_pre = {}
        # links_array的扩充，把训练集中没有出现的也加入到字典中
        # {user1:[user2,user3,...], user2:[],...}
        # 应该表示用来辅助预测的好友关系
        self.till_record = {}
        print("start getting test link...")
        f = open("data/"+self.data_name+".test.link")
        line = f.readline()
        while line:
            line = line.rstrip()
            arr = line.split('\t')
            user1 = int(arr[0])
            user2 = int(arr[1])

            if user1 not in self.last_pre.keys():
                self.last_pre[user1] = [user2]
            else:
                self.last_pre[user1].append(user2)
            if user1 not in self.till_record.keys():
                if user1 in self.links_array.keys():
                    self.till_record[user1] = self.links_array[user1]
                else:
                    self.till_record[user1] = []
            line = f.readline()
        f.close()

        test_link = {}
        test_link['last_pre'] = self.last_pre
        test_link['till_record'] = self.till_record
        with open("data/test_link_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(test_link, f)
        print("finish getting test link...")

    def __getitem__(self, fetch_id):
        # 下面两个量根据NJM代码
        # 下面都是针对一条记录（user，item）的
        user_node_N = self.u_counter + 1
        train_T = self.train_step
        if self.mode == 'train':
            # spot_attr[t,i]=r 表示用户i在t step对这条记录中的物品的打分是r
            # spot_attr记录的是某个物品被交互的总记录
            spot_attr = np.zeros((train_T, user_node_N), dtype='float32')

            spot_attr[0] = np.random.normal(size=(user_node_N))

            one = self.data[fetch_id]
            # one['spot_attr']=[[userID,rating,step],...]
            for record in one['spot_attr']:
                # 训练数据中step从0开始，test的step=train_T = self.train_step
                if record[2] < train_T - 1: # 这个记录不是训练数据中最后一个step的
                    # 将互动记录赋值到spot_attr上，step=i对应第i+1行
                    spot_attr[record[2] + 1][record[0]] = record[1]

            # 当前用户的所有好友关系记录
            link = np.zeros((train_T, user_node_N), dtype='int32')
            # 下面这个包括了负采样的边
            link_weight = np.zeros((train_T, user_node_N), dtype='float32')
            # one['link_res']=[[user2ID,step],...]
            for record in one['link_res']:
                link[record[1]][record[0]] = 1


            for record in one['link_predict_weight']:
                # 有点问题，明明应该包括负采样的边，而且已经记录了权重但是全赋值为1
                # record = [userID, step, weight]
                link_weight[record[1]][record[0]] = 1.0

            res = {
                # 这条互动的userid
                'user_id_list': one['user_id'],
                # 这条互动的itemid
                'spot_id_list': one['spot_id'],
                # 构建一个长为train_T的向量，每个元素都是spot_id
                'spot_id_list_list': np.repeat(one['spot_id'], train_T),
                # rating_indicator表明当前的用户和物品在哪些step发生了互动
                'rating_indicator_list': one['rating_indicator'],
                # rating_pre表明在当前的用户和物品在所有step上互动的评分
                'rating_pre_list': one['rating_pre'],
                # spot_attr记录的是当前被交互的总记录,类似物品的”社交“图
                'spot_attr_list': spot_attr,
                # 当前用户在所有step上的好友关系记录
                'link_list': link,
                # 当前用户在所有step上的好友关系权重
                'link_weight_list': link_weight
            }
            return res
            # user_id_list.append(one['user_id'])
            # spot_id_list.append(one['spot_id'])
            # spot_id_list_list.append(np.repeat(one['spot_id'], self.train_T))
            # rating_indicator_list.append(one['rating_indicator'])
            # # consumption_weight_list.append(consumption_weight)
            # spot_attr_list.append(spot_attr)
            # rating_pre_list.append(one['rating_pre'])
            # link_list.append(link)
            # link_weight_list.append(link_weight)
        elif self.mode == 'test':
            # self.data['ids']所有的互动列表，[(user,item),...]
            user_id = self.data['ids'][fetch_id][0]
            spot_id = self.data['ids'][fetch_id][1]

            # 最后一个也就是测试预测的step长度的向量，每个都是物品的id
            spot_list = np.repeat(spot_id, train_T + 1)
            # rating 所有的评分列表，[r1，r2,...]
            rating = self.data['rating'][fetch_id]

            spot_attr = np.zeros((train_T + 1, user_node_N), dtype='float32')
            # link_res = np.zeros((self.train_T , self.user_node_N), dtype='int32')
            friend_record = np.zeros((user_node_N), dtype='float32')
            spot_attr[0] = np.random.normal(size=(user_node_N))
            # in 表示spot_id在训练集中出现过，在spot_attr上记录这个物品的互动情况
            if spot_id in self.data['spot_dict'].keys():
                for record in self.data['spot_dict'][spot_id]:
                    spot_attr[record[2] + 1][record[0]] = record[1]
            # 记录当前用户都跟谁互动过，friend是一个user数量加1长度的向量
            if user_id in self.data['links'].keys():
                for link in self.data['links'][user_id]:
                    friend_record[link[0]] = 1.0

            res = {
                'user_id_list': user_id,
                'spot_id_list': spot_id,
                'spot_id_list_list': spot_list,
                'spot_attr_list': spot_attr,
                'friend_record_list': friend_record,
                'rating_pre_list': rating
            }
            return res
            # user_id_list.append(user_id)
            # spot_id_list.append(spot_id)
            # spot_id_list_list.append(spot_list)
            # spot_attr_list.append(spot_attr)
            # friend_record_list.append(friend_record)
            # rating_pre_list.append(rating)
        elif self.mode == 'debug':
            return {k: v[fetch_id] for k, v in self.data.items()}

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        elif self.mode == 'test':
            return len(self.data['ids'])
        elif self.mode == 'debug':
            return len(self.data['user_id_list'])

class MyDataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            total_num,
            device
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.total_num = total_num
        self.total_batch_num = int(total_num / batch_size) + 1
        self.device = device

    def get_batch_data(self):
        for batch_id in range(self.total_batch_num):
            res = {}
            for id in range(self.batch_size):
                if (id + batch_id * self.batch_size) > (self.total_num - 1):
                    fetch_id = (id + batch_id * self.batch_size) % self.total_num
                else:
                    fetch_id = id + batch_id * self.batch_size

                one = self.dataset[fetch_id]

                for k in one.keys():
                    if k not in res.keys():
                        res[k] = [one[k]]
                    else:
                        res[k].append(one[k])

            for k in res.keys():
                res[k] = np.array(res[k])

            yield {k: torch.tensor(res[k], device=self.device) for k in res.keys()}


def dataset_test():
    # 训练集测试
    mydataset = Dataset4NJM(mode='test')
    mydataset.generate()
    print(mydataset[0])
    batch_size = 16
    # myloader = DataLoader(mydataset, batch_size=16, shuffle=True)
    total_num = len(mydataset)
    total_batch_num = int(total_num / batch_size) + 1

    for batch_id in range(total_batch_num):
        for id in range(batch_size):
            if (id + batch_id * batch_size) > (total_num - 1):
                fetch_id = (id + batch_id * batch_size) % total_num
            else:
                fetch_id = id + batch_id * batch_size
            one = mydataset[fetch_id]

            print(one)
            print('=' * 20)
            break

    # for batch_x in myloader:
    #     print(batch_x)
    #     print('='*20)
    #     break

    # # 测试集
    # mydataset = Dataset4NJM(mode='test')
    # mydataset.load_data()


if __name__ == '__main__':
    # with open("data/train_" + 'epinions' + ".pkl", 'rb') as f:
    #     data = pickle.load(f)
    dataset_test()

