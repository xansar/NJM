import numpy as np
import time,datetime
import random
import pickle

# torch
from torch.utils.data import Dataset


def Normalization_gowalla(inX):
    return 1.0 / (1 + np.exp(-inX))

def Normalization_epinions(x,Max=5.0,Min=0):
    x = (x - Min) / (Max - Min)
    return x

class Dataset(object):
    def __init__(
            self,
            path='data/',
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
        # train_id_list 保存互动记录，[user1, item1], [user1, item2], [user2, item1]...
        self.train_id_list = []
        self.train_rating_list = []
        # [[user1, item1], [user2,item2],...]
        self.test_id_list = []
        # 跟idlist对应，[rating1,rating2,...]
        self.test_rating_list = []
        self.train_data = {}
        self.test_data = {}
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


    #
    def generate(self):
        import os
        if os.path.exists("data/inter_"+self.data_name+".pkl"):
            self.load_data()
        else:
            self.get_inter_data()
            self.get_train_data()
            self.get_test_rating()
            self.get_test_link()

    def load_data(self):
        # inner_data
        with open("data/inter_"+self.data_name+".pkl", 'rb') as f:
            inter_data = pickle.load(f)
        self.train_id_list = inter_data['ids']
        self.user_dict = inter_data['user_dict']
        self.spot_dict = inter_data['spot_dict']
        self.links = inter_data['links']
        self.links_array = inter_data['links_array']
        print("finish getting inter_data...")

        # train_data
        ## train_data will be loaded in training process
        print("finish getting train_data...")

        # test_data
        with open("data/test_rating_"+self.data_name+".pkl", 'rb') as f:
            test_data = pickle.load(f)
        self.test_id_list = test_data['ids']
        self.test_rating_list = test_data['rating']
        self.user_dict = test_data['user_dict']
        self.spot_dict = test_data['spot_dict']
        self.links = test_data['links']
        print("finish getting test rating...")

        # test link
        with open("data/test_link_"+self.data_name+".pkl", 'rb') as f:
            test_link = pickle.load(f)
        self.last_pre = test_link['last_pre']
        self.till_record = test_link['till_record']
        print("finish getting test link...")


    def get_inter_data(self):
        # 保存交互的记录
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
        res_t = {}
        # self.train_id_list里每条记录就是一次用户-物品互动，其中训练数据中没有出现的用户都被赋予了1-20这二十条记录
        for i in range(len(self.train_id_list)):
            res_one = self.get_train_data_by_id(id = i)
            res_t[i] = res_one
        print("finish getting train_data...")
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
        test_data['ids'] = self.test_id_list
        test_data['rating'] = self.test_rating_list
        test_data['user_dict'] = self.user_dict
        test_data['spot_dict'] = self.spot_dict
        test_data['links'] = self.links
        print("finish getting test rating...")
        with open("data/test_rating_"+self.data_name+".pkl", 'wb') as f:
            pickle.dump(test_data, f)

    def get_test_link(self):
        # 最后一个step的link，{user1:[user2,user3,...],...}
        self.last_pre = {}
        # links_array的扩充，把训练集中没有出现的也加入到字典中，{user1:[user2,user3,...],...}
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




