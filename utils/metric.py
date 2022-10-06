#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   metric.py
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/6 14:21   zxx      1.0         None
"""
import numpy as np
import random

class BaseMetric:
    def __init__(self):
        self.init_metric()
        self.metric_dict = dict()

    def init_metric(self, **metric):
        self.metric_dict = dict()

    def compute_metric(self, *input_):
        pass

    def get_batch_metric(self, *input_):
        pass

class NJMMetric(BaseMetric):
    def __init__(self):
        super(NJMMetric, self).__init__()
        self.init_metrics()

    def init_metrics(self):
        self.metric_dict = {
            'rmse' : {'value': 0., 'cnt': 0, 'best': 1e8},
            'precision' : {'value': 0., 'cnt': 0, 'best': 0.},
            'recall' : {'value': 0., 'cnt': 0, 'best': 0.},
            'f1' : {'value': 0., 'cnt': 0, 'best': 0.},
        }

    def clear_metrics(self):
        for k, v in self.metric_dict.items():
            self.metric_dict[k]['value'] = 0
            self.metric_dict[k]['cnt'] = 0


    def _rmse(self, input_dict):
        prediction = input_dict['rating_prediction'][0]['rating_prediction'].cpu().detach().numpy()
        rating_pre_list = input_dict['rating_prediction'][0]['rating_pre_list'].cpu().detach().numpy()
        bsz = prediction.shape[0]
        rmse = np.sum(np.square(prediction - rating_pre_list)) / bsz
        self.metric_dict['rmse']['value'] += rmse
        self.metric_dict['rmse']['cnt'] += 1

    def _f1(self, input_dict):
        predict_link = input_dict['predict_link'].cpu().detach().numpy()
        last_pre = input_dict['last_pre']
        user_id = input_dict['user_id']
        till_record = input_dict['till_record']
        till_record_keys = input_dict['till_record_keys']
        user_node_N = input_dict['user_node_N']

        precision = 0.0
        recall = 0.0

        candidate = np.arange(user_node_N - 1)
        candidate = candidate + 1  # 从1开始，跟用户编号对应
        # viewed_link 训练集+测试集所有的好友关系列表
        viewed_link = []
        # last_pre 是从test数据集中获取的user_id的好友列表
        for user_viewed in last_pre:
            if user_viewed not in viewed_link:
                viewed_link.append(user_viewed)
        # till_record 最后一个step之前的好友关系列表
        if user_id in till_record_keys:
            for user_viewed in till_record:
                if user_viewed not in viewed_link:
                    viewed_link.append(user_viewed)
        # 所有用户中没有跟当前用户链接的用户列表
        candidate = np.array([x for x in candidate if x not in viewed_link])
        # print(len(candidate))
        # 随机抽100个负样本
        candidate = random.sample(list(candidate), 100)
        candidate_value = {}

        for user in candidate:
            # 模型对负样本的预测得分
            candidate_value[user] = predict_link[0][user]
        for user in last_pre:
            # 模型对gt的打分
            candidate_value[user] = predict_link[0][user]

        candidate_value = sorted(candidate_value.items(), key=lambda item: item[1], reverse=True)
        y_predict = []
        # 选前5个打分最高的
        for i in range(5):
            y_predict.append(candidate_value[i][0])

        tp = 0.0
        fp = 0
        if len(last_pre) < 5:
            total_ture = len(last_pre)
        else:
            total_ture = 5.0
        for y_ in y_predict:
            if y_ in last_pre:
                tp += 1.0
            else:
                fp += 1
        precision += tp / 5.0
        recall += tp / total_ture

        self.metric_dict['precision']['value'] += precision
        self.metric_dict['precision']['cnt'] += 1
        self.metric_dict['recall']['value'] += recall
        self.metric_dict['recall']['cnt'] += 1


    def compute_metric(self, input_dict, mode='rmse'):
        if mode == 'rmse':
            self._rmse(input_dict)
        elif mode == 'link':
            self._f1(input_dict)

    def get_batch_metric(self):
        for k in self.metric_dict.keys():
            if k == 'f1':
                continue
            self.metric_dict[k]['value'] /= self.metric_dict[k]['cnt']
            if k == 'rmse':
                self.metric_dict[k]['value'] = np.sqrt(self.metric_dict[k]['value'])
                if self.metric_dict[k]['value'] < self.metric_dict[k]['best']:
                    self.metric_dict[k]['best'] = self.metric_dict[k]['value']
            else:
                if self.metric_dict[k]['value'] > self.metric_dict[k]['best']:
                    self.metric_dict[k]['best'] = self.metric_dict[k]['value']
            self.metric_dict[k]['cnt'] = -1

        precision = self.metric_dict['precision']['value']
        recall = self.metric_dict['recall']['value']
        if precision != 0 and recall != 0:
            self.metric_dict['f1']['value'] = 2 * precision * recall / (precision + recall)
        if self.metric_dict['f1']['value'] > self.metric_dict['f1']['best']:
            self.metric_dict['f1']['best'] = self.metric_dict['f1']['value']
        self.metric_dict['f1']['cnt'] += -1

    def print_best_metric(self):
        metric_str = ''
        for k in self.metric_dict.keys():
            metric_str += f"best {k}: {self.metric_dict[k]['best']:.4f}\n"
        return metric_str



