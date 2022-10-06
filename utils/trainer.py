#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/6 14:21   zxx      1.0         None
'''

import torch
import numpy as np

from tqdm import tqdm
import pickle

class Trainer:
    def __init__(self, model, loss_func, optimizer, metric, train_loader, test_loader, config):
        self.config = config
        self.random_seed = eval(self.config['TRAIN']['random_seed'])
        self.log_pth = self.config['TRAIN']['log_pth'] + str(self.random_seed) + '_njm_torch.txt'
        self.print_config()

        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metric = metric
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.data_name = config['DATA']['data_name']
        self.device = self.config['TRAIN']['device']

        self.to(self.device)

    def print_config(self):
        config_str = ''
        config_str += '=' * 10 + "Config" + '=' * 10 + '\n'
        for k, v in self.config.items():
            config_str += k + ': \n'
            for _k, _v in v.items():
                config_str += f'\t{_k}: {_v}\n'
        config_str += ('=' * 25 + '\n')
        tqdm.write(self.log(config_str, mode='w'))


    def to(self, device=None):
        if device is None:
            self.model = self.model.to(self.config['TRAIN']['device'])
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['TRAIN']['device'])
            self.config['TRAIN']['device'] = device

    def step(self, batch_data, mode='train'):
        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model.step(batch_data, mode=mode)
            loss = self.loss_func(output)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                output = self.model.step(batch_data, mode=mode)
                loss = self.loss_func(output)
                self.metric.compute_metric(output)
                return loss.item()
        elif mode == 'evaluate_link':
            with torch.no_grad():
                self.model.eval()
                ipt = {'link_test_user_id': torch.tensor(batch_data['user_id'], device=self.device, dtype=torch.long)}
                # 这里输入好像只有userid，应该是直接过mlp打分
                output = self.model.step(ipt, mode='evaluate_link')
                metric_input_dict = batch_data
                metric_input_dict.update({
                    'predict_link': output['predict_link'],
                    'user_node_N': output['user_node_N']
                })
                self.metric.compute_metric(metric_input_dict, mode='link')
                return
        else:
            raise ValueError("Wrong Mode")

    def _compute_metric(self, metric_str):
        self.metric.get_batch_metric()
        for k, v in self.metric.metric_dict.items():
            metric_str += f'{k}: {self.metric.metric_dict[k]["value"]:4f}\n'
        self.metric.clear_metrics()
        return metric_str

    def log(self, str_, mode='a'):
        with open(self.log_pth, mode, encoding='utf-8') as f:
            f.write(str_)
            f.write('\n')
        return str_

    def train(self):
        tqdm.write(self.log("=" * 10 + "TRAIN BEGIN" + "=" * 10))
        epoch = eval(self.config['TRAIN']['epoch'])
        self.metric.init_metrics()
        for e in range(1, epoch + 1):
            all_loss = 0.0
            for s, batch_data in enumerate(tqdm(self.train_loader, desc='train')):
                loss = self.step(batch_data, mode='train')
                all_loss += loss

            all_loss /= s + 1
            metric_str = f'Train Epoch: {e}\nLoss: {all_loss:.4f}\n'
            if e % 1 == 0:
                all_loss = 0.0
                self.metric.clear_metrics()
                for s, batch_data in enumerate(tqdm(self.test_loader, desc='evaluate')):
                    loss = self.step(batch_data, mode='evaluate')
                    all_loss += loss

                all_loss /= s + 1

                with open("data/test_link_" + self.data_name + ".pkl", 'rb') as f:
                    test_link = pickle.load(f)
                for user_id in tqdm(test_link['last_pre'].keys(), desc='link_evaluate'):
                    # print(user_id)
                    if len(test_link['last_pre'].keys()) >= 1:
                        batch_data = {
                            'last_pre': test_link['last_pre'][user_id],
                            'user_id': user_id,
                            'till_record': test_link['till_record'][user_id],
                            'till_record_keys': test_link['till_record'].keys(),
                        }
                        self.step(batch_data, mode='evaluate_link')
                metric_str += f'Valid Epoch: {e}\n'
                metric_str += f'valid rating loss: {all_loss:.4f}\n'
                metric_str = self._compute_metric(metric_str)

                tqdm.write(self.log(metric_str))

        tqdm.write(self.log(self.metric.print_best_metric()))
        tqdm.write("=" * 10 + "TRAIN END" + "=" * 10)