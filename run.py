#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run.py
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/6 14:19   zxx      1.0         None
"""

# 加载依赖
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from Dataset_torch import Dataset4NJM
from NJM_Torch import NJM, NJMLossFunction
from utils.metric import NJMMetric
from utils.trainer import Trainer
from utils.config_parser import MyConfigParser
from utils.weight_init import weight_init

import argparse

def get_config(config_pth):
	config = MyConfigParser()
	config.read('./config/' + config_pth, encoding='utf-8')
	return config._sections


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)

def run(config_pth):
	# get config
	config = get_config(config_pth)

	seed = eval(config['TRAIN']['random_seed'])
	setup_seed(seed)

	# get data loader
	## train data
	train_dataset = Dataset4NJM(mode=config['TRAIN']['mode'])
	train_dataset.generate()
	train_loader = DataLoader(
		train_dataset,
		batch_size=eval(config['DATA']['train_batch_size']),
		shuffle=False,
		pin_memory=True,
		drop_last=True
	)

	## test data
	test_dataset = Dataset4NJM(mode='test')
	test_dataset.generate()
	test_loader = DataLoader(
		test_dataset,
		batch_size=eval(config['DATA']['test_batch_size']),
		shuffle=False,
		pin_memory=True,
		drop_last=True
	)

	# 初始化模型，损失和优化器
	model = NJM(config)
	model.apply(weight_init)

	paras = dict(model.named_parameters())

	lr = eval(config['OPTIM']['learning_rate'])
	# weight_decay
	common_weight_decay = eval(config['OPTIM']['common_weight_decay'])
	balance_weight_decay = eval(config['OPTIM']['balance_weight_decay'])
	mlp_weight_decay = eval(config['OPTIM']['mlp_weight_decay'])
	transformation_weight_decay = eval(config['OPTIM']['transformation_weight_decay'])
	weights_weight_decay = eval(config['OPTIM']['weights_weight_decay'])
	# 1e-1 weight_decay
	# for k, v in paras.items():
	#     print(k.ljust(30), str(v.shape).ljust(30), 'grad:', v.requires_grad)
	paras_new = []
	for k, v in paras.items():
		if 'mlp.0' in k:
			paras_new += [{'params': [v], 'lr': lr, 'weight_decay': mlp_weight_decay}]
		elif 'weights.' in k:
			if 'balance' in k:
				paras_new += [{'params': [v], 'lr': lr, 'weight_decay': balance_weight_decay}]
			elif 'transformation' in k:
				paras_new += [{'params': [v], 'lr': lr, 'weight_decay': transformation_weight_decay}]
			else:
				paras_new += [{'params': [v], 'lr': lr, 'weight_decay': weights_weight_decay}]
		else:
			paras_new += [{'params': [v], 'lr': lr, 'weight_decay': common_weight_decay}]

	optimizer = optim.Adam(params=paras_new)

	loss_func = NJMLossFunction(config)
	metric = NJMMetric()

	# 构建trainer
	trainer = Trainer(
		model=model,
		loss_func=loss_func,
		optimizer=optimizer,
		metric=metric,
		train_loader=train_loader,
		test_loader=test_loader,
		config=config,
	)

	trainer.train()
def parse_args():
	########
	# Parses the NJM arguments.
	#######
	parser = argparse.ArgumentParser(description="Run NJM.")

	parser.add_argument('--config_pth', type=str, default=1 ,
						help='Choose config')
	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	run(args.config_pth)