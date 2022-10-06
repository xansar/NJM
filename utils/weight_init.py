#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   weight_init.py    
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/6 22:05   zxx      1.0         None
"""
import torch.nn as nn

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_normal_(m.weight.data)

