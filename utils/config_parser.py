#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   config_parser.py
@Contact :   xxzhang16@fudan.edu.cn

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/10/6 14:57   zxx      1.0         None
"""

from configparser import ConfigParser

class MyConfigParser(ConfigParser):
    def __init__(self,defaults=None):
        super(MyConfigParser, self).__init__()
    def optionxform(self, optionstr):
        return optionstr