# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 16:30
# @Author  : yilan.wang
# @Email   : wangyilan521@gmail.com
# @File    : train.py
# @Software: PyCharm

# 此训练文件为示例文件，目前使用coco当作示例，resnet50当作basemodel。
# 我们的基本思路是，train-数据集-base网络-改动.py这种方式，而不采用
# 类似argparse这种方式，因为往往argparse是成熟期代码的简洁表达方式，
# 这种表达不太适合research，或者开发期使用。argparse或者其他配置参数
# 的表达方式将会在后期集成到ycv后采用。


import time
import os
import sys
import collections

import numpy as np

import torch as t
import torchvision as tv

def main():
    assert t.cuda.is_available()

