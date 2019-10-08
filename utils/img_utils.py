# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 17:33
# @Author  : yilan.wang
# @Email   : wangyilan521@gmail.com
# @File    : img_utils.py
# @Software: PyCharm

# 定义一些图像的基础操作，主要是因为：
# 1.尽量不使用torchvision
# 2.为了更多操作我们的变换是基于cv2的
# 3.而且torchvision还是基于PIL读取图像的，需要稍作处理
# 毕竟opencv已经很稳定了，torchvision还是一天一个样

import cv2
import numpy as np
# import random


def random_mirror(img,)

