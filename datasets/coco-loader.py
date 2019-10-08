# code for loader coco

import sys
import os
import torch
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import cv2

class cocoDataset(Dataset):
    # 这一段代码主要来自 https://github.com/yhenon/pytorch-retinanet/blob/master/dataloader.py
    # 因为它真的写的不错～
    def __init__(self, coco_dir, coco_name='train_2017', transform=None):
        self.coco_dir=coco_dir
        self.coco_name=coco_name
        self.transform=transform

        self.coco=COCO(os.path.join(self.coco_dir,'annotations', 'instances_' + self.coco_name + '.json'))
        self.img_idx=self.coco.getImgIds()

        self.load_class()

    def load_class(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        # getCatIds: catNms=[], supNms=[], catIds=[], 如果都为空则直接返回全部类名
        categories.sort(key=lambda x: x['id'])
        self.classes={}
        self.coco_labels={}
        self.coco_labels_inverse={}
        for cat in categories:
            self.coco_labels[len(self.classes)] = cat['id']
            self.coco_labels_inverse[cat['id']] = len(self.classes)
            self.classes[cat['name']] = len(self.classes)
