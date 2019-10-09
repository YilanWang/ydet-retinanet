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
        self.coco_dir = coco_dir  # coco数据和json文件存放位置
        self.coco_name = coco_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.coco_dir, 'annotations', 'instances_' + self.coco_name + '.json'))
        self.img_ids = self.coco.getImgIds()

        self.load_class()

    def load_class(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        # getCatIds: catNms=[], supNms=[], catIds=[], 如果都为空则直接返回全部类名
        # categories[0]={'supercategory': 'person', 'id': 1, 'name': 'person'}
        categories.sort(key=lambda x: x['id'])  # 类别按照id来排序
        # 以下主要是按照上方dict形式创造词典，生成存放{类名：真实类别号}，{coco标签：真实标签号}，{真实标签号：coco标签}3个dict
        # 主要是因为coco的id不是按照从0-79排序
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        self.labels = {}
        for cat in categories:
            self.coco_labels[len(self.classes)] = cat['id']
            self.coco_labels_inverse[cat['id']] = len(self.classes)
            self.classes[cat['name']] = len(self.classes)
            self.labels[len(self.classes)] = self.classes[cat['name']]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        # All datasets that represent a map from keys to data samples should subclass it.
        # All subclasses should overwrite __getitem__(), supporting fetching a data sample for a given key.

        img = self.load_image(index)
        ann = self.load_annotation(index)
        sample = {'img': img, 'ann': ann}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        '''
        example: self.coco.loadImgs(self.img_idx[0])[0]
        {'license': 3,
         'file_name': '000000391895.jpg',
         'coco_url': 'http://images.cocodataset.org/train2017/000000391895.jpg',
         'height': 360,
         'width': 640,
         'date_captured': '2013-11-14 11:18:45',
         'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
         'id': 391895}
        :param image_index:
        :return: img
        '''

        image_info = self.coco.loadImgs(self.img_ids[image_index])[0]
        img_path = os.path.join(self.coco_dir, self.coco_name, self.coco_name,
                                image_info['file_name'])  # 这一步找图像，中间的需根据自己情况修改
        img = cv2.imread(img_path)

        assert len(img.shape[-1]) == 3

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr2rgb

        return img

    def load_annotation(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.img_ids[image_index], iscrowd=False)
        annotations = np.zeros((1, 5))
        if annotations_ids == []:  # gt bbox为空
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        # 组织如下：
        # {'segmentation':list[float],
        # 'area':626.9852500000001 float,
        # 'iscrowd': 0,
        # 'image_id': 391895,
        # 'bbox': [486.01, 183.31, 30.63, 34.98],
        # 'category_id': 2,
        # 'id': 1766676}
        # bbox是top-left，w，h
        for idx, ann in enumerate(coco_annotations):
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:  # 太小
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = ann['bbox']
            annotation[0, 4] = self.cocolabel_to_truelabel(ann['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # [x_top-left, y_top-left, w, h] -> [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def cocolabel_to_truelabel(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def truelabel_to_cocolabel(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.img_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

