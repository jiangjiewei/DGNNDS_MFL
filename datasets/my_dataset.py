from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict
import csv
import json
import numpy as np
import os
import sys
from sklearn.metrics import roc_curve, auc

data_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize(256),
        # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        transforms.RandomResizedCrop(224),
        # 图像用于翻转
        transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # train: normMean = [0.39276576, 0.21989475, 0.11854827], normStd = [0.3156468, 0.19685638, 0.1492506]
        # test: normMean = [0.3932243, 0.22020523, 0.1196063], normStd = [0.31652802, 0.20019625, 0.15570432]
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.39276576, 0.21989475, 0.11854827], [0.3156468, 0.19685638, 0.1492506])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.3932243, 0.22020523, 0.1196063], [0.31652802, 0.20019625, 0.15570432])
    ]),
}

# Image.open(iamge_Dir).convert('RGB')

class My_Data_Set(Dataset):
    def __init__(self, image_dir, image_list, transform=None, target_transform=None):
        super(My_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        self.image_dir = image_dir
        fp = open(image_list, 'r', encoding='utf-8')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.replace('(','')
            line.replace(')','')
            line.rstrip()
            information = line.split(',')
            images.append(information[0])
            # 将标签信息由str类型转换为float类型
            labels.append([float(l) for l in information[1:len(information)]])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = Image.open(os.path.join(self.image_dir, imageName)).convert('RGB')
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        # 需要将标签转换为float类型，BCELoss只接受float类型
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        '''
            返回数据总数
        :return:
        '''
        return len(self.images)



