import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from glob import glob
import random
import imutils
table={}


for i in glob('./datasets/images/*.png'):
    fname=i.split('/')[-1].split('.')[0]
    table[fname]=i

class FreezeNetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(FreezeNetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        name = self.annotation_lines[index]
        png = cv2.imread(name,0)
        jpg = cv2.imread(table[name.split('/')[-1].split('.')[0]])
        jpg, png = self.get_random_data(jpg, png, self.input_shape, Train = self.train)

        jpg = np.transpose(np.array(jpg, np.float64)/255, [2,0,1])
        png = np.array(png)
        png=(png!=0).astype(np.uint8)
        

        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]

        seg_labels = seg_labels.reshape([self.input_shape[0], self.input_shape[1], self.num_classes + 1])


        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=0.08, sat=0.3, val=0.4, Train=True):
    
        image=cv2.resize(image,(input_shape[0],input_shape[0]))
        label=cv2.resize(label,(input_shape[0],input_shape[0]))
        

        if Train: 
            angle = random.randrange(-180, 181)
            center = (self.input_shape[0] // 2, self.input_shape[1] // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (self.input_shape[0], self.input_shape[1]), flags=cv2.INTER_NEAREST)
            label = cv2.warpAffine(label, M, (self.input_shape[0], self.input_shape[1]), flags=cv2.INTER_NEAREST)

            r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
            dtype = image.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image, label


def freezenet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
