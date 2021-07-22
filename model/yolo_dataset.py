import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from model.yolo_util import prep_image


class data_set(Dataset):
    def __init__(self, folder, size, train=True):
        self.folder = folder
        self.size = size
        self.train = train
        img_list = []
        img_list.extend(glob.glob(os.path.join(
            self.folder, '*.jpg')))
        self.img_list = img_list

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = prep_image(cv2.imread(img_path), self.size).squeeze(
            0).requires_grad_()
        (_, img_name) = os.path.split(img_path)
        (name, _) = os.path.splitext(img_name)

        label_path = os.path.join(self.folder, '{}.txt'.format(name))

        with open(label_path, 'r') as f:
            lines = f.read().split('\n')
            lines = [x for x in lines if len(x) > 0]
            label = []
            for l in lines:
                label.append([x.rstrip().lstrip() for x in l.split(' ')])

        for i in range(len(label)):
            label[i][0] = int(label[i][0])
            for j in range(1, 5):
                label[i][j] = float(label[i][j])

        # normalize into 0-1
        label = np.array(label)
        label = np.pad(
            label, ((0, 8 - label.shape[0]), (0, 0)), 'constant', constant_values=0)
        label = torch.tensor(label)

        return img, label

    def __len__(self):
        return len(self.img_list)
