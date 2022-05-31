import os
import math
import random
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import gc;

gc.enable()  # memory is tight
import torch
import pdb
import torch.utils.data as data
import cv2
from image import flip, color_aug
from image import get_affine_transform, affine_transform

dtype = "float32"


## HeatMap Genrating Functions

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian1D(shape, sigma=1):
    m = (shape - 1.) / 2.
    y = np.ogrid[-m:m + 1]
    h = np.exp(-(y * y) / (2 * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, y_0, y_1, x_0, x_1, radius, k=1, is_train_x0=True):
    diameter = 2 * radius + 1
    # pdb.set_trace()
    # gaussian = gaussian1D((diameter), sigma=diameter / 6)
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)  # array([[0.01831564, 0.13533528, 0.01831564],
    # [0.13533528, 1.        , 0.13533528],
    # [0.01831564, 0.13533528, 0.01831564]])
    y, x = int(center[1]), int(center[0])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_gaussian = gaussian[radius:radius + bottom, radius]  # [0:3, 0:3]
    masked_gaussian_low = gaussian[radius - top:radius + 1, radius]  # [0:3, 0:3]
    if is_train_x0:

        # masked_heatmap = heatmap[y - top :y + bottom, x_0 : x_1]
        # masked_gaussian = np.expand_dims(gaussian[radius - top:radius + bottom,radius],axis=1) * np.ones_like(masked_heatmap)
        masked_heatmap = heatmap[y_0:y_0 + bottom, x_0]  # array([[0., 0., 0.],
        #       [0., 0., 0.],
        #       [0., 0., 0.]], dtype=float32)
        masked_heatmap_e = heatmap[y_0:y_0 + bottom, x_1]
        if min(masked_gaussian.shape) > 0 and (
                min(masked_heatmap.shape) > 0 or min(masked_heatmap_e.shape) > 0):  # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
            np.maximum(masked_heatmap_e, masked_gaussian * k, out=masked_heatmap_e)
    else:
        masked_heatmap_low = heatmap[y_1 - top:y_1 + 1, x_0]
        masked_heatmap_lowe = heatmap[y_1 - top:y_1 + 1, x_1]
        if min(masked_gaussian.shape) > 0 and (
                min(masked_heatmap_low.shape) > 0 or min(masked_heatmap_lowe.shape) > 0):  # TODO debug
            np.maximum(masked_heatmap_low, masked_gaussian_low * k, out=masked_heatmap_low)
            np.maximum(masked_heatmap_lowe, masked_gaussian_low * k, out=masked_heatmap_lowe)

    return heatmap


def draw_umich_ind(ind_heatmap, center, x_0, x_1):
    y, x = int(center[1]), int(center[0])

    ind_heatmap[y, x_0: x_1] = 1
    return ind_heatmap


def reverse_res_to_bbox(res):
    down_ratio = res['input'].shape[1] / res['hm'].shape[1]
    output_h, output_w = res['hm'].shape[0], res['hm'].shape[1]
    num_objs = np.sum(res['reg_mask'])
    bbox = np.zeros((num_objs, 4), dtype='float32')
    ct = res['reg'][:num_objs]
    ct[:, 0] += res['ind'][:num_objs] % output_w
    ct[:, 1] += res['ind'][:num_objs] // output_w
    h, w = res['wh'][:num_objs, 1], res['wh'][:num_objs, 0]
    bbox[:, 0] = (ct[:, 0] * 2 - w) / 2
    bbox[:, 2] = (ct[:, 0] * 2 + w) / 2
    bbox[:, 1] = (ct[:, 1] * 2 - h) / 2
    bbox[:, 3] = (ct[:, 1] * 2 + h) / 2
    bbox *= down_ratio
    return bbox


def reverse_res_to_bbox_centerline_ind(res):
    # pdb.set_trace()
    down_ratio = res['input'].shape[1] / res['hm'].shape[1]
    output_h, output_w = res['hm'].shape[0], res['hm'].shape[1]
    num_objs = np.sum(res['reg_mask'])
    bbox = np.zeros((num_objs, 4), dtype='float32')
    ct = np.zeros((num_objs, 2), dtype="float32")
    ct_y = res['reg_y'][:num_objs]
    bbox[:, 0] = res['ind'][:num_objs][:, 0] % output_w
    bbox[:, 2] = res['ind'][:num_objs][:, 1] % output_w
    ct_y += res['ind'][:num_objs][:, 0] // output_w
    ct[:, 1] = ct_y
    # h, w = res['wh'][:num_objs, 1], res['wh'][:num_objs, 0]
    h_d, h_u = res['h_ud'][:num_objs, 1], res['h_ud'][:num_objs, 0]

    bbox[:, 1] = (ct[:, 1] + h_d)
    bbox[:, 3] = (ct[:, 1] + h_u)
    bbox *= down_ratio
    return bbox


def reverse_res_to_bbox_centerline(res):
    # pdb.set_trace()
    down_ratio = res['input'].shape[1] / res['hm'].shape[1]
    output_h, output_w = res['hm'].shape[0], res['hm'].shape[1]
    num_objs = np.sum(res['reg_mask'])
    bbox = np.zeros((num_objs, 4), dtype='float32')
    ct = np.zeros((num_objs, 2), dtype="float32")
    ct_y = np.squeeze(res['reg_y'][:num_objs])
    bbox[:, 0] = res['ind'][:num_objs][:, 0] % output_w
    bbox[:, 2] = res['ind'][:num_objs][:, 1] % output_w
    ct_y += res['ind'][:num_objs][:, 0] // output_w
    # ct[:, 1] = ct_y
    # h, w = res['wh'][:num_objs, 1], res['wh'][:num_objs, 0]
    h_ud = np.squeeze(res['h_ud'][:num_objs])

    bbox[:, 1] = ct_y
    bbox[:, 3] = (ct_y + h_ud)
    bbox *= down_ratio
    return bbox


class ctDataset(data.Dataset):

    def __init__(self, split="test"):
        # img_dir = 'kitti dataset/'
        # ../input/kitti_single/training/label_2/
        self.keep_res = False
        self.split = split
        self.not_rand_crop = True
        self.no_color_aug = False
        self.original_shift = 0
        self.original_scale = 0
        self.scale = [0, 0.3, 0.5, 0.7]  # , 1.2, 1.5, 1.8, 2.0]
        self.shift = [0, 0.1, 0.2, 0.3]  # , 0.5, 0.7, 0.8, 0.6]
        # self.scale = [0, 0.7, 0.9, 1.2]
        # self.shift = [0, 0.1,0.2]
        self.flip = 0.5
        self.scale_shift_values = 0
        self.num_classes = 4
        self.max_objs = 80
        self.down_ratio = 2
        self.color_aug_value = 0.2
        self.input_h, self.input_w = 512, 512
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]], dtype=np.float32)
        self.mean = [0.64566372, 0.64566372, 0.64566372]
        self.std = [0.08311639, 0.08311639, 0.08311639]
        root = '/media/zjn/F/Data/TX_data/all_bbox/'
        self.root = root
        self.image_dir = os.path.join(root, "RGB/4K_800_801_data_imag_add_mode/")
        self.label_dir = os.path.join(root, "label_txt/4K_800_801_label_txt_add_mode_only_start-stop-points/")
        #self.image_dir = os.path.join(root,"RGB/1K_2000_3200_data_imag_1000/")  # ceshi/")#short_data_imag_4000/")
        #self.label_dir = os.path.join(root, "label_txt/1K_2000_3200_data_imag_1000/")
        if self.split == "train":
            imageset_txt = os.path.join(root, "imageSets", "train.txt")
        elif self.split == "test":
            imageset_txt = os.path.join(root, "imageSets", "test.txt")
        elif self.split == "validation":
            imageset_txt = os.path.join(root, "imageSets", "validation.txt")
        image_files = []
        for line in open(imageset_txt, "r"):
            image_name = line.replace("\n", "")
            image_files.append(image_name)
            # image_name_2 = image_name.replace("RGB_500","RGB_1000")
            # image_files.append(image_name_2)
            # image_name_3 = image_name.replace("RGB_500", "RGB_2000")
            # image_files.append(image_name_3)
            # image_name_4 = image_name.replace("RGB_500", "RGB_4000")
            # image_files.append(image_name_4)
        self.image_files = image_files
        self.label_files = [i.replace(".jpg", ".txt") for i in self.image_files]
        self.num_samples = len(self.image_files)

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[1], box[0], box[3], box[2]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        original_idx = self.image_files[idx].replace(".jpg", "")
        img_path = os.path.join(self.image_dir + self.image_files[idx])
        img = cv2.imread(img_path)
        # default_resolution = [1601, 400]  # [375, 1242]
        # height, width = img.shape[0], img.shape[1]
        # print("Before pre-reshape: ", height, width)
        # transform to 512 * 512

        # reshape image to default size (some samples have slightly different size)
        height, width = img.shape[0], img.shape[1]
        input_h, input_w = self.input_h, self.input_w
        scale_h, scale_w = input_h / height, input_w / width

        inp = cv2.resize(img, (input_w, input_h))
        # scale_h, scale_w = input_h / height, input_w / width
        inp = (inp.astype(np.float32) / 255.)
        inp = inp.transpose(2, 0, 1)
        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio
        num_classes = self.num_classes
        if self.split in ["test", "validation", "train"]:
            res = {'image': img, \
                   'input': inp, \
                   'index': idx, \
                   'ori_index': original_idx}
            return res
        # get the labels
        label_path = os.path.join(self.label_dir + self.label_files[idx])
        with open(label_path) as f:
            content = f.readlines()
        content = [x.split() for x in content]
        # print(content)

        draw_gaussian = draw_umich_gaussian
        draw_ind = draw_umich_ind
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cls = np.zeros((self.max_objs), dtype=np.float32)
        # con_signal = np.zeros((max_objs), dtype=np.int16)
        # wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        h_ud = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_y = np.zeros((self.max_objs, 1), dtype=np.float32)
        ind = np.zeros((self.max_objs, 2), dtype=np.int64)
        # ind_hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

        count = 0
        for c in content:
            # if (c[0] == "Car"):
            # pdb.set_trace()
            bbox = np.array(c[1:5], dtype="float32")
            bbox = self._coco_box_to_bbox(bbox)
        res = {'image': img, \
               'input': inp, \
               'hm': hm, \
               'reg_mask': reg_mask, \
               'ind': ind, \
               # 'ind_hm': ind_hm, \
               # 'wh': wh, \
               'h_ud': h_ud, \
               # 'reg': reg, \
               'reg_y': reg_y, \
               'cl': cls, \
               # 'con_id':con_signal, \
               'index': idx, \
               'ori_index': original_idx}

        return res


if __name__ == "__main__":
    im_idx = 0
    my_dataset = ctDataset(split="validation")
    res = my_dataset.__getitem__(im_idx)
    # pdb.set_trace()
    img = res['image']
    inp = res['input'].transpose(1, 2, 0)
    plt.title("Original Image")
    plt.imshow(img)
    plt.show()

    plt.title("transform Image")
    plt.imshow(inp)
    plt.show()
