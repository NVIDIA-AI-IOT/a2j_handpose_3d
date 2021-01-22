'''
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import cv2
import json
import random
import imutils
import numpy as np

from glob import glob
from scipy.io import loadmat
from torch.utils.data import Dataset

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const
from dataloader.pre_process import preprocess

class A2J_NYU_DataLoader(Dataset):
    def __init__(self, train=True, dataset_path=const.DATASET_PATH, target_size=const.TARGET_SIZE, \
                depth_threshold=const.DEPTH_THRESHOLD, num_joints=const.NUM_JOINTS, joint_list=const.JOINT_LIST, \
                rand_crop_shift=const.RAND_CROP_SHIFT, rand_rotate=const.RANDOM_ROTATE, rand_scale=const.RAND_SCALE):
        """
        Class initilizer

        :param img_name_list: None
        :param dataset_path: the full path to the image directory
        :param target_size: tuple, the input size of the nework
        """
        if train:
            self.dataset_path = os.path.join(dataset_path, "train")
        else:
            self.dataset_path = os.path.join(dataset_path, "test")

        if const.CAMERA_VIEW == "ALL":
            self.img_name_list = glob(f"{self.dataset_path}/depth*.png")
        else:
            self.img_name_list = glob(f"{self.dataset_path}/depth_1_*.png")
        
        joint_path = os.path.join(self.dataset_path, "joint_data.mat")
        self.joint = loadmat(joint_path)["joint_uvd"]
        
        if const.CAMERA_VIEW == "ALL":
            self.length = 0
            for i in self.joint:
                self.length += len(i)
            assert self.length == len(self.img_name_list)

        self.target_size = target_size
        self.joint_list = joint_list
        self.train = train
        self.depth_threshold = depth_threshold
        self.num_joints = num_joints
        self.rand_crop_shift = rand_crop_shift
        self.rand_rotate = rand_rotate
        self.rand_scale = rand_scale
    
    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        """
        Return
        :img: training transformed image
        :joint: transformed joints wrt img
        :xy_location: hand boundingbox
        :depth_img: original depth_img
        :joints: original joint annotations
        :median: the median depth of the hand
        """
        dataset_path = self.img_name_list[idx]
        img_name = dataset_path.split("/")[-1].split(".")[0]
        depth_img = cv2.imread(dataset_path)
        new_depth_img = np.left_shift(depth_img[:,:,1].astype(np.uint16), 8) + depth_img[:,:,0].astype(np.uint16) 
        
        camera_num = int(img_name.split("_")[1])
        frame_num = int(img_name.split("_")[-1].split(".")[0])   

        joints = self.joint[camera_num-1][frame_num-1][self.joint_list]
        x_min = max(0, int(joints[:,0:1].min()))
        x_max = min(int(joints[:,0:1].max()), new_depth_img.shape[1])
        y_min = max(0, int(joints[:,1:2].min()))
        y_max = min(int(joints[:,1:2].max()), new_depth_img.shape[0])
        median = (np.median(new_depth_img.copy()[y_min:y_max, x_min:x_max])).item()
        
        img, joint, xy_location, median = preprocess(new_depth_img, joints, median, self.target_size, \
                                            depth_thresh=self.depth_threshold, num_joints=self.num_joints , augment=self.train,\
                                            rand_crop_shift=self.rand_crop_shift , rand_rotate=self.rand_rotate, rand_scale=self.rand_scale)
        
        return img, joint, xy_location, new_depth_img.astype(float), joints, median, img_name
