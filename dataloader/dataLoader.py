import os
import sys
import cv2
import json
import random
import imutils
import numpy as np
from glob import glob
from torch.utils.data import Dataset

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const
from dataloader.pre_process import preprocess

class A2J_DataLoader(Dataset):
    def __init__(self, img_name_list, train=True, \
                img_path=const.DEPTH_IMG_PATH, joint_path=const.JOINT_JSON_PATH, target_size=const.TARGET_SIZE, \
                depth_threshold=const.DEPTH_THRESHOLD, num_joints=const.NUM_JOINTS, rand_crop_shift=const.RAND_CROP_SHIFT,\
                rand_rotate=const.RANDOM_ROTATE, rand_scale=const.RAND_SCALE):
        """
        Class initilizer

        :param img_name_list: a list of the name of the images that will be used for (training/validation)
        :param img_path: the full path to the image directory 
        :param joint_path: the full path to the stored joints directory
        :param target_size: tuple, the input size of the nework
        """
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.joint_path = joint_path
        self.target_size = target_size
        
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
        
        img_name = os.path.join(self.img_path, f"{self.img_name_list[idx]}.png")
        depth_img = cv2.imread(img_name, cv2.COLOR_BGR2GRAY).astype(np.uint16)

        json_name = os.path.join(self.joint_path, f"{self.img_name_list[idx]}.json")
        with open(json_name, 'r') as jf:
            joint_info = json.load(jf)
        median = joint_info["median"]
        joints = np.array(joint_info["joints"])

        img, joint, xy_location, median = preprocess(depth_img, joints, median, self.target_size, \
                                            depth_thresh=self.depth_threshold, num_joints=self.num_joints , augment=self.train,\
                                            rand_crop_shift=self.rand_crop_shift , rand_rotate=self.rand_rotate, rand_scale=self.rand_scale)
        
        return img, joint, xy_location, depth_img.astype(float), joints, median