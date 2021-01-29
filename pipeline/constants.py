'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)

DATASET = "NYU"
DATASET_PATH = "/home/analog/Desktop/DataSets/NYU_dataset"

# Set the checkpoint directory
SAVE_PATH = ""
SAVE_PATH = os.path.join(ROOT_PATH, "check_point")


# Select which backbone network you want to train with
BACKBONE_NAME_SWITCHER = {
    "resnet18": False,
    "resnet34": False,
    "resnet50": True,
    "resnet101": False,
    "resnet152": False,
    "mobilenet": False,
}

BACKBONE_NAME = [elem[0] for idx, elem in enumerate(BACKBONE_NAME_SWITCHER.items()) if elem[1]][0]
PRE_TRAINED = True # Set to true for pre trained backbone on ImageNet

# NYU Dataset has 3 sets of cameras:
#   1: corresponds to the front view camera
#   ALL: corresponds to all 3 different views
# 
# Authors of A2J use the front view Camera
CAMERA_VIEW = "1" # ALL, 1

NUM_JOINTS = 16 # 16, (36 Total NYU). You can set it ither to 16 or 36
JOINT_LIST = [0, 2, 5, 6, 8, 11, 12, 14, 17, 18, 20, 23, 24, 26, 28, 34]
if NUM_JOINTS == 36:
    JOINT_LIST = [i for i in range(36)]
"""
IF 16 Joints is chosen
pinky 0, 2, 5
ring 6, 8, 11
middle 12, 14, 17
index 18, 20, 23
thumb 24, 26, 28
palm 34
"""

TARGET_SIZE = (176, 176)
DEPTH_THRESHOLD = 180
RAND_CROP_SHIFT = 5
RANDOM_ROTATE = 180
RAND_SCALE = (1., 0.5)

TRAIN_VAL_SPLIT = 90
SAVE_FREQ = 1
MAX_EPOCH = 32

BATCH_SIZE = 64
STRIDE = 16

LR_RATE = 0.00036
# LR_RATE = 0.00035
WEIGHT_DECAY = 1e-4
STEP_SIZE = 10

GAMMA = 0.2
SPACIAL_FACTOR = 0.5
REG_LOSS_FACTOR = 3
