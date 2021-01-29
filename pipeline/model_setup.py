'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import torch
import random
from glob import glob
from torch.utils.data import DataLoader, sampler

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const
from model.a2j import A2J
from model.a2j_utilities.post_processing import PostProcess
from model.a2j_utilities.a2j_utils import A2JLoss

from dataloader.nyuDataLoader import A2J_NYU_DataLoader

random.seed(0)

class ModelSetup(object):
    def __init__(self, dataset_path=const.DATASET_PATH, train=True, test=False, load=None,\
                    num_classes=const.NUM_JOINTS, backbone_name=const.BACKBONE_NAME, backbone_pre_trained=const.PRE_TRAINED,\
                    target_size=const.TARGET_SIZE, stride=const.STRIDE, p_h=None, p_w=None, spacial_factor=const.SPACIAL_FACTOR,\
                    lr=const.LR_RATE, w_d=const.WEIGHT_DECAY, step_size=const.STEP_SIZE, gamma=const.GAMMA, reg_loss_fac=const.REG_LOSS_FACTOR,\
                    bs=const.BATCH_SIZE, max_epoch=const.MAX_EPOCH, save_path=const.SAVE_PATH, save_freq=const.SAVE_FREQ, train_spit=const.TRAIN_VAL_SPLIT):

        print("Setting up model...")
        self.dataset_path = dataset_path
        self.save_path = save_path

        self.num_classes = num_classes
        self.backbone_name = backbone_name

        self.train = train
        self.test = test

        self.max_epoch = max_epoch
        self.save_freq = save_freq

        self.model = A2J(num_joints=num_classes, backbone_name=backbone_name, backbone_pretrained=backbone_pre_trained)
        self.reg_loss_factor = reg_loss_fac

        self.post_precess = PostProcess(shape=(target_size[1]//16, target_size[0]//16),stride=stride, p_h=p_h, p_w=p_w)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=w_d)
        self.sched = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=gamma)
        self.criterion = A2JLoss(shape=(target_size[1]//16, target_size[0]//16), stride=stride,\
                                                    spacial_factor=spacial_factor, p_h=p_h, p_w=p_w)
        if load:
            print(f"Loading model...{load}")
            check_point = torch.load(load, map_location=torch.device('cpu'))
            self.num_class = check_point["num_classes"]
            self.model = A2J(num_joints=num_classes, backbone_name=backbone_name, backbone_pretrained=backbone_pre_trained)
            self.model.load_state_dict(check_point["model"])
            self.optim.load_state_dict(check_point["optim"])
            self.sched.load_state_dict(check_point["sched"])
            self.epoch = check_point["epoch"]+1
        else:
            self.epoch = 0

        self.train_data = A2J_NYU_DataLoader(train=True, dataset_path=self.dataset_path)
        self.valid_data = A2J_NYU_DataLoader(train=False, dataset_path=self.dataset_path)
        self.test_data = A2J_NYU_DataLoader(train=False, dataset_path=self.dataset_path)

        self.load_train = DataLoader(
            self.train_data,
            batch_size=bs,
            sampler=sampler.RandomSampler(self.train_data),
            drop_last=False,
            num_workers=8,
        )

        self.load_valid = DataLoader(
            self.valid_data,
            batch_size=bs//4,
            sampler=sampler.RandomSampler(self.valid_data),
            drop_last=False,
            num_workers=8,
        )

        self.load_test = DataLoader(
            self.test_data,
            batch_size=bs//8,
            sampler=sampler.RandomSampler(self.test_data),
            drop_last=False,
            num_workers=8,
        )
        print("Model setup finished!")

    def save(self, epoch):
        save_path = self.save_path
        if not os.path.isdir(save_path):
            # os.mkdir(save_path)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

        save_path += f"/{const.DATASET}_{const.CAMERA_VIEW}_{self.backbone_name}_{self.num_classes}_a2j.pth"

        torch.save(dict(
            num_classes=self.num_classes,
            model=self.model.state_dict(),
            optim=self.optim.state_dict(),
            sched=self.sched.state_dict(),
            epoch=epoch,
        ), save_path)
        print(f"Model saved at {save_path}")
