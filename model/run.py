'''
Copyright (c) 2019 Boshen Zhang
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import time
import torch
import numpy as np

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const
from pipeline.model_setup import ModelSetup
from model.a2j_utilities.a2j_utils import Summary, SummaryList


def back_to_normal(pred_joints, true_joints, xy_bb, median_depth, mean_error_list, max_error_list, train):
    xy_bb_all = xy_bb.detach().cpu().numpy()
    median_depth_all = median_depth.detach().cpu().numpy()

    true_joints = true_joints.cpu()
    all_true_joints = true_joints.detach().numpy()
    mean_err = []
    max_err = 0
    for i in range(len(pred_joints)):
        pred_joint = pred_joints[i].cpu()
        pred_joint = pred_joint.detach().numpy()
        true_joints = all_true_joints[i]

        xy_bb = xy_bb_all[i]
        median_depth = median_depth_all[i]

        p_j = np.ones((const.NUM_JOINTS, 3))
        true_joint = np.ones((const.NUM_JOINTS, 3))
        x_len = abs(xy_bb[0] - xy_bb[2])
        y_len = abs(xy_bb[1] - xy_bb[3])

        p_j[:,0] = ((pred_joint[:,1] * x_len) / const.TARGET_SIZE[0]) + xy_bb[0]
        p_j[:,1] = ((pred_joint[:,0] * y_len) / const.TARGET_SIZE[1]) + xy_bb[1]
        p_j[:,2] = pred_joint[:,2] + median_depth

        true_joint[:,0] = ((true_joints[:,1] * x_len) / const.TARGET_SIZE[0]) + xy_bb[0]
        true_joint[:,1] = ((true_joints[:,0] * y_len) / const.TARGET_SIZE[1]) + xy_bb[1]
        true_joint[:,2] = true_joints[:,2] + median_depth

        temp_max = np.sqrt((np.abs(p_j - true_joint)**2).sum(1).max())
        
        if temp_max > max_err:
            max_err = temp_max

        temp_mean = np.sqrt((np.abs(p_j - true_joint)**2).sum(1).mean())
        mean_err.append(temp_mean)
        
        if not train:
            max_error_list.update(temp_max)
            mean_error_list.update(temp_mean)
    
    if not mean_err:
        mean_err = [0]
    return sum(mean_err)/len(mean_err), max_err


def run_model(model_setup, train=True, test=False):
    if train:
        dataset = model_setup.load_train
        model_setup.model.train()
    else:
        dataset = model_setup.load_valid
        model_setup.model.eval()
    
    if test:
        dataset = model_setup.load_test
        model_setup.model.eval()  
    
    train_loss_avg = Summary()
    class_loss_avg = Summary()
    reg_loss_avg = Summary()
    iter_time_avg = Summary()
    mean_error = Summary()
    max_error = Summary()
    mean_error_list = SummaryList()
    max_error_list = SummaryList()
    start_time = time.time()

    transformed_image = torch.tensor([], dtype=torch.float)
    transformed_joints = torch.tensor([], dtype=torch.float)
    xy_boundingbox = torch.tensor([], dtype=torch.float)
    depth_images = torch.tensor([], dtype=torch.float)
    out_joints = torch.tensor([], dtype=torch.float)
    median_depths = torch.tensor([], dtype=torch.float)
    pred_joints = torch.tensor([], dtype=torch.float)

    if torch.cuda.is_available():
        model_setup.model = model_setup.model.cuda()  
        transformed_image = transformed_image.cuda()
        transformed_joints = transformed_joints.cuda()
        xy_boundingbox = xy_boundingbox.cuda()
        depth_images = depth_images.cuda()
        out_joints = out_joints.cuda()
        median_depths = median_depths.cuda()
        pred_joints = pred_joints.cuda()
        model_setup.criterion.cuda()
        model_setup.post_precess.cuda()

    for i, (img, joint, xy_locations, depth_img, joints, median_depth, img_name_list) in enumerate(dataset):
        iter_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            joints = joints.cuda()
            img = img.cuda()
            joint = joint.cuda()

        joint_classification, offset_regression, depth_regression = model_setup.model(img)
        pred_points = model_setup.post_precess(joint_classification, offset_regression, depth_regression)
        
        if train:
            model_setup.optim.zero_grad()
        class_loss, regression_loss = model_setup.criterion(joint_classification, offset_regression, depth_regression, joint)
        
        loss = 1*class_loss + regression_loss*model_setup.reg_loss_factor

        if train:
            loss.backward()
            model_setup.optim.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        train_loss_avg.update(loss.item())
        class_loss_avg.update(class_loss.item())
        reg_loss_avg.update(regression_loss.item())

        if test:
            t_image = torch.tensor(img, dtype=torch.float)
            t_joints = torch.tensor(joint, dtype=torch.float)
            xy_bb = torch.tensor(xy_locations, dtype=torch.float)
            d_img = torch.tensor(depth_img, dtype=torch.float)
            true_joints = torch.tensor(joints, dtype=torch.float)
            med_depth = torch.tensor(median_depth, dtype=torch.float)
            
            transformed_image = torch.cat((transformed_image.cpu(), t_image.cpu()))

            transformed_joints = torch.cat((transformed_joints.cpu(), t_joints.cpu()))
            
            xy_boundingbox = torch.cat((xy_boundingbox.cpu(), xy_bb.cpu()))
            
            depth_images = torch.cat((depth_images.cpu(), d_img.cpu()))
            
            out_joints = torch.cat((out_joints.cpu(), true_joints.cpu()))
            
            median_depths = torch.cat((median_depths.cpu(), med_depth.cpu()))
            
            pred_points = model_setup.post_precess(joint_classification, offset_regression, depth_regression)
            pred = torch.stack(pred_points)
            pred_joints = torch.cat((pred_joints.cpu(), pred.cpu()))
            
        stop_iter_time = time.time()
        iter_time_avg.update(stop_iter_time - iter_time)
        mean_err, max_err = back_to_normal(pred_points, joint, xy_locations, median_depth, mean_error_list, max_error_list, train)
        
        mean_error.update(mean_err)
        max_error.update(max_err)
        
        if (i+1)%10 == 0:
            print(f"epoch: {model_setup.epoch} [{i}/{len(dataset)}] : {i/len(dataset):1.2f},\
            	 classification loss: {class_loss_avg.avg:1.4f},	 regression loss: {reg_loss_avg.avg:1.4f},	 total loss: {train_loss_avg.avg:1.4f},\
            	 mean_err: {mean_error.avg: 1.2f},	 max_err: {max_error.maximum: 1.2f},	 time: {iter_time_avg.sum:1.2f},	 lr rate: {model_setup.sched.get_lr()[0]:1.9f}")
            iter_time_avg._reset_summary()
            
        if (i+1)%10==0 and test:
            break
            
    end_time = time.time()

    total_time = end_time - start_time

    out = dict(
        avg_loss = train_loss_avg.avg,
        avg_reg_loss = reg_loss_avg.avg,
        avg_class_loss = class_loss_avg.avg,
        max_err=max_error.maximum,
        mean_err=mean_error.avg,
    )

    if test:
        return out, transformed_image, transformed_joints, xy_boundingbox, depth_images, out_joints, pred_joints, median_depths
    elif train:
        return out
    else:
        return out, mean_error_list, max_error_list
