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
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_PATH, os.path.pardir)
ROOT_PATH = os.path.join(MODEL_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# Import Project Libraries
import pipeline.constants as const



def generate_anchors(p_h=None, p_w=None):
    """
    Generate anchor shape

    :param p_h: anchor hieght layout
    :param p_w: anchor width layout
    """
    if p_h is None:
        p_h = np.array([2, 6, 10, 14])
    
    if p_w is None:
        p_w = np.array([2, 6, 10, 14])
    
    num_anchors = len(p_h) * len(p_w)

    # Initialize the anchor points
    k = 0
    anchors = np.zeros((num_anchors, 2))
    for i in range(len(p_w)):
        for j in range(len(p_h)):
            anchors[k,1] = p_w[j]
            anchors[k,0] = p_h[i]
            k += 1
    return anchors

def shift(shape, stride, anchor):
    """
    Create the locations of all the anchonrs in the in put image

    :param shape: common trunk (H, W)
    :param stride: the downsampling factor from input to common trunk
    :param anchor: anchor 
    """
    shift_h = np.arange(0, shape[0]) * stride # (shape[0]) 10
    shift_w = np.arange(0, shape[1]) * stride # (shape[1]) 9

    shift_h, shift_w = np.meshgrid(shift_h, shift_w) # (shape[1], shape[0]) (9, 10), (shape[1], shape[0]) (9, 10)
    shifts = np.vstack( (shift_h.ravel(), shift_w.ravel()) ).transpose() # (shape[0]*shape[1], 2) (90, 2)

    A = anchor.shape[0] # 16
    K = shifts.shape[0] # (shape[0]*shape[1]) (90)

    all_anchors = (anchor.reshape(1,A,2) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))) # (shape[0]*shape[1], A, 2)
    all_anchors = all_anchors.reshape((K*A, 2)) # (shape[0]*shape[1]*A, 2)
    return all_anchors


class A2JLoss(nn.Module):
    """
    A2J loss class
    """
    def __init__(self, p_h=None, p_w=None, shape=[const.TARGET_SIZE[1]//16, const.TARGET_SIZE[0]//16],\
                    stride=const.STRIDE, spacial_factor=0.1):
        """
        Class constructor

        :param p_h:
        :param p_w:
        :param shape: common trunk (H, W)
        :param stride: the downsampling factor from input to common trunk
        :param spacial_factor: regression loss spacial factor
        """
        super(A2JLoss, self).__init__()
        anchors = generate_anchors(p_h=p_h, p_w=p_w)
        self.all_anchors = torch.from_numpy(shift(shape, stride, anchors)).float()
        self.spacial_factor = spacial_factor

    def forward(self, joint_classifications, offset_regressions, depth_regressions, annotations):
        """
        forward pass through the module

        :param joint_classifications: type torch.tensor, joint classification output of the model  -->  (N, shape[0]*shape[1]*anchor_stride, num_joints)
        :param offset_regressions:  type torch.tensor, offset regression output of the model  -->  (N, 2*shape[0]*shape[1]*anchor_stride, num_joints)
        :param depth_regressions:  type torch.tensor, depth rgression output of the model  -->  (N, shape[0]*shape[1]*anchor_stride, num_joints)
        :param annotations:  type torch.tensor, true joint annotations  -->  (N, num_joints, 3)
        """
        DEVICE = joint_classifications.device
        
        batch_size = joint_classifications.shape[0]
        anchor = self.all_anchors.to(DEVICE) # (shape[0]*shape[1]*anchor_stride, 2) (1440, 2)
        regression_losses = list()
        anchor_reg_losses = list()

        for i in range(batch_size):
            joint_classification = joint_classifications[i] # (shape[0]*shape[1]*anchor_stride, num_joints) (1440, 18)
            offset_regression = offset_regressions[i] # (shape[0]*shape[1]*anchor_stride, num_joints, 2) (1440, 18, 2)
            depth_regression = depth_regressions[i] # (shape[0]*shape[1]*anchor_stride, num_joints) (1440, 18)

            annotation = annotations[i] # (num_joints, 3)

            reg_weight = F.softmax(joint_classification, dim=0) # (shape[0]*shape[1]*anchor_stride, num_joints) (1440, 18)

            reg_weight_xy = reg_weight.unsqueeze(2).expand(reg_weight.shape[0], reg_weight.shape[1], 2).to(DEVICE) # (shape[0]*shape[1]*anchor_stride, num_joints, 2) (1440, 18, 2)

            annotation_xy = annotation[:,:2] # (num_joints, 2)

            anchor_diff_xy = torch.abs(annotation_xy - (reg_weight_xy * anchor.unsqueeze(1).to(DEVICE)).sum(0)) # (num_joints, 2)

            anchor_loss = torch.where(
                torch.le(anchor_diff_xy, 1),
                0.5 * 1* torch.pow(anchor_diff_xy, 2),
                anchor_diff_xy - 0.5/1
            )

            anchor_reg_loss = anchor_loss.mean()
            anchor_reg_losses.append(anchor_reg_loss)

            ######### Regression Losses for spacial #########

            # xy_regression: is the location of each anchor point + the offset
            # offset_regression: is giving us the offset
            xy_regression = anchor.unsqueeze(1).to(DEVICE) + offset_regression # (shape[0]*shape[1]*anchor_stride, 2) (1440, 18, 2)
            regression_diff = torch.abs(annotation_xy - (reg_weight_xy * xy_regression).sum(0)) # (num_joints, 2)

            regression_loss = torch.where(
                torch.le(regression_diff, 1),
                0.5 * 1 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 1,
            )

            regression_loss = regression_loss.mean() * self.spacial_factor

            annotation_z = annotation[:,2] # (num_joints, 1)
            depth_regression_diff = torch.abs(annotation_z - (reg_weight*depth_regression).sum(0)) # (num_joints, 1)
            
            regression_loss_depth = torch.where(
                torch.le(depth_regression_diff, 3),
                0.5 * (1/3) * torch.pow(depth_regression_diff, 2),
                depth_regression_diff - 0.5 / (1/3),
            )
            regression_loss += regression_loss_depth.mean()

            regression_losses.append(regression_loss)
        
        return torch.stack(anchor_reg_losses).mean(dim=0, keepdim=True),\
                    torch.stack(regression_losses).mean(dim=0, keepdim=True)


class Summary(object):
    def __init__(self):
        self._reset_summary()
    
    def _reset_summary(self):
        self.maximum = 0
        self.minimum = 0
        self.value = 0
        self.count = 0
        self.sum = 0
        self.avg = 0
    
    def update(self, value, count=1):
        self.value = value
        self.sum += value
        self.count += count
        if self.minimum > value:
            self.minimum = value
        if self.maximum < value:
            self.maximum = value
        self.avg = self.sum / self.count

class SummaryList(object):
    def __init__(self):
        self._reset_summary()
    
    def _reset_summary(self):
        self.values = []
    
    def update(self, value, count=1):
        self.values.append(value)
