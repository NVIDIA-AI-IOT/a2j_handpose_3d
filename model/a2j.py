import os
import sys
import torch.nn as nn

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# Import Project Library
from model.back_bone.resnet import ResnetBackbone
from model.back_bone.mobilenet import MobileNet
from model.a2j_utilities.a2j_branchs import DepthRegression, OffsetRegression, JointClassification

A2J_BACKBONE_CONFIG = {
    "resnet18": {"backbone": ResnetBackbone, "common_trunk": 256, "Regression_trunk": 512},
    "resnet34": {"backbone": ResnetBackbone, "common_trunk": 256, "Regression_trunk": 512},
    "resnet50": {"backbone": ResnetBackbone, "common_trunk": 1024, "Regression_trunk": 2048},
    "resnet101": {"backbone": ResnetBackbone, "common_trunk": 1024, "Regression_trunk": 2048},
    "resnet152": {"backbone": ResnetBackbone, "common_trunk": 1024, "Regression_trunk": 2048},
    "mobilenet": {"backbone": MobileNet, "common_trunk": 512, "Regression_trunk": 1024},
}

class A2J(nn.Module):
    """
    A2J model class
    """
    def __init__(self, num_joints=21, backbone_name="resnet18", backbone_pretrained=True):
        """
        Class constructor

        :param num_joints: number of joints to predict
        :param backbone_name: the name of the backbone network
        :param backbone_pretrained: load a pretrained backbone network
        """
        super(A2J, self).__init__()
        Backbone_Model = A2J_BACKBONE_CONFIG[backbone_name]["backbone"]
        self.back_bone = Backbone_Model(name=backbone_name, pretrained=backbone_pretrained)

        self.offset_regression = OffsetRegression(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["Regression_trunk"], num_joints=num_joints)
        self.depth_regression = DepthRegression(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["Regression_trunk"], num_joints=num_joints)
        self.joint_classification = JointClassification(input_channels=A2J_BACKBONE_CONFIG[backbone_name]["common_trunk"], num_joints=num_joints)
    
    
    def forward(self, x):
        out3, out4 = self.back_bone(x)
        offset_regression = self.offset_regression(out4)
        depth_regression = self.depth_regression(out4)
        joint_classification = self.joint_classification(out3)

        return joint_classification, offset_regression, depth_regression