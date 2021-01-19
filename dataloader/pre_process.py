import os
import sys
import cv2
import torch
import numpy as np

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const

MEAN = -0.66877532
STD = 28.32958208

def transform(img, joints, matrix, num_joints=const.NUM_JOINTS, target_size=const.TARGET_SIZE):
    """
    Rotating the Image and Joints Xy location with repect to the Matrix

    img: (H, W)
    Joints: (18, 2)
    Matrix: 2x3
    """
    img_out = cv2.warpAffine(img, matrix, target_size)
    joints_out = np.ones((num_joints, 3))
    joints_out[:,:2] = joints[:,:2].copy()
    joints_out = np.matmul(matrix, joints_out.transpose())
    joints_out = joints_out.transpose()
    return img_out, joints_out


# height : 160, width : 144 --> target_size = (width, height)
def preprocess(img, joints, median, target_size=const.TARGET_SIZE, depth_thresh=const.DEPTH_THRESHOLD, num_joints=const.NUM_JOINTS, \
                augment=False, rand_crop_shift=const.RAND_CROP_SHIFT, rand_rotate=const.RANDOM_ROTATE, rand_scale=const.RAND_SCALE):

    img_output = np.ones((target_size[1], target_size[0], 1), dtype="float32")
    joint_output = np.ones((num_joints, 3), dtype="float32")

    if augment:
        random_offset_1 = np.random.randint(-1*rand_crop_shift, rand_crop_shift)
        random_offset_2 = np.random.randint(-1*rand_crop_shift, rand_crop_shift)
        random_offset_3 = np.random.randint(-1*rand_crop_shift, rand_crop_shift)
        random_offset_4 = np.random.randint(-1*rand_crop_shift, rand_crop_shift)
        random_offset_depth = np.random.randint(-1*40, 40)
        random_rotate = np.random.randint(-1*rand_rotate, rand_rotate)
        random_scale = np.random.rand()*rand_scale[0]+rand_scale[1]

    else:
        random_offset_1, random_offset_2, random_offset_3, random_offset_4 = 0, 0, 0, 0
        random_offset_depth = 0
        random_rotate = 0
        random_scale = 1 

    joints = np.array(joints)
    if median < joints[:,2:3].min():
        median = np.median(joints[:,2:3]) + 40
    else:
        median = median 

    joints = np.array(joints)
    x = joints[:,0:1] 
    y = joints[:,1:2] 
    z = joints[:,2:3]

    if augment and np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
        width = img.shape[1]
        x = width - x 

    left_top = [x.min(), y.min()]
    right_bottom = [x.max(), y.max()]

    center = [(left_top[0]+right_bottom[0])/2, (left_top[1]+right_bottom[1])/2]
    left_top = [center[0]-110, center[1]-110]
    right_bottom = [center[0]+110, center[1]+110]
    
    matrix = cv2.getRotationMatrix2D((target_size[0]/2, target_size[1]/2), random_rotate, random_scale)
    
    new_Xmin = int(max(left_top[0] + random_offset_1, 0))
    new_Ymin = int(max(left_top[1] + random_offset_2, 0))
    new_Xmax = int(min(right_bottom[0] + random_offset_3, img.shape[1]-1))
    new_Ymax = int(min(right_bottom[1] + random_offset_4, img.shape[0]-1))

    img_crop = img[new_Ymin:new_Ymax, new_Xmin:new_Xmax].copy()

    img_resize = cv2.resize(img_crop, target_size, interpolation=cv2.INTER_NEAREST)
    img_resize = np.asarray(img_resize, dtype="float32")
    
    median_depth = median + random_offset_depth 
    img_resize[np.where(img_resize >= median_depth + depth_thresh)] = median_depth 
    img_resize[np.where(img_resize <= median_depth - depth_thresh)] = median_depth
    img_resize = (img_resize - median_depth)*random_scale
    img_resize = (img_resize - MEAN)/STD

    # joints
    joints_xy = np.ones((num_joints, 2), dtype="float32")
    joints_xy[:,0] = (x[:,0].copy() - new_Xmin)*target_size[1]/(abs(new_Xmax - new_Xmin))
    joints_xy[:,1] = (y[:,0].copy() - new_Ymin)*target_size[0]/(abs(new_Ymax - new_Ymin))

    if augment:
        img_resize, joints_xy = transform(img_resize, joints_xy, matrix)
    
    img_output[:,:,0] = img_resize

    joint_output[:,1] = joints_xy[:,0]
    joint_output[:,0] = joints_xy[:,1]
    joint_output[:,2] = (joints[:,2] - median_depth)*random_scale

    img_output = np.asarray(img_output)
    img_NCHW_out = img_output.transpose(2, 0, 1)
    img_NCHW_out = np.asarray(img_NCHW_out)
    joint_output = np.asarray(joint_output)

    img_out, joint_out = torch.from_numpy(img_NCHW_out), torch.from_numpy(joint_output)
    return img_out, joint_out, torch.tensor([new_Xmin, new_Ymin, new_Xmax, new_Ymax]), median_depth


#####################
#### TESTING
#####################
"""
import matplotlib.pyplot as plt
from scipy.io import loadmat
from glob import glob

img_path=const.DEPTH_IMG_PATH
joint_path=const.JOINT_JSON_PATH
target_size=const.TARGET_SIZE
depth_threshold=const.DEPTH_THRESHOLD
num_joints=const.NUM_JOINTS
rand_crop_shift=const.RAND_CROP_SHIFT
rand_rotate=const.RANDOM_ROTATE
rand_scale=const.RAND_SCALE
       
img_path = os.path.join(img_path, "train")

img_name_list = glob(f"{img_path}/depth_1*.png")
joint_path = os.path.join(img_path, "joint_data.mat")
joint = loadmat(joint_path)["joint_uvd"]
img_name_list.sort()

target_size = target_size

train = True
depth_threshold = depth_threshold
num_joints = num_joints
rand_crop_shift = rand_crop_shift
rand_rotate = rand_rotate
rand_scale = rand_scale

img_path = img_name_list[0]
img_name = img_path.split("/")[-1].split(".")[0]
depth_img = cv2.imread(img_path)
new_depth_img = np.left_shift(depth_img[:,:,1].astype(np.uint16), 8) + depth_img[:,:,0].astype(np.uint16) 

camera_num = int(img_name.split("_")[1])
frame_num = int(img_name.split("_")[-1].split(".")[0])   

joints = joint[camera_num-1][frame_num-1]
x_min = max(0, int(joints[:,0:1].min()))
x_max = min(int(joints[:,0:1].max()), new_depth_img.shape[1])
y_min = max(0, int(joints[:,1:2].min()))
y_max = min(int(joints[:,1:2].max()), new_depth_img.shape[0])
z_min = max(0, int(joints[:,2:3].min()))
z_max = int(joints[:,2:3].max())

median = np.median(new_depth_img[y_min:y_max, x_min:x_max])

img, joint, xy_location, median = preprocess(new_depth_img, joints, median, target_size, \
                                    depth_thresh=depth_threshold, num_joints=num_joints , augment=train,\
                                    rand_crop_shift=rand_crop_shift , rand_rotate=rand_rotate, rand_scale=rand_scale)

img = img.numpy()[0]
joint = joint.numpy()

fig = plt.figure(figsize=(6, 8))
ax1 = fig.add_subplot(2,1,1)
ax1.imshow(new_depth_img)


ax2 = fig.add_subplot(2,1,2)
ax2.scatter(x=joint[:,1:2], y=joint[:,0:1], c="r", s=20)
ax2.imshow(img)

plt.show()
"""