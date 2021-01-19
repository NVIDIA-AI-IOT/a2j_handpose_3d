import os
import sys
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

import pipeline.constants as const
from pipeline.utils import * 
from pipeline.model_setup import ModelSetup
from model.run import run_model
# from visualizer.create_point_cloud import CreatePointCloud

def back_to_normal(pred_joints, true_joints, xy_bb, median_depth):
    pred_joints = pred_joints.detach().numpy()
    true_joints = true_joints.detach().numpy()
    xy_bb = xy_bb.detach().numpy()
    median_depth = median_depth.detach().numpy()

    pred_joint = np.ones((const.NUM_JOINTS, 3))
    true_joint = np.ones((const.NUM_JOINTS, 3))
    x_len = abs(xy_bb[0] - xy_bb[2])
    y_len = abs(xy_bb[1] - xy_bb[3])

    pred_joint[:,0] = ((pred_joints[:,1] * x_len) / const.TARGET_SIZE[0]) + xy_bb[0]
    pred_joint[:,1] = ((pred_joints[:,0] * y_len) / const.TARGET_SIZE[1]) + xy_bb[1]
    pred_joint[:,2] = pred_joints[:,2] + median_depth

    true_joint[:,0] = ((true_joints[:,1] * x_len) / const.TARGET_SIZE[0]) + xy_bb[0]
    true_joint[:,1] = ((true_joints[:,0] * y_len) / const.TARGET_SIZE[1]) + xy_bb[1]
    true_joint[:,2] = true_joints[:,2] + median_depth

    return pred_joint, true_joint

def pick_points(pcd, pred_joints_point_cloud, true_joints_point_cloud):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pred_joints_point_cloud)
    vis.add_geometry(true_joints_point_cloud)
    vis.add_geometry(pcd)
    vis.run()# user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


"""
def visualize_point_cloud(depth_img, pred_joint, true_joint, median_depth):
    depth_img = depth_img.detach().numpy()
    cpc = CreatePointCloud()
    point_cloud_data, true_joint, pred_joint = cpc.create_point_cloud_from_2D(depth_img, true_joint, pred_joint)

    pred_color = [[255, 0, 0] for i in pred_joint]
    true_color = [[0, 255, 0] for i in true_joint]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    pred_joints_point_cloud = o3d.geometry.PointCloud()
    pred_joints_point_cloud.points = o3d.utility.Vector3dVector(pred_joint)
    pred_joints_point_cloud.colors = o3d.utility.Vector3dVector(pred_color)

    true_joints_point_cloud = o3d.geometry.PointCloud()
    true_joints_point_cloud.points = o3d.utility.Vector3dVector(true_joint)
    true_joints_point_cloud.colors = o3d.utility.Vector3dVector(true_color)

    points = pick_points(pcd, pred_joints_point_cloud, true_joints_point_cloud)
    import pdb; pdb.set_trace()

    o3d.visualization.draw_geometries([pcd, pred_joints_point_cloud, true_joints_point_cloud])
"""

def draw_img(img, pred_joint, true_joint):
    img = img.detach().numpy()
    pred_joint = pred_joint.detach().numpy()
    true_joint = true_joint.detach().numpy()

    fig, ax = plt.subplots(1)

    ax.imshow(img[0], cmap="hot", interpolation='nearest')
    ax.scatter(x=pred_joint[:,1], y=pred_joint[:,0], c="b", s=20)
    ax.scatter(x=true_joint[:,1], y=true_joint[:,0], c="g", s=20)
    plt.show()

def draw_img_normal(img, pred_joint, true_joint, median):
    """
    :param img: torch tensor of an image (C, H, W)
    :param pred_joint: numpy array of predicted joint
    :param true_joint: numpy  array of True joint
    :param median: integer of the median depth
    """
    img = img.detach().numpy()

    fig, ax = plt.subplots(1)

    palette = plt.cm.jet_r
    palette.set_bad(color='black')
    img = np.ma.masked_where(img < 0.05, img)
    ax.imshow(img, cmap=palette, interpolation='nearest', vmin=median-const.DEPTH_THRESHOLD, vmax=median+const.DEPTH_THRESHOLD)

    ax.scatter(x=pred_joint[:,0], y=pred_joint[:,1], c="b", s=20)
    ax.scatter(x=true_joint[:,0], y=true_joint[:,1], c="g", s=20)
    plt.show()

def vizualize_frams(ax_1, ax_2, depth_image, pred_joints, true_joints, median, out_joints):
    pred_joints = np.array(pred_joints)
    true_joints = np.array(true_joints)

    max_range = [0, 0, 0]  
    min_range = [float("inf"), float("inf"), float("inf")] 

    min_x = true_joints[:,0].min()
    if min_x < min_range[0]:  
        min_range[0] = min_x  
    min_y = true_joints[:,1].min()
    if min_y < min_range[1]:  
        min_range[1] = min_y  
    min_z = true_joints[:,2].min()
    if min_z < min_range[2]:  
        min_range[2] = min_z  

    max_x = true_joints[:,0].max()
    if max_x > max_range[0]:  
        max_range[0] = max_x  
    max_y = true_joints[:,1].max()
    if max_y > max_range[1]:  
        max_range[1] = max_y  
    max_z = true_joints[:,2].max()
    if max_z > max_range[2]:  
        max_range[2] = max_z

    mid_x = (max_range[0] + min_range[0])/2 
    mid_y = (max_range[1] + min_range[1])/2 
    mid_z = (max_range[2] + min_range[2])/2

    # fig = plt.figure(figsize=(6,10))

    # ax_1 = fig.add_subplot(2,1,1)
    depth_image = depth_image.detach().numpy()

    palette = plt.cm.jet_r
    palette.set_bad(color='black')
    depth_image = np.ma.masked_where(depth_image < 0.05, depth_image)
    ax_1.scatter(x=out_joints[:,0:1], y=out_joints[:,1:2], c="r", s=20)
    ax_1.imshow(depth_image, cmap=palette, interpolation='nearest', vmin=median-const.DEPTH_THRESHOLD, vmax=median+const.DEPTH_THRESHOLD)

    ax_1.set_yticklabels([]) 
    ax_1.set_xticklabels([])


    # Second subplot
    # ax_2 = fig.add_subplot(111, projection="3d")
    # ax_2 = fig.add_subplot(2, 1, 2, projection='3d')
    ax_2.grid(True)
    ax_2.set_xticklabels([])
    ax_2.set_yticklabels([]) 
    ax_2.set_zticklabels([])

    ax_2.set_xlim(mid_x - max_range[0]/2, mid_x + max_range[0]/2) 
    ax_2.set_ylim(mid_y - max_range[1]/2, mid_y + max_range[1]/2) 
    ax_2.set_zlim(mid_z - max_range[2]/2, mid_z + max_range[2]/2) 
    scat_1 = ax_2.scatter(true_joints[:,0], true_joints[:,1], true_joints[:,2], c='g', marker='o', s=20) 
    scat_2 = ax_2.scatter(pred_joints[:,0], pred_joints[:,1], pred_joints[:,2], c='r', marker='^', s=20) 
    # ax_2.plot(pred_joints[0:4,0], pred_joints[0:4,1], pred_joints[0:4,2], color='b')
    # ax_2.plot(pred_joints[4:8,0], pred_joints[4:8,1], pred_joints[4:8,2], color='b')
    # ax_2.plot(pred_joints[8:12,0], pred_joints[8:12,1], pred_joints[8:12,2], color='b')
    # ax_2.plot(pred_joints[12:16,0], pred_joints[12:16,1], pred_joints[12:16,2], color='b')
    # ax_2.plot(pred_joints[16:20,0], pred_joints[16:20,1], pred_joints[16:20,2], color='b')
    
    # ax_2.plot([pred_joints[3,0], pred_joints[20,0]], [pred_joints[3,1], pred_joints[20,1]], [pred_joints[3,2], pred_joints[20,2]], color='b')
    # ax_2.plot([pred_joints[7,0], pred_joints[20,0]], [pred_joints[7,1], pred_joints[20,1]], [pred_joints[7,2], pred_joints[20,2]], color='b')
    # ax_2.plot([pred_joints[11,0], pred_joints[20,0]], [pred_joints[11,1], pred_joints[20,1]], [pred_joints[11,2], pred_joints[20,2]], color='b')
    # ax_2.plot([pred_joints[15,0], pred_joints[20,0]], [pred_joints[15,1], pred_joints[20,1]], [pred_joints[15,2], pred_joints[20,2]], color='b')
    # ax_2.plot([pred_joints[19,0], pred_joints[20,0]], [pred_joints[19,1], pred_joints[20,1]], [pred_joints[19,2], pred_joints[20,2]], color='b')

    ax_2.view_init(-70, -70) 
    plt.draw()
    plt.pause(1)

    scat_1.remove()
    scat_2.remove()
    ax_1.clear()
    ax_2.clear()

def visualize(model_chk):
    model_setup = ModelSetup(load=model_chk, test=True)
    out, transformed_image, transformed_joints, xy_boundingbox, depth_images, out_joints, pred_joints, median_depths = run_model(model_setup, train=False, test=True)
    print(f"epoch: {model_setup.epoch},\t avg_loss: {out['avg_loss']:1.4f},\t regression loss: {out['avg_reg_loss']:1.4f},\t classification loss: {out['avg_class_loss']:1.4f}")

    # draw_img(transformed_image[0], pred_joints[0], transformed_joints[0])
    fig = plt.figure(figsize=(6,10))

    ax_1 = fig.add_subplot(2,1,1)
    ax_2 = fig.add_subplot(2, 1, 2, projection='3d')
    
    for i in range(transformed_image.shape[0]):
        pred_joint, true_joint = back_to_normal(pred_joints[i], transformed_joints[i], xy_boundingbox[i], median_depths[i])
        # draw_img_normal(depth_images[i], pred_joint, true_joint, median_depths[i].item())
        vizualize_frams(ax_1, ax_2, depth_images[i], pred_joint, true_joint, median_depths[i].item(), out_joints[i])
    
    # visualize_point_cloud(depth_images[0], pred_joint, true_joint, median_depths[0].item())

def main():
    args = get_arguments()
    model_path = get_latest_model(args)
    visualize(model_path)

if __name__ == "__main__":
    main()