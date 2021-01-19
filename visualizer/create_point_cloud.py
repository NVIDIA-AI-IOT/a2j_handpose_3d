import os
import sys
import cv2
import json
import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.join(os.getcwd(), os.pardir)
sys.path.append(PROJECT_ROOT)

import pipeline.constants as const

'''
CLass CreatePointCloud create a point cloud from the depth image.
the depth image resolution is (576, 640) stored as a one channel png file with uint16 values.
'''
class CreatePointCloud(object):
    def __init__(self, data_set_path= const.DATASET):
        self.dataset_path = data_set_path
        self.xTable = self._load_xTable()
        self.yTable = self._load_yTable()

    def _load_xTable(self):
        xTable = np.empty((576, 640))
        with open(os.path.join(self.dataset_path, "../info/xTable.txt"), 'r') as fp:
            for i in range(xTable.shape[0]):
                xTable[i] = np.array( [float(i) for i in  fp.readline().split(" ")[:-1]])
        return xTable

    def _load_yTable(self):
        yTable = np.empty((576, 640))
        with open(os.path.join(self.dataset_path, "../info/yTable.txt"), 'r') as fp:
            for i in range(yTable.shape[0]):
                yTable[i] = np.array( [float(i) for i in  fp.readline().split(" ")[:-1]])
        return yTable

    def create_point_cloud(self, frameNumber="000000"):
        depth_img_path = os.path.join(self.dataset_path, f"depth_image/{frameNumber}.png")
        joint_path = os.path.join(self.dataset_path, f"joint_locations_point_cloud/{frameNumber}.json")
        
        with open(joint_path, 'r') as f:
            data = json.load(f)
        self.joint_data = np.array(data['body_joints'][0]['joint_pos_xyz'])
        median_depth = np.median(self.joint_data[:,2])
        depth_img = cv2.imread(depth_img_path, cv2.COLOR_BGR2GRAY)

        point_cloud = []
        for i in range(depth_img.shape[0]):
            for j in range(depth_img.shape[1]):
                x = depth_img[i][j] * self.xTable[i][j]
                y = depth_img[i][j] * self.yTable[i][j]
                z = depth_img[i][j]
                if (z != 0\
                    and median_depth-500 <= z <= median_depth+500):
                    point_cloud.append([x, y, z])
        self.point_cloud = np.array(point_cloud)

        return self.point_cloud, self.joint_data

    def create_point_cloud_from_2D(self, depth_img, t_joint, p_joint):
        
        self.true_joint_data = t_joint
        self.pred_joint_data = p_joint

        median_depth = np.median(self.true_joint_data[:,2])
        point_cloud = []
        for i in range(depth_img.shape[0]):
            for j in range(depth_img.shape[1]):
                x = depth_img[i][j] * self.xTable[i][j]
                y = depth_img[i][j] * self.yTable[i][j]
                z = depth_img[i][j]
                # if (z != 0):
                if (z != 0\
                    and median_depth-500 <= z <= median_depth+500):
                    point_cloud.append([x, y, z])
        self.point_cloud = np.array(point_cloud)

        # import pdb; pdb.set_trace()
        self.true_joint_data = [ [i[2] * self.xTable[int(i[1])][int(i[0])], i[2] * self.yTable[int(i[1])][int(i[0])], i[2]] for i in t_joint]
        self.pred_joint_data = [ [i[2] * self.xTable[int(i[1])][int(i[0])], i[2] * self.yTable[int(i[1])][int(i[0])], i[2]] for i in p_joint]

        return self.point_cloud, self.true_joint_data, self.pred_joint_data 
    
    def vizualize_point_cloud(self, point_cloud=None, joints=None):
        if (point_cloud is None):
            point_cloud = self.point_cloud
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(point_cloud)

        if joints is not None:
            self.joint_data = [self.joint_data[i] for i in joints]
            
        red = [[255, 0, 0] for i in self.joint_data]
        joint_pc = o3d.geometry.PointCloud()
        joint_pc.points = o3d.utility.Vector3dVector(self.joint_data)
        joint_pc.colors = o3d.utility.Vector3dVector(red)

        o3d.visualization.draw_geometries([pc, joint_pc])
    
    def store_point_cloud(self, point_cloud=None, point_cloud_path="Users/mnoori/Desktop/visualize_dataset/dataset/NAN/record0", frame_num=0):
        point_cloud_path = os.path.join(point_cloud_path, f"{str(frame_num).zfill(6)}.ply")
        if (point_cloud is None):
            point_cloud = self.point_cloud
        with open(point_cloud_path, "w") as fp:
            fp.write("ply\n")
            fp.write("format ascii 1.0\n")
            fp.write(f"element vertex {len(point_cloud)}\n")
            fp.write("property float x\n")
            fp.write("property float y\n")
            fp.write("property float z\n")
            fp.write("end_header\n")
            for point in point_cloud:
                fp.write(f"{point[0]} {point[0]} {point[0]}\n")
