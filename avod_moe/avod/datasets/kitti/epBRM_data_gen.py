"""Dataset utils for preparing data for the network."""

import itertools
import fnmatch
import os
import random

import numpy as np
import cv2
import tensorflow as tf
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils

from avod.core import box_3d_encoder
from avod.core import constants
from avod.datasets.kitti import kitti_aug
from avod.datasets.kitti.kitti_utils import KittiUtils
import enum 
"""
class epBRM_config_cars(enum.Enum):
    def __str__(self):
        return str(self.value)
    cylinder_radius = 2.4
    cylinder_bottom_z = -0.5
    cylinder_top_z = 2.5
    data_aug_scale_min =  0.9
    data_aug_scale_max = 1.1
    data_aug_rot_min = -0.392699
    data_aug_rot_max = 0.392699
    dist_bound = 0.15
    sample_points_num = 512
 """   

class epBRMDataset():

    
    def __init__(self, kitti_dataset, dataset_config):
        self.kitti_dataset = kitti_dataset

        epBRM_config_cars = {
            "cylinder_radius" : 2.4,
            "cylinder_bottom_z" : -0.5,
            "cylinder_top_z" : 2.5,
            "data_aug_scale_min" :  0.9,
            "data_aug_scale_max" : 1.1,
            "data_aug_rot_min" : -0.392699,
            "data_aug_rot_max" : 0.392699,
            "dist_bound" : 0.15,
            "sample_points_num" : 512
        }
        self._config = epBRM_config_cars 


    def get_samples(self, batch_size=1):
        print("KITTIII", self.kitti_dataset)
        samples = self.kitti_dataset.next_batch(batch_size=batch_size, shuffle=True)
        # only support batch_size 1
        samples = samples[0]
        # point_cloud: NX3 each row [x,y,z]
        point_cloud = samples[constants.KEY_POINT_CLOUD]
        print("POINT_CLOUD_SHAPE_1", point_cloud[0, :].shape)
        # first select points whose x is larger than 0
        #point_cloud_front = point_cloud[point_cloud[:,0]>0]
        point_cloud_front = point_cloud
        print("POINT_CLOUD_FRONT_SHAPE", point_cloud_front.shape)
        # label_box_3d: NX7 each row [x,y,z,w,h,l,theta]
        label_box_3d = samples[constants.KEY_LABEL_BOXES_3D]
        cam_to_velo = samples["cam_to_velo"]
        # all are N rows arrays
        center_gt = label_box_3d[:,:3]
        dim_gt = label_box_3d[:,3:6]
        angle_gt = label_box_3d[:,6]

        # create new gt with augmentation
        new_center_gt = np.zeros_like(center_gt)
        new_dim_gt = np.zeros_like(dim_gt)
        new_angle_gt = np.zeros_like(angle_gt)

        cylinder_radius = self._config["cylinder_radius"]
        cylinder_top_z = self._config["cylinder_top_z"]
        cylinder_bottom_z = self._config["cylinder_bottom_z"]
        data_aug_scale_extents = [self._config["data_aug_scale_min"], self._config["data_aug_scale_max"]]
        data_aug_rot_extents = [self._config["data_aug_rot_min"], self._config["data_aug_rot_max"]]
        dist_bound = self._config["dist_bound"]
        sample_points_num = self._config["sample_points_num"]

        final_sample_points = []
        for ind in range(center_gt.shape[0]):
            
            # sample points in cylinder, sample_points shape: [num_cylinders, points_num_in_one_cylinder, 3]
            sample_points = self.cylinder_filter(point_cloud_front, center_gt[ind], cylinder_radius, 
                                    cylinder_bottom_z, cylinder_top_z, sample_points_num,cam_to_velo)
            
            # translate sample points centering at center_gt
            sample_points_center_gt = sample_points - center_gt[ind].reshape((1,3))

            # rotate sample points -angle_gt[ind] to make the bounding box aligned with y axis
            rot_sample_points_center_gt = self.rotate_z(sample_points_center_gt, -angle_gt[ind])

            # randomly scale the rotated sample points in x, y, z direction
            scaled_rot_sample_points_center_gt, scale_dim_gt = self.random_scale(rot_sample_points_center_gt, 
                                dim_gt[ind], data_aug_scale_extents[0], data_aug_scale_extents[1])
            new_dim_gt[ind] = scale_dim_gt
            
            # rotate the scaled sample points back to original orientation plus a random angle
            random_angle = np.random.uniform(data_aug_rot_extents[0], data_aug_rot_extents[1])
            new_angle_gt[ind] = angle_gt[ind] + random_angle
            rotate_back_sample_points = self.rotate_z(scaled_rot_sample_points_center_gt, new_angle_gt[ind])

            # randomly translated the point cloud to a new position
            translated_sample_points, translated_center_gt = self.random_translate(rotate_back_sample_points, 
                                                                center_gt[ind], dist_bound)
            final_sample_points.extend(translated_sample_points)
            new_center_gt[ind] = translated_center_gt
        new_angle_gt = new_angle_gt.reshape((new_angle_gt.shape[0],1))
        # create sample dict to return
        sample_dict = {
            "center_gt": new_center_gt,
            "dim_gt": new_dim_gt,
            "angle_gt": new_angle_gt,
            "sampled_point_cloud": final_sample_points
        }
        print("angle", new_angle_gt)
        return sample_dict

    def cylinder_filter(self, point_cloud, center, radius, zmin, zmax, sample_num, cam_to_velo):
        radius_square = radius ** 2
        print("POINT_CLOUD", point_cloud)
        print("CENTER", center)
        print("ZMIN", zmin)
        print("ZMAX", zmax)
        #print("POINT_CLOUD_SHAPE", point_cloud.shape)
        point_cloud = np.reshape(point_cloud, (-1, 3))
        print("POINT_CLOUD_SHAPE", point_cloud.shape)
        print("cam to velo dim", cam_to_velo)
        
        #center = np.vstack((np.transpose(center).reshape((3,1)), np.ones((1,1))))
        #center_in_velo = np.dot(cam_to_velo, center)
        #print("center shape", center_in_velo)
        """
        filtered_point_cloud = point_cloud[np.any(point_cloud[:,2]>=zmin) 
                                            and np.any(point_cloud[:,2]<=zmax)
                                            and ((point_cloud[:,0]-center_in_velo[0])**2+
                                            (point_cloud[:,1]-center_in_velo[1])**2<=radius_square)]
        """
        mask1 = (point_cloud[:,1]- center[1]) >= -zmax
        mask2 = (point_cloud[:,1] - center[1])<=-zmin
        mask3 = ((point_cloud[:,0]-center[0])**2 + (point_cloud[:,2]-center[2])**2) <= radius_square
        for i in range(20000):
            print(mask1[i], mask2[i], mask3[i])
        """
        filtered_point_cloud = point_cloud[np.where((point_cloud[:,1]- center[1])>= -zmax) 
                                            and np.where((point_cloud[:,1] - center[1])<=-zmin)]
                                            #* ((point_cloud[:,0]-center[0])**2+
                                             #   (point_cloud[:,2]-center[2])**2<=radius_square)]
        """
        filtered_point_cloud = point_cloud[mask1 * mask2 * mask3]
        """
        for i in range(point_cloud.shape[0]):
            if point_cloud[i,2] > center[2]:
                print("Above center z", point_cloud[i,:])
        """
        """
        for i in range(point_cloud.shape[0]):
            if point_cloud[i, 2]>=zmin:
                if point_cloud[i,2]<=zmax:
                    #print("Found")
                    if ((point_cloud[i,0]-center_in_velo[0])**2+(point_cloud[i,1]-center_in_velo[1])**2)<=(radius_square):
                        print("Found", point_cloud[i, :])
        """
                        
        # TODO: filtered_point_cloud is empty sometimes
        print("FILTERED", np.sort(filtered_point_cloud))
        
        #filtered_point_cloud = point_cloud
        np.random.shuffle(filtered_point_cloud)
        filtered_point_cloud = filtered_point_cloud[:sample_num,:]
        return filtered_point_cloud.reshape((1, filtered_point_cloud.shape[0], filtered_point_cloud.shape[1]))

    def rotate_z(self, point_cloud, angle):
        rot_mat = np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle), np.cos(angle),0],
                            [            0,             0,1]])
        rot_mat_T = rot_mat.T
        print("POINT_CLOUD_SHAPE_ROTATE", point_cloud.shape)
        return point_cloud.dot(rot_mat_T)

    def random_scale(self, point_cloud, gt_box_dim, rmin, rmax):
        """
        gt_box_dim(w,h,l) corresponding to y,z,x direction
        
        Return:
            scaled_point_cloud(NX3)
            scaled_box_dim(w,h,l)
        """
        rx = np.random.uniform(rmin, rmax)
        ry = np.random.uniform(rmin, rmax)
        rz = np.random.uniform(rmin, rmax)
        scaled_box_dim = gt_box_dim * np.array([ry,rz,rx])
        scale_array = np.array([rx,ry,rz]).reshape((1,1,3))
        assert scale_array.shape[2] == point_cloud.shape[2]
        scaled_point_cloud = point_cloud * scale_array
        return scaled_point_cloud, scaled_box_dim
    
    def random_translate(self, point_cloud, center_gt, dist_bound):
        x = np.random.uniform(-dist_bound, dist_bound)
        y = np.random.uniform(-dist_bound, dist_bound)
        z = np.random.uniform(-dist_bound, dist_bound)
        tran = np.array([x,y,z]).reshape((1, 1,3))
        assert tran.shape[2] == point_cloud.shape[2]
        translated_point_cloud = point_cloud + tran
        translated_center_gt = center_gt + tran
        return translated_point_cloud, translated_center_gt


        

