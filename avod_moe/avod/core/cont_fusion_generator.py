import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim
from wavedata.tools.core import calib_utils

def add_margin_to_regions(bev_proposal_boxes, bev_extents, margin_size_in_m=0.5):
    """
    Arguments:
        bev_proposal_boxes: Nx[x1,y1,x2,y2] normalized coordinates in bev_extents
        bev_extents: [[xmin,xmax],[ymin,ymax]]
        margin_size_in_m: marginal space intended to add
    Return:
        bev_mar_boxes, bev_mar_boxes_norm
    """
    bev_xmin, bev_xmax = bev_extents[0]
    bev_ymin, bev_ymax = bev_extents[1]
    mar = np.array([-0.5,-0.5,0.5,0.5], dtype=np.float32).reshape((1,4))
    bev_mar_boxes = bev_proposal_boxes + mar
    x1 = bev_mar_boxes[:,0]/(bev_xmax-bev_xmin)
    x2 = bev_mar_boxes[:,2]/(bev_xmax-bev_xmin)
    y1 = bev_mar_boxes[:,1]/(bev_ymax-bev_ymin)
    y2 = bev_mar_boxes[:,3]/(bev_ymax-bev_ymin)
    bev_mar_boxes_norm = np.vstack([x1,y1,x2,y2]).T
    return bev_mar_boxes, bev_mar_boxes_norm

def bev_pixel_eq_1_loc(bev_preprocessed):
    """
    Return: NX[batch_ind, row_ind, col_ind, layer_ind] tf.tensor
    """
    print("bev_preprocessed.shape: ", bev_preprocessed.shape)
    b, h, w, l = bev_preprocessed.shape
    # print bev_preprocessed to see max and min
    bev_pixels_loc = tf.where(bev_preprocessed > 0)
    # print("bev_pixels_loc shape: ", bev_pixels_loc.shape)
    print("bev_pixels_loc.shape: ", bev_pixels_loc.shape)
    # bev_pixels_loc = tf.stack([bev_pixels_loc[:,0],bev_pixels_loc[:,3], bev_pixels_loc[:,1],
    #                             bev_pixels_loc[:,2]], axis=1)
    return bev_pixels_loc


def bev_pixel_loc_to_3d_velo(bev_pixels_loc, bev_shape, height_list, bev_extents):
    """
    Arguments:
        bev_pixels_loc: Nx[batch_ind, row_ind, col_ind, layer_ind]
        bev_shape: (row_num, col_num)
        height_list: z value of the point is height_list[layer_ind]
        bev_extents: [[xmin,xmax],[ymin,ymax]]
    Return:
        velo_pc: Nx[x,y,z] in velodyne coordinate frame
    """
    # bev x --->  y |
    # velodyne x | y <--
    print("bev_pixels_loc shape: ", bev_pixels_loc.shape)
    bev_xmin, bev_xmax = bev_extents[0]
    bev_ymin, bev_ymax = bev_extents[1]
    h = bev_shape[0] # 700
    w = bev_shape[1] # 800
    h = tf.cast(h, dtype=tf.int64)
    w = tf.cast(w, dtype=tf.int64)
    x = (h - bev_pixels_loc[:,1]) * (bev_ymax - bev_ymin) / h + bev_ymin
    y = (w - bev_pixels_loc[:,2]) * (bev_xmax - bev_xmin) / w + bev_xmin
    z = tf.gather(height_list, bev_pixels_loc[:,3])
    z = tf.cast(z, tf.float64)
    # print("x,y,z shape: ", x.shape, y.shape, z.shape)
    velo_pc = tf.stack([x,y,z],axis=1)
    print("velo_pc shape: ", velo_pc.shape)
    # raise Exception("velo_pc finished!")
    return velo_pc

def project_to_image(point_cloud, calib_mat):
    """
    Arguments:
        point_cloud: Nx3 matrix
        calib_mat: 3x4 calibration matrix 
    Return:
        pts_2d: Nx2
    """
    pts_2d = tf.tensordot(calib_mat, tf.stack(tf.transpose(point_cloud),
                                 tf.ones((1, point_cloud.shape[0])),
                                 axis=0))

    pts_2d[0, :] = pts_2d[0, :] / pts_2d[2, :]
    pts_2d[1, :] = pts_2d[1, :] / pts_2d[2, :]
    return tf.transpose(pts_2d[:2,:])

def create_fused_bev(bev_shape, bev_pixels_loc, img_features):
    """
    Arguments:
        bev_shape: [Batch_size, h, w, l]
        bev_pixels_loc: NX[batch_ind, row_ind, col_ind, layer_ind]
        img_features: NX32
    """

    with tf.variable_scope("image_features_compression", "ifc", [img_features]):
        fc1 = slim.fully_connected(img_features, 16, scope='fc1')
        fc2 = slim.fully_connected(fc1, 1, activation_fn=None, scope='fc2')

    print("fc2: ", fc2)
    print("fc2 shape: ", fc2.shape)
    print("bev_pixels_loc: ", bev_pixels_loc.shape)
    print("bev_shape: ", bev_shape)
    # bev_shape = tf.cast(bev_shape, tf.int64)
    fc2 = tf.reshape(fc2, [-1])
    fused_bev = tf.Variable(tf.zeros(bev_shape))
    fused_bev = tf.scatter_nd_update(fused_bev, bev_pixels_loc, fc2)
    return fused_bev

    


    