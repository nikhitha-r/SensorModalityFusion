import os
import numpy as np
import open3d as o3d
import cv2
import argparse
import scipy
import scipy.interpolate

# objectness threshold
obj_thres = 0.8
# input read path
# data_index = "009000"
data_index = "000008"
# data_index = "009001"
# iteration_index = "100000"
# folder = "./Aug_sample/Aug_sample/"
folder = "./"
pred_folder_default = "final_boxes_prediction_dropout_05_119000_noise/"


# point_cloud_path = folder + "velodyne/" + data_index + ".bin"
# image_path = folder + "image_2/" + data_index + ".png"
# label_path = folder + "label_2/" + data_index + ".txt"
# calib_path = folder + "calib/" + data_index + ".txt"
# pred_filepath = folder + "prediction_{}/".format(iteration_index) + data_index + ".txt"
# pred_filepath = folder + "final_boxes_prediction_dropout_05_119000/" + data_index + ".txt"
# pred_filepath = folder + "final_boxes_prediction_new_bev_110000/" + data_index + ".txt"
# pred_filepath = folder + "final_boxes_prediction_dropout_05_119000_noise/" + data_index + ".txt"


def project_to_feature_map(feature_map, projected_points, calib_matrix):
    """
    Project points in lidar frame to camera frame and interpretate values
    Args:
        feature_map: feature map extracted from rgb image
        projected_points: (NX3 np.array) points in lidar frame to be projected
        calib_matrix: (3X4 np.array) transformation from lidar frame to camera 
                      fram
    Return:
        features: (NX1 np.array) interpolated features of N projected points
    """
    # w, h = feature_map.shape
    projected_points = projected_points[projected_points[:, 0] > 0]
    N, _ = projected_points.shape
    projected_points = np.transpose(projected_points)
    projected_points = np.vstack((projected_points, np.ones((1, N))))
    points_2d = np.dot(calib_matrix, projected_points)
    points_2d = points_2d / points_2d[2]

    # h,w = feature_map.shape[:2]
    # xx,yy = np.meshgrid(range(w),range(h))
    # xx_yy_list = list(zip(xx.flat,yy.flat))
    # features1 = scipy.interpolate.interp2d(range(w),range(h), feature_map[:,:,0], 'linear')
    # features2 = scipy.interpolate.interp2d(range(w),range(h), feature_map[:,:,1], 'linear')
    # features3 = scipy.interpolate.interp2d(range(w),range(h), feature_map[:,:,2], 'linear')

    # print(xx.shape)
    # print(points_2d.shape)
    # features1 = scipy.interpolate.interpn((xx_yy_list[0],xx_yy_list[1]), feature_map[:,:,0], points_2d[:2].T, 'linear')
    # features2 = scipy.interpolate.interpn(xx_yy_list, feature_map[:,:,1].flat, points_2d[:2], 'linear')
    # features3 = scipy.interpolate.interpn(xx_yy_list, feature_map[:,:,2].flat, points_2d[:2], 'linear')
    # features = np.array([features1(points_2d[0],points_2d[1]),features2(points_2d[0],points_2d[1]),
    # features3(points_2d[0],points_2d[1])])
    # features = np.array([features1(points_2d[0],points_2d[1])])

    p_2d = np.round(points_2d[:2]).astype(int)
    fh, fw, _ = feature_map.shape
    mask1 = p_2d[0] < fw  # 1242  # and p_2d[0] >=0 and p_2d[1] < 375 and p_2d[1] >= 0
    mask2 = p_2d[0] > 0
    mask3 = p_2d[1] < fh  # 374
    mask4 = p_2d[1] > 0
    mask = mask1 * mask2 * mask3 * mask4
    mask_l = mask.shape[0]
    mask = mask.reshape((mask_l, 1))
    print("mask shape: ", mask.shape)
    print(p_2d.shape)
    mask = mask.T
    print(np.sum(mask))
    mask_2d_p = np.repeat(mask, 2, axis=0)
    p_2d = p_2d[mask_2d_p].reshape((2, -1))
    print("mask before: ", mask.shape)
    mask = (mask == True)
    print(mask)
    print("mask after: ", mask.shape)
    mask_3d_p = np.repeat(mask, 4, axis=0)
    print("p_2d shape: ", p_2d.shape)
    print("mask_3d_p shape: ", mask_3d_p.shape)
    try:
        features = feature_map[p_2d[1], p_2d[0]]
    except:
        print('p_2d max', np.max(p_2d[1]))
        raise
    print(features.shape)
    features = features / 255.
    features = features[:, (2, 1, 0)]

    return projected_points[mask_3d_p].reshape((4, -1)), features, mask_3d_p[0]


def draw_point_cloud(point_cloud_path, image_path, paint_color="depth"):
    """
    return a point cloud object for visualization
    Args:
        point_cloud_path: path to the .bin file storing point cloud data
        paint_color     : "depth" or "image", "depth" means the color of the point
                          will change according to the depth, "image" will paint 
                          the point cloud with rgb values in the camera it project to
    """
    with open(point_cloud_path, "rb") as f:
        pc = np.fromfile(f, np.single)
        pc = pc.reshape(-1, 4)
        print(pc[:10])
        print(pc.shape)

    pc = pc[:, :3]

    if paint_color == "image":
        im = cv2.imread(image_path)
        """Frame Calibration Holder
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters.

        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.

        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        """
        tf_velo2cam0 = read_calib(calib_path)
        tf_cam02rect = read_calib(calib_path, tf_name="R0_rect")
        tf_rect2cam2 = read_calib(calib_path, tf_name="P2")
        calib_mat = np.dot(np.dot(tf_rect2cam2, tf_cam02rect), tf_velo2cam0)
        pc, colors, mask = project_to_feature_map(im, pc, calib_mat)
        print("pc shape: ", pc.shape)
        print("mask shape: ", mask.shape)
        pc = pc[:3, :].T
        origin_colors = np.zeros((pc.shape[0], 3))
        origin_colors = colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    if paint_color == 'image':
        pcd.colors = o3d.utility.Vector3dVector(origin_colors)

    o3d.io.write_point_cloud("{}.ply".format(os.path.basename(point_cloud_path).split(".")[0]), pcd)
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def box_paras2corners(box_paras, calib_mat):
    """
    Input box parameters of a series of boxes, return the corners for these boxes
    Args: box_paras(NX7 np.array) : dimensions, coordinate, rotation
          calib_mat(4X4 np.array) : lidar frame to camera frame
    Return: box_corners(NX8X3 np.array)
    """
    box_corners = []
    for box_para in box_paras:
        w, h, l = box_para[0:3]
        x, y, z = box_para[3:6]
        print("w,h,l: ", w, h, l)
        print(x, y, z)
        rotate = box_para[6]
        corner = np.array([
            [x - l / 2., y - w, z - h / 2],
            [x + l / 2., y - w, z - h / 2],
            [x - l / 2., y, z - h / 2],
            [x - l / 2., y - w, z + h / 2],
            [x - l / 2., y, z + h / 2],
            [x + l / 2., y, z - h / 2],
            [x + l / 2., y - w, z + h / 2],
            [x + l / 2., y, z + h / 2],
        ])

        corner -= np.array([x, y, z])

        rotate_matrix = np.array([
            [np.cos(rotate), 0, np.sin(rotate)],
            [0, 1, 0],
            [-np.sin(rotate), 0, np.cos(rotate)]
        ])

        a = np.dot(corner, rotate_matrix.transpose())
        a += np.array([x, y, z])
        calib_mat_inv = np.linalg.inv(calib_mat)
        a = np.dot(calib_mat_inv[:3, :3], a.T)
        t = calib_mat_inv[:3, 3].reshape((3, 1))
        a = a + t
        box_corners.append(a.T)
    box_corners = np.array(box_corners)
    return box_corners


def read_calib(calib_path, tf_name="Tr_velo_to_cam"):
    """
    Return the matrix in calibration text file
    Args:
        calib_path: path to the calib file
        tf_name   : Could be one of "P0" "P1" "P2" "P3" 
                    "R0_rect" "Tr_velo_to_cam" "Tr_imu_to_velo".
                    "P0"~"P3" refers to transformation from rectified 0 image to x image
                    "R0_rect" cam 0 coordinate to rectified cam 0 coordinate
                    "Tr_velo_to_cam" velodyne frame to camera 0 frame
    Return:
        calib_mat:(4X4 np.array) transformation matrix specified by tf_name
    """
    with open(calib_path, "r") as f:
        calib_mat = np.identity(4)
        for i, line in enumerate(f):
            row = line.strip("\n").split(" ")
            if row[0].strip(":") == "R0_rect" == tf_name:
                calib_mat[:3, :3] = np.array([float(x) for x in row[1:]]).reshape((3, 3))
                break
            elif row[0].strip(":") == tf_name:
                calib_mat[:3, :4] = np.array([float(x) for x in row[1:]]).reshape((3, 4))
                break
    return calib_mat


def draw_bbox_lines(box_corners, calib_mat, color=[1, 0, 0]):
    line_sets = []
    box_sets = []
    for i, one_box_corners in enumerate(box_corners):
        # if i != 1:
        #     continue
        # for one_box_corners in box_corners:
        lines = [[0, 1], [0, 2], [0, 3], [1, 6], [6, 7], [6, 3], [1, 5], [5, 7], [5, 2], [4, 7], [4, 3], [4, 2]]
        colors = [color for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(one_box_corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

    # draw z axis
    line_set = o3d.geometry.LineSet()
    points = [[0, 0, 0], [0, 0, 10]]
    lines = [[0, 1]]
    colors = [[0, 0, 1]]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_sets.append(line_set)

    # draw y axis
    line_set = o3d.geometry.LineSet()
    points = [[0, 0, 0], [0, 10, 0]]
    lines = [[0, 1]]
    colors = [[0, 1, 0]]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_sets.append(line_set)

    # draw y axis
    line_set = o3d.geometry.LineSet()
    points = [[0, 0, 0], [10, 0, 0]]
    lines = [[0, 1]]
    colors = [[1, 0, 0]]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_sets.append(line_set)
    return line_sets


def draw_gt_bbox(label_path, calib_path, label_class="Car"):
    """
    Draw bounding boxes on point cloud scene
    """
    box_paras = []
    with open(label_path, "r") as f:
        for line in f:
            row = line.split(" ")
            line_class = row[0]
            if line_class == label_class:
                # 3 dim(h,w,l), 3 coordinate, 1 rotation[-pi,pi]
                box_paras.append(np.array([float(x) for x in row[8:]]))
    tf_velo2cam0 = read_calib(calib_path)
    tf_cam02rect = read_calib(calib_path, tf_name="R0_rect")
    calib_mat = np.dot(tf_cam02rect, tf_velo2cam0)
    box_paras = np.array(box_paras)
    box_corners = box_paras2corners(box_paras, calib_mat)
    line_sets = draw_bbox_lines(box_corners, calib_mat)
    return line_sets


def draw_pred_bbox(pred_filepath, calib_path, objectness_threshold=0.90, color=[0, 1, 0]):
    """
    Draw bounding boxes on point cloud scene
    """
    box_paras = []
    with open(pred_filepath, "r") as f:
        for line in f:
            row = line.split(" ")
            # 3 dim(h,w,l), 3 coordinate, 1 rotation[-pi,pi]
            row_np = np.array([float(x) for x in row])
            if row_np[7] > objectness_threshold:
                reordered_row = np.zeros_like(row_np)
                reordered_row[:3] = row_np[[5, 4, 3]]
                reordered_row[3:6] = row_np[:3]
                reordered_row[6] = row_np[6]
                box_paras.append(reordered_row)
    tf_velo2cam0 = read_calib(calib_path)
    tf_cam02rect = read_calib(calib_path, tf_name="R0_rect")
    calib_mat = np.dot(tf_cam02rect, tf_velo2cam0)
    box_paras = np.array(box_paras)
    box_corners = box_paras2corners(box_paras, calib_mat)
    line_sets = draw_bbox_lines(box_corners, calib_mat, color=color)
    return line_sets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3d bounding box.')
    parser.add_argument('-p', '--pred_folder', dest='pred_folder', default=pred_folder_default,
                        help='prediction file folder')
    parser.add_argument('-s', '--scene_index', dest='scene_index', default=data_index, help='scene index, e.g 000008')
    parser.add_argument('-d', '--data_folder', dest='data_folder', default='./',
                        help='data folder which contains 4 subfolders: image_2, label_2, velodyne, calib')
    parser.add_argument('-t', '--threshold', dest='obj_thres', type=float, default=obj_thres,
                        help='threshold for the objectness score in prediction, only boxes with objectness above this threshold will be shown')
    args = parser.parse_args()

    point_cloud_path = os.path.join(args.data_folder, "velodyne/", args.scene_index + ".bin")
    image_path = os.path.join(args.data_folder + "image_2/", args.scene_index + ".png")
    label_path = os.path.join(args.data_folder + "label_2/", args.scene_index + ".txt")
    calib_path = os.path.join(args.data_folder + "calib/", args.scene_index + ".txt")
    pred_filepath = os.path.join(args.data_folder, "final_boxes_prediction_dropout_05_119000_noise/",
                                 args.scene_index + ".txt")

    pcd = draw_point_cloud(point_cloud_path, image_path, paint_color="image")
    gt_line_sets = draw_gt_bbox(label_path, calib_path)
    pred_line_sets = draw_pred_bbox(pred_filepath, calib_path, objectness_threshold=args.obj_thres, color=[0, 1, 0])
    line_sets = gt_line_sets + pred_line_sets

    # pcd = draw_point_cloud(point_cloud_path, image_path, paint_color="image")
    # gt_line_sets = draw_gt_bbox(label_path, calib_path)
    # pred_line_sets = draw_pred_bbox(pred_filepath, calib_path, objectness_threshold=obj_thres, color=[0, 1, 0])
    # line_sets = gt_line_sets + pred_line_sets
    # o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd, *line_sets])
