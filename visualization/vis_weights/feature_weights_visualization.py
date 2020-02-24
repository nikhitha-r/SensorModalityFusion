import numpy as np
import cv2
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import os


def feature_pca(feat):
    b, h, w, d = feat.shape
    print("feat shape: ", h, w, d)
    reshaped_feat = feat.reshape((-1, d))
    mean = np.mean(reshaped_feat, axis=1).reshape((-1, 1))
    std = np.std(reshaped_feat, axis=1).reshape((-1, 1))
    std_zero_mask = np.tile((std == 0), (1, d))
    std_feat = np.true_divide((reshaped_feat - mean), std)
    std_feat[std_zero_mask] = 0
    # std_feat = reshaped_feat
    pca = PCA(n_components=3, whiten=True, svd_solver='randomized')
    pca_feat = pca.fit_transform(reshaped_feat)
    pca_feat = ((pca_feat - np.min(pca_feat)) / (np.max(pca_feat) - np.min(pca_feat)) * 255).astype(np.int8)
    # pca_feat = (pca_feat*255).astype(np.int8)
    pca_feat = pca_feat.reshape((b, h, w, -1))

    return pca_feat


def read_feature(filename, channels, feat_ind=-1, show='False', average='False'):
    """
    filename(str): path to the .npy file
    channels(list): a list of three integers range from 0~31, specifying which channels to visualize
    feat_ind(int): an integer indicating which box's feature is visualize, if -1 just return the whole image
    average(bool): if True, average all channels and return the averaged feature
    """
    feat = np.load(filename)
    print("feat shape: ", feat.shape)
    if channels == -1:
        selected_feat = feature_pca(feat)
    else:
        selected_feat = (feat[:, :, :, channels] * 255).astype(np.int8)
    # reverse rgb channel
    # selected_feat = selected_feat[:,:,:,range(selected_feat.shape[3]-1,-1,-1)]

    if feat_ind != -1:
        if average == 'True':
            selected_feat = (np.mean(feat, axis=3) * 255).astype(np.int8)
            feat2image = Image.fromarray(selected_feat[feat_ind, :, :], 'L')
        else:
            print("feat2image shape: ", selected_feat.shape)
            feat2image = Image.fromarray(selected_feat[feat_ind, :, :, :], 'RGB')
        feat2image = feat2image.resize((700, 700))
    else:
        if average == 'True':
            selected_feat = (np.mean(feat, axis=2) * 255).astype(np.int8)
            feat2image = Image.fromarray(selected_feat, 'L')
        else:
            print("feat2image shape: ", selected_feat.shape)
            feat2image = Image.fromarray(selected_feat[feat_ind, :, :, :], 'RGB')

    # feat2image = feat2image.resize((100,100))
    # feat2image.show()
    if show == 'True':
        feat2image.show()
    return feat2image


def show_all_feat(filename, num_per_row=4):
    feat = np.load(filename)
    b, h, w, c = feat.shape
    img_num = int(c / 3) + (1 if c % 3 != 0 else 0)
    num_per_col = np.ceil(float(img_num) / float(num_per_row))
    final_img = Image.new('RGB', (int(w * num_per_row), int(h * num_per_col)))
    for i in range(img_num):
        f = feat[0, :, :, i * 3:i * 3 + 3]
        if f.shape[2] < 3:
            f = np.concatenate([f[:, :, :], np.zeros((h, w, 3 - f.shape[2]))], axis=2)
        f = Image.fromarray((f * 255).astype(np.int8), 'RGB')
        final_img.paste(f, (
            int(i % num_per_row) * w, int(i / num_per_row) * h, int(i % num_per_row + 1) * w,
            int(i / num_per_row + 1) * h))
    final_img.show()


def read_img(filename, show='False'):
    img2image = Image.open(filename)
    if show == 'True':
        img2image.show()
    return img2image


def draw_boxes(filename, weights_filename, img, mode='bev', box_num=10):
    """
    filename: filename for boxes .npy file
    img: Image object
    mode: 'bev' or 'img'
    """
    bev_xmin = -40
    bev_xmax = 40
    bev_ymin = 0
    bev_ymax = 70

    imgw, imgh = img.size
    print("imgw, imgh: ", imgw, imgh)

    # boxes [x1,y1,x2,y2] in meters, so we need to normalize them
    boxes = np.load(filename)
    print("###################")
    print((boxes[1][0], boxes[1][1], boxes[1][2], boxes[1][3]))
    weights = get_weight(weights_filename)
    print("boxes shape: ", boxes.shape)
    # print("boxes: ", boxes)
    draw = ImageDraw.Draw(img)
    line_color = (255, 0, 0)

    # vis_ind_list = {1,2,10,4,7,9,3,21,15} # for 000248
    # vis_ind_list = {18,17,9,0,6,16}#set(range(20)) # for 008017
    # vis_ind_list = {0,4,7,18,13,1,12}
    # vis_ind_list = {13,2,36,41,42,43,100} # for 009001 black block
    # vis_ind_list = {11,24,25,77,26,59,58} # for 009001 ground block
    # vis_ind_list = {1, 13, 57, 73, 27, 24}  # for 000008
    # vis_ind_list = {28, 71, 30, 83}  # for 009001 noisy block

    vis_ind_list = set(range(box_num))
    for i in range(boxes.shape[0]):
        if i in vis_ind_list:
            if mode == 'bev':
                x1 = (boxes[i][0]) * imgw / (bev_xmax - bev_xmin)
                x2 = (boxes[i][2]) * imgw / (bev_xmax - bev_xmin)
                y1 = (boxes[i][1]) * imgh / (bev_ymax - bev_ymin)
                y2 = (boxes[i][3]) * imgh / (bev_ymax - bev_ymin)
                a, b, c, d = gen_box_corners(x1, y1, x2, y2)
            else:
                a, b, c, d = gen_box_corners(*boxes[i])

            # if i == 1:
            # print((x1,y1,x2,y2))
            # c_img = img.crop((x1,y1,x2,y2))
            # c_img.show()
            draw.line([a, b, c, d, a], fill=line_color, width=1)
            # draw.text(list([a,b,c,d])[np.random.choice(range(4))],
            #             "{} weight: {:.3}".format(i, weights[i]),(0,255,0))
            draw.text([b[0], np.random.uniform(b[1], c[1])],
                      "{} weight: {:.3}".format(i, weights[i]), (0, 255, 0))
        if i > 100:
            break

    img.show()


def get_boxes(boxes_filename, mode='bev', bev_img_size=(0, 0)):
    bev_xmin = -40
    bev_xmax = 40
    bev_ymin = 0
    bev_ymax = 70
    imgw, imgh = bev_img_size
    boxes = np.load(boxes_filename)
    if mode == 'bev':
        x1 = boxes[:, 0] * imgw / (bev_xmax - bev_xmin)
        x2 = boxes[:, 2] * imgw / (bev_xmax - bev_xmin)
        y1 = boxes[:, 1] * imgh / (bev_ymax - bev_ymin)
        y2 = boxes[:, 3] * imgh / (bev_ymax - bev_ymin)
        print("###################")
        print((boxes[1][0], boxes[1][1], boxes[1][2], boxes[1][3]))
        boxes = np.vstack([x1, y1, x2, y2]).T
    print("new_boxes shape: ", boxes.shape)
    print("new boxes: ", boxes)
    return boxes


def gen_box_corners(x1, y1, x2, y2):
    return (x1, y1), (x2, y1), (x2, y2), (x1, y2)


def get_weight(filename):
    weights = np.load(filename)
    return weights


def crop_and_draw_weight(img, boxes, weights_filename, ind):
    weights = get_weight(weights_filename)
    croped_img = img.crop(boxes[ind])
    resized_croped_img = croped_img.resize((croped_img.size[0] * 10, croped_img.size[1] * 10))
    draw = ImageDraw.Draw(resized_croped_img)
    draw.text((20, 20), "weights: {:.3}".format(weights[ind]))
    resized_croped_img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizing features and region weights from MoE")
    parser.add_argument('-s', '--scene_index', dest='scene_index', type=str, default='000008',
                        help='scene index, e.g 000008')
    parser.add_argument('-d', '--data_folder', dest='data_folder', default='weights_dropout_05_119000_noise',
                        help='data folder which contains weights subfolder and rois subfolder')
    parser.add_argument('-id', '--image_directory', dest='img_folder', default='./',
                        help='the directory to the scene image')

    # show preprocessed bev or image
    parser.add_argument('-bp', '--show_bev', dest='show_bev', action='store_true', default=False,
                        help='whether to show whole preprocessed bev')
    parser.add_argument('-bpc', '--bev_channels', dest='bev_channels', type=list, default=[0, 1, 2],
                        help='specify three of 0~5 as the channel indexes')
    parser.add_argument('-i', '--show_image', dest='show_image', action='store_true', default=False,
                        help='whether to show whole preprocessed image')

    # show bev or image features in pca
    parser.add_argument('-bf', '--show_bev_feature', dest='show_bev_feature', action='store_true', default=False,
                        help='whether to show whole bev feature in pca')
    parser.add_argument('-if', '--show_img_feature', dest='show_img_feature', action='store_true', default=False,
                        help='whether to show whole image feature in pca')

    # show bev or image features of all channels
    parser.add_argument('-bfa', '--show_bev_feature_all', dest='show_bev_feature_all', action='store_true',
                        default=False,
                        help='whether to show whole bev features of all channels')
    parser.add_argument('-ifa', '--show_img_feature_all', dest='show_img_feature_all', action='store_true',
                        default=False,
                        help='whether to show whole image feature of all channels')

    # show bev or image bounding boxes
    parser.add_argument('-bb', '--show_bev_box', dest='show_bev_box', action='store_true', default=False,
                        help='whether to show bev region proposal boxes and weights')
    parser.add_argument('-ib', '--show_img_box', dest='show_img_box', action='store_true', default=False,
                        help='whether to show image region proposal boxes weights')
    parser.add_argument('-bnum', '--box_num', dest='box_num', type=int, default=10,
                        help='how many bounding boxes to show')

    args = parser.parse_args()

    # read_bev_feature("./rois/bev_feat/000001/000001.npy",[0,1,2])
    # sample_name = "009001"
    # sample_name = "009002"
    # sample_name = "009002"
    # folder = "weights_dropout_05_119000_noise"
    # im = np.load("./{}/rois/bev_pre/{}/{}.npy".format(folder,sample_name, sample_name))

    sample_name = args.scene_index
    folder = args.data_folder
    image_folder = args.img_folder
    box_num = args.box_num

    # read bev and show
    if args.show_bev:
        show_f = 'True'
    else:
        show_f = 'False'
    # im1 = read_feature("./{}/rois/bev_pre/{}/{}.npy".format(folder,sample_name, sample_name),[0,1,2], show='True')
    im1 = read_feature("./{}/rois/bev_pre/{}/{}.npy".format(folder, sample_name, sample_name), [0, 1, 2], show=show_f)
    # im2 = read_feature("./{}/rois/bev_pre/{}/{}.npy".format(folder,sample_name, sample_name),[3,4,5])

    # read some channels in whole image and bev
    # bev_feat_whole = read_feature("./{}/rois/bev_feat_whole/{}/{}.npy".format(folder,sample_name, sample_name),[0,1,2], show='True')
    # img_feat_whole = read_feature("./{}/rois/img_feat_whole/{}/{}.npy".format(folder,sample_name, sample_name),[0,1,2], show='True')

    # draw pca featuers
    if args.show_bev_feature:
        bev_feat_whole = read_feature("./{}/rois/bev_feat_whole/{}/{}.npy".format(folder, sample_name, sample_name), -1,
                                      show='True')
    if args.show_img_feature:
        img_feat_whole = read_feature("./{}/rois/img_feat_whole/{}/{}.npy".format(folder, sample_name, sample_name), -1,
                                      show='True')
    # bev_feat_whole = show_all_feat("./{}/rois/bev_feat_whole/{}/{}.npy".format(folder,sample_name, sample_name))
    # img_feat_whole = show_all_feat("./{}/rois/img_feat_whole/{}/{}.npy".format(folder,sample_name, sample_name))

    # read image and show
    if args.show_image:
        image1 = read_img(os.path.join(image_folder, "{}.png".format(sample_name)), show='True')
    else:
        image1 = read_img(os.path.join(image_folder, "{}.png".format(sample_name)), show='False')
    # image1 = read_img("{}.png".format(sample_name), show='False')

    # bev_feat = read_feature("./{}/rois/bev_feat/{}/{}.npy".format(folder,sample_name, sample_name),-1,feat_ind=12,show='True')
    # img_feat = read_feature("./{}/rois/img_feat/{}/{}.npy".format(folder,sample_name, sample_name),-1,feat_ind=12,show='True')

    # get weights and boxes
    bev_weights_filename = "./{}/weights/{}/bev_weights_{}.npy".format(folder, sample_name, sample_name)
    img_weights_filename = "./{}/weights/{}/img_weights_{}.npy".format(folder, sample_name, sample_name)
    bev_boxes_filename = "./{}/rois/bev_box/{}/{}.npy".format(folder, sample_name, sample_name)
    img_boxes_filename = "./{}/rois/img_box/{}/{}.npy".format(folder, sample_name, sample_name)

    # get boxes
    # draw boxes and bev or image
    if args.show_bev_box:
        bev_boxes = get_boxes(bev_boxes_filename, bev_img_size=im1.size[:2])
        draw_boxes("./{}/rois/bev_box/{}/{}.npy".format(folder, sample_name, sample_name),
                   bev_weights_filename, im1, 'bev', box_num=box_num)
    if args.show_img_box:
        img_boxes = get_boxes(img_boxes_filename, mode='img')
        draw_boxes("./{}/rois/img_box/{}/{}.npy".format(folder, sample_name, sample_name),
                   img_weights_filename, image1, 'img', box_num=box_num)

    # crop_and_draw_weight(im1, bev_boxes, bev_weights_filename, 1)
    # crop_and_draw_weight(image1, img_boxes, img_weights_filename, 1)
