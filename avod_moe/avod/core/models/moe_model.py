import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

class MoeModel():
    def __init__(self, img_feature_maps, bev_feature_maps, img_proposal_boxes, bev_proposal_boxes):
        # feature maps after bottle neck to one channel
        self.img_feature_maps = img_feature_maps
        self.bev_feature_maps = bev_feature_maps
        self.img_proposal_boxes = img_proposal_boxes
        self.bev_proposal_boxes = bev_proposal_boxes
        self.gating_net_input = tf.concat([tf.reshape(self.img_feature_maps, [1,-1]), 
                                           tf.reshape(self.bev_feature_maps,[1,-1])], axis=1)
        self.placeholders = dict()

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        print(self.img_feature_maps.shape)
        _, imgh, imgw, _ = self.img_feature_maps.shape
        _, bevh, bevw, _ = self.bev_feature_maps.shape
        with tf.variable_scope("img_feature_maps"):
            self._add_placeholder(tf.float32, [None,imgh,imgw], "img_feature_maps_pl")
        with tf.variable_scope("bev_feature_maps"):
            self._add_placeholder(tf.float32, [None,bevh,bevw],"bev_feature_maps_pl")


    def build(self):

        with tf.variable_scope("mix_of_experts", "moe", [self.gating_net_input]):
            tensor_in = self.gating_net_input
            print(tensor_in.shape)
            self.fc1 = slim.fully_connected(tensor_in, 128, scope='fc1')
            self.fc2 = slim.fully_connected(self.fc1,2,scope='fc2')
            self.out = slim.softmax(self.fc2)#tf.div(self.fc2, tf.reduce_sum(self.fc2,axis=1))

        prediction = dict()
        prediction['img_weight'] = self.out[0,0]
        prediction['bev_weight'] = self.out[0,1]
        return prediction