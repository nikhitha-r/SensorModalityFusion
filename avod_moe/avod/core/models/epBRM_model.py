import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from avod.core import losses
from avod.datasets.kitti.epBRM_data_gen import epBRMDataset
from avod.datasets.kitti.kitti_dataset import KittiDataset
from avod.core import orientation_encoder


class epBRM():
    KEY_LOCATION = "location"
    KEY_ORIENT = "orientation"
    KEY_SIZE = "size"
    PRED_LOCATION =  "pred_location"
    PRED_ORIENT = "pred_orient"
    PRED_SIZE = "pred_size"
    PRED_LOCATION_GT =  "pred_location_gt"
    PRED_ORIENT_GT = "pred_orient_gt"
    PRED_SIZE_GT = "pred_size_gt"
    POINT_CLOUD_INPUT = "input_point_cloud"
    def __init__(self, model_config, dataset):
        self.dataset = epBRMDataset(dataset, model_config)
        self.model_config = model_config
        # Inputs to network placeholders
        self._placeholder_inputs = dict()
        self.placeholders = dict()


    """
    Build the epBRM model
    """
    def build(self):
        dist_bound = 0.15
        # Setup input placeholders
        self._set_up_input_pls()
        #self.input = self._placeholder_inputs[self.POINT_CLOUD_INPUT]
        with tf.variable_scope('epbrm'):
            num_points = self._point_input.get_shape()[1].value

            # Build the network
            self.convnet1 = slim.conv2d(self._point_input, 64, 1, scope='convnet1')
            self.convnet2 = slim.conv2d(self.convnet1, 128, 1, scope='convnet2')
            self.convnet3 = slim.conv2d(self.convnet2, 256, 1, scope='convnet3')

            self.maxpool1 = slim.max_pool2d(self.convnet3, [2, 2], 2, scope='maxpool1')

            self.convnet4 = slim.conv2d(self.maxpool1, 256, 1, scope='convnet4')
            self.convnet5 = slim.conv2d(self.convnet4, 128, 1, scope='convnet5')
            self.convnet6 = slim.conv2d(self.convnet5, 64, 1, scope='convnet6')

            #self.convnet6 = slim.conv2d(self.convnet2, 7, [1,1], scope='convnet6')

            #self.flatten1 = slim.flatten(self.convnet6)  

            # Get the output of the last layers
            fc_layers_out = self.fc_layers_build(self.convnet6)

            loc_coords = fc_layers_out[self.KEY_LOCATION]
            loc_coords_x = tf.gather(loc_coords, 0)
            loc_coords_y = tf.gather(loc_coords, 1)
            loc_coords_z = tf.gather(loc_coords, 2)

            dist_x = 0.5 * dist_bound
            dist_y = 0.5 * dist_bound
            dist_z = 0.5 * dist_bound
            loc_x = 2 * (tf.nn.sigmoid(loc_coords_x) - 0.5) * dist_x
            loc_y = 2 * (tf.nn.sigmoid(loc_coords_y) - 0.5) * dist_y
            loc_z = 2 * (tf.nn.sigmoid(loc_coords_z) - 0.5) * dist_z

            angle_vec = fc_layers_out[self.KEY_ORIENT]
            size_out = fc_layers_out[self.KEY_SIZE]

            prediction_dict = dict()
            prediction_dict[self.PRED_LOCATION] = [loc_x, loc_y, loc_z]
            prediction_dict[self.PRED_ORIENT] = angle_vec
            prediction_dict[self.PRED_SIZE] = size_out
            """
            prediction_dict[self.PRED_LOCATION_GT] = self._placeholder_inputs[self.PRED_LOCATION_GT]
            prediction_dict[self.PRED_ORIENT_GT] = self._placeholder_inputs[self.PRED_ORIENT_GT]
            prediction_dict[self.PRED_SIZE_GT] = self._placeholder_inputs[self.PRED_SIZE_GT]
            """
            prediction_dict[self.PRED_LOCATION_GT] = self._location_gt
            prediction_dict[self.PRED_ORIENT_GT] = self._angle_gt
            prediction_dict[self.PRED_SIZE_GT] = self._size_gt
            return prediction_dict

            # Fetch the ground truth values

    """
    Build the fully connected layers
    Input: tensor_in : tensor input
    """
    def fc_layers_build(self, tensor_in):
        # Location output with size 3
        locs_out = slim.fully_connected(tensor_in,
                                      3,
                                      activation_fn=None,
                                      scope='locs_out')

        # Size output with size 3
        size_out = slim.fully_connected(tensor_in,
                                      3,
                                      activation_fn=None,
                                      scope='size_out')
        
        # Orientation output with size 2
        orient_out = slim.fully_connected(tensor_in,
                                      2,
                                      activation_fn=None,
                                      scope='orient_out')
        # Create a dictonary for the three outputs
        fc_output_layers = dict()
        fc_output_layers[self.KEY_LOCATION] = locs_out
        fc_output_layers[self.KEY_SIZE] = size_out
        fc_output_layers[self.KEY_ORIENT] = orient_out

        return fc_output_layers
        
    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def _set_up_input_pls(self):
        """Sets up input placeholders by adding them to self._placeholders.
        Keys are defined as self.PL_*.
        """
        

        with tf.variable_scope('point_cloud_input'):
            # Placeholder for BEV image input, to be filled in with feed_dict
            point_input_placeholder = self._add_placeholder(tf.float32, [None, None, 3],
                                                          self.POINT_CLOUD_INPUT)

            self._point_input = tf.expand_dims(
                point_input_placeholder, axis=0)
        
        with tf.variable_scope('GT_holder'):    
            location_gt_placeholder = self._add_placeholder(tf.float32, [None, 3],
                                                          self.PRED_LOCATION_GT)
            self._location_gt = tf.expand_dims(
                location_gt_placeholder, axis=0)

            size_gt_placeholder = self._add_placeholder(tf.float32, [None, 3],
                                                          self.PRED_SIZE_GT)
            self._size_gt = tf.expand_dims(
                size_gt_placeholder, axis=0)

            angle_gt_placeholder = self._add_placeholder(tf.float32, [None, 1],
                                                          self.PRED_ORIENT_GT)
            self._angle_gt = tf.expand_dims(
                angle_gt_placeholder, axis=0)                                            
        
            

    def create_feed_dict(self, sample_index=None):
        """ Fills in the placeholders with the actual input values.
            Currently, only a batch size of 1 is supported

        Args:
            sample_index: optional, only used when train_val_test == 'test',
                a particular sample index in the dataset
                sample list to build the feed_dict for

        Returns:
            a feed_dict dictionary that can be used in a tensorflow session
        """
        
        feed_samples = self.dataset.get_samples(batch_size=1)
        self._placeholder_inputs[self.POINT_CLOUD_INPUT] = feed_samples["sampled_point_cloud"]
        self._placeholder_inputs[self.PRED_SIZE_GT] = feed_samples["dim_gt"]
        self._placeholder_inputs[self.PRED_LOCATION_GT]= feed_samples["center_gt"]
        self._placeholder_inputs[self.PRED_ORIENT_GT] = feed_samples["angle_gt"]
        feed_dict = dict()
        for key, value in self.placeholders.items():
            print(key)
            print(value.shape)
            feed_dict[value] = self._placeholder_inputs[key]
        return feed_dict

    """

    Calculate the loss of the network

    """
    def loss(self, prediction_dict):
        # Fetch the ground truth values
        location_gt = prediction_dict[self.PRED_LOCATION_GT]
        orientation_gt = prediction_dict[self.PRED_ORIENT_GT]
        orientation_vec_gt = orientation_encoder.tf_orientation_to_angle_vector(
                orientation_gt)
        size_gt = prediction_dict[self.PRED_SIZE_GT]

        # Fetch the prediction values
        with tf.variable_scope('epbrm_prediction_batch'):
            location_pred = prediction_dict[self.PRED_LOCATION]
            orientation_vec_pred = prediction_dict[self.PRED_ORIENT]
            size_pred = prediction_dict[self.PRED_SIZE]
        # Calculate the losses
        with tf.variable_scope('epbrm_losses'):
            # Location loss
            with tf.variable_scope('location'):
                # Fetch the loss function
                reg_loss = losses.WeightedSmoothL1Loss()

                location_loss = reg_loss(location_pred,
                                            location_gt,
                                            weight=5.0)
            # Size loss
            with tf.variable_scope('size'):
                # Fetch the loss function
                reg_loss = losses.WeightedSmoothL1Loss()

                size_loss = reg_loss(size_pred,
                                        size_gt,
                                           weight=5.0)
            
            # Orientation loss
            with tf.variable_scope('orientation'):
                # Fetch the loss function
                reg_loss = losses.WeightedSmoothL1Loss()

                orientation_loss = reg_loss(orientation_vec_pred,
                                        orientation_vec_gt,
                                           weight=1.0)
        with tf.variable_scope('epBRM_loss'):
            total_loss = tf.reduce_sum(location_loss) + tf.reduce_sum(size_loss) + tf.reduce_sum(orientation_loss)
        return total_loss

            




            
