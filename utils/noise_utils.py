# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2019-04-08 21:07:10
# @Last Modified by:   twankim
# @Last Modified time: 2019-05-20 21:39:16
# Modified by Nikhitha Radhakrishna Naik
import numpy as np
from wavedata.tools.core import calib_utils

class SINFields:
    VALID_SIN_TYPES = ['rand','rect','vert','lowres']
    SIN_INPUT_NAMES = ['image']
    VALID_MAX_MAGTD = {'image': 255,
                       'lidar': 0.2}
    SIN_LEVEL_MAX = 10.0
    SIN_LEVEL_MIN = 0.0
   


def genSINtoInputs(image_input,
                   sin_type='rand',
                   sin_level=1,
                   sin_input_name='image',
                   mask_2d=None,
                   frame_calib_p2=None):
    """
        Add Single Input Noise to a given input
    """
    print("Called")
    assert sin_type in SINFields.VALID_SIN_TYPES,\
        "sin_type must be one of [{}]".format(','.join(SINFields.VALID_SIN_TYPES))
    assert sin_input_name in SINFields.SIN_INPUT_NAMES,\
        "sin_input_name must be one of [{}]".format(','.join(SINFields.SIN_INPUT_NAMES))

    if sin_input_name == 'image':
        sin_image_input = np.copy(image_input).astype(np.float)
        sin_image_input = genSIN(sin_image_input,
                                 sin_type, sin_level,
                                 sin_input_name = sin_input_name,
                                 mask_2d = mask_2d,
                                 frame_calib_p2=frame_calib_p2)
        # Fit to range and type
        np.clip(sin_image_input, 0.0, 255.0, out=sin_image_input)
        sin_image_input = sin_image_input.astype(np.uint8)
   
    else:
        sin_image_input = image_input
    
    return sin_image_input


def genSIN(image_input,
           sin_type='rand', sin_level=1, sin_input_name='image', mask_2d=None,frame_calib_p2=None):
    """Apply noise to the data"""

    if sin_input_name == 'image':
        input_shape = np.shape(image_input)
    else:
        input_shape = np.shape(image_input)

    max_magnitude = SINFields.VALID_MAX_MAGTD[sin_input_name]
    sin_factor = (np.clip(sin_level,SINFields.SIN_LEVEL_MIN,SINFields.SIN_LEVEL_MAX) \
                  / SINFields.SIN_LEVEL_MAX) *4.5*max_magnitude
    # Set to 1.5-sigma region
    if sin_type == 'rand':
        # Get the noisy input to be added to the image 
        sin_input = sin_factor*np.random.randn(*input_shape)
        if sin_input_name == 'image':
            sin_input += image_input
        else:
            sin_input += image_input
    return sin_input


