"""
Sample function to create noisy image data
"""
import noise_utils
import cv2
import numpy as np
import random
from random import randint
from random import sample

image_dir = "/storage/remote/atcremers61/w0017/KITTI/PointRCNN/KITTI/object/training/image_2/"
noisy_img_dir = "/storage/remote/atcremers61/w0017/test_noise123/"
# Open the val file
with open("/usr/prakt/w0017/try_avod_moe/avod_moe/avod/split/val.txt", "r") as valFile:
    num_set = sum(1 for line in valFile)
    # Number of noisy data
    noisy_num = 1500
    #lines = valFile.read().splitlines()
with open("/usr/prakt/w0017/try_avod_moe/avod_moe/avod/split/val.txt", "r") as valFile:
    # Create an aug file to keep a track of the ids of the data samples that were modified.
    with open("/usr/prakt/w0017/try_avod_moe/avod_moe/avod/split/aug1.txt", "w") as noisyFile:
        num_list = []
        for i in valFile:
            num_list.append(i)
        image_id = "9100"

        # Loop through the number of noisy data
        for i in range(0, noisy_num):
            # Select a random id
            lineSelect = random.randint(0, num_set-1)
            random_line = num_list[lineSelect].strip()
            noisyFile.write(random_line)
            noisyFile.write("\n")
           
            # Read the image 
            orig_image = cv2.imread(image_dir + random_line.strip() + ".png", 1)
            # Modify the image data
            image_mod = noise_utils.genSINtoInputs(orig_image,
                   sin_type='rand',
                   sin_level=1,
                   sin_input_name='image',
                   mask_2d=None,
                   frame_calib_p2=None)

            image_id = int(image_id) + 1
            # Save the new noisy data
            cv2.imwrite(noisy_img_dir + "00" + str(image_id) + ".png", image_mod)

