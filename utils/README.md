
## Generating data augmented and noisy samples

Augmented and noisy data samples for evaluation can be obtained by running scripts in `~/utils` path.
1. `data_aug.py` - Update the corresponding paths in this file to obtain the augmented data. This file generates images, labels, velodyne, calib folders with the augmented data. Only images are augmented and rest of the corresponding data is copied from the main location. A file called `aug.txt` is created that stores IDs of the scenes that were used for augmentation.  

2. `create_noise_data.py` - Update the paths. This file creates noisy image data. 
