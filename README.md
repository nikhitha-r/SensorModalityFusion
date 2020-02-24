# Low-level Sensor Fusion for 3D Object detection

## Installation of required packages

We recommend you to use conda. create a conda environment and install the following packages. Just use the following command line.
```bash
conda create --name avod python=3.5 && conda activate avod && conda install matplotlib -y && conda install numpy -y && conda install -c conda-forge opencv -y && conda install -c conda-forge pandas -y && conda install -c conda-forge pillow -y && conda install -c conda-forge scipy -y && conda install -c anaconda scikit-learn -y && conda install -c anaconda tensorflow-gpu==1.3.0 -y && conda install -c open3d-admin open3d -y
```
If you want to install the packages one by one, you can also refer to the following command lines.
```bash
conda create --name avod python=3.5
conda activate avod
conda install matplotlib -y
conda install numpy -y
conda install -c conda-forge opencv -y
conda install -c conda-forge pandas -y
conda install -c conda-forge pillow -y
conda install -c conda-forge scipy -y
conda install -c anaconda scikit-learn -y
conda install -c anaconda tensorflow-gpu==1.3.0 -y
conda install -c open3d-admin open3d -y
```

## Training and evaluation of models

Please refer to the README file in the subdirectory `avod_moe`, click [instruction for using the model](./avod_moe/README.md) to directly access the README.

## Visualization of final prediction results

### How to use the code for visualization?

The code for visualization is in the subdirectory `./visualization`. Inside the folder, there are two more subfolder `vis_3d` and `vis_weights`. To visualize the final 3d bounding box prediction, you should use the code in `vis_3d`, to visualize the weights for different region proposals, you should use the code in `vis_weights`.

#### To visualize 3d bounding box prediction

If you want to visualize the 3d bounding box prediction in the point cloud, you need to do the following steps:

1. Use the following code to run the visualization.

```bash
cd ./visualization/vis_3d
python vis_point_cloud.py -d ./path_to_data_folder -s scene_index -p prediction_folder -t objectness_threshold
```

you can also use the following lines for help with the arguments:

```bash
python vis_point_cloud.py --help
```

if you just want to see the effect without your own prediction files folder, just run the following to see the effect:

```bash
python vis_point_cloud.py
```

#### To visualize weights for MoE

To visualize the MoE, you need to do the following steps:

1. Specify the path to weights data folder `weights`. How to generate `weights`? Please see [this README](./avod_moe/README.md).
2. Use the following code to run the visualization.

For help, run the following:
```bash
cd ./visualization/vis_weights
python feature_weighs_visualization.py -h # read the help to learn how to use it
```

To visualize scene 000008 the image features and bev features in pca:
```bash
python feature_weights_visualization.py -id path_to_image_folder -s 000008 -d directory_of_data -bf -if
```

To draw bounding boxes of scene 000008 on the image and bev:
```bash
python feature_weights_visualization.py -id path_to_image_folder -s 000008 -d directory_of_data -bb -ib
```
