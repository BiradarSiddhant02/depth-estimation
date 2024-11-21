# **Depth Estimation**

A solution for Monocular Depth Estimation problem which will be applied for development of an autonomous UAV for obstacle detection and path planning

The model implements a Deep Neural Network of [UNET architecture](https://arxiv.org/abs/1505.04597) First proposed by Olaf Ronneberger, Philipp Fischer, Thomas Brox in the year 2015. The paper uses UNET model for segmentation of biomedical images.

The model has two major parts. The Encode and the Decoder. The encoder has been replaced with a [DenseNet](https://arxiv.org/abs/1608.06993) model in our solution. 

The Dataset used to train the model is the [NYU-depth-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) dataset. The dataset has:
- 1449 densely labeled pairs of aligned RGB and depth images
- 464 new scenes taken from 3 cities
- 407,024 new unlabeled frames

# **Acknowledgments**

Much of the inspiration for this codebase is taken from https://github.com/alinstein/Depth_estimation. I extend my gratitude to the original author.

[UNET Architecture](https://arxiv.org/abs/1505.04597): Olaf Ronneberger, Philipp Fischer, and Thomas Brox for their pioneering work on the UNET architecture, as described in their 2015 paper UNET: Convolutional Networks for Biomedical Image Segmentation.

[NYU-depth-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html): Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. He for their significant contributions to DenseNet, as detailed in their 2016 paper Densely Connected Convolutional Networks.

# Usage

install miniconda and create an environment

Note: all models should be stored in the ***models*** directory

clone the repository
```
git clone https://github.com/BiradarSiddhant02/depth-estimation.git
cd depth-estimation
```

install dependencies. Ubuntu 22.04 with up-to-date Nvidia drivers is recommended
```
pip install -r requirments.txt
```
Install pytorch seperately according to your system from [pytorch.org](pytorch.org)

### **To run inference on a single image:**
```
python image_inference --input <input mode> --output_folder <folder to store outputs> --model <path to model>
```
**input mode:**
- ***camera*** to take an image from the device's camera
- ***path/to/image*** to take image from a specific path

### **To run inference through camera feed**
```
python video_inference <path/to/model>
```

### **RPi Setup**
```
pip install -r requirements_aarch64.txt
```
create a folder named inputs and upload an image.
```
python inference/rpi_inference.py    \
    --input <path/to/input/image>    \
    --output <path/to/output/folder> \
    --model <path/to/model>
```
download the output image and view