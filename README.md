# DGNNDS_MFL

This repository contains the source code for developing a multi-lesion diagnosis method for fundus images with a fusion deep graph neural network（DGNNDS_MFL）. This method extracts the lesion features in the fundus image with a convolutional neural network, and then inputs the extracted features into a graph neural network to learn the correlation between different lesions. Finally, the graph is fused. The input features and output features of the neural network are diagnosed by a multi-label classifier.

This system provides a practical strategy for Automatic diagnosis of fundus lesions.


## Environment
- Ubuntu: 18.04 lts
- Python 3.7.8
- Pytorch 1.6.0
- NVIDIA GPU + CUDA_10.0 CuDNN_7.5

## Installation

- pytorch: 1.8.0
- wheel: 0.34.2
- yaml: 0.2.5
- scipy: 1.5.2
- joblib: 0.16.0
- opencv-python: 4.3.0.38
- scikit-image: 0.17.2
- numpy: 1.19.1
- sikit-learn：0.23.2
- mmcv-full: 1.2.4

## Install dependencies
pip install -r requirements.txt



## Models && adjacency matrices
You can download the data files and our best models [here](https://pan.baidu.com/s/1xUu3Frta_lH0riMYoBVh7Q)
password: f113

## Usage
git clone https://github.com/Ophthalmology-CAD/DGNNDS_MFL.git

cd DGNNDS_MFL

mkdir data  (download the data needed and put here)


- The file "main_resnet_lesion. sh" in /DGNNDS_MFL is used for model training of ResNet fusion depth map neural network in multi-lesion diagnosis of fundus.
- The file "main_densenet_lesion.sh" in /DGNNDS_MFL is used for model training of DenseNet fusion depth map neural network in multi-lesion diagnosis of fundus.


## Citation
Please feel free to contact us for any questions or comments: Jiewei Jiang, E-mail: jiangjw924@126.com or Liufei Guo, E-mail: liufei.guo@foxmail.com.


## Contributing
For any questions, feel free to open an issue or contact us (tianshuichen@gmail.com & xumx7@mail2.sysu.edu.cn & huixlu@mail2.sysu.edu.cn)







 









