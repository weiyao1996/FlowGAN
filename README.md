# FlowGAN

This repository is the official implementation of Flow-based GAN for 3D Point Cloud Generation from a Single Image (BMVC 2022)

[[Project page](https://bmvc2022.mpi-inf.mpg.de/569/)]

## Abstract
Generating a 3D point cloud from a single 2D image is of great importance for 3D scene understanding applications. To reconstruct the whole 3D shape of the object shown in the image, the existing deep learning based approaches use either explicit or implicit generative modeling of point clouds, which, however, suffer from limited quality. In this work, we aim to alleviate this issue by introducing a hybrid explicit-implicit generative modeling scheme, which inherits the flow-based explicit generative models for sampling point clouds with arbitrary resolutions while improving the detailed 3D structures of point clouds by leveraging the implicit generative adversarial networks (GANs). We evaluate on the large-scale synthetic dataset ShapeNet, with the experimental results demonstrating the superior performance of the proposed method. In addition, the generalization ability of our method is demonstrated by performing on cross-category synthetic images as well as by testing on real images from PASCAL3D+ dataset.

![image](https://github.com/weiyao1996/weiyao1996.github.io/blob/master/img/bmvc2022.png)  

## Dependencies 
Environment: Ubuntu 20.04.4 LTS, CUDA 11.3.
· Pytorch == 1.1.0  
· Python == 3.6.7  
· Numpy == 1.16.3  
· opencv == 4.1.0  
· PyMaxflow == 1.2.12  
· scipy  == 1.2.1  
· Cython == 0.29.13  

## Datasets

### ShapeNetCore.v1

### PASCAL3D+
  
## Usage  
### Visualization
We adopt [Mitsuba renderer](https://github.com/mitsuba-renderer/mitsuba2) for the visualization of 3D point clouds. Please refer to the original repository.

## Acknowledgement
We build our code based on the following codebases, we thank their authors and recommand citing their works as well if you find this code is useful for your work. 

[PointFlow](https://github.com/stevenygd/PointFlow) [Yang et al., ICCV'19]
[DPF-Nets](https://github.com/Regenerator/dpf-nets) [Klokov et al., ECCV'20]
[MixNFs](https://github.com/janisgp/go_with_the_flows) [Postels et al., 3DV'21]
