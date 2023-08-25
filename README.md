# FlowGAN

The official implementation of _Flow-based GAN for 3D Point Cloud Generation from a Single Image_ (BMVC 2022)

### [Project page](https://bmvc2022.mpi-inf.mpg.de/569/)

![image](https://github.com/weiyao1996/weiyao1996.github.io/blob/master/img/bmvc2022.png)  
  
## Usage

### Data Preparation

First of all, download ShapeNet dataset from [ShapeNetCore.v1](https://shapenet.org/) and [3D-R2N2](http://3d-r2n2.stanford.edu/). Then, please refer to [DPF-Nets](https://github.com/Regenerator/dpf-nets) for pre-processing scripts, because the data should be stored in hdf5 format. Here, we provide *ShapeNet_Airplane* via [GoogleDrive](https://drive.google.com/drive/folders/1hkWJykin2kJWZKdakgtg2N2s9MDIRT1T?usp=sharing) | [百度网盘](https://pan.baidu.com/s/14M2KBOg-n_AbeOlNmZ3YHw) (提取码2uk2).

### Visualization
We adopt [Mitsuba renderer](https://github.com/mitsuba-renderer/mitsuba2) for the visualization of 3D point clouds. Please refer to the original repository.

##  Citation

Please cite our work if you find this code is useful in your research.
```
@inproceedings{Wei_2022_BMVC,
author    = {Yao Wei and George Vosselman and Michael Ying Yang},
title     = {Flow-based GAN for 3D Point Cloud Generation from a Single Image},
booktitle = {33rd British Machine Vision Conference 2022, {BMVC} 2022, London, UK, November 21-24, 2022},
publisher = {{BMVA} Press},
year      = {2022},
url       = {https://bmvc2022.mpi-inf.mpg.de/0569.pdf}
}
```

## Acknowledgement
We build our code based on the following codebases, we are very grateful to the authors.

[PointFlow](https://github.com/stevenygd/PointFlow) [Yang et al., ICCV'19]
[DPF-Nets](https://github.com/Regenerator/dpf-nets) [Klokov et al., ECCV'20]
[MixNFs](https://github.com/janisgp/go_with_the_flows) [Postels et al., 3DV'21]
