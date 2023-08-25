# FlowGAN

The official implementation of _Flow-based GAN for 3D Point Cloud Generation from a Single Image_ (BMVC 2022)

[[Project page](https://bmvc2022.mpi-inf.mpg.de/569/)]

![image](https://github.com/weiyao1996/weiyao1996.github.io/blob/master/img/bmvc2022.png)  
  
## Usage

### Data Preparation

_ShapeNetCore.v1_ and _PASCAL3D+_ can be downloaded via the following link. If you would like to apply this work to other datasets, please refer to [DPF-Nets](https://github.com/Regenerator/dpf-nets) for pre-processing scripts.

### Visualization
We adopt [Mitsuba renderer](https://github.com/mitsuba-renderer/mitsuba2) for the visualization of 3D point clouds. Please refer to the original repository.

##  Citation
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
We build our code based on the following codebases, we thank their authors and recommand citing their works as well if you find this code is useful for your work. 

[PointFlow](https://github.com/stevenygd/PointFlow) [Yang et al., ICCV'19]
[DPF-Nets](https://github.com/Regenerator/dpf-nets) [Klokov et al., ECCV'20]
[MixNFs](https://github.com/janisgp/go_with_the_flows) [Postels et al., 3DV'21]
