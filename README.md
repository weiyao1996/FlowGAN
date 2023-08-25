# FlowGAN

The official implementation of _Flow-based GAN for 3D Point Cloud Generation from a Single Image_ (BMVC 2022)

### [Project page](https://bmvc2022.mpi-inf.mpg.de/569/)

![image](https://github.com/weiyao1996/weiyao1996.github.io/blob/master/img/bmvc2022.png)  
  
## Usage

### Data Preparation

First of all, download ShapeNet dataset from [ShapeNetCore.v1](https://shapenet.org/) and [3D-R2N2](http://3d-r2n2.stanford.edu/). Please refer to [DPF-Nets](https://github.com/Regenerator/dpf-nets) for pre-processing scripts, because the data should be stored in hdf5 format. Here, we provide *Airplane* category via [GoogleDrive](https://drive.google.com/drive/folders/1hkWJykin2kJWZKdakgtg2N2s9MDIRT1T?usp=sharing) | [百度网盘](https://pan.baidu.com/s/14M2KBOg-n_AbeOlNmZ3YHw) (提取码2uk2).

### Train

All configurations can be found in `configs/`. The trained models are saved under `logs/models/`.

```
python train.py ./configs/airplane.yaml svr_model_02691156 20 0.000256
python train.py ./configs/airplane.yaml svr_model_02691156 30 0.000064 --resume
```

### Evaluation

The generated point clouds will be stored under `logs/` in .h5 format.

```
python evaluate.py ./configs/airplane.yaml svr_model_02691156 test 2500 2500 reconstruction --weights_type learned_weights --reps 1 --f1_threshold_lst 0.0001 --cd --f1 --emd --unit_scale_evaluation
```

### Visualization

We adopt [Mitsuba renderer](https://github.com/mitsuba-renderer/mitsuba2) for the visualization of 3D point clouds. `$path_mitsuba` is supposed to be `.../mitsuba2/build/dist/`.
```
python render_mitsuba.py --path_h5 $path_h5 --path_png $path_png --path_mitsuba $path_mitsuba --name_png $name_png --indices 1 2 3
```

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
We build our code based on the following codebases, many thanks to the contributors.

[PointFlow](https://github.com/stevenygd/PointFlow) [Yang et al., ICCV'19]
[DPF-Nets](https://github.com/Regenerator/dpf-nets) [Klokov et al., ECCV'20]
[MixNFs](https://github.com/janisgp/go_with_the_flows) [Postels et al., 3DV'21]
