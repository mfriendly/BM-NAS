# BM-NAS: Bilevel Multimodal Neural Architecture Search (AAAI 2022)

Yihang Yin, Siyu Huang, Xiang Zhang

## Full Paper

Please check the our arXiv version [here](https://arxiv.org/abs/2104.09379) for the full paper with supplementary.

## Requirements

The latest tested versions are:

```
pytorch==1.10.1
opencv-python==4.5.5.62
sklearn==1.10.1
tqdm
IPython
```

## Pre-trained Backbones and Pre-processed Datasets

The backbones (checkpoints) and pre-processed datasets (BM-NAS_dataset) are available at [here](https://www.aliyundrive.com/s/1c55RCPdaAo), you can download them and put them in the root directory.

## MM-IMDB Experiments

Will be available later.


## NTU RGB-D Experiments

You can just use our pre-processed dataset, but you should cite the original [NTU RGB-D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset. 

### Dataset Pre-processing

If you want to use the original one, you can follow these steps. 

First, request and download the NTU RGB+D dataset (not NTU RGB+D 120) from the official [site](https://rose1.ntu.edu.sg/dataset/actionRecognition/). We only use the **3D skeletons (body joints)** and **RGB videos** modality. 

Then, run the following script to reshape all RGB videos to 256x256 with 30 fps:
```shell
$ python datasets/prepare_ntu.py --dir=<dir of RGB videos>
```

## EgoGesture Experiments

Will be available later.

## Citation

If you find this work helpful, please kindly cite our [paper](https://arxiv.org/abs/2104.09379).

```latex
@article{yin2021bm,
  title={BM-NAS: Bilevel Multimodal Neural Architecture Search},
  author={Yin, Yihang and Huang, Siyu and Zhang, Xiang},
  journal={arXiv preprint arXiv:2104.09379},
  year={2021}
}
```
