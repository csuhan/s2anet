### Align Deep Features for Oriented Object Detection

![](demo/network.png)

> **[Align Deep Features for Oriented Object Detection](https://arxiv.org/abs/2008.09397)**,            
> Jiaming Han, Jian Ding, Jie Li, Gui-Song Xia,        
> arXiv preprint ([arXiv:2008.09397](https://arxiv.org/abs/2008.09397))  

The repo is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

Two versions are provided here: [Original version](https://github.com/csuhan/s2anet/tree/original_version) and [v20210104](https://github.com/csuhan/s2anet).

### Introduction
The past decade has witnessed significant progress on detecting objects in aerial images that are often distributed with large scale variations and arbitrary orientations. However most of existing methods rely on heuristically defined anchors with different scales, angles and aspect ratios and usually suffer from severe misalignment between anchor boxes and axis-aligned convolutional features, which leads to the common inconsistency between the classification score and localization accuracy. To address this issue, we propose a **Single-shot Alignment Network** (S<sup>2</sup>A-Net) consisting of two modules: a Feature Alignment Module (FAM) and an Oriented Detection Module (ODM). The FAM can generate high-quality anchors with an Anchor Refinement Network and adaptively align the convolutional features according to the corresponding anchor boxes with a novel Alignment Convolution. The ODM first adopts active rotating filters to encode the orientation information and then produces orientation-sensitive and orientation-invariant features to alleviate the inconsistency between classification score and localization accuracy. Besides, we further explore the approach to detect objects in large-size images, which leads to a better speed-accuracy trade-off. Extensive experiments demonstrate that our method can achieve state-of-the-art performance on two commonly used aerial objects datasets (*i.e.*, DOTA and HRSC2016) while keeping high efficiency.


## Changelog

* **2021-01-01.** **Big changes!** Following mmdetection v2, we made a lot of changes to our code. Our original code contains many unnecessary functions and inappropriate modifications. So we modified related codes, e.g, dataset preprocessing and loading, unified function names, iou calculator between OBBs, and evaluation. Besides, we also implement a **Cascade S<sup>2</sup>A-Net**. Compared with previous versions, the updated version is more straightforward and easy to understand. 


## Benchmark and model zoo
* **[Original implementation](https://github.com/csuhan/s2anet/tree/original_version) on DOTA**

|Model          |    Backbone     |    MS  |  Rotate | Lr schd  | Inf time (fps) | box AP (ori./now) | Download|
|:-------------:| :-------------: | :-----:| :-----: | :-----:  | :------------: | :----: | :---------------------------------------------------------------------------------------: |
|RetinaNet      |    R-50-FPN     |   -     |   -    |   1x     |      16.0      |  68.05/68.40 |        [model](https://drive.google.com/file/d/1ZUc8VUDOkTnVA1FFNuINm2U39h0anLPm/view?usp=sharing)        |
|S<sup>2</sup>A-Net         |    R-50-FPN     |   -     |   -    |   1x     |      16.0      |  74.12/73.99|    [model](https://drive.google.com/file/d/19gwDSzCx0uToqI9LyeAg_yXNLgK3sbl_/view?usp=sharing)    |
|S<sup>2</sup>A-Net         |    R-50-FPN     |   ✓     |  ✓     |   1x     |      16.0      |  79.42 |    [model](https://drive.google.com/file/d/1W-JPfoBPHdOxY6KqsD0ZhhLjqNBS7UUN/view?usp=sharing)    |
|S<sup>2</sup>A-Net         |    R-101-FPN    |   ✓     |  ✓     |   1x     |      12.7      |  79.15 |    [model](https://drive.google.com/file/d/1Jkbx-WvKhokEOlWR7WLKxTpH4hDTp-Tb/view?usp=sharing)            |

*Note that the mAP reported here is a little different from the original paper. All results are reported on DOTA-v1.0 *test-dev*. 
All checkpoints here are trained with the [Original version](https://github.com/csuhan/s2anet/tree/original_version), and **not compatible** with the updated version.

* **20210104 updated version**

|Model                      |Data           |    Backbone     |    MS  |  Rotate | Lr schd  | box AP | Download|
|:-------------:            |:-------------:| :-------------: | :-----:| :-----: | :-----:  | :----: | :---------------------------------------------------------------------------------------: |
|RetinaNet                  |HRSC2016       |    R-50-FPN     |   -    |   ✓    |   6x     |  81.63 |    [cfg](configs/hrsc2016/retinanet_obb_r50_fpn_6x_hrsc2016.py) [model](https://drive.google.com/file/d/1vb3dTsNnyM1EBG81oi0TPfVqwTYAX2WO/view?usp=sharing) [log](https://drive.google.com/file/d/16h1YjoCNLvyja4ik6_unOKwobwAZohfY/view?usp=sharing)        |
|CS<sup>2</sup>A-Net-1s     |HRSC2016       |    R-50-FPN     |   -    |   ✓    |   4x     |  84.58 |    [cfg](configs/hrsc2016/cascade_s2anet_1s_r50_fpn_4x_hrsc2016.py) [model](https://drive.google.com/file/d/1Nu0Xa9DhsQfP5nUic1LVxI9013-xo1w_/view?usp=sharing) [log](https://drive.google.com/file/d/1F50yegKejAxQ9SQg9oxUkaVFmyZX5f0f/view?usp=sharing)        |
|CS<sup>2</sup>A-Net-2s     |HRSC2016       |    R-50-FPN     |   -    |   ✓    |   3x     |  89.96 |    [cfg](configs/hrsc2016/cascade_s2anet_2s_r50_fpn_3x_hrsc2016.py) [model](https://drive.google.com/file/d/1Xa2rDg9-LHvfiRmpCY7Aoow61vIcqSQE/view?usp=sharing) [log](https://drive.google.com/file/d/1vH_VyCVvcoNDga63fU-13Fzkp-nBq95c/view?usp=sharing)        |
|S<sup>2</sup>A-Net         |HRSC2016       |    R-101-FPN    |   -    |   ✓    |   3x     |  90.00 |    [cfg](configs/hrsc2016/s2anet_r101_fpn_3x_hrsc2016.py) [model](https://drive.google.com/file/d/1T11d37BJXA__8t99CttRPHYqqOKtJOVw/view?usp=sharing)    |
|CS<sup>2</sup>A-Net-1s     |DOTA           |    R-50-FPN     |   -    |   -    |   1x     |  69.06 |    [cfg](configs/dota/cascade_s2anet_1s_r50_fpn_1x_dota.py) [model](https://drive.google.com/file/d/13S9dFMVmwQeaojB5mVa6Kw_I0UMAjkct/view?usp=sharing) [log](https://drive.google.com/file/d/1H4_IqNWjLUgyCYLe0xrBsuf3-JQNI8z0/view?usp=sharing)        |
|CS<sup>2</sup>A-Net-2s     |DOTA           |    R-50-FPN     |   -    |   -    |   1x     |  73.67 |    [cfg](configs/dota/cascade_s2anet_2s_r50_fpn_1x_dota.py) [model](https://drive.google.com/file/d/1OOHcMsBzV0OxOSCxhVLO8Vc2dmFKuCYq/view?usp=sharing) [log](https://drive.google.com/file/d/19Eos7bdTrA9VduPTH4AJDvn8SrrijTBU/view?usp=sharing)        |
|S<sup>2</sup>A-Net         |DOTA           |    R-50-FPN     |   -    |   -    |   1x     |  74.04 |    [cfg](configs/dota/s2anet_r50_fpn_1x_dota.py) [model](https://drive.google.com/file/d/1OyKKhc1rgexf8otCex6XGiJIsb8fZBNQ/view?usp=sharing)    |

CS<sup>2</sup>A-Net-*n*s indicates Cascade S<sup>2</sup>A-Net with *n* stages. For more information, please refer to [cascade_s2anet.md](docs/cascade_s2anet.md)

The checkpoints of S<sup>2</sup>A-Net are converted from the original version.

If you cannot get access to Google Drive, BaiduYun download link can be found [here](https://pan.baidu.com/s/1vsRDUD09RMC1hr9yU7Gviw) with extracting code **ABCD**.


## Installation

Please refer to [install.md](docs/INSTALL.md) for installation and dataset preparation.


## Getting Started

Please see [getting_started.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.



## Citation

```
@article{han2020align,
  title = {Align Deep Features for Oriented Object Detection},
  author = {Han, Jiaming and Ding, Jian and Li, Jie and Xia, Gui-Song},
  journal = {arXiv preprint arXiv:2008.09397},
  year = {2020}
}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}

@InProceedings{Ding_2019_CVPR,
  author = {Ding, Jian and Xue, Nan and Long, Yang and Xia, Gui-Song and Lu, Qikai},
  title = {Learning RoI Transformer for Oriented Object Detection in Aerial Images},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}

@article{chen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
