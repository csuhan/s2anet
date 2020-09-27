## Installation

### Requirements

- Linux (Windows is not officially supported)
- Python 3.5+ (Python 2 is not supported)
- PyTorch **1.3** or higher
- CUDA 9.0 or higher
- NCCL 2
- GCC(G++) **4.9** or higher
- [mmcv](https://github.com/open-mmlab/mmcv)==**0.2.14**

Note that some cuda extensions, e.g., ```box_iou_rotated``` and ```nms_rotated``` require pytorch>=1.3 and gcc>=4.9.

We have tested the following versions of OS and softwares:

- OS:  CentOS 7.2
- CUDA: 10.0
- NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
- GCC(G++): 4.9
- pytorch: 1.3.1

### Install

a. Create a conda virtual environment and activate it.

```shell
conda create -n s2anet python=3.7 -y
conda activate s2anet
```

b. Install PyTorch stable or nightly and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch=1.3 torchvision cudatoolkit=10.0 -c pytorch
```

c. Clone the s2anet repository.

```shell
git clone https://github.com/csuhan/s2anet.git
cd s2anet
```

d. Install s2anet (other dependencies will be installed automatically).

```shell
pip install -r requirements.txt

python setup.py develop
# or "pip install -v -e ."

# install orn
cd mmdet/ops/orn
python setup.py build_ext --inplace
```
Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

### Install DOTA_devkit
```
sudo apt-get install swig
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```

### Prepare datasets

For DOTA, we provide scripts to split the original images into chip images (e.g., 1024*1024). Please refer to [DOTA_devkit/prepare_dota1_ms.py](https://github.com/csuhan/s2anet/blob/master/DOTA_devkit/prepare_dota1_ms.py).

For HRSC2016, please refer to [DOTA_devkit/prepare_hrsc2016.py](https://github.com/csuhan/s2anet/blob/master/DOTA_devkit/prepare_hrsc2016.py).

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── dota_1024_s2anet
│   │   ├── trainval_split
│   │   │    │─── images
│   │   │    │─── labelTxt
│   │   │    │─── trainval.json
│   │   ├── test_split
│   │   │    │─── images
│   │   │    │─── test.json
│   ├── HRSC2016 (optinal)
│   │   ├── Train
│   │   │    │─── images
│   │   │    │─── labelTxt
│   │   │    │─── trainval.json
│   │   ├── Test
│   │   │    │─── images
│   │   │    │─── test.json
```


### Multiple versions

If there are more than one mmdetection on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
