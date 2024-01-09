### HAVE ADMINISTRATOR RIGHTS ON YOUR WINDOWS PC
### HAVE AN NVIDIA GRAPHICS CARD COMPATIBLE WITH CUDA >= 11.2
### USE THE ANACONDA POWERSHELL PROMPT RATHER THAN WINDOWS CMD

----------------------------------------------------------------

### Download and Install ###

- Swig for compiling python into c++: https://www.swig.org/download.html
Get the windows version if this link is deprecated : 
	http://prdownloads.sourceforge.net/swig/swigwin-4.1.1.zip


- MinGW to use gcc, which is only available on Linux and is needed to compile: https://winlibs.com/ \
Download the archive of any version :
https://winlibs.com/#download-release \
UnZip it on the desktop to get a tree like this one:\
\mingw64 \
&nbsp;&nbsp;&nbsp;\include\
&nbsp;&nbsp;&nbsp;\lib\
&nbsp;&nbsp;&nbsp;\bin

- Cuda 11.3 (check that your NVidia gpu supports this version, otherwise go to 11.8, 11.2)\
Download the local or network installer and use it:
	https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Windows


- cuDnn is a library for deep learning on your NVidia card. \
Download the archive that goes with your version of cuda (here 11.x.x) from this NVidia site: 
	https://developer.nvidia.com/rdp/cudnn-archive \
If you have the cuda 11.3.0 version here is the direct download link: https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.4/local_installers/11.x/cudnn-windows-x86_64-8.9.4.25_cuda11-archive.zip/ \
Go to your CUDA installation folder: \
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3 \
and drag all the files from the downloaded cuDnn archive into the folder (do "replace files" if Windows asks you to).


----------------------------------------------------------------


### Environment variables under Windows ###

CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3

CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3

CUDA_PATH_V11_3 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3

SWIG = C:\Users\youruser\Desktop\swig-4.1.1

PATH add (user AND all users): 
- C:\Users\youruser\Desktop\mingw64\include
- C:\Users\youruser\Desktop\mingw64\lib
- C:\Users\youruser\Desktop\mingw64\bin
- C:\Users\youruser\Desktop\swig-4.1.1

----------------------------------------------------------------


### Creation and activation of the conda environment : Pytorch 1.10.1 + cuda 11.3 + cudnn ###

> conda create -n s2anet python=3.7 pytorch==1.10.1 cudatoolkit=11.3 torchvision -c pytorch -y \
> conda activate s2anet


----------------------------------------------------------------


### S2ANET INSTALLATION ###

> git clone -b pytorch1.9 https://github.com/csuhan/s2anet.git \
> cd s2anet \
> pip install -r requirements.txt \
> python setup.py develop


----------------------------------------------------------------


### Compiling ###

> cd DOTA_devkit/polyiou \
> swig -c++ -python csrc/polyiou.i \
> python setup.py build_ext --inplace \

----------------------------------------------------------------


### Use ###

Use it like this:

> python demo/demo_inference.py model.py model.pth test_img/ test_out/ \

- model.py and model.pth are the model files.
- test_img/ is the directory where you have to put the images you want to process inference on.
- test_out/ is the directory where you will recover inferenced images with bounding boxes.