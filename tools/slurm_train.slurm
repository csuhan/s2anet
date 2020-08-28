#!/usr/bin/env bash

#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH -o train_s2anet_r50_fpn_1x.log

module load scl/gcc4.9
module load nvidia/cuda/10.0
nvidia-smi
./tools/dist_train.sh \
  configs/dota/s2anet_r50_fpn_1x.py 4