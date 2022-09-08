#!/bin/bash
#SBATCH -J wy36da
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=dcu:4
#SBATCH --mem 90G
#SBATCH -x j10r2n07,e13r4n02,b01r2n17,e13r1n02

date
export PYTHONPATH=$PYTHONPATH:/public/home/ssct004t/project/zenglb/Digital_twin_brain
export PYTHONPATH=$PYTHONPATH:/public/home/ssct004t/project/zenglb/Digital_twin_brain/cuda/python
module switch compiler/rocm/4.0.1
mpirun python DataAssimilation_demo.py \
  --task='rest' \
  --ip=11.5.4.1:50051 \
  --bold_path='/public/home/ssct004t/project/wenyong36/dti_voxel_subfeng_1e8_22703/single/rest_state_afterzscoreV2.npy' \
  --block_path='/public/home/ssct004t/project/yeleijun/Digital_twin_brain/PD-sub401/dti_distribution_100m/ensembles/' \
  --path_out='/public/home/ssct004t/project/wenyong36/Digital_twin_brain/toymodel/' \
  --para_ind='10' \
  --gui_real='0.01935443 0.00052911 0.09493671 0.02250352' \
  --bold_sigma=1e-8 \
  --hp_sigma=0.25 \
  --hp_range_rate='4 2' \
  --noise_rate=0.01\
  --ensembles=10 \
  --gui_alpha=5 \
  --label='DTB_restfengvoxel_b25_b8_h5_a5'
