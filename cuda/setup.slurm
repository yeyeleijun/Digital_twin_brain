#!/bin/bash
#SBATCH -J jmdsetup
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH --mem-per-cpu=90G
date
#hostfile=./$SLURM_JOB_ID
#scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
module rm compiler/rocm/3.3
module rm compiler/rocm/2.9
module rm compiler/rocm/3.9
module rm compiler/rocm/3.5
module add compiler/cmake/3.15.6
#module add compiler/rocm/4.0.1
module add compiler/rocm/dtk-22.04.2
module list
./gencode.sh
cd 3rdparty/jsoncpp
mkdir build
cd build && rm -rf ./*
cmake ../ && make clean && make -j16
cd ../../../
#dbg=0 make -j16 all
make clean && dbg=0 make -j16 all
