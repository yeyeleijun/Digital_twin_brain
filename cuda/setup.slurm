#!/bin/bash
#SBATCH -J setup
#SBATCH -p kshdexclu04
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH --mem-per-cpu=90G

date
module rm compiler/rocm/2.9
module add compiler/rocm/4.0.1
make clean && make -j16 all
