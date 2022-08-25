# -*- coding: utf-8 -*-
# @Time : 2021/8/25 22:04
# @Author : lepold
# @File : da_version_1.py


import argparse
from DA_voxel import DA_Rest_Whole_Brain_Voxel
import numpy as np
import os

block_dict = {"ip": "10.5.4.1:50051",
              "block_path": "./",
              "noise_rate": 0.01,
              "delta_t": 1.,
              "print_stat": False,
              "froce_rebase": True}
bold_dict = {"epsilon": 200,
             "tao_s": 0.8,
             "tao_f": 0.4,
             "tao_0": 1,
             "alpha": 0.2,
             "E_0": 0.8,
             "V_0": 0.02}

parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
parser.add_argument("--print_stat", type=bool, default=False)
parser.add_argument("--force_rebase", type=bool, default=True)
parser.add_argument("--block_path", type=str,
                    default="/public/home/ssct004t/project/wenyong36/dti_voxel_outlier_10m/dti_n4_d100/single")
parser.add_argument("--whole_brain_info", type=str,
                    default="/public/home/ssct004t/project/yeleijun/spiking_nn_for_brain_simulation/data/DBS/bold_rest_jianfeng.npy")
parser.add_argument("--write_path", type=str,
                    default="/public/home/ssct004t/project/wenyong36/dti_voxel_outlier_10m/dti_n4_d100/")
parser.add_argument("--T", type=int, default=450)
parser.add_argument("--noise_rate", type=float, default=0.01)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--hp_sigma", type=float, default=0.1)
parser.add_argument("--bold_sigma", type=float, default=1e-6)
parser.add_argument("--ensembles", type=int, default=100)

args = parser.parse_args()
block_dict.update(
    {"ip": args.ip, "block_path": args.block_path, "noise_rate": args.noise_rate, "print_stat": args.print_stat,
     "force_rebase": args.force_rebase})
whole_brain_info = args.whole_brain_info

da_rest = DA_Rest_Whole_Brain_Voxel(block_dict, bold_dict, whole_brain_info, steps=args.steps,
                              ensembles=args.ensembles, time=args.T, hp_sigma=args.hp_sigma, bold_sigma=args.bold_sigma)

#real_parameter = np.array([[6.1644921e-03, 2.9690875e-02], [6.1644921e-03, 2.9690875e-02]], dtype=np.float32).reshape((1, 4))
real_parameter = np.array([0.00618016, 0.07027743,  0.00618016, 0.07027743], dtype=np.float32).reshape((1, 4))
# real_parameter = np.array([0.00904649, 0.05379498], dtype=np.float32).reshape((1, 2))
# real_parameter = np.array([0.00808016, 0.04027743], dtype=np.float32).reshape((1, 2))
print("real parameter:", real_parameter)
para_ind = np.array([10, 12], dtype=np.int64)
os.makedirs(args.write_path, exist_ok=True)
da_rest.run(real_parameter, para_ind, whole_brain_info=whole_brain_info, write_path=args.write_path)

