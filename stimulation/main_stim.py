#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/25 12:20
# @Author  : Leijun Ye

import argparse
from stimulation import StimulationVoxel
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Stimulation")
parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
parser.add_argument("--block_path", type=str, default="./")
parser.add_argument("--write_path", type=str, default="./")
parser.add_argument("--whole_brain_info", type=str, default="./")
parser.add_argument("--print_info", type=bool, default=False)
parser.add_argument("--vmean_option", type=bool, default=False)
parser.add_argument("--sample_option", type=bool, default=False)
parser.add_argument("--imean_option", type=bool, default=False)
parser.add_argument("--step", type=int, default=800)
parser.add_argument("--observation_time", type=int, default=300)
parser.add_argument("--hp_path", type=str, default="./")
parser.add_argument("--hp_index", type=str, default="10 12")

args = parser.parse_args()
hp_index = np.array([int(s) for s in args.hp_index.split()]).reshape(-1)
kwargs = {"name": "test_simulation",
          "write_path": args.write_path,
          "print_info": args.print_info,
          "vmean_option": args.vmean_option,
          "sample_option": args.sample_option,
          "imean_option": args.imean_option}
stimulation_voxel = StimulationVoxel(args.ip, args.block_path, route_path=None, **kwargs)
stimulation_voxel.run(step=args.step, observation_time=args.observation_time, hp_path=args.hp_path, hp_index=hp_index, whole_brain_info=args.whole_brain_info)