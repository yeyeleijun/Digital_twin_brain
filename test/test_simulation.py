# -*- coding: utf-8 -*- 
# @Time : 2022/8/20 21:04 
# @Author : lepold
# @File : test_simulation.py


import os
import argparse
import numpy as np
from simulation.simulation import simulation


def get_args():
    parser = argparse.ArgumentParser(description="Model simulation")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--block_dir", type=str, default=None)
    parser.add_argument("--write_path", type=str, default=None)
    parser.add_argument("--aal_info_path", type=str, default=None)
    parser.add_argument("--hp_after_da_path", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()
    return args


def rest_column_simulation(args):
    """
    simulation of resting brain at micro-column version.

    Parameters
    ----------
    args: dict
        some needed parameter in simulation object.

    Returns
    -------

    """
    block_path = os.path.join(args.block_dir, "single")
    model = simulation(args.ip, block_path, dt=1., route_path=None, column=True, print_info=True, vmean_option=True,
                       sample_option=True, name=args.name, write_path=args.write_path)
    aal_region = np.load(args.aal_info_path)['aal_region']
    population_base = np.load(os.path.join(args.block_dir, "supplementary_info", "population_base.npy"))
    model.sample(aal_region, population_base)
    model(observation_time=100, hp_index=[10], hp_path=args.hp_after_da_path)


if __name__ == "__main__":
    args = get_args()
    rest_column_simulation(args)
