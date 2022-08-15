import os
import argparse
import time
import numpy as np
from DataAssimilation import *


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
    parser.add_argument("--task", type=str, default="rest")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--is_cuda", type=bool, default=True)
    parser.add_argument("--block_path", type=str, default=None)
    parser.add_argument("--bold_path", type=str, default=None)
    parser.add_argument("--path_out", type=str, default=None)
    parser.add_argument("--bold_idx_path", type=str, default=None)
    parser.add_argument("--para_ind", type=str, default="10 12")
    parser.add_argument("--hp_range_rate", type=str, default="4 2")
    parser.add_argument("--solo_rate", type=float, default=0.5)
    parser.add_argument("--noise_rate", type=float, default=0.01)
    parser.add_argument("--gui_alpha", type=int, default=5)
    parser.add_argument("--I_alpha", type=int, default=1)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--hp_sigma", type=float, default=1)
    parser.add_argument("--bold_sigma", type=float, default=1e-8)
    parser.add_argument("--ensembles", type=int, default=100)
    parser.add_argument("--sample_path", type=str, default=None)
    parser.add_argument("--gui_real", type=str, default="6.1644921e-03 8.9986715e-04 2.9690875e-02 1.9053727e-03")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--I_ext", type=float, default=1.)
    parser.add_argument("--task_bold_path", type=str, default=None)
    parser.add_argument("--w_path", type=str, default=None)
    parser.add_argument("--gui_label", type=str, default=None)
    parser.add_argument("--gui_path", type=str, default=None)
    parser.add_argument("--I_path", type=str, default=None)
    return parser


def regular_dict(**kwargs):
    return kwargs


def rest_demo(args):
    # make dirs
    property_index = np.array([int(s) for s in args.para_ind.split()]).reshape(-1)
    text_para_ind = '_'.join([str(i) for i in property_index.flatten()])
    hp_range_rate = np.array([float(s) for s in args.hp_range_rate.split()]).reshape(2, -1)
    text_hp_range_rate = '_'.join([str(i) for i in hp_range_rate.flatten()])
    path_out = args.path_out + args.label + text_para_ind + text_hp_range_rate + str(args.solo_rate) + str(
        args.ensembles) + '/assimilation/'
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(path_out + 'figure/', exist_ok=True)
    # get real bold signal
    bold_real = get_bold_signal(args.bold_path, b_min=0.02, b_max=0.05, lag=0)
    print('=============================prepare DA configuration=========================================')
    # da_rest
    da_rest_parameter = regular_dict(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    da_rest = DataAssimilation(args.block_path, args.ip, column=False, **da_rest_parameter)
    da_rest.clear_mode()

    # gui boundary initialize
    gui_real = np.array([float(s) for s in args.gui_real.split()]).reshape(-1, 1)
    gui_number = len(property_index)
    gui_low = gui_real[property_index - 10] / hp_range_rate[0]  # shape = gui_number
    gui_high = gui_real[property_index - 10] * hp_range_rate[1]
    gui_low = torch.tensor(gui_low, dtype=torch.float32).reshape(1, gui_number).repeat(da_rest.single_voxel_number, 1)
    gui_high = torch.tensor(gui_high, dtype=torch.float32).reshape(1, gui_number).repeat(da_rest.single_voxel_number, 1)
    # da_rest initialize
    gui = da_rest.hp_random_initialize(gui_low, gui_high, gui_number)
    da_rest.da_property_initialize(property_index, args.gui_alpha, gui)
    da_rest.get_hidden_state(steps=1e3)
    # da_rest run
    da_rest.run(bold_real, path_out)


def task_demo(args):
    pass


if __name__ == "__main__":
    parser = get_args()
    if parser.parse_args().task == 'rest':
        rest_demo(parser.parse_args())
    if parser.parse_args().task == 'task':
        task_demo(parser.parse_args())
