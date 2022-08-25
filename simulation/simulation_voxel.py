#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/19 21:56
# @Author  : Leijun Ye

import os
import copy
import time
import numpy as np
import torch
from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from models.bold_model_pytorch import BOLD
import argparse
import prettytable as pt
import matplotlib.pyplot as mp
from scipy.io import loadmat, savemat
import h5py

def plot_bold(path_out, bold_real, bold_da, bold_index, lag):
    T = bold_da.shape[0]
    iteration = [i for i in range(T-lag-50)]
    assert len(bold_da.shape) == 2
    for i in bold_index:
        print("show_bold" + str(i))
        fig = mp.figure(figsize=(8, 4), dpi=500)
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(iteration, bold_real[50:T-lag, i], 'r-', label="Raw")
        ax1.plot(iteration, bold_da[50+lag:T, i], 'b-', label="Assimilation")
        mp.legend()
        mp.ylim((0.0, 0.08))
        ax1.set(xlabel='Observation time/800ms', ylabel='BOLD', title=str(i + 1))
        mp.savefig(os.path.join(path_out, "bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
        mp.close(fig)


def simulate_voxel_after_assimilation_for_rest_state(block_path, ip, noise_rate, whole_brain_info=None,
                                                     T=400, step=800,  write_path=None,
                                                     hp_path=None, re_parameter_ind=[10, 12]):
    """

    Parameters
    ----------
    ip : str
        server ip
    step : int
        total steps to simulate
    number : int
        iteration number in one step
    whole_brain_info : str
        should be specified, it contains the whole brain voxel information
    write_path: str
        which dir to write
    noise_rate: float
        noise rate in block model.
    block_path: str
        connection block setting path
    da_mode : str
        such as "rest_whole_brain" , "task_v1"
    kwargs : key
        other keyword argument contain: {re_parameter_ind, hp]
    """
    start_time = time.time()
    os.makedirs(write_path, exist_ok=True)

    block_model = block_gpu(ip, block_path, 1., force_rebase=True)
    bold = BOLD(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)

    populations_per_voxel = 2
    population_id = block_model.subblk_id.cpu().numpy()
    num_populations = int(block_model.total_subblks)
    num_voxels = num_populations // populations_per_voxel
    num_neurons = int(block_model.total_neurons)
    neurons_per_population_cpu = block_model.neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population_cpu)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    neurons_per_voxel_cpu = np.histogram(population_id, weights=neurons_per_population_cpu, bins=num_voxels, range=(0, num_populations))[0]

    tb = pt.PrettyTable()
    tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
    tb.add_row([1, "name", "whole_brain", "ensembles", 1])
    tb.add_row([2, "num_neurons", num_neurons, "voxels", num_voxels])
    tb.add_row([3, "num_populations", num_populations, "noise_rate", noise_rate])
    tb.add_row([4, "T", T, "step", step])
    print(tb)

    re_parameter_ind = np.array([int(s) for s in re_parameter_ind.split()], dtype=np.int64).reshape(-1)
    print(f"Re_parameter_ind: {re_parameter_ind}")
    
    for k in re_parameter_ind:
        population_info = np.stack(np.meshgrid(population_id, k, indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        alpha = torch.ones(num_populations, device="cuda:0") * 5
        beta = torch.ones(num_populations, device="cuda:0") * 5
        block_model.gamma_property_by_subblk(population_info, alpha, beta)

    re_parameter_time = time.time()
    print(f"\n re_parameter have Done, Cost time {re_parameter_time - start_time:.2f} ")

    hp_da = np.load(os.path.join(hp_path, "hp.npy"))
    assert hp_da.shape[1] == num_voxels
    hp_da = torch.from_numpy(hp_da.astype(np.float32)).cuda()
    population_info = np.stack(np.meshgrid(population_id, re_parameter_ind, indexing="ij"),
                               axis=-1).reshape((-1, 2))
    population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()

    file = h5py.File(whole_brain_info, 'r')
    bold_real = np.array(file['dti_rest_state']).T
    bold_real = bold_real[20:]
    bold_real = (bold_real - bold_real.mean(axis=0, keepdims=True)) / bold_real.std(axis=0, keepdims=True)  # z-score
    bold_real = 0.02 + 0.03 * (bold_real - bold_real.min()) / (bold_real.max() - bold_real.min())
    
    bold_sim = np.zeros([T, num_voxels], dtype=np.float32)
    simulate_start = time.time()
    for ii in range(T):
        print("Run simulation || ", ii)
        simulate_start_1 = time.time()
        block_model.mul_property_by_subblk(population_info, hp_da[ii, :, :].reshape(-1))
        for freqs in block_model.run(800, freqs=True, vmean=False, sample_for_show=False):
            freqs = freqs.float().cpu().numpy()
            act = np.histogram(population_id, weights=freqs, bins=num_voxels, range=(0, num_populations))[0]
            act = (act / neurons_per_voxel_cpu).reshape(-1)
            act = torch.from_numpy(act).cuda()
            out = bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
        print('act_max:', act.max(), 'act_min:', act.min(), 'act_mean:', act.mean())
        bold_sim[ii] = out.cpu().numpy()
        print("Cost Time: ", time.time() - simulate_start_1)

    lag = 4
    plot_bold(write_path, bold_real, bold_sim, np.arange(100), lag)
    r = np.zeros(num_voxels)
    for i in range(num_voxels):
        r[i] = np.corrcoef(bold_real[50 - lag:-lag, i], bold_sim[50:, i])[0, 1]
    r_mean = r.mean()
    print("Mean correlation between raw and assimilation BOLD is: ", r_mean)
    savemat(os.path.join(write_path, 'simulation_after_assim' + str(lag) + '.mat'), {'bold_real': bold_real, 'bold_sim': bold_sim, 'r': 'r', 'r_mean': r_mean})

    print(f"\n simulation have Done, Cost time {time.time() - simulate_start:.2f} ")
    print(f"\n Totally have Done, Cost time {time.time() - start_time:.2f} ")
    block_model.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_path', type=str,
                        default="test/laminar_structure_whole_brain_include_subcortical/200m_structure/n40d100/single")
    parser.add_argument('--ip', type=str, default='11.5.4.2:50051')
    parser.add_argument("--noise_rate", type=str, default="0.01")
    parser.add_argument('--re_parameter', type=bool, default=True)
    parser.add_argument("--whole_brain_info", type=str, default="./")
    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--step", type=int, default=800)
    parser.add_argument("--write_path", type=str, default="./")
    parser.add_argument("--hp_path", type=str, default="./")
    parser.add_argument("--re_parameter_ind", type=str, default="10 12")

    FLAGS, unparsed = parser.parse_known_args()
    simulate_voxel_after_assimilation_for_rest_state(FLAGS.block_path,
                                                     FLAGS.ip,
                                                     float(FLAGS.noise_rate),
                                                     whole_brain_info=FLAGS.whole_brain_info,
                                                     step=FLAGS.step,
                                                     T=FLAGS.T,
                                                     write_path=FLAGS.write_path,
                                                     hp_path=FLAGS.hp_path,
                                                     re_parameter_ind=FLAGS.re_parameter_ind)

