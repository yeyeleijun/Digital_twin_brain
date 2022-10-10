# -*- coding: utf-8 -*- 
# @Time : 2021/12/10 11:20 
# @Author : lepold
# @File : simulate_22703_largescale.py


import os
import time
import numpy as np
import torch
from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from models.bold_model_pytorch import BOLD
import argparse
from scipy.io import loadmat
import prettytable as pt
import pandas as pd
import matplotlib.pyplot as plt

v_th = -50


def pretty_print(content):
    screen_width = 80
    text_width = len(content)
    box_width = text_width + 6
    left_margin = (screen_width - box_width) // 2
    print()
    print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
    print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
    print(' ' * left_margin + '|' + content + ' ' * (box_width - text_width - 4) + '|')
    print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
    print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
    print()


def np_move_avg(a, n=10, mode="valid"):
    if a.ndim > 1:
        tmp = []
        for i in range(a.shape[1]):
            tmp.append(np.convolve(a[:, i], np.ones((n,)) * 1000 / n, mode=mode))
        tmp = np.stack(tmp, axis=1)
    else:
        tmp = np.convolve(a, np.ones((n,)) * 1000 / n, mode=mode)
    return tmp


def torch_2_numpy(u, is_cuda=True):
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out

def sample_new(aal_region, neurons_per_population_base, populations_id, choice_all=None):
    subcortical = np.array([37, 38, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78], dtype=np.int64) - 1  # region index from 0
    subblk_base = [0]
    tmp = 0
    for i in range(len(aal_region)):
        if aal_region[i] in subcortical:
            subblk_base.append(tmp + 2)
            tmp = tmp + 2
        else:
            subblk_base.append(tmp + 8)
            tmp = tmp + 8
    subblk_base = np.array(subblk_base)
    sample_idx = np.empty([90 * 300, 4], dtype=np.int64)
    for i in np.arange(90):
        print("sampling for region: ", i)
        if choice_all is None:
            choice = np.random.choice(np.where(aal_region == i)[0], 1)
        else:
            choice = choice_all[300 * i]
        if i in subcortical:
            index = np.where(np.logical_and(populations_id < (choice+1) * 10, populations_id>=choice * 10))[0]
            sub_populations = populations_id[index]
            assert len(np.unique(sub_populations)) ==2
            popu1 = index[np.where(sub_populations % 10 == 6)]
            neurons = np.concatenate([np.arange(neurons_per_population_base[id], neurons_per_population_base[id+1]) for id in popu1])
            sample1 = np.random.choice(neurons, size=240, replace=False)
            popu2 = index[np.where(sub_populations % 10 == 7)]
            neurons = np.concatenate(
                [np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1]) for id in popu2])
            sample2 = np.random.choice(neurons, size=60, replace=False)
            sample = np.concatenate([sample1, sample2])
            sub_blk = np.concatenate(
                [np.ones_like(sample1) * (subblk_base[choice]), np.ones_like(sample2) * (subblk_base[choice] +1)])[:,
                      None]
            # print("sample_shape", sample.shape)
            sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
            sample = np.concatenate([sample, sub_blk, np.ones((300, 1)) * i], axis=-1)
            sample_idx[300 * i:300 * (i + 1), :] = sample
        else:
            index = np.where(np.logical_and(populations_id < (choice + 1) * 10, populations_id >= choice * 10))[0]
            sub_populations = populations_id[index]
            assert len(np.unique(sub_populations)) == 8
            sample_this_region = []
            sub_blk = []
            for yushu, size in zip(np.arange(2, 10), np.array([80, 20, 80, 20, 20, 10, 60, 10])):
                popu1 = index[np.where(sub_populations % 10 == yushu)]
                neurons = np.concatenate(
                    [np.arange(neurons_per_population_base[id], neurons_per_population_base[id + 1]) for id in popu1])
                sample1 = np.random.choice(neurons, size=size, replace=False)
                sample_this_region.append(sample1)
                sub_blk.append(np.ones(size) * (subblk_base[choice] + yushu))
            sample_this_region = np.concatenate(sample_this_region)
            sub_blk = np.concatenate(sub_blk)
            sample = np.stack([sample_this_region, np.ones(300) * choice, sub_blk, np.ones(300) * i], axis=-1)
            sample_idx[300 * i:300 * (i + 1), :] = sample
    return sample_idx.astype(np.int64)


def sample_in_cortical_and_subcortical(aal_region, neurons_per_population_base):
    base = neurons_per_population_base
    subcortical = np.array([37, 38, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78], dtype=np.int64) - 1  # region index from 0
    subblk_base = [0]
    tmp = 0
    for i in range(len(aal_region)):
        if aal_region[i] in subcortical:
            subblk_base.append(tmp + 2)
            tmp = tmp + 2
        else:
            subblk_base.append(tmp + 8)
            tmp = tmp + 8
    subblk_base = np.array(subblk_base)
    sample_idx = np.empty([92 * 300, 4], dtype=np.int64)
    # the (, 0): neuron idx; (, 1): voxel idx, (,2): subblk(population) idx, (, 3) voxel idx belong to which brain region
    for i in np.arange(92):
        # print("sampling for voxel: ", i)
        choice = np.random.choice(np.where(aal_region == i)[0], 1)
        if i in subcortical:
            sample1 = np.random.choice(
                np.arange(start=base[subblk_base[choice]], stop=base[subblk_base[choice] + 1], step=1), 240,
                replace=False)
            sample2 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 1], stop=base[subblk_base[choice] + 2], step=1), 60,
                replace=False)
            sample = np.concatenate([sample1, sample2])
            sub_blk = np.concatenate(
                [np.ones_like(sample1) * subblk_base[choice], np.ones_like(sample2) * (subblk_base[choice] + 1)])[:,
                      None]
            # print("sample_shape", sample.shape)
            sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
            sample = np.concatenate([sample, sub_blk, np.ones((300, 1)) * i], axis=-1)
            sample_idx[300 * i:300 * (i + 1), :] = sample
        else:
            sample1 = np.random.choice(
                np.arange(start=base[subblk_base[choice]], stop=base[subblk_base[choice] + 1], step=1), 80,
                replace=False)
            sample2 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 1], stop=base[subblk_base[choice] + 2], step=1), 20,
                replace=False)
            sample3 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 2], stop=base[subblk_base[choice] + 3], step=1), 80,
                replace=False)
            sample4 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 3], stop=base[subblk_base[choice] + 4], step=1), 20,
                replace=False)
            sample5 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 4], stop=base[subblk_base[choice] + 5], step=1), 20,
                replace=False)
            sample6 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 5], stop=base[subblk_base[choice] + 6], step=1), 10,
                replace=False)
            sample7 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 6], stop=base[subblk_base[choice] + 7], step=1), 60,
                replace=False)
            sample8 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 7], stop=base[subblk_base[choice] + 8], step=1), 10,
                replace=False)
            sample = np.concatenate([sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8])
            # print("sample_shape", sample.shape)
            sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
            sub_blk = np.concatenate(
                [np.ones(j) * (subblk_base[choice] + k) for k, j in enumerate([80, 20, 80, 20, 20, 10, 60, 10])])[:,
                      None]
            sample = np.concatenate([sample, sub_blk, np.ones((300, 1)) * i], axis=-1)
            sample_idx[300 * i:300 * (i + 1), :] = sample
    return sample_idx.astype(np.int64)


def simulate_hetero_model_after_assimilation_for_rest_state(block_path, ip, noise_rate, rout_path=None,
                                                            step=400, number=800, whole_brain_info=None,
                                                            write_path=None, **kwargs):
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
        should be specifed, it contain the whole brain voxel information
    write_path: str
        which dir to write
    noise_rate: float
        noise rate in block model.
    block_path: str
        connection block setting path
    kwargs : key
        other keyword argument contain: {re_parameter_ind, hp]
    """
    start_time = time.time()
    os.makedirs(write_path, exist_ok=True)

    if rout_path=="None":
        rout_path = None
    print(block_path)
    block_model = block_gpu(ip, block_path, 1.,
                            route_path=rout_path,
                            force_rebase=False)
    # route_path = "/public/home/ssct004t/project/spiking_nn_for_brain_simulation/route_dense_10k_22703.json"

    populations = block_model.subblk_id  # such as [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, ...]
    populations_ccpu = populations.cpu().numpy()
    total_populations = int(block_model.total_subblks)
    total_neurons = int(block_model.total_neurons)

    # 包含卡间的populations的重叠 比如卡一population[1, 2], 卡二[1, 2],合起来[1, 2, 1, 2]
    neurons_per_population = block_model._neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    populations_cpu = block_model._subblk_id[block_model._subblk_idx].cpu().numpy()

    # neurons_per_population = block_model.neurons_per_subblk.cpu().numpy()
    # neurons_per_population_base = np.add.accumulate(neurons_per_population)
    # neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)

    num_cortical_voxel = int(np.load(whole_brain_info)["divide_point"])
    aal_region = np.load(whole_brain_info)["aal_region"]
    num_voxel = len(aal_region)
    assert len(aal_region) == num_voxel
    num_subcortical_voxel = num_voxel - num_cortical_voxel
    neurons_per_voxel, _ = np.histogram(populations_cpu, weights=neurons_per_population, bins=num_voxel,
                                        range=(0, num_voxel * 10))

    tb = pt.PrettyTable()
    tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
    tb.add_row([1, "name", "large_scale_rest_brain", "ensembles", 1])
    tb.add_row([2, "neurons", total_neurons, "voxels", num_voxel])
    tb.add_row([3, "cortical_voxel", num_cortical_voxel, "subcortical_voxel", num_subcortical_voxel])
    tb.add_row([4, "toal_populations", total_populations, "noise_rate", noise_rate])
    tb.add_row([5, "step", step, "number", number])
    print(tb)


    pretty_print("Init Noise")
    population_info = torch.stack(torch.meshgrid(populations, torch.tensor([0], dtype=torch.int64, device="cuda:0")),
                                  dim=-1).reshape((-1, 2))
    alpha = torch.ones(total_populations, device="cuda:0") * noise_rate * 1e8
    beta = torch.ones(total_populations, device="cuda:0") * 1e8
    block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    pretty_print("Init gui for 4 channels")
    gui_laminar = np.array([[0.00659512, 0.00093751, 0.1019024, 0.00458985],
                            [0.01381911, 0.00196363, 0.18183651, 0.00727698],
                            [0.00659512, 0.00093751, 0.1019024, 0.00458985],
                            [0.01381911, 0.00196363, 0.18183651, 0.00727698],
                            [0.00754673, 0.00106148, 0.09852575, 0.00431849],
                            [0.0134587, 0.00189199, 0.15924831, 0.00651926],
                            [0.00643689, 0.00091055, 0.10209763, 0.00444712],
                            [0.01647443, 0.00234132, 0.21505809, 0.00796669],
                            [0.00680198, 0.00095797, 0.06918744, 0.00324963],
                            [0.01438906, 0.00202573, 0.14674303, 0.00587307]], dtype=np.float64)
    gui_voxel = np.array([0.00618016, 0.00086915, 0.07027743, 0.00253291], dtype=np.float64)
    ss, edges = np.histogram(populations_ccpu, bins=num_voxel, range=(0, num_voxel * 10))
    cor = np.where(ss == 8)[0]
    subcor = np.where(ss == 2)[0]
    assert len(cor) + len(subcor) == len(ss)
    for i in np.arange(2, 10, 1):
        cor_layer = edges[cor] + i
        ll = len(cor_layer)
        for j in np.arange(0, 4, 1):
            population_info = np.stack(np.meshgrid(cor_layer, np.array([10+j]), indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            alpha = torch.ones(ll, device="cuda:0") * 1e8 * gui_laminar[i, j]
            beta = torch.ones(ll, device="cuda:0") * 1e8
            block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)
    for i in np.arange(6, 8, 1):
        cor_layer = edges[subcor] + i
        ll = len(cor_layer)
        for j in np.arange(0, 4, 1):
            population_info = np.stack(np.meshgrid(cor_layer, np.array([10+j]), indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            alpha = torch.ones(ll, device="cuda:0") * 1e8 * gui_voxel[j]
            beta = torch.ones(ll, device="cuda:0") * 1e8
            block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    # pretty_print("Init Gamma distribution for gui")
    # re_parameter_ind = kwargs.get("re_parameter_ind", 10)
    # re_parameter_ind = torch.tensor([re_parameter_ind], dtype=torch.int64).cuda()
    # print(f"Re_parameter_ind: {re_parameter_ind}")
    # try:
    #     hp_initial = kwargs["hp_initial"]
    # except KeyError:
    #     raise KeyError("hyper parameter is not stated in re_parameter module")
    # # hp_total = np.load(os.path.join(hp_initial, "hp_new.npy"))
    # hp_total = np.load(os.path.join(hp_initial, "hp.npy"))
    # assert hp_total.shape[1] == total_populations
    # hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
    # population_info = torch.stack(torch.meshgrid(populations, re_parameter_ind),
    #                               dim=-1).reshape((-1, 2))
    # gamma = torch.ones(total_populations, device="cuda:0") * 5.
    # block_model.gamma_property_by_subblk(population_info, gamma, gamma, debug=False)
    # block_model.mul_property_by_subblk(population_info, hp_total[0, :])
    # hp_total = hp_total[1:, ]
    # total_T = hp_total.shape[0]

    pretty_print("Set sample idx")
    sample_file = np.load('/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/simulate/simulation_10b_cby_generation_nodebug/result_file/sample_idx.npy')
    choice_all = sample_file[:, 1]
    # choice_all = None
    sample_idx = load_if_exist(sample_new, os.path.join(write_path, "sample_idx"),
                               aal_region=aal_region,
                               neurons_per_population_base=neurons_per_population_base, populations_id=populations_cpu, choice_all=choice_all)
    sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
    sample_number = sample_idx.shape[0]
    block_model.set_samples(sample_idx)
    load_if_exist(lambda: block_model.neurons_per_subblk.cpu().numpy(), os.path.join(write_path, "blk_size"))

    pretty_print("Discard Transition Init")
    for j in range(4):
        t13 = time.time()
        temp_fre = []
        for return_info in block_model.run(800, freqs=True, vmean=True, sample_for_show=True):
            Freqs, vmean, spike, vi = return_info
            temp_fre.append(Freqs)
        temp_fre = torch.stack(temp_fre, dim=0)
        t14 = time.time()
        print(
            f"{j}th step number {number}, max fre: {torch.max(torch.mean(temp_fre / block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t14 - t13:.1f}")
    firerate = torch.mean(temp_fre / block_model.neurons_per_subblk.float() * 1000, dim=0)
    firerate = torch_2_numpy(firerate)
    np.savez(os.path.join(write_path, "firerate.npz"), firerate=firerate, populations=populations_ccpu)

    Init_end = time.time()
    bold1 = BOLD(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")
    step = step - 1
    bolds_out = np.zeros([step, num_voxel], dtype=np.float32)

    pretty_print("Begin Simulation")
    # print(f"\nTotal time is {total_T}, we only simulate {step}\n")
    simulate_start = time.time()
    for ii in range((step - 1) // 50 + 1):
        nj = min(step - ii * 50, 50)
        FFreqs = np.zeros([nj, number, total_populations], dtype=np.uint32)
        Vmean = np.zeros([nj, number, total_populations], dtype=np.float32)
        Spike = np.zeros([nj, number, sample_number], dtype=np.uint8)
        Noise_Spike = np.zeros([nj, number, sample_number], dtype=np.uint8)
        Vi = np.zeros([nj, number, sample_number], dtype=np.float32)

        for j in range(nj):
            i = ii * 50 + j
            t13 = time.time()
            temp_fre = []
            temp_spike = []
            temp_noise_spike = []
            temp_vi = []
            temp_vmean = []
            # block_model.mul_property_by_subblk(population_info, hp_total[i, :])
            bold_out1 = None
            for return_info in block_model.run(number, freqs=True, vmean=True, sample_for_show=True):
                Freqs, vmean, spike, vi = return_info
                freqs = Freqs.cpu().numpy()
                act, _ = np.histogram(populations_ccpu, weights=freqs, bins=num_voxel, range=(0, num_voxel * 10))
                act = (act / neurons_per_voxel).reshape(-1)
                act = torch.from_numpy(act).cuda()
                noise_spike = spike & (torch.abs(vi - v_th) / 50 >= 1e-5)
                spike &= (torch.abs(vi - v_th) / 50 < 1e-5)
                temp_fre.append(Freqs)
                temp_vmean.append(vmean)
                temp_spike.append(spike)
                temp_noise_spike.append(noise_spike)
                temp_vi.append(vi)
                bold_out1 = bold1.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
            bolds_out[i, :] = torch_2_numpy(bold_out1)
            temp_fre = torch.stack(temp_fre, dim=0)
            temp_vmean = torch.stack(temp_vmean, dim=0)
            temp_spike = torch.stack(temp_spike, dim=0)
            temp_noise_spike = torch.stack(temp_noise_spike, dim=0)
            temp_vi = torch.stack(temp_vi, dim=0)
            t14 = time.time()
            FFreqs[j] = torch_2_numpy(temp_fre)
            Vmean[j] = torch_2_numpy(temp_vmean)
            Spike[j] = torch_2_numpy(temp_spike)
            Noise_Spike[j] = torch_2_numpy(temp_noise_spike)
            Vi[j] = torch_2_numpy(temp_vi)
            print(
                f"{i}th step, max fre: {torch.max(torch.mean(temp_fre / block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t14 - t13:.1f}")
        np.save(os.path.join(write_path, "spike_after_assim_{}.npy".format(ii)), Spike)
        np.save(os.path.join(write_path, "noise_spike_after_assim_{}.npy".format(ii)), Noise_Spike)
        np.save(os.path.join(write_path, "vi_after_assim_{}.npy".format(ii)), Vi)
        np.save(os.path.join(write_path, "freqs_after_assim_{}.npy".format(ii)), FFreqs)
        np.save(os.path.join(write_path, "vmean_after_assim_{}.npy".format(ii)), Vmean)
    np.save(os.path.join(write_path, "bold_after_assim.npy"), bolds_out)

    pretty_print(f"Simulation have Done, Cost time {time.time() - simulate_start:.2f} ")
    pretty_print(f"Totally have Done, Cost time {time.time() - start_time:.2f} ")
    block_model.shutdown()


def draw(log_path, freqs, block_size, sample_idx, write_path, name_path, bold_path, real_bold_path, vmean_path,
         vsample):
    log = np.load(log_path, )
    Freqs = np.load(freqs)
    vmean = np.load(vmean_path)
    vsample = np.load(vsample)
    if len(log.shape) > 2:
        log = log.reshape([-1, log.shape[-1]])
        vsample = vsample.reshape([-1, vsample.shape[-1]])
        Freqs = Freqs.reshape([-1, Freqs.shape[-1]])
        vmean = vmean.reshape((-1, vmean.shape[-1]))
    os.makedirs(write_path, exist_ok=True)
    block_size = np.load(block_size)
    name = loadmat(name_path)['AAL']
    bold_simulation = np.load(bold_path)
    # bold_simulation = bold_simulation[30:, ]
    T, voxels = bold_simulation.shape
    bold_y = np.load(real_bold_path)["rest_bold"]
    bold_y = bold_y.T
    bold_y = 0.02 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
    real_bold = bold_y[:T, :voxels]

    # the (, 0): neuron idx; (, 1): voxel idx, (,2): subblk(population) idx, (, 3) voxel idx belong to which brain region
    property = np.load(sample_idx)
    unique_voxel = np.unique(property[:, 1])

    def run_voxel(i):
        print("draw voxel: ", i)
        idx = np.where(property[:, 1] == i)[0]
        subpopu_idx = np.unique(property[idx, 2])
        sub_log = log[:, idx]
        sub_vsample = vsample[:, idx]
        region = property[idx[0], 3]
        if region < 90:
            sub_name = name[region // 2][0][0] + '-' + ['L', 'R'][region % 2]
        elif region == 91:
            sub_name = 'LGN-L'
        else:
            sub_name = 'LGN-R'
        if real_bold is None:
            sub_real_bold = None
        else:
            sub_real_bold = real_bold[:, i]
        sub_sim_bold = bold_simulation[:, i]
        sub_vmean = vmean[:, subpopu_idx]

        _, split = np.unique(property[idx, 2], return_counts=True)
        split = np.add.accumulate(split)
        split = np.insert(split, 0, 0)
        index = np.unique(property[idx, 2])
        fire_rate = Freqs[:, index]
        return write_path, block_size, i, sub_log, split, sub_name, fire_rate, index, sub_real_bold, sub_sim_bold, sub_vsample, sub_vmean

    n_nlocks = [process_block(*run_voxel(i)) for i in unique_voxel]

    table = pd.DataFrame({'Name': [b[1] for b in n_nlocks],
                          'Visualization': ['\includegraphics[scale=0.04125]{IMG%i.JPG}' % (i + 1,) for i in
                                            range(len(n_nlocks))],
                          'Neuron Sample': ['\includegraphics[scale=0.275]{log_%i.png}' % (i,) for i in
                                            range(len(n_nlocks))],
                          'Layer fr': ['\includegraphics[scale=0.275]{frpoup_%i.png}' % (i,) for i in
                                   range(len(n_nlocks))],
                          'Bold': ['\includegraphics[scale=0.275]{bold_%i.png}' % (i,) for i in
                                   range(len(n_nlocks))],
                          'FR pdf': ['\includegraphics[scale=0.275]{statis_%i.png}' % (i,) for i in
                                            range(len(n_nlocks))],
                          'CV': ['\includegraphics[scale=0.275]{cv_%i.png}' % (i,) for i in
                                                 range(len(n_nlocks))],
                          })
    column_format = '|l|c|c|c|c|c|c|c|'

    with open(os.path.join(write_path, 'chart.tex'), 'w') as f:
        f.write("""
                    \\documentclass[varwidth=25cm]{standalone}
                    \\usepackage{graphicx}
                    \\usepackage{longtable,booktabs}
                    \\usepackage{multirow}
                    \\usepackage{multicol}
                    \\begin{document}
                """)
        f.write(table.to_latex(bold_rows=True, longtable=True, multirow=True, multicolumn=True, escape=False,
                               column_format=column_format))

        f.write("""
                    \\end{document}
                """)

    print('-')


def process_block(write_path, real_block, block_i, log, split, name, fire_rate, subblk_index, bold_real=None,
                  bold_sim=None, sub_vsample=None, sub_vmean=None, time=1200, slice_window=800, stride=200):
    block_size = log.shape[-1]
    real_block_size = real_block[subblk_index]
    names = ['L2/3', 'L4', 'L5', 'L6']

    # frequence = log.sum() * 1000 / log.shape[0] / activate_idx.shape[0]
    frequence_map = torch.from_numpy(log.astype(np.float32)).transpose(0, 1).unsqueeze(1)
    frequence_map = 1000 / slice_window * torch.conv1d(frequence_map, torch.ones([1, 1, slice_window]),
                                                       stride=stride).squeeze().transpose(0, 1).numpy()
    fig_fre = plt.figure(figsize=(4, 4), dpi=500)
    fig_fre.gca().hist(frequence_map.reshape([-1]), 100, density=True)
    fig_fre.gca().set_yscale('log')
    fig_fre.savefig(os.path.join(write_path, "statis_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_fre)

    # fire_rate_ = (fire_rate[-time:, ::2] + fire_rate[-time:, 1::2]) / (real_block_size[::2] + real_block_size[1::2])
    # fire_rate_ = np_move_avg(fire_rate_, n=5, mode="same")
    # fig_frequence = plt.figure(figsize=(4, 4), dpi=500)
    # ax1 = fig_frequence.add_subplot(1, 1, 1)
    # ax1.grid(False)
    # ax1.set_xlabel('time(ms)')
    # ax1.set_ylabel('Instantaneous fr(hz)')
    # for i in range(fire_rate_.shape[1]):
    #     ax1.plot(fire_rate_[:, i], label=names[i])
    # ax1.legend(loc='best')
    # fig_frequence.savefig(os.path.join(write_path, "fr_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    # plt.close(fig_frequence)

    activate_idx = (log.sum(0) > 0).nonzero()[0]  # log.shape=[100*800, 300]
    cvs = []
    for i in activate_idx:
        out = log[:, i].nonzero()[0]
        if out.shape[0] >= 3:
            fire_interval = out[1:] - out[:-1]
            cvs.append(fire_interval.std() / fire_interval.mean())

    cv = np.array(cvs)
    fig_cv = plt.figure(figsize=(4, 4), dpi=500)
    fig_cv.gca().hist(cv, 100, range=(0, 2), density=True)
    fig_cv.savefig(os.path.join(write_path, "cv_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_cv)

    fig = plt.figure(figsize=(4, 4), dpi=500)
    axes = fig.add_subplot(1, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    fire_rate_ = np_move_avg(fire_rate[-2000:, ], n=10, mode="valid")
    if len(split) > 3:
        df = pd.DataFrame(fire_rate_, columns=['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I'])
    else:
        df = pd.DataFrame(fire_rate_, columns=['E', 'I'])
    df.plot.box(vert=False, showfliers=False, widths=0.2, color=color, ax=axes)
    fig.savefig(os.path.join(write_path, "frpopu_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    valid_idx = np.where(log.mean(axis=0) > 0.001)[0]
    instanous_fr = log[-2000:-800, valid_idx].mean(axis=1)
    instanous_fr = np_move_avg(instanous_fr, 10, mode="valid")
    length = len(instanous_fr)
    fig = plt.figure(figsize=(8, 4), dpi=500)
    ax1 = fig.add_subplot(1, 1, 1, frameon=False)
    ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax1.grid(False)
    ax1.set_xlabel('time(ms)')
    axes = fig.add_subplot(2, 1, 1)
    if len(split) > 3:
        sub_vmean = sub_vmean[-2000:-800, :] * np.array(
            [0.24355972, 0.05152225, 0.25995317, 0.07025761, 0.11709602, 0.03512881, 0.18501171, 0.03747072])
        sub_vmean = sub_vmean.sum(axis=-1)
        for t in range(8):
            x, y = log[-2000:-800, split[t]:split[t + 1]].nonzero()
            if t % 2 == 0:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="blue")
            else:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="red")
        names = ['L2/3', 'L4', 'L5', 'L6']
        names_loc = split[:-1][::2]
    else:
        sub_vmean = sub_vmean[-2000:-800, :] * np.array([0.8, 0.2])
        sub_vmean = sub_vmean.sum(axis=-1)
        for t in range(2):
            x, y = log[-2000:-800, split[t]:split[t + 1]].nonzero()

            if t % 2 == 0:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="blue")
            else:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="red")
        names = ["E", "I"]
        names_loc = split[:-1]
    axes.set_title("fre of spiking neurons: %.2f" % instanous_fr.mean())
    axes.set_xlim((0, length))
    axes.set_ylim((0, block_size))
    plt.yticks(names_loc, names)
    axes.invert_yaxis()
    axes.set_aspect(aspect=1)
    axes = fig.add_subplot(2, 1, 2)
    axes.plot(instanous_fr, c="black")
    fig.savefig(os.path.join(write_path, "log_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    # fig_vi = plt.figure(figsize=(8, 4), dpi=500)
    # ax1 = fig_vi.add_subplot(1, 1, 1, frameon=False)
    # ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # ax1.grid(False)
    # ax1.set_xlabel('time(ms)')
    # axes = fig_vi.add_subplot(2, 1, 1)
    # sub_vsample = sub_vsample[-2000:-800, :]
    # axes.imshow(sub_vsample.T, vmin=-65, vmax=-50, cmap='jet', origin="lower")
    # axes = fig_vi.add_subplot(2, 1, 2)
    # axes.plot(sub_vmean, c="r")
    # fig_vi.savefig(os.path.join(write_path, "vi_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    # plt.close(fig_vi)

    fig_bold = plt.figure(figsize=(4, 4), dpi=500)
    if bold_real is not None:
        fig_bold.gca().plot(np.arange(len(bold_real)), bold_real, "r-", label="real")
    fig_bold.gca().plot(np.arange(len(bold_sim)), bold_sim, "b-", label="sim")
    fig_bold.gca().set_ylim((0., 0.08))
    fig_bold.gca().legend(loc="best")
    fig_bold.gca().set_xlabel('time')
    fig_bold.gca().set_ylabel('bold')
    fig_bold.savefig(os.path.join(write_path, "bold_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_bold)

    return real_block_size.sum(), name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_path', type=str,
                        default="/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_comp/dti_2k_10G/single")
    parser.add_argument('--ip', type=str, default='11.5.4.2:50051')
    parser.add_argument("--noise_rate", type=str, default="0.01")
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--number", type=int, default=800)
    parser.add_argument("--whole_brain_info", type=str,
                        default="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz")
    parser.add_argument("--route_path", type=str,
                        default="/public/home/ssct004t/project/spiking_nn_for_brain_simulation/dti_comp/dti_2k_10G/route_dense.json")
    parser.add_argument("--write_path", type=str, default="./simulation_largescale_rest")
    parser.add_argument("--hp_initial", type=str,
                        default="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/da_simulation/rest_whole_brain")
    parser.add_argument("--re_parameter_ind", type=int, default=10)

    parser.add_argument("--real_bold_path", type=str,
                        default="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz")
    parser.add_argument("--log_name", type=str, default="spike_after_assim_0.npy")
    parser.add_argument("--freqs_name", type=str, default="freqs_after_assim_0.npy")
    parser.add_argument("--block_size_name", type=str, default="blk_size.npy")
    parser.add_argument("--sample_idx_name", type=str, default="sample_idx.npy")
    parser.add_argument("--name_path", type=str,
                        default="/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/aal_names.mat")
    parser.add_argument("--sim_bold_path", type=str, default="bold_after_assim.npy")
    parser.add_argument("--vmean_path", type=str, default="vmean_after_assim_0.npy")
    parser.add_argument("--vsample_path", type=str, default="vi_after_assim_0.npy")

    FLAGS, unparsed = parser.parse_known_args()
    write_file_path = os.path.join(FLAGS.write_path, "result_file")
    simulate_hetero_model_after_assimilation_for_rest_state(FLAGS.block_path,
                                                            FLAGS.ip,
                                                            float(FLAGS.noise_rate),
                                                            step=FLAGS.step,
                                                            rout_path=FLAGS.route_path,
                                                            number=FLAGS.number,
                                                            whole_brain_info=FLAGS.whole_brain_info,
                                                            write_path=write_file_path,
                                                            hp_initial=FLAGS.hp_initial,
                                                            re_parameter_ind=FLAGS.re_parameter_ind, )

    write_fig_path = os.path.join(FLAGS.write_path, "fig")
    log_path = os.path.join(write_file_path, FLAGS.log_name)
    freqs = os.path.join(write_file_path, FLAGS.freqs_name)
    block_size = os.path.join(write_file_path, FLAGS.block_size_name)
    sample_idx = os.path.join(write_file_path, FLAGS.sample_idx_name)
    simulation_bold = os.path.join(write_file_path, FLAGS.sim_bold_path)
    vmean = os.path.join(write_file_path, FLAGS.vmean_path)
    vsample = os.path.join(write_file_path, FLAGS.vsample_path)
    draw(log_path, freqs, block_size, sample_idx, write_fig_path, FLAGS.name_path, simulation_bold,
         FLAGS.real_bold_path, vmean, vsample)
