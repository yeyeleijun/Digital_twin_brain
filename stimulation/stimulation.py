#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/18 13:32
# @Author  : Leijun Ye

import os
import copy
import time
import numpy as np
import torch
from default_param import bold_params, v_th
from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block
from models.bold_model_pytorch import BOLD
from utils.pretty_print import pretty_print, table_print
from utils.helpers import load_if_exist, torch_2_numpy
import h5py


class StimulationVoxel:

    @staticmethod
    def merge_data(a, weights=None, bins=10, range=None):
        """
        merge a dataset to a desired shape. such as merge different populations to
        a voxel to calculate its size.

        Parameters
        ----------
        a : ndarray
            Input data. It must be a flattened array.

        weights: ndarray
            An array of weights, of the same shape as `a`

        bins : int
            such as the number of voxels in this simulation object

        range: (float, float)
            The lower and upper range of the bins, for micro-column,
            its upper range should be divided by 10, and for voxel should be divided by 2.

        Returns
        ----------
        merged data : array
            The values of the merged result.

        """
        return np.histogram(a, weights=weights, bins=bins, range=range)[0]

    def __init__(self, ip: str, block_path: str, route_path=None, **kwargs):
        self.block_model = block(ip, block_path, 1., route_path=route_path)
        self.bold = BOLD(**bold_params)

        self.populations_per_voxel = 2  # for voxels
        self.population_id = self.block_model.subblk_id
        self.population_id_cpu = self.population_id.cpu().numpy()
        self.num_populations = int(self.block_model.total_subblks)
        self.num_voxels = self.num_populations // self.populations_per_voxel
        self.num_neurons = int(self.block_model.total_neurons)
        self.num_neurons_per_population = self.block_model.neurons_per_subblk
        self.num_neurons_per_population_cpu = self.num_neurons_per_population.cpu().numpy()
        self.num_neurons_per_voxel_cpu = self.merge_data(self.population_id_cpu,
                                                         weights=self.num_neurons_per_population_cpu,
                                                         bins=self.num_voxels, range=(0, self.num_populations))

        self.write_path = kwargs.get("write_path", './')
        os.makedirs(self.write_path, exist_ok=True)
        self.name = kwargs.get('name', "stimulation")
        self.print_info = kwargs.get("print_info", False)
        self.vmean_option = kwargs.get("vmean_option", False)
        self.sample_option = kwargs.get("sample_option", False)
        self.imean_option = kwargs.get("imean_option", False)

    def clear_mode(self):
        """
        Make this simulation to a clear state with no printing and only default freqs and bold as the output info.
        Returns
        -------

        """
        self.sample_option = False
        self.vmean_option = False
        self.imean_option = False
        self.print_info = False

    def update(self, param_index, param):
        """
        Use a distribution of gamma(alpha, beta) to update neuronal params,
        where alpha = hyper_param * 1e8, beta = 1e8.

        Thus, the obtained distribution is approximately a delta distribution.
        In other words, we use the same value to update a certain set of parameters of neurons.

        Parameters
        ----------
        param_index: int
            indicate the column index among 21 attributes of LIF neuron.

        param: float

        """
        print(f"update {param_index}th attribute, to value {param:.3f}\n")
        population_info = torch.stack(
            torch.meshgrid(self.population_id, torch.tensor([param_index], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        alpha = torch.ones(self.num_populations, device="cuda:0") * param * 1e8
        beta = torch.ones(self.num_populations, device="cuda:0") * 1e8
        self.block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    def gamma_initialize(self, param_index, alpha=5., beta=5.):
        """
        Use a distribution of gamma(alpha, beta) to update neuronal params,
        where alpha = beta = 5.

        Parameters
        ----------

        param_index: list
            indicate the column index among 21 attributes of LIF neuron.

        alpha: float, default=5.

        beta: float, default=5.

        """

        print(f"gamma_initialize {param_index}th attribute, to value gamma({alpha}, {beta}) distribution\n")
        population_info = torch.stack(
            torch.meshgrid(self.population_id, torch.tensor(param_index, dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        alpha = torch.ones(self.num_populations, device="cuda:0") * alpha
        beta = torch.ones(self.num_populations, device="cuda:0") * beta
        self.block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    def evolve(self, step, vmean_option=False, sample_option=False, imean_option=False) -> tuple:
        """

        The block evolve one TR time, i.e, 800 ms as default setting.

        Parameters
        ----------
        step: int, default=800
            iter number in one observation time point.

        sample_option: bool, default=True
            if True, block.run will contain freqs statistics, Tensor: shape=(num_populations, ).

        vmean_option: bool, default=False
            if True, block.run will contain vmean statistics, Tensor: shape=(num_populations, ).

        Returns
        ----------
        freqs: torch.Tensor
            total freqs of different populations, shape=(step, num_populations).

        vmean: torch.Tensor
            average membrane potential of different populations, shape=(step, num_populations).

        sample_spike:: torch.Tensor
            the spike of sample neurons, shape=(step, num_sample_voxels).
        sample_v: torch.Tensor
            the membrane potential of sample neurons, shape=(step, num_sample_voxels).
        bold_out: torch.Tensor
            the bold out of voxels at the last step in one observation time poitn, shape=(num_voxels).

            .. versionadded:: 1.0

        """
        total_res = []

        for return_info in self.block_model.run(step, freqs=True, vmean=vmean_option, sample_for_show=sample_option,
                                                imean=imean_option):
            Freqs, *others = return_info
            print(Freqs.shape)
            total_res.append(return_info)

            freqs = Freqs.cpu().numpy()
            print(return_info.shape)
            print(freqs.shape)
            print(self.population_id_cpu.shape)
            act = self.merge_data(self.population_id_cpu, freqs, self.num_voxels, (0, self.num_populations))
            act = (act / self.num_neurons_per_voxel_cpu).reshape(-1)
            act = torch.from_numpy(act).cuda()
            bold_out = self.bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
        temp_freq = torch.stack([x[0] for x in total_res], dim=0)
        out = (temp_freq,)
        if vmean_option:
            temp_vmean = torch.stack([x[1] for x in total_res], dim=0)
            out += (temp_vmean,)
        if sample_option:
            if vmean_option:
                temp_vsample = torch.stack([x[3] for x in total_res], dim=0)
                temp_spike = torch.stack([x[2] for x in total_res], dim=0)
                temp_spike &= (torch.abs(temp_vsample - v_th) / 50 < 1e-5)
                out += (temp_spike, temp_vsample,)
            else:
                temp_vsample = torch.stack([x[2] for x in total_res], dim=0)
                temp_spike = torch.stack([x[1] for x in total_res], dim=0)
                temp_spike &= (torch.abs(temp_vsample - v_th) / 50 < 1e-5)
                out += (temp_spike, temp_vsample,)
        if imean_option:
            temp_imean = torch.stack([x[-1] for x in total_res], dim=0)
            out += (temp_imean,)
        out += (bold_out, )
        return out

    def run(self, step=800, observation_time=100, hp_path='./', hp_index=None, whole_brain_info='./'):
        """
        Run this block and save the returned block information.

        Parameters
        ----------
        step: int, default=800
             iter number in one observation time point.

        observation_time: int, default=100
            total time points, equal to the time points of bold signal.

        hp_path: str, default='./'
            path of saved hyperparameters after assimilation

        """
        info = {'name': self.name, "num_neurons": self.num_neurons, 'num_voxels': self.num_voxels,
                'num_populations': self.num_populations}
        table_print(info)

        hp_total = np.load(hp_path)
        assert hp_total.shape[1] == self.num_voxels
        hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
        hp_index = torch.tensor(hp_index, dtype=torch.int64, device="cuda:0")
        start_time = time.time()
        for k in hp_index:
            self.gamma_initialize(k)
        population_info = torch.stack(torch.meshgrid(self.population_id, hp_index), dim=-1).reshape((-1, 2))
        self.block_model.mul_property_by_subblk(population_info, hp_total[0].reshape(-1))
        freqs, _ = self.evolve(3200, vmean_option=False, sample_option=False, imean_option=False)
        Init_end = time.time()
        if self.print_info:
            print(
                f"max fre: {torch.max(torch.mean(freqs / self.num_neurons_per_population.float() * 1000, dim=0)):.1f}")
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")

        hp_total = hp_total[1:]
        total_T = hp_total.shape[0]
        assert observation_time <= total_T

        # file = h5py.File(whole_brain_info, 'r')
        # aal_region = np.array(file["dti_AAL_stn"], dtype=np.int32).squeeze() - 1
        # if self.sample_option:
        #     self.sample(aal_region, specified_info=None)

        pretty_print("Begin Simulation")
        print(f"Total time is {total_T}, we only simulate {observation_time}\n")

        bold_out = np.zeros([observation_time, self.num_voxels], dtype=np.float32)
        for ii in range((observation_time - 1) // 50 + 1):
            nj = min(observation_time - ii * 50, 50)
            FFreqs = np.zeros([nj, step, self.num_populations], dtype=np.uint32)
            if self.vmean_option:
                Vmean = np.zeros([nj, step, self.num_populations], dtype=np.float32)
            if self.sample_option:
                Spike = np.zeros([nj, step, self.num_sample], dtype=np.uint8)
                Vi = np.zeros([nj, step, self.num_sample], dtype=np.float32)
            if self.imean_option:
                Imean = np.zeros([nj, step, self.num_populations], dtype=np.float32)

            for j in range(nj):
                i = ii * 50 + j
                t_sim_start = time.time()
                self.block_model.mul_property_by_subblk(population_info, hp_total[i].reshape(-1))
                out = self.evolve(step, vmean_option=self.vmean_option, sample_option=self.sample_option,
                                  imean_option=self.imean_option)
                FFreqs[j] = torch_2_numpy(out[0])
                if self.vmean_option:
                    Vmean[j] = torch_2_numpy(out[1])
                if self.sample_option:
                    if self.vmean_option:
                        Spike[j] = torch_2_numpy(out[2])
                        Vi[j] = torch_2_numpy(out[3])
                    else:
                        Spike[j] = torch_2_numpy(out[1])
                        Vi[j] = torch_2_numpy(out[2])
                if self.imean_option:
                    Imean[j] = torch_2_numpy(out[-2])
                bold_out[i, :] = torch_2_numpy(out[-1])
                t_sim_end = time.time()
                print(
                    f"{i}th observation time, max fre: {torch.max(torch.mean(out[0] / self.num_neurons_per_population.float() * 1000, dim=0)):.1f}, cost time {t_sim_end - t_sim_start}:.1f")
            np.save(os.path.join(self.write_path, "freqs_after_assim_{}.npy".format(ii)), FFreqs)
            if self.vmean_option:
                np.save(os.path.join(self.write_path, "vmean_after_assim_{}.npy".format(ii)), Vmean)
            if self.sample_option:
                np.save(os.path.join(self.write_path, "spike_after_assim_{}.npy".format(ii)), Spike)
                np.save(os.path.join(self.write_path, "vi_after_assim_{}.npy".format(ii)), Vi)
            if self.imean_option:
                np.save(os.path.join(self.write_path, "imean_after_assim_{}.npy".format(ii)), Imean)
        np.save(os.path.join(self.write_path, "bold_after_assim.npy"), bold_out)

        pretty_print(f"Total simulation have done, Cost time {time.time() - start_time:.2f}")

        self.block_model.shutdown()
