#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/8/18 13:32
# @Author  : Leijun Ye

import os
import copy
import time
import numpy as np
import torch
from default_params import bold_params, v_th
from utils.pretty_print import pretty_print, table_print
from utils.helpers import load_if_exist, torch_2_numpy
from simulation.simulation import simulation
from utils.sample import sample_voxel
import h5py


class SimulationVoxel(simulation):

    def __init__(self, ip: str, block_path: str, dt: float, route_path=None, column=True, **kwargs):
        super(SimulationVoxel, self).__init__(ip, block_path, dt, route_path, column, **kwargs)

    def sample(self, aal_region, population_base, num_sample_voxel_per_region, num_neurons_per_voxel, specified_info=None):
        """

        set sample idx of neurons in this simulation object.
        Due to the routing algorithm, we utilize the disorder population idx and neurons num.

        Parameters
        ----------
        aal_region: ndarray
            indicate the brain regions label of each voxel appeared in this simulation.

        population_base: ndarray
            indicate range of neurons in each population.

        """

        sample_idx = load_if_exist(sample_voxel, os.path.join(self.write_path, "sample_idx"),
                                   aal_region=aal_region,
                                   neurons_per_population_base=population_base,
                                   num_sample_voxel_per_region=num_sample_voxel_per_region, num_neurons_per_voxel=num_neurons_per_voxel)
        sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
        num_sample = sample_idx.shape[0]
        assert sample_idx.max() < self.num_neurons
        self.block_model.set_samples(sample_idx)
        load_if_exist(lambda: self.block_model.neurons_per_subblk.cpu().numpy(),
                      os.path.join(self.write_path, "blk_size"))
        self.num_sample = num_sample

        return num_sample

    def info_specified_region(self, aal_region, specified_region=(90, 91)):
        """
        Return the information of specified regions

        Parameters
        ----------
        aal_region: ndarray
            the information of all voxels belong to which region
        specified_region: tuple
            the region to be inquired
        Returns
        ----------
        population_id_belong_to_specified_region: array

        neuron_id_belong_to_region: array
        """
        specified_region = np.array(specified_region)
        voxel_id_belong_to_specified_region = np.isin(aal_region, specified_region).nonzero()[0]
        population_id_belong_to_specified_region = np.concatenate(
            [np.arange(temp * self.populations_per_voxel, (temp + 1) * self
                       .populations_per_voxel) for temp in voxel_id_belong_to_specified_region])
        neuron_id_belong_to_region = []
        num_neurons_per_population_base_cpu = np.add.accumulate(self.num_neurons_per_population_cpu)
        num_neurons_per_population_base_cpu = np.insert(num_neurons_per_population_base_cpu, 0, 0)
        for idx in population_id_belong_to_specified_region:
            neuron_id_belong_to_region.append(
                np.arange(num_neurons_per_population_base_cpu[idx], num_neurons_per_population_base_cpu[idx + 1]))
        neuron_id_belong_to_region = np.concatenate(neuron_id_belong_to_region)
        print(f"{len(voxel_id_belong_to_specified_region)} voxels, {len(population_id_belong_to_specified_region)} populations and {len(neuron_id_belong_to_region)} neurons in region {specified_region}")
        return population_id_belong_to_specified_region.astype(np.int64), neuron_id_belong_to_region.astype(np.int64)

    def run(self, step=800, observation_time=100, hp_total=None, hp_index=None, whole_brain_info="./"):
        """
        Run this block and save the returned block information.

        Parameters
        ----------
        step: int, default=800
             iter number in one observation time point.

        observation_time: int, default=100
            total time points, equal to the time points of bold signal.

        hp_total: default=None
            hyperparameters after assimilation

        """

        start_time = time.time()
        if hp_total is not None:
            state = "after"
            for k in hp_index:
                self.gamma_initialize(self.population_id, k)
            population_info = torch.stack(torch.meshgrid(self.population_id, hp_index), dim=-1).reshape((-1, 2))
            self.block_model.mul_property_by_subblk(population_info, hp_total[0].reshape(-1))

            hp_total = hp_total[1:]
            total_T = hp_total.shape[0]
            assert observation_time <= total_T
        else:
            state = "before"
            total_T = None

        freqs, _ = self.evolve(3200, vmean_option=False, sample_option=False, imean_option=False)
        Init_end = time.time()
        if self.print_info:
            print(
                f"mean fre: {torch.mean(torch.mean(freqs / self.num_neurons_per_population.float() * 1000, dim=0)):.1f}")
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")

        pretty_print("Begin Simulation")
        print(f"Total time is {total_T}, we only simulate {observation_time}\n")


        # set sample
        if self.sample_option:
            num_neurons_per_population_cpu_base = np.add.accumulate(self.num_neurons_per_population_cpu)
            num_neurons_per_population_cpu_base = np.insert(num_neurons_per_population_cpu_base, 0, 0)
            file = h5py.File(whole_brain_info, 'r')
            aal_region = np.array(file["dti_AAL_stn"], dtype=np.int32).squeeze() - 1
            self.sample(aal_region, num_neurons_per_population_cpu_base, num_sample_voxel_per_region=1, num_neurons_per_voxel=200)

        bolds_out = np.zeros([observation_time, self.num_voxels], dtype=np.float32)
        for ii in range((observation_time - 1) // 50 + 1):
            nj = min(observation_time - ii * 50, 50)
            FFreqs = np.zeros([nj, step, self.num_populations], dtype=np.uint32)
            # Noise_Spike = np.zeros([nj, step, self.num_sample], dtype=np.uint8)
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
                if hp_total is not None:
                    self.block_model.mul_property_by_subblk(population_info, hp_total[i].reshape(-1))
                out = self.evolve(step, vmean_option=self.vmean_option, sample_option=self.sample_option)
                FFreqs[j] = torch_2_numpy(out[0])
                out_base = 1
                if self.vmean_option:
                    Vmean[j] = torch_2_numpy(out[out_base])
                    out_base += 1
                if self.sample_option:
                    Spike[j] = torch_2_numpy(out[out_base])
                    Vi[j] = Spike[j] = torch_2_numpy(out[out_base + 1])
                    out_base += 2
                if self.imean_option:
                    Imean[i] = torch_2_numpy(out[out_base])

                bolds_out[i, :] = torch_2_numpy(out[-1])
                t_sim_end = time.time()
                print(
                    f"{i}th observation_time, mean fre: {torch.mean(torch.mean(out[0] / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t_sim_end - t_sim_start:.1f}")
            if self.sample_option:
                np.save(os.path.join(self.write_path, f"spike_{state}_assim_{ii}.npy"), Spike)
                np.save(os.path.join(self.write_path, f"vi_{state}_assim_{ii}.npy"), Vi)
            if self.vmean_option:
                np.save(os.path.join(self.write_path, f"vmean_{state}_assim_{ii}.npy"), Vmean)
            if self.imean_option:
                np.save(os.path.join(self.write_path, f"imean_{state}_assim_{ii}.npy"), Imean)
            np.save(os.path.join(self.write_path, f"freqs_{state}_assim_{ii}.npy"), FFreqs)
        np.save(os.path.join(self.write_path, f"bold_{state}_assim.npy"), bolds_out)

        pretty_print(f"Total simulation have done, Cost time {time.time() - start_time:.2f}")

    def __call__(self, step=800, observation_time=100,  hp_path=None, hp_index=None, whole_brain_info='./'):
        """
        Give this objects the ability to behave like a function, to simulate the LIF nn.

        Parameters
        ----------
        hp_path: str, default is None.

        hp_index: list, default is None.

        Notes
        ---------

        if hp info is None, it equally says that we don't need to update neuronal attributes
        during the simulation period.

        """

        info = {'name': self.name, "num_neurons": self.num_neurons, 'num_voxels': self.num_voxels,
                'num_populations': self.num_populations}
        table_print(info)
        if hp_path is not None:
            hp_total = np.load(hp_path)
            assert hp_total.shape[1] == self.num_populations
            hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
            hp_index = torch.tensor(hp_index, dtype=torch.int64, device="cuda:0")
            self.run(step=step, observation_time=observation_time, hp_index=hp_index, hp_total=hp_total, whole_brain_info=whole_brain_info)
        else:
            self.run(step=step, observation_time=observation_time, hp_index=None, hp_total=None, whole_brain_info=whole_brain_info)

        self.block_model.shutdown()


class SimulationVoxelCritical(SimulationVoxel):
    def __init__(self, ip, block_path, dt, route_path, **kwargs):
        # global bold_params
        # bold_params['delta_t'] = 1e-4
        SimulationVoxel.__init__(self, ip, block_path, dt, route_path, **kwargs)

    def evolve(self, step, vmean_option=False, sample_option=False, imean_option=False, bold_detail=False) -> tuple:

        total_res = []
        for return_info in self.block_model.run(step, freqs=True, vmean=vmean_option, sample_for_show=sample_option,
                                                imean=imean_option):
            total_res.append(return_info)
        temp_freq = torch.stack([x[0] for x in total_res], dim=0)
        out = (temp_freq,)
        temp_freq = temp_freq.cpu().numpy()
        temp_freq = temp_freq.reshape([800, step//800, -1]).sum(axis=1)

        for k in range(800):
            freqs = temp_freq[k]
            act = self.merge_data(self.population_id_cpu, freqs, self.num_voxels, (0, self.num_populations))
            act = (act / self.num_neurons_per_voxel_cpu).reshape(-1)
            act = torch.from_numpy(act).cuda()
            bold_out = self.bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
        if bold_detail:
            return act, self.bold.s, self.bold.q, self.bold.v, self.bold.f_in
        out_base = 1
        if vmean_option:
            temp_vmean = torch.stack([x[1] for x in total_res], dim=0)
            out += (temp_vmean,)
            out_base += 1
        if sample_option:
            temp_vsample = torch.stack([x[out_base+1] for x in total_res], dim=0)
            temp_spike = torch.stack([x[out_base] for x in total_res], dim=0)
            temp_spike &= (torch.abs(temp_vsample - v_th) / 50 < 1e-5)
            temp_spike = temp_spike.reshape([800, step//800, -1]).sum(axis=1)
            assert temp_spike.max() <= 1, f"{torch.where(temp_spike > 0)[0].shape}"
            out += (temp_spike, temp_vsample,)
            out_base += 2
        if imean_option:
            temp_imean = torch.stack([x[-1] for x in total_res], dim=0)
            out += (temp_imean,)
        out += (bold_out,)
        return out

    def __call__(self, step=800, observation_time=100,  hp_path=None, hp_index=None, whole_brain_info='./'):
        info = {'name': self.name, "num_neurons": self.num_neurons, 'num_voxels': self.num_voxels,
                'num_populations': self.num_populations}
        table_print(info)
        if hp_path is not None:
            hp_total = np.load(hp_path)
            assert hp_total.shape[1] == self.num_populations
            hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
            hp_index = torch.tensor(hp_index, dtype=torch.int64, device="cuda:0")
            self.run(step=step, observation_time=observation_time, hp_index=hp_index, hp_total=hp_total, whole_brain_info=whole_brain_info)
        else:
            self.run(step=step, observation_time=observation_time, hp_index=None, hp_total=None, whole_brain_info=whole_brain_info)

        self.block_model.shutdown()


class StimulationVoxel(SimulationVoxel):
    def __init__(self, ip, block_path, dt, route_path, **kwargs):
        SimulationVoxel.__init__(self, ip, block_path, dt, route_path, **kwargs)

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
            self.gamma_initialize(self.population_id, k)
        population_info = torch.stack(torch.meshgrid(self.population_id, hp_index), dim=-1).reshape((-1, 2))
        self.block_model.mul_property_by_subblk(population_info, hp_total[0].reshape(-1))
        freqs, _ = self.evolve(3200, vmean_option=False, sample_option=False, imean_option=False)
        Init_end = time.time()
        if self.print_info:
            print(
                f"mean fre: {torch.mean(torch.mean(freqs / self.num_neurons_per_population.float() * 1000, dim=0)):.1f}")
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")

        hp_total = hp_total[1:]
        total_T = hp_total.shape[0]
        assert observation_time <= total_T

        file = h5py.File(whole_brain_info, 'r')
        aal_region = np.array(file["dti_AAL_stn"], dtype=np.int32).squeeze() - 1
        # if self.sample_option:
        #     self.sample(aal_region, specified_info=None)

        # stimulation setting to mimic beta oscillation
        stim_density = 0.1  # Amplitude
        freqs_stim = 15  # Hz
        sample_rate = 1000  # Hz
        T = observation_time * step
        current_beta = stim_density * np.sin(np.linspace(0, 2 * np.pi * freqs_stim * T / sample_rate, T), dtype=np.float32)
        current_beta = torch.from_numpy(current_beta).cuda()

        population_id_stim, _ = self.info_specified_region(aal_region, specified_region=(74, 75, 90, 91))
        population_id_stim = torch.from_numpy(population_id_stim).cuda()
        num_populations_stim = len(population_id_stim)
        self.gamma_initialize(population_id_stim, 2)
        pro_idx_extern_input_current = torch.tensor([2], dtype=torch.int64, device="cuda:0")   # property idx 2 is I_extern_Input
        population_stim_info = torch.stack(torch.meshgrid(population_id_stim, pro_idx_extern_input_current), dim=-1).reshape((-1, 2))
        self.block_model.mul_property_by_subblk(population_stim_info, torch.ones(num_populations_stim, device="cuda:0") * 0.0001)

        # deep brain stimulation setting
        current_dbs = np.ones([T], dtype=np.float32) * 0.0001
        freqs_dbs = 125
        for i in range(T//2, T, sample_rate//freqs_dbs):
            current_dbs[i] = -6
        current_dbs = torch.from_numpy(current_dbs).cuda()

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
                # for i_step in range(step//10):
                #     self.block_model.mul_property_by_subblk(population_stim_info,
                #                                                 -1 * torch.ones(num_populations_stim, device="cuda:0") * 2)
                #     out = self.evolve(1, vmean_option=self.vmean_option, sample_option=self.sample_option,
                #                           imean_option=self.imean_option)
                #     FFreqs[j, i_step*10:i_step*10+1, :] = torch_2_numpy(out[0])
                #     self.block_model.mul_property_by_subblk(population_stim_info,
                #                                                 torch.ones(num_populations_stim, device="cuda:0") * 0.0001)
                #     out = self.evolve(9, vmean_option=self.vmean_option, sample_option=self.sample_option,
                #                           imean_option=self.imean_option)
                #     FFreqs[j, i_step*10+1:i_step*10+10, :] = torch_2_numpy(out[0])
                for i_step in range(step):
                    curr_stim = torch.ones(num_populations_stim, device="cuda:0") * current_beta[i * step + i_step]
                    curr_stim[-4:] += current_dbs[i * step + i_step]
                    self.block_model.mul_property_by_subblk(population_stim_info, curr_stim)
                    out = self.evolve(1, vmean_option=self.vmean_option, sample_option=self.sample_option,
                                      imean_option=self.imean_option)
                    FFreqs[j, i_step, :] = torch_2_numpy(out[0])
                    if self.vmean_option:
                        Vmean[j, i_step, :] = torch_2_numpy(out[1])
                    if self.sample_option:
                        if self.vmean_option:
                            Spike[j, i_step, :] = torch_2_numpy(out[2])
                            Vi[j, i_step, :] = torch_2_numpy(out[3])
                        else:
                            Spike[j, i_step, :] = torch_2_numpy(out[1])
                            Vi[j, i_step, :] = torch_2_numpy(out[2])
                    if self.imean_option:
                        Imean[j, i_step, :] = torch_2_numpy(out[-2])
                bold_out[i, :] = torch_2_numpy(out[-1])
                t_sim_end = time.time()
                print(
                    f"{i}th observation time, mean fre: {torch.mean(torch.mean(out[0] / self.num_neurons_per_population.float() * 1000, dim=0)):.1f}, cost time {t_sim_end - t_sim_start:.1f}")
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

