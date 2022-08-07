# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 13:07 
# @Author : lepold
# @File : simulation.py


import os
import time
import numpy as np
import torch
from default_param import bold_params, v_th
from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block
from brain_block.bold_model_pytorch import BOLD
from utils import *


class simulation(object):
    """
    By giving the iP and block path, a simulation object is initialized.
    Usually in the case: a one ensemble, i.e., one simulation ensemble.

    default params can be found in *params.py*,
    This is for micro-column version.

    Parameters
    ----------

    block_path: str
        the dir which saves the block.npz

    ip: str
        the server listening address

    route_path : str, default is None
        the routing results.

    kwargs: others which are specified.

    """

    def __init__(self, block_path: str, ip: str, route_path=None, **kwargs):
        self.block_model = block(ip, block_path, 1., route_path=route_path, force_rebase=False)
        self.bold = BOLD(**bold_params)
        self.populations = self.block_model.subblk_id
        self.neurons_per_population = self.block_model.neurons_per_subblk

        self.total_populations = int(self.block_model.total_subblks)
        self.total_neurons = int(self.block_model.total_neurons)
        self.num_voxel = int(self.populations.max() + 9 // 10)

        self.write_path = kwargs.get('write_path', "./")
        self.name = kwargs.get('name', "simulation")

    def update(self, param_index, param):
        """
        Use a distribution of gamma(alpha, beta) to update neuronal params,
        where alpha = hpyer_param * 1e8, beta = 1e8.

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
            torch.meshgrid(self.populations, torch.tensor([param_index], dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        alpha = torch.ones(self.total_populations, device="cuda:0") * param * 1e8
        beta = torch.ones(self.total_populations, device="cuda:0") * 1e8
        self.block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    def sample(self, aal_region, specified_info=None):
        """

        set sample idx of neurons in this simulation object.
        Due to the routing algorithm, we utilize the disorder population idx and neurons num.

        Parameters
        ----------
        aal_region: ndarray
            indicate the brain regions label of each voxel appeared in this simulation.

        """

        num_voxel = len(aal_region)
        neurons_per_population = self.block_model._neurons_per_subblk.cpu().numpy()
        neurons_per_population_base = np.add.accumulate(neurons_per_population)
        neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
        populations_cpu = self.block_model._subblk_id[self.block_model._subblk_idx].cpu().numpy()

        sample_idx = load_if_exist(sample, os.path.join(self.write_path, "sample_idx"),
                                   aal_region=aal_region,
                                   neurons_per_population_base=neurons_per_population_base,
                                   populations_id=populations_cpu, specified_info=specified_info)
        sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
        num_sample = sample_idx.shape[0]
        assert sample_idx[:, 0].max() < self.total_neurons
        self.block_model.set_samples(sample_idx)
        load_if_exist(lambda: self.block_model.neurons_per_subblk.cpu().numpy(),
                      os.path.join(self.write_path, "blk_size"))
        self.num_sample = num_sample
        return num_sample, num_voxel

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
            torch.meshgrid(self.populations, torch.tensor(param_index, dtype=torch.int64, device="cuda:0")),
            dim=-1).reshape((-1, 2))
        alpha = torch.ones(self.total_populations, device="cuda:0") * alpha
        beta = torch.ones(self.total_populations, device="cuda:0") * beta
        self.block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    def __call__(self, number=800, step=100, hp_index=None, hp_path=None):
        """
        Give this objects the ability to behave like a function, to simulate the LIF nn.

        Parameters
        ----------

        hp_index: list, default is None.

        hp_path: str, default is None.


        Notes
        ---------

        if hp info is None, it equally says that we don't need to update neuronal attributes
        during the simulation period.

        """

        info = {'name': self.name, "num_neurons": self.total_neurons, 'num_voxel':self.num_voxel,
                'num_populations': self.total_populations}
        table_print(info)
        if hp_path is not None:
            hp_total = np.load(hp_path)
            assert hp_total.shape[1] == self.total_populations
            hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
            hp_index = torch.tensor(hp_index, dtype=torch.int64, device="cuda:0")
            self.run(number=800, step=100, hp_index=hp_index, hp_total=hp_total)
        else:
            self.run(number=800, step=100, hp_index=None, hp_total=None)

        self.block_model.shutdown()

    def run(self, number=800, step=100, hp_index=None, hp_total=None):
        """
        Run this block and save the returned block information.

        Parameters
        ----------
        number: int, default=800
            minimum iter

        step: int, default=100
            total steps, equal to the time points of bold signal.

        """
        populations_cpu = self.populations.cpu().numpy()
        neurons_per_population = self.neurons_per_population.cpu().numpy()
        neurons_per_voxel, _ = np.histogram(populations_cpu, weights=neurons_per_population, bins=self.num_voxel,
                                            range=(0, self.num_voxel * 10))
        if not hasattr(self, 'num_sample'):
            raise NotImplementedError('Please set the sampling neurons first')

        if hp_total is not None:
            self.gamma_initialize(hp_index)
            population_info = torch.stack(torch.meshgrid(self.populations, hp_total), dim=-1).reshape((-1, 2))
            self.block_model.mul_property_by_subblk(population_info, hp_total[0, :])

        hp_total = hp_total[1:, ]
        total_T = hp_total.shape[0]
        start_time = time.time()
        for j in range(4):
            t13 = time.time()
            temp_fre = []
            for return_info in self.block_model.run(800, freqs=True, vmean=True, sample_for_show=True):
                Freqs, vmean, spike, vi = return_info
                temp_fre.append(Freqs)
            temp_fre = torch.stack(temp_fre, dim=0)
            t14 = time.time()
            print(
                f"{j}th step number {number}, max fre: {torch.max(torch.mean(temp_fre / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t14 - t13:.1f}")

        Init_end = time.time()
        assert step <= total_T
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")
        bolds_out = np.zeros([step, self.num_voxel], dtype=np.float32)

        pretty_print("Begin Simulation")
        print(f"Total time is {total_T}, we only simulate {step}\n")
        simulate_start = time.time()
        for ii in range((step - 1) // 50 + 1):
            nj = min(step - ii * 50, 50)
            FFreqs = np.zeros([nj, number, self.total_populations], dtype=np.uint32)
            Vmean = np.zeros([nj, number, self.total_populations], dtype=np.float32)
            Spike = np.zeros([nj, number, self.num_sample], dtype=np.uint8)
            # Noise_Spike = np.zeros([nj, number, self.num_sample], dtype=np.uint8)
            Vi = np.zeros([nj, number, self.num_sample], dtype=np.float32)

            for j in range(nj):
                i = ii * 50 + j
                t13 = time.time()
                temp_fre = []
                temp_spike = []
                # temp_noise_spike = []
                temp_vi = []
                temp_vmean = []
                if hp_total is not None:
                    self.block_model.mul_property_by_subblk(population_info, hp_total[i, :])
                for return_info in self.block_model.run(number, freqs=True, vmean=True, sample_for_show=True):
                    Freqs, vmean, spike, vi = return_info
                    freqs = Freqs.cpu().numpy()
                    act, _ = np.histogram(populations_cpu, weights=freqs, bins=self.num_voxel, range=(0, self.num_voxel * 10))
                    act = (act / neurons_per_voxel).reshape(-1)
                    act = torch.from_numpy(act).cuda()
                    # noise_spike = spike & (torch.abs(vi - v_th) / 50 >= 1e-5)
                    spike &= (torch.abs(vi - v_th) / 50 < 1e-5)
                    temp_fre.append(Freqs)
                    temp_vmean.append(vmean)
                    temp_spike.append(spike)
                    # temp_noise_spike.append(noise_spike)
                    temp_vi.append(vi)
                    bold_out = self.bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
                bolds_out[i, :] = torch_2_numpy(bold_out)
                temp_fre = torch.stack(temp_fre, dim=0)
                temp_vmean = torch.stack(temp_vmean, dim=0)
                temp_spike = torch.stack(temp_spike, dim=0)
                # temp_noise_spike = torch.stack(temp_noise_spike, dim=0)
                temp_vi = torch.stack(temp_vi, dim=0)
                t14 = time.time()
                FFreqs[j] = torch_2_numpy(temp_fre)
                Vmean[j] = torch_2_numpy(temp_vmean)
                Spike[j] = torch_2_numpy(temp_spike)
                # Noise_Spike[j] = torch_2_numpy(temp_noise_spike)
                Vi[j] = torch_2_numpy(temp_vi)
                print(
                    f"{i}th step, max fre: {torch.max(torch.mean(temp_fre / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t14 - t13:.1f}")
            np.save(os.path.join(self.write_path, "spike_after_assim_{}.npy".format(ii)), Spike)
            # np.save(os.path.join(self.write_path, "noise_spike_after_assim_{}.npy".format(ii)), Noise_Spike)
            np.save(os.path.join(self.write_path, "vi_after_assim_{}.npy".format(ii)), Vi)
            np.save(os.path.join(self.write_path, "freqs_after_assim_{}.npy".format(ii)), FFreqs)
            np.save(os.path.join(self.write_path, "vmean_after_assim_{}.npy".format(ii)), Vmean)
        np.save(os.path.join(self.write_path, "bold_after_assim.npy"), bolds_out)

        pretty_print(f"Simulation have Done, Cost time {time.time() - simulate_start:.2f} ")
        pretty_print(f"Totally have Done, Cost time {time.time() - start_time:.2f} ")
