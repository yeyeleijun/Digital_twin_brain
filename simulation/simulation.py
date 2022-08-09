# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 13:07 
# @Author : lepold
# @File : simulation.py


import os
import time
import numpy as np
import torch
from default_param import bold_params, v_th
# from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block
from models.bold_model_pytorch import BOLD
from utils.pretty_print import pretty_print, table_print
from utils.helpers import load_if_exist, torch_2_numpy
from utils.sample import sample


class simulation(object):
    """
    By giving the iP and block path, a simulation object is initialized.
    Usually in the case: a one ensemble, i.e., one simulation ensemble.

    default params can be found in *params.py*,
    This is for micro-column version or voxel version.

    Specifically, for micro-column version, the overlap=10 and voxel version 1.
    It means that in DA procedure, the step units in indexing the populations between ensembles.

    Parameters
    ----------

    block_path: str
        the dir which saves the block.npz

    ip: str
        the server listening address

    route_path : str, default is None
        the routing results.

    kwargs: other positional params which are specified.

    """

    @staticmethod
    def merge_data(a, weights=None, bins=10, range=None,):
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

    def __init__(self, block_path: str, ip: str, route_path=None, column=True, **kwargs):
        if column:
            self.block_model = block(ip, block_path, 1., route_path=route_path, overlap=10)
            self.voxel_populations = 10
        else:
            self.block_model = block(ip, block_path, 1., route_path=route_path, overlap=1)
            self.voxel_populations = 2
        self.bold = BOLD(**bold_params)
        self.populations = self.block_model.subblk_id
        self.populations_cpu = self.populations.cpu().numpy()

        if column:
            self.num_voxel = int(self.populations_cpu.max() + 9 // 10)
        else:
            self.num_voxel = int(self.populations_cpu.max() + 1 // 2)

        self.neurons_per_population = self.block_model.neurons_per_subblk
        neurons_per_population = self.neurons_per_population.cpu().numpy()
        self.neurons_per_voxel_cpu = self.merge_data(self.populations_cpu, neurons_per_population, self.num_voxel,
                                            (0, self.num_voxel * self.voxel_populations))

        self.total_populations = int(self.block_model.total_subblks)
        self.total_neurons = int(self.block_model.total_neurons)

        self.write_path = kwargs.get('write_path', "./")
        self.name = kwargs.get('name', "simulation")
        self.print_info = kwargs.get("print_info", False)
        self.vmean_option = kwargs.get("vmean_option", False)
        self.sample_option = kwargs.get("sample_option", False)

    def clear_mode(self):
        """
        Make this simulation to a clear state with no printing and only default freqs and bold as the output info.
        Returns
        -------

        """
        self.sample_option = False
        self.vmean_option = False
        self.print_info = False

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

    def evolve(self, step, vmean_option=False, sample_option=True) -> tuple:
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
        self.vmean_option = vmean_option
        self.sample_option = sample_option
        total_res = []

        for return_info in self.block_model.run(step, freqs=True, vmean=self.vmean_option, sample_for_show=self.sample_option):
            Freqs, *others = return_info
            total_res.append(return_info)

            freqs = Freqs.cpu().numpy()
            act = self.merge_data(self.populations_cpu, freqs, self.num_voxel, (0, self.num_voxel * self.voxel_populations))
            act = (act / self.neurons_per_voxel_cpu).reshape(-1)
            act = torch.from_numpy(act).cuda()
            bold_out = self.bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
        temp_freq = torch.stack([x[0] for x in total_res], dim=0)
        out = (temp_freq, )
        if vmean_option:
            temp_vmean = torch.stack([x[1] for x in total_res], dim=0)
            out += (temp_vmean, )
        if sample_option:
            temp_vsamle = torch.stack([x[3] for x in total_res], dim=0)
            temp_spike = torch.stack([x[2] for x in total_res], dim=0)
            temp_spike &= (torch.abs(temp_vsamle - v_th) / 50 < 1e-5)
            out += (temp_spike, temp_vsamle, )
        out += (bold_out)
        return out

    def run(self, step=800, observation_time=100, hp_index=None, hp_total=None):
        """
        Run this block and save the returned block information.

        Parameters
        ----------
        step: int, default=800
             iter number in one observation time point.

        observation_time: int, default=100
            total time points, equal to the time points of bold signal.

        """

        if not hasattr(self, 'num_sample'):
            raise NotImplementedError('Please set the sampling neurons first in simulation case')

        start_time = time.time()
        if hp_total is not None:
            self.gamma_initialize(hp_index)
            population_info = torch.stack(torch.meshgrid(self.populations, hp_total), dim=-1).reshape((-1, 2))
            self.block_model.mul_property_by_subblk(population_info, hp_total[0, :])

        hp_total = hp_total[1:, ]
        total_T = hp_total.shape[0]
        assert observation_time <= total_T
        freqs, _ = self.evolve(3200, vmean_option=False, sample_option=False)
        Init_end = time.time()
        if self.print_info:
            print(f"max fre: {torch.max(torch.mean(freqs / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}")
        pretty_print(f"Init have Done, cost time {Init_end - start_time:.2f}")

        pretty_print("Begin Simulation")
        print(f"Total time is {total_T}, we only simulate {observation_time}\n")

        bolds_out = np.zeros([observation_time, self.num_voxel], dtype=np.float32)
        for ii in range((observation_time - 1) // 50 + 1):
            nj = min(observation_time - ii * 50, 50)
            FFreqs = np.zeros([nj, step, self.total_populations], dtype=np.uint32)
            # Noise_Spike = np.zeros([nj, step, self.num_sample], dtype=np.uint8)
            if self.vmean_option:
                Vmean = np.zeros([nj, step, self.total_populations], dtype=np.float32)
            if self.sample_option:
                Spike = np.zeros([nj, step, self.num_sample], dtype=np.uint8)
                Vi = np.zeros([nj, step, self.num_sample], dtype=np.float32)

            for j in range(nj):
                i = ii * 50 + j
                t_sim_start = time.time()
                if hp_total is not None:
                    self.block_model.mul_property_by_subblk(population_info, hp_total[i, :])
                out = self.evolve(step)
                FFreqs[j] = torch_2_numpy(out[0])
                if self.vmean_option:
                    Vmean[j] = torch_2_numpy(out[1])
                if self.sample_option:
                    Spike[j] = torch_2_numpy(out[-3])
                    Vi[j] = Spike[j] = torch_2_numpy(out[-2])
                bolds_out[i, :] = torch_2_numpy(out[-1])
                t_sim_end= time.time()
                print(
                    f"{i}th observation_time, max fre: {torch.max(torch.mean(out[0] / self.block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t_sim_end - t_sim_start:.1f}")
            if self.sample_option:
                np.save(os.path.join(self.write_path, "spike_after_assim_{}.npy".format(ii)), Spike)
                np.save(os.path.join(self.write_path, "vi_after_assim_{}.npy".format(ii)), Vi)
            if self.vmean_option:
                np.save(os.path.join(self.write_path, "vmean_after_assim_{}.npy".format(ii)), Vmean)
            np.save(os.path.join(self.write_path, "freqs_after_assim_{}.npy".format(ii)), FFreqs)
        np.save(os.path.join(self.write_path, "bold_after_assim.npy"), bolds_out)

        pretty_print(f"Totally have Done, Cost time {time.time() - start_time:.2f} ")

    def __call__(self, step=800, observation_time=100, hp_index=None, hp_path=None):
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
            self.run(step=800, observation_time=100, hp_index=hp_index, hp_total=hp_total)
        else:
            self.run(step=800, observation_time=100, hp_index=None, hp_total=None)

        self.block_model.shutdown()
