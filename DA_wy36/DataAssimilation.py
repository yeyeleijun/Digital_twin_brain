# -*- coding: utf-8 -*-
# @Time : 2022/8/10 11:20
# @Author : wy36
# @File : DataAssimilation.py


import os
import time
import torch
import numpy as np
from simulation.simulation import simulation
import matplotlib.pyplot as mp
mp.switch_backend('Agg')


def get_bold_signal(bold_path, b_min=None, b_max=None, lag=0):
    bold_y = np.load(bold_path)[lag:]
    if b_max is not None:
        bold_y = b_min + (b_max - b_min) * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
    return bold_y


def torch2numpy(u, is_cuda=True):
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def numpy2torch(u, is_cuda=True):
    assert isinstance(u, np.ndarray)
    if is_cuda:
        return torch.from_numpy(u).cuda()
    else:
        return torch.from_numpy(u)


def diffusion_EnKF(w_hat, bold_sigma, bold_t, solo_rate, debug=False):
    ensembles, brain_num, state_num = w_hat.shape
    w = w_hat.clone()  # ensemble, brain_n, hp_num+hemodynamic_state
    w_mean = torch.mean(w_hat, dim=0, keepdim=True)
    w_diff = w_hat - w_mean
    w_cx = w_diff[:, :, -1] * w_diff[:, :, -1]
    w_cxx = torch.sum(w_cx, dim=0) / (ensembles - 1) + bold_sigma
    temp = w_diff[:, :, -1] / (w_cxx.reshape([1, brain_num])) / (ensembles - 1)  # (ensemble, brain)
    # kalman = torch.mm(temp.T, w_diff.reshape([ensembles, brain_n*hp_num]))  # (brain_n, w_shape[1])
    model_noise = bold_sigma ** 0.5 * torch.normal(0, 1, size=(ensembles, brain_num)).type_as(temp)
    w += solo_rate * (bold_t + model_noise - w_hat[:, :, -1])[:, :, None] \
         * torch.sum(temp[:, :, None] * w_diff.reshape([ensembles, brain_num, state_num]), dim=0, keepdim=True)
    w += (1 - solo_rate) * torch.mm(torch.mm(bold_t + model_noise - w_hat[:, :, -1], temp.T) / brain_num,
                               w_diff.reshape([ensembles, brain_num * state_num])).reshape(
        [ensembles, brain_num, state_num])
    if debug:
        print(w_cxx.max(), w_cxx.min())
        print(w[:, :, :state_num - 5].max(), w[:, :, :state_num - 5].min())
        w_debug = w_hat[:, :10, -1][None, :, :] + (bold_t + model_noise - w_hat[:, :, -1]).T[:, :, None] \
                  * torch.mm(temp.T, w_diff[:, :10, -1])[:, None, :]  # brain, ensemble, 10
        return w, w_debug
    else:
        return w


class DataAssimilation(simulation):
    def __init__(self, block_path: str, ip: str, route_path: str, column: bool, **kwargs):
        super().__init__(self, block_path, ip, route_path, column, **kwargs)
        self._ensemble_number = kwargs.get("ensemble", 100)
        self._hp_sigma = kwargs.get("hp_sigma", 1.)
        self._bold_sigma = kwargs.get("bold_sigma", 1e-8)
        self._solo_rate = kwargs.get("solo_rate", 0.5)
        assert (self.total_neurons % self._ensemble_number == 0)
        self._single_block_neurons_number = int(self.total_neurons // self._ensemble_number)
        self._single_voxel_number = int(self.total_populations // self._ensemble_number)
        self._hidden_state = None
        self._property_index = None
        self._hp_index_in_voxel = None
        self._hp_num = None
        self._hp_log = None
        self._hp = None
        self._hp_low = None
        self._hp_high = None

    @staticmethod
    def log_torch(val, lower, upper, scale=10):
        assert len(val.shape) == 2
        assert len(lower.shape) == 1
        if isinstance(val, torch.Tensor):
            if (val <= upper).all() and (val >= lower).all():
                out = scale * (torch.log(val - lower) - torch.log(upper - val))
                return out
            else:
                print('val <= upper).all() and (val >= lower).all()?')
        elif isinstance(val, np.ndarray):
            if (val <= upper).all() and (val >= lower).all():
                out = scale * (np.log(val - lower) - np.log(upper - val))
                return out
            else:
                print('val <= upper).all() and (val >= lower).all()?')
        else:
            print('torch.Tensor or np.ndarray?')

    @staticmethod
    def sigmoid_torch(val, lower, upper, scale=10):
        assert len(val.shape) == 2
        assert len(lower.shape) == 1
        if isinstance(val, torch.Tensor):
            out = lower + (upper - lower) * torch.sigmoid(val / scale)
            return out
        elif isinstance(val, np.ndarray):
            out = lower + (upper - lower) * 1 / (1 + np.exp(-val.astype(np.float32)) / scale)
            return out
        else:
            print('torch.Tensor or np.ndarray?')

    @staticmethod
    def plot_bold(path_out, bold_real, bold_da, bold_index):
        T = bold_da.shape[0]
        iteration = [i for i in range(T)]
        assert len(bold_da.shape) == 3
        for i in bold_index:
            print("show_bold" + str(i))
            fig = mp.figure(figsize=(8, 4), dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(iteration, bold_real[:T, i], 'r-')
            ax1.plot(iteration, np.mean(bold_da[:T, :, i], axis=1), 'b-')
            if bold_da.shape[1] != 1:
                mp.fill_between(iteration, np.mean(bold_da[:T, :, i], axis=1) -
                                np.std(bold_da[:T, :, i], axis=1), np.mean(bold_da[:T, :, i], axis=1)
                                + np.std(bold_da[:T, :, i], axis=1), color='b', alpha=0.2)
            mp.ylim((0.0, 0.08))
            ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
            mp.savefig(os.path.join(path_out, "figure/bold" + str(i) + ".pdf"), bbox_inches='tight', pad_inches=0)
            mp.close(fig)

    @staticmethod
    def plot_hp(path_out, hp_real, hp, bold_index, hp_num, label='hp'):
        T = hp.shape[0]
        iteration = [i for i in range(T)]
        assert len(hp.shape) == 4
        for i in bold_index:
            for j in range(hp_num):
                print("show_hp", i, 'and', j)
                fig = mp.figure(figsize=(8, 4), dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(iteration, np.mean(hp[:T, :, i, j], axis=1), 'b-')
                if hp_real is not None:
                    ax1.plot(iteration, np.tile(hp_real[j], T), 'r-')
                if hp.shape[1] != 1:
                    mp.fill_between(iteration, np.mean(hp[:T, :, i, j], axis=1) -
                                    np.sqrt(np.var(hp[:T, :, i, j], axis=1)), np.mean(hp[:T, :, i, j], axis=1)
                                    + np.sqrt(np.var(hp[:T, :, i, j], axis=1)), color='b', alpha=0.2)
                ax1.set(xlabel='observation time/800ms', ylabel='hyper parameter')
                mp.savefig(os.path.join(path_out, 'figure/' + label + str(i) + "_" + str(j) + ".pdf"), bbox_inches='tight',
                           pad_inches=0)
                mp.close(fig)

    @property
    def hp_sigma(self):
        return self._hp_sigma

    @property
    def ensemble_number(self):
        return self._ensemble_number

    @property
    def bold_sigma(self):
        return self._bold_sigma

    @property
    def solo_rate(self):
        return self._solo_rate

    @property
    def single_voxel_number(self):
        return self._single_voxel_number

    @property
    def single_block_neurons_number(self):
        return self._single_block_neurons_number

    def hp_random_initialize(self, gui_low, gui_high, gui_number, voxel_index=None):
        """
        Generate initial hyper parameters of each ensemble brain samples

        Complete assignment: self._hp_low, self._hp_high, self._hp, self._hp_log, self._hp_num

        Parameters
        ----------

        gui_low: torch.tensor, shape=(single_voxel_number, gui_number)
            low bound of hyper-parameters

        gui_high: torch.tensor, shape=(single_voxel_number, gui_number)
            low bound of hyper-parameters

        gui_number: int
            the number of hyper-parameters assimilated in brain property

        voxel_index: torch.tensor int64, shape=(-1)
            index of voxel assimilated or updated

        Returns
        -------

        self._hp: torch.tensor float32 shape=(ensemble_number*single_voxel_number*gui_number)

        """
        voxel_index = self.single_voxel_number if voxel_index is None else voxel_index
        assert gui_low.shape[0] == len(voxel_index)
        self._hp_low = gui_low.reshape(-1)
        self._hp_high = gui_high.reshape(-1)
        self._hp_num = gui_number
        self._hp_log = torch.linspace(-20, 20, self.ensemble_number).repeat(voxel_index * gui_number, 1)
        self._hp_log = self._hp_log.T.reshape(self.ensemble_number, voxel_index, gui_number)
        for i in range(gui_number):
            idx = np.random.choice(self.ensemble_number, self.ensemble_number, replace=False)
            self._hp_log[:, :, i] = self._hp_log[idx, :, i]
        self._hp = self.sigmoid_torch(self._hp_log.reshape(self.ensemble_number, -1), self._hp_low, self._hp_high)
        return self._hp.reshape(-1)

    def da_property_initialize(self, property_index, alpha, gui, voxel_index=None):
        """
        Update g_ui parameters and hyper-parameters

        Complete assignment: self._property_index, self._hp_index_in_voxel

        Parameters
        ----------

        property_index: list
            gui index in brain property

        alpha: int
            concentration of Gamma distribution which gui parameters follows

        gui: torch.tensor float32 shape=(len(voxel_index, gui_number))
            value of hyper parameters updated

        voxel_index: torch.tensor int64
            index of voxel assimilated or updated

        """
        self.gamma_initialize(property_index, alpha, alpha)
        self._property_index = torch.tensor(property_index).type_as(self.populations).reshape(-1)
        voxel_index = self.populations if voxel_index is None else voxel_index
        self._hp_index_in_voxel = torch.stack((torch.meshgrid(voxel_index, self._property_index)), dim=1).reshape(-1, 2)
        self.mul_property_by_subblk(self._hp_index_in_voxel.type_as(torch.int64), gui.reshape(-1))

    def get_hidden_state(self, steps, show_info=False):
        """
        The block evolve one TR time, i.e, 800 ms as default setting.

        Complete assignment: self._hidden_state

        Parameters
        ----------
        steps: int default=800
            iter number in one observation time point.

        show_info: bool, default=False
            Show frequency and BOLD signals if show_info is True

        """
        out = self.evolve(step=steps, vmean_option=False, sample_option=False, bold_detail=True)
        self._hidden_state = torch.cat((self._hp_log.reshape(self.ensemble_number, self.single_voxel_number, -1),
                                        out[0].reshape(self.ensemble_number, self.single_voxel_number, 1),
                                        out[1].reshape(self.ensemble_number, self.single_voxel_number, 1),
                                        out[2].reshape(self.ensemble_number, self.single_voxel_number, 1),
                                        out[3].reshape(self.ensemble_number, self.single_voxel_number, 1),
                                        out[4].reshape(self.ensemble_number, self.single_voxel_number, 1),), dim=2)
        if show_info:
            print(f'(Frequency.max, mean, min)={out[0].max(), out[0].mean(), out[0].min()}')
            print(f'(BOLD.max, mean, min)={out[4].max(), out[4].mean(), out[4].min()}')

    def da_evolve(self, steps=800):
        """
        The block evolve one TR time, i.e, 800 ms as default setting.

        Parameters
        ----------
        steps: int, default=800
            iter number in one observation time point.

        """
        self._hp_log += torch.normal(0, self._hp_sigma**0.5, size=self._hp_log.shape).type_as(self._hp_log)
        self._hp = self.sigmoid_torch(self._hp_log.reshape(self.ensemble_number, -1), self._hp_low, self._hp_high)
        self.mul_property_by_subblk(self._hp_index_in_voxel, self._hp.reshape(-1))
        self.get_hidden_state(steps)

    def da_filter(self, bold_real_t, debug=False):
        """
        Correct hidden_state by diffusion ensemble Kalman filter

        Parameters
        ----------
        bold_real_t: torch.tensor, shape=(1, single_voxel_number)
            Real BOLD signal to assimilate

        debug: bool, default=False
            Return hidden_state to debug if debug is true

        """
        if debug:
            self._hidden_state, w_debug = diffusion_EnKF(self._hidden_state, self.bold_sigma, bold_real_t,
                                                         self.solo_rate, debug)
            return w_debug
        else:
            self._hidden_state = diffusion_EnKF(self._hidden_state, self.bold_sigma, bold_real_t, self.solo_rate)
        self._hp_log = self._hidden_state[:, :, :self._hp_num]
        self.bold.state_update(self._hidden_state[:, :, -5:-1])

    def da_rest_run(self, bold_real, write_path, observation_times=None):
        """
        Run data assimilation to simulate the BOLD signals and estimate the hyper-parameters

        Save hyper-parameters, hidden state and draw figures

        Parameters
        ----------
        bold_real: torch.tensor, shape=(t, single_voxel_number)
            Real BOLD signal to assimilate

        write_path: str
            Path where array and fig save

        observation_times: int
            iteration time of data assimilation
            Set to observation times of BOLD signals if observation_times is None

        """
        w_save = []
        w_fix = []
        observation_times = len(bold_real) if observation_times is None else observation_times
        for t in range(observation_times):
            start_time = time.time()
            self.da_evolve(800)
            w_save.append(torch2numpy(self._hidden_state))
            self.da_filter(bold_real[t].reshape(1, self.single_voxel_number))
            w_fix.append(torch2numpy(self._hidden_state))
            if t <= 9 or t % 50 == 49 or t == (observation_times - 1):
                np.save(os.path.join(write_path, "w.npy"), w_save)
                np.save(os.path.join(write_path, "w_fix.npy"), w_fix)
            print("------------run da" + str(t) + ":" + str(time.time() - start_time))
        bold_assimilation = torch.stack(w_save)[:, :, :, -1]
        np.save(os.path.join(write_path, "bold_assimilation.npy"), bold_assimilation)
        self.plot_bold(write_path, bold_real, bold_assimilation, np.arange(10))
        hp_save_log = torch.stack(w_save)[:, :, :, :self._hp_num]
        hp_sm = self.sigmoid_torch(hp_save_log, self._hp_low, self._hp_high)
        np.save(os.path.join(write_path, "hp_sm.npy"), hp_sm.mean(1))
        self.plot_hp(write_path, None, hp_sm, np.arange(10), self._hp_num, 'hp_sm')
        hp_ms = self.sigmoid_torch(hp_save_log.mean(1), self._hp_low, self._hp_high)
        np.save(os.path.join(write_path, "hp_ms.npy"), hp_ms)
        self.plot_hp(write_path, None, np.expand_dims(hp_ms, 1), np.arange(10), self._hp_num, 'hp_ms')
