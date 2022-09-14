import os
import prettytable as pt
from cuda_resample0710.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
# from cuda.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from models.bold_model_pytorch import BOLD
import time
import torch
import numpy as np
import matplotlib.pyplot as mp
import h5py

mp.switch_backend('Agg')


class DA_MODEL:
    def __init__(self, block_dict: dict, bold_dict: dict, steps=800, ensembles=100, time=400, hp_sigma=1.,
                 bold_sigma=1e-6):
        """
        Mainly for the whole brain model consisting of cortical functional column structure and canonical E/I=4:1 structure.

       Parameters
       ----------
       block_name: str
           block name.
       block_dict : dict
           contains the parameters of the block model.
       bold_dict : dict
           contains the parameters of the bold model.
        """
        self.block = block_gpu(block_dict['ip'], block_dict['block_path'], block_dict['noise_rate'],
                               block_dict['delta_t'], block_dict['print_stat'], block_dict['force_rebase'])  # !!!!
        # self.block = block_gpu(block_dict['ip'], block_dict['block_path'],
        #                        block_dict['delta_t'], print_stat=block_dict['print_stat'],
        #                        force_rebase=block_dict['force_rebase'])  # !!!!

        self.noise_rate = block_dict['noise_rate']
        self.delta_t = block_dict['delta_t']
        self.bold = BOLD(bold_dict['epsilon'], bold_dict['tao_s'], bold_dict['tao_f'], bold_dict['tao_0'],
                         bold_dict['alpha'], bold_dict['E_0'], bold_dict['V_0'])
        self.ensembles = ensembles
        self.num_populations = int(self.block.total_subblks)
        print("num_populations", self.num_populations)
        self.num_populations_in_one_ensemble = int(self.num_populations / self.ensembles)
        self.num_neurons = int(self.block.total_neurons)
        self.num_neurons_in_one_ensemble = int(self.num_neurons / self.ensembles)
        self.populations_id = self.block.subblk_id.cpu().numpy()
        print("len(populations_id)", len(self.populations_id))
        self.neurons = self.block.neurons_per_subblk
        self.populations_id_per_ensemble = np.split(self.populations_id, self.ensembles)

        # Update noise rate to setting noise rate
        # population_info = np.stack(np.meshgrid(self.populations_id, np.array([0], dtype=np.int64), indexing="ij"),
        #                            axis=-1).reshape((-1, 2))
        # population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        # alpha = torch.ones(self.num_populations, device="cuda:0") * block_dict['noise_rate'] * 1e8
        # beta = torch.ones(self.num_populations, device="cuda:0") * 1e8
        # self.block.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

        self.T = time
        self.steps = steps
        self.hp_sigma = hp_sigma
        self.bold_sigma = bold_sigma

    @staticmethod
    def log(val, low_bound, up_bound, scale=10):
        assert torch.all(torch.le(val, up_bound)) and torch.all(
            torch.ge(val, low_bound)), "In function log, input data error!"
        return scale * (torch.log(val - low_bound) - torch.log(up_bound - val))

    @staticmethod
    def sigmoid(val, low_bound, up_bound, scale=10):
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(val / scale)
        elif isinstance(val, np.ndarray):
            assert np.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(
                torch.from_numpy(val.astype(np.float32)) / scale).numpy()
        else:
            raise ValueError("val type is wrong!")

    @staticmethod
    def torch_2_numpy(u, is_cuda=True):
        assert isinstance(u, torch.Tensor)
        if is_cuda:
            return u.cpu().numpy()
        else:
            return u.numpy()

    @staticmethod
    def show_bold(W, bold, T, path, voxel_num):
        iteration = [i for i in range(T)]
        for i in range(voxel_num):
            print("show_bold" + str(i))
            fig = mp.figure(figsize=(5, 3), dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(iteration, bold[:T, i], 'r-', label="Raw")
            ax1.plot(iteration, np.mean(W[:T, :, i, -1], axis=1), 'b-', label="Assimilation")
            mp.fill_between(iteration, np.mean(W[:T, :, i, -1], axis=1) -
                            np.std(W[:T, :, i, -1], axis=1), np.mean(W[:T, :, i, -1], axis=1)
                            + np.std(W[:T, :, i, -1], axis=1), color='b', alpha=0.2)
            mp.legend()
            mp.ylim((0.0, 0.08))
            ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
            mp.savefig(os.path.join(path, "bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
            mp.close(fig)
        return None

    @staticmethod
    def show_hp(hp, T, path, voxel_num, hp_num, hp_real=None):
        iteration = [i for i in range(T)]
        for i in range(voxel_num):
            for j in range(hp_num):
                print("show_hp", i, 'and', j)
                fig = mp.figure(figsize=(5, 3), dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(iteration, np.mean(hp[:T, :, i, j], axis=1), 'b-')
                if hp_real is None:
                    pass
                else:
                    ax1.plot(iteration, np.tile(hp_real[j], T), 'r-')
                mp.fill_between(iteration, np.mean(hp[:T, :, i, j], axis=1) -
                                np.sqrt(np.var(hp[:T, :, i, j], axis=1)), np.mean(hp[:T, :, i, j], axis=1)
                                + np.sqrt(np.var(hp[:T, :, i, j], axis=1)), color='b', alpha=0.2)
                ax1.set(xlabel='observation time/800ms', ylabel='hyper parameter')
                mp.savefig(os.path.join(path, "hp" + str(i) + "_" + str(j) + ".png"), bbox_inches='tight', pad_inches=0)
                mp.close(fig)
        return None

    def initial_model(self, real_parameter, para_ind):
        """
        initialize the block model, and then determine the random walk range of hyper parameter,
        -------
        """
        raise NotImplementedError

    def evolve(self, steps=800):
        """
        evolve the block model and obtain prediction observation,
        here we apply the MC method to evolve samples drawn from initial Gaussian distribution.
        -------

        """
        raise NotImplementedError

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        use kalman filter to filter the latent state.
        -------

        """
        raise NotImplementedError


class DA_Rest_Whole_Brain_Voxel(DA_MODEL):
    def __init__(self, block_dict: dict, bold_dict: dict, whole_brain_info: str, steps, ensembles, time, hp_sigma,
                 bold_sigma):
        super().__init__(block_dict, bold_dict, steps, ensembles, time, hp_sigma, bold_sigma)
        self.device = "cuda:0"
        self.populations_per_voxel = 2
        self.num_voxels = self.num_populations // self.populations_per_voxel  # split EI = True
        self.num_voxel_in_one_ensemble = self.num_populations_in_one_ensemble // self.populations_per_voxel  # split EI = True
        assert self.populations_id.max() == self.num_populations_in_one_ensemble * self.ensembles - 1, "population_id is not correct!"
        self.neurons_per_population = self.block.neurons_per_subblk.float()  # !!! Tensor.cuda()
        self.neurons_per_voxel_cpu = \
        np.histogram(self.populations_id, weights=self.neurowhowhns_per_population.cpu().numpy(), bins=self.num_voxels,
                     range=(0, self.num_populations))[0]
        self.neurons_per_voxel = torch.from_numpy(self.neurons_per_voxel_cpu).cuda()

        self.hp_num = None
        self.up_bound = None
        self.low_bound = None
        self.hp = None
        self.hp_log = None

    def __str__(self):
        print("DA FOR REST WHOLE BRAIN")

    @staticmethod
    def random_walk_range(real_parameter, num_voxels, up_times=3, low_times=2):
        if real_parameter.ndim == 2:
            temp_up = np.tile(real_parameter * up_times, (num_voxels, 1))
            temp_low = np.tile(real_parameter / low_times, (num_voxels, 1))
            return temp_up.reshape((num_voxels, -1)), temp_low.reshape((num_voxels, -1))
        else:
            raise NotImplementedError("real_parameter.ndim=1 is waiting to complete")

    def initial_model(self, real_parameter, para_ind, up_times=3, low_times=2):
        start = time.time()

        assert isinstance(real_parameter, np.ndarray)
        para_ind = para_ind.astype(np.int64)  # [10, 12]

        real_parameter = real_parameter.astype(np.float32)  # shape [1, 4]
        self.hp_num = real_parameter.shape[0] * real_parameter.shape[1]  # 4

        for i in para_ind:
            _, _ = self.block.update_property_by_property_idx(torch.LongTensor([i]).cuda(), 5, 5)  # !!!!
        # for k in para_ind:
        #     population_info = np.stack(np.meshgrid(self.populations_id, k, indexing="ij"),
        #                                axis=-1).reshape((-1, 2))
        #     population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        #     alpha = torch.ones(self.num_populations, device="cuda:0") * 5
        #     beta = torch.ones(self.num_populations, device="cuda:0") * 5
        #     self.block.gamma_property_by_subblk(population_info, alpha, beta)

        # CPU numpy to GPU ndarray
        self.up_bound, self.low_bound = self.random_walk_range(real_parameter, self.num_voxel_in_one_ensemble,
                                                               up_times=up_times, low_times=low_times)
        self.hp = np.linspace(self.low_bound, self.up_bound, num=3 * self.ensembles,
                              dtype=np.float32)[
                  self.ensembles: - self.ensembles]  # shape: [ensembles, num_voxel, num_hp], 4 for split_EI=True and para_ind=[10, 12]
        for i in range(self.hp_num):
            idx = np.random.choice(self.ensembles, self.ensembles, replace=False)
            self.hp[:, :, i] = self.hp[idx, :, i]
        self.hp = torch.from_numpy(self.hp).cuda()

        self.up_bound, self.low_bound = torch.from_numpy(self.up_bound).cuda(), torch.from_numpy(self.low_bound).cuda()

        print(f"Multiply populations hp")
        hp_info = np.stack(
            np.meshgrid(self.populations_id.astype(np.int64), para_ind.astype(np.int64), indexing="ij"),
            axis=-1).reshape((-1, 2))
        hp_info = torch.from_numpy(hp_info).cuda()
        self.block.mul_property_by_subblk(hp_info, self.hp.reshape(-1))
        print(f"=================Initial DA MODEL done! cost time {time.time() - start:.2f}==========================")

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        distributed ensemble kalman filter. for single voxel, it modified both by its own observation and also
        by other's observation with distributed rate.

        Parameters


        w_hat  :  store the state matrix, shape=(ensembles, voxels, states)
        bold_y_t : shape=(voxels)
        rate : distributed rate
        """
        ensembles, voxels, total_state_num = w_hat.shape  # ensemble, brain_n, hp_num+act+hemodynamic(total state num)
        assert total_state_num == self.hp_num + 6
        w = w_hat.clone()
        w_mean = torch.mean(w_hat, dim=0, keepdim=True)
        w_diff = w_hat - w_mean
        w_cx = w_diff[:, :, -1] * w_diff[:, :, -1]
        w_cxx = torch.sum(w_cx, dim=0) / (self.ensembles - 1) + self.bold_sigma
        temp = w_diff[:, :, -1] / (w_cxx.reshape([1, voxels])) / (self.ensembles - 1)  # (ensemble, brain)
        model_noise = self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(self.ensembles, voxels)).type_as(temp)
        w += rate * (bold_y_t + model_noise - w_hat[:, :, -1])[:, :, None] * torch.sum(
            temp[:, :, None] * w_diff.reshape([self.ensembles, voxels, total_state_num]), dim=0, keepdim=True)
        # print("min1:", torch.min(w[:, :, :self.hp_num]).item(), "max1:", torch.max(w[:, :, :self.hp_num]).item())
        w += (1 - rate) * torch.mm(torch.mm(bold_y_t + model_noise - w_hat[:, :, -1], temp.T) / voxels,
                                   w_diff.reshape([self.ensembles, voxels * total_state_num])).reshape(
            [self.ensembles, voxels, total_state_num])
        print("after filter", "hp_log_min:",
              torch.min(w[:, :, :self.hp_num]).item(), "hp_log_max:",
              torch.max(w[:, :, :self.hp_num]).item())
        return w

    def evolve(self, steps=800):
        print(f"evolve:")
        start = time.time()
        act = None
        out = None
        steps = steps if steps is not None else self.steps
        for freqs in self.block.run(steps, freqs=True, vmean=False, sample_for_show=False):
            freqs = freqs.float().cpu().numpy()
            act = np.histogram(self.populations_id, weights=freqs, bins=self.num_voxels, range=(0, self.num_populations))[0]
            act = (act / self.neurons_per_voxel.cpu().numpy()).reshape(-1)
            act = torch.from_numpy(act).cuda()
            out = self.bold.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))

        print(
            f'active: {act.mean().item():.3f},  {act.min().item():.3f} ------> {act.max().item():.3f}')

        bold = torch.stack(
            [self.bold.s, self.bold.q, self.bold.v, self.bold.f_in, out])

        # print("cortical max bold_State: s, q, v, f_in, bold ", bold1.max(1)[0].data)
        print("bold range:", bold[-1].min().data, "------>>", bold[-1].max().data)

        w = torch.cat((self.hp_log, act.reshape([self.ensembles, -1, 1]),
                       bold.T.reshape([self.ensembles, -1, 5])), dim=2)
        print(f'End evolving, totally cost time: {time.time() - start:.2f}')
        return w

    def run(self, real_parameter, para_ind, whole_brain_info="whole_brain_voxel_info.npz", write_path="./"):
        total_start = time.time()

        tb = pt.PrettyTable()
        tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
        tb.add_row([1, "name", "rest_brain", "ensembles", self.ensembles])
        tb.add_row([2, "total_populations", self.num_populations, "populations_per_ensemble",
                    self.num_populations_in_one_ensemble])
        tb.add_row([3, "total_neurons", self.num_neurons, "neurons_per_ensemble", self.num_neurons_in_one_ensemble])
        tb.add_row([4, "voxels_per_ensemble", self.num_voxel_in_one_ensemble, "populations_per_voxel",
                    self.num_populations_in_one_ensemble // self.num_voxel_in_one_ensemble])
        tb.add_row([5, "total_time", self.T, "steps", self.steps])
        tb.add_row([6, "hp_sigma", self.hp_sigma, "bold_sigma", self.bold_sigma])
        tb.add_row([7, "noise_rate", self.noise_rate, "bold_range", "0.02-0.05"])
        tb.add_row([8, "walk_up_bound", "x4", "walk_low_bound", "/4"])
        print(tb)

        self.initial_model(real_parameter, para_ind, up_times=4, low_times=4)  # !!!
        self.hp_log = self.log(self.hp, self.low_bound, self.up_bound)

        for i in range(5):
            w = self.evolve(steps=800)
        file = h5py.File(whole_brain_info, 'r')
        bold_rest = np.array(file['dti_rest_state']).T
        bold_rest = bold_rest[20:]
        bold_y = (bold_rest - bold_rest.mean(axis=0, keepdims=True)) / bold_rest.std(axis=0, keepdims=True)  # z-score
        bold_y = 0.02 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
        assert bold_y.shape[1] == self.num_voxel_in_one_ensemble
        bold_y = torch.from_numpy(bold_y.astype(np.float32)).cuda()

        w_save = [self.torch_2_numpy(w, is_cuda=True)]
        print("\n                 BEGIN DA               \n")
        # self.T = bold_y.shape[0]
        print("The time point of BOLD is ", self.T)
        for t in range(self.T - 1):
            print("\n   PROCESSING || %d  \n " % t)
            bold_y_t = bold_y[t].reshape([1, self.num_voxel_in_one_ensemble])
            self.hp_log = w[:, :, :self.hp_num] + (
                    self.hp_sigma ** 0.5 * torch.normal(0, 1, size=(
                self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num))).type_as(w)
            print("self.hp_log", self.hp_log.min().item(), self.hp_log.max().item())
            self.hp = self.sigmoid(self.hp_log, self.low_bound, self.up_bound)
            print("Hp, eg: ", self.hp[0, 0, :self.hp_num].data)

            print(f"Multiply populations hp")
            hp_info = np.stack(
                np.meshgrid(self.populations_id.astype(np.int64), para_ind.astype(np.int64), indexing="ij"),
                axis=-1).reshape((-1, 2))
            hp_info = torch.from_numpy(hp_info).cuda()
            self.block.mul_property_by_subblk(hp_info, self.hp.reshape(-1))

            w_hat = self.evolve()
            w_hat[:, :, -5:] += (self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(
                self.ensembles, self.num_voxel_in_one_ensemble, 5))).type_as(w_hat)

            w = self.filter(w_hat, bold_y_t, rate=0.8)
            self.bold.state_update(
                w[:, :self.num_voxel_in_one_ensemble, (self.hp_num + 1):(self.hp_num + 5)])
            w_save.append(self.torch_2_numpy(w_hat, is_cuda=True))

        print("\n                 END DA               \n")
        np.save(os.path.join(write_path, "W.npy"), w_save)
        del w_hat, w
        path = write_path + '/show/'
        os.makedirs(path, exist_ok=True)

        w_save = np.array(w_save)
        self.show_bold(w_save, self.torch_2_numpy(bold_y, is_cuda=True), self.T, path, 100)  # !!!
        hp_save = self.sigmoid(
            w_save[:, :, :, :self.hp_num].reshape(self.T * self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num),
            self.torch_2_numpy(self.low_bound), self.torch_2_numpy(self.up_bound))
        hp = hp_save.reshape(self.T, self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num).mean(axis=1)
        hp = hp.reshape([self.T, self.num_populations_in_one_ensemble, -1])
        np.save(os.path.join(write_path, "hp.npy"), hp)
        self.show_hp(
            hp_save.reshape(self.T, self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num),
            self.T, path, 100, self.hp_num)
        self.block.shutdown()
        print("\n\n Totally cost time:\t", time.time() - total_start)
        print("=================================================\n")

