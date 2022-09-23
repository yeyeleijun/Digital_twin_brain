# -*- coding: utf-8 -*- 
# @Time : 2022/8/10 14:31 
# @Author : lepold
# @File : test_generation.py
import os
import unittest
import h5py
# from mpi4py import MPI
import numpy as np

from generation.make_block import *
import sparse


class TestBlock(unittest.TestCase):
    @staticmethod
    def _make_directory_tree(root_path, scale, degree, init_min, init_max, extra_info):
        """
        make directory tree for each subject.

        Parameters
        ----------
        root_path: str
            each subject has a root path.

        scale: int
            number of neurons of whole brain.
        degree:
            in-degree of each neuron.

        init_min: float
            the lower bound of uniform distribution where w is sampled from.

        init_max: float
            the upper bound of uniform distribution where w is sampled from.

        extra_info: str
            supplementary information.

        Returns
        ----------
        second_path: str
            second path to save connection table

        """
        os.makedirs(root_path, exist_ok=True)
        os.makedirs(os.path.join(root_path, "raw_data"), exist_ok=True)
        second_path = os.path.join(root_path, f"dti_distribution_{int(scale//1e6)}m_d{degree}_w{init_min}_{init_max}_{extra_info}")
        os.makedirs(second_path, exist_ok=True)
        os.makedirs(os.path.join(second_path, "single"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "ensembles"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "supplementary_info"), exist_ok=True)
        os.makedirs(os.path.join(second_path, "DA"), exist_ok=True)

        return second_path

    @staticmethod
    def _add_laminar_cortex_model(conn_prob, gm, canonical_voxel=False):
        """
        Process the connection probability matrix, grey matter and degree scale for DTB with pure voxel and micro-column
        structure.  Each voxel is split into 2 populations (E and I). Each micro-column is spilt into 10 populations
        (L1E, L1I, L2/3E, L2/3I, L4E, L4I, L5E, L5I, L6E, L6I).

        Parameters
        ----------
        conn_prob: numpy.ndarray, shape [N, N]
            the connectivity probability matrix between N voxels/micro-columns.

        gm: numpy.ndarray, shape [N]
            the normalized grey matter in each voxel/micro-column.

        canonical_voxel: bool
            Ture for voxel structure; False for micro-column structure.

        Returns
        -------
        out_conn_prob: numpy.ndarray
            connectivity probability matrix between populations (shape [2*N, 2*N] for voxel; shape[10*N, 10*N] for micro
            -column) in the sparse matrix form.

        out_gm: numpy.ndarray
            grey matter for populations in DTB (shape [2*N] for voxel; shape[10*N] for micro-column).

        out_degree_scale: numpy.ndarray
            scale of degree for populations in DTB (shape [2*N] for voxel; shape[10*N] for micro-column).

        """
        if not canonical_voxel:
            lcm_connect_prob = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 3554, 804, 881, 45, 431, 0, 136, 0, 1020],
                                         [0, 0, 1778, 532, 456, 29, 217, 0, 69, 0, 396],
                                         [0, 0, 417, 84, 1070, 690, 79, 93, 1686, 0, 1489],
                                         [0, 0, 168, 41, 628, 538, 36, 0, 1028, 0, 790],
                                         [0, 0, 2550, 176, 765, 99, 621, 596, 363, 7, 1591],
                                         [0, 0, 1357, 76, 380, 32, 375, 403, 129, 0, 214],
                                         [0, 0, 643, 46, 549, 196, 327, 126, 925, 597, 2609],
                                         [0, 0, 80, 8, 92, 3, 159, 11, 76, 499, 1794]], dtype=np.float64
                                        )

            lcm_gm = np.array([0, 0,
                               33.8 * 78, 33.8 * 22,
                               34.9 * 80, 34.9 * 20,
                               7.6 * 82, 7.6 * 18,
                               22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
        else:
            # 4:1 setting, setting from wenyong
            # lcm_connect_prob = np.array([[0.3, 0.2, 0.5],
            #                              [0.3, 0.2, 0.5]], dtype=np.float64)
            # 4:1 setting, setting inferred from micro-column
            lcm_connect_prob = np.array([[4 / 7, 1 / 7, 2 / 7],
                                         [4 / 7, 1 / 7, 2 / 7]], dtype=np.float64)
            lcm_gm = np.array([0.8, 0.2], dtype=np.float64)

        lcm_gm /= lcm_gm.sum()

        syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
        lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
        lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
        lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)

        if conn_prob.shape[0] == 1:
            conn_prob[:, :] = 1
        else:
            conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
            conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)

        conn_prob[np.isnan(conn_prob)] = 0
        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape([-1])
        out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape([-1])
        conn_prob = sparse.COO(conn_prob)
        # only e5 is allowed to output.
        corrds1 = np.empty([4, conn_prob.coords.shape[1] * lcm_connect_prob.shape[0]], dtype=np.int64)
        if not canonical_voxel:
            corrds1[3, :] = 6
        else:
            corrds1[3, :] = 0
        corrds1[(0, 2), :] = np.broadcast_to(conn_prob.coords[:, :, None],
                                             [2, conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([2, -1])
        corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                          [conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([1, -1])

        data1 = (conn_prob.data[:, None] * lcm_connect_prob[:, -1]).reshape([-1])

        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, conn_prob.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(np.arange(conn_prob.shape[0], dtype=np.int64)[:, None],
                                        [conn_prob.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                                [conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

        out_conn_prob = sparse.COO(coords=np.concatenate([corrds1, corrds2], axis=1),
                                   data=np.concatenate([data1, data2], axis=0),
                                   shape=[conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                                          lcm_connect_prob.shape[1] - 1])

        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        if conn_prob.shape[0] == 1:
            out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
        return out_conn_prob, out_gm, out_degree_scale

    @staticmethod
    def _add_laminar_cortex_include_subcortical_in_whole_brain(conn_prob, gm, divide_point=22703):
        lcm_connect_prob = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 3554, 804, 881, 45, 431, 0, 136, 0, 1020],
                                     [0, 0, 1778, 532, 456, 29, 217, 0, 69, 0, 396],
                                     [0, 0, 417, 84, 1070, 690, 79, 93, 1686, 0, 1489],
                                     [0, 0, 168, 41, 628, 538, 36, 0, 1028, 0, 790],
                                     [0, 0, 2550, 176, 765, 99, 621, 596, 363, 7, 1591],
                                     [0, 0, 1357, 76, 380, 32, 375, 403, 129, 0, 214],
                                     [0, 0, 643, 46, 549, 196, 327, 126, 925, 597, 2609],
                                     [0, 0, 80, 8, 92, 3, 159, 11, 76, 499, 1794]], dtype=np.float64
                                    )

        # wenyong setting (0.3, 0.2, 0.5), cortical column is (4/7, 1/7, 2/7)
        lcm_connect_prob_subcortical = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 4 / 7, 1 / 7, 0, 0, 2 / 7],
                                                 [0, 0, 0, 0, 0, 0, 4 / 7, 1 / 7, 0, 0, 2 / 7],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                 ], dtype=np.float64)

        lcm_gm = np.array([0, 0,
                           33.8 * 78, 33.8 * 22,
                           34.9 * 80, 34.9 * 20,
                           7.6 * 82, 7.6 * 18,
                           22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
        lcm_gm /= lcm_gm.sum()

        syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
        lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
        lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
        lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)

        if conn_prob.shape[0] == 1:
            conn_prob[:, :] = 1
        else:
            conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
            conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)

        conn_prob[np.isnan(conn_prob)] = 0
        N = len(conn_prob)
        cortical = np.arange(divide_point, dtype=np.int32)
        sub_cortical = np.arange(divide_point, N, dtype=np.int32)
        conn_prob_cortical = conn_prob[cortical]
        conn_prob_subcortical = conn_prob[sub_cortical]

        out_gm = (gm[:, None] * lcm_gm[None, :]).reshape(
            [-1])  # shape[cortical_voxel, 10] reshape to [10 * cortical_voxel]
        for i in sub_cortical:
            out_gm[10 * i:10 * (i + 1)] = 0.
            out_gm[10 * i + 6] = gm[i] * 0.8
            out_gm[10 * i + 7] = gm[i] * 0.2

        out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape(
            [-1])  # shape[cortical_voxel, 10] reshape to [10 * cortical_voxel]
        for i in sub_cortical:
            out_degree_scale[10 * i:10 * (i + 1)] = 0.
            out_degree_scale[10 * i + 6] = 1.
            out_degree_scale[10 * i + 7] = 1.

        """
        deal with outer_connection of cortical voxel 
        """
        conn_prob = sparse.COO(conn_prob)
        conn_prob_cortical = sparse.COO(conn_prob_cortical)
        index_cortical = np.in1d(conn_prob.coords[0], cortical)
        coords_cortical = conn_prob.coords[:, index_cortical]
        # only e5 is allowed to output.
        corrds1 = np.empty([4, coords_cortical.shape[1] * lcm_connect_prob.shape[0]], dtype=np.int64)

        corrds1[3, :] = 6
        corrds1[(0, 2), :] = np.broadcast_to(coords_cortical[:, :, None],
                                             [2, coords_cortical.shape[1], lcm_connect_prob.shape[0]]).reshape([2, -1])
        corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                          [coords_cortical.shape[1], lcm_connect_prob.shape[0]]).reshape([1, -1])
        data1 = (conn_prob_cortical.data[:, None] * lcm_connect_prob[:, -1]).reshape([-1])

        """
        deal with outer_connection of subcortical voxel
        """
        conn_prob_subcortical = sparse.COO(conn_prob_subcortical)
        index_subcortical = np.in1d(conn_prob.coords[0], sub_cortical)
        coords_subcortical = conn_prob.coords[:, index_subcortical]
        coords3 = np.empty([4, coords_subcortical.shape[1] * 2], dtype=np.int64)
        coords3[3, :] = 6
        coords3[(0, 2), :] = np.broadcast_to(coords_subcortical[:, :, None],
                                             [2, coords_subcortical.shape[1], 2]).reshape([2, -1])
        coords3[(1), :] = np.broadcast_to(np.arange(6, 8, dtype=np.int64)[None, :],
                                          [coords_subcortical.shape[1], 2]).reshape([1, -1])
        data3 = (conn_prob_subcortical.data[:, None] * lcm_connect_prob_subcortical[6:8, -1]).reshape([-1])

        """
        deal with inner_connection of cortical voxel
        """
        lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
        corrds2 = np.empty([4, cortical.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
        corrds2[0, :] = np.broadcast_to(cortical[:, None],
                                        [cortical.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
        corrds2[2, :] = corrds2[0, :]
        corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                             [2, cortical.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
            [2, -1])
        data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                                [cortical.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

        """
        deal with inner_connection of subcortical voxel
        """
        lcm_connect_prob_inner_subcortical = sparse.COO(lcm_connect_prob_subcortical[:, :-1])
        corrds4 = np.empty([4, sub_cortical.shape[0] * lcm_connect_prob_inner_subcortical.data.shape[0]],
                           dtype=np.int64)
        corrds4[0, :] = np.broadcast_to(sub_cortical[:, None],
                                        [sub_cortical.shape[0],
                                         lcm_connect_prob_inner_subcortical.data.shape[0]]).reshape(
            [-1])
        corrds4[2, :] = corrds4[0, :]
        corrds4[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner_subcortical.coords[:, None, :],
                                             [2, sub_cortical.shape[0],
                                              lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
            [2, -1])
        data4 = np.broadcast_to(lcm_connect_prob_inner_subcortical.data[None, :],
                                [sub_cortical.shape[0], lcm_connect_prob_inner_subcortical.coords.shape[1]]).reshape(
            [-1])

        out_conn_prob = sparse.COO(coords=np.concatenate([corrds1, corrds2, coords3, corrds4], axis=1),
                                   data=np.concatenate([data1, data2, data3, data4], axis=0),
                                   shape=[conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                                          lcm_connect_prob.shape[1] - 1])

        out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                               conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
        if conn_prob.shape[0] == 1:
            out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
        return out_conn_prob, out_gm, out_degree_scale

    def test_make_small_block(self, write_path="../data/small_blocks/critical_blocks_d100", initial_parameter=(2.13037975e-02, 1.39240506e-04, 1.58227848e-01, 1.89873418e-02)):
        prob = torch.tensor([[0.8, 0.2], [0.8, 0.2]])
        tau_ui = (8, 40, 10, 50)
        population_kwards = [{'g_Li': 0.03,
                              'g_ui': initial_parameter,
                              'T_ref': 5,
                              "V_reset": -65,
                              "noise_rate": 0.0003,
                              'tao_ui': tau_ui,
                              'size': num} for num in [1600, 400]]
        conn = connect_for_multi_sparse_block(prob, population_kwards, degree=100, init_min=1,
                                       init_max=1, prefix=None)
        merge_dti_distributation_block(conn, write_path,
                                       MPI_rank=None,
                                       number=4,
                                       dtype="single",
                                       debug_block_dir=None)
        print("Done")

    def _test_make_whole_brain_include_cortical_laminar_and_subcortical_voxel_model(self,
                                                                                   path="./laminar_structure_whole_brain_include_subcortical/200m_structure_d100",
                                                                                   degree=100,
                                                                                   minmum_neurons_for_block=0,
                                                                                   scale=int(2e8),
                                                                                   blocks=40):
        """
        generate the whole brian connection table at the cortical-column version, and generate index file of populations.
        In simulation, we can use the population_base.npy to sample which neurons we need to track.

        Parameters
        ----------
        path: str
            the path to save information.

        degree: int
            default is 100.

        minmum_neurons_for_block: int
            In cortical-column version, it must be zero.

        scale: int
            simualation size.

        blocks : int
            equals number of gpu cards.


        """
        os.makedirs(path, exist_ok=True)
        whole_brain = np.load(
            '/public/home/ssct004t/project/zenglb/spiking_nn_for_simulation/whole_brain_voxel_info.npz')
        conn_prob = whole_brain["conn_prob"]
        block_size = whole_brain["block_size"]
        divide_point = int(whole_brain['divide_point'])
        cortical = np.arange(divide_point, dtype=np.int32)
        conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
        conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)
        block_size = block_size / block_size.sum()
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_include_subcortical_in_whole_brain(conn_prob,
                                                                                                          block_size,
                                                                                                          divide_point)
        # for 100 degree
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

        degree = np.maximum((degree * degree_scale).astype(np.uint16),
                            1)
        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui_laminar[i % 10] if np.isin(i // 10, cortical) else gui_voxel,
                   "size": int(max(b * scale, minmum_neurons_for_block))
                   }
                  for i, b in enumerate(block_size)]

        population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
        population_base = np.add.accumulate(population_base)
        population_base = np.insert(population_base, 0, 0)
        os.makedirs(os.path.join(path, 'supplementary_info'), exist_ok=True)
        np.save(os.path.join(path, 'supplementary_info', "population_base.npy"), population_base)

        conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           number=blocks,
                                           dtype="single",
                                           debug_block_dir=None)

    def _test_generate_normal_voxel_whole_brain(self, root_path="./data/jianfeng_normal", degree=100,
                                                minimum_neurons_for_block=(200, 50),
                                                scale=int(1e8), init_min=1, init_max=1):
        second_path = self._make_directory_tree(root_path, scale, degree, init_min, init_max, "critical")
        blocks = int(scale / 5e6)
        print(f"Total {scale} neurons for DTB, merge to {blocks} blocks")
        file = h5py.File(
            '/public/home/ssct004t/project/yeleijun/spiking_nn_for_brain_simulation/data/jianfeng_normal/A1_1_DTI_voxel_structure_data_jianfeng.mat',
            'r')
        block_size = file['dti_grey_matter'][0]
        dti = np.float32(file['dti_net_full'])

        nonzero_gm = (block_size > 0).nonzero()[0]
        nonzero_dti = (dti.sum(axis=1) > 0).nonzero()[0]
        nonzero_all = np.intersect1d(nonzero_gm, nonzero_dti)
        print(f"valid voxel index length {len(nonzero_all)}")
        print(f"valid voxel index {nonzero_all}")
        block_size = block_size[nonzero_all]
        block_size /= block_size.sum()
        conn_prob = dti[np.ix_(nonzero_all, nonzero_all)]
        conn_prob[np.diag_indices_from(dti)] = 0
        # conn_prob[np.diag_indices_from(dti)] = conn_prob.sum(axis=1) * 5 / 3  # intra_E:inter_E:I=5:3:2
        conn_prob /= conn_prob.sum(axis=1, keepdims=True)
        conn_prob, block_size, degree_scale = self._add_laminar_cortex_model(conn_prob, block_size,
                                                                             canonical_voxel=True)
        # gui = np.array([6.1644921e-03, 8.9986715e-04, 2.9690875e-02, 1.9053727e-03], dtype=np.float64) # old setting, noise rate=0.01, totally steady spike, 1ms iteration
        gui = np.array([0.01837975, 0.00072405, 0.10759494, 0.02180028], dtype=np.float64)  # find in sub critical in 3hz noise rate and 0.1ms iteration resolution.
        degree = np.maximum((degree * degree_scale).astype(np.uint16), 1)

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   'tao_ui': (8, 40, 10, 50),  # old setting: [2, 40, 10, 50]
                   'noise_rate': 0.003,  # old setting: 0.01 Hz
                   "size": int(max(b * scale, minimum_neurons_for_block[i % 2]))}
                  for i, b in enumerate(block_size)]

        population_base = np.array([kword['size'] for kword in kwords], dtype=np.int64)
        population_base = np.add.accumulate(population_base)
        population_base = np.insert(population_base, 0, 0)
        np.save(os.path.join(second_path, "supplementary_info", "population_base.npy"), population_base)

        conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=1,  # have modified as in the criticalNN experiment.
                                              init_max=1)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, blocks, size):
            merge_dti_distributation_block(conn, second_path,
                                           MPI_rank=i,
                                           number=blocks,
                                           dtype="single",
                                           debug_block_dir=None)

    def _test_generate_EI_big_network(self, path="./canonical_ei_big_network", degree=100,
                                      minmum_neurons_for_block=1000,
                                      scale=int(1e6)):

        degree_scale = np.array([1., 1.])
        gui = np.array([(0.0200, 0.0010, 0.0851, 0.0063)], dtype=np.float64)
        degree = np.maximum((degree * degree_scale).astype(np.uint16), 1)
        block_size = np.array([0.8, 0.2])
        conn_prob = np.array([[0.8, 0.2],
                              [0.8, 0.2]])

        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': gui,
                   'tao_ui': (2, 40, 10, 50),
                   "size": int(max(b * scale, minmum_neurons_for_block))
                   }
                  for i, b in enumerate(block_size)]
        conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        for i in range(rank, 1, size):
            merge_dti_distributation_block(conn, path,
                                           MPI_rank=i,
                                           number=1,
                                           dtype="single",
                                           debug_block_dir=None)


if __name__ == "__main__":
    unittest.main()
