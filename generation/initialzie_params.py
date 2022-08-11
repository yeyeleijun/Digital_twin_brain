# -*- coding: utf-8 -*- 
# @Time : 2022/8/10 14:25 
# @Author : lepold
# @File : initialzie_params.py

import torch
import os
from generation.read_block import connect_for_block
from models.block import block
from make_block import connect_for_multi_sparse_block


def initilalize_gui_in_homo_block(delta_t=1, default_Hz=20, max_output_Hz=100, T_ref=5, degree=100, g_Li=0.03,
                                 V_L=-75,
                                 V_rst=-65, V_th=-50, path="./initialize_gui"):
    """
    A heuristic method to find appropriate parameter for single neuron.

    """

    gap = V_th - V_rst

    noise_rate = 1 / (1000 / delta_t / default_Hz)
    max_delta_raise = gap / (1000 / delta_t / max_output_Hz - T_ref)
    default_delta_raise = gap / (1000 / delta_t / default_Hz - T_ref)

    leaky_compensation = g_Li * ((V_th + V_rst) / 2 - V_L)

    label = torch.tensor([0.7 * (max_delta_raise + leaky_compensation),
                          0.3 * (max_delta_raise + leaky_compensation),
                          0.7 * (max_delta_raise - default_delta_raise),
                          0.3 * (max_delta_raise - default_delta_raise)])
    print(label.tolist())

    gui = label

    def _test_gui(max_iter=4000, noise_rate=noise_rate):
        property, w_uij = connect_for_block(os.path.join(path, 'single'))
        property = property.cuda()
        w_uij = w_uij.cuda()
        B = block(
            node_property=property,
            w_uij=w_uij,
            delta_t=delta_t,
        )
        out_list = []
        Hz_list = []

        for k in range(max_iter):
            B.run(noise_rate=noise_rate, isolated=True)
            out_list.append(B.I_ui.mean(-1).abs())
            Hz_list.append(float(B.active.sum()) / property.shape[0])
        out = torch.stack(out_list[-1000:]).mean(0)
        Hz = sum(Hz_list[-1000:])
        print('out:', out.tolist(), "Hz: ", Hz)
        return out.cpu(), Hz

    for i in range(20):
        if os.path.exists(os.path.join(path, 'block_0.npz')):
            os.remove(os.path.join(path, 'block_0.npz'))
        prob = torch.tensor([[0.8, 0.2], [0.8, 0.2]], dtype=torch.float32)
        population_kwards = [{'g_Li': g_Li,
                              'g_ui': gui,
                              "V_reset": -65,
                              'tao_ui': (2, 40, 10, 50),
                              'size': size} for size in [1600, 400]]
        connect_for_multi_sparse_block(prob, population_kwards, degree=degree, init_min=0,
                                       init_max=1, prefix=path)
        out, Hz = _test_gui()
        gui = gui * label / out
        print('gui:', gui.tolist())
        with open(os.path.join(path, "./iteraion_log.txt"), "a") as f:
            f.write("\n\n")
            f.write("iteration: {}".format(iter))
            f.write("\n")
            f.write("gui: " + str(gui))
            f.write("\n")
            f.write("Hz: " + str(Hz))

    print("Done, information is write in iteration_log.txt")