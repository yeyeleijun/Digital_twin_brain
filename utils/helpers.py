# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 14:48 
# @Author : lepold
# @File : helpers.py

import torch
import numpy as np


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
