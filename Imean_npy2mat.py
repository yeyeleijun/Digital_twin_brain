#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 19:49
# @Author  : Leijun Ye
import os.path

import numpy as np
from scipy.io import savemat

root_path = "/public/home/ssct004t/project/yeleijun/Digital_twin_brain/PD-sub401/stimulation_PALSTN_15Hz_0.1_STN_125Hz_neg6"
Imean = np.load(os.path.join(root_path, "imean_after_assim_0.npy"))
Imean = Imean.reshape([-1, Imean.shape[-1]])

for k in range(Imean.shape[0] // 10000):
    lfp = Imean[k*10000:(k+1)*10000]
    savemat(os.path.join(root_path, f"lfp_{k}.mat"), {"lfp": lfp})

lfp_stn = Imean[:, -2:]
savemat(os.path.join(root_path, "lfp_stn.mat"), {"lfp_stn": lfp_stn})