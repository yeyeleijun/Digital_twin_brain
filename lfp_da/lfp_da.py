#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/26 9:54
# @Author  : Leijun Ye

import numpy as np
import h5py

lfp_real = np.array(h5py.File("lfp_stnl_downsample_betaband.mat", "r")["lfp_stnl_downsample_betaband"]).squeeze()

