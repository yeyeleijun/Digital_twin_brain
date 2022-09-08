# -*- coding: utf-8 -*- 
# @Time : 2022/8/7 13:15 
# @Author : lepold
# @File : default_param.py


def regular_dict(**kwargs):
    return kwargs


bold_params = regular_dict(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
v_th = -50
