import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
import pdb
import yaml

def mha_tflops(num_in, num_out, feat_dim):
    k_tflops = 2 * num_out * feat_dim * feat_dim
    v_tflops = 2 * num_out * feat_dim * feat_dim
    q_tflops = 2 * num_in * feat_dim * feat_dim
    qk_tflops = 2 * num_in * num_out * feat_dim
    va_tflops = 2 * num_in * num_out * feat_dim
    out_tflops = 2 * num_in * feat_dim * feat_dim
    total_tflops = k_tflops + v_tflops + q_tflops + qk_tflops + va_tflops + out_tflops
    return total_tflops

def mlp_tflops(num_in, feat_dim, mlp_dim):
    up_tflops = 2 * num_in * feat_dim * mlp_dim
    gate_tflops = 2 * num_in * feat_dim * mlp_dim
    down_tflops = 2 * num_in * feat_dim * mlp_dim
    total_tflops = up_tflops + gate_tflops + down_tflops
    return total_tflops

def get_tflops_fastv_lpm(tgt_layer, tgt_num, tflops_config = None):
    num_total_tks = tflops_config.num_sys_token + tflops_config.num_img_token + tflops_config.num_txt_token
    num_tgt_tks = tflops_config.num_sys_token + tgt_num + tflops_config.num_txt_token
    num_tks = [[num_total_tks, num_total_tks, num_total_tks] for _ in range(tgt_layer+1)]
    num_tks = num_tks + [[num_tgt_tks, num_tgt_tks, num_tgt_tks] for _ in range(tgt_layer+1,tflops_config.num_layer)]
    total_tflops = 0
    for idl in range(tflops_config.num_layer):
#         print(idl)
        total_tflops += mha_tflops(num_tks[idl][1], num_tks[idl][0], tflops_config.feat_dim)
        total_tflops += mlp_tflops(num_tks[idl][2], tflops_config.feat_dim, tflops_config.mlp_dim)
    return total_tflops

def get_tflops_strategy(pruning_strategy, tflops_config = None):
    pruning_strategy = set(tuple([tuple(pos) for pos in pruning_strategy]))
    token_nums = [tflops_config.num_sys_token, tflops_config.num_img_token, tflops_config.num_txt_token]
    if tflops_config is not None and 'token_nums' in tflops_config:
        token_nums = tflops_config['token_nums']
    num_total_tks = sum(token_nums)
#     print(num_layer)
    num_tks = [[num_total_tks, num_total_tks, num_total_tks] for _ in range(tflops_config.num_layer)]
    for pos in pruning_strategy:
        idl, idg, ido = pos
        cur_token_nums = token_nums
        num_tks[idl][ido] -= cur_token_nums[idg]
    total_tflops = 0
    # pdb.set_trace()

    for idl in range(tflops_config.num_layer):
        total_tflops += mha_tflops(num_tks[idl][1], num_tks[idl][0], tflops_config.feat_dim)
        total_tflops += mlp_tflops(num_tks[idl][2], tflops_config.feat_dim, tflops_config.mlp_dim)
    return total_tflops

def get_gsop_trajectory(folder_path, prune_config):
    sorting_path = os.path.join(folder_path, "final_sorting.npy")
    sorting= np.load(sorting_path, allow_pickle=True)
    cur_policy = sorting[0]
    total_flops = get_tflops_strategy([], prune_config)
    cur_flops = get_tflops_strategy(cur_policy, prune_config)
    cur_flops_ratio = cur_flops/total_flops
    gsop_trajectory = {}
    gsop_trajectory[0] = (cur_flops_ratio, cur_policy)
    rk_layer = prune_config.remove_key_layer
    if rk_layer is not None:
        for step in range(1, len(sorting)):
            new_ops = []
            for oid in range(len(sorting[step])):
                idl, idg, ido = sorting[step][oid]
                if idg == 1 and idl <= rk_layer[ido]:
                    print("removing key op:", sorting[step][oid])
                else:
                    new_ops.append(sorting[step][oid])
            sorting[step] = new_ops
    for step in range(1, len(sorting)):
        cur_policy += sorting[step]
        print(sorting[step])
        log_policy = list(set(tuple([tuple(pos) for pos in cur_policy])))
        cur_flops = get_tflops_strategy(log_policy, prune_config)
        cur_flops_ratio = cur_flops/total_flops
        gsop_trajectory[step] = (cur_flops_ratio, log_policy)
    return gsop_trajectory