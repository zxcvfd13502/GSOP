import copy
import pdb
import os
import re
import numpy as np
import time
import yaml

def iterative_eval_prune_acc(search_config, eval_configs):
    gsop_sorting = []
    search_config.cur_ll_poss = None
    cur_ll_poss, cur_policy, ll_records = get_lossless(search_config, eval_configs)
    mid_res_dir = search_config.mid_out_dir
    if not os.path.exists(mid_res_dir):
        os.makedirs(mid_res_dir, exist_ok=True)
    cfg_path = os.path.join(mid_res_dir, 'configs.yaml')
    sorting_path = os.path.join(mid_res_dir, 'final_sorting.npy')
    with open(cfg_path, 'w') as f:
        yaml.dump(search_config, f, allow_unicode=True, sort_keys=False)
    gsop_sorting.append(cur_policy)
    np.save(sorting_path, gsop_sorting)
    poos = get_single_poos(search_config, eval_configs, cur_policy)

    poo_mr_path = os.path.join(mid_res_dir, 'poo_0.npy')
    bsl_val = search_config.bsl_val
    poo_results = {"poos": poos, "bsl_val":bsl_val,"search_config": search_config}
    np.save(poo_mr_path, poo_results)

    greedy_steps = search_config.greedy_steps
    acc_threshold = search_config.acc_threshold_ratio * bsl_val
    acc_stride = acc_threshold / greedy_steps
    records = []
    ids = 0
    idp = 1
    last_policy = copy.deepcopy(cur_policy)
    while idp <= greedy_steps:
        ids += 1
        cur_policy, all_poo_flag, poos = get_next_policy(poos, cur_policy, search_config)
        eval_acc = eval_sparse_lm(cur_policy, search_config, eval_configs)
        print("step ", ids, "acc:", eval_acc)
        if all_poo_flag:
            break
        if (bsl_val - eval_acc > idp * acc_stride):
            cur_policy = last_policy
            poos = get_single_poos(search_config, eval_configs, cur_policy)
            idp += 1
            poo_mr_path = os.path.join(mid_res_dir, 'poo_{}.npy'.format(ids))
            poo_results = {"poos": poos, "bsl_val":bsl_val, 'acc_threshold':ids * acc_stride, "search_config": search_config}
            np.save(poo_mr_path, poo_results)
        else:
            cur_op = list(set(cur_policy) - set(last_policy))
            gsop_sorting.append(cur_op)
            try:
                np.save(sorting_path, np.array(gsop_sorting, dtype=object))
            except:
                pdb.set_trace()
            
            gp_mr_path = os.path.join(mid_res_dir, 'gp_step{}.npy'.format(ids))
            gp_results = {"cur_policy": cur_policy, "all_poo_flag": all_poo_flag, "eval_acc":eval_acc, "search_config": search_config}
            np.save(gp_mr_path, gp_results)

            last_policy = copy.deepcopy(cur_policy)

def get_lossless(search_config, eval_configs):
    ll_groups = search_config.prune_groups
    bsl_val = search_config.bsl_val
    num_layers = search_config.num_layer
    cur_ll_poss = search_config.cur_ll_poss
    if cur_ll_poss is None:
        cur_ll_poss = [-1 for _ in range(len(search_config.token_groups) * 3)]
    cur_policy = []
    ll_records = {}
    cur_thresh = 0

    for idg in ll_groups:
        if idg == 1 and search_config.remove_key_layer[0] >= 0:
            cur_start_layer = search_config.remove_key_layer[0]
            print("group 1, remove key layer", search_config.remove_key_layer[0], "cur_start_layer:", cur_start_layer)
            # pdb.set_trace()
        else:
            cur_start_layer = None
        ll_ops = copy.deepcopy(search_config.prune_ops)
        if idg == 1 and search_config.remove_key_out:
            ll_ops.pop()
            print("group 1, remove key out", ll_ops)
            # pdb.set_trace()
        while len(ll_ops) > 0:
            llids = []
            for ido in ll_ops:
                llids.append(idg * 3 + ido)
            print("llids:", llids)
            cur_ll_poss, cur_policy, ll_records = binary_search(cur_ll_poss, llids, num_layers, bsl_val, cur_thresh, cur_policy, search_config, eval_configs, ll_records, start_layer = cur_start_layer)
            ll_ops.pop()
    return cur_ll_poss, cur_policy, ll_records

def binary_search(cur_ll_poss, gids, num_layers, bsl_val, acc_threshold, base_policy, search_config, eval_configs, records = {}, start_layer = None):
    if start_layer is None:
        start_layer = 0
    sys_mip_flag = True
    for gid in gids:
        sys_mip_flag = gid in [1,2]
    if sys_mip_flag:
        cur_ll_poss = update_ll_poss(cur_ll_poss, gids, 0)
        cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
        cur_acc = eval_sparse_lm(cur_policy, search_config, eval_configs)
        records[tuple(cur_ll_poss)] = cur_acc
        return cur_ll_poss, cur_policy, records
    if search_config.ll_half:
        left = num_layers // 2 - 1
    else:
        left = start_layer
    if cur_ll_poss[gids[0]] >= 0 and cur_ll_poss[gids[0]] <= left:
        return cur_ll_poss, base_policy, records
    try:
        right = cur_ll_poss[gids[0]] if cur_ll_poss[gids[0]] > 0 else num_layers - 1
    except:
        pdb.set_trace()
    cur_llid = cur_ll_poss[gids[0]]
    
    cur_ll_poss = update_ll_poss(cur_ll_poss, gids, right)
    cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
    if tuple(cur_ll_poss) not in records:
        cur_acc = eval_sparse_lm(cur_policy, search_config, eval_configs)
    else:
        cur_acc = records[tuple(cur_ll_poss)]
    acc_diff = bsl_val - cur_acc
    records[tuple(cur_ll_poss)] = cur_acc
    
    if acc_diff > acc_threshold:
        cur_ll_poss = update_ll_poss(cur_ll_poss, gids, cur_llid)
        cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
        return cur_ll_poss, cur_policy, records
    
    cur_ll_poss = update_ll_poss(cur_ll_poss, gids, left)
    cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
    if tuple(cur_ll_poss) not in records:
        cur_acc = eval_sparse_lm(cur_policy, search_config, eval_configs)
    else:
        cur_acc = records[tuple(cur_ll_poss)]
    acc_diff = bsl_val - cur_acc
    records[tuple(cur_ll_poss)] = cur_acc
    if acc_diff <= acc_threshold:
        cur_ll_poss = update_ll_poss(cur_ll_poss, gids, left)
        cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
        return cur_ll_poss, cur_policy, records

    cur_llid_traces = []
    while left <= right:
        mid = (left + right) // 2
        cur_ll_poss = update_ll_poss(cur_ll_poss, gids, mid)
        cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
        if tuple(cur_ll_poss) not in records:
            cur_acc = eval_sparse_lm(cur_policy, search_config, eval_configs)
        else:
            cur_acc = records[tuple(cur_ll_poss)]
        
        records[tuple(cur_ll_poss)] = cur_acc
        acc_diff = bsl_val - cur_acc
        if acc_diff <= acc_threshold:
            if cur_ll_poss[gids[0]] < cur_llid or cur_llid == -1:
                cur_llid = cur_ll_poss[gids[0]]
            right = mid - 1
        else:
            left = mid + 1
        cur_llid_traces.append((cur_llid, cur_acc))
    cur_ll_poss = update_ll_poss(cur_ll_poss, gids, cur_llid)
    cur_policy = update_policy(base_policy, loss_less = cur_ll_poss, num_layers = num_layers)
    print(cur_llid_traces)
    return cur_ll_poss, cur_policy, records

def update_ll_poss(ll_poss, gids, cur_val):
    for gid in gids:
        ll_poss[gid] = cur_val
    return ll_poss

def eval_sparse_lm(cur_policy, search_config, eval_configs):
    eval_configs['lm'].model.clean_pruning()
    eval_configs['lm'].model.apply_pruning_policy(cur_policy, search_config)
    evaluate = eval_configs.pop('eval_func')
    cur_task = search_config.greedy_task
    metric = search_config.greedy_metric
    results = evaluate(**eval_configs)
    eval_configs['eval_func'] = evaluate
    if isinstance(results, dict):
        return results['results'][cur_task][metric]
    else:
        return results

def get_next_policy(poos, base_policy, search_config):
    # pdb.set_trace()
    cur_policy = base_policy
    sorted_candidates = sorted(poos, key=poos.get)
    if len(sorted_candidates) == 0:
        return cur_policy, True, poos
    next_candidate = sorted_candidates.pop()
    while next_candidate in cur_policy:
        next_candidate = sorted_candidates.pop()
    cur_policy.append(next_candidate)
    print("checking the constraint 1: group")
    cur_policy = group_constraint_process(cur_policy)
    print("checking the constraint 2: depth")
    num_layers = search_config.num_layer
    cur_policy = depth_constraint_process(cur_policy, num_layers)
    for prune_op in cur_policy:
        if prune_op in poos:
            poos.pop(prune_op)
    return cur_policy, False, poos

def get_single_poos(search_config, eval_configs, base_policy):
    poo_groups = search_config.prune_groups
    poo_ops = search_config.prune_ops
    # prune_edits = (search_config.remove_key_out, search_config.remove_key_layer)
    num_layers = search_config.num_layer
    poo_candidates = get_candidates(poo_groups, poo_ops, base_policy, num_layers, start_layer = 0, remove_key_layer = search_config.remove_key_layer)
    cur_poo_dict = {}
    for cur_poo in poo_candidates:
        cur_policy = update_policy(base_policy, cur_poo)
        if len(search_config.token_groups) > 3:
            print("checking the constraint 1: group")
            cur_policy = group_constraint_process(cur_policy)
        print("checking the constraint 2: depth")
        cur_policy = depth_constraint_process(cur_policy, num_layers)
        acc = eval_sparse_lm(cur_policy, search_config, eval_configs)
        cur_poo_dict[cur_poo] = acc
    # good_poo_dict, bad_poss = update_good_bad_poos(cur_poo_dict, bad_poss, search_config.bsl_val, search_config.acc_threshold_ratio)
    return cur_poo_dict

def get_candidates(poo_groups, poo_ops, base_policy, num_layers = 32, start_layer = 0, remove_key_layer = None):
    poo_candidates = []
    # remove_key_out, remove_key_layer = prune_edits
    for idl in range(start_layer, num_layers):
        for gid in poo_groups:
            for ido in poo_ops:
                if (idl, gid, ido) not in base_policy:
                    if remove_key_layer is not None and gid == 1 and idl < remove_key_layer[ido]:
                        print("removing key layer for group 1 in candidates", (idl, gid, ido))
                        continue
                    if (idl, gid, ido) in [(0,1,0),(0,2,0)]:
                        print("removing first layer out op for the 2 input id bug", (idl, gid, ido))
                        continue
                    poo_candidates.append((idl, gid, ido))
    return poo_candidates



def update_policy(base_policy, single_pos = None, loss_less = None, num_layers = 32):
    cur_policy = copy.deepcopy(base_policy)
    if single_pos is not None:
        cur_policy.append(single_pos)
    if loss_less is not None:
        num_group = len(loss_less) // 3
        for idg in range(num_group):
            for ido in range(3):
                llid = idg * 3 + ido
                if loss_less[llid] >= 0:
                    cur_policy += [(idl, idg, ido) for idl in range(loss_less[llid], num_layers)]
    cur_policy = list(set(cur_policy))
    cur_policy.sort()
    return cur_policy

def group_constraint_process(pruning_strategy):
    for pos in pruning_strategy:
        idl, idg, ido = pos
        if idg == 1 and (idl, 2,ido) not in pruning_strategy:
            print("adding group 2 op", (idl, 2,ido))
            pruning_strategy.append((idl, 2,ido))
    return pruning_strategy

def depth_constraint_process(pruning_strategy, layer_num=32):
    pruned_layers_by_group = {}
    org_policy_set = set(pruning_strategy)
    
    for pos in pruning_strategy:
        idl, idg, ido = pos
        if ido == 0:
            if idg not in pruned_layers_by_group:
                pruned_layers_by_group[idg] = set()
            pruned_layers_by_group[idg].add(idl)
    group_out_depth = {}
    for idg, pruned_layers in pruned_layers_by_group.items():
        max_depth = -1
        for layer in range(layer_num - 1, -1, -1):
            if layer in pruned_layers:
                continue
            else:
                max_depth = layer
                break
        group_out_depth[idg] = max_depth
    new_pruning_strategy = list(pruning_strategy)
    
    for idg, max_depth in group_out_depth.items():
        if max_depth >= 0:
            for layer in range(max_depth + 1, layer_num):
                new_pruning_strategy.append((layer, idg, 1))
                new_pruning_strategy.append((layer, idg, 2))
    new_policy_set = set(new_pruning_strategy)
    new_pruning_strategy= list(new_policy_set)
    print("for depth constraint, we add", new_policy_set - org_policy_set)
    
    return new_pruning_strategy