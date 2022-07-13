import numpy as np
from evaluations.patched_nn import group_cost_from_file


def calc_perwindow_cost(group_cost, tmin, keepall=False):
    res = 0.
    all_res = []
    for i in range(group_cost.shape[0] - tmin):
        cost = np.min(group_cost[i, i+tmin]) / tmin
        res += cost
        all_res.append(cost)
    if keepall:
        return res / (group_cost.shape[0] - tmin), all_res
    else:
        return res / (group_cost.shape[0] - tmin)


def perwindow_nn(src_file, tgt_files, tmin, use_pos=False, keepall=False):
    group_cost = group_cost_from_file(src_file, tgt_files, use_pos)
    return calc_perwindow_cost(group_cost, tmin, keepall)


def coverage(src_file, tgt_files, tmin=30, use_pos=False, threshold=2.0):
    res = []
    if not isinstance(tgt_files, list):
        tgt_files = [tgt_files]
    for tgt_file in tgt_files:
        group_cost = group_cost_from_file(tgt_file, src_file, use_pos)
        for i in range(group_cost.shape[0] - tmin):
            cost = np.min(group_cost[i, i+tmin]) / tmin
            res.append(1.0 if cost < threshold else 0.)

    return np.mean(np.array(res))
