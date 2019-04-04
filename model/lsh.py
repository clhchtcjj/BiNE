#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'
from datasketch import MinHashLSHForest, MinHash, MinHashLSH
import random

def construct_lsh(obj_dict):
    lsh_0 = MinHashLSH(threshold=0, num_perm=128,params=None)
    lsh_5 = MinHashLSH(threshold=0.6, num_perm=128,params=None)
    # forest = MinHashLSHForest(num_perm=128)
    keys = list(obj_dict.keys())
    values = list(obj_dict.values())
    ms = []
    for i in range(len(keys)):
        temp = MinHash(num_perm=128)
        for d in values[i]:
            temp.update(d.encode('utf8'))
        ms.append(temp)
        lsh_0.insert(keys[i], temp)
        lsh_5.insert(keys[i], temp)
    return lsh_0,lsh_5, keys, ms

def get_negs_by_lsh(user_dict, item_dict, num_negs):
    sample_num_u = max(300, int(len(user_dict)*0.01*num_negs))
    sample_num_v = max(300, int(len(item_dict)*0.01*num_negs))
    negs_u = call_get_negs_by_lsh(sample_num_u,user_dict)
    negs_v = call_get_negs_by_lsh(sample_num_v,item_dict)
    return negs_u,negs_v

def call_get_negs_by_lsh(sample_num, obj_dict):
    lsh_0,lsh_5, keys, ms = construct_lsh(obj_dict)
    visited = []
    negs_dict = {}
    for i in range(len(keys)):
        record = []
        if i in visited:
            continue
        visited.append(i)
        record.append(i)
        total_list = set(keys)
        sim_list = set(lsh_0.query(ms[i]))
        high_sim_list = set(lsh_5.query(ms[i]))
        total_list = list(total_list - sim_list)
        for j in high_sim_list:
            total_list = set(total_list)
            ind = keys.index(j)
            if ind not in visited:
                visited.append(ind)
                record.append(ind)
            sim_list_child = set(lsh_0.query(ms[ind]))
            total_list = list(total_list - sim_list_child)
        total_list = random.sample(list(total_list), min(sample_num, len(total_list)))
        for j in record:
            key = keys[j]
            negs_dict[key] = total_list
    return negs_dict
