#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'
from datasketch import MinHashLSHForest, MinHash
import random

def construct_lsh(obj_dict):
    forest = MinHashLSHForest(num_perm=128)
    keys = obj_dict.keys()
    values = obj_dict.values()
    ms = []
    for i in range(len(keys)):
        temp = MinHash(num_perm=128)
        for d in values[i]:
            temp.update(d.encode('utf8'))
        ms.append(temp)
        forest.add(keys[i], temp)
    forest.index()
    return forest, keys, ms

def get_negs_by_lsh(user_dict, item_dict, num_negs):
    k_u = max(100, len(user_dict) // 50)
    k_v = max(100, len(item_dict) // 50) 
    sample_num_u = max(250, int(len(user_dict)*0.01*num_negs))
    sample_num_v = max(250, int(len(item_dict)*0.01*num_negs))
    negs_u = call_get_negs_by_lsh(k_u,sample_num_u,user_dict)
    negs_v = call_get_negs_by_lsh(k_v,sample_num_v,item_dict)
    return negs_u,negs_v

def call_get_negs_by_lsh(k, sample_num, obj_dict):
    forest, keys, ms = construct_lsh(obj_dict)
    visted = []
    negs_dict = {}
    for i in range(len(keys)):
        record = []
        if i in visted:
            continue
        visted.append(i)
        record.append(i)
        total_list = set(keys)
        sim_list = set(forest.query(ms[i], k))
        total_list = list(total_list - sim_list)
        for j in sim_list:
            total_list = set(total_list)
            ind = keys.index(j)
            if ind not in visted:
                visted.append(ind)
                record.append(ind)
            sim_list_child = set(forest.query(ms[ind], k))
            total_list = list(total_list - sim_list_child)
        total_list = random.sample(list(total_list), min(sample_num, len(total_list)))
        for j in record:
            key = keys[j]
            negs_dict[key] = total_list
    return negs_dict
