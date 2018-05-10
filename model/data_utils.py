#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'CLH'

import random
from io import open
import os

class DataUtils(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def rename(self, datafile):
        """
        Distinguish two types of node and rename
        """
        with open(os.path.join(self.model_path,"_ratings.dat"), "w") as fw:
            with open(datafile, "r", encoding="UTF-8") as fin:
                line = fin.readline()
                while line:
                    user, item, rating = line.strip().split("\t")
                    fw.write("u" + user + "\t" + "i" + item + "\t" + rating + "\n")
                    line = fin.readline()

    def split_data(self, percent):
        """
        split data
        :param percent:
        :return:
        """
        test_user,test_item,test_rate,rating = set(), set(), {},{}
        with open(os.path.join(self.model_path, "ratings.dat"), "r") as fin, open(os.path.join(self.model_path, "ratings_train.dat"),"w") as ftrain, open(os.path.join(self.model_path,"ratings_test.dat"), "w") as ftest:
            for line in fin.readlines():
                user, item, rate = line.strip().split("\t")
                if rating.get(user) is None:
                    rating[user] = {}
                rating[user][item] = rate
            for u in rating.keys():
                item_list = rating[u].keys()
                sample_list = random.sample(item_list, int(len(item_list) * percent))
                for item in item_list:
                    if item in sample_list:
                        ftrain.write(u + "\t" + item + "\t" + rating[u][item] + "\n")
                    else:
                        if test_rate.get(u) is None:
                            test_rate[u] = {}
                        test_rate[u][item] = float(rating[u][item])
                        test_user.add(u)
                        test_item.add(item)
                        ftest.write(u + "\t" + item + "\t" + rating[u][item] + "\n")
        return test_user, test_item, test_rate

    def read_data(self,filename=None):
        if filename is None:
            filename = os.path.join(self.model_path,"ratings_test.dat")
        users,items,rates = set(), set(), {}
        with open(filename, "r", encoding="UTF-8") as fin:
            line = fin.readline()
            while line:
                user, item, rate = line.strip().split()
                if rates.get(user) is None:
                    rates[user] = {}
                rates[user][item] = float(rate)
                users.add(user)
                items.add(item)
                line = fin.readline()
        return users, items, rates



