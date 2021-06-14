#coding: utf-8

from time import time
from tqdm import tqdm
import argparse
import numpy as np
import random
import math
import os
import sys
import json
import pickle
import yaml
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from confgf import utils, dataset

import multiprocessing
from functools import partial 





if __name__ == '__main__':


    multiprocessing.set_start_method('spawn', force=True)
   
    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--input', type=str)
    parser.add_argument('--core', type=int, default=1, help='path of dataset')
    args = parser.parse_args()
    print(args)

    with open(args.input, 'rb') as fin:
        data_list = pickle.load(fin)
    print(len(data_list))


    bad_case = 0

    filtered_data_list = []
    for i in tqdm(range(len(data_list))):
        if '.' in data_list[i].smiles:
            bad_case += 1
            continue
        # filter corrupted mols with #confs less than 1000
        if data_list[i].num_pos_ref < 1000:
            bad_case += 1
            continue
        filtered_data_list.append(data_list[i])

    cnt_conf = 0
    for i in range(len(filtered_data_list)):
        cnt_conf += filtered_data_list[i].num_pos_ref
    print('%d bad cases, use %d mols with total %d confs' % (bad_case, len(filtered_data_list), cnt_conf))

    pool = multiprocessing.Pool(args.core)
    func = partial(utils.evaluate_distance, ignore_H=True)


    s_mmd_all = []
    p_mmd_all = []
    a_mmd_all = []
    

    for result in tqdm(pool.imap(func, filtered_data_list), total=len(filtered_data_list)):
        stats_single, stats_pair, stats_all = result
        s_mmd_all += [e['mmd'] for e in stats_single]
        p_mmd_all += [e['mmd'] for e in stats_pair]
        a_mmd_all.append(stats_all['mmd'])

    print('SingleDist | Mean: %.4f | Median: %.4f | Min: %.4f | Max: %.4f' % \
                            (np.mean(s_mmd_all), np.median(s_mmd_all), np.min(s_mmd_all), np.max(s_mmd_all)))
    print('PairDist | Mean: %.4f | Median: %.4f | Min: %.4f | Max: %.4f' % \
                            (np.mean(p_mmd_all), np.median(p_mmd_all), np.min(p_mmd_all), np.max(p_mmd_all)))
    print('AllDist | Mean: %.4f | Median: %.4f | Min: %.4f | Max: %.4f' % \
                            (np.mean(a_mmd_all), np.median(a_mmd_all), np.min(a_mmd_all), np.max(a_mmd_all)))                                                        


    pool.close()
    pool.join()
