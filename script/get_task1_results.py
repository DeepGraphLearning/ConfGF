#coding: utf-8

from time import time
from tqdm import tqdm
import argparse
import numpy as np
import pickle

from confgf import utils

import multiprocessing
from functools import partial 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--input', type=str)
    parser.add_argument('--core', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of COV score')
    parser.add_argument('--FF', action='store_true', help='only for rdkit')

    args = parser.parse_args()
    print(args)

    with open(args.input, 'rb') as fin:
        data_list = pickle.load(fin)
        assert len(data_list) == 200

    bad_case = 0
    filtered_data_list = []
    for i in tqdm(range(len(data_list))):
        if '.' in data_list[i].smiles:
            bad_case += 1
            continue
        filtered_data_list.append(data_list[i])

    cnt_conf = 0
    for i in range(len(filtered_data_list)):
        cnt_conf += filtered_data_list[i].num_pos_ref.item()
    print('%d bad cases, use %d mols with total %d confs' % (bad_case, len(filtered_data_list), cnt_conf))

    pool = multiprocessing.Pool(args.core)
    func = partial(utils.evaluate_conf, useFF=args.FF, threshold=args.threshold)

    covs = []
    mats = []
    for result in tqdm(pool.imap(func, filtered_data_list), total=len(filtered_data_list)):
        covs.append(result[0])
        mats.append(result[1])
    covs = np.array(covs)
    mats = np.array(mats)

    print('Coverage Mean: %.4f | Coverage Median: %.4f | Match Mean: %.4f | Match Median: %.4f' % \
                        (covs.mean(), np.median(covs), mats.mean(), np.median(mats)))
    pool.close()
    pool.join()
