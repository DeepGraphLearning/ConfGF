import os
import time
import argparse
import torch
import pickle

import copy
import numpy as np
from tqdm import tqdm

import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem

from confgf import utils, dataset

import multiprocessing
from functools import partial 

def generate_conformers(mol, num_confs):
    mol = copy.deepcopy(mol)
    mol.RemoveAllConformers()
    assert mol.GetNumConformers() == 0

    AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=num_confs, 
        maxAttempts=0,
        ignoreSmoothingFailures=True,
    )
    if mol.GetNumConformers() != num_confs:
        print('Warning: Failure cases occured, generated: %d , expected: %d.' % (mol.GetNumConformers(), num_confs, ))

    return mol



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=50)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--core', type=int, default=6)
    parser.add_argument('--FF', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    print(args)



    with open(args.input, 'rb') as f:
        data_raw = pickle.load(f)
        if 'pos_ref' in data_raw[0]:
            data_list = data_raw
        else:
            data_list = dataset.GEOMDataset_PackedConf(data_raw)


    generated_data_list = []
    for i in tqdm(range(args.start_idx, len(data_list))):
        return_data = copy.deepcopy(data_list[i])

        if args.num_samples > 0:
            num_confs = args.num_samples
        else:
            num_confs = -args.num_samples*return_data.num_pos_ref.item()
        mol = generate_conformers(return_data.rdmol, num_confs=num_confs)
        num_pos_gen = mol.GetNumConformers()
        all_pos = []

        if num_pos_gen == 0:
            continue

        for j in range(num_pos_gen):
            all_pos.append(torch.tensor(mol.GetConformer(j).GetPositions(), dtype=torch.float32))

        return_data.pos_gen = torch.cat(all_pos, 0) # (num_pos_gen * num_node, 3)
        return_data.num_pos_gen = torch.tensor([len(all_pos)], dtype=torch.long)
        generated_data_list.append(return_data)  

    with open(args.output, "wb") as fout:
        pickle.dump(generated_data_list, fout)
    print('save generated conf to %s done!' % args.output)
    


    
    if args.eval:
        print('start getting results!')

        with open(args.output, 'rb') as fin:
            data_list = pickle.load(fin)
        bad_case = 0


        filtered_data_list = []
        for i in tqdm(range(len(data_list))):
            if '.' in data_list[i].smiles:
                bad_case += 1
                continue
            filtered_data_list.append(data_list[i])

        cnt_conf = 0
        for i in range(len(filtered_data_list)):
            cnt_conf += filtered_data_list[i].num_pos_ref
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
                

