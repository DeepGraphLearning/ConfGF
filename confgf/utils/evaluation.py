import numpy as np
from tqdm.auto import tqdm

import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

from confgf import utils


def get_rmsd_confusion_matrix(data: Data, useFF=False):
    data.pos_ref = data.pos_ref.view(-1, data.num_nodes, 3)
    data.pos_gen = data.pos_gen.view(-1, data.num_nodes, 3)
    num_gen = data.pos_gen.size(0)
    num_ref = data.pos_ref.size(0)

    assert num_gen == data.num_pos_gen.item()
    assert num_ref == data.num_pos_ref.item()

    rmsd_confusion_mat = -1 * np.ones([num_ref, num_gen],dtype=np.float)
    
    for i in range(num_gen):
        gen_mol = utils.set_rdmol_positions(data.rdmol, data.pos_gen[i])
        if useFF:
            #print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(gen_mol)
        for j in range(num_ref):
            ref_mol = utils.set_rdmol_positions(data.rdmol, data.pos_ref[j])
            
            rmsd_confusion_mat[j,i] = utils.GetBestRMSD(gen_mol, ref_mol)

    return rmsd_confusion_mat
    

def evaluate_conf(data: Data, useFF=False, threshold=0.5):
    rmsd_confusion_mat = get_rmsd_confusion_matrix(data, useFF=useFF)
    rmsd_ref_min = rmsd_confusion_mat.min(-1)
    return (rmsd_ref_min<=threshold).mean(), rmsd_ref_min.mean()


def evaluate_distance(data: Data, ignore_H=True):
    data.pos_ref = data.pos_ref.view(-1, data.num_nodes, 3) # (N, num_node, 3)
    data.pos_gen = data.pos_gen.view(-1, data.num_nodes, 3) # (M, num_node, 3)
    num_ref = data.pos_ref.size(0) # N
    num_gen = data.pos_gen.size(0) # M
    assert num_gen == data.num_pos_gen.item()
    assert num_ref == data.num_pos_ref.item()
    smiles = data.smiles

    edge_index = data.edge_index
    atom_type = data.atom_type

    # compute generated length and ref length 
    ref_lengths = (data.pos_ref[:, edge_index[0]] - data.pos_ref[:, edge_index[1]]).norm(dim=-1) # (N, num_edge)
    gen_lengths = (data.pos_gen[:, edge_index[0]] - data.pos_gen[:, edge_index[1]]).norm(dim=-1) # (M, num_edge)
    # print(ref_lengths.size(), gen_lengths.size())
    #print(ref_lengths.size())
    #print(gen_lengths.size())

    stats_single = []
    first = 1
    for i, (row, col) in enumerate(tqdm(edge_index.t())):
        if row >= col: 
            continue
        if ignore_H and 1 in (atom_type[row].item(), atom_type[col].item()): 
            continue
        gen_l = gen_lengths[:, i]
        ref_l = ref_lengths[:, i]
        if first:
            print(gen_l.size(), ref_l.size())
            first = 0
        mmd = compute_mmd(gen_l.view(-1, 1).cuda(), ref_l.view(-1, 1).cuda()).item()
        stats_single.append({
            'edge_id': i,
            'elems': '%s - %s' % (utils.get_atom_symbol(atom_type[row].item()), utils.get_atom_symbol(atom_type[col].item())),
            'nodes': (row.item(), col.item()),
            'gen_lengths': gen_l.cpu(),
            'ref_lengths': ref_l.cpu(),
            'mmd': mmd
        })

    first = 1
    stats_pair = []
    for i, (row_i, col_i) in enumerate(tqdm(edge_index.t())):
        if row_i >= col_i: 
            continue
        if ignore_H and 1 in (atom_type[row_i].item(), atom_type[col_i].item()): 
            continue
        for j, (row_j, col_j) in enumerate(edge_index.t()):
            if (row_i >= row_j) or (row_j >= col_j): 
                continue
            if ignore_H and 1 in (atom_type[row_j].item(), atom_type[col_j].item()): 
                continue

            gen_L = gen_lengths[:, (i,j)]   # (N, 2)
            ref_L = ref_lengths[:, (i,j)]   # (M, 2)
            if first:
                # print(gen_L.size(), ref_L.size())
                first = 0
            mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

            stats_pair.append({
                'edge_id': (i, j),
                'elems': (
                    '%s - %s' % (utils.get_atom_symbol(atom_type[row_i].item()), utils.get_atom_symbol(atom_type[col_i].item())), 
                    '%s - %s' % (utils.get_atom_symbol(atom_type[row_j].item()), utils.get_atom_symbol(atom_type[col_j].item())),                         
                ),
                'nodes': (
                    (row_i.item(), col_i.item()),
                    (row_j.item(), col_j.item()),
                ),
                'gen_lengths': gen_L.cpu(),
                'ref_lengths': ref_L.cpu(),
                'mmd': mmd
            })    

    edge_filter = edge_index[0] < edge_index[1]
    if ignore_H:
        for i, (row, col) in enumerate(edge_index.t()): 
            if 1 in (atom_type[row].item(), atom_type[col].item()):
                edge_filter[i] = False

    gen_L = gen_lengths[:, edge_filter]    # (N, Ef)
    ref_L = ref_lengths[:, edge_filter]    # (M, Ef)
    # print(gen_L.size(), ref_L.size())
    mmd = compute_mmd(gen_L.cuda(), ref_L.cuda()).item()

    stats_all = {
        'gen_lengths': gen_L.cpu(),
        'ref_lengths': ref_L.cpu(),
        'mmd': mmd
    }
    return stats_single, stats_pair, stats_all

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Params:
	    source: n * len(x)
	    target: m * len(y)
	Return:
		sum(kernel_val): Sum of various kernel matrices
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)#/len(kernel_val)
 
def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    Params:
	    source: (N, D)
	    target: (M, D)
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

    return loss


"""
Another implementation:
    https://github.com/martinepalazzo/kernel_methods/blob/master/kernel_methods.py
"""