import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from confgf import utils, layers

class DistanceScoreMatch(torch.nn.Module):

    def __init__(self, config):
        super(DistanceScoreMatch, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        self.output_mlp = layers.MultiLayerPerceptron(2 * self.hidden_dim, \
                                [self.hidden_dim, self.hidden_dim // 2, 1], activation=self.config.model.mlp_act)

        self.model = layers.GraphIsomorphismNetwork(hidden_dim=self.hidden_dim, \
                                 num_convs=self.config.model.num_convs, \
                                 activation=self.config.model.gnn_act, \
                                 readout="sum", short_cut=self.config.model.short_cut, \
                                 concat_hidden=self.config.model.concat_hidden)
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

        """
        Techniques from "Improved Techniques for Training Score-Based Generative Models"
        1. Choose sigma1 to be as large as the maximum Euclidean distance between all pairs of training data points.
        2. Choose sigmas as a geometric progression with common ratio gamma, where a specific equation of CDF is satisfied.
        3. Parameterize the Noise Conditional Score Networks with f_theta_sigma(x) =  f_theta(x) / sigma
        """

    
    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order+1):
                adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order+1):
                order_mat += (adj_mats[i] - adj_mats[i-1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data   
      

    @torch.no_grad()
    def get_score(self, data: Data, d, sigma):
        """
        Input:
            data: torch geometric batched data object
            d: edge distance, shape (num_edge, 1)
            sigma: noise level, tensor (,)
        Output:
            log-likelihood gradient of distance, tensor with shape (num_edge, 1)         
        """
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)      
        d_emb = self.input_mlp(d) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]] # (num_edge, hidden)
        distance_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature) # (num_edge, 1)
        scores = scores * (1. / sigma) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)
        return scores

    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        self.device = self.sigmas.device
        data = self.extend_graph(data, self.order)
        data = self.get_distance(data)

        assert data.edge_index.size(1) == data.edge_length.size(0)
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]        

        # sample noise level
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)
        used_sigmas = used_sigmas[edge2graph].unsqueeze(-1) # (num_edge, 1)

        # perturb
        d = data.edge_length # (num_edge, 1)

        if self.noise_type == 'symmetry':
            num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.device), node2graph) # (num_graph)
            num_cum_nodes = num_nodes.cumsum(0) # (num_graph)
            node_offset = num_cum_nodes - num_nodes # (num_graph)
            edge_offset = node_offset[edge2graph] # (num_edge)

            num_nodes_square = num_nodes**2 # (num_graph)
            num_nodes_square_cumsum = num_nodes_square.cumsum(-1) # (num_graph)
            edge_start = num_nodes_square_cumsum - num_nodes_square # (num_graph)
            edge_start = edge_start[edge2graph]

            all_len = num_nodes_square_cumsum[-1]

            node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
            #node_in, node_out = node_index.t()
            node_large = node_index.max(dim=-1)[0]
            node_small = node_index.min(dim=-1)[0]
            undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start

            symm_noise = torch.cuda.FloatTensor(all_len, device=self.device).normal_()
            d_noise = symm_noise[undirected_edge_id].unsqueeze(-1) # (num_edge, 1)

        elif self.noise_type == 'rand':
            d_noise = torch.randn_like(d)
        else:
            raise NotImplementedError('noise type must in [distance_symm, distance_rand]')
        assert d_noise.shape == d.shape
        perturbed_d = d + d_noise * used_sigmas   
        #perturbed_d = torch.clamp(perturbed_d, min=0.1, max=float('inf'))    # distances must be greater than 0



        # get target, origin_d minus perturbed_d
        target = -1 / (used_sigmas ** 2) * (perturbed_d - d) # (num_edge, 1)

        # estimate scores
        node_attr = self.node_emb(data.atom_type) # (num_node, hidden)
        edge_attr = self.edge_emb(data.edge_type) # (num_edge, hidden)
        d_emb = self.input_mlp(perturbed_d) # (num_edge, hidden)
        edge_attr = d_emb * edge_attr # (num_edge, hidden)

        output = self.model(data, node_attr, edge_attr)
        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]] # (num_edge, hidden)

        distance_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
        scores = self.output_mlp(distance_feature) # (num_edge, 1)
        scores = scores * (1. / used_sigmas) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        target = target.view(-1) # (num_edge)
        scores = scores.view(-1) # (num_edge)
        loss =  0.5 * ((scores - target) ** 2) * (used_sigmas.squeeze(-1) ** self.anneal_power) # (num_edge)
        loss = scatter_add(loss, edge2graph) # (num_graph)
        return loss
