# coding=utf-8
from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add


class MeanReadout(nn.Module):
    """Mean readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).

        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_mean(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output


class SumReadout(nn.Module):
    """Sum readout operator over graphs with variadic sizes."""

    def forward(self, data, input):
        """
        Perform readout over the graph(s).

        Parameters:
            data (torch_geometric.data.Data): batched graph
            input (Tensor): node representations

        Returns:
            Tensor: graph representations
        """
        output = scatter_add(input, data.batch, dim=0, dim_size=data.num_graphs)
        return output



class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.

    Note there is no activation or dropout in the last layer.

    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


