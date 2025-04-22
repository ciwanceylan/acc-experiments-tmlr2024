from typing import Union, List
import torch
from torch import nn

from .utils import create_activation
from .layers import OneDirSAGEConv


class ACCGraphSAGE(nn.Module):
    def __init__(self,
                 directed: bool,
                 in_dim: int,
                 max_dim: int,
                 num_layers,
                 activation,
                 feat_drop,
                 norm=None,
                 encoding=False,
                 bias: bool = False,
                 min_add_dim: int = 2
                 ):
        super(ACCGraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.norm = norm if encoding else None
        self.is_encoding = encoding
        self.in_dim = in_dim
        self.max_dim = max_dim
        self.feat_drop = feat_drop
        self.bias = bias

        out_dim, init_dim, layer_dims = get_emb_dims(
            in_dim=in_dim,
            max_dim=max_dim,
            num_steps=num_layers,
            min_add_dim=min_add_dim,
            directed=directed
        )
        self.out_dim = out_dim

        self.init_layer = nn.Sequential(
                    nn.Dropout(self.feat_drop),
                    nn.Linear(in_features=self.in_dim, out_features=init_dim, bias=self.bias),
                    create_activation(self.activation))
        self.acc_layers = nn.ModuleList()
        if directed:
            self.init_directed_model(DirACCConv, init_dim=init_dim, layer_dims=layer_dims)
        else:
            self.init_directed_model(ACCConv, init_dim=init_dim, layer_dims=layer_dims)

    def init_directed_model(self, conv_model: Union['DirACCConv', 'ACCConv'], init_dim: int, layer_dims: List[int]):

        if self.num_layers > 0:
            self.acc_layers.append(conv_model(
                init_dim, layer_dims[0], aggregator_type='mean', feat_drop=self.feat_drop,
                activation=create_activation(self.activation), norm=self.norm))

            for l in range(1, self.num_layers):
                self.acc_layers.append(conv_model(
                    layer_dims[l - 1], layer_dims[l], aggregator_type='mean', feat_drop=self.feat_drop,
                    activation=create_activation(self.activation), norm=self.norm))

    def forward(self, g, inputs):
        h = self.init_layer(inputs)
        z = [h]

        for l, acc_layer in enumerate(self.acc_layers):
            h = acc_layer(g, h)
            z.append(h)

        z = torch.cat(z, dim=1)
        return z


class DirACCConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 *,
                 feat_drop=0.,
                 aggregator_type='mean',
                 activation=None,
                 bias=False,
                 norm=None):
        super(DirACCConv, self).__init__()

        self.fc_neigh = nn.Linear(in_features=2 * in_feats, out_features=out_feats, bias=bias)
        self.activation = activation
        self.norm = norm

        self.fwd_sage_conv = OneDirSAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            feat_drop=feat_drop,
            aggregator_type=aggregator_type
        )
        self.bwd_sage_conv = OneDirSAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            feat_drop=feat_drop,
            aggregator_type=aggregator_type
        )

    def forward(self, graph, feat):
        _, fwd_neigh = self.fwd_sage_conv(graph, feat)
        _, bwd_neigh = self.bwd_sage_conv(graph.reverse(), feat)
        # torch.cat((h_self, fwd_neigh, bwd_neigh), dim=0)
        neigh_rst = self.fc_neigh(torch.cat((fwd_neigh, bwd_neigh), dim=1))
        if self.activation:
            neigh_rst = self.activation(neigh_rst)
        if self.norm is not None:
            neigh_rst = self.norm(neigh_rst)
        return neigh_rst


class ACCConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 *,
                 feat_drop=0.,
                 aggregator_type='mean',
                 activation=None,
                 bias=False,
                 norm=None):
        super(ACCConv, self).__init__()

        self.fc_neigh = nn.Linear(in_features=in_feats, out_features=out_feats, bias=bias)
        self.activation = activation
        self.norm = norm

        self.sage_conv = OneDirSAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            feat_drop=feat_drop,
            aggregator_type=aggregator_type
        )

    def forward(self, graph, feat):
        _, neigh_rst = self.sage_conv(graph, feat)
        if self.activation:
            neigh_rst = self.activation(neigh_rst)
        if self.norm is not None:
            neigh_rst = self.norm(neigh_rst)
        return neigh_rst


def get_emb_dims(in_dim: int, max_dim: int, num_steps: int, min_add_dim: int, directed: bool):
    # if emb_dim < (num_steps + 1) * min_add_dim:
    #     raise ValueError(
    #         f"Embedding dimension cannot be smaller than num_steps * min_add_dim. "
    #         f"Currently: emb_dim='{emb_dim}', num_steps='{num_steps}', min_add_dim='{min_add_dim}'."
    #     )
    factor = (1 + int(directed))
    dim_per_step = max(max_dim // (num_steps + 1), min_add_dim)
    dims = (1 + num_steps) * [dim_per_step]
    dims = [min(factor ** i * in_dim, dim) for i, dim in enumerate(dims)]

    emb_dim = sum(dims)
    init_dim = dims[0]
    layer_dims = dims[1:]
    return emb_dim, init_dim, layer_dims
