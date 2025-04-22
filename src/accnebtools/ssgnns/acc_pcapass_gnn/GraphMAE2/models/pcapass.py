from typing import Union, List
import torch
from torch import nn

from .utils import create_activation
from .layers import OneDirSAGEConv


class PCAPassGraphSAGE(nn.Module):
    def __init__(self,
                 directed: bool,
                 in_dim: int,
                 hidden_dim: int,
                 num_layers,
                 activation,
                 feat_drop,
                 norm=None,
                 encoding=False,
                 bias: bool = False,
                 ):
        super(PCAPassGraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        self.norm = norm if encoding else None
        self.is_encoding = encoding
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.feat_drop = feat_drop
        self.bias = bias

        self.init_layer = nn.Sequential(
            nn.Dropout(self.feat_drop),
            nn.Linear(in_features=self.in_dim, out_features=hidden_dim, bias=self.bias),
            create_activation(self.activation))
        self.pcapass_layers = nn.ModuleList()
        if directed:
            self.init_directed_model(DirPCAPassConv, hidden_dim=hidden_dim)
        else:
            self.init_directed_model(PCAPassConv, hidden_dim=hidden_dim)

    def init_directed_model(self, conv_model: Union['DirPCAPassConv', 'PCAPassConv'], hidden_dim: int):

        if self.num_layers > 0:
            self.pcapass_layers.append(conv_model(
                hidden_dim, hidden_dim, aggregator_type='mean', feat_drop=self.feat_drop,
                activation=create_activation(self.activation), norm=self.norm))

            for l in range(1, self.num_layers):
                self.pcapass_layers.append(conv_model(
                    hidden_dim, hidden_dim, aggregator_type='mean', feat_drop=self.feat_drop,
                    activation=create_activation(self.activation), norm=self.norm))

    def forward(self, g, inputs):
        h = self.init_layer(inputs)

        for l, pcapass_layer in enumerate(self.pcapass_layers):
            h = pcapass_layer(g, h)

        return h


class DirPCAPassConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 *,
                 feat_drop=0.,
                 aggregator_type='mean',
                 activation=None,
                 bias=False,
                 norm=None):
        super(DirPCAPassConv, self).__init__()

        self.fc_self_neigh = nn.Linear(in_features=3 * in_feats, out_features=out_feats, bias=bias)
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
        h_self, fwd_neigh = self.fwd_sage_conv(graph, feat)
        _, bwd_neigh = self.bwd_sage_conv(graph.reverse(), feat)
        # torch.cat((h_self, fwd_neigh, bwd_neigh), dim=0)
        rst = self.fc_self_neigh(torch.cat((h_self, fwd_neigh, bwd_neigh), dim=1))
        if self.activation:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        return rst


class PCAPassConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 *,
                 feat_drop=0.,
                 aggregator_type='mean',
                 activation=None,
                 bias=False,
                 norm=None):
        super(PCAPassConv, self).__init__()

        self.fc_self_neigh = nn.Linear(in_features=2 * in_feats, out_features=out_feats, bias=bias)
        self.activation = activation
        self.norm = norm

        self.sage_conv = OneDirSAGEConv(
            in_feats=in_feats,
            out_feats=out_feats,
            feat_drop=feat_drop,
            aggregator_type=aggregator_type
        )

    def forward(self, graph, feat):
        h_self, neigh_rst = self.sage_conv(graph, feat)
        rst = self.fc_self_neigh(torch.cat((h_self, neigh_rst), dim=1))
        if self.activation:
            rst = self.activation(rst)
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
