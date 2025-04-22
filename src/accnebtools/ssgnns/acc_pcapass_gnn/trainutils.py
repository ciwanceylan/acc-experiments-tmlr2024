import dataclasses as dc
from typing import Sequence, Literal, Tuple, Optional

import numpy as np
import torch

import nebtools.data.graph as dgraph
from nebtools.ssgnns.utils import SSGNNTrainer, get_features

from .GraphMAE2.models.edcoder import PreModel

ENCODERS = Literal['acc', 'pcapass', 'dir_graphsage', 'mlp', 'linear']


@dc.dataclass(frozen=True)
class Parameters:
    num_epochs: int = 1000
    lr: float = 0.001
    wd: float = 1e-4
    max_emb_dim: int = 1024
    num_layers: int = 6
    mask_rate: float = 0.5
    replace_rate: float = 0.05
    alpha_l: int = 3
    lam: float = 1.0
    use_scheduler: bool = True
    add_degree: bool = True
    add_lcc: bool = True
    standardize: bool = True
    encoder: ENCODERS = 'acc'
    decoder: ENCODERS = 'dir_graphsage'
    bias: bool = False


def build_model(directed: bool, num_in_features: int, params: Parameters):
    max_emb_dim = params.max_emb_dim
    num_layers = params.num_layers
    mask_rate = params.mask_rate
    replace_rate = params.replace_rate
    alpha_l = params.alpha_l
    lam = params.lam
    encoder_type = params.encoder
    decoder_type = params.decoder
    bias = params.bias

    norm = None
    in_drop = 0.2
    drop_edge_rate = 0.0
    activation = 'prelu'
    loss_fn = 'sce'
    remask_method = "fixed"
    mask_method = "random"
    remask_rate = 0.5
    momentum = 0.996
    num_remasking = 3
    num_dec_layers = 1
    delayed_ema_epoch = 0
    zero_init = False

    model = PreModel(
        directed=directed,
        in_dim=num_in_features,
        max_emb_dim=max_emb_dim,
        num_layers=num_layers,
        num_dec_layers=num_dec_layers,
        num_remasking=num_remasking,
        activation=activation,
        feat_drop=in_drop,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        remask_rate=remask_rate,
        mask_method=mask_method,
        norm=norm,
        loss_fn=loss_fn,
        drop_edge_rate=drop_edge_rate,
        alpha_l=alpha_l,
        lam=lam,
        delayed_ema_epoch=delayed_ema_epoch,
        replace_rate=replace_rate,
        remask_method=remask_method,
        momentum=momentum,
        zero_init=zero_init,
        bias=bias
    )
    return model


class GraphMAE2Trainer(SSGNNTrainer):

    def __init__(self, graph: dgraph.SimpleGraph, params: Parameters, device: torch.device):
        self.graph = graph.to_dgl_graph()
        self.graph = self.graph.remove_self_loop()
        self.feat = get_features(graph, add_degree=params.add_degree, add_lcc=params.add_lcc,
                                 standardize=params.standardize)
        self.graph = self.graph.to(device)
        self.feat = self.feat.to(device)
        self.target_nodes = torch.arange(self.feat.shape[0], device=device, dtype=torch.long)
        self.params = params
        self.num_nodes = graph.num_nodes
        self.model = build_model(directed=graph.directed, num_in_features=self.feat.shape[1], params=params)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr, weight_decay=params.wd)

        if params.use_scheduler and params.num_epochs > 0:
            self.scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / params.num_epochs)) * 0.5
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.scheduler)
        else:
            self.scheduler = None

        self.device = device

    def step(self, step: int):
        self.model.train()

        loss = self.model(self.graph, self.feat, targets=self.target_nodes)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def get_embeddings(self) -> torch.Tensor:
        self.model.eval()
        embeds = self.model.embed(self.graph.to(self.device), self.feat.to(self.device))
        return embeds

    def get_embeddings_for_graph(self, graph: dgraph.SimpleGraph) -> torch.Tensor:
        dgl_graph = graph.to_dgl_graph()
        dgl_graph = dgl_graph.remove_self_loop().add_self_loop()
        feat = get_features(graph, add_degree=self.params.add_degree, add_lcc=self.params.add_lcc,
                            standardize=self.params.standardize)
        self.model.eval()
        embeds = self.model.embed(dgl_graph.to(self.device), feat.to(self.device))
        return embeds
