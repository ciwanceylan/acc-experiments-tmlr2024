import argparse
import json
import time
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import load_npz
from src.pred import eucledian_dist, convex_init

import accmp.acc as acc
import accmp.transforms as accmptrns
import accmp.preprocessing as preproc


def parse_args():
    parser = argparse.ArgumentParser(description="Run Fugal alignment")
    parser.add_argument('adjA', help='Edge list of input graph A.')
    parser.add_argument('adjB', help='Edge list of input graph B.')
    parser.add_argument('output_alignment', help='Output path for alignment matrix.')
    parser.add_argument('--metadata_path', nargs='?', help='Output path for metadata file.')
    parser.add_argument('--iter', default=15, type=int, help='Num iterations')
    parser.add_argument('--mu', default=1, type=float, help='Mu parameter')
    parser.add_argument('--K', default=6, type=float, help='Num message-passing iterations')

    return parser.parse_args()


def compute_acc_embs(num_nodes, edges, K):
    dtype = np.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    init_params = preproc.InitFeaturesWeightsParams(
        use_degree=True,
        use_log1p_degree=False,
        use_lcc=True,
        use_weights=False,
        use_node_attributes=False,
        as_undirected=False,
        dtype=dtype
    )
    norm = None

    normalization = accmptrns.FeatureNormalization(
        mode=norm,
        subtract_mean=True,
        before_prune=True,
        before_propagate=False
    )

    params = acc.ACCParams(
        max_steps=K,
        initial_feature_standardization=accmptrns.FeatureNormalization(mode='std', subtract_mean=True,
                                                                       before_prune=False, before_propagate=False),
        mp_feature_normalization=normalization,
        init_params=init_params,
        decomposed_layers=1,
        normalized_weights=True,
        min_add_dim=2,
        max_dim=512,
        return_us=False,
        sv_thresholding='rtol',
        theta=1e-8,
        use_rsvd=False,
    )

    embeddings = acc.agg_compress_cat_embeddings(
        edge_index=edges.T, num_nodes=num_nodes, directed_conv=True, params=params, weights=None,
        node_attributes=None, device=device, return_np=False)
    embeddings = embeddings.cpu()
    return embeddings


def main(args):
    print("Fugal")
    mu = args.mu
    iter = args.iter
    dtype = np.float64
    torch.set_num_threads(40)

    adjA = load_npz(args.adjA)
    adjB = load_npz(args.adjB)
    adj_comb = sp.block_diag((adjA, adjB))
    edges = np.stack((adj_comb.row, adj_comb.col), axis=1).astype(np.int64)
    num_nodes = adj_comb.shape[0]

    embeddings = compute_acc_embs(edges=edges, num_nodes=num_nodes, K=args.K).to(torch.float64)
    F1 = embeddings[:adjA.shape[0], :]
    F2 = embeddings[adjA.shape[0]:, :]

    Src = load_npz(args.adjA).astype(dtype).toarray()
    Tar = load_npz(args.adjB).astype(dtype).toarray()

    start = time.time()
    for i in range(Src.shape[0]):
        row_sum1 = np.sum(Src[i, :])

        # If the sum of the row is zero, add a self-loop
        if row_sum1 == 0:
            Src[i, i] = 1
    for i in range(Tar.shape[0]):
        row_sum = np.sum(Tar[i, :])

        # If the sum of the row is zero, add a self-loop
        if row_sum == 0:
            Tar[i, i] = 1
    n1 = Tar.shape[0]
    n2 = Src.shape[0]
    n = max(n1, n2)

    A = torch.from_numpy(Src).to(dtype=torch.float64)
    B = torch.from_numpy(Tar).to(dtype=torch.float64)

    D = eucledian_dist(F1, F2, n)
    D = torch.from_numpy(D).to(dtype=torch.float64)

    P = convex_init(A, B, D, mu, iter)
    finish = time.time()

    np.save(args.output_alignment, P)
    if args.metadata_path:
        with open(args.metadata_path, 'w') as fp:
            json.dump({
                "total_duration": finish - start,
            }, fp, indent=2)

    return P


if __name__ == "__main__":
    args = parse_args()
    main(args)
