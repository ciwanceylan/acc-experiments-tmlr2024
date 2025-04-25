import argparse
import json
import time
import scipy.sparse as sp
from scipy.sparse import load_npz
import numpy as np
import torch
from src.GromovWassersteinGraphToolkit import recursive_direct_graph_matching


def parse_args():
    parser = argparse.ArgumentParser(description="Run S-GWL alignment")
    parser.add_argument('adjA', help='Edge list of input graph A.')
    parser.add_argument('adjB', help='Edge list of input graph B.')
    parser.add_argument('output_alignment', help='Output path for alignment matrix.')
    parser.add_argument('--metadata_path', nargs='?', help='Output path for metadata file.')
    parser.add_argument('--clus', default=2, help='Num clusters')
    parser.add_argument('--level', default=3, help='Level parameter')
    parser.add_argument('--max-cpu', default=0, help='Num cpu threads')

    return parser.parse_args()


def compute_normalized_degrees(adj):
    deg_out = np.asarray(adj.sum(axis=1)).ravel()
    deg_in = np.asarray(adj.sum(axis=0)).ravel()
    degrees = deg_in + deg_out
    norm_degs = degrees / np.sum(degrees)
    return norm_degs.reshape(-1, 1)


def add_self_loops_for_isolated_nodes(adj):
    out_deg = np.asarray(adj.sum(axis=1)).ravel()
    nodes = np.nonzero(out_deg < 1)[0]
    self_loop_mat = sp.coo_matrix((np.ones(len(nodes), dtype=float), (nodes, nodes)), shape=adj.shape).tocsr()
    adj = adj + self_loop_mat
    return adj


def main(args):
    print("SGWL")
    adjA = load_npz(args.adjA).astype(float)
    adjB = load_npz(args.adjB).astype(float)

    start = time.time()
    adjA = add_self_loops_for_isolated_nodes(adjA)
    adjB = add_self_loops_for_isolated_nodes(adjB)

    cost_s = adjA
    cost_t = adjB
    p_s = compute_normalized_degrees(adjA)
    p_t = compute_normalized_degrees(adjB)
    idx2node_s = {i: i for i in range(adjA.shape[0])}
    idx2node_t = {i: i for i in range(adjB.shape[0])}

    if args.max_cpu > 0:
        torch.set_num_threads(args.max_cpu)

    ot_dict = {
        'loss_type': 'L2',  # the key hyperparameters of GW distance
        'ot_method': 'proximal',
        'beta': 0.2,  # 0.025-0.1 depends on degree
        # outer, inner iteration, error bound of optimal transport
        'outer_iteration': adjA.shape[0],  # num od nodes
        'iter_bound': 1e-10,
        'inner_iteration': 2,
        'sk_bound': 1e-30,  # --mine
        'node_prior': 1000,  # --mine
        'max_iter': 4,  # --mine  # iteration and error bound for calcuating barycenter
        'cost_bound': 1e-26,
        'update_p': False,  # optional updates of source distribution
        'lr': 0,
        'alpha': 1
    }

    pairs_idx, pairs_name, pairs_confidence, trans = recursive_direct_graph_matching(
        0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t,
        idx2node_s, idx2node_t, ot_dict, weights=None, predefine_barycenter=False,
        cluster_num=args.clus, partition_level=args.level, max_node_num=200
    )
    finish = time.time()

    np.save(args.output_alignment, trans)
    if args.metadata_path:
        with open(args.metadata_path, 'w') as fp:
            json.dump({
                "total_duration": finish - start,
            }, fp, indent=2)

    return trans


if __name__ == "__main__":
    args = parse_args()
    main(args)
