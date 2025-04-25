import json

try:
    import pickle as pickle
except ImportError:
    import pickle

import argparse
import time
import numpy as np
from scipy.sparse import load_npz
from src.embedding import netmf
from src.conealign import align_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Run CONE Align.")

    parser.add_argument('adjA', help='Edge list of input graph A.')
    parser.add_argument('adjB', help='Edge list of input graph B.')
    parser.add_argument('output_alignment', help='Output path for alignment matrix.')
    parser.add_argument('--metadata_path', nargs='?', help='Output path for metadata file.')
    # parser.add_argument('--store_align', action='store_true', help='Store the alignment matrix.')

    # Node Embedding
    parser.add_argument('--embmethod', nargs='?', default='netMF', help='Node embedding method.')
    # netMF parameters
    parser.add_argument("--rank", default=256, type=int,
                        help='Number of eigenpairs used to approximate normalized graph Laplacian.')
    parser.add_argument("--dim", default=512, type=int, help='Dimension of embedding.')
    parser.add_argument("--window", default=10, type=int, help='Context window size.')
    parser.add_argument("--negative", default=1.0, type=float, help='Number of negative samples.')

    # Embedding Space Alignment
    # convex initialization parameters
    parser.add_argument('--niter_init', type=int, default=10, help='Number of iterations.')
    parser.add_argument('--reg_init', type=float, default=1.0, help='Regularization parameter.')
    # WP optimization parameters
    parser.add_argument('--nepoch', type=int, default=5, help='Number of epochs.')
    parser.add_argument('--niter_align', type=int, default=10, help='Iterations per epoch.')
    parser.add_argument('--reg_align', type=float, default=0.05, help='Regularization parameter.')
    parser.add_argument('--bsz', type=int, default=10, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate.')

    # Matching Nodes
    parser.add_argument('--embsim', nargs='?', default='euclidean', help='Metric for comparing embeddings.')
    parser.add_argument('--alignmethod', nargs='?', default='greedy', help='Method to align embeddings.')
    parser.add_argument('--numtop', type=int, default=10,
                        help='Number of top similarities to compute with kd-tree.  If None, computes all pairwise similarities.')

    return parser.parse_args()


def main(args):
    adjA = load_npz(args.adjA).astype(float)
    adjB = load_npz(args.adjB).astype(float)

    num_nodesA = adjA.shape[0]
    num_nodesB = adjB.shape[0]
    if num_nodesA != num_nodesB:
        raise ValueError("GraphA and GraphB must have the same number of nodes.")

    dim = min(num_nodesA - 1, args.dim)
    cone_args = vars(args)

    start = time.time()
    same_emb_dims = False
    while not same_emb_dims:
        emb_matrixA = netmf(
            adjA, dim=dim, window=args.window, b=args.negative, normalize=True)

        emb_matrixB = netmf(
            adjB, dim=dim, window=args.window, b=args.negative, normalize=True)
        same_emb_dims = emb_matrixA.shape[1] == emb_matrixB.shape[1]
        dim = min(emb_matrixA.shape[1], emb_matrixB.shape[1])

    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    alignment_matrix = align_embeddings(
        emb_matrixA,
        emb_matrixB,
        CONE_args=cone_args,
        adj1=adjA,
        adj2=adjB,
        struc_embed=None,
        struc_embed2=None
    )
    total_time = time.time() - start
    # print(("time for CONE-align (in seconds): %f" % total_time))
    np.save(args.output_alignment, alignment_matrix.toarray())

    if args.metadata_path:
        with open(args.metadata_path, 'w') as fp:
            json.dump({
                "total_duration": total_time
            }, fp, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
