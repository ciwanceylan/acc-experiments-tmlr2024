# Fugal Algorithm was provided by anonymous authors.
import argparse
import json
import time
import numpy as np
import torch
import networkx as nx
from scipy.sparse import load_npz
from src.pred import feature_extraction, eucledian_dist, convex_init


def parse_args():
    parser = argparse.ArgumentParser(description="Run Fugal alignment")
    parser.add_argument('adjA', help='Edge list of input graph A.')
    parser.add_argument('adjB', help='Edge list of input graph B.')
    parser.add_argument('output_alignment', help='Output path for alignment matrix.')
    parser.add_argument('--metadata_path', nargs='?', help='Output path for metadata file.')
    parser.add_argument('--iter', default=15, type=int, help='Num iterations')
    parser.add_argument('--mu', default=1, type=float, help='Mu parameter')

    return parser.parse_args()


# def are_matrices_equal(matrix1, matrix2):
#     # Check if dimensions are the same
#     if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
#         return False
#
#     # Check element-wise equality
#     for i in range(len(matrix1)):
#         for j in range(len(matrix1[0])):
#             if matrix1[i][j] != matrix2[i][j]:
#                 return False
#
#     # If no inequality is found, matrices are equal
#     return True


# def main(data, iter, simple, mu, EFN=5):
def main(args):
    print("Fugal")
    mu = args.mu
    iter = args.iter
    dtype = np.float64
    torch.set_num_threads(40)

    Src = load_npz(args.adjA).astype(dtype).toarray()
    Tar = load_npz(args.adjB).astype(dtype).toarray()

    directed = not np.allclose(Src, Src.T)

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

    simple = True
    if directed:
        Src1 = nx.from_numpy_array(Src, create_using=nx.DiGraph)
        Tar1 = nx.from_numpy_array(Tar, create_using=nx.DiGraph)
        F1 = np.concatenate((
            feature_extraction(Src1, simple),
            feature_extraction(Src1.reverse(), simple)
        ),
            axis=1
        )
        F2 = np.concatenate((
            feature_extraction(Tar1, simple),
            feature_extraction(Tar1.reverse(), simple)
        ),
            axis=1
        )
    else:
        Src1 = nx.from_numpy_array(Src, create_using=nx.Graph)
        Tar1 = nx.from_numpy_array(Tar, create_using=nx.Graph)
        F1 = feature_extraction(Src1, simple)
        F2 = feature_extraction(Tar1, simple)

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
