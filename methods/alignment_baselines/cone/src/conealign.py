import numpy as np
import sklearn.metrics.pairwise

try:
    import pickle as pickle
except ImportError:
    import pickle
import scipy.sparse as sps
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import unsup_align


# import embedding

def align_embeddings(embed1, embed2, CONE_args, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        if CONE_args['embsim'] == "cosine":
            corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
        else:
            corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
            corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1):
            adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2):
            adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = unsup_align.convex_init_sparse(
            embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False, niter=CONE_args['niter_init'],
            reg=CONE_args['reg_init'], P=corr)
    else:
        init_sim, corr_mat = unsup_align.convex_init(
            embed1, embed2, apply_sqrt=False, niter=CONE_args['niter_init'], reg=CONE_args['reg_init'], P=corr)

    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=CONE_args['lr'], bsz=CONE_args['bsz'], nepoch=CONE_args['nepoch'],
        niter=CONE_args['niter_align'], reg=CONE_args['reg_align'])

    aligned_embed1 = embed1.dot(dim_align_matrix)

    alignment_matrix = kd_align(
        aligned_embed1, embed2, distance_metric=CONE_args['embsim'], num_top=CONE_args['numtop'])

    return alignment_matrix


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    # print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix(
        (data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()
