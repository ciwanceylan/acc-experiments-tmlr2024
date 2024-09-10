import time
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import train_test_split

import accnebtools.data.graph as dgraphs
import accnebtools.experiments.classification as nodeclassification
import structfeatures.main as stf
import common


def svdpass_embeddings(x, adj_f, adj_b, dim=512):
    start = time.perf_counter()
    features = np.concatenate((x, adj_f @ x, adj_b @ x), axis=1)
    u, s, vh = np.linalg.svd(features, full_matrices=False)
    v = vh[:dim, :].T
    features = features @ v
    duration = time.perf_counter() - start
    print("SVDPASS embedding duration:", duration)
    return features, v


def pcapass_embeddings(x, adj_f, adj_b, dim=512):
    start = time.perf_counter()
    features = np.concatenate((x, adj_f @ x, adj_b @ x), axis=1)
    features = features - np.mean(features, axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(features, full_matrices=False)
    v = vh[:dim, :].T
    features = features @ v
    duration = time.perf_counter() - start
    print("PCAPASS embedding duration:", duration)
    return features, v


def acc_embeddings(x, adj_f, adj_b, dim=512):
    start = time.perf_counter()
    ux, sx, vhx = np.linalg.svd(x, full_matrices=False)
    vx = vhx[:dim // 2, :].T
    features_x = x @ vx

    adj_features = np.concatenate((adj_f @ features_x, adj_b @ features_x), axis=1)
    u_adjx, s_adjx, vh_adjx = np.linalg.svd(adj_features, full_matrices=False)
    v_adjx = vh_adjx[:dim // 2, :].T
    adj_features = adj_features @ v_adjx
    duration = time.perf_counter() - start

    features = np.concatenate((features_x, adj_features), axis=1)
    print("ACC embedding duration:", duration)
    return features, vx, v_adjx


def acc_pca_embeddings(x, adj_f, adj_b, dim=512):
    start = time.perf_counter()
    x = x - np.mean(x, axis=0, keepdims=True)
    ux, sx, vhx = np.linalg.svd(x, full_matrices=False)
    vx = vhx[:dim // 2, :].T
    features_x = x @ vx

    adj_features = np.concatenate((adj_f @ features_x, adj_b @ features_x), axis=1)
    adj_features = adj_features - np.mean(adj_features, axis=0, keepdims=True)
    u_adjx, s_adjx, vh_adjx = np.linalg.svd(adj_features, full_matrices=False)
    v_adjx = vh_adjx[:dim // 2, :].T
    adj_features = adj_features @ v_adjx
    duration = time.perf_counter() - start

    features = np.concatenate((features_x, adj_features), axis=1)
    print("ACC PCA embedding duration:", duration)
    return features, vx, v_adjx


def grad_boost_eval(features, node_labels, train_idx, test_idx, seed):
    model, model_name = nodeclassification.get_default_classification_model(
        "multiclass",
        feature2node_ratio=0,
        random_state=seed,
        model_name="grad_boost"
    )
    start = time.perf_counter()
    model = model.fit(features[train_idx], y=node_labels[train_idx])
    accuracy = model.score(features[test_idx], y=node_labels[test_idx])
    duration = time.perf_counter() - start
    print("Classification duration:", duration)
    print("Accuracy:", accuracy)
    return accuracy


def run_eval(dataroot: str, dataset_spec: dgraphs.DatasetSpec, seed: int):
    data_graph = dgraphs.SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)
    node_labels, node_labels_type = nodeclassification.read_node_labels(dataroot=dataroot,
                                                                        dataset=dataset_spec.data_name)

    bf_params = stf.BaseFeatureParams(
        use_weights=False,
        use_node_attributes=data_graph.is_node_attributed,
        as_undirected=data_graph.is_undirected,
        use_degree=True,
        use_lcc=True,
        use_egonet_edge_counts=False,
        use_legacy_egonet_edge_counts=False
    )

    _edge_index, weights, node_attributes = stf.prepare_inputs(edge_index=data_graph.edges,
                                                               num_nodes=data_graph.num_nodes,
                                                               bf_params=bf_params,
                                                               node_attributes=data_graph.node_attributes)
    base_features, _ = stf.get_structural_initial_features(edge_index=_edge_index, num_nodes=data_graph.num_nodes,
                                                           bf_params=bf_params, weights=weights,
                                                           node_attributes=node_attributes)

    std = np.std(base_features, axis=0)
    std[std == 0] = 1
    x = (base_features - np.mean(base_features, axis=0)) / std

    adj = sp.coo_array((np.ones(data_graph.num_edges), (data_graph.edges[:, 1], data_graph.edges[:, 0])),
                       shape=(data_graph.num_nodes, data_graph.num_nodes))
    out_degrees = adj.sum(axis=0)
    inv_out_degrees = np.zeros_like(out_degrees)
    inv_out_degrees[out_degrees > 0] = 1. / out_degrees[out_degrees > 0]
    in_degrees = adj.sum(axis=0)
    inv_in_degrees = np.zeros_like(in_degrees)
    inv_in_degrees[in_degrees > 0] = 1. / in_degrees[in_degrees > 0]

    adj_f = inv_in_degrees.reshape(-1, 1) * adj
    adj_b = inv_out_degrees.reshape(-1, 1) * adj.T

    # x_svs = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    # adjf_svs = np.linalg.svd(adj_f @ x, full_matrices=False, compute_uv=False)
    # adjb_svs = np.linalg.svd(adj_b @ x, full_matrices=False, compute_uv=False)
    #
    # svdpass_features, v_svdpass = svdpass_embeddings(x, adj_f, adj_b, dim=512)
    # acc_features, v_x_acc, v_adj_acc = acc_embeddings(x, adj_f, adj_b, dim=512)
    #
    pcapass_features, v_pcapass = pcapass_embeddings(x, adj_f, adj_b, dim=512)
    acc_features, v_x_acc, v_adj_acc = acc_pca_embeddings(x, adj_f, adj_b, dim=512)

    train_idx, test_idx = train_test_split(np.arange(data_graph.num_nodes, dtype=np.int64), test_size=0.2,
                                           random_state=seed)
    x_accuracy = grad_boost_eval(x, node_labels, train_idx, test_idx, seed=seed)
    adjf_x_accuracy = grad_boost_eval(adj_f @ x, node_labels, train_idx, test_idx, seed=seed)
    adjb_x_accuracy = grad_boost_eval(adj_b @ x, node_labels, train_idx, test_idx, seed=seed)
    pcapass_accuracy = grad_boost_eval(pcapass_features, node_labels, train_idx, test_idx, seed=seed)
    acc_accuracy = grad_boost_eval(acc_features, node_labels, train_idx, test_idx, seed=seed)
    data = {"pcapass_accuracy": pcapass_accuracy, "acc_accuracy": acc_accuracy,
            "x_accuracy": x_accuracy, "adjf_x_accuracy": adjf_x_accuracy, "adjb_x_accuracy": adjb_x_accuracy,
            "v_x_acc": v_x_acc, "v_adj_acc": v_adj_acc, "v_pcapass": v_pcapass,
            # "x_svs": x_svs, "adjf_svs": adjf_svs, "adjb_svs": adjb_svs
            }
    return data


def main():
    experiment_name = "nc_feature_importance"
    parser = common.get_common_parser()
    args = parser.parse_args()
    results_path, resources, dataset_spec, args = common.setup_experiment(experiment_name, args)

    data = run_eval(dataroot=args.dataroot, dataset_spec=dataset_spec, seed=args.seed)
    np.savez(results_path[:-5] + "_results.npz", **data)


if __name__ == "__main__":
    main()
