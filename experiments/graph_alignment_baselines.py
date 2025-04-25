import copy
import os
import tempfile
import json
from typing import Sequence, Literal

import pandas as pd
import tqdm.auto as tqdm
import numpy as np
import scipy.sparse as sp

import accnebtools.data.graph as dgraphs
from accnebtools.data.graph import SimpleGraph, DatasetSpec
import accnebtools.algs.utils as algutils
import accnebtools.experiments.alignment as alignment
from accnebtools.utils import NEB_ROOT
import common


def make_alignment_graph(graph: SimpleGraph, noise_model: str, p: float, rng: np.random.Generator):
    graph2, alignment_objective = alignment.create_permuted(graph, rng=rng)
    if noise_model.lower() == "add":
        graph2, actual_p = dgraphs.add_noise_edges(graph2, p=p, rng=rng)
    elif noise_model.lower() == "remove":
        graph2, actual_p = dgraphs.remove_edges_noise(graph2, p=p, rng=rng)
    else:
        raise NotImplementedError(f"Noise model {noise_model} is not found.")

    return graph, graph2, alignment_objective, actual_p


def load_magna_alignment_graph(dataroot: str, p: float):
    noise_level = int(100 * p)
    index_path = os.path.join(dataroot, "data_index.json")
    with open(index_path, 'r') as fp:
        dataset_info = json.load(fp)["magna"]
    g1_graph = SimpleGraph.from_dataset_index(dataroot=dataroot, data_name='magna', as_undirected=True,
                                              as_unweighted=True)
    g2_path = os.path.join(dataset_info['datapath'], dataset_info['alignment_graphs_dir'], f'graph+{noise_level}e.npz')
    g2_graph = SimpleGraph.load(g2_path, as_canonical_undirected=True, add_symmetrical_edges=True, use_weights=False)

    align_obj_path = os.path.join(dataset_info['datapath'], dataset_info['alignment_graphs_dir'],
                                  f'graph+{noise_level}e_align_obj.csv')
    alignment_objective = alignment.AlignedGraphs.load_from_file(align_obj_path)
    return g1_graph, g2_graph, alignment_objective, p


def run_conealign(adjA_path, adjB_path, output_file_name, metadata_file_name, tempdir: str, timeout: int):
    os.makedirs(tempdir, exist_ok=True)
    method_dir = os.path.join(NEB_ROOT, 'methods/alignment_baselines/cone')
    run_command = [
        "conda", "run", "-n", "conealign_env",
        "python", f"{os.path.join(NEB_ROOT, method_dir, 'conealign.py')}",
        f"{adjA_path}", f"{adjB_path}", f"{output_file_name}",
        "--metadata_path", f"{metadata_file_name}",
        "--embmethod", "netMF",
    ]
    outcome, error_out = algutils.run_command(command=run_command, timeout_time=timeout,
                                              cwd=method_dir)

    return outcome, error_out


def run_sgwl(adjA_path, adjB_path, output_file_name, metadata_file_name, tempdir: str, timeout: int):
    os.makedirs(tempdir, exist_ok=True)
    method_dir = os.path.join(NEB_ROOT, 'methods/alignment_baselines/sgwl')
    run_command = [
        "conda", "run", "-n", "acc_neb_env",
        "python", f"{os.path.join(NEB_ROOT, method_dir, 'run_sgwl.py')}",
        f"{adjA_path}", f"{adjB_path}", f"{output_file_name}",
        "--metadata_path", f"{metadata_file_name}",
    ]
    outcome, error_out = algutils.run_command(command=run_command, timeout_time=timeout, cwd=method_dir)
    return outcome, error_out


def run_fugal(adjA_path, adjB_path, output_file_name, metadata_file_name, tempdir: str, timeout: int):
    os.makedirs(tempdir, exist_ok=True)
    method_dir = os.path.join(NEB_ROOT, 'methods/alignment_baselines/fugal')
    run_command = [
        "conda", "run", "-n", "acc_neb_env",
        "python", f"{os.path.join(NEB_ROOT, method_dir, 'run_fugal.py')}",
        f"{adjA_path}", f"{adjB_path}", f"{output_file_name}",
        "--metadata_path", f"{metadata_file_name}",
    ]
    outcome, error_out = algutils.run_command(command=run_command, timeout_time=timeout, cwd=method_dir)
    return outcome, error_out


METHOD_DICT = {
    "conealign": run_conealign,
    "sgwl": run_sgwl,
    "fugal": run_fugal
}


def eval_alignment(methods, graphA: dgraphs.SimpleGraph, graphB: dgraphs.SimpleGraph,
                   alignment_objective: alignment.AlignedGraphs, matching_alg: Literal['nearest', 'sort_greedy'],
                   tempdir: str, timeout: int):
    results = []
    os.makedirs(tempdir, exist_ok=True)
    adjA = sp.coo_matrix(
        (np.ones(graphA.num_edges, dtype=np.float32), (graphA.edges[:, 0], graphA.edges[:, 1])),
        shape=(graphA.num_nodes, graphB.num_nodes)
    ).tocsr()
    adjB = sp.coo_matrix(
        (np.ones(graphB.num_edges, dtype=np.float32), (graphB.edges[:, 0], graphB.edges[:, 1])),
        shape=(graphB.num_nodes, graphB.num_nodes)
    ).tocsr()

    with tempfile.NamedTemporaryFile(dir=tempdir, delete=True, suffix='.npz') as graphA_file, \
            tempfile.NamedTemporaryFile(dir=tempdir, delete=True, suffix='.npz') as graphB_file:
        sp.save_npz(graphA_file, adjA)
        sp.save_npz(graphB_file, adjB)

        for method in tqdm.tqdm(methods):
            with tempfile.NamedTemporaryFile(dir=tempdir, delete=True, suffix='.npy') as output_file, \
                    tempfile.NamedTemporaryFile(dir=tempdir, delete=True, suffix='.json') as metadata_file:
                outcome, error_out = METHOD_DICT[method](
                    adjA_path=graphA_file.name,
                    adjB_path=graphB_file.name,
                    output_file_name=output_file.name,
                    metadata_file_name=metadata_file.name,
                    tempdir=tempdir,
                    timeout=timeout
                )

                if outcome == 'completed' and os.path.getsize(output_file.name) > 0:
                    similarity = np.load(output_file)
                    metadata = json.load(metadata_file)
                    matching = alignment.create_matching(similarity, matching_alg)
                else:
                    print(f"{method} failed: {outcome}.")
                    print("Error message: ")
                    print("\n".join(error_out))
                    matching = None
                    metadata = None

            if matching is not None:
                accuracy = compute_alignment_accuracy(alignment_objective, matching)
                results.append({"method": method, "k@1": accuracy, "outcome": outcome,
                                "duration": metadata["total_duration"]})
            else:
                results.append({"method": method, "outcome": outcome, "error": "\n".join(error_out)})
    return results


def compute_alignment_accuracy(align_obj: alignment.AlignedGraphs, matching):
    alignments = pd.Series(index=matching[:, 0], data=matching[:, 1])
    count = 0
    for g1_index, g2_index in align_obj.g1_to_g2.items():
        if g1_index in alignments and alignments[g1_index] == g2_index:
            count += 1
    accuracy = count / len(align_obj.g1_to_g2)
    return accuracy


def run_eval(dataroot: str, dataset_spec: DatasetSpec, methods, noise_model: str, seed: int,
             matching_alg: Literal['nearest', 'sort_greedy'], noise_levels: Sequence[float], num_reps: int = 5,
             tempdir: str = "./", results_path: str = None, timeout: int = 3600, debug: bool = False):
    if debug:
        num_reps = 1
        noise_levels = [0.01]

    all_results = []
    rng = np.random.default_rng(seed)
    seed_spawners = np.random.SeedSequence(seed).spawn(len(noise_levels))

    data_graph = SimpleGraph.from_dataset_spec(dataroot=dataroot, dataset_spec=dataset_spec)

    for noise_p, ss in zip(tqdm.tqdm(noise_levels), seed_spawners):
        spawned_seeds = ss.generate_state(num_reps)
        for rep, rep_seed in zip(tqdm.trange(num_reps), spawned_seeds):
            if dataset_spec.data_name == "magna":
                graphA, graphB, align_obj, noise_p_actual = load_magna_alignment_graph(
                    dataroot=dataroot, p=noise_p
                )
            else:
                graphA, graphB, align_obj, noise_p_actual = make_alignment_graph(
                    data_graph,
                    noise_model=noise_model,
                    p=noise_p,
                    rng=rng
                )
            results = eval_alignment(
                methods=methods,
                graphA=graphA,
                graphB=graphB,
                alignment_objective=align_obj,
                matching_alg=matching_alg,
                tempdir=tempdir,
                timeout=timeout
            )
            for entry in results:
                res = copy.deepcopy(entry)
                res["seed"] = rep_seed
                res["rep"] = rep
                res["noise_p"] = noise_p
                res["noise_p_actual"] = noise_p_actual
                all_results.append(res)
        if results_path:
            pd.DataFrame(all_results).to_json(results_path, indent=2, orient="records")
    return all_results


def main():
    experiment_name = "network_alignment_baselines"
    parser = common.get_common_parser()
    parser.add_argument("--noisemodel", type=str, help="Which method to add noise to the graphs",
                        default="remove")
    parser.add_argument("--noise-p", type=str, default="full",
                        help="Amount of edge noise to use.")
    parser.add_argument("--num-reps", type=int, default=5,
                        help="Num repeats")
    parser.add_argument("--matching-alg", type=str, default='nearest',
                        help="Which matching alg to use, 'nearest' or 'sort_greedy'. (Default: 'nearest')")
    args = parser.parse_args()

    if args.noise_p == "low":
        noise_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    elif args.noise_p == "high":
        noise_levels = [0.075, 0.1, 0.15, 0.2, 0.25]
    elif args.noise_p == "full":
        noise_levels = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
    elif args.noise_p == "selected":
        noise_levels = [0.05, 0.1, 0.2]
    else:
        noise_levels = [float(args.noise_p)]

    experiment_name = f"{experiment_name}_{args.noise_p}/{args.matching_alg}"

    results_path, dataset_spec, args = common.setup_experiment(experiment_name, args)

    results = run_eval(dataroot=args.dataroot, methods=args.methods,
                       dataset_spec=dataset_spec, noise_model=args.noisemodel, matching_alg=args.matching_alg,
                       tempdir=args.tempdir, results_path=results_path, timeout=args.timeout,
                       seed=args.seed, debug=args.debug, noise_levels=noise_levels,
                       num_reps=args.num_reps)
    pd.DataFrame(results).to_json(results_path, indent=2, orient="records")


if __name__ == "__main__":
    main()
