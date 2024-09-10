#!/usr/bin/env bash


mkdir -p "nc_logs"

CPU_WORKERS = 32  # Used to set paths for Numba caches
GPU_WORKERS = 1  # TODO

declare -a algs=("node_attribute_only" "acc_cr" "acc_cr_pca" "svdpass_cr" "pcapass_cr" "sgcn" "graphmae" "graphmae2" "graphmae2_gs" "bgrl" "bgrl_gs" "ccassg" "dgi" "gae" "spgcl" "mvgrl" "greet")
declare -a datasets=("pyg_chameleon" "pyg_squirrel" "roman_empire" "ogb_arxiv_year")


for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --cpu-workers $CPU_WORKERS --gpu-workers $GPU_WORKERS --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 10000 --pp-mode all --clf log_reg_sgd::grad_boost --num-reps 5 --dims 512 >> nc_logs/${d}_${a}.log 2>> nc_logs/${d}_${a}_error.log
    done
done


# SNAP PATENTS
declare -a datasets=("snap_patents")
declare -a algs=("node_attribute_only" "acc" "pcapass_dirank" "sgcn" "graphmae" "graphmae2" "graphmae2_gs" "ccassg" "gae")

for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --cpu-workers 32 --cpu-memory 128 --gpu-workers 0 --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 86400 --pp-mode all --clf log_reg_sgd::grad_boost --num-reps 1 --dims 256 >> nc_logs/${d}_${a}.log 2>> nc_logs/${d}_${a}_error.log
    done
done