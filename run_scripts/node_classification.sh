#!/usr/bin/env bash


mkdir -p "nc_logs"

declare -a algs=("node_attribute_only" "acc" "pcapass" "sgcn" "graphmae" "graphmae2" "graphmae2_gs" "bgrl" "bgrl_gs" "ccassg" "dgi" "gae" "spgcl" "mvgrl" "greet")
declare -a datasets=("pyg_chameleon" "pyg_squirrel" "roman_empire" "ogb_arxiv_year")


for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 10000 --pp-mode all --clf log_reg_sgd::grad_boost --num-reps 5 --dims 512 >> nc_logs/${d}_${a}.log 2>> nc_logs/${d}_${a}_error.log
    done
done


# SNAP PATENTS
declare -a datasets=("snap_patents")
declare -a algs=("node_attribute_only" "acc" "pcapass" "sgcn" "graphmae" "graphmae2" "graphmae2_gs" "ccassg" "gae")

for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataroot ./data --resultdir ./neb_gcp_results --tempdir /tmp --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 86400 --pp-mode all --clf log_reg_sgd::grad_boost --num-reps 1 --dims 256 >> nc_logs/${d}_${a}.log 2>> nc_logs/${d}_${a}_error.log
    done
done