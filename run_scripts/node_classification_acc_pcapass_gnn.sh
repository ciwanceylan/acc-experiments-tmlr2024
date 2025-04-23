#!/usr/bin/env bash

declare -a algs=("accgnn" "pcapassgnn")
declare -a datasets=("chameleon" "squirrel" "roman_empire" "arxiv_year")


for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 10000 --pp-mode standardize --clf log_reg_sgd::grad_boost --num-reps 5 --dims 512
    done
done


# SNAP PATENTS
declare -a datasets=("snap_patents")

for a in "${algs[@]}"
do
    for d in "${datasets[@]}"
    do
        echo "$(date) Starting $a for dataset $d"
        python experiments/node_classification.py --dataset $d --methods $a --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 86400 --pp-mode standardize --clf log_reg_sgd::grad_boost --num-reps 1 --dims 256
    done
done