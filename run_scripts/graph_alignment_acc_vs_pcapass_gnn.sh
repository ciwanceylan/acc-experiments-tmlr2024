#!/usr/bin/env bash

python experiments/graph_alignment.py --dataset arenas --methods acc_gnn_ga_steps pcapass_gnn_ga_steps --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset ppi --methods acc_gnn_ga_steps pcapass_gnn_ga_steps --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset polblogs --methods acc_gnn_ga_steps pcapass_gnn_ga_steps --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset enron_na --methods acc_gnn_ga_steps pcapass_gnn_ga_steps --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512

python experiments/magna_graph_alignment.py --dataset magna --methods acc_gnn_ga_steps pcapass_gnn_ga_steps --undirected 1 --seed 23423 --timeout 3600 --noise-level 15 --pp-mode whiten --dims 512
