#!/usr/bin/env bash

python experiments/embedding_info_analysis.py  --dataset arenas --weighted 0 --node-attributed 1 --undirected 1 --methods cikm_acc_pca_svs --dims 512
python experiments/embedding_info_analysis.py  --dataset ppi_small --weighted 0 --node-attributed 1 --undirected 1 --methods cikm_acc_pca_svs --dims 512
python experiments/embedding_info_analysis.py  --dataset magna --weighted 0 --node-attributed 1 --undirected 1 --methods cikm_acc_pca_svs --dims 512
python experiments/embedding_info_analysis.py  --dataset enron_na --weighted 0 --node-attributed 1 --undirected 0 --methods cikm_acc_pca_svs --dims 512
python experiments/embedding_info_analysis.py  --dataset polblogs --weighted 0 --node-attributed 1 --undirected 0 --methods cikm_acc_pca_svs --dims 512
