# ACC experiments framework

This repository contains the graph experiments and benchmarking framework used for the paper "Full-Rank Unsupervised Node Embeddings for Directed
Graphs via Message Aggregation" submitted to TLMR 2024. See usage instructions below.


## Cloning without LFS files
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/ciwanceylan/node-embedding-benchmarks.git
```

For windows 
```cmd
set GIT_LFS_SKIP_SMUDGE=1  
git clone https://github.com/ciwanceylan/node-embedding-benchmarks.git
```

# Installation

#### CPU installation
```bash
conda env create --file environment_cpu_full_env.yml
conda activate neb_main_cpu_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/repo.html
pip install GitPython
```

#### GPU installation
```bash
conda env create --file environment_cuda_full_env.yml
conda activate neb_main_cuda_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html
pip install GitPython
```


### Troubleshooting

Both pytorch geometric and DGL may raise errors related to a bad LD_LIBRARY_PATH.
To fix these, try
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```


