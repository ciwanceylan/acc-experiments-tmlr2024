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

### Install dependencies

You first need to have a [conda](https://docs.anaconda.com/miniconda/) installed on your system.
Possibly [mamba](https://mamba.readthedocs.io/en/latest/index.html) will also work, but this has not been tested.

Then you need to install the packages listed in [the requirements file](required_packages.txt) into an conda environment 
named 'acc_neb_env'.
The most convenient way to do this, and to use the same package versions as in our paper, is to first use the provided [environment file](basic_environment.yml):
```commandline
conda env create --file basic_environment.yml
```
Then, activate the environment, and install the following dependencies via pip:
```commandline
conda activate acc_neb_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/${CUDA}
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/${CUDA}/repo.html
```
where `${CUDA}` is either `cu118` for GPU, or `cpu` for only CPU support.

### Install framework
To install the framework, while standing in the top repo directory, run
```commandline
pip install -e structfeatures/
pip install -e .
```

### Install ACC
Clone ACC from <URL> into a suitable location, and then install ACC using
```commandline
pip install -e <path/to/ACC-repo>
```


### Troubleshooting

If you see an error along the lines of
```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found.
```

Both pytorch geometric and DGL may raise errors related to a bad LD_LIBRARY_PATH.


To fix these, try
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

In some setup, for example in Google Cloud, `conda` is not a valid command for interacting with conda from subprocesses
from within python. I'm not sure why this is, but a fix is do the following replacement on Line...
```python
conda_command = f"conda run -n {alg.spec.env_name}".split()
conda_command = f"__conda_exe run -n {alg.spec.env_name}".split()
```

