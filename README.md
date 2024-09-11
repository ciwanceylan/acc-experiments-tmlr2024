# ACC experiments framework

This repository contains the graph experiments and benchmarking framework used for the paper "Full-Rank Unsupervised Node Embeddings for Directed
Graphs via Message Aggregation" submitted to TLMR 2024. See usage instructions below.

# Installation

### Install dependencies

You first need to have a [conda](https://docs.anaconda.com/miniconda/) installed on your system.
It is possible that [mamba](https://mamba.readthedocs.io/en/latest/index.html) will also work, but this has not been tested.

Then you need to install the packages listed in [the requirements file](required_packages.txt) into an conda environment 
named 'acc_neb_env'. You can do this conveniently by first using the provided [environment files](basic_environment_cu118.yml):
```commandline
conda env create --file basic_environment_${CUDA}.yml
```
where `${CUDA}` is either `cu118` for GPU, or `cpu` for only CPU support.
Then, activate the environment, and install the following dependencies via pip:
```commandline
conda activate acc_neb_env
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/${CUDA}
pip install torch_geometric
pip install pyg_lib==0.3.1 torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/${CUDA}/repo.html
pip install ogb==1.3.6
```

To run [clustering_rank_deficiency_noise_demo.py](experiments/clustering_rank_deficiency_noise_demo.py) and 
[squirrel_analysis.py](experiments/squirrel_analysis.py), you also need to install [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) and [umap](https://umap-learn.readthedocs.io/en/latest/).


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


# Datasets

All the graph alignment datasets are available under `data/inrepo`. To index these datasets and to download the node 
classification datasets, please run
```commandline
python index_and_download_datasets.py
```

Snap Patents has to be downloaded manually from here [LINK](https://github.com/CUAI/Non-Homophily-Large-Scale?tab=readme-ov-file#dataset-preparation).
Once this is done, you can prepare and index it by providing the path to the downloaded .mat file to the dataset preparation script:
```commandline
python index_and_download_datasets.py --snap-patents-path <path/to/snap_patents.mat>
```


# Troubleshooting

You may encounder errors along the lines of
```
ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found.
```
or
```commandline
OSError: libcusparse.so.11: cannot open shared object file: No such file or directory.
```
Both of these can be fixed by updating the LD_LIBRARY_PATH
```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib
```
If you are running the code via PyCharm, you might also need to set this path inside the IDE.


In some setup, for example in Google Cloud, `conda` is not a valid command for interacting with conda from subprocesses
from within python. I'm not sure why this is, but a fix is do the following replacement on Line...
```diff
- conda_command = f"conda run -n {alg.spec.env_name}".split()
+ conda_command = f"__conda_exe run -n {alg.spec.env_name}".split()
```

