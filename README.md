# FreeHGC: Training-free Heterogeneous Graph Condensation via Data Selection

## Requirements

## Our environment configuration
* python          3.8
* torch           1.12.1+cu113
* torch_geometric 2.1.0
* torch_sparse    0.6.15
* torch_scatter   2.0.9
* numpy           1.21.5

## Data preparation

For experiments in Motivation section and on four medium-scale datasets, please download datasets `DBLP.zip`, `ACM.zip`, `IMDB.zip`, `Freebase.zip` from [the source of HGB benchmark](https://cloud.tsinghua.edu.cn/d/a2728e52cd4943efa389/), and extract content from these compresesed files under the folder `'./data/'`.

For experiments on the large dataset AMiner, The dataset will be downloaded automatically. If the download fails, you can view the source code of `torch_geometric.datasets` and update the url.

## Run
For medium-scale datasets:

`python train_hgb.py --dataset ACM --method FreeHGC --reduction-rate 0.1 --pr 0.95 --gpu 0 --num-hops 3 --num-hidden 128 --lr 0.001 --dropout 0.5 --ff-layer-2 2 --ACM-keep-F`

For large-scale dataset:

`python train_ogbn_pr.py --dataset aminer --method FreeHGC --reduction-rate 0.05 --num-hops 2`
