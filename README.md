# Transformers Meet Directed Graphs

This repository contains the accompanying code for the paper Transformers Meet
Directed Graphs.

For external resources (not published by DeepMind), please see the 
[homepage of TU Munich](https://www.cs.cit.tum.de/daml/digraph-transformer/).

*Disclaimer: The provided scripts are for illustrative purposes. Due to the
single-machine setup the scalability is limited.*

## Setup

You can set up Python virtual environment (you might need to install the
`python3-venv` package first) with all needed dependencies inside the forked
`deepmind_research` repository using:

```bash
python3 -m venv /tmp/digraph_transformer
source /tmp/digraph_transformer/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt
```

Note that you will also need the headers of libtbb and graphviz:

```
sudo apt install libtbb-dev graphviz-dev
```

The default configurations for the eigenvectors of the (Magnetic) Laplacian are
being precomputed. For all other configurations, the dataloader will use
numba's multithreading capabilities for the eigendecomposition. This might not
be desired if using a multi-threaded backend for numpy.

## Generate datasets

We provide utility scripts to generate the used datasets, where `DATA_PATH` is
a local storage location of your choice:
- Positional encodings playground: `/bin/bash run_sn_dataset_gen.sh -r ${DATA_PATH}`
- Sorting network: `/bin/bash run_sn_dataset_gen.sh -r ${DATA_PATH}`
- OGB Code2 dataset: `/bin/bash run_ogb_dataset_gen.sh -r ${DATA_PATH}`

Note that `distance` is used as an identifyer for the positional encodings
playground in the code base. Moreover, note that loading the
`graphidx2code.json.gz` provided by OGB will require a
substantial amount of main memory (> 50 GB).

## Train models

For training the models with Magnetic Laplacian positional encodings you
may use the commands detailed in this sections. For other models you mainly
need to adjust the `maglappos` preset.

You may monitor the run via the tensorboard logs in the respective checkpoint
directory.

### Positional Encodings Playground

```bash
python experiment.py \
    --jaxline_mode=train_eval_multithreaded \
    --config=./config.py:${TASK},bignn,maglappos,bbs \
    --config.experiment_kwargs.config.data_root=${DATA_PATH}/distance \
    --config.random_seed=1000 \
    --config.checkpoint_dir=${DATA_PATH}/distance/checkpoints/ \
    --config.experiment_kwargs.config.model.gnn_config.se=
```

`TASK` is the respective task of the playground (e.g., `export TASK=adj`):
- `adj`/`adja` predicting the adjacency for regular/acyclic graphs
- `con`/`cona` predicting the connectedness for regular/acyclic graphs
- `dist`/`dista` directed distance for regular/acyclic graphs
- `distu`/`distau` undirected distance for regular/acyclic graphs

### Sorting Network

```bash
python experiment.py \
    --jaxline_mode=train_eval_multithreaded \
    --config=./config.py:${TASK},bignn,maglappos,bbs \
    --config.experiment_kwargs.config.data_root=${DATA_PATH}/sorting_network \
    --config.random_seed=1000 \
    --config.checkpoint_dir=${DATA_PATH}/sorting_network/checkpoints/ \
    --config.experiment_kwargs.config.model.gnn_config.se=
```

### OGB Code2

```bash
python experiment.py \
    --jaxline_mode=train_eval_multithreaded \
    --config=./config.py:${TASK},bignn,maglappos,bbs \
    --config.experiment_kwargs.config.data_root=${DATA_PATH}/ogb \
    --config.random_seed=1000 \
    --config.checkpoint_dir=${DATA_PATH}/ogb/checkpoints/ \
```

## Evaluating pretrained models

To run, e.g., pretrained models for OGB you may run

```bash
python ./experiment.py \
    --jaxline_mode="eval" \
    --config="./config.py:dfogb,bignn,maglappos,bbs" \
    --config.restore_path=${RESTORE_PATH} \
    --config.experiment_kwargs.config.data_root=${DATA_ROOT}/ogb \
    --config.one_off_evaluate=True
```

where `RESTORE_PATH` is the path to the model's folder (i.e., the parent of
the `*.dill` file). `DATA_ROOT` is the location of the preprocessed data.

## Citation

To cite this work:
```latex
@inproceedings{deepmind2023digraph_transformer,
  author = {Geisler, Simon and Li, Yujia and Mankowitz, Daniel and Cemgil, Taylan and G\"unnemann, Stephan and Paduraru, Cosmin},
  title = {Transformers Meet Directed Graphs},
  year = {2023},
  booktitle = {International Conference on Machine Learning, {ICML}},
}
```

## License and disclaimer

Copyright 2022 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
