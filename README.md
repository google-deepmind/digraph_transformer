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

For training the models, we provide convenient [hydra](https://hydra.cc/)
configurations in `./conf`.

For training the models with Magnetic Laplacian positional encodings you
may use:
- Positional encodings playground: `python hydra_runner.py -m +task=adj,adja,con,cona,dist,dista,distu,distau +model=maglap` to sweep over the tasks
  - `adj`/`adja` predicting the adjacency for regular and acyclic graphs
  - `con`/`cona` predicting the connectedness for regular and acyclic graphs
  - `dist`/`dista` directed distance for regular and acyclic graphs
  - `distu`/`distau` undirected distance for regular and acyclic graphs
- Sorting network: `python hydra_runner.py -m +task=sn +model=maglap`
- OGB Code2 dataset: `python hydra_runner.py -m +task=dfogb +model=maglap`

See `conf/reruns` for configurations about randomized reruns. For OGB Code2 we
used `conf/reruns/10seed.yaml`.

If you have weights and biases (`wandb`) installed, the experiment will
automatically log results there. Otherwise, see the tensorboard logs.

### Evaluating pretrained models

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
