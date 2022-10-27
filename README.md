# Transformers Meet Directed Graphs

This repository contains the accompanying code for the paper Transformers Meet
Directed Graphs.

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

## Sorting Networks

We now explain how to reproduce the results using the transformer with magnetic
laplacian positional encodings on the sorting networks dataset.

*Disclaimer: This script is provided for illustrative purposes. Due to the
single machine setup the scalability is limited.*

### Generate dataset

Where `DATA_PATH` is a local storage location of your choice.

To generate the dataset, please run:

```bash
/bin/bash run_sn_dataset_gen.sh -r ${DATA_PATH}
```

### Training the model

After dataset generation, to train a model, please run:

```bash
/bin/bash run_sn_training.sh -r ${DATA_PATH}
```

### Evaluating the model

With a trained model, please run:

```bash
/bin/bash run_sn_eval.sh -r ${DATA_PATH}  -c ${MODEL_PATH}
```
```

For example, our `MODEL_PATH` was `"~/checkpoints/models/best"` (i.e. pointing
to a folder containing a `checkpoint.dill` file).

## Open Graph Benchmark Code2

To generate the dataset with the data-flow centric graph construction, we
provide the `dataflow_parser.py`. This parser can be used to convert python
source code into graphs (source code is provided by OGB in
`<ogb_root>/ogbg_code2/mapping/graphidx2code.json`). To run this module you
also need to install `pip3 install python-graphs` (at the time of usage there
was an issue with bare "*" as function parameters that we fixed manually).

As you can see from the configuration, we took the 11,973 most frequent
attributes (all words that appear at least 100 times in the dataset). All other
node/edge attributes/features are exhaustively mapped to tokens.

You first need to convert the dataset (see
`script_generate_sorting_network_np.py` for a possible template). Thereafter,
you can use to train the model using with eigenvectors of the Magnetic
Laplacian as positional encodings:
```bash
python "${SCRIPT_DIR}"/experiment.py \ --jaxline_mode="train" \
--config="${SCRIPT_DIR}/config.py:dfogb,bignn,maglappos,bbs" \
--config.random_seed=${MODEL_SEED} \ --config.checkpoint_dir=${CHECKPOINT_DIR} \
--config.experiment_kwargs.config.data_root=${DATA_ROOT}
```
For more details see also also `run_sn_training.sh`.

## Citation

To cite this work:

```latex
@article{deepmind2022magnetic_laplacian,
  author = {Geisler, Simon and Paduraru, Cosmin and Li, Yujia and Cemgil, Taylan},
  title = {Transformers Meet Directed Graphs},
  year = {2022},
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
