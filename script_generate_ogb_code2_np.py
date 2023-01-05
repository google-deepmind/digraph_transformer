# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generates the sorting network dataset."""
from collections import Counter
import os

from absl import app
from absl import flags
from absl import logging
import jraph
import numpy as np
from ogb.graphproppred import GraphPropPredDataset
import pandas as pd
from tqdm import tqdm

import ogb_utils
import dataflow_parser
from script_postcompute_eigedecomp import process_graph


_OUT_PATH = flags.DEFINE_string('out_path', './data/ogb/ogbg-code2-norev-df',
                                'The path to write datasets to.')
_RAW_PATH = flags.DEFINE_string('raw_path', './data/ogb',
                                'The path to write datasets to.')
_SHARD_SIZE = flags.DEFINE_integer(
    'shard_size', 10_000, 'The number of times to store in each file.')
_NUM_VOCAB = flags.DEFINE_integer('num_vocab', 5000,
                                  'the number of vocabulary used for sequence prediction')
_MAX_SEQ_LEN = flags.DEFINE_integer(
  'max_seq_len', 5, 'maximum sequence length to predict')
_DATA_FLOW = flags.DEFINE_bool(
  'data_flow', True, 'Data-flow centric graph construction')


maglap_configs = [
  dict(k=15, k_excl=0, q=0.1, q_absolute=False, norm_comps_sep=False,
       sign_rotate=False, use_symmetric_norm=True, l2_norm=True,
       exclude_cfg=False, exclude_ns=True)
]


def process_graph_(graph, config):
  config = config.copy()
  exclude_control_flow_edges = config.pop('exclude_cfg')
  exclude_next_syntax_edges = config.pop('exclude_ns')

  mask = np.ones_like(graph.edges["edge_type"], dtype=bool)
  if exclude_control_flow_edges:
    # These edges are the sequential edges of python_graphs
    mask = graph.edges["edge_type"] != 1
  if exclude_next_syntax_edges:
    # These edges are the original control flow edges of python_graphs
    mask = mask & (graph.edges["edge_type"] != 9)

  mask = mask.squeeze()

  senders = graph.senders[mask]
  receivers = graph.receivers[mask]
  n_dropped_edges = (~mask).sum()
  n_edge = graph.n_edge - n_dropped_edges

  graph_ = graph._replace(
      edges=None, senders=senders, receivers=receivers, n_edge=n_edge)

  return process_graph(graph_, **config)


def to_graphs_tuple(raw_graph, ogb_edge_types=True, max_token_length=1_023):
  """Converts the OGB Code 2 to a GraphsTuple."""
  if ogb_edge_types:
    edge_index, edge_attr = ogb_utils.augment_edge(
        raw_graph['edge_index'],
        raw_graph['node_is_attributed'],
        reverse=False)
  else:
    edge_index = raw_graph['edge_index']
    edge_attr = dict(edge_type=raw_graph['edge_type'],
                     edge_name=raw_graph['edge_name'],
                     edge_order=raw_graph['edge_order'])
  senders = edge_index[0]
  receivers = edge_index[1]
  nodes = dict(node_feat=raw_graph['node_feat'],
               node_depth=raw_graph['node_depth'])
  if 'node_feat_raw' in raw_graph:
    nodes['node_feat_raw'] = raw_graph['node_feat_raw']
  if 'node_feat_orig' in raw_graph:
    nodes['node_feat_orig'] = raw_graph['node_feat_orig']
  edges = edge_attr
  n_node = np.array([raw_graph['num_nodes']])
  n_edge = np.array([len(senders)])

  graph = jraph.GraphsTuple(
      nodes=nodes,
      edges=edges,
      senders=senders,
      receivers=receivers,
      n_node=n_node,
      n_edge=n_edge,
      globals=None)

  if graph.n_node <= max_token_length:
    precomputed = tuple(
        (config, *process_graph_(graph, config)) for config in maglap_configs)
  else:
    # To make sure the shapes are constant for tfds
    def dummy_eigenvec(k):
      return np.full((graph.nodes['node_feat'].shape[0], k),
                     np.nan, dtype=np.complex64)

    precomputed = tuple(
        (config,
         np.full(config['k'], np.nan, dtype=np.float32),
         dummy_eigenvec(config['k']))
        for config in maglap_configs)
  graph = graph._replace(globals=dict(eigendecomposition=precomputed))

  return graph


def encode_raw_target(target):
  return np.array([s.encode('utf-8') for s in target], dtype='object')


def main(*args, dataset_name='ogbg-code2', **kwargs):
  ds = GraphPropPredDataset(dataset_name, root=_RAW_PATH.value)

  vocab2idx, idx2vocab = ogb_utils.get_vocab_mapping(
      [ds[i][1] for i in ds.get_idx_split()['train']], _NUM_VOCAB.value)

  metadata = {
      'vocab2idx': vocab2idx,
      'idx2vocab': idx2vocab,
      # +2 to account for end of sequence and unknown token
      'num_vocab': str(_NUM_VOCAB.value + 2).encode('utf-8'),
      'max_seq_len': str(_MAX_SEQ_LEN.value).encode('utf-8')
  }

  if _DATA_FLOW.value:
    mapping_dir = os.path.join(_RAW_PATH.value, 'ogbg_code2', 'mapping')

    attr2idx = dict()
    for line in pd.read_csv(os.path.join(mapping_dir,
                                         'attridx2attr.csv.gz')).values:
      attr2idx[line[1]] = int(line[0])
    type2idx = dict()
    for line in pd.read_csv(os.path.join(mapping_dir,
                                         'typeidx2type.csv.gz')).values:
      type2idx[line[1]] = int(line[0])

    code_dict_file_path = os.path.join(mapping_dir, 'graphidx2code.json.gz')
    code_dict = pd.read_json(code_dict_file_path, orient='split')

  split_ids = ds.get_idx_split()

  if _DATA_FLOW.value:
    node_type_counter = Counter()
    node_value_counter = Counter()
    edge_name_counter = Counter()

    for split, ids in split_ids.items():
      file_path = os.path.join(_OUT_PATH.value, split)
      os.makedirs(file_path, exist_ok=True)
      os.makedirs(os.path.join(file_path, 'raw'), exist_ok=True)

      buffer = []
      start_id = ids[0]
      for id_ in tqdm(list(ids)):
        try:
          df_graph = dataflow_parser.py2ogbgraph(
              code_dict.iloc[id_].code, attr2idx, type2idx)[0]

          for node_type in df_graph['node_feat_raw'][:, 0]:
            node_type = node_type.decode('utf-8')
            node_type_counter[node_type] += 1
          for node_value in df_graph['node_feat_raw'][:, 1]:
            node_value = node_value.decode('utf-8')
            node_value_counter[node_value] += 1
          for edge_name in df_graph['edge_name'].squeeze():
            edge_name = edge_name.decode('utf-8')
            edge_name_counter[edge_name] += 1

          buffer.append((df_graph,
                         ogb_utils.encode_y_to_arr(
                             ds[id_][1], vocab2idx, _MAX_SEQ_LEN.value),
                         encode_raw_target(ds[id_][1]),
                         np.array(id_)))
        except:  # pylint: disable=bare-except
          print(f'Error for graph {id_}')
          print(code_dict.iloc[id_].code)

        if len(buffer) >= _SHARD_SIZE.value or id_ == ids[-1]:
          file_name = os.path.join(
              file_path, 'raw', f'{start_id}_{id_}_raw.npz')
          np.savez_compressed(file_name, data=np.array(buffer, dtype='object'))
          logging.info('Wrote %d to %s', len(buffer), file_name)
          buffer = []
          start_id = id_

    topk_node_values = node_value_counter.most_common(11_972)
    node_values_to_idx = {k: v for v, (k, _) in enumerate(topk_node_values)}

    meta = os.path.join(_OUT_PATH.value, 'train', 'meta.npz')
    metadata = np.load(meta, allow_pickle=True)["data"].item()

    node_values_to_idx = metadata['node_values_to_idx']
    node_type_to_idx = metadata['node_type_to_idx']
    edge_name_to_idx = metadata['edge_name_to_idx']

    def node_values_to_idx_with_default(value):
      if value in node_values_to_idx:
        return node_values_to_idx[value]
      return len(node_values_to_idx)

    node_type_to_idx = {
        k: v for v, (k, _) in enumerate(node_type_counter.most_common())}
    edge_name_to_idx = {
        k: v for v, (k, _) in enumerate(edge_name_counter.most_common())}

    metadata['node_values_to_idx'] = node_values_to_idx
    metadata['node_type_to_idx'] = node_type_to_idx
    metadata['edge_name_to_idx'] = edge_name_to_idx

    file = os.path.join(_OUT_PATH.value, 'meta.npz')
    np.savez_compressed(file, data=np.array(metadata, dtype='object'))
    logging.info('Wrote %s', file)

    for split in split_ids.keys():
      file_path = os.path.join(_OUT_PATH.value, split)

      files = os.listdir(os.path.join(file_path, 'raw'))
      for f in tqdm(files):
        if 'raw' not in f:
          continue

        buffer = []
        for graph, *remainder in np.load(
                os.path.join(file_path, 'raw', f), allow_pickle=True)["data"]:
          node_types = [
              node_type_to_idx[node_type.decode('utf-8')]
              for node_type in graph['node_feat_raw'][:, 0]
          ]
          node_values = [
              node_values_to_idx_with_default(node_value.decode('utf-8'))
              for node_value in graph['node_feat_raw'][:, 1]
          ]
          graph['node_feat_orig'] = graph['node_feat']
          graph['node_feat'] = np.array((node_types, node_values),
                                        dtype=np.int64).transpose()
          del graph['node_feat_raw']

          edge_names = [
              edge_name_to_idx[edge_name.decode('utf-8')]
              for edge_name in graph['edge_name'].squeeze()
          ]
          graph['edge_name'] = np.array(
            edge_names, dtype=np.int64)[:, None]

          graphs_tuple = to_graphs_tuple(graph, ogb_edge_types=False)
          buffer.append((graphs_tuple, *remainder))

        file_name = os.path.join(file_path, f.replace('_raw', ''))
        np.savez_compressed(file_name, data=np.array(buffer, dtype='object'))
        logging.info('Wrote %d to %s', len(buffer), file_name)

    return

  file = os.path.join(_OUT_PATH.value, 'meta.npz')
  np.savez_compressed(file, data=np.array(metadata, dtype='object'))
  logging.info('Wrote %s', file)

  for split, ids in split_ids.items():
    file_path = os.path.join(_OUT_PATH.value, split)

    buffer = []
    start_id = ids[0]
    for id_ in tqdm(list(ids)):
      graphs_tuple = to_graphs_tuple(ds[id_][0])
      buffer.append((
          graphs_tuple,
          ogb_utils.encode_y_to_arr(ds[id_][1], vocab2idx, _MAX_SEQ_LEN.value),
          encode_raw_target(ds[id_][1]),
          np.array(id_)))

      if len(buffer) >= _SHARD_SIZE.value or id_ == ids[-1]:
        file_name = os.path.join(file_path, f'{start_id}_{id_}.npz')
        np.savez_compressed(file_name, data=np.array(buffer, dtype='object'))
        logging.info('Wrote %d to %s', len(buffer), file_name)
        buffer = []
        start_id = id_


if __name__ == '__main__':
  app.run(main)
