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
"""Dataset utility functions for the graph property prediction task."""
import functools
import multiprocessing
import os
from typing import Any, Dict, Optional, Sequence

from absl import logging
import jax
import jax.numpy as jnp
import jraph
import numba
import numpy as np
import tensorflow as tf

# pylint: disable=g-bad-import-order
import config
import utils

# This param should be chosen to be reasonable for the pipeline/hardware
NUM_THREADS_EIGENVECTORS = 96


@jax.curry(jax.tree_map)
def _downcast_ints(x):
  if x.dtype == tf.int64:
    return tf.cast(x, tf.int32)
  return x


@jax.curry(jax.tree_map)
def _make_writable(x):
  if not x.flags["WRITEABLE"]:
    return np.array(x)
  return x


def _add_target_to_globals(dataset_config: config.Dataset,
                           graph: jraph.GraphsTuple, target: tf.Tensor,
                           target_raw: Optional[tf.Tensor],
                           graph_index: Optional[int], is_training: bool):
  """Adds the labels to globals of the graph for convenience."""
  def set_global_shape(x):
    if dataset_config.sequence_length < 0:
      return x
    return tf.ensure_shape(x, [1, dataset_config.sequence_length])

  globals_dict = {
      "target": set_global_shape(tf.expand_dims(target, axis=0)),
      **(graph.globals if isinstance(graph.globals, dict) else {})
  }
  if graph_index is not None:
    globals_dict["graph_index"] = tf.expand_dims(graph_index, axis=0)
  if not is_training and target_raw is not None:
    globals_dict["target_raw"] = tf.expand_dims(target_raw, axis=0)

  return graph._replace(globals=globals_dict)


def _pad_to_next_power_of_two(graph: jraph.GraphsTuple,
                              cfg: config.Dataset) -> jraph.GraphsTuple:
  """Pads GraphsTuple nodes and edges to next power of two."""
  graph = _make_writable(graph)

  n_node_max = graph.n_node.max()
  n_edge_max = graph.n_edge.max()
  n_node = np.power(2, np.ceil(np.log2(n_node_max + 1)))
  n_edge = np.power(2, np.ceil(np.log2(n_edge_max + 1)))

  # Handle a trailing dimension of size 1
  if len(graph.n_edge.shape) == 3 and graph.n_edge.shape[-1] == 1:
    graph = graph._replace(n_edge=graph.n_edge[..., 0])

  batch = graph.n_node.shape[0]

  pad_n_node = int(n_node - n_node_max)
  pad_n_edge = int(n_edge - n_edge_max)
  pad_n_empty_graph = 1

  def pad(leaf, n_pad: int):
    padding = np.zeros(
        (
            batch,
            n_pad,
        ) + leaf.shape[2:], dtype=leaf.dtype)
    padded = np.concatenate([leaf, padding], axis=1)
    return padded

  if cfg.name.startswith('dist'):
    tree_nodes_pad = functools.partial(pad, n_pad=int(n_node))
    tree_edges_pad = functools.partial(pad, n_pad=int(n_edge))

    def tree_globs_pad(globals_):
      if globals_.shape[-1] == 1:
        return pad(globals_, n_pad=pad_n_empty_graph)

      indices = jnp.tile(jnp.arange(n_node), (batch, 1))
      padding_mask = indices < graph.n_node[:, 0, None]
      padding_mask = padding_mask[:, :, None] * padding_mask[:, None, :]

      if cfg.num_classes > 0 and not cfg.name.endswith('con'):
        # adj = np.zeros((batch, int(n_node), int(n_node)), dtype=np.int32)
        adj = np.full((batch, int(n_node), int(n_node)), -1, dtype=np.int32)

        adj[padding_mask] = 0

        batch_idx = np.arange(batch)[:, None]
        batch_idx = np.tile(batch_idx, (1, graph.senders.shape[-1]))
        adj[batch_idx, graph.senders, graph.receivers] = 1

        adj[:, jnp.arange(int(n_node)), jnp.arange(int(n_node))] = -1
        return adj
      else:
        sq_globals = np.full((batch, int(n_node), int(n_node)), -1.)

        for idx_batch, nb_nodes in enumerate(graph.n_node[:, 0]):
          idx = np.arange(nb_nodes)
          idx = np.stack(np.meshgrid(idx, idx)).reshape((2, -1))
          sq_globals[idx_batch, idx[0], idx[1]] = globals_[
              idx_batch, 0, :int(nb_nodes ** 2)]

        sq_globals[:, np.arange(int(n_node)), np.arange(int(n_node))] = -1
        if cfg.name.endswith('con'):
          sq_globals[~np.isfinite(sq_globals)] = 0
          sq_globals[sq_globals > 0] = 1
        else:
          sq_globals[~np.isfinite(sq_globals)] = -1
        return sq_globals
  else:
    tree_nodes_pad = functools.partial(pad, n_pad=pad_n_node)
    tree_edges_pad = functools.partial(pad, n_pad=pad_n_edge)

    def tree_globs_pad(globals_):
      return pad(globals_, n_pad=pad_n_empty_graph)

  # Correct zero padding of senders and receivers
  edge_pad_idx = np.tile(
      np.arange(graph.senders.shape[1]),
      (graph.senders.shape[0], 1)) >= graph.n_edge
  senders = graph.senders
  senders[edge_pad_idx] = -1
  receivers = graph.receivers
  receivers[edge_pad_idx] = -1

  # Only OGB hast edge features, while Sorting Networks has not
  edges = graph.edges
  if isinstance(edges, dict) or (edges.shape[-1] > 1):
    edges = jax.tree_map(tree_edges_pad, edges)

  padded_graph = jraph.GraphsTuple(
      n_node=np.concatenate([graph.n_node, n_node - graph.n_node], axis=1),
      n_edge=np.concatenate([graph.n_edge, n_edge - graph.n_edge], axis=1),
      nodes=jax.tree_map(tree_nodes_pad, graph.nodes),
      edges=edges,
      globals=jax.tree_map(tree_globs_pad, graph.globals),
      senders=np.concatenate(
          [senders, np.full([batch, pad_n_edge], -1, dtype=np.int32)], axis=1),
      receivers=np.concatenate(
          [receivers,
           np.full([batch, pad_n_edge], -1, dtype=np.int32)], axis=1))
  return padded_graph


def build_dataset_iterator(dataset_config: config.Dataset,
                           ds: tf.data.Dataset,
                           batch_size: int,
                           debug: bool = False,
                           do_bucket_by_size: bool = False,
                           bucket_boundaries: Sequence[int] = (255, 511),
                           bucket_batch_size_factors: Sequence[int] = (
                             4, 2, 1),
                           is_training: bool = True,
                           max_token_length: int = 1023,
                           max_number_of_instances: int = -1,
                           exclude_control_flow_edges: bool = True,
                           exclude_next_syntax_edges: bool = False,
                           num_parallel_batchers: Optional[int] = None,
                           posenc_config: Optional[Dict[str, Any]] = None):
  """Creates a dataset generator and does the important preprocessing steps."""
  num_local_devices = jax.local_device_count()

  if debug:
    max_items_to_read_from_dataset = int(num_local_devices * batch_size)
    prefetch_buffer_size = 1
    shuffle_buffer_size = 1
    num_parallel_batchers = 1
    drop_remainder = False
  else:
    max_items_to_read_from_dataset = -1  # < 0 means no limit.
    prefetch_buffer_size = 64
    # It can take a while to fill the shuffle buffer with k fold splits.
    shuffle_buffer_size = int(1e6)
    if is_training and num_parallel_batchers is None:
      num_parallel_batchers = 4
    drop_remainder = is_training

  ds = ds.filter(
      lambda graph, *args: tf.math.reduce_any(graph.n_node < max_token_length))

  if is_training:
    logging.info("Shard dataset %d / %d",
                 jax.process_index() + 1, jax.process_count())
    ds = ds.shard(jax.process_count(), jax.process_index())

  ds = ds.take(max_items_to_read_from_dataset)
  ds = ds.cache()
  if is_training:
    ds = ds.shuffle(shuffle_buffer_size)

  # Only take a random subset (must be after shuffle)
  if max_number_of_instances > 0:
    ds = ds.take(max_number_of_instances)

  def map_fn(graph, target, target_raw=None, graph_index=None):
    graph = _add_target_to_globals(dataset_config, graph, target, target_raw,
                                   graph_index, is_training)
    graph = _downcast_ints(graph)

    if "-raw" in dataset_config.name:
      graph.nodes["node_feat"] = graph.nodes["node_feat_orig"]

    if "-df" in dataset_config.name:
      mask = tf.ones_like(graph.edges["edge_type"], dtype=tf.bool)
      if exclude_control_flow_edges:
        # These edges are our data-flow centric control flow edges
        mask = graph.edges["edge_type"] != 1
      if exclude_next_syntax_edges:
        # These edges are the original control flow edges of python_graphs
        mask = mask & (graph.edges["edge_type"] != 9)

      edges = jax.tree_util.tree_map(lambda x: tf.boolean_mask(x, mask, axis=0),
                                     graph.edges)
      senders = tf.boolean_mask(graph.senders, mask[:, 0], axis=0)
      receivers = tf.boolean_mask(graph.receivers, mask[:, 0], axis=0)

      n_dropped_edges = tf.reduce_sum(
          tf.cast(tf.math.logical_not(mask), tf.int32))
      n_edge = graph.n_edge - n_dropped_edges

      graph = graph._replace(
          edges=edges, senders=senders, receivers=receivers, n_edge=n_edge)

    return graph

  ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

  ### Explicit static batching due to self-attention over nodes. ###
  if not do_bucket_by_size or not is_training:
    ds = ds.padded_batch(
        num_local_devices * batch_size, drop_remainder=drop_remainder)
  else:
    full_batch_size = num_local_devices * batch_size
    ds = ds.bucket_by_sequence_length(
        lambda graph: tf.reduce_max(graph.n_node),
        bucket_boundaries=bucket_boundaries,
        # This is a rather pessimistic linear scaling
        bucket_batch_sizes=[
            factor * full_batch_size for factor in bucket_batch_size_factors
        ],
        drop_remainder=drop_remainder)

  if is_training:
    ds = ds.repeat()
  ds = ds.prefetch(prefetch_buffer_size)  # tf.data.experimental.AUTOTUNE)

  def reshape(leaf):
    return leaf.reshape((num_local_devices,
                         leaf.shape[0] // num_local_devices) + leaf.shape[1:])

  calc_eigenvals_and_vecs = (
      posenc_config is not None and
      posenc_config.get("posenc_type", "") == "maglap")
  if calc_eigenvals_and_vecs:
    eigv_magnetic_config = dict(
        k=posenc_config.get("top_k_eigenvectors", 5),
        k_excl=posenc_config.get("excl_k_eigenvectors", 1),
        q=posenc_config.get("maglap_q", 0.25),
        q_absolute=posenc_config.get("maglap_q_absolute", True),
        use_symmetric_norm=posenc_config.get("maglap_symmetric_norm", False),
        norm_comps_sep=posenc_config.get("maglap_norm_comps_sep", False),
        sign_rotate=posenc_config.get("maglap_sign_rotate", True))
    eigv_magnetic_laplacian = functools.partial(
        utils.eigv_magnetic_laplacian_numba_batch, **eigv_magnetic_config)
    if "-df" in dataset_config.name:
      eigv_magnetic_config['l2_norm'] = posenc_config.get("maglap_l2_norm",
                                                          True)
      eigv_magnetic_config['exclude_cfg'] = exclude_control_flow_edges
      eigv_magnetic_config['exclude_ns'] = exclude_next_syntax_edges

    numba.set_num_threads(
        min(NUM_THREADS_EIGENVECTORS, numba.get_num_threads()))
    logging.info("Numba uses %d threads for eigenvector calculation",
                 numba.get_num_threads())
    logging.info("Number of cores %d", multiprocessing.cpu_count())

  for sample in ds.as_numpy_iterator():

    # Precomputed eigenvectors need to be added prior to final padding
    eigenvalues, eigenvectors = None, None
    if 'eigendecomposition' in sample.globals:
      eigendecomposition = sample.globals.pop('eigendecomposition')
      if calc_eigenvals_and_vecs:
        for entry, val, vec in eigendecomposition:
          # Assuming that the elements are always in the same order
          entry = jax.tree_map(lambda x: x[0], entry)
          if all([k in entry and np.isclose(v, entry[k])
                  for k, v in eigv_magnetic_config.items()]):
            eigenvalues, eigenvectors = val, vec

        if eigenvectors is not None:
          if not isinstance(sample.nodes, dict):
            sample = sample._replace(nodes={"node_feat": sample.nodes})
          sample.nodes["eigenvectors"] = eigenvectors

    # Doing these preprocessing here is not optimal, however, it is a simple
    # approach to maneuvers around some limitations of TFDS etc.
    sample = _pad_to_next_power_of_two(sample, dataset_config)

    if calc_eigenvals_and_vecs and eigenvalues is None:
      logging.debug("Start eigenvalues and vectors calculation")
      try:
        eigenvalues, eigenvectors = eigv_magnetic_laplacian(
            sample.senders, sample.receivers, sample.n_node)
      except:
        logging.warning("MagLap calculation error in %s",
                        str(sample.globals["graph_index"]))
        raise
      eigenvalues = eigenvalues.astype(np.float32)
      if np.iscomplexobj(eigenvectors):
        eigenvectors = eigenvectors.astype(np.complex64)
      else:
        eigenvectors = eigenvectors.astype(np.float32)

      if not isinstance(sample.nodes, dict):
        sample = sample._replace(nodes={"node_feat": sample.nodes})
      sample.nodes["eigenvectors"] = eigenvectors
      # This is not accurate, but globals are treated differently
      sample.nodes["eigenvalues"] = eigenvalues
      logging.debug("Finished eigenvalues and vectors calculation")

    if calc_eigenvals_and_vecs:
      # This is not accurate, but globals are treated differently later on
      sample.nodes["eigenvalues"] = eigenvalues

    sample = jax.tree_map(
        lambda x: jnp.array(x) if x.dtype != object else x, sample)
    if jax.device_count() > 1 and is_training:
      sample = jax.tree_map(reshape, sample)
    yield sample


def dataset_generator(path: str, split: str):
  """Loads the data from a folder stored as `npz` files, assuming the data is located in `$path/$split`.
  """
  base_path = os.path.join(path, split)
  for file in os.listdir(base_path):
    if not file.endswith('.npz') or 'meta' in file:
      continue
    for instance in np.load(
            os.path.join(base_path, file), allow_pickle=True)["data"]:
      yield tuple(instance)
