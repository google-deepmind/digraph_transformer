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
"""All the important shortcuts to ease configuration of experiments."""

from typing import Text, List
from ml_collections import config_dict


def apply_presets(config: config_dict.ConfigDict,
                  unique_presets: List[Text]) -> config_dict.ConfigDict:
  """Applies the defined presets."""

  all_presets = {
      # Select dataset
      'norevogb': norevogb_preset,
      'dfogb': dfogb_preset,
      'sn': sorting_network_preset,
      'adj': adjacency_preset,
      'adja': adjacency_acyclic_preset,
      'adju': adjacency_undirected_preset,
      'con': is_connected_preset,
      'cona': is_connected_acyclic_preset,
      'conu': is_connected_undirected_preset,
      'dist': distance_preset,
      'dista': distance_acyclic_preset,
      'distu': distance_undirected_preset,
      'distau': distance_acyclic_undirected_preset,
      # Basic architecture and positional encodings
      'bignn': bignn_preset,
      'pprpos': ppr_posenc_preset,
      'rwpos': random_walk_posenc_preset,
      'maglappos': magnetic_laplacian_preset,
      'lappos': laplacian_preset,
      'bbs': bucket_by_size_preset
  }

  for preset in unique_presets:
    if preset in all_presets:
      all_presets[preset](config)
    else:
      raise ValueError(f'Invalid preset value `{preset}`')

  return config


def norevogb_preset(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the vanilla OGB (without reversed edges)."""
  config_ = config.experiment_kwargs.config

  config_.dataset_name = 'ogbg-code2-norev'

  # General hyperparams (With tuning batch size 56)
  config_.optimizer.use_agc = True
  config_.optimizer.optimizer_kwargs.weight_decay = 6e-5
  config_.optimizer.optimizer_kwargs.b1 = 0.9
  config_.optimizer.optimizer_kwargs.b2 = 0.95
  config_.optimizer.lr_schedule.peak_value = 3e-3 / 56
  config_.optimizer.lr_schedule.init_value = 5e-6 / 56
  config_.optimizer.lr_schedule.end_value = 6e-9 / 56
  config_.optimizer.agc_kwargs.clipping = 0.05

  # Model params
  config_.model.attention_config.with_attn_dropout = False
  config_.model.activation = 'gelu'
  config_.model.attention_config.with_bias = True
  config_.model.global_readout = 'cls'
  config_.model.encoder_config.attribute_dropout = 0.15
  config_.model.attention_config.dropout_p = 0.18
  config_.model.gnn_config.bidirectional = True
  config_.model.gnn_config.k_hop = 3
  config_.model.gnn_config.mlp_layers = 1

  # Offset
  config_.evaluation.unk_offset = 0.75
  config_.evaluation.eos_offset = 0.3
  return config


def dfogb_preset(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the data-flow centric OGB (without reversed edges)."""
  config_ = config.experiment_kwargs.config

  config_.dataset_name = 'ogbg-code2-norev-df'
  config_.dataset_config.exclude_control_flow_edges = False
  config_.dataset_config.exclude_next_syntax_edges = True

  # General hyperparams (With tuning batch size 56)
  config_.optimizer.use_agc = True
  config_.optimizer.optimizer_kwargs.weight_decay = 7.5e-5
  config_.optimizer.optimizer_kwargs.b1 = 0.75
  config_.optimizer.optimizer_kwargs.b2 = 0.935
  config_.optimizer.lr_schedule.peak_value = 3e-3 / 56
  config_.optimizer.lr_schedule.init_value = 4.5e-6 / 56
  config_.optimizer.lr_schedule.end_value = 6.5e-9 / 56
  config_.optimizer.agc_kwargs.clipping = 0.1

  # Model params
  config_.model.attention_config.with_attn_dropout = False
  config_.model.activation = 'gelu'
  config_.model.attention_config.with_bias = True
  config_.model.global_readout = 'cls'
  config_.model.encoder_config.attribute_dropout = 0.15
  config_.model.attention_config.dropout_p = 0.185
  config_.model.gnn_config.bidirectional = True
  config_.model.gnn_config.k_hop = 3
  config_.model.gnn_config.mlp_layers = 1

  # offset
  config_.evaluation.unk_offset = 0.75
  config_.evaluation.eos_offset = 0.45

  # Adaptions to save memory in comparison to TPU setup
  config_.optimizer.accumulate_gradient_k = 12
  config_.training.batch_size = 32
  # config_.evaluation.batch_size = 32
  return config


def sorting_network_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = dfogb_preset(config)

  # Reverse adaptions to save memory in comparison to TPU setup
  config.experiment_kwargs.config.optimizer.accumulate_gradient_k = 8
  config.experiment_kwargs.config.training.batch_size = 48
  # config.experiment_kwargs.config.evaluation.batch_size = 48

  config.epochs = 15

  config.experiment_kwargs.config.dataset_name = 'sn-7to11-12-13to16'
  config.experiment_kwargs.config.model.gnn_config.use_edge_attr = False
  config.experiment_kwargs.config.evaluation.max_number_of_instances = 40_000
  config.best_model_eval_metric = 'accuracy'
  config.experiment_kwargs.config.optimizer.agc_kwargs.clipping = 0.075
  config.experiment_kwargs.config.optimizer.optimizer_kwargs.weight_decay = 6e-5
  config.experiment_kwargs.config.optimizer.optimizer_kwargs.b1 = 0.7
  config.experiment_kwargs.config.optimizer.optimizer_kwargs.b2 = 0.9
  config.experiment_kwargs.config.optimizer.lr_schedule.peak_value = 4e-4 / 48
  config.experiment_kwargs.config.optimizer.lr_schedule.init_value = 2e-6 / 48
  config.experiment_kwargs.config.optimizer.lr_schedule.end_value = 2e-9 / 48

  config.experiment_kwargs.config.model.deg_sensitive_residual = False
  config.experiment_kwargs.config.model.posenc_config.exclude_canonical = True

  # offset
  config.experiment_kwargs.config.evaluation.unk_offset = 0
  config.experiment_kwargs.config.evaluation.eos_offset = 0
  return config


def adjacency_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = sorting_network_preset(config)

  config.save_checkpoint_interval = 240

  config.epochs = 15  # Due to the more aggressive bucketing, this is roughly 30
  config_ = config.experiment_kwargs.config

  config_.dataset_name = 'adj_c'
  config_.evaluation.max_number_of_instances = -1
  config.best_model_eval_metric = 'f1'

  config_.optimizer.accumulate_gradient_k = 1

  config_.model.posenc_config.exclude_canonical = True
  # config_.model.global_readout = 'sum_n'

  config_.dataset_config.bucket_boundaries = (127, 255, 511)
  config_.dataset_config.bucket_batch_size_factors = (8, 4, 2, 1)

  # Offset - will be auto tuned
  # config_.training.batch_size = 24
  config_.evaluation.unk_offset = 0.0
  config_.evaluation.eos_offset = 0.0

  config_.evaluation.batch_size = 128
  # config_.evaluation.batch_size = 24
  return config


def adjacency_acyclic_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = adjacency_preset(config)

  config.experiment_kwargs.config.dataset_name = 'adj_ca'

  return config


def adjacency_undirected_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = adjacency_preset(config)

  config.experiment_kwargs.config.dataset_name = 'adj_c_u'

  return config


def is_connected_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = adjacency_preset(config)

  config.experiment_kwargs.config.dataset_name = 'con_c'

  return config


def is_connected_acyclic_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = is_connected_preset(config)

  config.experiment_kwargs.config.dataset_name = 'con_ca'

  return config


def is_connected_undirected_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = is_connected_preset(config)

  config.experiment_kwargs.config.dataset_name = 'con_c_u'

  return config


def distance_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = adjacency_preset(config)

  config.save_checkpoint_interval = 600

  config_ = config.experiment_kwargs.config

  config_.dataset_name = 'dist_c'
  config_.evaluation.batch_size = 64

  config.best_model_eval_metric = 'rmse'
  config.best_model_eval_metric_higher_is_better = False

  return config


def distance_acyclic_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = distance_preset(config)

  config.experiment_kwargs.config.dataset_name = 'dist_ca'

  return config


def distance_undirected_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = distance_preset(config)

  config.experiment_kwargs.config.dataset_name = 'dist_c_u'

  return config


def distance_acyclic_undirected_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the sorting network dataset."""
  config = distance_preset(config)

  config.experiment_kwargs.config.dataset_name = 'dist_ca_u'

  return config


def bignn_preset(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Params for the bidirectional GNN."""
  config_ = config.experiment_kwargs.config.model.gnn_config
  config_.gnn_type = 'gnn'
  config_.k_hop = 3
  config_.se = 'gnn'
  config_.mlp_layers = 1
  config_.tightening_factor = 2
  config_.bidirectional = True
  config_.concat = True
  config_.residual = True
  return config


def ppr_posenc_preset(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Personalized Page Rank positional encodings."""
  exclude_canonical = 'ogb' not in config.experiment_kwargs.config.dataset_name
  posenc_config = config_dict.ConfigDict(
      dict(
          posenc_type='ppr',
          do_also_reverse=True,
          ppr_restart_probability=0.1,
          relative_positional_encodings=True,
          exclude_canonical=exclude_canonical,
          random_walk_steps=-1))
  config.experiment_kwargs.config.model.posenc_config = posenc_config
  return config


def random_walk_posenc_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Random Walk positional encodings."""
  exclude_canonical = 'ogb' not in config.experiment_kwargs.config.dataset_name
  posenc_config = config_dict.ConfigDict(
      dict(
          posenc_type='rw',
          do_also_reverse=True,
          ppr_restart_probability=0.05,
          relative_positional_encodings=False,
          exclude_canonical=exclude_canonical,
          random_walk_steps=3))
  config.experiment_kwargs.config.model.posenc_config = posenc_config
  return config


def magnetic_laplacian_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  config_ = config.experiment_kwargs.config

  posenc_config = config_.model.posenc_config
  posenc_config.posenc_type = 'maglap'
  # posenc_config.exclude_canonical = False
  posenc_config.relative_positional_encodings = False
  posenc_config.top_k_eigenvectors = 15
  posenc_config.maglap_q = 0.1
  posenc_config.maglap_q_absolute = False
  posenc_config.excl_k_eigenvectors = 0
  posenc_config.concatenate_eigenvalues = True
  posenc_config.maglap_symmetric_norm = True
  posenc_config.maglap_transformer = True
  posenc_config.maglap_use_signnet = True
  posenc_config.maglap_use_gnn = True
  posenc_config.maglap_norm_comps_sep = False
  posenc_config.maglap_l2_norm = True
  posenc_config.maglap_dropout_p = 0.10
  posenc_config.maglap_sign_rotate = False

  if 'sn-' in config.experiment_kwargs.config.dataset_name:
    config_.optimizer.use_agc = True
    config_.optimizer.optimizer_kwargs.weight_decay = 6e-5
    config_.optimizer.optimizer_kwargs.b1 = 0.6
    config_.optimizer.optimizer_kwargs.b2 = 0.9
    config_.optimizer.lr_schedule.peak_value = 5e-4 / 48
    config_.optimizer.lr_schedule.init_value = 1.5e-7 / 48
    config_.optimizer.lr_schedule.end_value = 1.5e-9 / 48
    config_.optimizer.agc_kwargs.clipping = 0.005

    posenc_config.top_k_eigenvectors = 25
    posenc_config.relative_positional_encodings = False
    posenc_config.maglap_q = 0.25
    posenc_config.maglap_q_absolute = False
    posenc_config.maglap_transformer = True
    posenc_config.maglap_use_gnn = False
    posenc_config.maglap_norm_comps_sep = False
    posenc_config.maglap_l2_norm = True
    posenc_config.maglap_dropout_p = 0.15
    posenc_config.maglap_use_signnet = False
    posenc_config.maglap_sign_rotate = True

  if ('adj' in config.experiment_kwargs.config.dataset_name or
      'con' in config.experiment_kwargs.config.dataset_name or
      'dist' in config.experiment_kwargs.config.dataset_name):
    posenc_config.top_k_eigenvectors = 16
    config_.model.global_readout = 'sum_n'  # Avoid adding a new virtual node
    posenc_config.maglap_use_gnn = False
    posenc_config.maglap_l2_norm = True
    posenc_config.maglap_sign_rotate = True
    posenc_config.maglap_use_signnet = False

  return config


def laplacian_preset(config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """Regular Laplacian positional encodings."""
  config = magnetic_laplacian_preset(config)

  config_ = config.experiment_kwargs.config
  config_.model.posenc_config.maglap_q = 0
  config_.model.posenc_config.relative_positional_encodings = False
  return config


def bucket_by_size_preset(
        config: config_dict.ConfigDict) -> config_dict.ConfigDict:
  """To batch graphs of similar size together."""
  config.experiment_kwargs.config.dataset_config.do_bucket_by_size = True
  return config


def model_size_preset(config: config_dict.ConfigDict,
                      layers: int) -> config_dict.ConfigDict:
  """To adjust model size and batch size accordingly."""
  config_ = config.experiment_kwargs.config

  inflate_factor = layers / config_.model.num_layers

  config_.training.batch_size = int(config_.training.batch_size /
                                    inflate_factor)
  config_.model.attention_config.num_heads = int(
      config_.model.attention_config.num_heads / inflate_factor)
  config_.model.num_layers = int(config_.model.num_layers / inflate_factor)
  return config
