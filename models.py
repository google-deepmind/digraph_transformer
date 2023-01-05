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
"""Contains top level models as well as important custom components."""

import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import warnings

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

# pylint: disable=g-bad-import-order
import config
import layers
import magnet
import utils
from ogb_utils import ASTNodeEncoder

# Inline important classes and methods
CallArgs = layers.CallArgs
BatchNorm, LayerNorm = layers.BatchNorm, layers.LayerNorm
mlp, MLP = layers.mlp, layers.MLP
MultiHeadAttention = layers.MultiHeadAttention
GraphConvolution = layers.GraphConvolution

Tensor = utils.Tensor
softmax_cross_entropy_loss = utils.softmax_cross_entropy_loss
count_edges = utils.count_edges
exact_ppr, k_step_random_walk = utils.exact_ppr, utils.k_step_random_walk
svd_encodings = utils.svd_encodings
sinusoid_position_encoding = utils.sinusoid_position_encoding


@dataclasses.dataclass
class PositionalEncodingsParams:
  """Parameters for positional encodings."""

  # Options: '', 'rw', 'ppr', 'maglap'
  # Random Walk, Personalized Page Rank, Magnetic Laplacian
  posenc_type: Optional[str] = None
  # For absolute we use aggregation for global position according to Pan Li et
  # al., 2020 or the real value of the (magnetic) laplacian eigenvectors
  # If using maglap, this implies that we have complex queries and key in attn.
  relative_positional_encodings: bool = True
  # Exclude AST depth in OGB or sinusoidal for sorting networks
  exclude_canonical: bool = False

  ### 'rw', 'ppr'
  # Not only consider the position in the direction of the graph
  do_also_reverse: bool = True
  # If using ppr/rw, the restart probability for the ppr
  ppr_restart_probability: float = 0.2
  # If using rw, the number of random walk steps

  ### 'maglap'
  random_walk_steps: int = 3
  # If using maglap, the k eigenvecotrs to use starting from excl_k_eigenvectors
  top_k_eigenvectors: int = 10
  # If using maglap, exclude the excl_k_eigenvectors most top eigenvectors
  excl_k_eigenvectors: int = 0
  # If using maglap and true, also the eigenvalues are considered
  concatenate_eigenvalues: bool = False
  # If using maglap, the q factor for directionality
  maglap_q: float = 0.25
  # If using maglap, if true `q` will be used, otherwise `q / n_node`.
  maglap_q_absolute: bool = True
  # If using maglap, True for symmetric and otherwise row normalization
  maglap_symmetric_norm: bool = False
  # If using maglap, True for a transformer or False for the MLP SignNet
  maglap_transformer: bool = False
  # If using maglap and `maglap_transformer`= True, use gnn for raw eigenvec.
  maglap_use_gnn: bool = False
  # If using maglap and `maglap_transformer`= True and if true, real and imag
  # components are separately normalized.
  maglap_norm_comps_sep: bool = False
  # If using maglap and False, we normalize the eigenvectors to span [0,1]
  # (ignoring eigenvectors of very low magnitude)
  maglap_l2_norm: bool = True
  # To force the network to work also with a subset of vectors
  maglap_dropout_p: float = 0.0
  # If using maglap, we can either use a SignNet (True) approach or scale as
  # well as rotate eigenvectors according to a convention.
  maglap_use_signnet: bool = True
  # Use the sign convention and rotation (i.e. the absolute largest real value
  # is positive and phase shift starts at 0)
  maglap_sign_rotate: bool = True
  # Rotation invariant MagLapNet
  maglap_net_type: str = 'signnet'


@dataclasses.dataclass
class LossConfig:
  """Config for the loss."""
  # Only punish first EOS token
  only_punish_first_end_of_sequence_token: bool = False


class DataFlowASTEncoder(hk.Module):
  """Encodes the AST for our graph construction procedure."""

  def __init__(self,
               emb_dim,
               num_nodetypes,
               num_nodeattributes,
               max_depth,
               num_edge_df_types,
               num_edge_ast_types,
               attribute_dropout: float = 0.0,
               name=None):
    super().__init__(name=name)

    self.emb_dim = emb_dim
    self.num_nodetypes = num_nodetypes
    self.num_nodeattributes = num_nodeattributes
    self.max_depth = max_depth
    self.num_edge_df_types = num_edge_df_types
    self.num_edge_ast_types = num_edge_ast_types
    self.attribute_dropout = attribute_dropout

  def __call__(
      self,
      graph: jraph.GraphsTuple,
      depth: Optional[Tensor] = None,
      call_args: CallArgs = CallArgs(True)
  ) -> jraph.GraphsTuple:
    nodes, edges = graph.nodes, graph.edges

    node_type_encoder = hk.Embed(self.num_nodetypes, self.emb_dim)
    node_attribute_encoder = hk.Embed(self.num_nodeattributes, self.emb_dim)

    node_type = nodes[..., 0]
    node_attribute = nodes[..., 1]

    if call_args.is_training:
      mask = hk.dropout(hk.next_rng_key(), self.attribute_dropout,
                        jnp.ones_like(node_attribute))
      node_attribute = jnp.where(mask > 0, node_attribute,
                                 self.num_nodeattributes - 1)  # set to unknown

    nodes = (
        node_type_encoder(node_type) + node_attribute_encoder(node_attribute))

    if depth is not None:
      depth_encoder = hk.Embed(self.max_depth + 1, self.emb_dim)

      depth = jnp.where(depth > self.max_depth, self.max_depth, depth)
      nodes += depth_encoder(depth[..., 0])

    edge_df_type_encoder = hk.Embed(self.num_edge_df_types, self.emb_dim)
    edge_ast_type_encoder = hk.Embed(self.num_edge_ast_types, self.emb_dim)

    edges = (
        edge_df_type_encoder(edges['edge_type']) +
        edge_ast_type_encoder(edges['edge_name']))

    graph = graph._replace(nodes=nodes, edges=edges)

    return graph


class SortingNetworkEncoder(hk.Module):
  """Encodes the Sorting Network graph."""

  def __init__(self,
               emb_dim: int,
               encode_order: bool = True,
               num_nodeattributes: int = 26,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.emb_dim = emb_dim
    self.encode_order = encode_order
    self.num_nodeattributes = num_nodeattributes

  def __call__(
      self,
      graph: jraph.GraphsTuple,
      depth: Any = None,
      call_args: CallArgs = CallArgs(True)) -> jraph.GraphsTuple:

    assert depth is None
    nodes_int = graph.nodes.astype(jnp.int32)

    argument_1 = sinusoid_position_encoding(
        nodes_int[..., 1],
        max_timescale=int(self.num_nodeattributes),
        hidden_size=self.emb_dim // 2)
    argument_2 = sinusoid_position_encoding(
        nodes_int[..., 2],
        max_timescale=int(self.num_nodeattributes),
        hidden_size=self.emb_dim // 2)
    nodes = jnp.concatenate((argument_1, argument_2), axis=-1)

    nodes = hk.Linear(self.emb_dim)(nodes)

    if self.encode_order:
      node_id = sinusoid_position_encoding(nodes_int[..., 0], self.emb_dim)
      nodes += node_id

    graph = graph._replace(nodes=nodes)
    return graph


class DistanceEncoder(hk.Module):
  """Encodes the Positional Encoding Playground graph."""

  def __init__(self,
               emb_dim: int,
               encode_order: bool = True,
               num_nodeattributes: int = 11,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.emb_dim = emb_dim
    self.encode_order = encode_order
    self.num_nodeattributes = num_nodeattributes

  def __call__(
      self,
      graph: jraph.GraphsTuple,
      depth: Any = None,
      call_args: CallArgs = CallArgs(True)) -> jraph.GraphsTuple:

    assert depth is None

    nodes = jnp.zeros((*graph.nodes.shape, self.emb_dim), dtype=jnp.float32)

    if self.encode_order:
      node_id = sinusoid_position_encoding(nodes[..., 0], self.emb_dim)
      nodes += node_id

    graph = graph._replace(nodes=nodes)
    return graph


class NaiveMagLapNet(hk.Module):
  """For the Magnetic Laplacian's or Combinatorial Laplacian's eigenvectors.

    Args:
      d_model_elem: Dimension to map each eigenvector.
      d_model_aggr: Output dimension.
      num_heads: Number of heads for optional attention.
      n_layers: Number of layers for MLP/GNN.
      dropout_p: Dropout for attention as well as eigenvector embeddings.
      activation: Element-wise non-linearity.
      concatenate_eigenvalues: If True also concatenate the eigenvalues.
      norm: Optional norm.
      name: Name of the layer.
  """

  def __init__(
        self,
        d_model_aggr: int = 256,
        name: Optional[str] = None,
        *args,
        **kwargs):
    super().__init__(name=name)
    self.d_model_aggr = d_model_aggr
    
  
  def __call__(self, graph: jraph.GraphsTuple, eigenvalues: Tensor,
               eigenvectors: Tensor, call_args: CallArgs,
               mask: Optional[Tensor] = None) -> Tensor:

    # Naive version
    re = hk.Linear(self.d_model_aggr)(jnp.real(eigenvectors))
    im = hk.Linear(self.d_model_aggr)(jnp.imag(eigenvectors))
    return re + im


class MagLapNet(hk.Module):
  """For the Magnetic Laplacian's or Combinatorial Laplacian's eigenvectors.

    Args:
      d_model_elem: Dimension to map each eigenvector.
      d_model_aggr: Output dimension.
      num_heads: Number of heads for optional attention.
      n_layers: Number of layers for MLP/GNN.
      dropout_p: Dropout for attention as well as eigenvector embeddings.
      activation: Element-wise non-linearity.
      return_real_output: True for a real number (otherwise complex).
      consider_im_part: Ignore the imaginary part of the eigenvectors.
      use_signnet: If using the sign net idea f(v) + f(-v).
      use_gnn: If True use GNN in signnet, otherwise MLP.
      use_attention: If true apply attention between eigenvector embeddings for
        same node.
      concatenate_eigenvalues: If True also concatenate the eigenvalues.
      norm: Optional norm.
      name: Name of the layer.
  """

  def __init__(self,
               d_model_elem: int = 32,
               d_model_aggr: int = 256,
               num_heads: int = 4,
               n_layers: int = 1,
               dropout_p: float = 0.2,
               activation: Callable[[Tensor], Tensor] = jax.nn.relu,
               return_real_output: bool = True,
               consider_im_part: bool = True,
               use_signnet: bool = True,
               use_gnn: bool = False,
               use_attention: bool = False,
               concatenate_eigenvalues: bool = False,
               norm: Optional[Any] = None,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.concatenate_eigenvalues = concatenate_eigenvalues
    self.consider_im_part = consider_im_part
    self.use_signnet = use_signnet
    self.use_gnn = use_gnn
    self.use_attention = use_attention
    self.num_heads = num_heads
    self.dropout_p = dropout_p
    self.norm = norm

    if self.use_gnn:
      self.element_gnn = GNN(
          int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
          gnn_type='gnn',
          k_hop=n_layers,
          mlp_layers=n_layers,
          activation=activation,
          use_edge_attr=False,
          concat=True,
          residual=False,
          name='re_element')
    else:
      self.element_mlp = MLP(
          int(2 * d_model_elem) if self.consider_im_part else d_model_elem,
          n_layers=n_layers,
          activation=activation,
          with_norm=False,
          final_activation=True,
          name='re_element')

    self.re_aggregate_mlp = MLP(
        d_model_aggr,
        n_layers=n_layers,
        activation=activation,
        with_norm=False,
        final_activation=True,
        name='re_aggregate')

    self.im_aggregate_mlp = None
    if not return_real_output and self.consider_im_part:
      self.im_aggregate_mlp = MLP(
          d_model_aggr,
          n_layers=n_layers,
          activation=activation,
          with_norm=False,
          final_activation=True,
          name='im_aggregate')

  def __call__(self, graph: jraph.GraphsTuple, eigenvalues: Tensor,
               eigenvectors: Tensor, call_args: CallArgs,
               mask: Optional[Tensor] = None) -> Tensor:
    padding_mask = (eigenvalues > 0)[..., None, :]
    padding_mask = padding_mask.at[..., 0].set(True)
    attn_padding_mask = padding_mask[..., None] & padding_mask[..., None, :]

    trans_eig = jnp.real(eigenvectors)[..., None]

    if self.consider_im_part and jnp.iscomplexobj(eigenvectors):
      trans_eig_im = jnp.imag(eigenvectors)[..., None]
      trans_eig = jnp.concatenate((trans_eig, trans_eig_im), axis=-1)
    else:
      if not self.use_signnet:
        # Like to Dwivedi & Bresson (2021)
        rand_sign_shape = (*trans_eig.shape[:-3], 1, *trans_eig.shape[-2:])
        rand_sign = jax.random.rademacher(hk.next_rng_key(), rand_sign_shape)
        trans_eig = rand_sign * trans_eig
      # Lower impact of numerical issues, assumes `k_excl` = 0
      trans_eig = trans_eig.at[..., 0, :].set(
          jnp.absolute(trans_eig[..., 0, :]))
      eigenvalues = eigenvalues.at[..., 0].set(0)

    if self.use_gnn:
      trans = self.element_gnn(
          graph._replace(nodes=trans_eig, edges=None), call_args).nodes
      if self.use_signnet:
        trans_neg = self.element_gnn(
            graph._replace(nodes=-trans_eig, edges=None), call_args).nodes
        # assumes `k_excl` = 0
        if self.consider_im_part and jnp.iscomplexobj(eigenvectors):
          trans_neg = trans_neg.at[..., 0, :].set(0)
        trans += trans_neg
    else:
      trans = self.element_mlp(trans_eig)
      if self.use_signnet:
        trans_neg = self.element_mlp(-trans_eig)
        # assumes `k_excl` = 0
        if self.consider_im_part and jnp.iscomplexobj(eigenvectors):
          trans_neg = trans_neg.at[..., 0, :].set(0)
        trans += trans_neg

    if self.concatenate_eigenvalues:
      eigenvalues_ = jnp.broadcast_to(eigenvalues[..., None, :],
                                      trans.shape[:-1])
      trans = jnp.concatenate((eigenvalues_[..., None], trans), axis=-1)

    if self.use_attention:
      if self.norm is not None:
        trans = self.norm()(trans)
      attn = MultiHeadAttention(
          self.num_heads,
          key_size=trans.shape[-1] // self.num_heads,
          value_size=trans.shape[-1] // self.num_heads,
          model_size=trans.shape[-1],
          w_init=None,
          dropout_p=self.dropout_p,
          with_bias=False)
      trans += attn(
          trans,
          trans,
          trans,
          mask=attn_padding_mask,
          is_training=call_args.is_training)

    padding_mask = padding_mask[..., None]
    trans = trans * padding_mask
    trans = trans.reshape(trans.shape[:-2] + (-1,))

    if self.dropout_p and call_args.is_training:
      trans = hk.dropout(hk.next_rng_key(), self.dropout_p, trans)

    output = self.re_aggregate_mlp(trans)
    if self.im_aggregate_mlp is None:
      return output

    output_im = self.im_aggregate_mlp(trans)
    output = output + 1j * output_im
    return output


class GNN(hk.Module):
  """Standard GNN that supersedes a GCN implementation os used by the Open Graph Benchmark Code2 dataset and a standard bidirectional GNN.

    Attributes:
      d_model: number of hidden dimensions.
      activation: The activation function.
      gnn_type: Either `gcn` or `gnn`
      use_edge_attr: If True also the edge attributes are considered. Must be
        True for `gnn_type=gcn`.
      k_hop: Number of message passing steps.
      mlp_layers: Number of layers in MLP (only relevant for `gnn_type=gnn`).
      tightening_factor: The factor of dimensionality reduction for message
        passing in contrast to `d_model`.
      norm: The batch/layer norm.
      concat: If True all intermediate node embeddings are concatenated and then
        mapped to `d_model` in the output MLP.
      residual: If True the GNN embeddings
      bidirectional: If True, edges in both directions are considered (only
        relevant for `gnn_type=gnn`).
      name: Name of module.
      **kwargs:
  """

  def __init__(self,
               d_model: int = 256,
               activation: Callable[[Tensor], Tensor] = jax.nn.relu,
               gnn_type='gcn',
               use_edge_attr=True,
               k_hop=2,
               mlp_layers: int = 2,
               tightening_factor: int = 1,
               norm=None,
               concat: bool = False,
               residual: bool = True,
               bidirectional: bool = True,
               name: Optional[str] = None,
               **kwargs):
    super().__init__(name=name)
    self.d_model = d_model
    self.mlp_layers = mlp_layers
    self.tightening_factor = tightening_factor
    self.activation = activation
    self.gnn_type = gnn_type
    self.use_edge_attr = use_edge_attr
    self.k_hop = k_hop
    self.norm = norm
    self.concat = concat
    self.residual = residual
    self.bidirectional = bidirectional

    if kwargs:
      logging.info('GNN.__init__() received unexpected kwargs: %s', kwargs)

  def _layer(self,
             idx: int) -> Callable[[jraph.GraphsTuple], jraph.GraphsTuple]:
    if self.gnn_type == 'gcn':
      assert self.use_edge_attr, 'For GCN we must include edge features'

      def update_fn(x):
        return hk.Linear(self.d_model // self.tightening_factor)(x)

      backw_update_node_fn = backw_update_edge_fn = None
      if self.bidirectional:
        backw_update_node_fn = backw_update_edge_fn = update_fn

      layer = GraphConvolution(
          forw_update_node_fn=update_fn,
          forw_update_edge_fn=update_fn,
          backw_update_node_fn=backw_update_node_fn,
          backw_update_edge_fn=backw_update_edge_fn,
          activation=self.activation,
          add_self_edges=False)
    elif self.gnn_type == 'gnn':
      layer = jraph.GraphNetwork(
          update_edge_fn=functools.partial(
              self._update_fn,
              exclude_features=not self.use_edge_attr,
              name=f'edge_mlp{idx}'),
          update_node_fn=functools.partial(
              self._update_fn, exclude_features=False, name=f'node_mlp{idx}'))
    else:
      raise ValueError(f'`gnn_type` {self.gnn_type} is not supported')
    return layer

  def _update_fn(self, features: Tensor, sender_features: Tensor,
                 receiver_features: Tensor, globals_: Any,
                 exclude_features: bool, name: str):
    if self.bidirectional:
      stack = [features, sender_features, receiver_features]
    else:
      stack = [features, sender_features + receiver_features]
    if exclude_features:
      stack = stack[1:]
    concat_features = jnp.concatenate(stack, axis=-1)
    return mlp(
        concat_features,
        self.d_model // self.tightening_factor,
        activation=self.activation,
        with_norm=False,
        final_activation=True,
        n_layers=self.mlp_layers,
        name=name)

  def __call__(self,
               graph: jraph.GraphsTuple,
               call_args: CallArgs,
               mask: Optional[Tensor] = None,
               in_axes: int = 0,
               **kwargs):
    if self.k_hop == 0:
      return graph
    nodes_list = [graph.nodes]
    for idx in range(self.k_hop):
      new_graph = jax.vmap(self._layer(idx), in_axes=in_axes)(graph)
      if self.residual and self.tightening_factor == 1:
        graph = graph._replace(
            nodes=graph.nodes + new_graph.nodes,
            edges=graph.edges + new_graph.edges,
        )
      else:
        graph = new_graph
      if self.concat:
        nodes_list.append(graph.nodes)
    if self.concat:
      graph = graph._replace(nodes=jnp.concatenate(nodes_list, axis=-1))
    if self.norm:
      graph = graph._replace(nodes=self.norm()(graph.nodes, call_args, mask))
    if self.concat or self.tightening_factor > 1:
      graph = graph._replace(nodes=hk.Linear(self.d_model)(graph.nodes))
    return graph


class TransformerEncoderLayer(hk.Module):
  """The transformer encoder layer.

  Main differences to the common implementation is the option to apply a GNN to
  query and key, as well as handling complex valued positional encodings.

  Attributes:
    d_model: number of hidden dimensions.
    num_heads: in multi head attention.
    dense_widening_factor: factor to enlarge MLP after attention.
    dropout_p: Probability dropout.
    with_bias: If the linear projections shall have a bias.
    re_im_separate_projection: Apply a joint (False) or separate projection
      (True) for complex values query and key.
    se: If to apply a structural encoding to query and key (either `gnn` or ``).
    norm: The batch/layer norm.
    pre_norm: If applying norm before attention.
    activation: The activation function.
    gnn_config: Config for GNN.
    name: Name of the layer.
  """

  def __init__(  # pylint: disable=dangerous-default-value
      self,
      d_model=256,
      num_heads=4,
      dense_widening_factor=4,
      dropout_p=0.2,
      with_attn_dropout=True,
      re_im_separate_projection=False,
      with_bias=False,
      activation=jax.nn.relu,
      norm: Any = LayerNorm,
      pre_norm: bool = False,
      gnn_config=dict(),
      name=None):
    super().__init__(name=name)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dense_widening_factor = dense_widening_factor
    self.dropout_p = dropout_p
    self.with_bias = with_bias
    self.re_im_separate_projection = re_im_separate_projection
    if not isinstance(gnn_config, dict):
      gnn_config = gnn_config.to_dict()
    else:
      gnn_config = gnn_config.copy()
    self.se = gnn_config.pop('se', 'gnn')

    self.norm = norm
    self.pre_norm = pre_norm
    self.activation = activation

    if self.se == 'gnn':
      self.se = GNN(self.d_model, self.activation, norm=self.norm, **gnn_config)
    elif self.se:
      raise ValueError(f'unexpected structure extractor value {self.se}')

    self.linear_dm = functools.partial(hk.Linear, self.d_model)
    self.linear_ff = functools.partial(
        hk.Linear, int(self.d_model * self.dense_widening_factor))

    self.attn = MultiHeadAttention(
        self.num_heads,
        key_size=self.d_model // self.num_heads,
        value_size=self.d_model // self.num_heads,
        model_size=self.d_model,
        w_init=None,
        dropout_p=self.dropout_p if with_attn_dropout else 0.,
        with_bias=self.with_bias,
        re_im_separate_projection=self.re_im_separate_projection)


  def __call__(self,
               graph: jraph.GraphsTuple,
               call_args: CallArgs,
               invnorm_degree: Optional[Tensor] = None,
               posenc: Optional[Tensor] = None,
               mask: Optional[Tensor] = None):
    if mask is not None:
      bn_mask = mask[..., :1]
    else:
      bn_mask = None

    if self.pre_norm:
      graph = graph._replace(nodes=self.norm()(graph.nodes, call_args, bn_mask))

    value = graph.nodes
    if posenc is not None and posenc.ndim <= value.ndim:
      value = jnp.real(value + posenc)
      graph = graph._replace(nodes=value)

    if not self.se:
      query = key = graph.nodes
    else:
      graph_se = self.se(graph, call_args, mask=bn_mask)

      query = key = graph_se.nodes

    logit_offset = None
    if posenc is not None:
      if posenc.ndim > query.ndim:
        logit_offset = posenc
        posenc = None
      else:
        query = key = query + posenc

    attn_emb = self.attn(
        query=query,
        key=key,
        value=value,
        is_training=call_args.is_training,
        logit_offset=logit_offset,
        mask=mask)

    if invnorm_degree is not None:
      attn_emb = invnorm_degree[..., None] * attn_emb
    if call_args.is_training:
      attn_emb = hk.dropout(hk.next_rng_key(), self.dropout_p, attn_emb)
    value = value + attn_emb

    value = self.norm()(value, call_args, bn_mask)

    fwd_emb = self.activation(self.linear_ff()(value))
    if call_args.is_training:
      fwd_emb = hk.dropout(hk.next_rng_key(), self.dropout_p, fwd_emb)
    fwd_emb = self.linear_dm()(fwd_emb)
    if call_args.is_training:
      fwd_emb = hk.dropout(hk.next_rng_key(), self.dropout_p, fwd_emb)

    value = value + fwd_emb

    if not self.pre_norm:
      value = self.norm()(value, call_args, bn_mask)

    return graph._replace(nodes=value)


class GraphTransformerEncoder(hk.Module):
  """Wrapper for multiple encoder layers."""

  def __init__(
      self,
      layer_sequence: Sequence[TransformerEncoderLayer],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.layer_sequence = layer_sequence

  def __call__(self, x, *args, **kwargs):
    output = x

    for layer in self.layer_sequence:
      output = layer(output, *args, **kwargs)

    return output


class StructureAwareTransformer(hk.Module):
  """Our implementation/extension of the Structure Aware Transformer, ICML22.

  Attributes:
    model_type: Either `sat` or `gnn`.
    num_class: Number of classes for graph classification.
    d_model: number of hidden dimensions.
    input_embedder: an embedder for the raw graph's features.
    gnn_config: Config for GNN
    attention_config: Config for Attention
    max_seq_len: number of prediction to be made per graph.
    global_readout: how to aggregate the node embeddings. One of `1`, `cls`,
      `mean`, `sum_n`.
    activation: The activation function.
    batch_norm: If True use BatchNorm, otherwise use LayerNorm.
    deg_sensitive_residual: If True normalize residual by node degree.
    with_pre_gnn: If True apply a GNN before the transformer.
    posenc_config: Configuration of positional encodings.
    loss_config: Configuration for loss.
    eps: a small constant.
    pmap_axis: Relevant for hk.BatchNorm to work within `pmap`.
    name: Name of the layer.
  """

  # pylint: disable=dangerous-default-value
  def __init__(self,
               model_type: str,
               num_class: int,
               d_model: int,
               input_embedder: Optional[Union[ASTNodeEncoder,
                                              DataFlowASTEncoder]] = None,
               num_layers=4,
               gnn_config=dict(),
               attention_config=dict(),
               max_seq_len=5,
               global_readout='mean',
               activation: str = 'relu',
               batch_norm: bool = False,
               deg_sensitive_residual: bool = True,
               with_pre_gnn: bool = False,
               posenc_config=PositionalEncodingsParams(),
               loss_config=LossConfig(),
               eps: float = 1e-7,
               pmap_axis: Optional[str] = None,
               name: Optional[str] = None):
    super().__init__(name=name)

    self.d_model = d_model
    self.deg_sensitive_residual = deg_sensitive_residual
    self.eps = eps
    self.embedding = input_embedder
    self.global_readout = global_readout
    self.posenc_config = posenc_config
    self.loss_config = loss_config
    self.max_seq_len = max_seq_len
    self.num_class = num_class

    self.activation = self.activation_fn(activation)

    if batch_norm:
      self.norm = functools.partial(
          BatchNorm,
          create_scale=True,
          create_offset=True,
          decay_rate=0.9,
          eps=1e-5,
          cross_replica_axis=pmap_axis)
    else:
      self.norm = functools.partial(LayerNorm, axis=-1, eps=1e-5)

    if model_type == 'sat':
      layer = functools.partial(
          TransformerEncoderLayer,
          d_model=d_model,
          activation=self.activation,
          gnn_config=gnn_config,
          norm=self.norm,
          **attention_config)
      self.encoder = GraphTransformerEncoder(
          [layer() for idx in range(num_layers)])
    elif model_type == 'magnet':
      self.encoder = magnet.MagNet(
          d_model, activation=self.activation, q=posenc_config.maglap_q,
          q_absolute=posenc_config.maglap_q_absolute, **gnn_config)
    else:
      warnings.warn(f'Falling back to `gnn` model (given type {model_type})')
      self.encoder = GNN(d_model, activation=self.activation, **gnn_config)

    self.pre_gnn = None
    if with_pre_gnn:

      # pylint: disable=unused-argument
      def update_edge_fn(features: Tensor, sender_features: Tensor,
                         receiver_features: Tensor, globals_: Any) -> Tensor:
        return mlp(
            features,
            self.d_model,
            activation=self.activation,
            with_norm=False,
            final_activation=True,
            n_layers=1,
            name='pre_gnn_update_edge_fn')

      def update_node_fn(features: Tensor, sender_features: Tensor,
                         receiver_features: Tensor, globals_: Any) -> Tensor:
        concat_features = jnp.concatenate(
            [features, sender_features, receiver_features], axis=-1)
        return mlp(
            concat_features,
            self.d_model,
            activation=self.activation,
            with_norm=False,
            final_activation=True,
            n_layers=1,
            name='pre_gnn_update_node_fn')

      self.pre_gnn = jraph.GraphNetwork(
          update_edge_fn=update_edge_fn, update_node_fn=update_node_fn)

    if self.posenc_config.posenc_type == 'maglap':
      use_rel_posenc = self.posenc_config.relative_positional_encodings
      if self.posenc_config.maglap_net_type == 'naive':
        self.maglap_net = NaiveMagLapNet(self.d_model)
      else:
        self.maglap_net = MagLapNet(
            self.d_model // 32,
            self.d_model,
            num_heads=attention_config.get('num_heads', 4),
            n_layers=1,
            dropout_p=self.posenc_config.maglap_dropout_p,
            concatenate_eigenvalues=self.posenc_config.concatenate_eigenvalues,
            consider_im_part=self.posenc_config.maglap_q != 0.,
            activation=self.activation,
            use_signnet=self.posenc_config.maglap_use_signnet,
            use_gnn=self.posenc_config.maglap_use_gnn,
            use_attention=self.posenc_config.maglap_transformer,
            norm=self.norm if self.posenc_config.maglap_transformer else None,
            return_real_output=not use_rel_posenc)

  def activation_fn(self, activation: str):
    """Get activation function from name."""
    if activation == 'relu':
      return jax.nn.relu
    elif activation == 'gelu':
      return jax.nn.gelu
    else:
      raise ValueError(f'unexpected activation {activation}')

  def positional_encodings(
      self,
      graph: jraph.GraphsTuple,
      eigenvalues: Optional[Tensor] = None,
      eigenvectors: Optional[Tensor] = None,
      call_args: CallArgs = CallArgs(True),
      mask: Optional[Tensor] = None
  ) -> Tuple[jraph.GraphsTuple, Optional[Tensor]]:
    """Adds the positional encodings."""
    if not self.posenc_config.posenc_type:
      return graph, None

    if self.posenc_config.posenc_type == 'ppr':
      ppr = exact_ppr(
          graph, restart_p=self.posenc_config.ppr_restart_probability)
      if self.posenc_config.do_also_reverse:
        backward_ppr = exact_ppr(
            graph,
            restart_p=self.posenc_config.ppr_restart_probability,
            reverse=True)
        posenc = jnp.stack((ppr, backward_ppr), axis=-1)
      else:
        posenc = ppr[..., None]
    elif self.posenc_config.posenc_type == 'rw':
      posenc = k_step_random_walk(
          graph,
          k=self.posenc_config.random_walk_steps,
          ppr_restart_p=self.posenc_config.ppr_restart_probability)
      if self.posenc_config.do_also_reverse:
        backward_rw = k_step_random_walk(
            graph,
            k=self.posenc_config.random_walk_steps,
            ppr_restart_p=self.posenc_config.ppr_restart_probability,
            reverse=True)
        posenc = jnp.concatenate((posenc, backward_rw), axis=-1)
    elif self.posenc_config.posenc_type == 'svd':
      posenc = svd_encodings(graph, rank=self.posenc_config.top_k_eigenvectors)
      # Like to Hussain (2022)
      rand_sign_shape = (*posenc.shape[:-2], 1, *posenc.shape[-1:])
      rand_sign = jax.random.rademacher(hk.next_rng_key(), rand_sign_shape)
      posenc = rand_sign * posenc
      posenc = mlp(
          posenc,
          self.d_model,
          activation=self.activation,
          n_layers=1,
          with_norm=False,
          final_activation=True,
          name='posenc_mlp')
      graph = graph._replace(nodes=graph.nodes + posenc)
      # Otherwise, we interpret the encoding as relative
      posenc = None
    elif self.posenc_config.posenc_type == 'maglap':
      # We might have already loaded the eigenvalues, -vectors
      if eigenvalues is None or eigenvectors is None:
        raise RuntimeError('Eigenvectors and eigenvalues were not provided.')

      posenc = self.maglap_net(
          graph, eigenvalues, eigenvectors, call_args=call_args, mask=mask)

      if not self.posenc_config.relative_positional_encodings:
        graph = graph._replace(nodes=graph.nodes + posenc)
        # Otherwise, we interpret the encoding as relative
        posenc = None
    else:
      raise ValueError(
          f'unexpected positional encoding type {self.posenc_config.posenc_type}'
      )

    if (not self.posenc_config.relative_positional_encodings and
        self.posenc_config.posenc_type in ['ppr', 'rw']):
      posenc = hk.Linear(posenc.shape[-1], name='posenc_linear')(posenc)
      posenc = (self.activation(posenc) * mask[..., None]).sum(axis=-3)
      posenc = mlp(
          posenc,
          self.d_model,
          activation=self.activation,
          n_layers=1,
          with_norm=False,
          final_activation=True,
          name='posenc_mlp')
      graph = graph._replace(nodes=graph.nodes + posenc)
      # Otherwise, we interpret the encoding as relative
      posenc = None

    return graph, posenc

  def readout(self, graph: jraph.GraphsTuple) -> Tensor:
    """For the aggregate prediction over for the graph."""
    if self.global_readout == 1 or self.global_readout == '1':
      return graph.nodes[..., 1, :]
    elif self.global_readout == 'cls':
      return graph.nodes[jnp.arange(graph.n_node.shape[0]),
                         graph.n_node[..., 0].astype(jnp.int32) - 1]

    batch, n_node = graph.nodes.shape[:2]
    indices = jnp.tile(jnp.arange(n_node), (batch, 1))
    padding_mask = indices < graph.n_node[:, 0, None]

    if self.global_readout == 'mean':
      output = (graph.nodes * padding_mask[..., None]).sum(-2)
      output /= (graph.n_node[:, 0, None] + self.eps)
    elif self.global_readout == 'sum_n':
      output = (graph.nodes * padding_mask[..., None]).sum(-2)
      n = graph.n_node[:, 0, None]
      output = jnp.concatenate((output, n), axis=-1)

    return output

  def get_padding_mask(self, graph: jraph.GraphsTuple) -> Tensor:
    batch, n_node = graph.nodes.shape[:2]
    indices = jnp.tile(jnp.arange(n_node), (batch, 1))
    padding_mask = indices < graph.n_node[:, 0, None]
    padding_mask = padding_mask[:, :, None] * padding_mask[:, None, :]
    return padding_mask

  def __call__(self,
               graph: jraph.GraphsTuple,
               call_args: CallArgs = CallArgs(False)):
    node_depth = None
    if 'node_depth' in graph.nodes and not self.posenc_config.exclude_canonical:
      node_depth = graph.nodes['node_depth']

    eigenvalues = eigenvectors = None
    if 'eigenvalues' in graph.nodes and 'eigenvectors' in graph.nodes:
      eigenvalues = graph.nodes['eigenvalues']
      eigenvectors = graph.nodes['eigenvectors']

    # Either nodes.['node_feat'] or the `nodes` represent the features
    if 'node_feat' in graph.nodes:
      graph = graph._replace(nodes=graph.nodes['node_feat'])

    if self.deg_sensitive_residual:
      sender_degree = count_edges(graph.senders, n_nodes=graph.nodes.shape[1])
      invnorm_degree = jax.lax.rsqrt((sender_degree + 1.))
    else:
      invnorm_degree = None

    if self.embedding is not None:
      graph = self.embedding(graph, node_depth, call_args)

    if self.pre_gnn is not None:
      graph = graph._replace(nodes=jax.vmap(self.pre_gnn)(graph).nodes)

    padding_mask = self.get_padding_mask(graph)
    graph, posenc = self.positional_encodings(graph, eigenvalues, eigenvectors,
                                              call_args, mask=padding_mask)

    if self.global_readout == 'cls':
      cls_token = hk.get_parameter('cls_token', (1, self.d_model), jnp.float32,
                                   hk.initializers.RandomNormal())
      n_node = graph.n_node.astype(jnp.int32)
      nodes = graph.nodes.at[jnp.arange(graph.nodes.shape[0]),
                             n_node[..., 0].astype(jnp.int32)].set(cls_token)
      n_node = n_node.at[..., 0].add(1)
      n_node = n_node.at[..., 1].add(-1)
      graph = graph._replace(nodes=nodes, n_node=n_node)
      # New padding mask due to added node
      padding_mask = self.get_padding_mask(graph)

    graph = self.encoder(
        graph,
        call_args=call_args,
        posenc=posenc,
        invnorm_degree=invnorm_degree,
        mask=padding_mask)

    output = self.readout(graph)

    if self.num_class <= 0 or self.max_seq_len <= 0:
      lead = len(graph.nodes.shape[:-2]) * (1,)
      nodes = graph.nodes.shape[-2]

      x_senders = jnp.tile(graph.nodes[..., :, None, :], lead + (1, nodes, 1))
      x_receivers = jnp.tile(graph.nodes[..., None, :, :], lead + (nodes, 1, 1))
      x_global = jnp.tile(output[..., None, None, :], lead + (nodes, nodes, 1))
      x = jnp.concatenate((x_senders, x_receivers, x_global), axis=-1)
      x = MLP(self.d_model, n_layers=2, activation=self.activation,
              with_norm=True, final_activation=True, name='adj_mlp')(x)

      dim_out = 1 if self.num_class <= 0 else self.num_class
      output = hk.Linear(dim_out, name='adj_out')(x)
      return output

    classifier = hk.Linear(self.num_class * self.max_seq_len)
    prediction = classifier(output).reshape(
        list(output.shape[:-1]) + [self.max_seq_len, self.num_class])
    return prediction

  def loss(self, graph: jraph.GraphsTuple, is_training=True):
    prediction = self.__call__(
        graph._replace(globals=None), call_args=CallArgs(is_training))
    if self.num_class > 0 and self.max_seq_len > 0:
      loss = softmax_cross_entropy_loss(
          prediction, graph.globals['target'][..., 0, :].astype(jnp.int32),
          self.num_class,
          self.loss_config.only_punish_first_end_of_sequence_token)
    else:
      if isinstance(graph.nodes, dict):
        graph = graph._replace(nodes=graph.nodes['node_feat'])
      target = graph.globals['target']
      mask = target >= 0

      if self.num_class > 0:
        target = jnp.where(mask, target, 0)
        targets_one_hot = jax.nn.one_hot(target, self.num_class)

        logits = jax.nn.log_softmax(prediction, axis=-1)
        loss = -jnp.sum(targets_one_hot * logits, axis=-1)

      else:
        prediction = jax.nn.softplus(prediction[..., 0])
        loss = (prediction - target) ** 2

      loss = (loss * mask).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
    return loss, prediction


def get_model(  # pylint: disable=dangerous-default-value
    dataset: config.Dataset,
    # For batch norm
    pmap_axis: str,
    model_type: str = 'sat',
    num_layers=4,
    d_model=256,
    activation: str = 'relu',
    batch_norm: bool = False,
    deg_sensitive_residual: bool = True,
    with_pre_gnn: bool = False,
    encoder_config=dict(),
    loss_kwargs=dict(),
    posenc_config=dict(),
    attention_config=dict(),
    gnn_config=dict(),
    global_readout='mean',
    **kwargs):
  """Creates the model for the given configuration."""

  if kwargs:
    logging.info('get_model() received kwargs: %s', kwargs)

  posenc_config = PositionalEncodingsParams(**posenc_config)

  if '-df' in dataset.name:
    input_encoder = DataFlowASTEncoder(d_model, dataset.num_nodetypes,
                                       dataset.num_nodeattributes,
                                       dataset.ast_depth, dataset.edge_df_types,
                                       dataset.edge_ast_types, **encoder_config)
  elif dataset.name.startswith('sn'):
    input_encoder = SortingNetworkEncoder(
        d_model,
        num_nodeattributes=dataset.num_nodeattributes,
        encode_order=not posenc_config.exclude_canonical)
  elif dataset.name.startswith('dist'):
    input_encoder = DistanceEncoder(
      d_model, encode_order=not posenc_config.exclude_canonical)
  else:
    edge_dim = d_model if gnn_config.get('residual', False) else d_model // 2
    input_encoder = ASTNodeEncoder(
        d_model,
        dataset.num_nodetypes,
        dataset.num_nodeattributes,
        dataset.ast_depth,
        edge_dim=edge_dim)

  model = StructureAwareTransformer(
      model_type,
      dataset.num_classes,
      d_model,
      input_encoder,
      attention_config=attention_config,
      num_layers=num_layers,
      max_seq_len=dataset.sequence_length,
      gnn_config=gnn_config,
      pmap_axis=pmap_axis,
      with_pre_gnn=with_pre_gnn,
      deg_sensitive_residual=deg_sensitive_residual,
      posenc_config=posenc_config,
      loss_config=LossConfig(**loss_kwargs),
      global_readout=global_readout,
      activation=activation,
      batch_norm=batch_norm,
      **kwargs)

  return model
