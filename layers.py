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
"""Mostly standard comnponents or adaptions to mimic PyTorch's behaviour."""
import dataclasses
from typing import Callable, Optional, Union
import warnings

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

Tensor = Union[np.ndarray, jnp.DeviceArray]


@dataclasses.dataclass
class CallArgs:
  """Common arguments to __call__ for most modules."""

  # Whether this is training or inference.
  is_training: bool

  # Whether local stats are used for batch norm when is_training=False.
  test_local_stats: bool = False


class ExponentialMovingAverage(hk.Module):
  """Maintains an exponential moving average.

  This uses the Adam debiasing procedure.
  See https://arxiv.org/pdf/1412.6980.pdf for details.
  """

  def __init__(
      self,
      decay,
      zero_debias: bool = True,
      warmup_length: int = 0,
      init_value: float = 0,
      name: Optional[str] = None,
  ):
    """Initializes an ExponentialMovingAverage module.

    Args:
      decay: The chosen decay. Must in ``[0, 1)``. Values close to 1 result in
        slow decay; values close to ``0`` result in fast decay.
      zero_debias: Whether to run with zero-debiasing.
      warmup_length: A positive integer, EMA has no effect until the internal
        counter has reached `warmup_length` at which point the initial value for
        the decaying average is initialized to the input value after
        `warmup_length` iterations.
      init_value: Value to warm start the moving average.
      name: The name of the module.
    """
    super().__init__(name=name)
    self.decay = decay
    self.warmup_length = warmup_length
    self.zero_debias = zero_debias
    self.init_value = init_value
    self.init = hk.initializers.Constant(init_value)

    if warmup_length < 0:
      raise ValueError(
          f'`warmup_length` is {warmup_length}, but should be non-negative.')

    if warmup_length and zero_debias:
      raise ValueError(
          'Zero debiasing does not make sense when warming up the value of the '
          'average to an initial value. Set zero_debias=False if setting '
          'warmup_length to a non-zero value.')

    if init_value != 0 and zero_debias:
      raise ValueError(
          'Do not set an inti value and zero_debias at the same time')

  def initialize(self, shape, dtype=jnp.float32):
    """If uninitialized sets the average to ``zeros`` of the given shape/dtype.
    """
    if hasattr(shape, 'shape'):
      warnings.warn(
          'Passing a value into initialize instead of a shape/dtype '
          'is deprecated. Update your code to use: '
          '`ema.initialize(v.shape, v.dtype)`.',
          category=DeprecationWarning)
      shape, dtype = shape.shape, shape.dtype

    hk.get_state('hidden', shape, dtype, init=self.init)
    hk.get_state('average', shape, dtype, init=self.init)

  def __call__(
      self,
      value: jnp.ndarray,
      update_stats: bool = True,
  ) -> jnp.ndarray:
    """Updates the EMA and returns the new value.

    Args:
      value: The array-like object for which you would like to perform an
        exponential decay on.
      update_stats: A Boolean, whether to update the internal state of this
        object to reflect the input value. When `update_stats` is False the
        internal stats will remain unchanged.

    Returns:
      The exponentially weighted average of the input value.
    """
    if not isinstance(value, jnp.ndarray):
      value = jnp.asarray(value)

    counter = hk.get_state(
        'counter', (),
        jnp.int32,
        init=hk.initializers.Constant(-self.warmup_length))
    counter = counter + 1

    decay = jax.lax.convert_element_type(self.decay, value.dtype)
    if self.warmup_length > 0:
      decay = jax.lax.select(counter <= 0, 0.0, decay)

    one = jnp.ones([], value.dtype)
    hidden = hk.get_state('hidden', value.shape, value.dtype, init=self.init)
    hidden = hidden * decay + value * (one - decay)

    average = hidden
    if self.zero_debias:
      average /= (one - jnp.power(decay, counter))

    if update_stats:
      hk.set_state('counter', counter)
      hk.set_state('hidden', hidden)
      hk.set_state('average', average)

    return average

  @property
  def average(self):
    return hk.get_state('average')


class LayerNorm(hk.LayerNorm):
  """Wrapper to allow for same interface as BatchNorm."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, create_scale=True, create_offset=True, **kwargs)

  def __call__(self,
               x: Tensor,
               call_args: Optional[CallArgs] = None,
               mask: Optional[Tensor] = None) -> Tensor:
    return super().__call__(x)


class BatchNorm(hk.BatchNorm):
  """Makes a BatchNorm Module that can be called with CallArgs."""

  def __init__(self,
               create_scale=False,
               create_offset=True,
               decay_rate=0.999,
               eps: float = 1e-3,
               initialize_running_stats: bool = True,
               name: Optional[str] = None,
               **kwargs):
    super().__init__(
        create_scale=create_scale,
        create_offset=create_offset,
        decay_rate=decay_rate,
        eps=eps,
        name=name,
        **kwargs)
    if initialize_running_stats:
      self.mean_ema = ExponentialMovingAverage(
          decay_rate,
          name='mean_ema',
          zero_debias=False,
          init_value=0.,
          warmup_length=0)
      self.var_ema = ExponentialMovingAverage(
          decay_rate,
          name='var_ema',
          zero_debias=False,
          init_value=1.,
          warmup_length=0)

  def __call__(self,
               x: Tensor,
               call_args: CallArgs,
               mask: Optional[Tensor] = None):
    return self.forward(
        x,
        is_training=call_args.is_training,
        test_local_stats=call_args.test_local_stats,
        mask=mask)

  def forward(
      self,
      inputs: Tensor,
      is_training: bool,
      mask: Optional[Tensor] = None,
      test_local_stats: bool = False,
      scale: Optional[Tensor] = None,
      offset: Optional[Tensor] = None,
  ) -> Tensor:
    """Computes the normalized version of the input with optional masking.

    Args:
      inputs: An array, where the data format is ``[..., C]``.
      is_training: Whether this is during training.
      mask: If provided, mask must broadcast to inputs where `false` elements
        are masked out for calculating the running statistics.
      test_local_stats: Whether local stats are used when is_training=False.
      scale: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the scale applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_scale=True``.
      offset: An array up to n-D. The shape of this tensor must be broadcastable
        to the shape of ``inputs``. This is the offset applied to the normalized
        inputs. This cannot be passed in if the module was constructed with
        ``create_offset=True``.

    Returns:
      The array, normalized across all but the last dimension.
    """
    if self.create_scale and scale is not None:
      raise ValueError(
          'Cannot pass `scale` at call time if `create_scale=True`.')
    if self.create_offset and offset is not None:
      raise ValueError(
          'Cannot pass `offset` at call time if `create_offset=True`.')

    channel_index = self.channel_index
    if channel_index < 0:
      channel_index += inputs.ndim

    if self.axis is not None:
      axis = self.axis
    else:
      axis = [i for i in range(inputs.ndim) if i != channel_index]

    if is_training or test_local_stats:
      if mask is None:
        mask = jnp.ones_like(inputs)
      n_elements = jnp.sum(mask, axis, keepdims=True)
      inputs *= mask
      isum = jnp.sum(inputs, axis, keepdims=True)
      isum_of_squares = jnp.sum(jnp.square(inputs), axis, keepdims=True)
      if self.cross_replica_axis and jax.device_count() > 1:
        isum = jax.lax.psum(
            isum,
            axis_name=self.cross_replica_axis,
            axis_index_groups=self.cross_replica_axis_index_groups)
        isum_of_squares = jax.lax.psum(
            isum_of_squares,
            axis_name=self.cross_replica_axis,
            axis_index_groups=self.cross_replica_axis_index_groups)
        n_elements = jax.lax.psum(
            n_elements,
            axis_name=self.cross_replica_axis,
            axis_index_groups=self.cross_replica_axis_index_groups)
      mean = isum / n_elements
      mean_of_squares = isum_of_squares / n_elements
      var = mean_of_squares - jnp.square(mean)
    else:
      mean = self.mean_ema.average.astype(inputs.dtype)
      var = self.var_ema.average.astype(inputs.dtype)

    if is_training:
      self.mean_ema(mean)
      self.var_ema(var)

    w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
    w_dtype = inputs.dtype

    if self.create_scale:
      scale = hk.get_parameter('scale', w_shape, w_dtype, self.scale_init)
    elif scale is None:
      scale = np.ones([], dtype=w_dtype)

    if self.create_offset:
      offset = hk.get_parameter('offset', w_shape, w_dtype, self.offset_init)
    elif offset is None:
      offset = np.zeros([], dtype=w_dtype)

    eps = jax.lax.convert_element_type(self.eps, var.dtype)
    inv = jax.lax.rsqrt(var + eps)
    scaled = scale * (inputs - mean) * inv + offset
    # It is technically not required to enforce zeros in the output
    scaled *= mask
    return scaled


UpdateFn = Callable[[jraph.NodeFeatures], jraph.NodeFeatures]


class GraphConvolution(hk.Module):
  """Returns a method that applies a Graph Convolution layer.

  This implementation also allows for edge features like the OGB sample code.

  Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,

  NOTE: This implementation does not add an activation after aggregation.
  If you are stacking layers, you may want to add an activation between
  each layer.

  Attributes:
    update_node_fn: function used to update the nodes. In the paper a single
      layer MLP is used.
    update_edge_fn: function used to aggregates the edge features.
      aggregate_nodes_fn: function used to aggregates the sender nodes.
    activation: to be applied. Default is relu.
    add_self_edges: whether to add self edges to nodes in the graph as in the
      paper definition of GCN. The number of graph.edges must match in either
      case Defaults to False.
    bidirectional: if True also messages in opposite edge direction are passed

  Returns:
    A method that applies a Graph Convolution layer.
  """

  def __init__(
      self,
      forw_update_node_fn: UpdateFn,
      forw_update_edge_fn: UpdateFn,
      backw_update_node_fn: Optional[UpdateFn] = None,
      backw_update_edge_fn: Optional[UpdateFn] = None,
      aggregate_nodes_fn: jraph.AggregateEdgesToNodesFn = jraph.segment_sum,
      activation: Callable[[Tensor], Tensor] = jax.nn.relu,
      add_self_edges: bool = False,
      name: Optional[str] = None):
    super().__init__(name)
    self.forw_update_node_fn = forw_update_node_fn
    self.forw_update_edge_fn = forw_update_edge_fn
    self.backw_update_node_fn = backw_update_node_fn
    self.backw_update_edge_fn = backw_update_edge_fn
    self.aggregate_nodes_fn = aggregate_nodes_fn
    self.activation = activation
    self.add_self_edges = add_self_edges

  def __call__(self, graph: jraph.GraphsTuple):
    """Applies a Graph Convolution layer."""
    orig_nodes, orig_edges, receivers, senders, _, _, _ = graph

    # Equivalent to jnp.sum(n_node), but jittable
    total_num_nodes = jax.tree_util.tree_leaves(orig_nodes)[0].shape[0]
    if self.add_self_edges:
      # We add self edges to the senders and receivers so that each node
      # includes itself in aggregation.
      # In principle, a `GraphsTuple` should partition by n_edge, but in
      # this case it is not required since a GCN is agnostic to whether
      # the `GraphsTuple` is a batch of graphs or a single large graph.
      conv_receivers = jnp.concatenate((receivers, jnp.arange(total_num_nodes)),
                                       axis=0)
      conv_senders = jnp.concatenate((senders, jnp.arange(total_num_nodes)),
                                     axis=0)
    else:
      conv_senders = senders
      conv_receivers = receivers

    # First pass nodes through the node updater.
    transf_nodes = self.forw_update_node_fn(orig_nodes)
    edges = self.forw_update_edge_fn(orig_edges)

    # Calculate the normalization values.
    count_edges = lambda x: jraph.segment_sum(  # pylint: disable=g-long-lambda
        jnp.ones_like(conv_senders), x, total_num_nodes)
    sender_degree = count_edges(conv_senders) + 1.
    receiver_degree = count_edges(conv_receivers) + 1.

    norm = (jax.lax.rsqrt(sender_degree)[conv_senders] *
            jax.lax.rsqrt(receiver_degree)[conv_receivers])[:, None]

    # Aggregate the pre normalized nodes.
    nodes = self.aggregate_nodes_fn(
        norm * self.activation(transf_nodes[conv_senders] + edges),
        conv_receivers, total_num_nodes)

    if self.backw_update_node_fn and self.backw_update_edge_fn:
      backw_nodes = self.backw_update_node_fn(orig_nodes)
      edges = self.backw_update_edge_fn(orig_edges)
      backw_nodes = self.aggregate_nodes_fn(
          norm * self.activation(transf_nodes[conv_receivers] + edges),
          conv_senders, total_num_nodes)
      nodes += backw_nodes

    root_emb = hk.get_parameter(
        'root_emb',
        shape=[1, transf_nodes.shape[-1]],
        dtype=jnp.float32,
        init=hk.initializers.RandomNormal()).astype(transf_nodes.dtype)

    nodes += self.activation(transf_nodes + root_emb) / receiver_degree[:, None]

    # pylint: enable=g-long-lambda
    return graph._replace(nodes=self.activation(nodes))


class MLP(hk.Module):
  """A simple MLP implementation."""

  def __init__(self,
               dim: int,
               activation=jax.nn.relu,
               n_layers: int = 2,
               with_norm: bool = True,
               final_activation: bool = True,
               name: Optional[str] = None):
    super().__init__(name=name)
    self.dim = dim
    self.activation = activation
    self.n_layers = n_layers
    self.with_norm = with_norm
    self.final_activation = final_activation

  def __call__(self, x: Tensor) -> Tensor:
    return mlp(
        x,
        dim=self.dim,
        activation=self.activation,
        n_layers=self.n_layers,
        with_norm=self.with_norm,
        final_activation=self.final_activation)


def mlp(x: Tensor,
        dim: int,
        activation=jax.nn.relu,
        n_layers: int = 2,
        with_norm: bool = True,
        final_activation: bool = True,
        name: Optional[str] = None):
  """Simple MLP layer with LayerNorm.

  Args:
    x: tensor of shape [b, *].
    dim: hidden and output dimensions, D.
    activation: a non-linearity. Default jax.nn.relu.
    n_layers: `int` number of layers. Default 2.
    with_norm: `bool` include LayerNorm. Default True.
    final_activation: `bool` include activation as last layer. Default True.
    name: name of the Sequential/MLP module.

  Returns:
    A tensor of shape [b, D]
  """
  layers = []
  for idx in range(n_layers):
    layers.append(hk.Linear(dim, name=f'{name}_linear{idx}' if name else None))
    if with_norm:
      norm = LayerNorm(
          axis=-1, name=f'{name}_layer_norm{idx}' if name else None)
      layers.append(norm)
    layers.append(activation)

  if not final_activation:
    layers = layers[:-1]

  return hk.Sequential(layers, name=name)(x)


class MultiHeadAttention(hk.Module):
  """Multi-headed attention (MHA) module.

  This module extends the haiku implementation by optional biases in the
  linear transofrmrations and dropout_p on the attention matrix.

  Rough sketch:
  - Compute keys (K), queries (Q), and values (V) as projections of inputs.
  - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
  - Output is another projection of WV^T.

  For more detail, see the original Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762.

  Glossary of shapes:
  - T: Sequence length.
  - D: Vector (embedding) size.
  - H: Number of attention heads.
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      w_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      dropout_p: float = 0.2,
      with_bias: bool = False,
      re_im_separate_projection: bool = False,
      name: Optional[str] = None,
  ):
    """Initialises the module.

    Args:
      num_heads: Number of independent attention heads (H).
      key_size: The size of keys (K) and queries used for attention.
      w_init: Initialiser for weights in the linear map.
      value_size: Optional size of the value projection (V). If None, defaults
        to the key size (K).
      model_size: Optional size of the output embedding (D'). If None, defaults
        to the key size multiplied by the number of heads (K * H).
      dropout_p: dropout_p after softmax of attention matrix.
      with_bias: if false (default), the linear projects will not have a bias.
      re_im_separate_projection: if true real and imaginary components are
        projected without weight sharing.
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads
    self.dropout_p = dropout_p
    self.with_bias = with_bias
    self.re_im_separate_projection = re_im_separate_projection

    self.w_init = w_init

  def __call__(
      self,
      query: Tensor,
      key: Tensor,
      value: Tensor,
      is_training: bool,
      logit_offset: Optional[Tensor] = None,
      mask: Optional[Tensor] = None,
  ) -> Tensor:
    """Computes (optionally masked) MHA with queries, keys & values.

    This module broadcasts over zero or more 'batch-like' leading dimensions.

    Args:
      query: Embeddings sequence used to compute queries; shape [..., T', D_q].
      key: Embeddings sequence used to compute keys; shape [..., T, D_k].
      value: Embeddings sequence used to compute values; shape [..., T, D_v].
      is_training: if True (not the default), dropout will not be applied. # #
      logit_offset: Optional offset/bias that is applied right before applying
        the softmax and before the mask for the attention scores (broadcast to
        [..., T', T, D_o]). A head specific linear transformation is applied.
      mask: Optional mask applied to attention weights; shape [..., H=1, T', T]
        or [..., T', T].

    Returns:
      A new sequence of embeddings, consisting of a projection of the
        attention-weighted value projections; shape [..., T', D'].
    """
    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    query_heads = projection(query, self.key_size, 'query')  # [T', H, Q=K]
    key_heads = projection(key, self.key_size, 'key')  # [T, H, K]
    value_heads = projection(value, self.value_size, 'value')  # [T, H, V]

    # Compute attention weights.
    attn_logits = jnp.einsum('...thd,...Thd->...htT', query_heads, key_heads)
    attn_logits = jnp.real(attn_logits)  # In case the logits are complex
    attn_logits = attn_logits / jnp.sqrt(self.key_size).astype(value.dtype)

    # E.g. to apply relative positional encodings or add edge bias
    if logit_offset is not None:
      logit_offset = hk.Linear(self.num_heads)(logit_offset)
      new_order = list(range(logit_offset.ndim - 3)) + [
          logit_offset.ndim - 1, logit_offset.ndim - 3, logit_offset.ndim - 2
      ]
      logit_offset = logit_offset.transpose(*new_order)
      attn_logits = attn_logits + logit_offset

    if mask is not None:
      if mask.ndim == attn_logits.ndim - 1:
        mask = mask[..., None, :, :]
      elif mask.ndim != attn_logits.ndim:
        raise ValueError(
            f'Mask dimensionality {mask.ndim} must match logits dimensionality '
            f'{attn_logits.ndim}.')
      attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

    if is_training and self.dropout_p > 0:
      attn_weights = hk.dropout(hk.next_rng_key(), self.dropout_p, attn_weights)

    # Weight the values by the attention and flatten the head vectors.
    attn = jnp.einsum('...htT,...Thd->...thd', attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    # Apply another projection to get the final embeddings.
    final_projection = hk.Linear(self.model_size, w_init=self.w_init)
    return final_projection(attn)  # [T', D']

  @hk.transparent
  def _linear_projection(
      self,
      x: Tensor,
      head_size: int,
      name: Optional[str] = None,
  ) -> Tensor:
    lin = hk.Linear(
        self.num_heads * head_size,
        w_init=self.w_init,
        name=name,
        with_bias=self.with_bias)
    if jnp.iscomplexobj(x):
      if self.re_im_separate_projection:
        y_re = lin(jnp.real(x))
        lin_im = hk.Linear(
            self.num_heads * head_size,
            w_init=self.w_init,
            name=name,
            with_bias=self.with_bias)
        y_im = lin_im(jnp.imag(x))
      else:
        y_re = lin(jnp.real(x))
        y_im = lin(jnp.imag(x))
      y = y_re + 1j * y_im
    else:
      y = lin(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))
