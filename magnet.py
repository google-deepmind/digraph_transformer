# Fairly naive implementation of MagNet (Zhang et al. 2021) using dense matrices.

from typing import Any, Callable, Optional, Tuple, Union

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import jraph

# pylint: disable=g-bad-import-order
import layers
import utils

# Inline important classes and methods
CallArgs = layers.CallArgs
mlp, MLP = layers.mlp, layers.MLP

Tensor = utils.Tensor


def magnetic_laplacian(
    graph: jraph.GraphsTuple,
    q: float = 0.25,
    q_absolute: bool = False,
    use_symmetric_norm: bool = False) -> Tuple[Tensor]:
  """k *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.

  Args:
    graph: the explicitly batched graph (i.e. nodes are of shape [b, n, d]).
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    use_symmetric_norm: symmetric (True) or row normalization (False).

  Returns:
    tensor of shape [b, n, n] the laplacian.
  """
  batch, n_nodes = (next(iter(graph.nodes.items()))[1] if isinstance(
      graph.nodes, dict) else graph.nodes).shape[:2]

  assign = jax.vmap(lambda a, s, r, d: a.at[s, r].add(d[s]))

  # Handle -1 padding
  edges_padding_mask = graph.senders >= 0

  adj = jnp.zeros((batch, n_nodes, n_nodes), dtype=jnp.float32)
  adj = assign(adj, graph.senders, graph.receivers, edges_padding_mask)
  adj = jnp.where(adj > 1, 1., adj)

  transpose_idx = tuple(range(adj.ndim - 2)) + (adj.ndim - 1, adj.ndim - 2)
  adj_transposed = adj.transpose(*transpose_idx)

  symmetric_adj = adj + adj_transposed
  symmetric_adj = jnp.where((adj != 0) & (adj_transposed != 0),
                            symmetric_adj / 2, symmetric_adj)

  symmetric_deg = symmetric_adj.sum(-2)

  if not q_absolute:
    m_imag = (adj != adj_transposed).sum((-2, -1)) / 2
    m_imag = jnp.where(m_imag > graph.n_node[..., 0], graph.n_node[..., 0],
                       m_imag)
    q = q / jnp.where(m_imag > 0, m_imag, 1)
  else:
    q = jnp.full(batch, q)

  theta = 1j * 2 * jnp.pi * q[..., None, None] * (adj - adj_transposed)

  if use_symmetric_norm:
    inv_deg = jnp.zeros((batch, n_nodes, n_nodes), dtype=jnp.float32)
    inv_deg_diag = inv_deg.at[:, jnp.arange(n_nodes), jnp.arange(n_nodes)]
    inv_deg = inv_deg_diag.set(jax.lax.rsqrt(jnp.clip(symmetric_deg, 1)))
    laplacian = jnp.eye(
        n_nodes) - (inv_deg @ symmetric_adj @ inv_deg) * jnp.exp(theta)

    idx = jnp.tile(jnp.arange(n_nodes)[None, :], [batch, 1])
    mask = idx < graph.n_node[:, :1]
    mask = mask[..., None] * mask[..., None, :]
    laplacian = mask * laplacian
  else:
    deg = jnp.zeros((batch, n_nodes, n_nodes), dtype=jnp.float32)
    deg_diag = deg.at[:, jnp.arange(n_nodes), jnp.arange(n_nodes)]
    deg = deg_diag.set(symmetric_deg)
    laplacian = deg - symmetric_adj * jnp.exp(theta)

  return laplacian


def complex_relu(value):
  mask = 1.0 * (value.real >= 0)
  return mask * value


class MagNet(hk.Module):
  """MagNet.

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
      bidirectional: If True, edhes in both directions are considered (only
        relevant for `gnn_type=gnn`).
      q: potential of magnetic laplacian.
      q_absolute: if False we use the graph specific normalizer.
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
               q: float = 0.25,
               q_absolute: float = 0.25,
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
    self.q = q
    self.q_absolute = q_absolute

    if kwargs:
      logging.info('GNN.__init__() received unexpected kwargs: %s', kwargs)


  def _update_fn(self, features: Tensor, sender_features: Tensor,
                 receiver_features: Tensor, globals_: Any,
                 exclude_features: bool, name: str):
    stack = [features, sender_features, receiver_features]
    if exclude_features:
      stack = stack[1:]
    if not self.bidirectional:
      stack = stack[:-1]
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
    
    x = graph.nodes

    maglap = magnetic_laplacian(
        graph, q=self.q, q_absolute=self.q_absolute, use_symmetric_norm=True)
    max_eigenvalue = jnp.linalg.eigvalsh(maglap).max()
    t_0 = jnp.eye(x.shape[-2])
    t_1 = 2 / max_eigenvalue * maglap - t_0[None, ...]

    for _ in range(self.k_hop):
      l_0 = hk.Linear(self.d_model)
      x_0 = t_0 @ x
      x_0 = l_0(x_0.real) + 1j * l_0(x_0.imag)
      l_1 = hk.Linear(self.d_model, with_bias=False)
      x_1 = t_1 @ x
      x_1 = l_1(x_1.real) + 1j * l_1(x_1.imag)
      x = complex_relu(x_0 + x_1)

    graph = graph._replace(nodes=jnp.concatenate((x.real, x.imag), axis=-1))
    return graph
  
