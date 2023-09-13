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
"""Misc utils covering helper function and positional encodings."""
import functools
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jraph
import numba
import numpy as np

# set the threading layer before any parallel target compilation
numba.config.THREADING_LAYER = 'safe'
numba.set_num_threads(max(int(3 / 4 * numba.get_num_threads()), 1))

Tensor = Union[np.ndarray, jnp.DeviceArray]

# Constant required for numerical reasons
EPS = 1e-8


def tp_fn_fp(prediction, target, mask=None):
  if mask is None:
    mask = jnp.ones_like(target)
  tp = ((prediction == target) & (prediction == 1) & mask).sum()
  fn = ((prediction != target) & (prediction == 0) & mask).sum()
  fp = ((prediction != target) & (prediction == 1) & mask).sum()
  return tp, fn, fp


def prec_rec_f1(tp, fn, fp):
  precision = tp / jnp.clip(tp + fp, a_min=1)
  recall = tp / jnp.clip(tp + fn, a_min=1)
  f1 = 2 * precision * recall / jnp.clip(precision + recall, a_min=1)
  return precision, recall, f1


def softmax_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    n_classes: int,
    only_punish_first_end_of_sequence_token: bool = False) -> jnp.DeviceArray:
  """Calculation of softmax loss for sequence of predictions/tokens."""
  targets_one_hot = jax.nn.one_hot(targets, n_classes)
  logits = jax.nn.log_softmax(logits)
  elem_loss = -jnp.sum(targets_one_hot * logits, axis=-1)

  if not only_punish_first_end_of_sequence_token:
    return jnp.mean(elem_loss, axis=-1)

  mask = jnp.cumsum(targets == n_classes - 1, axis=-1) < 2
  elem_loss *= mask
  return jnp.sum(elem_loss, axis=-1) / jnp.sum(mask, axis=-1)


def count_edges(idx, n_nodes):
  segment_sum = functools.partial(
      jraph.segment_sum,
      data=jnp.ones(1, dtype=jnp.int32),
      num_segments=n_nodes)
  return jax.vmap(segment_sum)(segment_ids=idx)


def dense_random_walk_matrix(graph: jraph.GraphsTuple,
                             reverse: bool = False) -> Tensor:
  """Returns the dense random walk matrix `A D^(-1)`.

  Args:
    graph: the explicitly batched graph (i.e. nodes are of shape [b, n, d]).
    reverse: If True the the graph is reversed. Default False.

  Returns:
    tensor of shape [b, n, n] containing the random walk probabilities
  """
  batch, n_nodes = graph.nodes.shape[:2]

  if reverse:
    senders = graph.receivers
    receivers = graph.senders
  else:
    senders = graph.senders
    receivers = graph.receivers

  deg = count_edges(senders, n_nodes)
  inv_deg = jnp.where(deg < 1, 0., 1. / deg)

  adj = jnp.zeros((batch, n_nodes, n_nodes), dtype=jnp.float32)
  assign = jax.vmap(lambda a, s, r, d: a.at[s, r].add(d[s]))
  adj = assign(adj, senders, receivers, inv_deg)
  # Once implemented swap next line with: adj = jnp.fill_diagonal(adj, deg < 1)
  adj = adj.at[:, jnp.arange(n_nodes), jnp.arange(n_nodes)].add(deg < 1)

  return adj


def k_step_random_walk(graph: jraph.GraphsTuple,
                       k: int = 3,
                       ppr_restart_p: Optional[float] = None,
                       reverse: bool = False) -> Tensor:
  """Returns the random walk matrices for k' in {1, ..., k} `I (A D^(-1))^k'`.

  Args:
    graph: the explicitly batched graph (i.e. nodes are of shape [b, n, d]).
    k: number of random walk steps.
    ppr_restart_p: if set, also the ppr is returned at `k + 1`-th dimension.
    reverse: If True the the graph is reversed. Default False.

  Returns:
    tensor of shape [b, n, n, k {+1}] containing the random walk probabilities
  """
  transition_probabilities = dense_random_walk_matrix(graph, reverse)

  rw_probabilities = transition_probabilities

  output = [rw_probabilities]
  for _ in range(k - 1):
    rw_probabilities = rw_probabilities @ transition_probabilities
    output.append(rw_probabilities)

  if ppr_restart_p:
    output.append(exact_ppr_from_trans(transition_probabilities, ppr_restart_p))

  output = jnp.stack(output, axis=-1)
  return output


def exact_ppr(graph: jraph.GraphsTuple,
              restart_p: float = 0.2,
              reverse: bool = False) -> Tensor:
  """Calculates the personalized page rank via matrix inversion.

  Args:
    graph: the explicitly batched graph (i.e. nodes are of shape [b, n, d]).
    restart_p: the personalized page rank restart probability. Default 0.2.
    reverse: If True the the graph is reversed. Default False.

  Returns:
    tensor of shape [b, n, n] containing the random walk probabilities
  """
  assert restart_p >= 0 and restart_p <= 1, 'Restart prob. must be in [0, 1]'

  transition_probabilities = dense_random_walk_matrix(graph, reverse)

  return exact_ppr_from_trans(transition_probabilities, restart_p)


def exact_ppr_from_trans(transition_prob: Tensor,
                         restart_p: float = 0.2) -> Tensor:
  """Calculates the personalized page rank via matrix inversion.

  Args:
    transition_prob: tensor of shape [b, n, n] containing transition
      probabilities.
    restart_p: the personalized page rank restart probability. Default 0.2.

  Returns:
    tensor of shape [b, n, n] containing the random walk probabilities
  """
  n_nodes = transition_prob.shape[-1]
  rw_matrix = jnp.eye(n_nodes) + (restart_p - 1) * transition_prob
  return restart_p * jnp.linalg.inv(rw_matrix)


def svd_encodings(graph: jraph.GraphsTuple, rank: int) -> Tensor:
  """SVD encodings following Hussain et al., Global Self-Attention as
  a Replacement for Graph Convolution, KDD 2022.

  Args:
      graph (jraph.GraphsTuple): to obtain the adjacency matrix.
      rank (int): for low rank approximation.

  Returns:
      Tensor: positional encodings.
  """
  batch, n_nodes = graph.nodes.shape[:2]
  senders = graph.senders
  receivers = graph.receivers

  adj = jnp.zeros((batch, n_nodes, n_nodes), dtype=jnp.float32)
  assign = jax.vmap(lambda a, s, r, d: a.at[s, r].add(d[s]))
  adj = assign(adj, senders, receivers, jnp.ones_like(senders))

  U, S, Vh = jax.lax.linalg.svd(adj)

  V = jnp.conjugate(jnp.transpose(Vh, axes=(0, 2, 1)))
  UV = jnp.stack((U, V), axis=-2)

  S = S[..., :rank]
  UV = UV[..., :rank]

  UV = UV * jnp.sqrt(S)[:, None, None, :]

  return UV.reshape(adj.shape[:-1] + (-1,))


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _norm_2d_along_first_dim_and_broadcast(array):
  """Equivalent to `linalg.norm(array, axis=0)[None, :] * ones_like(array)`."""
  output = np.zeros(array.shape, dtype=array.dtype)
  for i in numba.prange(array.shape[-1]):
    output[:, i] = np.linalg.norm(array[:, i])
  return output


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _max_2d_along_first_dim_and_broadcast(array):
  """Equivalent to `array.max(0)[None, :] * ones_like(array)`."""
  output = np.zeros(array.shape, dtype=array.dtype)
  for i in numba.prange(array.shape[-1]):
    output[:, i] = array[:, i].max()
  return output


@numba.njit([
    'Tuple((float64[::1], complex128[:, :], complex128[:, ::1]))(int64[:], ' +
    'int64[:], int64[:], int64, int64, int64, float64, b1, b1, b1, b1, b1)'
])
def eigv_magnetic_laplacian_numba(
    senders: np.ndarray, receivers: np.ndarray, n_node: np.ndarray,
    padded_nodes_size: int, k: int, k_excl: int, q: float, q_absolute: bool,
    norm_comps_sep: bool, l2_norm: bool, sign_rotate: bool,
    use_symmetric_norm: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """k *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.

  Args:
    senders: Origin of the edges of shape [m].
    receivers: Target of the edges of shape [m].
    n_node: array shape [2]
    padded_nodes_size: int the number of nodes including padding.
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    norm_comps_sep: If true first imaginary part is separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).

  Returns:
    array of shape [<= k] containing the k eigenvalues.
    array of shape [n, <= k] containing the k eigenvectors.
    array of shape [n, n] the laplacian.
  """
  # Handle -1 padding
  edges_padding_mask = senders >= 0

  adj = np.zeros(int(padded_nodes_size * padded_nodes_size), dtype=np.float64)
  linear_index = receivers + (senders * padded_nodes_size).astype(senders.dtype)
  adj[linear_index] = edges_padding_mask.astype(adj.dtype)
  adj = adj.reshape(padded_nodes_size, padded_nodes_size)
  # TODO(simongeisler): maybe also allow weighted matrices etc.
  adj = np.where(adj > 1, 1, adj)

  symmetric_adj = adj + adj.T
  symmetric_adj = np.where((adj != 0) & (adj.T != 0), symmetric_adj / 2,
                           symmetric_adj)

  symmetric_deg = symmetric_adj.sum(-2)

  if not q_absolute:
    m_imag = (adj != adj.T).sum() / 2
    m_imag = min(m_imag, n_node[0])
    q = q / (m_imag if m_imag > 0 else 1)

  theta = 1j * 2 * np.pi * q * (adj - adj.T)

  if use_symmetric_norm:
    inv_deg = np.zeros((padded_nodes_size, padded_nodes_size), dtype=np.float64)
    np.fill_diagonal(
        inv_deg, 1. / np.sqrt(np.where(symmetric_deg < 1, 1, symmetric_deg)))
    eye = np.eye(padded_nodes_size)
    inv_deg = inv_deg.astype(adj.dtype)
    deg = inv_deg @ symmetric_adj.astype(adj.dtype) @ inv_deg
    laplacian = eye - deg * np.exp(theta)

    mask = np.arange(padded_nodes_size) < n_node[:1]
    mask = np.expand_dims(mask, -1) & np.expand_dims(mask, 0)
    laplacian = mask.astype(adj.dtype) * laplacian
  else:
    deg = np.zeros((padded_nodes_size, padded_nodes_size), dtype=np.float64)
    np.fill_diagonal(deg, symmetric_deg)
    laplacian = deg - symmetric_adj * np.exp(theta)

  if q == 0:
    laplacian_r = np.real(laplacian)
    assert (laplacian_r == laplacian_r.T).all()
    # Avoid rounding errors of any sort
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_r)
    eigenvalues = eigenvalues[..., k_excl:k_excl + k]
    eigenvectors = eigenvectors[..., :, k_excl:k_excl + k]
    return eigenvalues.real, eigenvectors.astype(np.complex128), laplacian

  eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

  eigenvalues = eigenvalues[..., k_excl:k_excl + k]
  eigenvectors = eigenvectors[..., k_excl:k_excl + k]

  if sign_rotate:
    sign = np.zeros((eigenvectors.shape[1],), dtype=eigenvectors.dtype)
    for i in range(eigenvectors.shape[1]):
      argmax_i = np.abs(eigenvectors[:, i].real).argmax()
      sign[i] = np.sign(eigenvectors[argmax_i, i].real)
    eigenvectors = np.expand_dims(sign, 0) * eigenvectors

    argmax_imag_0 = eigenvectors[:, 0].imag.argmax()
    rotation = np.angle(eigenvectors[argmax_imag_0:argmax_imag_0 + 1])
    eigenvectors = eigenvectors * np.exp(-1j * rotation)

  if norm_comps_sep:
    # Only scale eigenvectors that seems to be more than numerical errors
    eps = EPS / np.sqrt(eigenvectors.shape[0])
    if l2_norm:
      scale_real = _norm_2d_along_first_dim_and_broadcast(np.real(eigenvectors))
      real = np.real(eigenvectors) / scale_real
    else:
      scale_real = _max_2d_along_first_dim_and_broadcast(
          np.abs(np.real(eigenvectors)))
      real = np.real(eigenvectors) / scale_real
    scale_mask = np.abs(
        np.real(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
    eigenvectors[:, scale_mask] = (
        real[:, scale_mask] + 1j * np.imag(eigenvectors)[:, scale_mask])

    if l2_norm:
      scale_imag = _norm_2d_along_first_dim_and_broadcast(np.imag(eigenvectors))
      imag = np.imag(eigenvectors) / scale_imag
    else:
      scale_imag = _max_2d_along_first_dim_and_broadcast(
          np.abs(np.imag(eigenvectors)))
      imag = np.imag(eigenvectors) / scale_imag
    scale_mask = np.abs(
        np.imag(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
    eigenvectors[:, scale_mask] = (
        np.real(eigenvectors)[:, scale_mask] + 1j * imag[:, scale_mask])
  elif not l2_norm:
    scale = _max_2d_along_first_dim_and_broadcast(np.absolute(eigenvectors))
    eigenvectors = eigenvectors / scale

  return eigenvalues.real, eigenvectors, laplacian


_eigv_magnetic_laplacian_numba_parallel_signature = [
    'Tuple((float64[:, :], complex128[:, :, :]))(int64[:, :], ' +
    'int64[:, :], int64[:, :], int64, int64, int64, float64, b1, b1, b1, b1, b1)'
]


@numba.njit(_eigv_magnetic_laplacian_numba_parallel_signature, parallel=True)
def eigv_magnetic_laplacian_numba_parallel(
    senders: np.ndarray,
    receivers: np.ndarray,
    n_node: np.ndarray,
    batch_size: int,
    k: int,
    k_excl: int,
    q: float,
    q_absolute: bool,
    norm_comps_sep: bool,
    l2_norm: bool,
    sign_rotate: bool,
    use_symmetric_norm: bool,
    # ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
) -> Tuple[np.ndarray, np.ndarray]:
  """k *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.

  Args:
    senders: Origin of the edges of shape [b, m].
    receivers: Target of the edges of shape [b, m].
    n_node: array shape [b, 2]
    batch_size: batch size b.
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    norm_comps_sep: If true first imaginary part is separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
      Will be treated as false if `norm_comps_sep` is true.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).

  Returns:
    list with arrays of shape [<= k] containing the k eigenvalues.
    list with arrays of shape [n_i, <= k] containing the k eigenvectors.
  """
  n = n_node.sum(-1).max()
  eigenvalues = np.zeros((batch_size, k), dtype=np.float64)
  eigenvectors = np.zeros((batch_size, n, k), dtype=np.complex128)

  n_node_wo_padding = n_node[:, 0]

  padding_maks = senders >= 0

  for i in numba.prange(0, batch_size, 1):
    eigenvalue, eigenvector, _ = eigv_magnetic_laplacian_numba(
        senders[i][padding_maks[i]],
        receivers[i][padding_maks[i]],
        n_node[i],
        padded_nodes_size=n_node_wo_padding[i],
        k=k,
        k_excl=k_excl,
        q=q,
        q_absolute=q_absolute,
        norm_comps_sep=norm_comps_sep,
        l2_norm=l2_norm,
        sign_rotate=sign_rotate,
        use_symmetric_norm=use_symmetric_norm)

    eigenvalues[i, :eigenvalue.shape[0]] = eigenvalue
    eigenvectors[i, :eigenvector.shape[0], :eigenvector.shape[1]] = eigenvector
  return eigenvalues, eigenvectors


def eigv_magnetic_laplacian_numba_batch(
    senders: np.ndarray,
    receivers: np.ndarray,
    n_node: np.ndarray,
    k: int = 10,
    k_excl: int = 1,
    q: float = 0.25,
    q_absolute: bool = True,
    norm_comps_sep: bool = False,
    l2_norm: bool = True,
    sign_rotate: bool = False,
    use_symmetric_norm: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
  """k *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.

  Args:
    senders: Origin of the edges of shape [m].
    receivers: Target of the edges of shape [m].
    n_node: array shape [b, 2]
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / n_node`.
    norm_comps_sep: If true real and imaginary part are separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).

  Returns:
    array of shape [k] containing the k eigenvalues.
    array of shape [n, k] containing the k eigenvectors.
  """
  eigenvalues, eigenvectors = eigv_magnetic_laplacian_numba_parallel(
      senders.astype(np.int64), receivers.astype(np.int64),
      n_node.astype(np.int64), senders.shape[0], int(k), int(k_excl), float(q),
      q_absolute, norm_comps_sep, l2_norm, sign_rotate, use_symmetric_norm)

  return eigenvalues, eigenvectors


def sinusoid_position_encoding(
    pos_seq: Tensor,
    hidden_size: int,
    max_timescale: float = 1e4,
    min_timescale: float = 2.,
) -> Tensor:
  """Creates sinusoidal encodings.

  Args:
    pos_seq: Tensor with positional ids.
    hidden_size: `int` dimension of the positional encoding vectors, D
    max_timescale: `int` maximum timescale for the frequency
    min_timescale: `int` minimum timescale for the frequency

  Returns:
    An array of shape [L, D]
  """
  freqs = np.arange(0, hidden_size, min_timescale)
  inv_freq = max_timescale**(-freqs / hidden_size)
  sinusoid_inp = jnp.einsum('bi,j->bij', pos_seq, inv_freq)
  pos_emb = jnp.concatenate(
      [jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
  return pos_emb
