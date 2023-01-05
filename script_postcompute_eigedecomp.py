from functools import partial
import os
from multiprocessing import Pool
from typing import Sequence

import numpy as np
from absl import app
from absl import flags
from tqdm import tqdm

from utils import eigv_magnetic_laplacian_numba


_SPLITS = flags.DEFINE_list(
    name="splits",
    default=["train", "valid", "test"],
    help="Data splits for which to precompute the eigenvectors.",
)
_DATA_ROOT = flags.DEFINE_string(
    name="data_root",
    default="data/sorting_network/7to11_12_13to16",
    help="Data root for the new dataset that contains the eigenvectors."
)
_NUM_CPU = flags.DEFINE_integer(
    name="num_cpu",
    default=1,
    help="Number of CPUs to use for the eigenvector calculation."
)

configs = [
  dict(k=25, k_excl=0, q=0.25,
       q_absolute=False, norm_comps_sep=False,
       sign_rotate=True, use_symmetric_norm=True),
  dict(k=25, k_excl=0, q=0,
       q_absolute=False, norm_comps_sep=False,
       sign_rotate=True, use_symmetric_norm=True)
]


def process_graph(graph_tuple, k, k_excl, q, q_absolute, norm_comps_sep,
                  sign_rotate, use_symmetric_norm, l2_norm=True):
  """Compute the first `k` maglap eigenvectors and values for a graph."""

  n_node = graph_tuple.n_node[0]
  eigenvalues = np.zeros(shape=(k), dtype=np.float32)
  eigenvectors = np.zeros(
    shape=(n_node, k), dtype=np.complex64)

  n_eigv = min(k, n_node)
  eigenvalues[:n_eigv], eigenvectors[:, :n_eigv], _ = eigv_magnetic_laplacian_numba(
      senders=graph_tuple.senders.astype(np.int64),
      receivers=graph_tuple.receivers.astype(np.int64),
      n_node=np.array([graph_tuple.n_node[0], 0]),
      padded_nodes_size=graph_tuple.n_node[0],
      k=k,
      k_excl=k_excl,
      q=q,
      q_absolute=q_absolute,
      norm_comps_sep=norm_comps_sep,
      l2_norm=l2_norm,
      sign_rotate=sign_rotate,
      use_symmetric_norm=use_symmetric_norm
  )

  if q == 0:
    eigenvectors = eigenvectors.real

  return eigenvalues, eigenvectors


def precalc_and_append(graph, configs):
  precomputed = tuple(
      (config, *process_graph(graph[0], **config)) for config in configs)

  if not isinstance(graph[0].globals, dict):
    graph[0] = graph[0]._replace(
        globals={'eigendecomposition': precomputed})
  else:
    graph[0].globals['eigendecomposition'] = precomputed

  return graph


def main(argv: Sequence[str]) -> None:

  for split in _SPLITS.value:
    base_path = os.path.join(_DATA_ROOT.value, split)
    for file in tqdm(os.listdir(os.path.join(base_path))):
      file_path = os.path.join(base_path, file)

      buffer = list(np.load(file_path, allow_pickle=True)["data"])
      with Pool(_NUM_CPU.value) as p:
        buffer = p.map(partial(precalc_and_append, configs=configs), buffer)

      np.savez_compressed(file_path, data=np.array(buffer, dtype='object'))


if __name__ == '__main__':
  app.run(main)
