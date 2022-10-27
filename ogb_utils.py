# MIT License

# Copyright (c) 2019 OGB Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Adapted to jax, source: https://github.com/snap-stanford/ogb/blob/c8efe8ec99d11279c80f2bcdbe1567675c1c5666/examples/graphproppred/code2/utils.py and converted to jax.
"""

# pylint: skip-file

import os

import haiku as hk
import jax.numpy as jnp
import jraph
import numpy as np
import pandas as pd
from sklearn import metrics

roc_auc_score, average_precision_score = metrics.roc_auc_score, metrics.average_precision_score


class ASTNodeEncoder(hk.Module):
  """Adapted from OGB.

  Input:

      x: default node feature. the first and second column represents node
      type and node attributes.
      depth: The depth of the node in the AST.
  Output:
      emb_dim-dimensional vector
  """

  def __init__(self,
               emb_dim: int,
               num_nodetypes: int,
               num_nodeattributes: int,
               max_depth: int,
               edge_dim: int,
               dtype=jnp.float32,
               name=None):
    super().__init__()

    self.emb_dim = emb_dim
    self.num_nodetypes = num_nodetypes
    self.num_nodeattributes = num_nodeattributes
    self.max_depth = max_depth
    self.edge_dim = edge_dim
    self.dtype = dtype

  def __call__(self, graph: jraph.GraphsTuple, depth: int, *args, **kwargs):
    x = graph.nodes

    type_encoder = hk.get_parameter(
        'type_encoder',
        shape=[self.num_nodetypes, self.emb_dim],
        dtype=self.dtype,
        init=hk.initializers.RandomNormal())
    attribute_encoder = hk.get_parameter(
        'attribute_encoder',
        shape=[self.num_nodeattributes, self.emb_dim],
        dtype=self.dtype,
        init=hk.initializers.RandomNormal())
    nodes = type_encoder[x[..., 0]] + attribute_encoder[x[..., 1]]

    if depth is not None:
      depth_encoder = hk.get_parameter(
          'depth_encoder',
          shape=[self.max_depth + 1, self.emb_dim],
          dtype=self.dtype,
          init=hk.initializers.RandomNormal())
      depth = jnp.where(depth > self.max_depth, self.max_depth, depth)
      nodes = nodes + depth_encoder[depth[..., 0]]

    edges = hk.Linear(self.edge_dim, with_bias=False)(graph.edges)

    graph = graph._replace(nodes=nodes, edges=edges)

    return graph


def get_vocab_mapping(seq_list, num_vocab):
  """Adapted from OGB.

    Input:

        seq_list: a list of sequences
        num_vocab: vocabulary size
    Output:
        vocab2idx:
            A dictionary that maps vocabulary into integer index.
            Additioanlly, we also index '__UNK__' and '__EOS__'
            '__UNK__' : out-of-vocabulary term
            '__EOS__' : end-of-sentence
        idx2vocab:
            A list that maps idx to actual vocabulary.
  """

  vocab_cnt = {}
  vocab_list = []
  for seq in seq_list:
    for w in seq:
      if w in vocab_cnt:
        vocab_cnt[w] += 1
      else:
        vocab_cnt[w] = 1
        vocab_list.append(w)

  cnt_list = np.array([vocab_cnt[w] for w in vocab_list])
  topvocab = np.argsort(-cnt_list, kind='stable')[:num_vocab]

  print('Coverage of top {} vocabulary:'.format(num_vocab))
  print(float(np.sum(cnt_list[topvocab])) / np.sum(cnt_list))

  vocab2idx = {
      vocab_list[vocab_idx]: idx for idx, vocab_idx in enumerate(topvocab)
  }
  idx2vocab = [vocab_list[vocab_idx] for vocab_idx in topvocab]

  # print(topvocab)
  # print([vocab_list[v] for v in topvocab[:10]])
  # print([vocab_list[v] for v in topvocab[-10:]])

  vocab2idx['__UNK__'] = num_vocab
  idx2vocab.append('__UNK__')

  vocab2idx['__EOS__'] = num_vocab + 1
  idx2vocab.append('__EOS__')

  # test the correspondence between vocab2idx and idx2vocab
  for idx, vocab in enumerate(idx2vocab):
    assert (idx == vocab2idx[vocab])

  # test that the idx of '__EOS__' is len(idx2vocab) - 1.
  # This fact will be used in decode_arr_to_seq, when finding __EOS__
  assert (vocab2idx['__EOS__'] == len(idx2vocab) - 1)

  return vocab2idx, idx2vocab


def augment_edge(edge_index, node_is_attributed, reverse: bool = True):
  """Adapted from OGB.

  Input:

      edge_index: senders and receivers stacked
      node_is_attributed: node attributes
      reverse: if true also reverse edges are added

  Output:
      data (edges are augmented in the following ways):
          data.edge_index: Added next-token edge. The inverse edges were
          also added.
          data.edge_attr (torch.Long):
              data.edge_attr[:,0]: whether it is AST edge (0) for
              next-token edge (1)
              data.edge_attr[:,1]: whether it is original direction (0) or
              inverse direction (1)
  """
  ##### AST edge
  edge_index_ast = edge_index
  edge_attr_ast = np.zeros((edge_index_ast.shape[1], 2))

  ##### Inverse AST edge
  if reverse:
    edge_index_ast_inverse = np.stack([edge_index_ast[1], edge_index_ast[0]],
                                      axis=0)

    edge_attr_ast_inverse = [
        np.zeros((edge_index_ast_inverse.shape[1], 1)),
        np.ones((edge_index_ast_inverse.shape[1], 1))
    ]
    edge_attr_ast_inverse = np.concatenate(edge_attr_ast_inverse, axis=1)

  ##### Next-token edge

  ## Obtain attributed nodes and get their indices in dfs order
  # attributed_node_idx = torch.where(data.node_is_attributed.view(-1,) == 1)[0]
  # attributed_node_idx_in_dfs_order = attributed_node_idx[torch.argsort(data.node_dfs_order[attributed_node_idx].view(-1,))]

  ## Since the nodes are already sorted in dfs ordering in our case, we can just do the following.
  attributed_node_idx_in_dfs_order = np.where(node_is_attributed[:, 0] == 1)[0]

  ## build next token edge
  # Given: attributed_node_idx_in_dfs_order
  #        [1, 3, 4, 5, 8, 9, 12]
  # Output:
  #    [[1, 3, 4, 5, 8, 9]
  #     [3, 4, 5, 8, 9, 12]
  edge_index_nextoken = [
      attributed_node_idx_in_dfs_order[:-1],
      attributed_node_idx_in_dfs_order[1:]
  ]
  edge_index_nextoken = np.stack(edge_index_nextoken, axis=0)
  edge_attr_nextoken = [
      np.ones((edge_index_nextoken.shape[1], 1)),
      np.zeros((edge_index_nextoken.shape[1], 1))
  ]
  edge_attr_nextoken = np.concatenate(edge_attr_nextoken, axis=1)

  ##### Inverse next-token edge
  if reverse:
    edge_index_nextoken_inverse = np.stack(
        [edge_index_nextoken[1], edge_index_nextoken[0]], axis=0)
    edge_attr_nextoken_inverse = np.ones((edge_index_nextoken.shape[1], 2))

    edge_index = [
        edge_index_ast, edge_index_ast_inverse, edge_index_nextoken,
        edge_index_nextoken_inverse
    ]
    edge_index = np.concatenate(edge_index, axis=1)

    edge_attr = [
        edge_attr_ast, edge_attr_ast_inverse, edge_attr_nextoken,
        edge_attr_nextoken_inverse
    ]
    edge_attr = np.concatenate(edge_attr, axis=0)
  else:
    edge_index = np.concatenate([edge_index_ast, edge_index_nextoken], axis=1)
    edge_attr = np.concatenate([edge_attr_ast, edge_attr_nextoken], axis=0)

  return edge_index, edge_attr


def encode_y_to_arr(seq, vocab2idx, max_seq_len):
  """Adapted from OGB.

  Input:

      data: PyG graph object
      output: add y_arr to data
  """

  y_arr = encode_seq_to_arr(seq, vocab2idx, max_seq_len)

  return y_arr


def encode_seq_to_arr(seq, vocab2idx, max_seq_len):
  """Adapted from OGB.

  Input:

      seq: A list of words
      output: add y_arr (jnp.array)
  """

  padded_seq = (
      seq[:max_seq_len] + ['__EOS__'] * max(0, max_seq_len - len(seq)))

  augmented_seq = [
      vocab2idx[w] if w in vocab2idx else vocab2idx['__UNK__']
      for w in padded_seq
  ]

  return jnp.array(augmented_seq)


def decode_arr_to_seq(arr, idx2vocab):
  """Adapted from OGB.

  Input: jnp.array 1d: y_arr Output: a sequence of words.

  IMPORTANT: we now filter for the unknown token to avoid inflating the FPs
  """
  # find the position of __EOS__ (the last vocab in idx2vocab)
  eos_idx_list = jnp.nonzero(arr == len(idx2vocab) - 1)[0]
  if len(eos_idx_list) > 0:
    clippted_arr = arr[:jnp.min(eos_idx_list)]  # find the smallest __EOS__
  else:
    clippted_arr = arr

  # Otherwise the UNK tokens are counted as a False Positive!
  clippted_arr = clippted_arr[clippted_arr != len(idx2vocab) - 2]

  return list(map(lambda x: idx2vocab[x], clippted_arr))


### Evaluator for graph classification
class Evaluator:

  def __init__(self, name, num_tasks: int = 1, eval_metric: str = 'F1'):
    self.name = name
    self.num_tasks = num_tasks
    self.eval_metric = eval_metric

  def _parse_and_check_input(self, input_dict):
    if (self.eval_metric == 'rocauc' or self.eval_metric == 'ap' or
        self.eval_metric == 'rmse' or self.eval_metric == 'acc'):
      if not 'y_true' in input_dict:
        raise RuntimeError('Missing key of y_true')
      if not 'y_pred' in input_dict:
        raise RuntimeError('Missing key of y_pred')

      y_true, y_pred = input_dict['y_true'], input_dict['y_pred']
      """
          y_true: numpy ndarray or torch tensor of shape (num_graphs,
          num_tasks)
          y_pred: numpy ndarray or torch tensor of shape (num_graphs,
          num_tasks)
      """
      ## check type
      if not isinstance(y_true, np.ndarray):
        raise RuntimeError(
            'Arguments to Evaluator need to be either numpy ndarray or torch tensor'
        )

      if not y_true.shape == y_pred.shape:
        raise RuntimeError('Shape of y_true and y_pred must be the same')

      if not y_true.ndim == 2:
        raise RuntimeError(
            'y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(
                y_true.ndim))

      if not y_true.shape[1] == self.num_tasks:
        raise RuntimeError(
            'Number of tasks for {} should be {} but {} given'.format(
                self.name, self.num_tasks, y_true.shape[1]))

      return y_true, y_pred

    elif self.eval_metric == 'F1':
      if not 'seq_ref' in input_dict:
        raise RuntimeError('Missing key of seq_ref')
      if not 'seq_pred' in input_dict:
        raise RuntimeError('Missing key of seq_pred')

      seq_ref, seq_pred = input_dict['seq_ref'], input_dict['seq_pred']

      if not isinstance(seq_ref, list):
        raise RuntimeError('seq_ref must be of type list')

      if not isinstance(seq_pred, list):
        raise RuntimeError('seq_pred must be of type list')

      if len(seq_ref) != len(seq_pred):
        raise RuntimeError('Length of seq_true and seq_pred should be the same')

      return seq_ref, seq_pred

    else:
      raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

  def eval(self, input_dict):

    if self.eval_metric == 'rocauc':
      y_true, y_pred = self._parse_and_check_input(input_dict)
      return self._eval_rocauc(y_true, y_pred)
    if self.eval_metric == 'ap':
      y_true, y_pred = self._parse_and_check_input(input_dict)
      return self._eval_ap(y_true, y_pred)
    elif self.eval_metric == 'rmse':
      y_true, y_pred = self._parse_and_check_input(input_dict)
      return self._eval_rmse(y_true, y_pred)
    elif self.eval_metric == 'acc':
      y_true, y_pred = self._parse_and_check_input(input_dict)
      return self._eval_acc(y_true, y_pred)
    elif self.eval_metric == 'F1':
      seq_ref, seq_pred = self._parse_and_check_input(input_dict)
      return self._eval_F1(seq_ref, seq_pred)
    else:
      raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

  @property
  def expected_input_format(self):
    desc = '==== Expected input format of Evaluator for {}\n'.format(self.name)
    if self.eval_metric == 'rocauc' or self.eval_metric == 'ap':
      desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
      desc += ('- y_true: numpy ndarray or torch tensor of shape (num_graphs, '
               'num_tasks)\n')
      desc += ('- y_pred: numpy ndarray or torch tensor of shape (num_graphs, '
               'num_tasks)\n')
      desc += 'where y_pred stores score values (for computing AUC score),\n'
      desc += 'num_task is {}, and '.format(self.num_tasks)
      desc += 'each row corresponds to one graph.\n'
      desc += 'nan values in y_true are ignored during evaluation.\n'
    elif self.eval_metric == 'rmse':
      desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
      desc += ('- y_true: numpy ndarray or torch tensor of shape (num_graphs, '
               'num_tasks)\n')
      desc += ('- y_pred: numpy ndarray or torch tensor of shape (num_graphs, '
               'num_tasks)\n')
      desc += 'where num_task is {}, and '.format(self.num_tasks)
      desc += 'each row corresponds to one graph.\n'
      desc += 'nan values in y_true are ignored during evaluation.\n'
    elif self.eval_metric == 'acc':
      desc += '{\'y_true\': y_true, \'y_pred\': y_pred}\n'
      desc += ('- y_true: numpy ndarray or torch tensor of shape (num_nodes '
               'num_tasks)\n')
      desc += ('- y_pred: numpy ndarray or torch tensor of shape (num_nodes '
               'num_tasks)\n')
      desc += 'where y_pred stores predicted class label (integer),\n'
      desc += 'num_task is {}, and '.format(self.num_tasks)
      desc += 'each row corresponds to one graph.\n'
    elif self.eval_metric == 'F1':
      desc += '{\'seq_ref\': seq_ref, \'seq_pred\': seq_pred}\n'
      desc += '- seq_ref: a list of lists of strings\n'
      desc += '- seq_pred: a list of lists of strings\n'
      desc += ('where seq_ref stores the reference sequences of sub-tokens, '
               'and\n')
      desc += 'seq_pred stores the predicted sequences of sub-tokens.\n'
    else:
      raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    return desc

  @property
  def expected_output_format(self):
    desc = '==== Expected output format of Evaluator for {}\n'.format(self.name)
    if self.eval_metric == 'rocauc':
      desc += '{\'rocauc\': rocauc}\n'
      desc += '- rocauc (float): ROC-AUC score averaged across {} task(s)\n'.format(
          self.num_tasks)
    elif self.eval_metric == 'ap':
      desc += '{\'ap\': ap}\n'
      desc += ('- ap (float): Average Precision (AP) score averaged across {} '
               'task(s)\n').format(self.num_tasks)
    elif self.eval_metric == 'rmse':
      desc += '{\'rmse\': rmse}\n'
      desc += ('- rmse (float): root mean squared error averaged across {} '
               'task(s)\n').format(self.num_tasks)
    elif self.eval_metric == 'acc':
      desc += '{\'acc\': acc}\n'
      desc += '- acc (float): Accuracy score averaged across {} task(s)\n'.format(
          self.num_tasks)
    elif self.eval_metric == 'F1':
      desc += '{\'F1\': F1}\n'
      desc += '- F1 (float): F1 score averaged over samples.\n'
    else:
      raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    return desc

  def _eval_rocauc(self, y_true, y_pred):
    """compute ROC-AUC averaged across tasks"""

    rocauc_list = []

    for i in range(y_true.shape[1]):
      #AUC is only defined when there is at least one positive data.
      if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        rocauc_list.append(
            roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
      raise RuntimeError(
          'No positively labeled data available. Cannot compute ROC-AUC.')

    return {'rocauc': sum(rocauc_list) / len(rocauc_list)}

  def _eval_ap(self, y_true, y_pred):
    """compute Average Precision (AP) averaged across tasks"""

    ap_list = []

    for i in range(y_true.shape[1]):
      #AUC is only defined when there is at least one positive data.
      if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
        # ignore nan values
        is_labeled = y_true[:, i] == y_true[:, i]
        ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled,
                                                                   i])

        ap_list.append(ap)

    if len(ap_list) == 0:
      raise RuntimeError(
          'No positively labeled data available. Cannot compute Average Precision.'
      )

    return {'ap': sum(ap_list) / len(ap_list)}

  def _eval_rmse(self, y_true, y_pred):
    """compute RMSE score averaged across tasks"""
    rmse_list = []

    for i in range(y_true.shape[1]):
      # ignore nan values
      is_labeled = y_true[:, i] == y_true[:, i]
      rmse_list.append(
          np.sqrt(((y_true[is_labeled] - y_pred[is_labeled])**2).mean()))

    return {'rmse': sum(rmse_list) / len(rmse_list)}

  def _eval_acc(self, y_true, y_pred):
    acc_list = []

    for i in range(y_true.shape[1]):
      is_labeled = y_true[:, i] == y_true[:, i]
      correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
      acc_list.append(float(np.sum(correct)) / len(correct))

    return {'acc': sum(acc_list) / len(acc_list)}

  def _eval_F1(self, seq_ref, seq_pred):
    """compute F1 score averaged over samples"""

    precision_list = []
    recall_list = []
    f1_list = []

    for l, p in zip(seq_ref, seq_pred):
      label = set(l)
      prediction = set(p)
      true_positive = len(label.intersection(prediction))
      false_positive = len(prediction - label)
      false_negative = len(label - prediction)

      if true_positive + false_positive > 0:
        precision = true_positive / (true_positive + false_positive)
      else:
        precision = 0

      if true_positive + false_negative > 0:
        recall = true_positive / (true_positive + false_negative)
      else:
        recall = 0
      if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
      else:
        f1 = 0

      precision_list.append(precision)
      recall_list.append(recall)
      f1_list.append(f1)

    return {
        'precision': np.average(precision_list),
        'recall': np.average(recall_list),
        'F1': np.average(f1_list)
    }
