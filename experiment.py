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
"""A jaxline experiment for predicting the correctness of sorting networks or for graph property prediction on OGB Code2 dataset.

https://ogb.stanford.edu/docs/graphprop/
"""

import datetime
import functools
import os
import threading
import traceback
from typing import Dict, NamedTuple, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import chex
import dill
import haiku as hk
import jax
import jax.numpy as jnp
from jaxline import experiment
# from jaxline import platform
from jaxline import utils as jutils
import jraph
import numpy as np
from ogb.graphproppred import GraphPropPredDataset, Evaluator
import optax
import tensorflow as tf
import tree

# pylint: disable=g-bad-import-order
import dataset_utils
import engine
import models
import ogb_utils
from utils import tp_fn_fp, prec_rec_f1

try:
  import wandb
except:
  wandb = None


hk.experimental.profiler_name_scopes(enabled=True)
jax.config.parse_flags_with_absl()

FLAGS = flags.FLAGS


class _Predictions(NamedTuple):
  predictions: np.ndarray
  indices: np.ndarray
  target: Optional[np.ndarray]


def _sort_predictions_by_indices(predictions: _Predictions):
  sorted_order = np.argsort(predictions.indices)
  return _Predictions(
      predictions=predictions.predictions[sorted_order],
      indices=predictions.indices[sorted_order],
      target=(None if predictions.target is None
              else predictions.target[sorted_order]))


def _disable_gpu_for_tf():
  tf.config.set_visible_devices([], "GPU")  # Hide local GPUs.
  os.environ["CUDA_VISIBLE_DEVICES"] = ""


def maybe_get_first(xs):
  if jax.device_count() == 1:
    return xs
  return jutils.get_first(xs)


def maybe_bcast_local_devices(xs):
  if jax.device_count() == 1:
    return xs
  return jutils.bcast_local_devices(xs)


class Experiment(experiment.AbstractExperiment):
  """OGB Graph Property Prediction GraphNet experiment."""

  # Holds a map from object properties that will be checkpointed to their name
  # within a checkpoint. Currently it is assumed that these are all sharded
  # device arrays.
  CHECKPOINT_ATTRS = {
      "_params": "params",
      "_opt_state": "opt_state",
      "_network_state": "network_state"
  }

  NON_BROADCAST_CHECKPOINT_ATTRS = {
      # Predictions are written by the evaluator and hence are
      # present only in 'best' ckpt.
      "_test_predictions": "test_predictions",
      "_valid_predictions": "valid_predictions",
  }

  def __init__(self, mode, init_rng, config):
    """Initializes experiment."""
    super(Experiment, self).__init__(mode=mode, init_rng=init_rng)
    self.mode = mode
    self.init_rng = init_rng
    self.config = config
    self.dataset_config = config.dataset
    self._test_predictions: _Predictions = None
    self._valid_predictions: _Predictions = None
    if mode not in ("train", "eval", "train_eval_multithreaded"):
      raise ValueError(f"Invalid mode {mode}.")

    self.loss = None
    self.forward = None
    # Needed for checkpoint restore.
    self._params = None
    self._network_state = None
    self._opt_state = None
    # For eval on ogb
    self.ogb_targets = None
    self.ogb_evaluator = None

  #  _             _
  # | |_ _ __ __ _(_)_ __
  # | __| "__/ _` | | "_ \
  # | |_| | | (_| | | | | |
  #  \__|_|  \__,_|_|_| |_|
  #
  def _loss(self, *graph) -> chex.ArrayTree:
    assert self.mode == "train"

    graph = jraph.GraphsTuple(*graph)
    model_instance = models.get_model(self.config.dataset,
                                      self.config.pmap_axis,
                                      **self.config.model)
    loss, prediction = model_instance.loss(graph)

    target = graph.globals["target"]
    scalars = {"loss": loss.mean()}
    if self.dataset_config.num_classes > 0:
      prediction = jnp.argmax(prediction, axis=-1)
      if self.dataset_config.name.startswith("dist"):
        mask = target >= 0
      else:
        target = target[..., 0, :]
        mask = jnp.ones_like(target, dtype=jnp.bool_)
      accuracy = ((prediction == target) & mask).sum() / mask.sum()
      scalars["accuracy"] = accuracy

    if self.dataset_config.num_classes == 2:
      scalars["tp"], scalars["fn"], scalars["fp"] = tp_fn_fp(
        prediction, target, mask)
      scalars["precision"], scalars["recall"], scalars["f1"] = prec_rec_f1(
        scalars["tp"], scalars["fn"], scalars["fp"]
      )

    return loss.sum(), scalars

  def _grads_stats(self,
                   grads: chex.ArrayTree,
                   divisor=1) -> Dict[str, jnp.DeviceArray]:

    def stack(x: chex.ArrayTree) -> jnp.DeviceArray:
      return jnp.array(jax.tree_util.tree_leaves(x))

    return {
        "gradient_mean":
            jnp.mean(stack(jax.tree_map(jnp.mean, grads))) / divisor,
        "gradient_absmean":
            jnp.mean(stack(jax.tree_map(lambda x: jnp.abs(x).mean(), grads))) /
            divisor,
        "gradient_min":
            jnp.min(stack(jax.tree_map(jnp.min, grads))) / divisor,
        "gradient_max":
            jnp.max(stack(jax.tree_map(jnp.max, grads))) / divisor,
    }

  def _update_parameters(self, params, network_state, opt_state, global_step,
                         rng, graph):
    """Updates parameters."""

    def get_loss(params, network_state, rng, *graph):
      (loss, scalars), network_state = self.loss.apply(params, network_state,
                                                       rng, *graph)
      loss = loss.mean()
      return loss, (scalars, network_state)

    grad_loss_fn = jax.grad(get_loss, has_aux=True)
    out = grad_loss_fn(params, network_state, rng, *graph)
    grads, (scalars, network_state) = out
    if jax.device_count() > 1:
      grads = jax.lax.psum(grads, axis_name=self.config.pmap_axis)
    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    scalars.update(self._grads_stats(grads, 1))
    return params, network_state, opt_state, scalars

  def _train_init(self):
    self.loss = hk.transform_with_state(self._loss)

    self._train_input = jutils.py_prefetch(
        lambda: self._build_numpy_dataset_iterator("train"), buffer_size=5)
    init_stacked_graphs = next(self._train_input)
    if jax.device_count() > 1:
      init_key = jutils.bcast_local_devices(self.init_rng)
      p_init = jax.pmap(self.loss.init, axis_name=self.config.pmap_axis)
    else:
      init_key = self.init_rng
      p_init = self.loss.init
    self._params, self._network_state = p_init(init_key, *init_stacked_graphs)

    # Learning rate scheduling.
    lr_schedule = optax.warmup_cosine_decay_schedule(
        **self.config.optimizer.lr_schedule)

    # @optax.inject_hyperparams
    def build_optimizer(learning_rate, kwargs):
      optimizer = getattr(optax, self.config.optimizer.name)(
          learning_rate=learning_rate, **kwargs)
      if not self.config.optimizer.use_agc:
        return optimizer
      else:
        return optax.chain(
            optimizer,
            optax.adaptive_grad_clip(**self.config.optimizer.agc_kwargs),
        )

    self.optimizer = build_optimizer(lr_schedule,
                                     self.config.optimizer.optimizer_kwargs)

    if self.config.optimizer.accumulate_gradient_k > 1:
      g_k = self.config.optimizer.accumulate_gradient_k
      self.optimizer = optax.MultiSteps(
          self.optimizer, use_grad_mean=False, every_k_schedule=g_k)

    if jax.device_count() > 1:
      self._opt_state = jax.pmap(self.optimizer.init)(self._params)
      self.update_parameters = jax.pmap(
          self._update_parameters,
          axis_name=self.config.pmap_axis,
          donate_argnums=(0, 1, 2))
    else:
      self._opt_state = jax.jit(self.optimizer.init)(self._params)
      self.update_parameters = jax.jit(
          self._update_parameters,
          donate_argnums=(0, 1, 2))

  def step(self, global_step, rng, **unused_args):
    """See base class."""
    if self.loss is None:
      self._train_init()

    graph = next(self._train_input)
    if jax.device_count() == 1:
      rng = rng[0]
    out = self.update_parameters(self._params, self._network_state,
                                 self._opt_state, global_step, rng, graph)
    (self._params, self._network_state, self._opt_state, scalars) = out

    scalars = maybe_get_first(scalars)
    scalars = jax.tree_map(lambda x: x.item(), scalars)

    scalars["local_device_count"] = jax.local_device_count()
    scalars["device_count"] = jax.device_count()
    scalars["process_count"] = jax.process_count()
    scalars["process_index"] = jax.process_index()

    return scalars

  def _build_numpy_dataset_iterator(self, split: str):
    """See base class."""
    batch_size = (
        self.config.training.batch_size
        if self.mode == "train" else self.config.evaluation.batch_size)

    max_number_of_instances = -1
    if split != "train":
      max_number_of_instances = self.config.evaluation.max_number_of_instances

    path = os.path.join(self.config.data_root, self.dataset_config.path)
    data_generator = functools.partial(dataset_utils.dataset_generator, path,
                                       split)
    example = next(data_generator())
    signature_from_example = tree.map_structure(_numpy_to_tensor_spec, example)
    dataset = tf.data.Dataset.from_generator(
        data_generator, output_signature=signature_from_example)

    return dataset_utils.build_dataset_iterator(
        self.dataset_config,
        dataset,
        batch_size=batch_size,
        debug=self.config.debug,
        is_training=self.mode == "train",
        posenc_config=self.config.model.posenc_config,
        max_number_of_instances=max_number_of_instances,
        **self.config.dataset_config)

  #                  _
  #   _____   ____ _| |
  #  / _ \ \ / / _` | |
  # |  __/\ V / (_| | |
  #  \___| \_/ \__,_|_|
  #
  def _forward(self, *graph) -> Tuple[np.ndarray, np.ndarray]:
    assert "eval" in self.mode

    graph = jraph.GraphsTuple(*graph)
    model_instance = models.get_model(self.dataset_config,
                                      self.config.pmap_axis,
                                      **self.config.model)
    loss, prediction = model_instance.loss(graph, is_training=False)
    return loss, prediction

  def _eval_init(self):
    self.forward = hk.transform_with_state(self._forward)
    self.eval_apply = jax.jit(self.forward.apply)

  def _ogb_performance_metrics(self, loss: np.ndarray, prediction: np.ndarray,
                               target: np.ndarray, target_raw: np.ndarray):
    """Creates unnormalised values for accumulation."""
    prediction_ = np.argmax(prediction, axis=-1)
    accuracy = prediction_ == target
    values = {"accuracy": accuracy.sum(), "loss": loss.sum()}
    counts = {"accuracy": accuracy.size, "loss": loss.size}
    scalars = {"values": values, "counts": counts}

    if self.dataset_config.idx2vocab is None:
      meta = np.load(os.path.join(self.config.data_root,
                                  self.config.dataset.path, 'meta.npz'),
                     allow_pickle=True)['data'].item()
      self.dataset_config.idx2vocab = meta['idx2vocab']
      self.dataset_config.vocab2idx = meta['vocab2idx']

    arr_to_seq = functools.partial(
        ogb_utils.decode_arr_to_seq, idx2vocab=self.dataset_config.idx2vocab)
    seq_pred = [arr_to_seq(seq) for seq in prediction_]

    seq_ref = [[el for el in seq if el]
               for seq in target_raw[:, 0].astype("U")]

    return scalars, seq_ref, seq_pred

  def _sn_performance_metrics(self, loss: np.ndarray, prediction: np.ndarray,
                              target: np.ndarray, seq_len: int):
    """Creates unnormalised values for accumulation."""
    prediction_ = np.argmax(prediction, axis=-1)
    target = target[..., 0, :]
    accuracy = prediction_ == target
    accuracy_sum = accuracy.sum()

    values = {
        "accuracy": accuracy_sum,
        f"accuracy_{seq_len}": accuracy_sum,
        "loss": loss.sum()
    }
    tp, fn, fp = tp_fn_fp(prediction_, target)
    values["tp"] = values[f"tp_{seq_len}"] = tp
    values["fn"] = values[f"fn_{seq_len}"] = fn
    values["fp"] = values[f"fp_{seq_len}"] = fp
    counts = {
        "tp": 0,
        "fn": 0,
        "fp": 0,
        f"tp_{seq_len}": 0,
        f"fn_{seq_len}": 0,
        f"fp_{seq_len}": 0,
        "accuracy": accuracy.size,
        f"accuracy_{seq_len}": accuracy.size,
        "loss": loss.size
    }
    scalars = {"values": values, "counts": counts}

    return scalars, target, []

  def _dist_performance_metrics(self, loss: np.ndarray, prediction: np.ndarray,
                                target: np.ndarray, skip=True):
    """Creates unnormalised values for accumulation."""
    if skip:
      return {"values": {}, "counts": {}}, target, loss

    mask = target >= 0

    if self.dataset_config.num_classes > 0:
      prediction_ = np.argmax(prediction, axis=-1)
      accuracy = (prediction_ == target) & mask
      accuracy_sum = accuracy.sum()

      values = {"loss": loss.sum(), "accuracy": accuracy_sum}
      values["tp"], values["fn"], values["fp"] = tp_fn_fp(
          prediction_, target, mask)

      counts = {
          "tp": 0,
          "fn": 0,
          "fp": 0,
          "loss": loss.size,
          "accuracy": mask.sum(),
      }

      nb_nodes = mask[..., 0].sum(-1) + 1
      for nb in np.unique(nb_nodes):
        inst_mask = nb_nodes == nb

        values[f"loss_{nb}"] = loss[inst_mask].sum()
        values[f"accuracy_{nb}"] = accuracy[inst_mask].sum()
        values[f"tp_{nb}"], values[f"fn_{nb}"], values[f"fp_{nb}"] = tp_fn_fp(
            prediction_[inst_mask], target[inst_mask], mask[inst_mask])

        counts.update({
            f"tp_{nb}": 0,
            f"fn_{nb}": 0,
            f"fp_{nb}": 0,
            f"loss_{nb}": inst_mask.sum(),
            f"accuracy_{nb}": mask[inst_mask].sum(),
        })
    else:

      values = {
          "loss": loss.sum(),
      }
      counts = {
          "loss": loss.size,
      }

      nb_nodes = mask[..., 0].sum(-1) + 1
      for nb in np.unique(nb_nodes):
        inst_mask = nb_nodes == nb
        values.update({
            f"loss_{nb}": loss[inst_mask].sum(),
        })
        counts.update({
            f"loss_{nb}": inst_mask.sum(),
        })

    scalars = {"values": values, "counts": counts}

    return scalars, target, []
    # return scalars, target.flatten(), prediction[..., 0].flatten()

  def _get_prediction(self, params, state, rng,
                      graph) -> Tuple[np.ndarray, np.ndarray]:
    """Returns predictions for all the graphs in the dataset split."""
    model_output, _ = self.eval_apply(params, state, rng, *graph)
    return model_output

  def _sum_agg_two_level_struct_with_default(self, structure, default=0):
    """Two level version of `tree.map_structure(lambda *l: sum(l), *all_scalars)` that handles missing keys.
    """
    accum = {}
    for element in structure:
      for ckey, container in element.items():
        if ckey not in accum:
          accum[ckey] = {}
        for vkey, values in container.items():
          if vkey not in accum[ckey]:
            accum[ckey][vkey] = default
          accum[ckey][vkey] += values
    return accum

  def _get_predictions(self, params, state, rng, m2_offest, m1_offset, split):
    graph_iterator = jutils.py_prefetch(
        lambda: self._build_numpy_dataset_iterator(split), buffer_size=5)

    all_scalars = []
    all_seq_refs = []
    all_seq_preds = []
    predictions = []
    graph_indices = []
    for i, graph in enumerate(graph_iterator):
      # Since jax does not support strings we cannot pass it to the model
      target_raw = None
      if "target_raw" in graph.globals:
        target_raw = graph.globals["target_raw"]
        del graph.globals["target_raw"]
      loss, prediction = self._get_prediction(params, state, rng, graph)
      if self.dataset_config.num_classes > 0:
        prediction = jax.nn.softmax(prediction, axis=-1)
        prediction = prediction.at[..., -2].add(-m2_offest)
        prediction = prediction.at[..., -1].add(-m1_offset)
      if "target" in graph.globals and not jnp.isnan(
              graph.globals["target"]).any():
        if self.dataset_config.name.startswith("dist"):
          scalars, seq_ref, seq_pred = self._dist_performance_metrics(
              loss, prediction, graph.globals["target"])
        elif self.dataset_config.name.startswith("sn"):
          scalars, seq_ref, seq_pred = self._sn_performance_metrics(
              loss, prediction, graph.globals["target"],
              int(graph.nodes["node_feat"][..., 1:3].max().item()) + 1)
        else:
          scalars, seq_ref, seq_pred = self._ogb_performance_metrics(
              loss, prediction, graph.globals["target"][..., 0, :], target_raw)
        all_scalars.append(scalars)
        all_seq_refs.extend(seq_ref)
        all_seq_preds.extend(seq_pred)
      predictions.append(prediction)
      graph_indices.append(graph.globals["graph_index"][:, 0])

      if i % 50 == 0:
        logging.info("Generated predictions for %d batches so far", i + 1)

    if "dist" in self.dataset_config.name:
      with jax.default_device(jax.devices("cpu")[0]):
        all_scalars = [
          self._dist_performance_metrics(
              np.stack(all_seq_preds), np.concatenate(predictions),
              np.stack(all_seq_refs), skip=False)[0]
        ]

    target = None
    if isinstance(all_seq_refs[0], jnp.ndarray):
      target = np.stack(all_seq_refs)
    predictions = _sort_predictions_by_indices(
        _Predictions(predictions=np.concatenate(predictions),
                     indices=np.concatenate(graph_indices),
                     target=target))

    if all_scalars:
      # Sum over graphs in the dataset.
      accum_scalars = self._sum_agg_two_level_struct_with_default(all_scalars)
      scalars = tree.map_structure(lambda x, y: x / max(y, 1),
                                   accum_scalars["values"],
                                   accum_scalars["counts"])
      if "ogbg-code2" in self.dataset_config.name:
        if self.ogb_targets is None:
          ds = GraphPropPredDataset("ogbg-code2", root=self.config.data_root)
          self.ogb_targets = {}
          for split_ in ["valid", "test"]:
            self.ogb_targets[split_] = {
                id_: ds[id_][1] for id_ in ds.get_idx_split()[split_]}
        if self.ogb_evaluator is None:
          self.ogb_evaluator = Evaluator("ogbg-code2")

        # De-bias filtering out large graphs (> 1023 nodes)
        indices = np.concatenate(graph_indices)
        seq_pred = {
          id_: prediction for id_, prediction in zip(indices, all_seq_preds)
        }
        seq_pred = [seq_pred[id_] if id_ in seq_pred else []
                    for id_ in self.ogb_targets[split].keys()]

        scalars.update(
            self.ogb_evaluator.eval({
                "seq_ref": list(self.ogb_targets[split].values()),
                "seq_pred": seq_pred
            }))
      elif (("dist" in self.dataset_config.name
             and self.dataset_config.num_classes == 2)
            or self.dataset_config.name.startswith("sn")):
        scalars["precision"], scalars["recall"], scalars["f1"] = prec_rec_f1(
          scalars["tp"], scalars["fn"], scalars["fp"]
        )
        for nb in np.unique([
                int(k.split('_')[-1]) for k in scalars.keys() if '_' in k]):
          prec_rec_f1_ = prec_rec_f1(
            scalars[f"tp_{nb}"], scalars[f"fn_{nb}"], scalars[f"fp_{nb}"]
          )
          scalars.update({
            f"precision_{nb}": prec_rec_f1_[0],
            f"recall_{nb}": prec_rec_f1_[1],
            f"f1_{nb}": prec_rec_f1_[2]
          })

      elif ("dist" in self.dataset_config.name
            and self.dataset_config.num_classes < 0):
        with jax.default_device(jax.devices("cpu")[0]):
          scalars["rmse"] = jnp.sqrt(scalars["loss"])
          for nb in np.unique([
                  int(k.split('_')[-1]) for k in scalars.keys() if '_' in k]):
            scalars[f"rmse_{nb}"] = jnp.sqrt(scalars[f"loss_{nb}"])
      scalars["local_device_count"] = jax.local_device_count()
      scalars["device_count"] = jax.device_count()
      scalars["process_count"] = jax.process_count()
      scalars["process_index"] = jax.process_index()
    else:
      scalars = {}
    return predictions, scalars

  def evaluate(self, global_step, rng, **unused_kwargs):
    """See base class."""
    if self.forward is None:
      self._eval_init()

    global_step = maybe_get_first(global_step).item()
    params = maybe_get_first(self._params)
    state = maybe_get_first(self._network_state)
    if rng.ndim == 2:
      rng = rng[0]

    params = jax.tree_map(lambda x: x.copy(), params)

    m2_offset = self.config.evaluation.unk_offset
    m1_offset = self.config.evaluation.eos_offset

    self._valid_predictions, scalars = self._get_predictions(
        params, state, rng, m2_offset, m1_offset, "valid")
    scalars["num_valid_predictions"] = len(self._valid_predictions.predictions)

    if self.config.evaluation.eval_also_on_test:
      self._test_predictions, test_scalars = self._get_predictions(
          params, state, rng, m2_offset, m1_offset, "test")
      scalars["num_test_predictions"] = len(self._test_predictions.predictions)
      scalars.update({f"test_{k}": v for k, v in test_scalars.items()})

    scalars = jax.tree_map(
      lambda x: x.item() if isinstance(x, jnp.ndarray) else x, scalars)

    logging.info("evaluate() - global_step: %d, %s", global_step, scalars)

    return scalars


def _numpy_to_tensor_spec(arr: np.ndarray) -> tf.TensorSpec:
  if not isinstance(arr, np.ndarray) and not isinstance(arr, jnp.ndarray):
    return tf.TensorSpec([],
                         dtype=tf.int32 if isinstance(arr, int) else tf.float32)
  elif arr.shape:
    return tf.TensorSpec((None,) + arr.shape[1:], arr.dtype)
  else:
    return tf.TensorSpec([], arr.dtype)


def _get_step_date_label(global_step: int):
  # Date removing microseconds.
  date_str = datetime.datetime.now().isoformat().split(".")[0]
  return f"step_{global_step}_{date_str}"


def _restore_state_to_in_memory_checkpointer(restore_path):
  """Initializes experiment state from a checkpoint."""

  # Load pretrained experiment state.
  python_state_path = os.path.join(restore_path, "checkpoint.dill")
  with open(python_state_path, "rb") as f:
    pretrained_state = dill.load(f)
  logging.info("Restored checkpoint from %s", python_state_path)

  # Assign state to a dummy experiment instance for the in-memory checkpointer,
  # broadcasting to devices.
  dummy_experiment = Experiment(
      mode="train", init_rng=0, config=FLAGS.config.experiment_kwargs.config)
  for attribute, key in Experiment.CHECKPOINT_ATTRS.items():
    setattr(dummy_experiment, attribute,
            maybe_bcast_local_devices(pretrained_state[key]))

  jaxline_state = dict(
      global_step=pretrained_state["global_step"],
      experiment_module=dummy_experiment)
  snapshot = jutils.SnapshotNT(0, jaxline_state)

  # Finally, seed the jaxline `jutils.InMemoryCheckpointer` global dict.
  jutils.GLOBAL_CHECKPOINT_DICT["latest"] = jutils.CheckpointNT(
      threading.local(), [snapshot])


def _save_state_from_in_memory_checkpointer(
        save_path, experiment_class: experiment.AbstractExperiment):
  """Saves experiment state to a checkpoint."""
  logging.info("Saving model.")
  for checkpoint_name, checkpoint in jutils.GLOBAL_CHECKPOINT_DICT.items():
    if not checkpoint.history:
      logging.info('Nothing to save in "%s"', checkpoint_name)
      continue

    for entry in reversed(checkpoint.history):
      try:
        pickle_nest = entry.pickle_nest
        global_step = pickle_nest["global_step"]

        state_dict = {"global_step": global_step}
        for attribute, key in experiment_class.CHECKPOINT_ATTRS.items():
          state_dict[key] = maybe_get_first(
              getattr(pickle_nest["experiment_module"], attribute))
        save_dir = os.path.join(save_path, checkpoint_name,
                                _get_step_date_label(global_step))
        python_state_path = os.path.join(save_dir, "checkpoint.dill")
        os.makedirs(save_dir, exist_ok=True)
        with open(python_state_path, "wb") as f:
          dill.dump(state_dict, f)
        logging.info('Saved %s checkpoint to %s',
                     checkpoint_name, python_state_path)
        break
      except Exception as e:
        print(e)


def main(argv, experiment_class: experiment.AbstractExperiment):
  # Maybe restore a model.
  restore_path = FLAGS.config.restore_path
  if restore_path:
    _restore_state_to_in_memory_checkpointer(restore_path)

  # Maybe save a model.
  save_dir = os.path.join(FLAGS.config.checkpoint_dir, "models")
  os.makedirs(save_dir, exist_ok=True)
  if FLAGS.config.one_off_evaluate:
    # No need to save checkpoint in this case.
    def save_model_fn(): return None
  else:
    save_model_fn = functools.partial(_save_state_from_in_memory_checkpointer,
                                      save_dir, experiment_class)

  def wandb_init(config):
    if config.experiment_kwargs.config.debug and wandb is not None:
      return None

    tags = config.wandb.tags

    return wandb.init(
      project=config.wandb.project, tags=tags, config=config.to_dict(),
      settings=wandb.Settings(**config.wandb.settings))

  try:
    engine.main(experiment_class, argv, wandb_init=wandb_init)
  except Exception as e:
    logging.error(traceback.format_exc())
  finally:
    save_model_fn()  # Save at the end of training or in case of exception.


if __name__ == "__main__":
  _disable_gpu_for_tf()
  flags.mark_flag_as_required("config")
  # pytype: disable=wrong-arg-types
  app.run(lambda argv: main(argv, Experiment))
