from collections import Sequence
import logging
import os
import socket
import subprocess
from subprocess import PIPE
from typing import Any, Iterator, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf


def serialize(value: Any) -> str:
  if value is None:
    return ''
  if isinstance(value, str):
    return value
  if isinstance(value, Sequence):
    return f'"{repr(tuple(value))}"'
  return repr(value)


def flatten(map: dict, prefix: str, separator='.') -> Iterator[Tuple[str, Any]]:
  for k, v in map.items():
    if isinstance(v, dict) or isinstance(v, DictConfig):
      yield from flatten(map[k], f'{prefix}{separator}{k}', separator)
    else:
      yield f'{prefix}{separator}{k}', serialize(v)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
  hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
  working_dir = hydra_cfg['runtime']['output_dir']
  logging.info(f'Host name: {socket.gethostname()}')
  logging.info(f'Working directory: {working_dir}')
  logging.info(OmegaConf.to_yaml(cfg))
  mode = cfg['config']['mode']
  seed = cfg['config']['random_seed']
  task = cfg['config']['task']
  presets = cfg['config']['presets']
  data = cfg['config']['data']
  checkpoint = cfg['config']['checkpoint']
  arguments = []
  if cfg['config']['wandb']:
    arguments += [
        f'{k}={v}' for k, v
        in flatten(cfg['config']['wandb'], '--config.wandb')]
  if cfg['overwrite']:
    arguments += [
        f'{k}={v}' for k, v in flatten(cfg['overwrite'], '--config')]

  command = ['python', 'experiment.py',
             f'--jaxline_mode={mode}',
             f'--config=./config.py:{task},{presets}',
             f'--config.experiment_kwargs.config.data_root={data}',
             f'--config.random_seed={seed}',
             f'--config.checkpoint_dir={working_dir}/checkpoint',
             *arguments]
  logging.info(' '.join(command))
  logging.info('VS Code debugger: "' + '",\n "'.join(
      [c.replace('"', '') for c in command[2:]]) + '"')
  try:
    process = subprocess.Popen(
      ' '.join(command),
      shell=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT  # Redirect standard error to standard output
    )

    while True:
      output = process.stdout.readline()
      if output == b'' and process.poll() is not None:
        break
      if output:
        # Redirect the output to logging
        logging.info(output.decode().strip())
  except Exception as e:
    logging.exception("Error during training")
    raise e


if __name__ == "__main__":
  my_app()
