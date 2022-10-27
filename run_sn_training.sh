#!/bin/bash
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
set -x

while getopts ":r:" opt; do
  case ${opt} in
    r )
      TASK_ROOT=$OPTARG
      ;;
    \? )
      echo "Usage: run_training.sh -r <Task root directory>"
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Get this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "
These scripts are provided for illustrative purposes. It is not practical for
actual training since it only uses a single machine, and likely requires
reducing the batch size and/or model size to fit on a single GPU.
"

# read -p "Press enter to continue"

MODEL_SEED=42
CHECKPOINT_DIR=${TASK_ROOT}/sorting_network/checkpoints/
DATA_ROOT=${TASK_ROOT}/sorting_network/

python "${SCRIPT_DIR}"/experiment.py \
    --jaxline_mode="train" \
    --config="${SCRIPT_DIR}/config.py:sn,bignn,maglappos,bbs" \
    --config.random_seed=${MODEL_SEED} \
    --config.checkpoint_dir=${CHECKPOINT_DIR} \
    --config.experiment_kwargs.config.data_root=${DATA_ROOT}
