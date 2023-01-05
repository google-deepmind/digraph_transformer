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
      echo "Usage: run_dist_dataset_gen.sh -r <Task root directory>"
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Get this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATA_ROOT=${TASK_ROOT}


python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --acyclic --connected \
    --n_train=16,18 --n_valid=18,20 --n_test=20,28

python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --connected \
    --n_train=16,18 --n_valid=18,20 --n_test=20,28

python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --acyclic --connected --target=undirected \
    --n_train=16,18 --n_valid=18,20 --n_test=20,28

python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --connected --target=undirected \
    --n_train=16,18 --n_valid=18,20 --n_test=20,28


python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --acyclic --connected \
    --n_train=16,64 --n_valid=64,72 --n_test=72,84

python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --connected \
    --n_train=16,64 --n_valid=64,72 --n_test=72,84

python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --acyclic --connected --target=undirected \
    --n_train=16,64 --n_valid=64,72 --n_test=72,84

python "${SCRIPT_DIR}"/script_generate_distance_np.py \
    --out_path=${DATA_ROOT}/distance --connected --target=undirected \
    --n_train=16,64 --n_valid=64,72 --n_test=72,84
