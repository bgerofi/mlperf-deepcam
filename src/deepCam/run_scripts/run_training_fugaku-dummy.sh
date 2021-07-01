#!/bin/bash
#PJM -L "elapse=05:00:00"
#PJM -L "rscunit=rscunit_ft0"
#PJM --llio sharedtmp-size=80Gi
## Stdout and stderr files
#PJM -o logs/%n.o.%J
#PJM -e logs/%n.e.%J
## Jobs statistics
#PJM --spath logs/%n.s.%J
#PJM -s

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if [ -z "${PJM_MPI_PROC}" ] || [ -z "${PJM_PROC_BY_NODE}" ] || [ -z "${PJM_NODE}" ] || [ -z ${PJM_JOBID} ]; then
	echo "error: you must run this script from a Fujitsu job environment"
	exit 1
fi

#if [ "${PJM_PROC_BY_NODE}" != "4" ]; then
#	echo "error: you must request 4 processes per node"
#	exit 1
#fi

NUM_NODES=${PJM_NODE}
NUM_PROCESSES_PER_NODE=${PJM_PROC_BY_NODE}
NUM_PROCESSES=$(echo ${NUM_PROCESSES_PER_NODE} \* ${NUM_NODES}|bc)
export OMP_NUM_THREADS=46

# Do not create empty output files
export PLE_MPI_STD_EMPTYFILE="off"

PYTORCH_ROOT=${HOME}/PyTorch-1.7.0/

#parameters
run_tag="dummy"
data_dir_prefix="/vol0004/share/ra000012/DeepCAM/original/All-Hist/"
#data_dir_prefix="/share/DeepCAM"
output_dir="${HOME}/DeepCAM/runs/dummy/"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

#run training
echo "Nodes: ${NUM_NODES}, MPI ranks: ${NUM_PROCESSES}, ranks/node: ${NUM_PROCESSES_PER_NODE}, OMP threads/rank: ${OMP_NUM_THREADS} .."
mpiexec -n ${NUM_PROCESSES} \
	-x LD_PRELOAD=libtcmalloc.so \
	-x LD_LIBRARY_PATH \
	-x OMP_NUM_THREADS \
	python -u ../train_hdf5_ddp_dummy_single.py \
     --wireup_method "mpi" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 0 \
     --model_prefix "classifier" \
     --optimizer "AdamW" \
     --start_lr 1e-3 \
     --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor $(( ${NUM_PROCESSES} / 8 )) \
     --weight_decay 1e-2 \
     --validation_frequency 200 \
     --training_visualization_frequency 200 \
     --validation_visualization_frequency 40 \
     --max_validation_steps 50 \
     --logging_frequency 0 \
     --save_frequency 400 \
     --max_epochs 1 \
     --amp_opt_level O1
     #--local_batch_size 2
