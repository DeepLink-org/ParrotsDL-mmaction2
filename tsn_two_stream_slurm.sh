#!/bin/bash

mkdir -p log/mmaction

T=`date +%m%d%H%M%S`
rgb_config="tsn_r50_1x1x3_75e_ucf101_rgb"
flow_config="tsn_r50_1x1x3_75e_ucf101_flow"
ROOT=.
EXTRA_ARGS=${@:3}

pyroot=$ROOT/models/mmaction
export PYTHONPATH=$pyroot:$PYTHONPATH
g=$(($2<8?$2:8))
SRUN_ARGS=${SRUN_ARGS:-""}

# avoid mm log repeat
export PARROTS_DEFAULT_LOGGER=FALSE
set -x

# train rgb
PYTHON_ARGS="python -u tools/train.py configs/recognition/tsn/${rgb_config}.py --launcher=slurm --validate"

srun -p $1 -n$2 --gres mlu:$g --ntasks-per-node $g --job-name=mmaction_${rgb_config} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmaction/train.${rgb_config}.log.$T

# train flow
PYTHON_ARGS="python -u tools/train.py configs/recognition/tsn/${flow_config}.py --launcher=slurm --validate"

srun -p $1 -n$2 --gres mlu:$g --ntasks-per-node $g --job-name=mmaction_${flow_config} ${SRUN_ARGS} \
    $PYTHON_ARGS $EXTRA_ARGS \
    2>&1 | tee $ROOT/log/mmaction/train.${flow_config}.log.$T


# test rgb
PYTHON_ARGS="python -u tools/test.py configs/recognition/tsn/${rgb_config}.py work_dirs/tsn_r50_1x1x3_75e_ucf101_split_1_rgb/latest.pth --launcher=slurm --out work_dirs/tsn_r50_1x1x3_75e_ucf101_split_1_rgb/rgb.pkl --eval top_k_accuracy"

srun -p $1 -n$2 --gres mlu:$g --ntasks-per-node $g --job-name=mmaction_${rgb_config} ${SRUN_ARGS} \
    $PYTHON_ARGS \
    2>&1 | tee $ROOT/log/mmaction/test.${rgb_config}.log.$T

# test flow
PYTHON_ARGS="python -u tools/test.py configs/recognition/tsn/${flow_config}.py work_dirs/tsn_r50_1x1x3_75e_ucf101_split_1_flow/latest.pth --launcher=slurm --out work_dirs/tsn_r50_1x1x3_75e_ucf101_split_1_flow/flow.pkl --eval top_k_accuracy"

srun -p $1 -n$2 --gres mlu:$g --ntasks-per-node $g --job-name=mmaction_${flow_config} ${SRUN_ARGS} \
    $PYTHON_ARGS \
    2>&1 | tee $ROOT/log/mmaction/test.${flow_config}.log.$T

# merge result
PYTHON_ARGS="python -u tools/analysis/report_accuracy.py --scores work_dirs/tsn_r50_1x1x3_75e_ucf101_split_1_rgb/rgb.pkl work_dirs/tsn_r50_1x1x3_75e_ucf101_split_1_flow/flow.pkl --datalist /mnt/lustre/share/openmmlab/datasets/action/ucf101/ucf101_val_split_1_rawframes.txt"

srun -p $1 -n 1 --gres mlu:1 --ntasks-per-node 1  --job-name=mmaction_merge ${SRUN_ARGS} \
    $PYTHON_ARGS \
    2>&1 | tee $ROOT/log/mmaction/merge_result.log.$T
