#!/bin/sh
set -x

mode=$5
echo ${mode}

if [ ${mode} == "flow" ]
then
    srun -p $1 -n1 --gres=mlu:1 python tools/deployment/pytorch2onnx.py $2 $3 --output-file $4 --shape 1 3 10 224 224 --do_simplify
else
    srun -p $1 -n1 --gres=mlu:1 python tools/deployment/pytorch2onnx.py $2 $3 --output-file $4 --shape 1 75 3 256 256 --do_simplify
fi

# usage:

# $1: camb_mlu290
# $2: configs/recognition/tsn/tsn_r50_1x1x3_75e_ucf101_rgb.py or configs/recognition/tsn/tsn_r50_1x1x3_30e_ucf101_flow.py
# $3: camb_pth/tsn/tsn_peoch_80.pth or camb_pth/tsn_flow/epoch_30.pth
# $4: camb_pth/tsn/tsn_epoch_80.onnx or camb_pth/tsn_flow/epoch_30.onnx
# $5: rgb or flow to choose the model be transed