#!/bin/bash
set -x
set -o pipefail

# 0. check the most important SMART_ROOT
echo  "!!!!!SMART_ROOT is" $SMART_ROOT
if $SMART_ROOT; then
    echo "SMART_ROOT is None,Please set SMART_ROOT"
    exit 0
fi

# 1. set env_path and build soft links for mm configs
if [[ $PWD =~ "mmaction" ]]
then 
    pyroot=$PWD
else
    pyroot=$PWD/mmaction
fi
echo $pyroot
if [ -d "$pyroot/algolib/configs" ]
then
    rm -rf $pyroot/algolib/configs
    ln -s $pyroot/configs $pyroot/algolib/
else
    ln -s $pyroot/configs $pyroot/algolib/
fi

# 2. build file folder for save log and set time
mkdir -p algolib_gen/mmaction/$3
now=$(date +"%Y%m%d_%H%M%S")
 
# 3. set env variables
export PYTORCH_VERSION=1.4
export PYTHONPATH=$pyroot:$PYTHONPATH
export MODEL_NAME=$3
export FRAME_NAME=mmaction    #customize for each frame
export PARROTS_DEFAULT_LOGGER=FALSE

# 4. init_path
export PYTHONPATH=${SMART_ROOT}:$PYTHONPATH
export PYTHONPATH=${SMART_ROOT}/common/sites:$PYTHONPATH
 
# 5. build necessary parameter
partition=$1  
name=$3
MODEL_NAME=$3
g=$(($2<8?$2:8))
array=( $@ )
EXTRA_ARGS=${array[@]:3}
EXTRA_ARGS=${EXTRA_ARGS//--resume/--resume-from}
SRUN_ARGS=${SRUN_ARGS:-""}

# 6. model list
case $MODEL_NAME in
    "stgcn_80e_ntu60_xsub_keypoint_3d")
        FULL_MODEL="skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint_3d"
        ;;
    *)
       echo "invalid $MODEL_NAME"
       exit 1
       ;; 
esac

# 7. set port and choice model
port=`expr $RANDOM % 10000 + 20000`
file_model=${FULL_MODEL##*/}
folder_model=${FULL_MODEL%/*}

# 8. run model
srun -p $1 -n$2\
        --gres gpu:$g \
        --ntasks-per-node $g \
        --job-name=${FRAME_NAME}_${MODEL_NAME} ${SRUN_ARGS}\
    python -u $pyroot/tools/train.py $pyroot/algolib/configs/$folder_model/$file_model.py --launcher=slurm  \
    --work-dir=algolib_gen/${FRAME_NAME}/${MODEL_NAME} $EXTRA_ARGS \
    2>&1 | tee algolib_gen/${FRAME_NAME}/${MODEL_NAME}/train.${MODEL_NAME}.log.$now
