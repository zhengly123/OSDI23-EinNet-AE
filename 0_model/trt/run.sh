#!/usr/bin/bash
export MODEL_DIR=/mnt/auxHome/models/einnet/
export TRT_PLUGIN_LIB=/home/osdi23ae/TensorRT-8.2.0.6/TensorRT/build/out/ 
source ../../env_lotus/env_einnet_eval.sh
source ../../env_lotus/env_trt8.sh

export ModelDir="$MODEL_DIR"
bash ./run_trt_e2e.sh
bash ./run_longformer.sh

grep -E -r "GPU Compute Time: m|\] min.*ms$" log/log_trt_"$(hostname)"* | awk '{print $1 $2 $3 $4 $5 $14 $15 " " $16 " " $17}'| column -t
