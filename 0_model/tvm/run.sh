#!/bin/bash

export MODEL_DIR=/mnt/auxHome/models/einnet/
source ../../env_lotus/env_tvm-v10.sh

export NVIDIA_TF32_OVERRIDE=0
ModelDir=${MODEL_DIR:-/path/to/model}
echo "ONNX Model dir is" $ModelDir

for model in infogan dcgan fsrcnn gcn resnet18 csrnet longformer-part1 longformer-part2; do
    for bs in 1 16; do
	 	fn=${ModelDir}/${model}.bs${bs}.onnx
		echo $fn
		log_dir=log/log_tvm_$(hostname)_${model}_${bs}
		mkdir -p $log_dir
        python3 run_onnx_tvm.py $ModelDir/$model.bs$bs.onnx cublas sm_80 > ${log_dir}/log.txt
	done
done

tail -n3 log/log_tvm_$(hostname)_*/log.txt
