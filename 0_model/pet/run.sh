#!/usr/bin/bash

export ModelDir=/mnt/auxHome/models/einnet/ 
export PET_HOME=/home/osdi23ae/PET
export TVM_HOME=/home/osdi23ae/tvm-v0.10.0

source ../../env_lotus/env_pet.sh
export PATH=${PET_HOME}/build:$PATH 
export PYTHONPATH=${PET_HOME}/build:${PET_HOME}/python:$TVM_HOME/python/
export LD_LIBRARY_PATH=/home/osdi23ae/PET/build:$LD_LIBRARY_PATH

export NVIDIA_TF32_OVERRIDE=0
echo "ONNX Model dir is" $ModelDir

echo gcc $(which gcc)
echo LD_LIBRARY_PATH $LD_LIBRARY_PATH
spack find --loaded
ldd $(which onnx_pet)
echo $(which python3)

for model in infogan dcgan fsrcnn gcn resnet18 csrnet; do
	for bs in 1 16; do
	 	fn=${ModelDir}/${model}.bs${bs}.onnx
		echo $fn
		log_dir=log/log_pet_$(hostname)_${model}_${bs}
		mkdir -p $log_dir
        onnx_pet ${fn} > ${log_dir}/log.txt
	done
done

for bs in 1 16; do
    log_dir=log/log_pet_$(hostname)_longformer_${bs}
	mkdir -p "$log_dir"
    echo "longformer_pet --gtest_filter=Longformer_PET.e2e_bs$bs > $log_dir/log.txt"
    longformer_pet --gtest_filter=Longformer_PET.e2e_bs$bs > "$log_dir/log.txt"
done

grep "Best Perf with correction" log/log_pet_$(hostname)*/log.txt | column -t
