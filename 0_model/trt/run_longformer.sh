#!/usr/bin/bash
echo "ONNX Model dir is" $ModelDir

for model in longformer; do
	for bs in 1 16; do
	 	fn=${ModelDir}/${model}.bs${bs}.onnx
		echo $fn
		log_dir=log/log_trt_$(hostname)_${model}_${bs}
		mkdir -p $log_dir
        LD_LIBRARY_PATH=$TRT_PLUGIN_LIB:$LD_LIBRARY_PATH trtexec --workspace=4096 --separateProfileRun --noTF32 --dumpProfile --iterations=100 --duration=0 --onnx=${fn} --plugins=${TRT_PLUGIN_LIB}/libnvinfer_plugin.so.8.2.0 > ${log_dir}/log.txt
	done
done
