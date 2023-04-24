export NVIDIA_TF32_OVERRIDE=0
ModelDir=${MODEL_DIR:-/path/to/model}
echo "ONNX Model dir is" $ModelDir

for model in infogan dcgan fsrcnn gcn resnet18 csrnet; do
	for bs in 1 16; do
	 	fn=${ModelDir}/${model}.bs${bs}.onnx
		echo $fn
		log_dir=log/log_trt_$(hostname)_${model}_${bs}
		mkdir -p $log_dir
		# Set a 4GB workspace to enable more optimizations
		trtexec --workspace=4096 --separateProfileRun --noTF32 --dumpProfile --iterations=100 --duration=0 --onnx=$fn > ${log_dir}/log.txt
	done
done
