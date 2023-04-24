export NVIDIA_TF32_OVERRIDE=0
cwd=`pwd`
ModelDir=/home/zly/models
for Device in 0 1; do
	if [[ $Device -eq 0 ]]; then
		DeviceName="A100"
	else
		DeviceName="V100"
	fi
	export CUDA_VISIBLE_DEVICES=$Device
	# for i in ${ModelDir}/infogan.bs*.onnx ${ModelDir}/gcn.bs*.onnx ${ModelDir}/csrnet.bs*.onnx ${ModelDir}/resnet18.bs*.onnx ${ModelDir}/dcgan.bs*.onnx ${ModelDir}/unet.bs*.onnx; do
	for i in  ${ModelDir}/gcn.bs*.onnx; do
		log_dir=220417_default_workspace_log_trt_${DeviceName}_cuDNN8.0_`hostname`_`basename $i` 
		mkdir $log_dir
		echo $log_dir
		# trtexec --separateProfileRun --noTF32 --dumpProfile --iterations=100 --duration=0 --onnx=$i > log.txt
		trtexec --workspace=4096 --separateProfileRun --noTF32 --dumpProfile --iterations=100 --duration=0 --onnx=$i > log.txt                          
		env &> env.txt
		mv env.txt log.txt trace.json prof.json $log_dir           
	done
done
egrep "GPU Compute Time: m|\] min.*ms$" 220417_default_workspace_log_*/log.txt
