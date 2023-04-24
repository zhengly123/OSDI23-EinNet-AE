# trtexec --workspace=4096 --separateProfileRun --noTF32 --dumpProfile --iterations=100 --duration=0 --onnx=/mnt/auxHome/models/einnet/longformer.bs1.onnx --plugins=/home/zly/Works/TensorRT-plugin/build-debug/out/libnvinfer_plugin_debug.so.8.2.0 --plugins=/home/zly/Works/TensorRT-plugin/build-debug/out/libnvonnxparser.so.8.2.0
# LD_LIBRARY_PATH=/home/hmy/data/TensorRT/build/out:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 trtexec --verbose --separateProfileRun --onnx=/mnt/auxHome/models/einnet/longformer.bs1.onnx --plugins=/home/hmy/data/TensorRT/build/out/libnvinfer_plugin.so.8.2.0 --exportTimes=trace.json --dumpProfile --exportProfile=prof.json

ModelDir=${MODEL_DIR:-/path/to/model}
echo "ONNX Model dir is" $ModelDir

TRT_PLUGIN=$NNET_HOME/0_model/trt/plugin
for i in $ModelDir/longformer.bs1.onnx $ModelDir/longformer.bs16.onnx; do
# for i in /mnt/auxHome/models/einnet/longformer.bs16.onnx; do
    LD_LIBRARY_PATH=$TRT_PLUGIN:$LD_LIBRARY_PATH trtexec --workspace=4096 --separateProfileRun --noTF32 --dumpProfile --iterations=100 --duration=0 --onnx=$i --plugins=$TRT_PLUGIN/libnvinfer_plugin.so.8.2.0 
done
