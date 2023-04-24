#!/bin/bash

ModelDir=${MODEL_DIR:-/path/to/model}
echo "ONNX Model dir is" $ModelDir

# GCC for nvcc
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
# Default FP32 for cuDNN
export NVIDIA_TF32_OVERRIDE=0
for model in infogan fsrcnn gcn csrnet resnet18 dcgan; do
for bs in 1 16; do
for backend in cublas ansor; do
    python3 run_onnx_tvm.py $ModelDir/$model.bs$bs.onnx $backend sm_80
done
done
done
