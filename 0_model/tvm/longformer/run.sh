#!/bin/bash
# GCC for nvcc
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
# Default FP32 for cuDNN
export NVIDIA_TF32_OVERRIDE=0
for model in /home/zly/Works/ml-baseline/tvm/result_model/longformer/longformerpart*.inferred.onnx; do
# for bs in 1 16; do
# for backend in cublas ansor; do
for backend in cublas; do
    python3 ../run_onnx_tvm.py $model $backend sm_80
done
# done
done
