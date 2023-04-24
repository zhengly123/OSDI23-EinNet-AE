#!/bin/bash
export NVIDIA_TF32_OVERRIDE=0
for model in longformerpart*.inferred.onnx; do
for backend in cublas; do
    python3 ../run_onnx_tvm.py $model $backend sm_80
done
done
