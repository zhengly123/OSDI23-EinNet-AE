#!/bin/bash
TVM_PYTHONPATH=/home/zly/Apps/tvm-v0.10.0/python
# GCC for nvcc
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
# Default FP32 for cuDNN
export NVIDIA_TF32_OVERRIDE=0
# export PYTHONPATH=$TVM_PYTHONPATH:$PYTHONPATH
python3 ../tune_op_ansor.py
