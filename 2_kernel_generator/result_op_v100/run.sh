#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# GCC for nvcc
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
# Default FP32 for cuDNN
export NVIDIA_TF32_OVERRIDE=0
python3 ../tune_op_ansor.py
