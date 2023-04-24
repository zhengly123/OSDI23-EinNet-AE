#!/usr/bin/env bash

# spack
. /home/spack/spack/share/spack/setup-env.sh
spack load googletest@1.10.0 cuda@11.0.2 cudnn@8.0.3.33-11.0
spack load python@3.7/2fcge

# python
source /home/osdi23ae/py37venv/bin/activate

# Use g++ 9 for nvcc --cubin
export CUDAHOSTCXX=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin/gcc
# spack load gcc@9.4.0 # This results in old libc++ detected and hinders other executables in OS

export NVIDIA_TF32_OVERRIDE=0

# export PYTHONPATH=${PET_HOME}/python:$PYTHONPATH

# # TVM errors on gcc: TVM has to use gcc 9 to compile CUDA code. Setting PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.2.0/gcc-9.4.0-33iybhs3rsr44bl57dyo5wp5qp5herwp/bin:$PATH before run onnx_pet and onnx_nnet can specify gcc-9 for TVM. However, this should not be set for the compilation of project PET itself
# export PYTHONPATH=/home/zly/Apps/tvm-211104/python:$PYTHONPATH
