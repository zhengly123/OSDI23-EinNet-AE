#!/usr/bin/bash
spack load cuda@11.0.2 cudnn@8.0.3.33-11.0
export CUDAHOSTCXX=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin/gcc
. /home/hsh/miniconda3/bin/activate # change it to your miniconda directory
conda activate nimble3

export CUDA_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.2.0/cuda-11.0.2-npdlw4kj3xsbaam3gedlxm3umfumpujb
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDNN_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.2.0/cudnn-8.0.3.33-11.0-2nsxaxc6dziw7mlxidagjvfv22xu4uqf
export CUDNN_LIB_DIR=$CUDNN_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDNN_HOME/include/
export CUDNN_LIBRARY=$CUDNN_HOME/lib64

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

export NIMBLE_HOME=/home/hsh/test-nimble/nimble
export TENSORRT_HOME=/home/osdi23ae/TensorRT-8.2.0.6
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
