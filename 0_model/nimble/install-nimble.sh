. /home/hsh/env_lotus.sh
spack load gcc@9.4.0
spack unload python
. ~hsh/miniconda3/bin/activate
conda create -n nimble3 python=3.7 -y
conda activate nimble3

export CUDA_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.2.0/cuda-11.0.2-npdlw4kj3xsbaam3gedlxm3umfumpujb
export CUDA_NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
export CUDNN_HOME=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-11.2.0/cudnn-8.0.3.33-11.0-2nsxaxc6dziw7mlxidagjvfv22xu4uqf
export CUDNN_LIB_DIR=$CUDNN_HOME/lib64
export CUDNN_INCLUDE_DIR=$CUDNN_HOME/include/
export CUDNN_LIBRARY=$CUDNN_HOME/lib64
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses -y
conda install -c pytorch magma-cuda110 -y

conda install -c conda-forge onnx -y

conda install -c conda-forge protobuf=3.9 -y

pip install future

export NIMBLE_HOME=/home/huangkz/repos/test-nimble/nimble
cd $NIMBLE_HOME
rm -rf ./build
TORCH_CUDA_ARCH_LIST="7.0;8.0" DEBUG=ON BUILD_TEST=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_NUMA=0 USE_MPI=0 python setup.py install

# pycuda
# pip install 'pycuda>=2019.1.1' --no-cache-dir

# addtional environment variables, we need this setting for every experiment using TensorRT
export TENSORRT_HOME=/mnt/auxHome/hmy/TensorRT-8.2.0.6
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH

# install TensorRT for python

# cd $TENSORRT_HOME/python
# pip install tensorrt-8.2.0.6-cp37-none-linux_x86_64.whl
# cd $TENSORRT_HOME/graphsurgeon
# pip install graphsurgeon-0.4.5-py2.py3-none-any.whl

pip install pandas

cd $NIMBLE_HOME/experiment/torchvision
python setup.py install
cd $NIMBLE_HOME/experiment/pretrained-models
python setup.py install
cd $NIMBLE_HOME/experiment/timm
python setup.py install
