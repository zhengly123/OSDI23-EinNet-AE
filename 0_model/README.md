## Baseline Evaluation for Figure 12

This folder contains the evaluation for baseline systems. To evalute these frameworks, 
1. build EinNet only on our privided server or build all baselines according to the following `Build from source` instructions. The build directory should be named `build` and placed in the top directory of EinNet, since evaluation scripts depend on the building results.
    - Build EinNet on our server
      ```
        git clone --recursive git@github.com:zhengly123/OSDI23-EinNet-AE.git
        cd OSDI23-EinNet-AE
        source env_lotus/env_einnet_build.sh
        make build && cd build
        cmake .. -DUSE_CUDA=ON -DTVM_INCLUDE_DIR="${TVM_HOME}/include" -DTVM_LIB_DIR="${TVM_HOME}/build" -DDMLC_INCLUDE_DIR="${TVM_HOME}/3rdparty/dmlc-core/include" -DDLPACK_INCLUDE_DIR="${TVM_HOME}/3rdparty/dlpack/include" -DBUILD_TEST=ON -DBUILD_TEST_CORE=ON -DBUILD_TEST_EINNET=ON
        make -j32
      ```
2. We provides `run.sh` scripts for each framework in the subdirectories. This script can be run out-of-the-box on our provided server. Evaluation on other platforms requires modifiying environment variables, which is set at the beginning of each script.


### Build from source
#### A.4.1 Install EinNet from source
Clone code from git
Install requirements
TVM: see A.4.5
Install EinNet
```
mkdir build; cd build
export TVM_HOME=/path/to/tvm
cmake .. -DUSE_CUDA=ON -DTVM_INCLUDE_DIR="${TVM_HOME}/include" -DTVM_LIB_DIR="${TVM_HOME}/build" -DDMLC_INCLUDE_DIR="${TVM_HOME}/3rdparty/dmlc-core/include" -DDLPACK_INCLUDE_DIR="${TVM_HOME}/3rdparty/dlpack/include" -DBUILD_TEST=ON -DBUILD_TEST_CORE=ON -DBUILD_TEST_EINNET=ON
make -j32
```

#### A.4.2 Install TensorFlow
Install pre-compiled libraries
```
pip install tensorflow-gpu==2.4
```

#### A.4.3 Install TensorRT
Download and extract the latest TensorRT 8.0 GA package for Ubuntu 18.04 and CUDA 10.2
```
tar -xvzf TensorRT-8.0.0.3.Linux.x86_64-gnu.cuda-11.0.cudnn8.2.tar.gz
export TRT_RELEASE=`pwd`/TensorRT-8.0.0.3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_RELEASE/lib
```

#### A.4.4 Install PET
see https://github.com/thu-pacman/PET/blob/master/README.pdf for detailed installation instruction
```
git clone --recursive https://github.com/thu-pacman/PET.git
cd PET
mkdir build; cd build; cmake ..
make -j 4
```

#### A.4.5 Install TVM & Ansor
See https://tvm.apache.org/docs/install/from_source.html for detailed installation instruction
```
git clone --recursive https://github.com/apache/tvm tvm
git checkout v0.10.0
mkdir build
cp cmake/config.cmake build
cd build
cmake ..
make -j4
```

#### A.4.6 Install Nimble
```
git clone –recursive https://github.com/snuspl/nimble.git nimble
cd nimble
# to fix correctness issue in nimble, we should apply a patch
git apply $NNET_HOME/0_model/nimble/nimble.patch
# On your cluster, you need to follow this guide to install nimble 
# On our cluster, see below
source $NNET_HOME/0_model/nimble/env_nimble.sh
bash $NNET_HOME/0_model/nimble/install-nimble.sh
```
