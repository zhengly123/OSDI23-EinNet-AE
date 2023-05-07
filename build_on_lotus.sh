#!/usr/bin/bash
source ./env_lotus/env_einnet_build.sh
mkdir build && cd build || exit
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DTVM_INCLUDE_DIR="${TVM_HOME}/include" -DTVM_LIB_DIR="${TVM_HOME}/build" -DDMLC_INCLUDE_DIR="${TVM_HOME}/3rdparty/dmlc-core/include" -DDLPACK_INCLUDE_DIR="${TVM_HOME}/3rdparty/dlpack/include" -DBUILD_TEST=ON -DBUILD_TEST_CORE=ON -DBUILD_TEST_EINNET=ON
make -j32
