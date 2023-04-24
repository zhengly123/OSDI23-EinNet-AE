#!/usr/bin/bash
source ../env_lotus/env_einnet_eval.sh
source ../env_lotus/env_einnet_build.sh

export NVIDIA_TF32_OVERRIDE=0
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin/:$PATH
./build.sh

if [ -z "${NNET_HOME}" ]; then
    echo  "NNET_HOME is not set"
    exit
fi

echo Evaluating time

$NNET_HOME/build/test_op_conv3x3 --gtest_filter="*.origin" |& tee conv3x3.origin.log
$NNET_HOME/build/test_op_conv3x3 --gtest_filter="*.optimized" |& tee conv3x3.opt.log

./conv2d_transposed 16 256 2 2 448 4 4 1 2 1 |& tee convtrans.origin.log
./conv_transpose2gemm 16 256 2 2 448 4 4 1 2 1 |& tee convtrans.opt.log

./conv -n 16 -c 32 -h 224 -w 224 -f 1 -g 1 -r 5 -s 5 -ph 2 -pw 2 -sh 1 -sw 1 -dh 1 -dw 1 -ca 4 |& tee conv5x5.origin.log
./conv -n 16 -c 32 -h 224 -w 224 -f 4 -g 1 -r 3 -s 3 -ph 2 -pw 1 -sh 1 -sw 1 -dh 1 -dw 1 -ca 7 |& tee conv5x5.opt.log

$NNET_HOME/build/test_op_g2bmm --gtest_filter="*.origin" |& tee g2bmm.origin.log
$NNET_HOME/build/test_op_g2bmm --gtest_filter="*.optimized" |& tee g2bmm.opt.log

grep -E -i "^time|^algo" ./*.log


if false; then
    echo Evaluating proformance metrics
    ./profile.sh $NNET_HOME/build/test_op_conv3x3 --gtest_filter="*.origin" |& tee conv3x3.origin.pro
    ./profile.sh $NNET_HOME/build/test_op_conv3x3 --gtest_filter="*.optimized" |& tee conv3x3.opt.pro

    ./profile.sh ./conv2d_transposed 16 256 2 2 448 4 4 1 2 1 |& tee convtrans.origin.pro
    ./profile.sh ./conv_transpose2gemm 16 256 2 2 448 4 4 1 2 1 |& tee convtrans.opt.pro

    ./profile.sh ./conv -n 16 -c 32 -h 224 -w 224 -f 1 -g 1 -r 5 -s 5 -ph 2 -pw 2 -sh 1 -sw 1 -dh 1 -dw 1 -ca 4 |& tee conv5x5.origin.pro
    ./profile.sh ./conv -n 16 -c 32 -h 224 -w 224 -f 4 -g 1 -r 3 -s 3 -ph 2 -pw 1 -sh 1 -sw 1 -dh 1 -dw 1 -ca 7 |& tee conv5x5.opt.pro

    ./profile.sh $NNET_HOME/build/test_op_g2bmm --gtest_filter="*.origin" |& tee g2bmm.origin.pro
    ./profile.sh $NNET_HOME/build/test_op_g2bmm --gtest_filter="*.optimized" |& tee g2bmm.opt.pro
    grep -E -i "^time|^algo" ./*.pro
fi

