#!/bin/bash

./profile.sh ../test_op_conv3x3 --gtest_filter="*.optimized" |& tee conv3x3.opt.log
