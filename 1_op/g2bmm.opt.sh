#!/bin/bash

./profile.sh ../test_op_g2bmm --gtest_filter="*.optimized" |& tee g2bmm.opt.log
