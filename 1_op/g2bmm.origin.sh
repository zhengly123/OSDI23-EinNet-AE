#!/bin/bash

./profile.sh ../test_op_g2bmm --gtest_filter="*.origin" |& tee g2bmm.origin.log
