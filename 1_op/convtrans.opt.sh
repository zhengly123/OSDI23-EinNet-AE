#!/bin/bash

./profile.sh ./conv_transpose2gemm 16 256 2 2 448 4 4 1 2 1 |& tee convtrans.opt.log
