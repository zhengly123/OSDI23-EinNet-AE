#!/bin/bash

./profile.sh ./conv -n 16 -c 32 -h 224 -w 224 -f 4 -g 1 -r 3 -s 3 -ph 2 -pw 1 -sh 1 -sw 1 -dh 1 -dw 1 -ca 7 |& tee conv5x5.opt.log
