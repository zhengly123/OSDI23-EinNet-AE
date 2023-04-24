#!/bin/bash

./profile.sh ./conv -n 16 -c 32 -h 224 -w 224 -f 1 -g 1 -r 5 -s 5 -ph 2 -pw 2 -sh 1 -sw 1 -dh 1 -dw 1 -ca 4 |& tee conv5x5.origin.log
