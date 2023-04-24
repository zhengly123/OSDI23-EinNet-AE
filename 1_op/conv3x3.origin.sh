#!/bin/bash

./profile.sh ./conv -n 1 -c 512 -h 7 -w 7 -f 512 -g 1 -r 3 -s 3 -ph 1 -pw 1 -sh 1 -sw 1 -dh 1 -dw 1 -ca 7 |& tee conv3x3.origin.log
