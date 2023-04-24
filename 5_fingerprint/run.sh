#!/usr/bin/bash

NNET_UseHash=0 NNET_MaxDepth=8 ./test_OpSearch &> out.no_fingerprint.txt
NNET_UseHash=1 NNET_MaxDepth=8 ./test_OpSearch &> out.fingerprint.txt

echo "Figure 18 (a)"
egrep "RUN|Intermediate" out.*fingerprint.txt

echo "Figure 18 (b)"
egrep "OK" out.*fingerprint.txt
