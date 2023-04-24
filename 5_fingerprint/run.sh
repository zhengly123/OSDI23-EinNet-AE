#!/usr/bin/bash
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
source ../env_lotus/env_einnet_build.sh 

NNET_UseHash=0 NNET_MaxDepth=7 ../build/test_OpSearch &> out.no_fingerprint.txt
NNET_UseHash=1 NNET_MaxDepth=7 ../build/test_OpSearch &> out.fingerprint.txt

echo "Figure 18 (a)"
grep -E "RUN|Intermediate" out.*fingerprint.txt

echo "Figure 18 (b)"
grep -E "OK" out.*fingerprint.txt
