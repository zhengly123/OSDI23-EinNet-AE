#!/usr/bin/bash
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
source ../env_lotus/env_einnet_build.sh 

for i in $(seq 1 11); do
    echo $i
    NNET_UseHash=1 NNET_MaxDepth=$i ../build/test_OpSearch &> out.searchDepthTest.$i.txt 
done

echo ">>> Figure 17 (a)"
grep -E "RUN|\[  .*ms" out.searchDepthTest*txt

echo ""

for i in Conv3x3 ConvTranspose Conv5x5 G2BMM; do
NNET_PrintAndExit=1 NNET_UseHash=1 NNET_MaxDepth=7 ../build/test_OpSearch  --gtest_filter="*$i" > out.steps.$i.txt
done

echo ">>> Figure 17 (b)"
grep "Steps" out.steps.*.txt | column -t
