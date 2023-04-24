#!/usr/bin/bash

for i in $(seq 1 11); do
    echo $i
    NNET_UseHash=1 NNET_MaxDepth=$i ./test_OpSearch &> out.searchDepthTest.$i.txt 
done

echo ">>> Figure 17 (a)"
egrep "RUN|#Intermediate states" out.searchDepthTest*txt

echo ""

for i in Conv3x3 ConvTranspose Conv5x5 G2BMM; do
NNET_PrintAndExit=1 NNET_UseHash=1 NNET_MaxDepth=7 ./test_OpSearch  --gtest_filter="*$i" > out.steps.$i.txt
done

echo ">>> Figure 17 (b)"
grep "Steps" out.steps.*.txt | column -t
