#!/usr/bin/bash
source ../env_lotus/env_tvm-v10.sh
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin/:$PATH

echo Figure 15 Ansor 

python3 ./tune_op_ansor.py > out.ansor.txt
grep "Time" out.ansor.txt | column -t

python3 ./longformer-eval.py gpu result_op_ansor_a100/ansor.longformer.A100.json > out.ansor.longformer.txt
grep "Time" out.ansor.longformer.txt | column -t

echo Figure 15 AutoTVM

python3 ./tune_op_autotvm.py > out.autotvm.txt
grep "Time" out.autotvm.txt | column -t
