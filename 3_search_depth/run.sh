#!/usr/bin/bash
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
source ../env_lotus/env_einnet_build.sh 

python3 ./evaluate_max_depth.py > out.txt

grep -E "=== Model|Depth =|Figure" out.txt
