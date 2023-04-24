#!/usr/bin/bash
source ../../env_lotus/env_einnet_build.sh
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
python3 ./run_models.py |& tee out_$(hostname).txt
grep -E "Figure|^=== " out_$(hostname).txt | column -t
