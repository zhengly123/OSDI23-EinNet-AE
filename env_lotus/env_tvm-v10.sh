#!/usr/bin/bash
spack load python@3.7.13/2fcgebh
spack load cudnn@8.0.3.33-11.0/2nsxaxc cuda@11.0.2/npdlw4k 
export PATH=/home/spack/spack/opt/spack/linux-ubuntu22.04-broadwell/gcc-9.4.0/gcc-9.4.0-st36klijpsnquihiy463hmedsyhoc3g6/bin:$PATH
export PYTHONPATH=/home/osdi23ae/tvm-v0.10.0/python:${PYTHONPATH}
