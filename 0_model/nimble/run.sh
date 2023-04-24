#!/usr/bin/bash
source ./env_nimble.sh
PYTHONPATH=/home/hsh/miniconda3/envs/nimble3/lib/python3.7/site-packages:/home/hsh/.local/lib/python3.7/site-packages ./evaluation.sh > out.$(hostname).txt

grep -iv "skip" out.$(hostname).txt | grep -B1 -A2 "============"
