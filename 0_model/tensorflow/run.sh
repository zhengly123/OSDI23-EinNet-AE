#!/usr/bin/bash
export ModelDir=/mnt/auxHome/models/einnet/ 
source /home/zly/env/tf2/bin/activate
source ../../env_lotus/env_einnet_eval.sh
python3 run.py
grep "Inference Time" log/tf_$(hostname)* | column -t
