`./run.sh` to get the execution time of models optimized by EinNet.

We provide TVM tuning logs on the provided server. Before running the script, you can copy them to this directory. Note that logs for A100 and V100 cannot exist at the same time, they should be switched when running on another device.
```bash
# For A100
cp -r /home/osdi23ae/tuned_kernels/cache_A100 ./.cache 
# For V100
cp -r /home/osdi23ae/tuned_kernels/cache_V100 ./.cache 
```
