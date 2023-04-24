This folder produces the resutls in Talbe 3 on the A100 GPU.

We provide `run.sh` to time and profile each case before and after optimization.

The correspondence between `ncu` output and Table 3 is
- DRAM: dram__bytes_read.sum
- L2: lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum

Since there can be multiple kernels for a case, the data in profiling output should be summed to get the overall result.

As the `ncu` may result in error status of GPUs in the provided server, we provide the profiling logs in `profiling_logs` for a convient verificatoin.

Note: we found a probabilistic performance regression on our A100 GPU. After using the `ncu` profiler, A100 may suffer from a persistent ~50% performance loss. One solution to fix this trouble is reload NVIDIA kernel module, which requires root privilege. Though we often check the status of our server, it can still be in the wrong status. If you meet such a problem, Please inform us of it.

Note: if the output of `ncu` is nan, which means the GPU is in a wrong status, a kernel module reload or system reboot are requried. Please inform us of it.
```
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                              byte                        (!) nan
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                        (!) nan
    ---------------------------------------------------------------------- --------------- ------------------------------
```
