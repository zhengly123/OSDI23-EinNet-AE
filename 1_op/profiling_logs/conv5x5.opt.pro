Profiling ./conv -n 16 -c 32 -h 224 -w 224 -f 4 -g 1 -r 3 -s 3 -ph 2 -pw 1 -sh 1 -sw 1 -dh 1 -dw 1 -ca 7
==PROF== Connected to process 2265597 (/home/zly/Work/OSDI23-EinNet-AE/1_op/conv)
Group count: 1, Math type: 
Input dims: 16, 32, 224, 224
Kernel dims: 32, 4, 3, 3
Output dims: 16, 4, 226, 224
==PROF== Profiling "winogradForwardData4x4" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "winogradForwardFilter4x4" - 1: 0%....50%....100% - 1 pass
==PROF== Profiling "gemmSN_NN_kernel" - 2: 0%....50%....100% - 1 pass
==PROF== Profiling "winogradForwardOutput4x4" - 3: 0%....50%....100% - 1 pass
Time: 67.3757 ms
Algo: CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
==PROF== Disconnected from process 2265597
[2265597] conv@127.0.0.1
  void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<T1, T2>), 2023-May-07 14:57:58, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                         103.08
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         232.41
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<T1, T2>), 2023-May-07 14:57:58, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                          12.67
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Kbyte                           4.61
    ---------------------------------------------------------------------- --------------- ------------------------------

  void gemmSN_NN_kernel<float, (int)128, (int)2, (int)4, (int)8, (int)4, (int)4, (bool)0, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatched<float>>(cublasGemmSmallNParams<T9, T10, T1>), 2023-May-07 14:57:58, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                         235.37
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         237.49
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<T1, T2>), 2023-May-07 14:58:00, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                          29.43
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                          29.42
    ---------------------------------------------------------------------- --------------- ------------------------------

