Profiling ./conv -n 16 -c 32 -h 224 -w 224 -f 1 -g 1 -r 5 -s 5 -ph 2 -pw 2 -sh 1 -sw 1 -dh 1 -dw 1 -ca 4
==PROF== Connected to process 2265431 (/home/zly/Work/OSDI23-EinNet-AE/1_op/conv)
Group count: 1, Math type: 
Input dims: 16, 32, 224, 224
Kernel dims: 32, 1, 5, 5
Output dims: 16, 1, 224, 224
==PROF== Profiling "regular_fft_pad" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "vector_fft" - 1: 0%....50%....100% - 1 pass
==PROF== Profiling "regular_fft_pad" - 2: 0%....50%....100% - 1 pass
==PROF== Profiling "vector_fft" - 3: 0%....50%....100% - 1 pass
==PROF== Profiling "transpose_readWrite_alignment..." - 4: 0%....50%....100% - 1 pass
==PROF== Profiling "transpose_readWrite_alignment..." - 5: 0%....50%....100% - 1 pass
==PROF== Profiling "gemv2T_kernel_val" - 6: 0%....50%....100% - 1 pass
==PROF== Profiling "transpose_readWrite_alignment..." - 7: 0%....50%....100% - 1 pass
==PROF== Profiling "vector_fft" - 8: 0%....50%....100% - 1 pass
==PROF== Profiling "regular_fft_clip" - 9: 0%....50%....100% - 1 pass
Time: 129.487 ms
Algo: CUDNN_CONVOLUTION_FWD_ALGO_FFT
==PROF== Disconnected from process 2265431
[2265431] conv@127.0.0.1
  void DSE::regular_fft_pad<(int)0, (int)1, (int)256, (int)16, (int)16, (int)1, float, float, float2>(T9 *, T7 *, int, int3, int3, int, int3, int3, int, int, int, int, int, bool), 2023-May-07 14:57:45, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                          32.13
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Kbyte                          17.82
    ---------------------------------------------------------------------- --------------- ------------------------------

  void DSE::vector_fft<(int)0, (int)1, (int)256, (int)16, (int)16, (int)1, float, float, float2>(T9 *, T9 *, int, int3, int3), 2023-May-07 14:57:45, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           8.47
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                           8.47
    ---------------------------------------------------------------------- --------------- ------------------------------

  void DSE::regular_fft_pad<(int)0, (int)1, (int)256, (int)16, (int)16, (int)1, float, float, float2>(T9 *, T7 *, int, int3, int3, int, int3, int3, int, int, int, int, int, bool), 2023-May-07 14:57:46, Context 1, Stream 19
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                         102.79
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         127.38
    ---------------------------------------------------------------------- --------------- ------------------------------

  void DSE::vector_fft<(int)0, (int)1, (int)256, (int)16, (int)16, (int)1, float, float, float2>(T9 *, T9 *, int, int3, int3), 2023-May-07 14:57:46, Context 1, Stream 19
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                         135.28
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         134.92
    ---------------------------------------------------------------------- --------------- ------------------------------

  void transpose_readWrite_alignment_kernel<float2, float2, (int)1, (bool)0, (int)6, (int)4, (int)4>(cublasTransposeParams<T2>, const T1 *, T1 *, const T2 *), 2023-May-07 14:57:46, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                         135.28
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         135.27
    ---------------------------------------------------------------------- --------------- ------------------------------

  void transpose_readWrite_alignment_kernel<float2, float2, (int)1, (bool)0, (int)6, (int)4, (int)4>(cublasTransposeParams<T2>, const T1 *, T1 *, const T2 *), 2023-May-07 14:57:46, Context 1, Stream 19
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           8.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                           8.45
    ---------------------------------------------------------------------- --------------- ------------------------------

  void gemv2T_kernel_val<int, int, float2, float2, float2, (int)128, (int)16, (int)2, (int)2, (bool)0, (bool)0, cublasGemvParams<cublasGemvTensorStridedBatched<const float2>, cublasGemvTensorStridedBatched<float2>, float2>>(T12, T5, T5), 2023-May-07 14:57:47, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                         143.73
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         152.04
    ---------------------------------------------------------------------- --------------- ------------------------------

  void transpose_readWrite_alignment_kernel<float2, float2, (int)1, (bool)0, (int)6, (int)4, (int)4>(cublasTransposeParams<T2>, const T1 *, T1 *, const T2 *), 2023-May-07 14:57:49, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           4.23
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                           4.23
    ---------------------------------------------------------------------- --------------- ------------------------------

  void DSE::vector_fft<(int)1, (int)2, (int)256, (int)16, (int)16, (int)1, float, float, float2>(T9 *, T9 *, int, int3, int3), 2023-May-07 14:57:49, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           4.24
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                           4.24
    ---------------------------------------------------------------------- --------------- ------------------------------

  void DSE::regular_fft_clip<(int)1, (int)2, (int)256, (int)16, (int)16, (int)1, float, float, float2>(T7 *, T9 *, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, T7 *, T7 *), 2023-May-07 14:57:49, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           4.28
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                           4.24
    ---------------------------------------------------------------------- --------------- ------------------------------

