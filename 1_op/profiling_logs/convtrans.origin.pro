Profiling ./conv2d_transposed 16 256 2 2 448 4 4 1 2 1
==PROF== Connected to process 2264844 (/home/zly/Work/OSDI23-EinNet-AE/1_op/conv2d_transposed)
==PROF== Profiling "scalePackedTensor_kernel" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 1: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 2: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 3: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 4: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 5: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 6: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 7: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 8: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 9: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 10: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 11: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 12: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 13: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 14: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 15: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 16: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 17: 0%....50%....100% - 1 pass
==PROF== Profiling "scalePackedTensor_kernel" - 18: 0%....50%....100% - 1 pass
==PROF== Profiling "dgrad_engine" - 19: 0%....50%....100% - 1 pass
Algo: CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
Time: 883.116 ms
883.116
==PROF== Disconnected from process 2264844
[2264844] conv2d_transposed@127.0.0.1
  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:04, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:07, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.53
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:07, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:08, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.42
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:08, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:08, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.51
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:08, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:09, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.27
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:09, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:11, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.57
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:11, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.20
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:11, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.40
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:12, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:12, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.40
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:12, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:13, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.42
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:13, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:13, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.42
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::ops::scalePackedTensor_kernel<float, float>(cudnnTensor4dStruct, T1 *, T2), 2023-May-07 14:54:15, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                           3.46
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                     byte                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

  void cudnn::detail::dgrad_engine<float, (int)512, (int)6, (int)5, (int)3, (int)3, (int)3, (bool)0>(int, int, int, const T1 *, int, const T1 *, int, T1 *, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int), 2023-May-07 14:54:15, Context 1, Stream 7
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           7.50
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         121.38
    ---------------------------------------------------------------------- --------------- ------------------------------

