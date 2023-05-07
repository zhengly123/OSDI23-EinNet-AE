Profiling /home/zly/Work/OSDI23-EinNet-AE/build/test_op_conv3x3 --gtest_filter=*.origin
Running main() from /home/zly/Work/OSDI23-EinNet-AE/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.origin
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_Conv3x3
[ RUN      ] op_Conv3x3.origin
==PROF== Connected to process 2264761 (/home/zly/Work/OSDI23-EinNet-AE/build/test_op_conv3x3)
Time: 0.157622 ms
==PROF== Profiling "generateWinogradTilesKernel" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "ampere_scudnn_winograd_128x12..." - 1: 0%....50%....100% - 1 pass
[       OK ] op_Conv3x3.origin (10654 ms)
[----------] 1 test from op_Conv3x3 (10654 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (10654 ms total)
[  PASSED  ] 1 test.
==PROF== Disconnected from process 2264761
[2264761] test_op_conv3x3@127.0.0.1
  void cudnn::winograd::generateWinogradTilesKernel<(int)0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<T2, T3>), 2023-May-07 14:53:47, Context 1, Stream 31
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           9.45
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                          11.26
    ---------------------------------------------------------------------- --------------- ------------------------------

  ampere_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148n_nt_v1, 2023-May-07 14:53:48, Context 1, Stream 31
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                          16.91
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                          18.24
    ---------------------------------------------------------------------- --------------- ------------------------------

