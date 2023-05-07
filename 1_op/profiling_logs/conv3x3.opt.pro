Profiling /home/zly/Work/OSDI23-EinNet-AE/build/test_op_conv3x3 --gtest_filter=*.optimized
Running main() from /home/zly/Work/OSDI23-EinNet-AE/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.optimized
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_Conv3x3
[ RUN      ] op_Conv3x3.optimized
==PROF== Connected to process 2264803 (/home/zly/Work/OSDI23-EinNet-AE/build/test_op_conv3x3)
Time: 0.0634241 ms
==PROF== Profiling "ampere_sgemm_32x32_sliced1x4_tn" - 0: 0%....50%....100% - 1 pass
==PROF== Profiling "reduce_merge_conv_3x3" - 1: 0%....50%....100% - 1 pass
[       OK ] op_Conv3x3.optimized (8081 ms)
[----------] 1 test from op_Conv3x3 (8081 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (8081 ms total)
[  PASSED  ] 1 test.
==PROF== Disconnected from process 2264803
[2264803] test_op_conv3x3@127.0.0.1
  ampere_sgemm_32x32_sliced1x4_tn, 2023-May-07 14:53:56, Context 1, Stream 31
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                           9.55
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                          27.88
    ---------------------------------------------------------------------- --------------- ------------------------------

  void reduce_merge_conv_3x3<float>(T1 *, T1 *, T1 *, int, int, int, int, int, int, int, int, int, int, int, int, int, int), 2023-May-07 14:53:57, Context 1, Stream 31
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Kbyte                         908.67
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                           1.18
    ---------------------------------------------------------------------- --------------- ------------------------------

