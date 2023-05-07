Profiling /home/zly/Work/OSDI23-EinNet-AE/build/test_op_g2bmm --gtest_filter=*.origin
Running main() from /home/zly/Work/OSDI23-EinNet-AE/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.origin
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.origin
==PROF== Connected to process 2265646 (/home/zly/Work/OSDI23-EinNet-AE/build/test_op_g2bmm)
Time: 10.6079 ms
==PROF== Profiling "sg2bmm_bs1_n10000_m64_w1000_d..." - 0: 0%....50%....100% - 1 pass
[       OK ] op_G2BMM.origin (7890 ms)
[----------] 1 test from op_G2BMM (7890 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (7890 ms total)
[  PASSED  ] 1 test.
==PROF== Disconnected from process 2265646
[2265646] test_op_g2bmm@127.0.0.1
  infini::sg2bmm_bs1_n10000_m64_w1000_d4_kernel0_tvm10(float *, float *, float *), 2023-May-07 14:58:09, Context 1, Stream 31
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                          20.84
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Gbyte                          19.75
    ---------------------------------------------------------------------- --------------- ------------------------------

