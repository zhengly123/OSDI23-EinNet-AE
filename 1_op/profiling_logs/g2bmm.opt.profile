../test_op_g2bmm --gtest_filter=*.optimized
Running main() from /home/whj/workspace/InfiniTensor/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.optimized
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.optimized
Graph Tensors:
Tensor 805, Fuid 301, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7fa2fe000000
Tensor 806, Fuid 302, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7fa2ff400000
Tensor 808, Fuid 303, shape [8,10000,2001], dtype Float32, tensorType 3, source 807, targets [], CUDA Runtime, 0x7fa218000000
Graph operators:
OP 807, pred [], succ [], G2BMM([width=1000,act=0],A=805,B=806,C=808, TTbmnkd: 8, 10000, 1000, 64, 1)

[       OK ] op_G2BMM.optimized (14449 ms)
[----------] 1 test from op_G2BMM (14449 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (14449 ms total)
[  PASSED  ] 1 test.
Running main() from /home/whj/workspace/InfiniTensor/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.optimized
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.optimized
==PROF== Connected to process 2614852 (/home/whj/workspace/InfiniTensor/cuda-build/test_op_g2bmm)
==PROF== Profiling "sg2bmm_bs1_n10000_m64_w1000_d..." - 1: 0%....50%....100%Graph Tensors:
Tensor 805, Fuid 301, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f7412000000
Tensor 806, Fuid 302, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f7410000000
Tensor 808, Fuid 303, shape [8,10000,2001], dtype Float32, tensorType 3, source 807, targets [], CUDA Runtime, 0x7f7358000000
Graph operators:
OP 807, pred [], succ [], G2BMM([width=1000,act=0],A=805,B=806,C=808, TTbmnkd: 8, 10000, 1000, 64, 1)

[       OK ] op_G2BMM.optimized (21568 ms)
[----------] 1 test from op_G2BMM (21568 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (21568 ms total)
[  PASSED  ] 1 test.
 - 13 passes
==PROF== Disconnected from process 2614852
[2614852] test_op_g2bmm@127.0.0.1
  infini::sg2bmm_bs1_n10000_m64_w1000_d1_kernel0_tvm10(float*, float*, float*), 2023-Apr-25 02:52:37, Context 1, Stream 25
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    DRAM Frequency                                                           cycle/nsecond                           1.21
    SM Frequency                                                             cycle/usecond                         763.55
    Elapsed Cycles                                                                   cycle                      2,184,141
    Memory [%]                                                                           %                          91.78
    SOL DRAM                                                                             %                           9.08
    Duration                                                                       msecond                           2.86
    SOL L1/TEX Cache                                                                     %                          92.13
    SOL L2 Cache                                                                         %                          26.39
    SM Active Cycles                                                                 cycle                   2,175,737.91
    SM [%]                                                                               %                          41.95
    ---------------------------------------------------------------------- --------------- ------------------------------
    OK    The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing workloads in the Memory Workload Analysis section.                                         

    Section: Launch Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Size                                                                                                        290
    Grid Size                                                                                                      23,000
    Registers Per Thread                                                   register/thread                             48
    Shared Memory Configuration Size                                                 Kbyte                          32.77
    Driver Shared Memory Per Block                                             Kbyte/block                           1.02
    Dynamic Shared Memory Per Block                                             byte/block                              0
    Static Shared Memory Per Block                                             Kbyte/block                           2.66
    Threads                                                                         thread                      6,670,000
    Waves Per SM                                                                                                    53.24
    ---------------------------------------------------------------------- --------------- ------------------------------
    WRN   Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 290    
          threads per block. Consequently, some threads in a warp are masked off and those hardware resources are       
          unused. Try changing the number of threads per block to be a multiple of 32 threads. Between 128 and 256      
          threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one      
          large thread block per multiprocessor if latency affects performance. This is particularly beneficial to      
          kernels that frequently call __syncthreads().                                                                 

    Section: Occupancy
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Limit SM                                                                   block                             32
    Block Limit Registers                                                            block                              4
    Block Limit Shared Mem                                                           block                             45
    Block Limit Warps                                                                block                              6
    Theoretical Active Warps per SM                                                   warp                             40
    Theoretical Occupancy                                                                %                          62.50
    Achieved Occupancy                                                                   %                          61.49
    Achieved Active Warps Per SM                                                      warp                          39.35
    ---------------------------------------------------------------------- --------------- ------------------------------

Running main() from /home/whj/workspace/InfiniTensor/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.optimized
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.optimized
==PROF== Connected to process 2615901 (/home/whj/workspace/InfiniTensor/cuda-build/test_op_g2bmm)
==PROF== Profiling "sg2bmm_bs1_n10000_m64_w1000_d..." - 1: 0%....50%....100%Graph Tensors:
Tensor 805, Fuid 301, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f6dea000000
Tensor 806, Fuid 302, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f6de8000000
Tensor 808, Fuid 303, shape [8,10000,2001], dtype Float32, tensorType 3, source 807, targets [], CUDA Runtime, 0x7f6d38000000
Graph operators:
OP 807, pred [], succ [], G2BMM([width=1000,act=0],A=805,B=806,C=808, TTbmnkd: 8, 10000, 1000, 64, 1)

[       OK ] op_G2BMM.optimized (20372 ms)
[----------] 1 test from op_G2BMM (20372 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (20372 ms total)
[  PASSED  ] 1 test.
 - 3 passes
==PROF== Disconnected from process 2615901
[2615901] test_op_g2bmm@127.0.0.1
  infini::sg2bmm_bs1_n10000_m64_w1000_d1_kernel0_tvm10(float*, float*, float*), 2023-Apr-25 02:52:58, Context 1, Stream 25
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                          20.68
    dram__bytes_write.sum                                                            Mbyte                         382.55
    dram__sectors_read.sum                                                          sector                        646,280
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Mbyte                         817.87
    lts__t_sectors_op_atom.sum                                                      sector                              0
    lts__t_sectors_op_read.sum                                                      sector                     26,297,034
    lts__t_sectors_op_red.sum                                                       sector                              0
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum                               inst                              0
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum                               inst                  5,122,560,000
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum                               inst                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

