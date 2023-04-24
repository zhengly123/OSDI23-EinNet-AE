../test_op_g2bmm --gtest_filter=*.origin
Running main() from /home/whj/workspace/InfiniTensor/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.origin
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.origin
Graph Tensors:
Tensor 805, Fuid 301, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7ff99e000000
Tensor 806, Fuid 302, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7ff99f400000
Tensor 808, Fuid 303, shape [8,10000,2001], dtype Float32, tensorType 3, source 807, targets [], CUDA Runtime, 0x7ff8b8000000
Graph operators:
OP 807, pred [], succ [], G2BMM([width=1000,act=0],A=805,B=806,C=808, TTbmnkd: 8, 10000, 1000, 64, 4)

[       OK ] op_G2BMM.origin (14536 ms)
[----------] 1 test from op_G2BMM (14536 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (14536 ms total)
[  PASSED  ] 1 test.
Running main() from /home/whj/workspace/InfiniTensor/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.origin
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.origin
==PROF== Connected to process 2613187 (/home/whj/workspace/InfiniTensor/cuda-build/test_op_g2bmm)
==PROF== Profiling "sg2bmm_bs1_n10000_m64_w1000_d..." - 1: 0%....50%....100%Graph Tensors:
Tensor 805, Fuid 301, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f482c000000
Tensor 806, Fuid 302, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f482a000000
Tensor 808, Fuid 303, shape [8,10000,2001], dtype Float32, tensorType 3, source 807, targets [], CUDA Runtime, 0x7f4778000000
Graph operators:
OP 807, pred [], succ [], G2BMM([width=1000,act=0],A=805,B=806,C=808, TTbmnkd: 8, 10000, 1000, 64, 4)

[       OK ] op_G2BMM.origin (21506 ms)
[----------] 1 test from op_G2BMM (21506 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (21506 ms total)
[  PASSED  ] 1 test.
 - 13 passes
==PROF== Disconnected from process 2613187
[2613187] test_op_g2bmm@127.0.0.1
  infini::sg2bmm_bs1_n10000_m64_w1000_d4_kernel0_tvm10(float*, float*, float*), 2023-Apr-25 02:51:38, Context 1, Stream 25
    Section: GPU Speed Of Light
    ---------------------------------------------------------------------- --------------- ------------------------------
    DRAM Frequency                                                           cycle/nsecond                           1.21
    SM Frequency                                                             cycle/usecond                         764.14
    Elapsed Cycles                                                                   cycle                     10,135,802
    Memory [%]                                                                           %                          82.53
    SOL DRAM                                                                             %                           1.61
    Duration                                                                       msecond                          13.26
    SOL L1/TEX Cache                                                                     %                          75.89
    SOL L2 Cache                                                                         %                          82.53
    SM Active Cycles                                                                 cycle                   9,834,924.98
    SM [%]                                                                               %                          11.57
    ---------------------------------------------------------------------- --------------- ------------------------------
    OK    The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device. To   
          further improve performance, work will likely need to be shifted from the most utilized to another unit.      
          Start by analyzing workloads in the Memory Workload Analysis section.                                         

    Section: Launch Statistics
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Size                                                                                                        667
    Grid Size                                                                                                       2,000
    Registers Per Thread                                                   register/thread                             80
    Shared Memory Configuration Size                                                 Kbyte                          65.54
    Driver Shared Memory Per Block                                             Kbyte/block                           1.02
    Dynamic Shared Memory Per Block                                             byte/block                              0
    Static Shared Memory Per Block                                             Kbyte/block                          32.16
    Threads                                                                         thread                      1,334,000
    Waves Per SM                                                                                                    18.52
    ---------------------------------------------------------------------- --------------- ------------------------------
    WRN   Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 667    
          threads per block. Consequently, some threads in a warp are masked off and those hardware resources are       
          unused. Try changing the number of threads per block to be a multiple of 32 threads. Between 128 and 256      
          threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one      
          large thread block per multiprocessor if latency affects performance. This is particularly beneficial to      
          kernels that frequently call __syncthreads().                                                                 

    Section: Occupancy
    ---------------------------------------------------------------------- --------------- ------------------------------
    Block Limit SM                                                                   block                             32
    Block Limit Registers                                                            block                              1
    Block Limit Shared Mem                                                           block                              5
    Block Limit Warps                                                                block                              3
    Theoretical Active Warps per SM                                                   warp                             21
    Theoretical Occupancy                                                                %                          32.81
    Achieved Occupancy                                                                   %                          32.61
    Achieved Active Warps Per SM                                                      warp                          20.87
    ---------------------------------------------------------------------- --------------- ------------------------------

Running main() from /home/whj/workspace/InfiniTensor/3rd-party/googletest/googletest/src/gtest_main.cc
Note: Google Test filter = *.origin
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from op_G2BMM
[ RUN      ] op_G2BMM.origin
==PROF== Connected to process 2613819 (/home/whj/workspace/InfiniTensor/cuda-build/test_op_g2bmm)
==PROF== Profiling "sg2bmm_bs1_n10000_m64_w1000_d..." - 1: 0%....50%....100%Graph Tensors:
Tensor 805, Fuid 301, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f9dfe000000
Tensor 806, Fuid 302, shape [8,10000,64], dtype Float32, tensorType 3, source None, targets [807], CUDA Runtime, 0x7f9dfc000000
Tensor 808, Fuid 303, shape [8,10000,2001], dtype Float32, tensorType 3, source 807, targets [], CUDA Runtime, 0x7f9dd4000000
Graph operators:
OP 807, pred [], succ [], G2BMM([width=1000,act=0],A=805,B=806,C=808, TTbmnkd: 8, 10000, 1000, 64, 4)

[       OK ] op_G2BMM.origin (20823 ms)
[----------] 1 test from op_G2BMM (20823 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (20823 ms total)
[  PASSED  ] 1 test.
 - 3 passes
==PROF== Disconnected from process 2613819
[2613819] test_op_g2bmm@127.0.0.1
  infini::sg2bmm_bs1_n10000_m64_w1000_d4_kernel0_tvm10(float*, float*, float*), 2023-Apr-25 02:52:00, Context 1, Stream 25
    Section: Command line profiler metrics
    ---------------------------------------------------------------------- --------------- ------------------------------
    dram__bytes_read.sum                                                             Mbyte                          20.96
    dram__bytes_write.sum                                                            Mbyte                         311.37
    dram__sectors_read.sum                                                          sector                        655,148
    lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum                    Gbyte                          19.75
    lts__t_sectors_op_atom.sum                                                      sector                              0
    lts__t_sectors_op_read.sum                                                      sector                    661,449,355
    lts__t_sectors_op_red.sum                                                       sector                              0
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum                               inst                              0
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum                               inst                  5,122,560,000
    smsp__sass_thread_inst_executed_op_fmul_pred_on.sum                               inst                              0
    ---------------------------------------------------------------------- --------------- ------------------------------

