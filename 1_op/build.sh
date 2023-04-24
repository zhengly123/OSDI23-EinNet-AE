nvcc -gencode arch=compute_80,code=sm_80 -O3 conv.cu -DUSE_FP32 -o conv -lcublas -lcudnn -lcurand -std=c++11
nvcc -gencode arch=compute_80,code=sm_80 -O3 conv_transpose2gemm.cu -DUSE_FP32 -o conv_transpose2gemm -lcublas -lcudnn -lcurand -std=c++11
nvcc -gencode arch=compute_80,code=sm_80 -O3 conv2d_transposed.cu -DUSE_FP32 -o conv2d_transposed -lcublas -lcudnn -lcurand -std=c++11
nvcc -gencode arch=compute_80,code=sm_80 -O3 conv2gemm.cu -DUSE_FP32 -o conv2gemm -lcublas -lcudnn -lcurand -std=c++11
