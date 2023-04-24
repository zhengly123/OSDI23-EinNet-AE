#include "cuda_fp16.h"
#include "dbg.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <string>
#include <sys/time.h>
using namespace std;

using dtype = half;

#define checkCUDA(ans)                                                         \
    { _checkCUDA((ans), __FILE__, __LINE__); }
inline void _checkCUDA(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s:%d %s\n", file, line,
                cudaGetErrorString(code));
        if (abort)
            exit(code);
    }
}

#define checkCURAND(ans)                                                       \
    { _checkCURAND((ans), __FILE__, __LINE__); }
inline void _checkCURAND(curandStatus_t code, const char *file, int line,
                         bool abort = true) {
    if (code != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRAND error: %s:%d %d\n", file, line, int(code));
        if (abort)
            exit(code);
    }
}

#define checkCUBLAS(ans)                                                       \
    { _checkCURAND((ans), __FILE__, __LINE__); }
inline void _checkCURAND(cublasStatus_t code, const char *file, int line,
                         bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s:%d %d\n", file, line, int(code));
        if (abort)
            exit(code);
    }
}

#define checkCUDNN(ans)                                                        \
    { _checkCUDNN((ans), __FILE__, __LINE__); }
inline void _checkCUDNN(cudnnStatus_t code, const char *file, int line,
                        bool abort = true) {
    if (code != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN error: %s:%d %d\n", file, line, int(code));
        if (abort)
            exit(code);
    }
}

const int warmup = 200, repeat = 200;

// size为转换前float数据个数，转换后由size/2个half2存储所有数据
__global__ void float2HalfVec(half *dst, float *src, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride)
        dst[i] = __float2half_rn(src[i]);
}

template <typename T> void d_rand(T *d_a, int size) { assert(0); }

template <> void d_rand<float>(float *d_a, int size) {
    static curandGenerator_t gen;
    static bool inited = false;
    if (!inited) {
        checkCURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        // checkCURAND(
        //     curandSetPseudoRandomGeneratorSeed(gen, (unsigned long
        //     long)clock()));
        checkCURAND(
            curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)0));
        inited = true;
    }
    checkCURAND(curandGenerateUniform(gen, d_a, size));
}

__global__ void d_fake_rand(float *d_a, int size) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < size; i += stride) {
        d_a[i] = i % 2;
        // d_a[i] = 1;
        // d_a[i] = 0;
    }
}

template <> void d_rand<half>(half *d_a, int size) {
    float *d_temp;
    checkCUDA(cudaMalloc(&d_temp, size * sizeof(float)));
    d_rand<float>(d_temp, size);
    // d_fake_rand<<<256, 256>>>(d_temp, size);
    float2HalfVec<<<256, 256>>>(d_a, d_temp, size);
    checkCUDA(cudaDeviceSynchronize());
}

template <typename T> T *d_malloc(size_t size, bool random = true) {
    T *ret;
    checkCUDA(cudaMalloc(&ret, size * sizeof(T)));
    if (random) {
        d_rand<T>(ret, size);
    }
    checkCUDA(cudaDeviceSynchronize());
    return ret;
}

__global__ void d_compare(half *x, half *y, int n, bool verbose=false) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    if (start >0)
        return;
    const half eps =  0.25;
    int n_err = 0;
    for (int i = 0; i < n; ++i) {
        half delta = (x[i] - y[i]);
        if ((__habs(delta) > eps) &&
            (__habs(delta / x[i]) > eps || __habs(delta / y[i]) > eps)) {
            ++n_err;
            if (verbose) {
                printf("Error on %d x= %f, y= %f\n", i, __half2float(x[i]),
                       __half2float(y[i]));
            }
        }
    }
    printf("n_err = %d\n", n_err);
}

__global__ void reduce_3x3(int n, int lda, int rs, half *x, half *y) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n2 = n / 2;
    half2 *x2 = (half2 *)x, *y2 = (half2 *)y;
#define idx(k, i) ((k) * (n2) + (i))

    for (int i = start; i < n2; i += stride) {
        half2 tmp = __hadd2(x2[idx(0, i)], x2[idx(1, i)]);
        for (int k = 2; k < rs; ++k)
            tmp = __hadd2(tmp, x2[idx(k, i)]);
        y2[i] = tmp;
    }
    // // first thread handles singleton for odd arrays
    // if (start == 0 && (n % 2))
    //     y[n - 1] = __hfma(a, x[n - 1], y[n - 1]);
}

// assert(f%2==0);
__global__ void reduce_3x3_offset(const int n, const int f, const int h,
                                  const int w, half *x, half *y) {
    // const int R = 3, S = 3;
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n2 = n / 2;
    // x[r,s,h,w,f]
    // two halves have different f but the same h,w
    half2 *x2 = (half2 *)x, *y2 = (half2 *)y;
#define idx(k, i) ((k) * (n2) + (i))
    for (int i = start; i < n2; i += stride) {
        const int hh = i * 2 / f / w;
        const int ww = i * 2 / f % w;
        half2 tmp;
        *(float *)&tmp = 0;
        for (int r = -1; r <= 1; ++r) {
            for (int s = -1; s <= 1; ++s) {
                const int sh = hh + r, sw = ww + s;
                if (0 <= sh && sh < h && 0 <= sw && sw < w) {
                    tmp = __hadd2(
                        tmp, x2[idx((r + 1) * 3 + (s + 1), i + r * w + s)]);
                    // if (hh<2 && ww <2 && 2*i %f ==0) {
                    //     half2 t = x2[idx((r + 1) * 3 + (s + 1), i + r * w + s)];
                    //     float a =
                    //         __half2float(__low2half(t));
                    //     float b =
                    //         __half2float(__high2half(t));
                    //     printf("i=%d (%d, %d) %d %d %f %f\n", i, hh, ww, r, s, a, b);
                    // }
                }
            }
        }
        y2[i] = tmp;
        // if (i==0)
        //     printf("i=%d sum=%f\n", i, __half2float(__high2half(tmp)));
    }
    // // first thread handles singleton for odd arrays
    // if (start == 0 && (n % 2))
    //     y[n - 1] = __hfma(a, x[n - 1], y[n - 1]);
}

__global__ void half2float(const int n, const half *x, float *y) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < n; i += stride) {
        y[i] = __half2float(x[i]);
    }
}

// [m,n] -> [n,m]
__global__ void transpose_rsfc2fcrs(dtype *x, dtype *y, int m, int n) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < m * n; i += stride) {
        int mm = i / n, nn = i % n;
        y[nn * m + mm] = x[i];
    }
}

__global__ void print_half(half *x, int len) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    if (start == 0) {
        for (int i = 0; i < len; ++i) {
            float t = __half2float(x[i]);
            printf("%.3f ", t);
        }
        printf("\n");
    }
}

#define index4_nchw(nn, ff, hh, ww, f, h, w)                                   \
    ((((nn)*f + (ff)) * h + (hh)) * w + (ww));

// [n,h,w,c]
#define index4_nhwc(nn, ff, hh, ww, f, h, w)                                   \
    ((((nn)*h + (hh)) * w + (ww)) * f + (ff));

// __global__ void hconv(half *input, half *kernel, half *output, int n, int
// c,
//                       int h, int w, int f, int r, int s) {
//     int start = threadIdx.x + blockDim.x * blockIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int nn = 0; nn < n; ++nn)
//         for (int ff = 0; ff < f; ++ff)
//             for (int hh = 0; hh < h; ++hh)
//                 for (int ww = 0; ww < w; ++ww) {
//                     output[index4_nhwc(nn, ff, hh, ww, f, h, w)] = 0;
//                     for (int cc = 0; cc < c; ++cc)
//                         for (int rr = 0; rr < r; ++rr)
//                             for (int ss = 0; ss < s; ++ss)
//                                 output[index4_nhwc(nn, ff, hh, ww, f, h,
//                                 w)]
//                                 +=
//                                     input[index4_nhwc(nn, cc, hh, ww, c,
//                                     h,
//                                                       w)] *
//                                     kernel[index4_nchw(ff, cc, 0, 0, c,
//                                     r, s)];
//                 }
// }

void conv_cudnn(int n, int c, int h, int w, int f, int r, int s, dtype *d_input,
                dtype *d_kernel, dtype *d_output) {
    auto DATA_TYPE =
        typeid(dtype) == typeid(float) ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    // input
    // const size_t input_size = n * c * h * w;
    // const size_t input_bytes = input_size * sizeof(dtype);

    // kernel
    // const size_t kernel_size = f * c * r * s;
    // const size_t kernel_bytes = kernel_size * sizeof(dtype);

    const int PAD_HEIGHT = 1, PAD_WIDTH = 1;
    const int VERTICAL_STRIDE = 1;
    const int HORIZONTAL_STRIDE = 1;
    const int DILATION_HEIGHT = 1;
    const int DILATION_WIDTH = 1;
    auto CONV_MODE = CUDNN_CROSS_CORRELATION;
    const int GROUP_COUNT = 1;
    const auto MATH_TYPE = CUDNN_DEFAULT_MATH;
    int OUTPUT_BATCH_SIZE = 0, OUTPUT_CHANNELS = 0, OUTPUT_HEIGHT = 0,
        OUTPUT_WIDTH = 0;
    cudnnTensorFormat_t INPUT_TENSOR_FORMAT = CUDNN_TENSOR_NHWC;
    cudnnTensorFormat_t KERNEL_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
    cudnnTensorFormat_t OUTPUT_TENSOR_FORMAT = CUDNN_TENSOR_NHWC;

    const int INPUT_BATCH_SIZE = n;
    const int INPUT_CHANNELS = c;
    const int INPUT_HEIGHT = h;
    const int INPUT_WIDTH = w;
    const int INPUT_CHANNELS_PER_GROUP = c;
    const int KERNEL_IN_CHANNELS = INPUT_CHANNELS_PER_GROUP;
    const int KERNEL_OUT_CHANNELS = f; // #kernels = #output.channels
    const int KERNEL_HEIGHT = r;
    const int KERNEL_WIDTH = s;
    const int choose_algo = 0;

    const cudnnConvolutionFwdAlgo_t total_conv_algo[] = {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
    const char algo_name[8][50] = {
        "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
        "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
        "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
        "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
        "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
        "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
        "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
        "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED",
    };
    const char math_types[3][50] = {
        "CUDNN_DEFAULT_MATH",
        "CUDNN_TENSOR_OP_MATH",
        "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION",
    };

    // const char *precision_types[3] = {"float32", "float64", "float16"};
    // descriptor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/INPUT_TENSOR_FORMAT,
                                          /*dataType=*/DATA_TYPE,
                                          /*batch_size=*/INPUT_BATCH_SIZE,
                                          /*channels=*/INPUT_CHANNELS,
                                          /*height=*/INPUT_HEIGHT,
                                          /*width=*/INPUT_WIDTH));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                          /*dataType=*/DATA_TYPE,
                                          /*format=*/KERNEL_TENSOR_FORMAT,
                                          /*out_channels=*/KERNEL_OUT_CHANNELS,
                                          /*in_channels=*/KERNEL_IN_CHANNELS,
                                          /*kernel_height=*/KERNEL_HEIGHT,
                                          /*kernel_width=*/KERNEL_WIDTH));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(
        cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/PAD_HEIGHT,
                                        /*pad_width=*/PAD_WIDTH,
                                        /*vertical_stride=*/VERTICAL_STRIDE,
                                        /*horizontal_stride=*/HORIZONTAL_STRIDE,
                                        /*dilation_height=*/DILATION_HEIGHT,
                                        /*dilation_width=*/DILATION_WIDTH,
                                        /*mode=*/CONV_MODE,
                                        /*conputeType=*/DATA_TYPE));

    if (GROUP_COUNT > 1)
        checkCUDNN(cudnnSetConvolutionGroupCount(convolution_descriptor,
                                                 /*group_count=*/GROUP_COUNT));
    checkCUDNN(cudnnSetConvolutionMathType(convolution_descriptor, MATH_TYPE));
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convolution_descriptor, input_descriptor, kernel_descriptor,
        &OUTPUT_BATCH_SIZE, &OUTPUT_CHANNELS, &OUTPUT_HEIGHT, &OUTPUT_WIDTH));

    // std::cout << "Precision: " << expr << std::endl;
    // std::cout << "Rounds: " << expr << std::endl;
    std::cout << "Group count: " << GROUP_COUNT << ", "
              << "Math type: " << math_types[MATH_TYPE] << std::endl;
    std::cout << "Input dims: " << INPUT_BATCH_SIZE << ", " << INPUT_CHANNELS
              << ", " << INPUT_HEIGHT << ", " << INPUT_WIDTH << std::endl;
    std::cout << "Kernel dims: " << KERNEL_OUT_CHANNELS << ", " << c << ", "
              << KERNEL_HEIGHT << ", " << KERNEL_WIDTH << std::endl;
    std::cout << "Output dims: " << OUTPUT_BATCH_SIZE << ", " << OUTPUT_CHANNELS
              << ", " << OUTPUT_HEIGHT << ", " << OUTPUT_WIDTH << std::endl;

    // size_t output_bytes = OUTPUT_BATCH_SIZE * OUTPUT_CHANNELS * OUTPUT_HEIGHT
    // *
    //                       OUTPUT_WIDTH * sizeof(dtype);

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/OUTPUT_TENSOR_FORMAT,
                                          /*dataType=*/DATA_TYPE,
                                          /*batch_size=*/OUTPUT_BATCH_SIZE,
                                          /*channels=*/OUTPUT_CHANNELS,
                                          /*height=*/OUTPUT_HEIGHT,
                                          /*width=*/OUTPUT_WIDTH));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    convolution_algorithm = total_conv_algo[choose_algo];

    std::cout << "Chosen algorithm: " << algo_name[convolution_algorithm]
              << std::endl;

    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, input_descriptor, kernel_descriptor, convolution_descriptor,
        output_descriptor, convolution_algorithm, &workspace_bytes));
    std::cout << "Workspace size: " << (workspace_bytes) << "B" << std::endl;

    void *d_workspace{nullptr};
    if (workspace_bytes > 0)
        checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));

    const float alpha = 1, beta = 0;
    // warmup
    checkCUDNN(cudnnConvolutionForward(
        cudnn, &alpha, input_descriptor, d_input, kernel_descriptor, d_kernel,
        convolution_descriptor, convolution_algorithm, d_workspace,
        workspace_bytes, &beta, output_descriptor, d_output));
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaFree(d_workspace));
}

void conv2gemm(const int n, const int c, const int h, const int w, const int f,
               const int r, const int s) {
    cublasHandle_t cublas;
    checkCUBLAS(cublasCreate(&cublas));
    assert(n == 1);
    const int B = n * r * s, M = f, N = h * w, K = c;
    dtype *d_input, *d_kernel, *d_temp, *d_output;
    size_t size_input = n * c * h * w, size_kernel = f * c * r * s,
           size_temp = r * s * f * h * w, size_output = f * h * w;
    d_input = d_malloc<dtype>(size_input);
    d_kernel = d_malloc<dtype>(size_kernel);
    d_temp = d_malloc<dtype>(size_temp, false);
    d_output = d_malloc<dtype>(size_output, false);

    int algo = 103;
    int transa = 1, transb = 0;
    int ld_input = K, ld_kernel = K, ld_temp = M;
    // seems alpha/beta on device result in seg fault
    half alpha = __float2half(1), beta = __float2half(0);

    cudaDataType_t data_type =
        (typeid(float) == typeid(dtype)) ? CUDA_R_32F : CUDA_R_16F;
    for (int t = 0; t < 2; ++t) {
        auto tbegin = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < warmup; ++i) {
            // clang-format off
            // Test[1].transa = 1;
            // Test[1].transb = 0;
            // Test[1].b = b;
            // Test[1].A = d_A;
            // Test[1].lda = k;
            // Test[1].B = d_B;
            // Test[1].ldb = k;
            // Test[1].C = d_C;
            // Test[1].ldc = m;
            // Test[1].m = m;
            // Test[1].n = n;
            // Test[1].k = k;
            cublasGemmStridedBatchedEx(
                cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, M, N, K,
                &alpha, 
                d_kernel, data_type, ld_kernel, M*K, 
                d_input, data_type, ld_input, 0, 
                &beta, d_temp, data_type, ld_temp, M*N, 
                B, data_type, (cublasGemmAlgo_t)algo);
            // clang-format on

            // reduce
            // reduce_3x3<<<128, (M * N + 255) / (2 * 128)>>>(M * N, M * N, 9,
            //                                                d_temp, d_output);
            reduce_3x3_offset<<<128, (M * N + 255) / (2 * 128)>>>(
                M * N, f, h, w, d_temp, d_output);
            // TODO: try cublasSaxpy. No cublasHaxpy
            // for (int i = 0; i < B; ++i) {
            //     cublasSaxpy(cublas, M * N, &alpha, d_, 1, d_, 1);
            // }
        }
        checkCUDA(cudaDeviceSynchronize());
        auto tend = std::chrono::high_resolution_clock::now();
        // cudaProfilerStop();
        double average_ms =
            std::chrono::duration_cast<std::chrono::duration<double>>(tend -
                                                                      tbegin)
                .count() *
            1000 / repeat;
        if (t == 1)
            printf("Time: %lf ms\n", average_ms);
        // dbg(average_ms);
    }
    // // Retrieve output in float
    // float *d_output_float = d_malloc<float>(size_output, false);
    // float *h_output_float = new float[size_output];
    // half2float<<<128, (M * N + 255) / (2 * 128)>>>(M * N, d_output,
    //                                                d_output_float);
    // checkCUDA(cudaMemcpy(h_output_float, d_output_float,
    //                      size_output * sizeof(float), cudaMemcpyDeviceToHost));

    // // cuDNN correctness
    // dtype *d_kernel_fcrs = d_malloc<dtype>(size_kernel, false);
    // transpose_rsfc2fcrs<<<128, 256>>>(d_kernel, d_kernel_fcrs, r * s, f * c);
    // // print_half<<<1, 1>>>(d_kernel, 14 * 14 + 1);
    // // print_half<<<1, 1>>>(d_kernel_fcrs, 14 * 14 + 1);
    // checkCUDA(cudaDeviceSynchronize());
    // dtype *d_ans = d_malloc<dtype>(size_output, false);
    // float *d_ans_float = d_malloc<float>(size_output, false);
    // float *h_ans_float = new float[size_output];
    // conv_cudnn(n, c, h, w, f, r, s, d_input, d_kernel_fcrs, d_ans);
    // half2float<<<128, (M * N + 255) / (2 * 128)>>>(M * N, d_ans, d_ans_float);
    // checkCUDA(cudaMemcpy(h_ans_float, d_ans_float, size_output * sizeof(float),
    //                      cudaMemcpyDeviceToHost));
    // for (int i = 0; i < 5; ++i) {
    //     // if (fabs(h_ans_float[i] - h_output_float[i]) > 1e-3)
    //     dbg(i, h_ans_float[i], h_output_float[i]);
    // }
    // // int start=256*14
    // for (int i = 256; i < 256 + 5; ++i) {
    //     // if (fabs(h_ans_float[i] - h_output_float[i]) > 1e-3)
    //     dbg(i, h_ans_float[i], h_output_float[i]);
    // }

    // // debug
    // // auto h_input = new float[size_input];
    // // auto d_input_float = d_malloc<float>(size_input);
    // // checkCUDA(cudaMemcpy(h_))
    // print_half<<<1, 1>>>(d_input, 5);
    // print_half<<<1, 1>>>(d_kernel, 5);
    // checkCUDA(cudaDeviceSynchronize());
    // dbg("d_kernel_fcrs");
    //     print_half<<<1, 1>>>(d_kernel_fcrs, c*r*s);
    // dbg("d_output");
    // for (int i = 0; i < 66; ++i) {
    //     // string s = "d_output[NHWC][0, " + to_string(i / w) + ", " +
    //     //            to_string(i % w) + ", _]";
    //     // dbg(s);
    //     print_half<<<1, 1>>>(d_output + f * i, 5);
    //     checkCUDA(cudaDeviceSynchronize());
    // }
    // dbg("d_ans");
    // for (int i = 0; i < 66; ++i) {
    //     // dbg("d_ans[NHWC][0, " + to_string(i / 14) + ", " + to_string(i % 14) +
    //     //     ", _]");
    //     print_half<<<1, 1>>>(d_ans + f * i, 5);
    //     checkCUDA(cudaDeviceSynchronize());
    // }
    // // for (int i = 0; i < 9; ++i) {
    // //     dbg("Temp " + to_string(i));
    // //     print_half<<<1, 1>>>(d_temp + i * M * N, 5);
    // //     checkCUDA(cudaDeviceSynchronize());
    // // }
    // d_compare<<<1,1>>>(d_output, d_ans, size_output, true);

    // delete[] h_ans_float;
    // delete[] h_output_float;
    // checkCUDA(cudaFree(d_ans_float));
    // checkCUDA(cudaFree(d_kernel_fcrs));
    // checkCUDA(cudaFree(d_output_float));
    checkCUDA(cudaFree(d_temp));
    checkCUDA(cudaFree(d_kernel));
    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_output));
    checkCUBLAS(cublasDestroy(cublas));
}

int main(int argc, char *argv[]) {
    int n, c, h, w, f, r, s;
    if (argc != 8)
        cerr << argv[0] << " n c h w f r s" << endl;
    n = atoi(argv[1]);
    c = atoi(argv[2]);
    h = atoi(argv[3]);
    w = atoi(argv[4]);
    f = atoi(argv[5]);
    r = atoi(argv[6]);
    s = atoi(argv[7]);

    conv2gemm(n, c, h, w, f, r, s);
}
