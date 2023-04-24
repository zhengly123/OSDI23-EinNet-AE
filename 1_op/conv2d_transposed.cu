#include "cuda_fp16.h"
// #include "dbg.h"
#include "cuda_profiler_api.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <cublas_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <functional>
#include <iostream>
#include <string>
#include <sys/time.h>
using namespace std;

using dtype = float;

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

double time_cuda(std::function<void(void)> foo, int warmup = 5,
                 int repeat = 5) {
    checkCUDA(cudaDeviceSynchronize());
    for (int i = 0; i < warmup; ++i)
        foo();
    checkCUDA(cudaDeviceSynchronize());
    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i)
        foo();
    checkCUDA(cudaDeviceSynchronize());
    auto tend = std::chrono::high_resolution_clock::now();
    // cudaProfilerStop();
    double average_ms =
        std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin)
            .count() *
        1000 / repeat;
    // dbg(average_ms);
    return average_ms;
}

// c and f are swapped compared with conv2d_fwd
void conv2d_transposed_cudnn(
    cudnnHandle_t cudnn, int n, int c, int h, int w, int f, int r, int s,
    const int stride, const int padding, const int dilation, dtype *d_input,
    dtype *d_kernel, dtype *d_output,
    cudnnTensorFormat_t INPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW,
    cudnnTensorFormat_t KERNEL_TENSOR_FORMAT = CUDNN_TENSOR_NCHW,
    cudnnTensorFormat_t OUTPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW) {
    auto DATA_TYPE =
        typeid(dtype) == typeid(float) ? CUDNN_DATA_FLOAT : CUDNN_DATA_HALF;
    // input
    // const size_t input_size = n * c * h * w;
    // const size_t input_bytes = input_size * sizeof(dtype);

    // kernel
    // const size_t kernel_size = f * c * r * s;
    // const size_t kernel_bytes = kernel_size * sizeof(dtype);

    const int PAD_HEIGHT = padding, PAD_WIDTH = padding;
    const int VERTICAL_STRIDE = stride;
    const int HORIZONTAL_STRIDE = stride;
    const int DILATION_HEIGHT = dilation;
    const int DILATION_WIDTH = dilation;
    auto CONV_MODE = CUDNN_CROSS_CORRELATION;
    const int GROUP_COUNT = 1;
    const auto MATH_TYPE = CUDNN_DEFAULT_MATH;
    int OUTPUT_BATCH_SIZE = 0, OUTPUT_CHANNELS = 0, OUTPUT_HEIGHT = 0,
        OUTPUT_WIDTH = 0;
    // cudnnTensorFormat_t INPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
    // cudnnTensorFormat_t KERNEL_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;
    // cudnnTensorFormat_t OUTPUT_TENSOR_FORMAT = CUDNN_TENSOR_NCHW;

    const int INPUT_BATCH_SIZE = n;
    const int INPUT_CHANNELS = f;
    const int INPUT_HEIGHT = h;
    const int INPUT_WIDTH = w;
    const int INPUT_CHANNELS_PER_GROUP = f;
    const int KERNEL_IN_CHANNELS = INPUT_CHANNELS_PER_GROUP;
    const int KERNEL_OUT_CHANNELS = c; // #kernels = #output.channels
    const int KERNEL_HEIGHT = r;
    const int KERNEL_WIDTH = s;
    int choose_algo = 0;

    const int NUM_ALGOS = 7;
    const cudnnConvolutionBwdDataAlgo_t total_conv_algo[NUM_ALGOS] = {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, /* non-deterministic */
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT};
    const char algo_name[7][50] = {
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0", /* non-deterministic */
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED",
        "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT"};
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
    checkCUDNN(cudnnSetFilter4dDescriptor(
        kernel_descriptor,
        /*dataType=*/DATA_TYPE,
        /*format=*/KERNEL_TENSOR_FORMAT,
        /*out_channels=*/KERNEL_IN_CHANNELS, // swapped for bwd
        /*in_channels=*/KERNEL_OUT_CHANNELS,
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
    // checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
    //     convolution_descriptor, input_descriptor, kernel_descriptor,
    //     &OUTPUT_BATCH_SIZE, &OUTPUT_CHANNELS, &OUTPUT_HEIGHT,
    //     &OUTPUT_WIDTH));
    OUTPUT_BATCH_SIZE = INPUT_BATCH_SIZE;
    OUTPUT_CHANNELS = KERNEL_OUT_CHANNELS;
    OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * VERTICAL_STRIDE - 2 * PAD_HEIGHT +
                    DILATION_HEIGHT * (KERNEL_HEIGHT - 1) + 1;
    OUTPUT_WIDTH = (INPUT_WIDTH - 1) * HORIZONTAL_STRIDE - 2 * PAD_WIDTH +
                   DILATION_WIDTH * (KERNEL_WIDTH - 1) + 1;

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                          /*format=*/OUTPUT_TENSOR_FORMAT,
                                          /*dataType=*/DATA_TYPE,
                                          /*batch_size=*/OUTPUT_BATCH_SIZE,
                                          /*channels=*/OUTPUT_CHANNELS,
                                          /*height=*/OUTPUT_HEIGHT,
                                          /*width=*/OUTPUT_WIDTH));

    cudnnConvolutionBwdDataAlgo_t convolution_algorithm;
    convolution_algorithm = total_conv_algo[choose_algo];

    // std::cout << "Chosen algorithm: " << algo_name[convolution_algorithm]
    //   << std::endl;

    size_t workspace_bytes = 0;
    void *d_workspace = nullptr;
    double ans = 9999;
    for (choose_algo = 0; choose_algo < NUM_ALGOS; choose_algo += 8) {
        convolution_algorithm = total_conv_algo[choose_algo];
        cudnnStatus_t status;
        if ((status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                 cudnn, kernel_descriptor, input_descriptor,
                 convolution_descriptor, output_descriptor,
                 convolution_algorithm, &workspace_bytes)) !=
            CUDNN_STATUS_SUCCESS) {
            std::cout << choose_algo << " failed." << std::endl;
            /*printf("Conv2d_T aglo %d : mean time(ms): %.3lf ms, %.3lf TFLPOS
               as " "gemm, status=%d\n", choose_algo, 9999.0, 0.0,
               (int)status);*/
        } else {
            // std::cout << "Workspace size: " << (workspace_bytes) << "B"
            //   << std::endl;
            if (workspace_bytes > 0)
                checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));

            const float alpha = 1, beta = 0;
            cudaProfilerStart();
            double average_ms = time_cuda([&]() {
                cudnnConvolutionBackwardData(
                    cudnn, &alpha, kernel_descriptor, d_kernel,
                    input_descriptor, d_input, convolution_descriptor,
                    convolution_algorithm, d_workspace, workspace_bytes, &beta,
                    output_descriptor, d_output);
            });
            cudaProfilerStop();
            /*printf(
                "Conv2d_T aglo %d : mean time(ms) %.3lf ms, %.3lf TFLPOS as
               gemm\n", choose_algo, average_ms,
                ((n * c * h * w * f * r * s * 2) / 1e9) / average_ms);*/
            checkCUDA(cudaFree(d_workspace));
            ans = std::min(ans, average_ms);
            std::cout << "Algo: " << algo_name[choose_algo] << std::endl;
            std::cout << "Time: " << average_ms << " ms" << std::endl;
        }
    }
    std::cout << ans << std::endl;
}

void conv2d_transpose(const int n, const int c, const int h, const int w,
                      const int f, const int r, const int s,
                      const int stride = 1, const int padding = 0,
                      const int dilation = 1) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    dtype *d_input, *d_kernel, *d_output;
    size_t size_input = n * c * h * w, size_kernel = f * c * r * s,
           size_output = f * h * w;
    d_input = d_malloc<dtype>(size_input);
    d_kernel = d_malloc<dtype>(size_kernel);
    d_output = d_malloc<dtype>(size_output, false);

    conv2d_transposed_cudnn(cudnn, n, c, h, w, f, r, s, stride, padding,
                            dilation, d_input, d_kernel, d_output);

    checkCUDA(cudaFree(d_kernel));
    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_output));
    checkCUDNN(cudnnDestroy(cudnn));
}

int main(int argc, char *argv[]) {
    int n, c, h, w, f, r, s;
    int stride = 1, padding = 0, dilation = 1;
    if (argc < 11)
        cerr << argv[0] << " n c h w f r s stride padding dilation" << endl;
    n = atoi(argv[1]);
    c = atoi(argv[2]);
    h = atoi(argv[3]);
    w = atoi(argv[4]);
    f = atoi(argv[5]);
    r = atoi(argv[6]);
    s = atoi(argv[7]);
    if (argc > 8) {
        if (argc != 11)
            cerr << argv[0] << " n c h w f r s stride padding dilation" << endl;
        stride = atoi(argv[8]);
        padding = atoi(argv[9]);
        dilation = atoi(argv[10]);
    }
    // conv2gemm(n, c, h, w, f, r, s);
    conv2d_transpose(n, c, h, w, f, r, s, stride, padding, dilation);
}
