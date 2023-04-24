#include "cuda_fp16.h"
// #include "dbg.h"
#include <cassert>
#include <chrono>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <curand.h>
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

// const int warmup = 200;
const int repeat = 200;

// size为转换前float数据个数，转换后由size/2个half2存储所有数据
__global__ void float2HalfVec(half *dst, float *src, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride)
        dst[i] = __float2half_rn(src[i]);
}

__global__ void d_fake_rand(float *d_a, int size) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < size; i += stride) {
        // d_a[i] = i % 2;
        d_a[i] = 1;
        // d_a[i] = 0;
    }
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
    d_fake_rand<<<256, 256>>>(d_a, size);
    // checkCURAND(curandGenerateUniform(gen, d_a, size));
}

// template <> void d_rand<float>(float *d_a, int size) {
//     float *d_temp;
//     checkCUDA(cudaMalloc(&d_temp, size * sizeof(float)));
//     d_rand<float>(d_temp, size);
//     d_fake_rand<<<256, 256>>>(d_temp, size);
//     // float2HalfVec<<<256, 256>>>(d_a, d_temp, size);
//     checkCUDA(cudaDeviceSynchronize());
// }

template <typename T> T *d_malloc(size_t size, bool random = false) {
    T *ret;
    checkCUDA(cudaMalloc(&ret, size * sizeof(T)));
    if (random) {
        d_rand<T>(ret, size);
    }
    checkCUDA(cudaDeviceSynchronize());
    return ret;
}

__global__ void d_compare(float *x, float *y, int n, bool verbose = false) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    if (start > 0)
        return;
    const float eps = 5e-6;
    int n_err = 0;
    for (int i = 0; i < n; ++i) {
        float delta = fabs(x[i] - y[i]);
        if ((delta / max(fabs(x[i]), fabs(y[i])) > eps)) {
            // && (fabs(delta / x[i]) > eps || __habs(delta / y[i]) > eps)) {
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
                    //     half2 t = x2[idx((r + 1) * 3 + (s + 1), i + r * w +
                    //     s)]; float a =
                    //         __half2float(__low2half(t));
                    //     float b =
                    //         __half2float(__high2half(t));
                    //     printf("i=%d (%d, %d) %d %d %f %f\n", i, hh, ww, r,
                    //     s, a, b);
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

__global__ void reduce_4x4(dtype *in, dtype *out, const int N, const int F,
                           const int H, const int W, const int IH,
                           const int IW) {
#define in_index(n, h, w, r, s, f)                                             \
    ((((((n)*IH + h) * IW + w) * R + r) * S + s) * F + f)
#define out_index(n, h, w, f) (((((n)*H) + (h)) * W + (w)) * F + (f))
    const int R = 4, S = 4;
    const int n_tasks = N * F * H * W;
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < n_tasks; i += stride) {
        int t = i, n, f, h, w;
        f = t % F;
        t /= F;
        w = t % W;
        t /= W;
        h = t % H;
        t /= H;
        n = t;

        // unroll this 2-iter loop
        float sum = 0;
        int x, y;
        for (int r = (h + 1) & 1; r < R; r += 2) {
            x = (h + 1 - r) / 2;
            if (x >= 0 && x < IH) {
                for (int s = (w + 1) & 1; s < S; s += 2) {
                    y = (w + 1 - s) / 2;
                    if (y >= 0 && y < IW) {
                        sum += in[in_index(n, x, y, r, s, f)];
                        // if (i==0)
                        //     printf("TTT nhwf= %d,%d,%d,%d x=%d y=%d, v=%f,
                        //     index=%d, rsf %d %d %d\n", n, h, w,
                        //            f, x, y, in[in_index(n, x, y, r, s, f)],
                        //            in_index(n, x, y, r, s, f), r,s,f);
                    }
                }
            }
        }
        out[out_index(n, h, w, f)] = sum;
    }
#undef in_index
#undef out_index
}

__global__ void half2float(const int n, const half *x, float *y) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < n; i += stride) {
        y[i] = __half2float(x[i]);
    }
}

// // [m,n] -> [n,m]
// __global__ void transpose_rsfc2fcrs(dtype *x, dtype *y, int m, int n) {
//     int start = threadIdx.x + blockDim.x * blockIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = start; i < m * n; i += stride) {
//         int mm = i / n, nn = i % n;
//         y[nn * m + mm] = x[i];
//     }
// }

// __global__ void transpose_rsfc2crsf(dtype *x, dtype *y, const int F, const
// int C, const int RS) {
//     int start = threadIdx.x + blockDim.x * blockIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = start; i < F * C * RS; i += stride) {
//         int rs = i / C / F;
//         int f = i / C % F;
//         int c = i % C;
//         y[(c * RS + rs) * F + f] = x[(rs * F + f) * C + c];
//     }
// }

// __global__ void transpose_rsfc2crsf(dtype *x, dtype *y, const int F, const
// int C, const int RS) {
//     int start = threadIdx.x + blockDim.x * blockIdx.x;
//     int stride = blockDim.x * gridDim.x;
//     for (int i = start; i < F * C * RS; i += stride) {
//         int rs = i / C / F;
//         int f = i / C % F;
//         int c = i % C;
//         y[(c * RS + rs) * F + f] = x[(rs * F + f) * C + c];
//     }
// }

__global__ void transpose_rsfc2cfrs(dtype *x, dtype *y, const int F,
                                    const int C, const int RS) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < F * C * RS; i += stride) {
        int rs = i / C / F;
        int f = i / C % F;
        int c = i % C;
        y[(c * F + f) * RS + rs] = x[(rs * F + f) * C + c];
    }
}

__global__ void transpose_nhwc2nchw(dtype *x, dtype *y, int N, int C, int HW) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start; i < N * C * HW; i += stride) {
        int n = i / C / HW;
        int hw = i / C % HW;
        int c = i % C;
        // if (i<10)
        //     printf("i=%d, n c hw %d %d %d\n", i, n, c, hw);
        y[(n * C + c) * HW + hw] = x[(n * HW + hw) * C + c];
    }
}

__global__ void print_float(float *x, int len) {
    int start = threadIdx.x + blockDim.x * blockIdx.x;
    if (start == 0) {
        for (int i = 0; i < len; ++i) {
            printf("%.3f ", x[i]);
        }
        printf("\n");
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

void conv_transposed2gemm(const int n, const int c, const int h, const int w,
                          const int f, const int r, const int s,
                          const int stride, const int pad, const int dilation) {
    cublasHandle_t cublas;
    checkCUBLAS(cublasCreate(&cublas));
    const int B = 1, M = r * s * f, N = n * h * w, K = c;
    dtype *d_input, *d_kernel, *d_temp, *d_output;

    int oh = (h - 1) * stride - 2 * pad + dilation * (r - 1) + 1;
    int ow = (w - 1) * stride - 2 * pad + dilation * (s - 1) + 1;
    assert(oh == ow);
    size_t size_input = n * c * h * w, size_kernel = f * c * r * s,
           size_temp = r * s * f * n * h * w, size_output = n * f * oh * ow;
    // dbg(n, c, h, w, f, r, s, stride, pad, dilation, size_temp, size_output);
    // dbg("Bgemm args", B, M, N, K);
    d_input = d_malloc<dtype>(size_input);
    d_kernel = d_malloc<dtype>(size_kernel);
    d_temp = d_malloc<dtype>(size_temp, false);
    d_output = d_malloc<dtype>(size_output, false);

    int algo = 0;
    int transa = 1, transb = 0;
    int ld_input = K, ld_kernel = K, ld_temp = M;
    float alpha = 1, beta = 0;

    cudaDataType_t data_type =
        (typeid(float) == typeid(dtype)) ? CUDA_R_32F : CUDA_R_16F;

    // // Retrieve output in float
    dtype *d_ans = d_malloc<dtype>(size_output, false);
    dtype *d_input_nchw = d_malloc<dtype>(size_input, false);
    dtype *d_kernel_cfrs = d_malloc<dtype>(size_kernel, false);
    dtype *d_ans_nhwc = d_malloc<dtype>(size_output, false);

    cudaProfilerStart();
    checkCUDA(cudaDeviceSynchronize());
    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; ++i) {
        cublasGemmStridedBatchedEx(
            cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, M, N,
            K, &alpha, d_kernel, data_type, ld_kernel, M * K, d_input,
            data_type, ld_input, 0, &beta, d_temp, data_type, ld_temp, M * N, B,
            data_type, (cublasGemmAlgo_t)algo);
        // clang-format on

        reduce_4x4<<<(M * N + 127) / 128, 128>>>(d_temp, d_output, n, f, oh, ow,
                                                 h, w);
    }
    checkCUDA(cudaDeviceSynchronize());
    cudaProfilerStop();
    auto tend = std::chrono::high_resolution_clock::now();
    double average_ms =
        std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin)
            .count() *
        1000 / repeat;
    // printf("Gemm+reduce mean time(ms): %lf\n", average_ms);
    printf("Time: %lf ms\n", average_ms);
    // dbg(average_ms);
    // __global__ void transpose_rsfc2cfrs(dtype *x, dtype *y, const int F,
    // const int C, const int RS) {
    // __global__ void transpose_nhwc2nchw(dtype *d_input, dtype *d_input_nchw,
    // int N, int C, int HW) {
    // print_float<<<1,1>>>(d_ans_nhwc, 10);
    // print_float<<<1,1>>>(d_ans, 10);

    checkCUDA(cudaFree(d_temp));
    checkCUDA(cudaFree(d_kernel));
    checkCUDA(cudaFree(d_input));
    checkCUDA(cudaFree(d_output));
    checkCUDA(cudaFree(d_ans));
    checkCUBLAS(cublasDestroy(cublas));
}

int main(int argc, char *argv[]) {
    int n, c, h, w, f, r, s;
    n = 1, c = 448, h = 2, w = 2, f = 256, r = 4, s = 4;
    int stride, pad, dilation;
    stride = 2, pad = 1, dilation = 1;
    // int params[][13] = [
    //     [ 1, 228, 1, 1, 448, 2, 2, 0, 0, 1, 1, 1, 1 ],
    //     [ 1, 448, 2, 2, 256, 4, 4, 1, 1, 2, 2, 1, 1 ],
    //     [ 1, 256, 4, 4, 128, 4, 4, 1, 1, 2, 2, 1, 1 ],
    //     [ 1, 128, 8, 8, 64, 4, 4, 1, 1, 2, 2, 1, 1 ],
    //     [ 1, 64, 16, 16, 3, 4, 4, 1, 1, 2, 2, 1, 1 ]
    // ];

    if (argc != 11) {
        cerr << argv[0] << " n c h w f r s stride pad dilation" << endl;
        exit(0);
    }
    n = atoi(argv[1]);
    c = atoi(argv[2]);
    h = atoi(argv[3]);
    w = atoi(argv[4]);
    f = atoi(argv[5]);
    r = atoi(argv[6]);
    s = atoi(argv[7]);
    stride = atoi(argv[8]);
    pad = atoi(argv[9]);
    dilation = atoi(argv[10]);

    conv_transposed2gemm(n, c, h, w, f, r, s, stride, pad, dilation);
}
