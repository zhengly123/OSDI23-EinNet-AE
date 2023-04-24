#include "cuda_fp16.h"
#include "cuda_profiler_api.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <iostream>
#include <sys/time.h>
#include <typeinfo>
#if !defined(USE_FP16) && !defined(USE_FP32)
#error "Specify USE_FP16 or USE_FP32"
#endif

#ifdef USE_FP32
typedef float d_type;
#else
typedef half d_type;
#endif

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if ((x) != cudaSuccess) {                                              \
            printf("Cuda error at %s:%d, %d\n", __FILE__, __LINE__,            \
                   EXIT_FAILURE);                                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CALL(x)                                                         \
    do {                                                                       \
        if ((x) != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("Cublas error at %s:%d, %d\n", __FILE__, __LINE__,          \
                   EXIT_FAILURE);                                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CURAND_CALL(x)                                                         \
    do {                                                                       \
        if ((x) != CURAND_STATUS_SUCCESS) {                                    \
            printf("Error at %s:%d, %d\n", __FILE__, __LINE__, EXIT_FAILURE);  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

curandGenerator_t gen;

template <typename T> class TestArg {
  public:
    int transa, transb;
    int b, m, n, k;
    T *A, *B, *C;
    int lda, ldb, ldc;

    float test(cublasHandle_t &cublas, int algo) {
        cudaEvent_t st, ed;
        cudaDataType_t data_type =
            (typeid(float) == typeid(T)) ? CUDA_R_32F : CUDA_R_16F;
        CUDA_CALL(cudaEventCreate(&st));
        CUDA_CALL(cudaEventCreate(&ed));
        float duration = 0.0;
        int warmup = 5, rounds = 5;
        const float alpha = 1.0, beta = 0.0;
        cublasStatus_t status = cublasGemmStridedBatchedEx(
            cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, m, n,
            k, &alpha, A, data_type, lda, m * k, B, data_type, ldb, k * n,
            &beta, C, data_type, ldc, m * n, b, data_type,
            (cublasGemmAlgo_t)algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Algo: %d failed\n", algo);
            return 10000;
        }
        for (int i = 0; i < warmup; ++i) {
            cublasGemmStridedBatchedEx(
                cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, m,
                n, k, &alpha, A, data_type, lda, m * k, B, data_type, ldb,
                k * n, &beta, C, data_type, ldc, m * n, b, data_type,
                (cublasGemmAlgo_t)algo);
        }
        cudaProfilerStart();
        for (int i = 0; i < rounds; ++i) {
            float durtime;
            CUDA_CALL(cudaEventRecord(st, 0));
            cublasGemmStridedBatchedEx(
                cublas, (cublasOperation_t)transa, (cublasOperation_t)transb, m,
                n, k, &alpha, A, data_type, lda, m * k, B, data_type, ldb,
                k * n, &beta, C, data_type, ldc, m * n, b, data_type,
                (cublasGemmAlgo_t)algo);
            CUDA_CALL(cudaEventRecord(ed, 0));
            CUDA_CALL(cudaEventSynchronize(st));
            CUDA_CALL(cudaEventSynchronize(ed));
            CUDA_CALL(cudaEventElapsedTime(&durtime, st, ed));
            duration += durtime;
        }
        cudaProfilerStop();
        std::cout << "Algo: " << algo << " Times(ms): " << duration / rounds
                  << std::endl;
        return duration / rounds;
    }
};

template <typename T> void initTestArgs(TestArg<T> *);

int b, m, n, k;
d_type *d_A, *d_B, *d_C;

// size为转换前float数据个数，转换后由size/2个half2存储所有数据
__global__ void float2HalfVec(half *dst, float *src, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < size; i += stride)
        dst[i] = __float2half_rn(src[i]);
}

void rand_fp16(half *d_A, int size) {
    float *d_tempA;
    CUDA_CALL(cudaMalloc((void **)&d_tempA, size * sizeof(float)));
    CURAND_CALL(curandGenerateUniform(gen, d_tempA, size));
    float2HalfVec<<<256, 256>>>(d_A, d_tempA, size);
    CUDA_CALL(cudaDeviceSynchronize());
}

int main(int argc, char *argv[]) {
    // scanf("%d%d%d%d", &b, &m, &n, &k);
    int transA = -1, transB = -1, transC = -1;
    if (argc < 5) {
        std::cout
            << argv[0]
            << " b m n k [Transpose_A=-1] [Transpose_B=-1] [Transpose_C=-1]"
            << std::endl;
        return 1;
    }
    b = atoi(argv[1]);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    k = atoi(argv[4]);
    if (argc >= 6)
        transA = atoi(argv[5]);
    if (argc >= 7)
        transB = atoi(argv[6]);
    if (argc >= 8)
        transC = atoi(argv[7]);
    assert(-1 <= transA && transA <= 1);
    assert(-1 <= transB && transB <= 1);
    assert(-1 <= transC && transC <= 1);

    CUDA_CALL(cudaMalloc((void **)&d_A, b * m * k * sizeof(d_type)));
    CUDA_CALL(cudaMalloc((void **)&d_B, b * k * n * sizeof(d_type)));
    CUDA_CALL(cudaMalloc((void **)&d_C, b * m * n * sizeof(d_type)));

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(
        curandSetPseudoRandomGeneratorSeed(gen, (unsigned long long)clock()));

#ifdef USE_FP32
    CURAND_CALL(curandGenerateUniform(gen, d_A, b * m * k));
    CURAND_CALL(curandGenerateUniform(gen, d_B, b * k * n));
#else
    rand_fp16(d_A, b * m * k);
    rand_fp16(d_B, b * k * n);
#endif

    cublasHandle_t cublas;
    CUBLAS_CALL(cublasCreate(&cublas));

    TestArg<d_type> Test[8];
    initTestArgs<d_type>(Test);

    int bestalgo = -2, bestmode = -1;
    double besttime = 10000;
    for (int i = 0; i < 8; i+=8) {
        // Row-major/Column-major policy
        if (transA >= 0 && transA != Test[i].transa)
            continue;
        if (transB >= 0 && transB != Test[i].transb)
            continue;
        // Transpose policy
        if ((transC == 0 && (i >= 4)) || (transC == 1 && (i < 4)))
            continue;
        for (int j = 0; j < 1; ++j) {
            printf("Num: %d\t", i);
            auto t = Test[i].test(cublas, j);
            if (t < besttime) {
                bestalgo = j;
                bestmode = i;
                besttime = t;
            }
        }
        if (typeid(half) == typeid(d_type)) {
            for (int j = 99; j < 116; ++j) {
                printf("Num: %d\t", i);
                auto t = Test[i].test(cublas, j);
                if (t < besttime) {
                    bestalgo = j;
                    bestmode = i;
                    besttime = t;
                }
            }
        }
    }
    std::cout << "best algo: " << bestalgo << ", best mode: " << bestmode
              << ", best time: " << besttime << std::endl;

    // Finalize
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
    CUBLAS_CALL(cublasDestroy(cublas));
    CURAND_CALL(curandDestroyGenerator(gen));

    return 0;
}

template <typename T> void initTestArgs(TestArg<T> *Test) {
    // d_C = d_A x d_B
    Test[0].transa = 0;
    Test[0].transb = 0;
    Test[0].b = b;
    Test[0].A = d_A;
    Test[0].lda = m;
    Test[0].B = d_B;
    Test[0].ldb = k;
    Test[0].C = d_C;
    Test[0].ldc = m;
    Test[0].m = m;
    Test[0].n = n;
    Test[0].k = k;

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

    // Test[2].transa = 0;
    // Test[2].transb = 1;
    // Test[2].b = b;
    // Test[2].A = d_A;
    // Test[2].lda = m;
    // Test[2].B = d_B;
    // Test[2].ldb = n;
    // Test[2].C = d_C;
    // Test[2].ldc = m;
    // Test[2].m = m;
    // Test[2].n = n;
    // Test[2].k = k;

    // Test[3].transa = 1;
    // Test[3].transb = 1;
    // Test[3].b = b;
    // Test[3].A = d_A;
    // Test[3].lda = k;
    // Test[3].B = d_B;
    // Test[3].ldb = n;
    // Test[3].C = d_C;
    // Test[3].ldc = m;
    // Test[3].m = m;
    // Test[3].n = n;
    // Test[3].k = k;

    // // trans(d_C) = trans(d_B) x trans(d_A)
    // Test[4].transa = 1;
    // Test[4].transb = 1;
    // Test[4].b = b;
    // Test[4].A = d_B;
    // Test[4].lda = k;
    // Test[4].B = d_A;
    // Test[4].ldb = m;
    // Test[4].C = d_C;
    // Test[4].ldc = n;
    // Test[4].m = n;
    // Test[4].n = m;
    // Test[4].k = k;

    // Test[5].transa = 0;
    // Test[5].transb = 1;
    // Test[5].b = b;
    // Test[5].A = d_B;
    // Test[5].lda = n;
    // Test[5].B = d_A;
    // Test[5].ldb = m;
    // Test[5].C = d_C;
    // Test[5].ldc = n;
    // Test[5].m = n;
    // Test[5].n = m;
    // Test[5].k = k;

    // Test[6].transa = 1;
    // Test[6].transb = 0;
    // Test[6].b = b;
    // Test[6].A = d_B;
    // Test[6].lda = k;
    // Test[6].B = d_A;
    // Test[6].ldb = k;
    // Test[6].C = d_C;
    // Test[6].ldc = n;
    // Test[6].m = n;
    // Test[6].n = m;
    // Test[6].k = k;

    // Test[7].transa = 0;
    // Test[7].transb = 0;
    // Test[7].b = b;
    // Test[7].A = d_B;
    // Test[7].lda = n;
    // Test[7].B = d_A;
    // Test[7].ldb = k;
    // Test[7].C = d_C;
    // Test[7].ldc = n;
    // Test[7].m = n;
    // Test[7].n = m;
    // Test[7].k = k;
}
