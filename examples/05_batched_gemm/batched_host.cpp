// Host launcher for batched_gemm kernel.
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _e = (call);                                             \
        if (_e != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(_e)             \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";     \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

extern "C" __global__ void batched_gemm(const float *A,
                                        const float *B,
                                        float *C,
                                        int M, int N, int K,
                                        int lda, int ldb, int ldc,
                                        int64_t strideA, int64_t strideB, int64_t strideC,
                                        int batchCount);

int main(int argc, char **argv) {
    int M = 128;
    int N = 128;
    int K = 128;
    int batch = 4;
    if (argc == 5) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
        batch = std::atoi(argv[4]);
    }

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    const int64_t strideA = static_cast<int64_t>(M) * lda;
    const int64_t strideB = static_cast<int64_t>(K) * ldb;
    const int64_t strideC = static_cast<int64_t>(M) * ldc;

    const size_t bytesA = static_cast<size_t>(strideA) * batch * sizeof(float);
    const size_t bytesB = static_cast<size_t>(strideB) * batch * sizeof(float);
    const size_t bytesC = static_cast<size_t>(strideC) * batch * sizeof(float);

    std::vector<float> hA(bytesA / sizeof(float));
    std::vector<float> hB(bytesB / sizeof(float));
    std::vector<float> hC(bytesC / sizeof(float), 0.0f);

    for (size_t i = 0; i < hA.size(); ++i) {
        hA[i] = static_cast<float>(1.0f);
    }
    for (size_t i = 0; i < hB.size(); ++i) {
        hB[i] = static_cast<float>(1.0f);
    }

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dB, bytesB));
    HIP_CHECK(hipMalloc(&dC, bytesC));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, bytesC));

    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              batch);

    hipLaunchKernelGGL(batched_gemm, grid, block, 0, 0,
                       dA, dB, dC,
                       M, N, K,
                       lda, ldb, ldc,
                       strideA, strideB, strideC,
                       batch);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hC.data(), dC, bytesC, hipMemcpyDeviceToHost));

    std::cout << "batch=0 C[0] = " << hC[0] << "\n";
    const size_t last_idx = static_cast<size_t>((batch - 1) * strideC + (M - 1) * ldc + (N - 1));
    std::cout << "batch=last C[last] = " << hC[last_idx] << "\n";

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return 0;
}
