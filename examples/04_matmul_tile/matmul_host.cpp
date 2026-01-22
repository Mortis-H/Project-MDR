// Host launcher for matmul_tile kernel.
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

extern "C" __global__ void matmul_tile(const float *A,
                                       const float *B,
                                       float *C,
                                       int M, int N, int K);

int main(int argc, char **argv) {
    int M = 128;
    int N = 128;
    int K = 128;
    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    const size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    const size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    const size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);
    std::vector<float> hC(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) {
        hA[i] = static_cast<float>(1.0f);
    }
    for (int i = 0; i < K * N; ++i) {
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
              1);

    hipLaunchKernelGGL(matmul_tile, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hC.data(), dC, bytesC, hipMemcpyDeviceToHost));

    std::cout << "C[0] = " << hC[0] << "\n";
    std::cout << "C[last] = " << hC.back() << "\n";

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return 0;
}
