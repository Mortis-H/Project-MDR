// Host launcher using hipModuleLoad (HSACO) for batched_gemm kernel.
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
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

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <kernel.hsaco> [M N K batch]\n";
        return 1;
    }

    const std::string hsaco_path = argv[1];
    int M = 128;
    int N = 128;
    int K = 128;
    int batch = 4;
    if (argc == 6) {
        M = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
        batch = std::atoi(argv[5]);
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

    std::vector<float> hA(bytesA / sizeof(float), 1.0f);
    std::vector<float> hB(bytesB / sizeof(float), 1.0f);
    std::vector<float> hC(bytesC / sizeof(float), 0.0f);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dB, bytesB));
    HIP_CHECK(hipMalloc(&dC, bytesC));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, bytesC));

    hipModule_t module = nullptr;
    hipFunction_t kernel = nullptr;
    HIP_CHECK(hipModuleLoad(&module, hsaco_path.c_str()));
    HIP_CHECK(hipModuleGetFunction(&kernel, module, "batched_gemm"));

    dim3 block(16, 16, 1);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y,
              batch);

    int lda_arg = lda;
    int ldb_arg = ldb;
    int ldc_arg = ldc;
    int64_t strideA_arg = strideA;
    int64_t strideB_arg = strideB;
    int64_t strideC_arg = strideC;
    int batch_arg = batch;

    void *kernel_params[] = {
        &dA, &dB, &dC,
        &M, &N, &K,
        &lda_arg, &ldb_arg, &ldc_arg,
        &strideA_arg, &strideB_arg, &strideC_arg,
        &batch_arg
    };

    HIP_CHECK(hipModuleLaunchKernel(kernel,
                                    grid.x, grid.y, grid.z,
                                    block.x, block.y, block.z,
                                    0, 0, kernel_params, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(hC.data(), dC, bytesC, hipMemcpyDeviceToHost));

    std::cout << "batch=0 C[0] = " << hC[0] << "\n";
    const size_t last_idx = static_cast<size_t>((batch - 1) * strideC + (M - 1) * ldc + (N - 1));
    std::cout << "batch=last C[last] = " << hC[last_idx] << "\n";

    HIP_CHECK(hipModuleUnload(module));
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return 0;
}
