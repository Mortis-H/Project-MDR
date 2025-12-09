// main.cpp
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#define HIP_CHECK(cmd)                                            \
    do {                                                          \
        hipError_t e = (cmd);                                     \
        if (e != hipSuccess) {                                    \
            std::cerr << "HIP error: " << hipGetErrorString(e)    \
                      << " at " << __FILE__ << ":" << __LINE__    \
                      << std::endl;                               \
            std::exit(1);                                         \
        }                                                         \
    } while (0)

int main() {
    const int N = 1;
    const size_t bytes = N * sizeof(float);

    // 準備 host 資料
    std::vector<float> hA(N), hB(N), hC(N);
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i) + 100;
        hB[i] = static_cast<float>(2 * i) + 100;
    }

    // 初始化 HIP、選裝置（可省略 hipInit，呼叫任何 HIP API 時會自動 init）
    HIP_CHECK(hipSetDevice(0));

    // 配置 device 記憶體
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));

    // host -> device
    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));

    // 載入 code object 模組
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, "vec_add_kernel.hsaco"));

    // 取得 kernel function handle
    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, module, "vec_add"));

    // 設定 launch 參數
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dC,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(
        func,
        gridSize, 1, 1,     // gridDim
        blockSize, 1, 1,    // blockDim
        0,                  // sharedMemBytes
        nullptr,            // stream
        kernelArgs,         // kernelParams
        nullptr             // extra
    ));

    HIP_CHECK(hipDeviceSynchronize());

    // device -> host
    HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

    // 驗證結果
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];
        if (std::fabs(hC[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at " << i
                      << ": got " << hC[i]
                      << ", expected " << expected << std::endl;
            ok = false;
            break;
        }
    }

    std::cout << "Result: " << (ok ? "OK" : "FAILED") << std::endl;
    std::cout << "C printed at host = " << std::fixed << std::setprecision(3) << hC[0] << std::endl;

    // 清理
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipModuleUnload(module));

    return ok ? 0 : 1;
}

