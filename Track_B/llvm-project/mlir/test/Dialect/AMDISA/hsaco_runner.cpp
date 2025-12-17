// hsaco_runner.cpp
// 通用的 .hsaco 執行器，用於驗證 GPU kernel
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstring>

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

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <hsaco_path> <kernel_name> <test_size>\n";
    std::cerr << "Example: " << prog << " kernel.hsaco vectorAdd 1024\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* hsaco_path = argv[1];
    const char* kernel_name = argv[2];
    const int N = std::atoi(argv[3]);

    std::cout << "========================================\n";
    std::cout << "HSACO Runner\n";
    std::cout << "========================================\n";
    std::cout << "HSACO:  " << hsaco_path << "\n";
    std::cout << "Kernel: " << kernel_name << "\n";
    std::cout << "Size:   " << N << "\n";
    std::cout << "========================================\n";

    const size_t bytes = N * sizeof(float);

    // 準備 host 資料
    std::vector<float> hA(N), hB(N), hC(N);
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i * 2);
        hC[i] = 0.0f;
    }

    // 初始化 HIP
    HIP_CHECK(hipSetDevice(0));

    // 配置 device 記憶體
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));

    // host -> device
    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), bytes, hipMemcpyHostToDevice));

    // 載入 code object 模組
    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, hsaco_path);
    if (err != hipSuccess) {
        std::cerr << "Failed to load module: " << hipGetErrorString(err) << "\n";
        std::cerr << "HSACO path: " << hsaco_path << "\n";
        return 1;
    }
    std::cout << "✓ Module loaded successfully\n";

    // 取得 kernel function handle
    hipFunction_t func;
    err = hipModuleGetFunction(&func, module, kernel_name);
    if (err != hipSuccess) {
        std::cerr << "Failed to get function '" << kernel_name << "': " 
                  << hipGetErrorString(err) << "\n";
        HIP_CHECK(hipModuleUnload(module));
        return 1;
    }
    std::cout << "✓ Function found: " << kernel_name << "\n";

    // 設定 launch 參數
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dC,
        (void*)&N
    };

    // 執行 kernel
    HIP_CHECK(hipModuleLaunchKernel(
        func,
        gridSize, 1, 1,     // gridDim
        blockSize, 1, 1,    // blockDim
        0,                  // sharedMemBytes
        nullptr,            // stream
        kernelArgs,         // kernelParams
        nullptr             // extra
    ));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    // device -> host
    HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

    // 驗證結果
    bool ok = true;
    int error_count = 0;
    const int max_errors_to_show = 5;
    
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];
        if (std::fabs(hC[i] - expected) > 1e-5f) {
            if (error_count < max_errors_to_show) {
                std::cerr << "Mismatch at [" << i << "]: "
                          << "got " << hC[i] << ", "
                          << "expected " << expected << "\n";
            }
            error_count++;
            ok = false;
        }
    }

    std::cout << "========================================\n";
    if (ok) {
        std::cout << "✅ PASS: All " << N << " elements correct\n";
        // 顯示前幾個結果
        std::cout << "Sample results:\n";
        int samples = std::min(5, N);
        for (int i = 0; i < samples; i++) {
            std::cout << "  [" << i << "] " << hA[i] << " + " << hB[i] 
                      << " = " << hC[i] << "\n";
        }
    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }
    std::cout << "========================================\n";

    // 清理
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipModuleUnload(module));

    return ok ? 0 : 1;
}

