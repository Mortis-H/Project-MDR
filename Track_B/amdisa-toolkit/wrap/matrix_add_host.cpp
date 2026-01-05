// Host code for MatrixAddGlobalMem kernel
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define HIP_CHECK(cmd)                                                         \
    do {                                                                       \
        hipError_t error = (cmd);                                              \
        if (error != hipSuccess) {                                             \
            std::cerr << "HIP error: " << hipGetErrorString(error)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

int main() {
    // 矩陣參數
    const int Width = 32;
    const int Height = 32;
    const int offset = 10;  // 用於測試 offset 參數
    const int N = Width * Height;
    
    std::cout << "Matrix Add with Offset Test" << std::endl;
    std::cout << "Width: " << Width << ", Height: " << Height << std::endl;
    std::cout << "Offset: " << offset << std::endl;
    std::cout << "Total elements: " << N << std::endl;
    
    // 分配 host 記憶體
    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_S(N);
    std::vector<float> h_S_ref(N);  // Reference result
    
    // 初始化資料
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }
    
    // 計算 reference（在 CPU 上）
    for (int i = 0; i < N; i++) {
        int k = i + offset;
        if (k < N) {
            h_S_ref[i] = h_A[k] + h_B[k];
        } else {
            h_S_ref[i] = 0.0f;  // 超出範圍
        }
    }
    
    // 分配 device 記憶體
    float *d_A, *d_B, *d_S;
    HIP_CHECK(hipMalloc(&d_A, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_S, N * sizeof(float)));
    
    // 複製資料到 device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_S, 0, N * sizeof(float)));
    
    // 載入 kernel module
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, "vec_add_kernel.hsaco"));
    
    // 取得 kernel function
    hipFunction_t func;
    HIP_CHECK(hipModuleGetFunction(&func, module, "vec_add"));
    
    // 設定 kernel 參數
    struct {
        float* A;
        float* B;
        float* S;
        int Width;
        int Height;
        int offset;
    } args;
    
    args.A = d_A;
    args.B = d_B;
    args.S = d_S;
    args.Width = Width;
    args.Height = Height;
    args.offset = offset;
    
    size_t args_size = sizeof(args);
    void* config[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
        HIP_LAUNCH_PARAM_END
    };
    
    // 設定 grid 和 block 維度（2D launch）
    dim3 blockDim(16, 16, 1);  // 16x16 threads per block
    dim3 gridDim((Width + blockDim.x - 1) / blockDim.x,
                 (Height + blockDim.y - 1) / blockDim.y,
                 1);
    
    std::cout << "\nLaunching kernel:" << std::endl;
    std::cout << "  Grid:  (" << gridDim.x << ", " << gridDim.y << ", " << gridDim.z << ")" << std::endl;
    std::cout << "  Block: (" << blockDim.x << ", " << blockDim.y << ", " << blockDim.z << ")" << std::endl;
    
    // 啟動 kernel
    HIP_CHECK(hipModuleLaunchKernel(
        func,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0,  // shared memory
        nullptr,  // stream
        nullptr,  // kernel args (使用 config)
        config
    ));
    
    // 等待完成
    HIP_CHECK(hipDeviceSynchronize());
    
    // 複製結果回 host
    HIP_CHECK(hipMemcpy(h_S.data(), d_S, N * sizeof(float), hipMemcpyDeviceToHost));
    
    // 驗證結果
    bool passed = true;
    int errors = 0;
    const int max_errors_to_print = 10;
    
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_S[i] - h_S_ref[i]) > 1e-5) {
            if (errors < max_errors_to_print) {
                std::cout << "Error at index " << i 
                          << ": GPU=" << h_S[i] 
                          << ", CPU=" << h_S_ref[i] << std::endl;
            }
            errors++;
            passed = false;
        }
    }
    
    if (passed) {
        std::cout << "\n✓ Test PASSED! All results match." << std::endl;
    } else {
        std::cout << "\n✗ Test FAILED! Found " << errors << " errors." << std::endl;
    }
    
    // 印出前幾個結果
    std::cout << "\nFirst 10 results:" << std::endl;
    for (int i = 0; i < std::min(10, N); i++) {
        int k = i + offset;
        std::cout << "  S[" << i << "] = A[" << k << "] + B[" << k << "] = "
                  << (k < N ? h_A[k] : 0.0f) << " + " << (k < N ? h_B[k] : 0.0f)
                  << " = " << h_S[i] 
                  << " (expected: " << h_S_ref[i] << ")" << std::endl;
    }
    
    // 清理
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_S));
    HIP_CHECK(hipModuleUnload(module));
    
    return passed ? 0 : 1;
}

