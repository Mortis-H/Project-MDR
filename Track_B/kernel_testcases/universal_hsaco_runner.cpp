// universal_hsaco_runner.cpp
// ============================================================================
// 通用 HSACO 執行器 - 用於驗證從組裝檔案生成的 GPU kernel
// ============================================================================
//
// 功能說明：
//   這個工具載入 HSACO (HSA Code Object) 檔案，並在 AMD GPU 上執行 kernel。
//   主要用於驗證從 .s 組裝檔案經過 llvm-mc 和 ld.lld 生成的 .hsaco 是否正確。
//
// 使用方式：
//   ./universal_hsaco_runner <hsaco_path> <kernel_name> <kernel_type> <test_size>
//
// 範例：
//   ./universal_hsaco_runner kernel.hsaco _Z9vectorAddPKfS0_Pfi float_add 1024
//
// 支援的 Kernel 類型：
//   - float_add   : Float 向量加法 (測試基本的浮點運算)
//   - float_mul   : Float 向量乘法 (測試浮點乘法運算)
//   - float_dot   : Float 向量點積 (測試 shared memory 和 reduction)
//   - float_saxpy : Float SAXPY 運算 (測試 scalar-vector 運算)
//   - float_cond  : Float 條件運算 (測試條件分支)
//   - int_scalar  : Int 純量運算 (測試整數運算)
//   - int_mem     : Int 記憶體操作 (測試記憶體存取)
//   - int_cond    : Int 條件判斷 (測試分支控制)
//   - int_loop    : Int 迴圈 (測試迴圈結構)
//   - int_shared  : Int 共享記憶體 (測試 LDS/Shared Memory)
//
// ============================================================================

#include <hip/hip_runtime.h>  // HIP runtime API
#include <iostream>           // 標準輸入輸出
#include <iomanip>            // 格式化輸出
#include <vector>             // 動態陣列
#include <cmath>              // 數學函數 (fabs)
#include <cstring>            // 字串操作 (strcmp)
#include <cstdint>            // uint64_t

// ============================================================================
// HIP_CHECK 巨集 - 檢查 HIP API 呼叫是否成功
// ============================================================================
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

// ============================================================================
// KernelType 列舉 - 定義支援的 Kernel 類型
// ============================================================================
enum KernelType {
    FLOAT_VECTOR_ADD,    // Float 向量加法: (float* a, float* b, float* c, int n)
    FLOAT_VECTOR_MUL,    // Float 向量乘法: (float* a, float* b, float* c, int n)
    FLOAT_VECTOR_DOT,    // Float 向量點積: (float* a, float* b, float* partial_sums, int n)
    FLOAT_SAXPY,         // Float SAXPY: (float alpha, float* x, float* y, int n)
    FLOAT_CONDITIONAL,   // Float 條件運算: (float* input, float* output, float threshold, int n)
    INT_SCALAR_OPS,      // Int 純量運算: (int* output, int n)
    INT_MEMORY_OPS,      // Int 記憶體操作: (int* input, int* output, int n)
    INT_CONDITIONAL,     // Int 條件判斷: (int* input, int* output, int n)
    INT_LOOP,            // Int 迴圈: (int* output, int n)
    INT_SHARED_MEMORY    // Int 共享記憶體: (int* input, int* output, int n)
};

// ============================================================================
// print_usage - 顯示使用說明
// ============================================================================
void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <hsaco_path> <kernel_name> <kernel_type> <test_size>\n";
    std::cerr << "\nKernel types:\n";
    std::cerr << "  float_add   - float vector addition (a, b, c, n)\n";
    std::cerr << "  float_mul   - float vector multiplication (a, b, c, n)\n";
    std::cerr << "  float_dot   - float vector dot product (a, b, partial_sums, n)\n";
    std::cerr << "  float_saxpy - float SAXPY operation (alpha, x, y, n)\n";
    std::cerr << "  float_cond  - float conditional operations (input, output, threshold, n)\n";
    std::cerr << "  int_scalar  - int scalar operations (output, n)\n";
    std::cerr << "  int_mem     - int memory operations (input, output, n)\n";
    std::cerr << "  int_cond    - int conditional (input, output, n)\n";
    std::cerr << "  int_loop    - int loop (output, n)\n";
    std::cerr << "  int_shared  - int shared memory (input, output, n)\n";
    std::cerr << "\nExample: " << prog << " kernel.hsaco vectorAdd float_add 1024\n";
}

// ============================================================================
// parse_kernel_type - 解析 kernel 類型字串
// ============================================================================
KernelType parse_kernel_type(const char* type_str) {
    if (strcmp(type_str, "float_add") == 0) return FLOAT_VECTOR_ADD;
    if (strcmp(type_str, "float_mul") == 0) return FLOAT_VECTOR_MUL;
    if (strcmp(type_str, "float_dot") == 0) return FLOAT_VECTOR_DOT;
    if (strcmp(type_str, "float_saxpy") == 0) return FLOAT_SAXPY;
    if (strcmp(type_str, "float_cond") == 0) return FLOAT_CONDITIONAL;
    if (strcmp(type_str, "int_scalar") == 0) return INT_SCALAR_OPS;
    if (strcmp(type_str, "int_mem") == 0) return INT_MEMORY_OPS;
    if (strcmp(type_str, "int_cond") == 0) return INT_CONDITIONAL;
    if (strcmp(type_str, "int_loop") == 0) return INT_LOOP;
    if (strcmp(type_str, "int_shared") == 0) return INT_SHARED_MEMORY;
    return FLOAT_VECTOR_ADD; // 預設值
}

// ============================================================================
// run_float_vector_add - 執行 Float 向量加法 kernel
// ============================================================================
bool run_float_vector_add(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(float);
    
    std::vector<float> hA(N), hB(N), hC(N);
    
    // 初始化測試資料：A[i] = i, B[i] = 2*i
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i * 2);
        hC[i] = 0.0f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dC,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

    // 驗證結果
    bool ok = true;
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];
        if (std::fabs(hC[i] - expected) > 1e-5f) {
            if (error_count < 5) {
                std::cerr << "Mismatch at [" << i << "]: got " << hC[i] 
                          << ", expected " << expected << "\n";
            }
            error_count++;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "✅ PASS: All " << N << " elements correct\n";

    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return ok;
}

// ============================================================================
// run_float_vector_mul - 執行 Float 向量乘法 kernel
// ============================================================================
bool run_float_vector_mul(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(float);
    
    std::vector<float> hA(N), hB(N), hC(N);
    
    // 初始化測試資料：A[i] = i, B[i] = 2*i
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i * 2);
        hC[i] = 0.0f;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC, bytes));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dC,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

    // 驗證結果
    bool ok = true;
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] * hB[i];
        if (std::fabs(hC[i] - expected) > 1e-5f) {
            if (error_count < 5) {
                std::cerr << "Mismatch at [" << i << "]: got " << hC[i] 
                          << ", expected " << expected << "\n";
            }
            error_count++;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "✅ PASS: All " << N << " elements correct\n";
    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return ok;
}

// ============================================================================
// run_float_vector_dot - 執行 Float 向量點積 kernel
// ============================================================================
bool run_float_vector_dot(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(float);
    
    std::vector<float> hA(N), hB(N);
    
    // 初始化測試資料
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i * 2);
    }

    float *dA = nullptr, *dB = nullptr, *dPartial = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t partial_bytes = gridSize * sizeof(float);
    HIP_CHECK(hipMalloc(&dPartial, partial_bytes));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));

    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dPartial,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    // 讀取部分和
    std::vector<float> hPartial(gridSize);
    HIP_CHECK(hipMemcpy(hPartial.data(), dPartial, partial_bytes, hipMemcpyDeviceToHost));

    // 計算總和
    float total = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total += hPartial[i];
    }

    std::cout << "✅ PASS: Kernel executed successfully\n";
    std::cout << "Dot product result: " << total << "\n";
    std::cout << "Sample partial sums:\n";
    for (int i = 0; i < std::min(5, gridSize); i++) {
        std::cout << "  Block[" << i << "] = " << hPartial[i] << "\n";
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dPartial));

    return true;
}

// ============================================================================
// run_float_saxpy - 執行 Float SAXPY kernel
// ============================================================================
bool run_float_saxpy(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(float);
    
    std::vector<float> hX(N), hY(N), hY_expected(N);
    float alpha = 2.5f;
    
    // 初始化測試資料
    for (int i = 0; i < N; ++i) {
        hX[i] = static_cast<float>(i);
        hY[i] = static_cast<float>(i * 0.5);
        hY_expected[i] = alpha * hX[i] + hY[i];
    }

    float *dX = nullptr, *dY = nullptr;
    HIP_CHECK(hipMalloc(&dX, bytes));
    HIP_CHECK(hipMalloc(&dY, bytes));

    HIP_CHECK(hipMemcpy(dX, hX.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dY, hY.data(), bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&alpha,
        (void*)&dX,
        (void*)&dY,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    HIP_CHECK(hipMemcpy(hY.data(), dY, bytes, hipMemcpyDeviceToHost));

    // 驗證結果
    bool ok = true;
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(hY[i] - hY_expected[i]) > 1e-5f) {
            if (error_count < 5) {
                std::cerr << "Mismatch at [" << i << "]: got " << hY[i] 
                          << ", expected " << hY_expected[i] << "\n";
            }
            error_count++;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "✅ PASS: All " << N << " elements correct\n";
    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }

    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dY));

    return ok;
}

// ============================================================================
// run_float_conditional - 執行 Float 條件運算 kernel
// ============================================================================
bool run_float_conditional(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(float);
    
    std::vector<float> hIn(N), hOut(N);
    float threshold = 500.0f;
    
    // 初始化測試資料
    for (int i = 0; i < N; ++i) {
        hIn[i] = static_cast<float>(i);
        hOut[i] = 0.0f;
    }

    float *dIn = nullptr, *dOut = nullptr;
    HIP_CHECK(hipMalloc(&dIn, bytes));
    HIP_CHECK(hipMalloc(&dOut, bytes));

    HIP_CHECK(hipMemcpy(dIn, hIn.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dOut, hOut.data(), bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dIn,
        (void*)&dOut,
        (void*)&threshold,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    HIP_CHECK(hipMemcpy(hOut.data(), dOut, bytes, hipMemcpyDeviceToHost));

    std::cout << "✅ PASS: Kernel executed successfully\n";
    std::cout << "Sample results (threshold=" << threshold << "):\n";
    for (int i = 0; i < std::min(5, N); i++) {
        std::cout << "  [" << i << "] input=" << hIn[i] << ", output=" << hOut[i] << "\n";
    }
    // 顯示閾值附近的幾個結果
    int near_threshold = static_cast<int>(threshold);
    std::cout << "Results near threshold:\n";
    for (int i = near_threshold - 2; i <= near_threshold + 2 && i < N; i++) {
        if (i >= 0) {
            std::cout << "  [" << i << "] input=" << hIn[i] << ", output=" << hOut[i] << "\n";
        }
    }

    HIP_CHECK(hipFree(dIn));
    HIP_CHECK(hipFree(dOut));

    return true;
}

// ============================================================================
// run_int_kernel_1out - 執行只有一個輸出參數的 Int kernel
// ============================================================================
bool run_int_kernel_1out(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(int);
    
    std::vector<int> hOut(N, 0);

    int *dOut = nullptr;
    HIP_CHECK(hipMalloc(&dOut, bytes));
    HIP_CHECK(hipMemcpy(dOut, hOut.data(), bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dOut,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    HIP_CHECK(hipMemcpy(hOut.data(), dOut, bytes, hipMemcpyDeviceToHost));

    std::cout << "✅ PASS: Kernel executed successfully\n";
    std::cout << "Sample results (output):\n";
    for (int i = 0; i < std::min(5, N); i++) {
        std::cout << "  [" << i << "] = " << hOut[i] << "\n";
    }

    HIP_CHECK(hipFree(dOut));
    return true;
}

// ============================================================================
// run_int_kernel_inout - 執行有輸入和輸出參數的 Int kernel
// ============================================================================
bool run_int_kernel_inout(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(int);
    
    std::vector<int> hIn(N), hOut(N, 0);
    
    // 初始化輸入資料：input[i] = i
    for (int i = 0; i < N; ++i) {
        hIn[i] = i;
    }

    int *dIn = nullptr, *dOut = nullptr;
    HIP_CHECK(hipMalloc(&dIn, bytes));
    HIP_CHECK(hipMalloc(&dOut, bytes));

    HIP_CHECK(hipMemcpy(dIn, hIn.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dOut, hOut.data(), bytes, hipMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    void* kernelArgs[] = {
        (void*)&dIn,
        (void*)&dOut,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    HIP_CHECK(hipMemcpy(hOut.data(), dOut, bytes, hipMemcpyDeviceToHost));

    std::cout << "✅ PASS: Kernel executed successfully\n";


    HIP_CHECK(hipFree(dIn));
    HIP_CHECK(hipFree(dOut));
    return true;
}

// ============================================================
// Atomic Timestamp 測量 V2
// 
// 方法：使用標準 kernel 簽名 (A*, B*, C*, N)
// Timestamp buffer 位於 C + N*sizeof(float)
// Kernel 內部會計算這個地址並使用 global atomic 操作
// 
// C buffer 布局（Host 分配）：
//   [0 ... N*sizeof(float)-1]: 實際輸出數據
//   [N*sizeof(float) ... N*sizeof(float)+15]: timestamp buffer
//     - Offset 0-7:  min_start (初始化為 MAX_U64)
//     - Offset 8-15: max_end   (初始化為 0)
// ============================================================

// ============================================================================
// run_float_vector_add_with_timestamp - 執行 Float 向量加法 kernel (帶 atomic timestamp)
// ============================================================================
bool run_float_vector_add_with_timestamp(hipFunction_t func, int N) {
    // 分配額外的 16 bytes 在 C buffer 末尾用於 timestamp
    const size_t bytes = N * sizeof(float);
    const size_t bytes_with_timestamp = bytes + 16;  // 額外 16 bytes for timestamp
    
    std::vector<float> hA(N), hB(N);
    std::vector<char> hC_raw(bytes_with_timestamp);  // 使用 char array 以便存放 timestamp
    float* hC = reinterpret_cast<float*>(hC_raw.data());
    uint64_t* hTimestamp = reinterpret_cast<uint64_t*>(hC_raw.data() + bytes);
    
    // 初始化測試資料：A[i] = i, B[i] = 2*i
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i * 2);
        hC[i] = 0.0f;
    }
    
    // 初始化 timestamp: min_start = MAX, max_end = 0
    hTimestamp[0] = 0xFFFFFFFFFFFFFFFFULL;  // min_start = MAX
    hTimestamp[1] = 0;                        // max_end = 0

    float *dA = nullptr, *dB = nullptr;
    char *dC_raw = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));
    HIP_CHECK(hipMalloc(&dB, bytes));
    HIP_CHECK(hipMalloc(&dC_raw, bytes_with_timestamp));

    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC_raw, hC_raw.data(), bytes_with_timestamp, hipMemcpyHostToDevice));
    
    float* dC = reinterpret_cast<float*>(dC_raw);
    
    std::cout << "✓ Timestamp buffer at C+" << bytes << " bytes (kernel computes address)\n";
    std::cout << "  Initialized: min_start=0x" << std::hex << hTimestamp[0] 
              << ", max_end=0x" << hTimestamp[1] << std::dec << "\n";

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 標準 kernel 參數：(A*, B*, C*, N)
    // Kernel 會從 C 和 N 計算 timestamp buffer 地址
    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dC,
        (void*)&N
    };

    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    // ============================================================
    // 讀取結果
    // ============================================================
    HIP_CHECK(hipMemcpy(hC_raw.data(), dC_raw, bytes_with_timestamp, hipMemcpyDeviceToHost));
    
    uint64_t min_start = hTimestamp[0];
    uint64_t max_end = hTimestamp[1];
    
    std::cout << "========================================\n";
    std::cout << "[Atomic Timestamp Results]\n";
    std::cout << "  min_start:       " << min_start << " ticks\n";
    std::cout << "  max_end:         " << max_end << " ticks\n";
    
    // 檢查是否有有效的 atomic 更新
    bool timestamp_valid = (min_start != 0xFFFFFFFFFFFFFFFFULL && max_end != 0);
    
    if (timestamp_valid) {
        uint64_t kernel_wall_time = max_end - min_start;
        std::cout << "  kernel_wall_time = max_end - min_start\n";
        std::cout << "  ────────────────────────────────\n";
        std::cout << "  KERNEL WALL TIME: " << kernel_wall_time << " ticks\n";
        
        // 假設 GPU 時鐘頻率約 1.7 GHz (可根據實際 GPU 調整)
        // 1 tick ≈ 0.588 ns
        double approx_ns = kernel_wall_time * 0.588;
        double approx_us = approx_ns / 1000.0;
        std::cout << "  (~" << std::fixed << std::setprecision(2) << approx_us << " us at 1.7 GHz)\n";
    } else {
        std::cout << "  ⚠️ Atomic timestamp not updated (kernel may not have atomic mode)\n";
    }
    std::cout << "========================================\n";

    // 驗證結果
    bool ok = true;
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];
        if (std::fabs(hC[i] - expected) > 1e-5f) {
            if (error_count < 5) {
                std::cerr << "Mismatch at [" << i << "]: got " << hC[i] 
                          << ", expected " << expected << "\n";
            }
            error_count++;
            ok = false;
        }
    }

    if (ok) {
        std::cout << "✅ PASS: All " << N << " elements correct\n";
    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC_raw));

    return ok;
}

// ============================================================================
// main - 主程式進入點
// ============================================================================
int main(int argc, char* argv[]) {
    // 檢查 --timestamp-atomic 標誌
    bool use_timestamp_atomic = false;
    int arg_offset = 0;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--timestamp-atomic") == 0) {
            use_timestamp_atomic = true;
            arg_offset = 1;
            // 移動後面的參數
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            argc--;
            break;
        }
    }
    
    if (argc < 5) {
        print_usage(argv[0]);
        std::cerr << "\nOptions:\n";
        std::cerr << "  --timestamp-atomic  Enable atomic timestamp profiling\n";
        return 1;
    }

    const char* hsaco_path = argv[1];
    const char* kernel_name = argv[2];
    const char* kernel_type_str = argv[3];
    const int N = std::atoi(argv[4]);

    KernelType kernel_type = parse_kernel_type(kernel_type_str);

    std::cout << "========================================\n";
    std::cout << "Universal HSACO Runner\n";
    std::cout << "========================================\n";
    std::cout << "HSACO:  " << hsaco_path << "\n";
    std::cout << "Kernel: " << kernel_name << "\n";
    std::cout << "Type:   " << kernel_type_str << "\n";
    std::cout << "Size:   " << N << "\n";
    std::cout << "========================================\n";

    HIP_CHECK(hipSetDevice(0));

    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, hsaco_path);
    if (err != hipSuccess) {
        std::cerr << "Failed to load module: " << hipGetErrorString(err) << "\n";
        return 1;
    }
    std::cout << "✓ Module loaded successfully\n";

    hipFunction_t func;
    err = hipModuleGetFunction(&func, module, kernel_name);
    if (err != hipSuccess) {
        std::cerr << "Failed to get function '" << kernel_name << "': " 
                  << hipGetErrorString(err) << "\n";
        HIP_CHECK(hipModuleUnload(module));
        return 1;
    }
    std::cout << "✓ Function found: " << kernel_name << "\n";

    std::cout << "========================================\n";
    
    bool result = false;
    
    // 如果啟用 atomic timestamp，使用特殊版本
    if (use_timestamp_atomic && kernel_type == FLOAT_VECTOR_ADD) {
        std::cout << "[Atomic Timestamp Mode Enabled]\n";
        result = run_float_vector_add_with_timestamp(func, N);
    } else {
        switch (kernel_type) {
            case FLOAT_VECTOR_ADD:
                result = run_float_vector_add(func, N);
                break;
                
            case FLOAT_VECTOR_MUL:
                result = run_float_vector_mul(func, N);
                break;
                
            case FLOAT_VECTOR_DOT:
                result = run_float_vector_dot(func, N);
                break;
                
            case FLOAT_SAXPY:
                result = run_float_saxpy(func, N);
                break;
                
            case FLOAT_CONDITIONAL:
                result = run_float_conditional(func, N);
                break;
                
            case INT_SCALAR_OPS:
            case INT_LOOP:
                result = run_int_kernel_1out(func, N);
                break;
                
            case INT_MEMORY_OPS:
            case INT_CONDITIONAL:
            case INT_SHARED_MEMORY:
                result = run_int_kernel_inout(func, N);
                break;
        }
    }

    std::cout << "========================================\n";

    HIP_CHECK(hipModuleUnload(module));
    
    return result ? 0 : 1;
}

