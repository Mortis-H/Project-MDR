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

// ============================================================================
// HIP_CHECK 巨集 - 檢查 HIP API 呼叫是否成功
// ============================================================================
// 這個巨集會檢查 HIP API 的返回值，如果發生錯誤會顯示錯誤訊息並終止程式。
// 用於簡化錯誤處理，避免重複寫 if (err != hipSuccess) 的程式碼。
//
// 使用範例：
//   HIP_CHECK(hipMalloc(&ptr, size));
//   HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
//
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
// 根據不同的 kernel 類型，會使用不同的記憶體配置和參數傳遞方式。
// 註解中的簽名表示該 kernel 預期的參數列表。
//
enum KernelType {
    FLOAT_VECTOR_ADD,    // Float 向量加法: (float* a, float* b, float* c, int n)
    INT_SCALAR_OPS,      // Int 純量運算: (int* output, int n)
    INT_MEMORY_OPS,      // Int 記憶體操作: (int* input, int* output, int n)
    INT_CONDITIONAL,     // Int 條件判斷: (int* input, int* output, int n)
    INT_LOOP,            // Int 迴圈: (int* output, int n)
    INT_SHARED_MEMORY    // Int 共享記憶體: (int* input, int* output, int n)
};

// ============================================================================
// print_usage - 顯示使用說明
// ============================================================================
// 當使用者提供的參數不正確時，顯示詳細的使用說明和範例。
//
// 參數：
//   prog - 程式名稱 (通常是 argv[0])
//
void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <hsaco_path> <kernel_name> <kernel_type> <test_size>\n";
    std::cerr << "\nKernel types:\n";
    std::cerr << "  float_add   - float vector addition (a, b, c, n)\n";
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
// 將命令列提供的 kernel 類型字串轉換為對應的 KernelType 列舉值。
//
// 參數：
//   type_str - kernel 類型字串 (例如 "float_add", "int_scalar")
//
// 返回值：
//   對應的 KernelType 列舉值，如果無法識別則返回 FLOAT_VECTOR_ADD (預設值)
//
KernelType parse_kernel_type(const char* type_str) {
    if (strcmp(type_str, "float_add") == 0) return FLOAT_VECTOR_ADD;
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
// 這個函數測試基本的浮點向量加法：C[i] = A[i] + B[i]
//
// 測試流程：
//   1. 準備測試資料：A[i] = i, B[i] = 2*i
//   2. 在 GPU 上執行 kernel
//   3. 驗證結果：檢查 C[i] 是否等於 A[i] + B[i]
//   4. 顯示前 5 個元素的結果
//
// 參數：
//   func - HIP kernel 函數指標
//   N    - 測試資料大小 (元素數量)
//
// 返回值：
//   true  - 所有結果正確
//   false - 有錯誤發生
//
bool run_float_vector_add(hipFunction_t func, int N) {
    // 計算需要的記憶體大小
    const size_t bytes = N * sizeof(float);
    
    // 在 Host (CPU) 端配置記憶體
    // hA, hB: 輸入陣列, hC: 輸出陣列
    std::vector<float> hA(N), hB(N), hC(N);
    
    // 初始化測試資料
    // A[i] = i, B[i] = 2*i, C[i] = 0
    // 預期結果：C[i] = A[i] + B[i] = i + 2*i = 3*i
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);
        hB[i] = static_cast<float>(i * 2);
        hC[i] = 0.0f;
    }

    // 在 Device (GPU) 端配置記憶體
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    HIP_CHECK(hipMalloc(&dA, bytes));  // 配置 dA
    HIP_CHECK(hipMalloc(&dB, bytes));  // 配置 dB
    HIP_CHECK(hipMalloc(&dC, bytes));  // 配置 dC

    // 將資料從 Host 複製到 Device
    HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hC.data(), bytes, hipMemcpyHostToDevice));

    // 設定 kernel 執行配置
    // blockSize: 每個 block 的 thread 數量 (256 是常用值)
    // gridSize: block 的數量，計算方式確保能覆蓋所有資料
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;  // 向上取整

    // 準備 kernel 參數
    // 注意：傳遞的是指標的位址 (&dA, &dB, &dC, &N)
    void* kernelArgs[] = {
        (void*)&dA,
        (void*)&dB,
        (void*)&dC,
        (void*)&N
    };

    // 啟動 kernel
    // 參數說明：
    //   func: kernel 函數
    //   gridSize, 1, 1: grid 維度 (x, y, z)
    //   blockSize, 1, 1: block 維度 (x, y, z)
    //   0: 共享記憶體大小 (此處不使用)
    //   nullptr: stream (使用預設 stream)
    //   kernelArgs: kernel 參數陣列
    //   nullptr: extra 參數
    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    // 等待 kernel 執行完成
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    // 將結果從 Device 複製回 Host
    HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

    // 驗證結果
    bool ok = true;
    int error_count = 0;
    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];  // 預期值
        // 使用 fabs 比較浮點數，容忍小誤差 (1e-5)
        if (std::fabs(hC[i] - expected) > 1e-5f) {
            // 只顯示前 5 個錯誤，避免輸出過多
            if (error_count < 5) {
                std::cerr << "Mismatch at [" << i << "]: got " << hC[i] 
                          << ", expected " << expected << "\n";
            }
            error_count++;
            ok = false;
        }
    }

    // 顯示測試結果
    if (ok) {
        std::cout << "✅ PASS: All " << N << " elements correct\n";
        std::cout << "Sample results:\n";
        // 顯示前 5 個元素的結果
        for (int i = 0; i < std::min(5, N); i++) {
            std::cout << "  [" << i << "] " << hA[i] << " + " << hB[i] 
                      << " = " << hC[i] << "\n";
        }
    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }

    // 釋放 Device 記憶體
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return ok;
}

// ============================================================================
// run_int_kernel_1out - 執行只有一個輸出參數的 Int kernel
// ============================================================================
// 這個函數用於測試簽名為 (int* output, int n) 的 kernel。
// 適用於：INT_SCALAR_OPS (純量運算) 和 INT_LOOP (迴圈)
//
// 測試流程：
//   1. 配置輸出陣列並初始化為 0
//   2. 在 GPU 上執行 kernel
//   3. 讀取並顯示結果 (不做驗證，因為不同 kernel 的預期輸出不同)
//
// 參數：
//   func - HIP kernel 函數指標
//   N    - 測試資料大小
//
// 返回值：
//   總是返回 true (因為沒有預期結果可比對)
//
bool run_int_kernel_1out(hipFunction_t func, int N) {
    // For kernels with signature: (int* output, int n)
    const size_t bytes = N * sizeof(int);
    
    // Host 端輸出陣列，初始化為 0
    std::vector<int> hOut(N, 0);

    // Device 端輸出陣列
    int *dOut = nullptr;
    HIP_CHECK(hipMalloc(&dOut, bytes));
    HIP_CHECK(hipMemcpy(dOut, hOut.data(), bytes, hipMemcpyHostToDevice));

    // 設定 kernel 執行配置
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 準備 kernel 參數
    void* kernelArgs[] = {
        (void*)&dOut,  // 輸出陣列
        (void*)&N      // 資料大小
    };

    // 啟動 kernel
    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    // 等待執行完成
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    // 讀取結果
    HIP_CHECK(hipMemcpy(hOut.data(), dOut, bytes, hipMemcpyDeviceToHost));

    // 顯示結果 (無法驗證正確性，因為不知道預期輸出)
    std::cout << "✅ PASS: Kernel executed successfully\n";
    std::cout << "Sample results (output):\n";
    for (int i = 0; i < std::min(5, N); i++) {
        std::cout << "  [" << i << "] = " << hOut[i] << "\n";
    }

    // 釋放記憶體
    HIP_CHECK(hipFree(dOut));
    return true;
}

// ============================================================================
// run_int_kernel_inout - 執行有輸入和輸出參數的 Int kernel
// ============================================================================
// 這個函數用於測試簽名為 (int* input, int* output, int n) 的 kernel。
// 適用於：INT_MEMORY_OPS (記憶體操作)、INT_CONDITIONAL (條件判斷)、
//        INT_SHARED_MEMORY (共享記憶體)
//
// 測試流程：
//   1. 準備輸入資料：input[i] = i
//   2. 配置輸出陣列並初始化為 0
//   3. 在 GPU 上執行 kernel
//   4. 讀取並顯示結果 (不做驗證)
//
// 參數：
//   func - HIP kernel 函數指標
//   N    - 測試資料大小
//
// 返回值：
//   總是返回 true (因為沒有預期結果可比對)
//
bool run_int_kernel_inout(hipFunction_t func, int N) {
    // For kernels with signature: (int* input, int* output, int n)
    const size_t bytes = N * sizeof(int);
    
    // Host 端陣列
    std::vector<int> hIn(N), hOut(N, 0);
    
    // 初始化輸入資料：input[i] = i
    for (int i = 0; i < N; ++i) {
        hIn[i] = i;
    }

    // Device 端陣列
    int *dIn = nullptr, *dOut = nullptr;
    HIP_CHECK(hipMalloc(&dIn, bytes));
    HIP_CHECK(hipMalloc(&dOut, bytes));

    // 複製資料到 Device
    HIP_CHECK(hipMemcpy(dIn, hIn.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dOut, hOut.data(), bytes, hipMemcpyHostToDevice));

    // 設定 kernel 執行配置
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 準備 kernel 參數
    void* kernelArgs[] = {
        (void*)&dIn,   // 輸入陣列
        (void*)&dOut,  // 輸出陣列
        (void*)&N      // 資料大小
    };

    // 啟動 kernel
    HIP_CHECK(hipModuleLaunchKernel(func, gridSize, 1, 1, blockSize, 1, 1, 
                                    0, nullptr, kernelArgs, nullptr));
    std::cout << "✓ Kernel launched (grid=" << gridSize << ", block=" << blockSize << ")\n";

    // 等待執行完成
    HIP_CHECK(hipDeviceSynchronize());
    std::cout << "✓ Kernel execution completed\n";

    // 讀取結果
    HIP_CHECK(hipMemcpy(hOut.data(), dOut, bytes, hipMemcpyDeviceToHost));

    // 顯示結果
    std::cout << "✅ PASS: Kernel executed successfully\n";
    std::cout << "Sample results:\n";
    for (int i = 0; i < std::min(5, N); i++) {
        std::cout << "  [" << i << "] input=" << hIn[i] << ", output=" << hOut[i] << "\n";
    }

    // 釋放記憶體
    HIP_CHECK(hipFree(dIn));
    HIP_CHECK(hipFree(dOut));
    return true;
}

// ============================================================================
// main - 主程式進入點
// ============================================================================
// 程式執行流程：
//   1. 解析命令列參數
//   2. 設定 GPU 裝置
//   3. 載入 HSACO 模組
//   4. 取得 kernel 函數
//   5. 根據 kernel 類型執行對應的測試函數
//   6. 清理資源並返回結果
//
// 命令列參數：
//   argv[1] - HSACO 檔案路徑
//   argv[2] - Kernel 名稱 (通常是 C++ mangled name)
//   argv[3] - Kernel 類型字串
//   argv[4] - 測試資料大小
//
// 返回值：
//   0 - 測試通過
//   1 - 測試失敗或發生錯誤
//
int main(int argc, char* argv[]) {
    // 檢查參數數量
    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    // 解析命令列參數
    const char* hsaco_path = argv[1];       // HSACO 檔案路徑
    const char* kernel_name = argv[2];      // Kernel 名稱
    const char* kernel_type_str = argv[3];  // Kernel 類型字串
    const int N = std::atoi(argv[4]);       // 測試資料大小

    // 解析 kernel 類型
    KernelType kernel_type = parse_kernel_type(kernel_type_str);

    // 顯示測試配置資訊
    std::cout << "========================================\n";
    std::cout << "Universal HSACO Runner\n";
    std::cout << "========================================\n";
    std::cout << "HSACO:  " << hsaco_path << "\n";
    std::cout << "Kernel: " << kernel_name << "\n";
    std::cout << "Type:   " << kernel_type_str << "\n";
    std::cout << "Size:   " << N << "\n";
    std::cout << "========================================\n";

    // 設定使用的 GPU 裝置 (裝置 0 是第一個 GPU)
    HIP_CHECK(hipSetDevice(0));

    // 載入 HSACO 模組
    // hipModule_t 是 HIP 模組的不透明句柄
    hipModule_t module;
    hipError_t err = hipModuleLoad(&module, hsaco_path);
    if (err != hipSuccess) {
        std::cerr << "Failed to load module: " << hipGetErrorString(err) << "\n";
        return 1;
    }
    std::cout << "✓ Module loaded successfully\n";

    // 從模組中取得指定的 kernel 函數
    // hipFunction_t 是 kernel 函數的不透明句柄
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
    
    // 根據 kernel 類型執行對應的測試函數
    bool result = false;
    switch (kernel_type) {
        case FLOAT_VECTOR_ADD:
            // Float 向量加法 - 有完整的驗證邏輯
            result = run_float_vector_add(func, N);
            break;
            
        case INT_SCALAR_OPS:
        case INT_LOOP:
            // Int 純量運算和迴圈 - 簽名: (int* output, int n)
            result = run_int_kernel_1out(func, N);
            break;
            
        case INT_MEMORY_OPS:
        case INT_CONDITIONAL:
        case INT_SHARED_MEMORY:
            // Int 記憶體操作、條件判斷、共享記憶體 - 簽名: (int* input, int* output, int n)
            result = run_int_kernel_inout(func, N);
            break;
    }

    std::cout << "========================================\n";

    // 卸載模組，釋放資源
    HIP_CHECK(hipModuleUnload(module));
    
    // 返回測試結果
    // 0 = 成功, 1 = 失敗
    return result ? 0 : 1;
}

