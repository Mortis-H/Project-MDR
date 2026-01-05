// host_program.cpp
// Host program that dynamically loads HSACO at runtime

#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// 錯誤檢查宏
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << "Error code: " << err << " - " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

class HSACOLoader {
private:
    hipModule_t module_;
    hipFunction_t function_;
    std::string hsaco_path_;
    std::string kernel_name_;

public:
    HSACOLoader(const std::string& hsaco_path, const std::string& kernel_name)
        : hsaco_path_(hsaco_path), kernel_name_(kernel_name), 
          module_(nullptr), function_(nullptr) {}

    ~HSACOLoader() {
        if (module_) {
            hipModuleUnload(module_);
        }
    }

    // 載入 HSACO 文件
    bool load() {
        std::cout << "Loading HSACO: " << hsaco_path_ << std::endl;
        
        // 讀取 HSACO 文件
        std::ifstream file(hsaco_path_, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Failed to open HSACO file: " << hsaco_path_ << std::endl;
            return false;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            std::cerr << "Failed to read HSACO file" << std::endl;
            return false;
        }

        std::cout << "HSACO file size: " << size << " bytes" << std::endl;

        // 從記憶體載入模組
        hipError_t err = hipModuleLoadData(&module_, buffer.data());
        if (err != hipSuccess) {
            std::cerr << "Failed to load module: " << hipGetErrorString(err) << std::endl;
            return false;
        }

        // 取得 kernel 函數
        err = hipModuleGetFunction(&function_, module_, kernel_name_.c_str());
        if (err != hipSuccess) {
            std::cerr << "Failed to get function '" << kernel_name_ 
                      << "': " << hipGetErrorString(err) << std::endl;
            return false;
        }

        std::cout << "Successfully loaded kernel: " << kernel_name_ << std::endl;
        return true;
    }

    // 執行 kernel
    template<typename... Args>
    void launch(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream, Args... args) {
        void* kernel_args[] = { (void*)&args... };
        
        HIP_CHECK(hipModuleLaunchKernel(
            function_,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            sharedMem,
            stream,
            kernel_args,
            nullptr
        ));
    }
};

// 測試向量加法
void testVectorAdd(HSACOLoader& loader, int n) {
    std::cout << "\n=== Testing Vector Addition (n=" << n << ") ===" << std::endl;

    // 分配 host memory
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);
    std::vector<float> h_c_ref(n);

    // 初始化輸入數據
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
        h_c_ref[i] = h_a[i] + h_b[i];
    }

    // 分配 device memory
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, n * sizeof(float)));

    // 複製數據到 device
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), n * sizeof(float), hipMemcpyHostToDevice));

    // 設定 kernel 參數
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    std::cout << "Launch configuration: grid=" << gridSize 
              << ", block=" << blockSize << std::endl;

    // 執行 kernel
    loader.launch(
        dim3(gridSize), 
        dim3(blockSize), 
        0,  // shared memory size
        0,  // stream
        d_a, d_b, d_c, n
    );

    // 等待完成
    HIP_CHECK(hipDeviceSynchronize());

    // 複製結果回 host
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, n * sizeof(float), hipMemcpyDeviceToHost));

    // 驗證結果
    bool passed = true;
    int error_count = 0;
    const int max_errors = 10;

    for (int i = 0; i < n; i++) {
        if (std::abs(h_c[i] - h_c_ref[i]) > 1e-5) {
            if (error_count < max_errors) {
                std::cerr << "Mismatch at index " << i << ": "
                         << "got " << h_c[i] << ", expected " << h_c_ref[i] << std::endl;
                error_count++;
            }
            passed = false;
        }
    }

    if (passed) {
        std::cout << "✓ Test PASSED! All results are correct." << std::endl;
    } else {
        std::cout << "✗ Test FAILED! Found " << error_count << " errors." << std::endl;
    }

    // 清理
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
}

int main(int argc, char** argv) {
    std::cout << "=== HIP Dynamic HSACO Loading Example ===" << std::endl;

    // 檢查命令行參數
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_hsaco> [kernel_name] [vector_size]" << std::endl;
        std::cerr << "Example: " << argv[0] << " vectorAdd.hsaco vectorAdd 1048576" << std::endl;
        return 1;
    }

    std::string hsaco_path = argv[1];
    std::string kernel_name = (argc >= 3) ? argv[2] : "vectorAdd";
    int n = (argc >= 4) ? std::atoi(argv[3]) : 1048576;  // 默認 1M 元素

    // 初始化 HIP
    HIP_CHECK(hipInit(0));
    
    // 獲取設備信息
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    std::cout << "Found " << deviceCount << " HIP device(s)" << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No HIP devices found!" << std::endl;
        return 1;
    }

    // 設定設備
    HIP_CHECK(hipSetDevice(0));

    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "gcnArchName: " << prop.gcnArchName << std::endl;

    // 載入 HSACO
    HSACOLoader loader(hsaco_path, kernel_name);
    if (!loader.load()) {
        std::cerr << "Failed to load HSACO" << std::endl;
        return 1;
    }

    // 執行測試
    try {
        testVectorAdd(loader, n);
    } catch (const std::exception& e) {
        std::cerr << "Error during test: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n=== Program completed successfully ===" << std::endl;
    return 0;
}

