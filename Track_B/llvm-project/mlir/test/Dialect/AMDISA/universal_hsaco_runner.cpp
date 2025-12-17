// universal_hsaco_runner.cpp
// 通用 HSACO 執行器，支持 float 和 int kernel
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

enum KernelType {
    FLOAT_VECTOR_ADD,    // (float*, float*, float*, int)
    INT_SCALAR_OPS,      // (int*, int)
    INT_MEMORY_OPS,      // (int*, int*, int)
    INT_CONDITIONAL,     // (int*, int*, int)
    INT_LOOP,            // (int*, int)
    INT_SHARED_MEMORY    // (int*, int*, int)
};

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

KernelType parse_kernel_type(const char* type_str) {
    if (strcmp(type_str, "float_add") == 0) return FLOAT_VECTOR_ADD;
    if (strcmp(type_str, "int_scalar") == 0) return INT_SCALAR_OPS;
    if (strcmp(type_str, "int_mem") == 0) return INT_MEMORY_OPS;
    if (strcmp(type_str, "int_cond") == 0) return INT_CONDITIONAL;
    if (strcmp(type_str, "int_loop") == 0) return INT_LOOP;
    if (strcmp(type_str, "int_shared") == 0) return INT_SHARED_MEMORY;
    return FLOAT_VECTOR_ADD; // default
}

bool run_float_vector_add(hipFunction_t func, int N) {
    const size_t bytes = N * sizeof(float);
    std::vector<float> hA(N), hB(N), hC(N);
    
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
        std::cout << "Sample results:\n";
        for (int i = 0; i < std::min(5, N); i++) {
            std::cout << "  [" << i << "] " << hA[i] << " + " << hB[i] 
                      << " = " << hC[i] << "\n";
        }
    } else {
        std::cout << "❌ FAIL: " << error_count << " errors found\n";
    }

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return ok;
}

bool run_int_kernel_1out(hipFunction_t func, int N) {
    // For kernels with signature: (int* output, int n)
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

bool run_int_kernel_inout(hipFunction_t func, int N) {
    // For kernels with signature: (int* input, int* output, int n)
    const size_t bytes = N * sizeof(int);
    std::vector<int> hIn(N), hOut(N, 0);
    
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
    std::cout << "Sample results:\n";
    for (int i = 0; i < std::min(5, N); i++) {
        std::cout << "  [" << i << "] input=" << hIn[i] << ", output=" << hOut[i] << "\n";
    }

    HIP_CHECK(hipFree(dIn));
    HIP_CHECK(hipFree(dOut));
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        print_usage(argv[0]);
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
    switch (kernel_type) {
        case FLOAT_VECTOR_ADD:
            result = run_float_vector_add(func, N);
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

    std::cout << "========================================\n";

    HIP_CHECK(hipModuleUnload(module));
    return result ? 0 : 1;
}

